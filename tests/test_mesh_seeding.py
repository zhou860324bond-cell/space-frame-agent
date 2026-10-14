"""布种与特征值/二阶分析的默认加密。

防的是一个真实而且最常见的问题：参数化生成的模型一杆一单元，频率、屈曲、
P-Δ 直接在它上面算，一致质量阵和几何刚度阵靠形函数装配，网格越粗报得越高
——偏不安全那一侧。实测单跨门式刚架屈曲因子 λ=1.66，收敛值 0.82（偏高
102%）；8 m 简支梁一个单元基频 +11%、屈曲 +21.6%。原来只附一条「网格太粗」
的提醒，让用户自己去加节点。

这里用闭合解核对：默认就剖够之后，单杆模型不加任何节点也落在 1% 以内。
另外核对布种本身的编译语义，以及线性静力不因布种而改变（内力在单元内解析
恢复，一段就是精确解）。
"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from agent import Session
from frame3d import solve
from model_compiler import ANALYSIS_MIN_DIVISIONS, compile_model
from model_io import validate_payload
from units import convert_model

E, NU, RHO = 2.0e11, 0.3, 7850.0
A, IY, IZ, J = 0.01, 1.0e-5, 2.0e-5, 5.0e-6


def _beam(length: float = 8.0, supports=None, **extra) -> dict:
    """沿 x 的单根杆件，一杆一单元，不带任何中间节点。"""
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": E, "nu": NU, "density": RHO}],
        "sections": [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": length, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "Steel"}],
        "supports": supports or [
            # 两端铰支：两端都不许平动，i 端再锁住绕杆轴的转动，防扭转机构。
            {"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
            {"node": 2, "fix": [0, 1, 1, 0, 0, 0]},
        ],
        **extra,
    }


def _portal() -> dict:
    """门式刚架，带均布、梯形与集中荷载，柱脚固接。"""
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": E, "nu": NU, "density": RHO}],
        "sections": [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 0, "y": 0, "z": 4},
                  {"id": 3, "x": 6, "y": 0, "z": 4}, {"id": 4, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "Steel"},
                    {"id": 2, "i": 2, "j": 3, "section": "S", "material": "Steel",
                     "releases": {"j": ["ry"]}},
                    {"id": 3, "i": 4, "j": 3, "section": "S", "material": "Steel"}],
        "supports": [{"node": 1, "fix": [1] * 6}, {"node": 4, "fix": [1] * 6}],
        "nodal_loads": [{"node": 2, "load": [5e3, 0, 0, 0, 0, 0]}],
        "member_spans": [
            {"member": 2, "kind": "uniform", "w1": [0, 0, -1e4]},
            {"member": 2, "kind": "point", "w1": [0, 0, -2e4], "a": 1.5},
            {"member": 1, "kind": "trapezoid", "w1": [2e3, 0, 0], "w2": [0, 0, 0]},
        ],
    }


def _node_u(frame, solution, node_id: int) -> np.ndarray:
    k = frame.index_of()[node_id]
    return solution.cases[solution.primary].U[6 * k:6 * k + 6]


# ------------------------------------------------------------------ 编译语义

def test_global_seed_splits_every_member_into_equal_segments():
    model = _beam(mesh={"divisions": 4})
    compiled = compile_model(model)

    assert len(compiled.mapping.element_ids(1)) == 4
    ratios = sorted(r for _, r in compiled.mapping.generated_nodes.values())
    assert ratios == pytest.approx([0.25, 0.5, 0.75])
    assert any("布种 4 段" in d for d in compiled.diagnostics)


def test_member_seed_overrides_global_and_analysis_minimum_takes_the_larger():
    model = _portal()
    model["mesh"] = {"divisions": 2, "members": {"2": 6}}
    compiled = compile_model(model)
    assert len(compiled.mapping.element_ids(1)) == 2
    # 杆 2 长 6 m 布种 6 段（间距 1 m），集中力在 1.5 m，落在两个布种点中间，
    # 离两边都超过 0.1 倍间距，所以多切一刀：7 个单元
    assert len(compiled.mapping.element_ids(2)) == 7

    refined = compile_model(model, min_divisions=ANALYSIS_MIN_DIVISIONS)
    assert len(refined.mapping.element_ids(1)) == ANALYSIS_MIN_DIVISIONS
    assert len(refined.mapping.element_ids(2)) == 7


def test_seed_point_next_to_a_point_load_is_merged_not_split_again():
    """集中力在 0.26L、布种点在 0.25L：再切一刀只会造出 0.01L 的碎单元。"""
    model = _beam(member_spans=[{"member": 1, "kind": "point",
                                 "w1": [0, 0, -1e3], "a": 0.26 * 8.0}],
                  mesh={"divisions": 4})
    compiled = compile_model(model)
    assert len(compiled.mapping.element_ids(1)) == 4
    ratios = sorted(r for _, r in compiled.mapping.generated_nodes.values())
    assert ratios == pytest.approx([0.26, 0.5, 0.75])


def test_releases_stay_at_the_physical_ends_after_seeding():
    model = _portal()
    model["mesh"] = {"divisions": 4}
    compiled = compile_model(model)
    elements = [compiled.analysis_model.members[e]
                for e in compiled.mapping.element_ids(2)]
    # 杆 2 长 6 m 分 4 段，1.5 m 处的集中力与第一个布种点重合，合并后仍是 4 段
    assert [bool(e.releases_j) for e in elements] == [False, False, False, True]
    assert not any(e.releases_i for e in elements)


@pytest.mark.parametrize("offset", [False, True])
def test_linear_static_is_unchanged_by_seeding(offset):
    """内力在单元内解析恢复：布种只加结果节点，原节点位移与反力逐位不变。"""
    base = _portal()
    if offset:
        # 刚域偏移只留在整根杆两端；布种点上没有荷载，可以和偏移同时用
        base["member_spans"] = [s for s in base["member_spans"] if s["kind"] != "point"]
        base["members"][0]["offset_j"] = [0.0, 0.0, -0.2]
    seeded = deepcopy(base)
    seeded["mesh"] = {"divisions": 5}

    coarse = compile_model(base).analysis_model
    fine = compile_model(seeded).analysis_model
    sol_c, sol_f = solve(coarse), solve(fine)
    for node in (1, 2, 3, 4):
        np.testing.assert_allclose(_node_u(fine, sol_f, node),
                                   _node_u(coarse, sol_c, node),
                                   rtol=1e-9, atol=1e-14)


def test_schema_rejects_a_seed_for_a_member_that_does_not_exist():
    errors = validate_payload(_beam(mesh={"members": {"7": 3}}))
    assert any("mesh/members" in e and "[7]" in e for e in errors)
    assert validate_payload(_beam(mesh={"divisions": 0}))


# ------------------------------------------------------- 闭合解：默认就剖够

def test_modal_of_a_single_member_beam_matches_the_closed_form():
    """简支梁基频 f₁ = (π/L)²·√(EI/ρA)/(2π)。一个单元时偏高 11%，这里不加任何节点。"""
    length = 8.0
    session = Session(model=_beam(length))
    got = session.modal_analysis(num_modes=4)
    assert got.ok, got.payload

    exact = (math.pi / length) ** 2 * math.sqrt(E * IY / (RHO * A)) / (2 * math.pi)
    f1 = got.payload["modes"][0]["frequency_Hz"]
    assert f1 == pytest.approx(exact, rel=5e-3)
    assert f1 >= exact * (1 - 1e-9), "一致质量阵应从上方逼近"
    assert "mesh_warning" not in got.payload
    assert got.payload["mesh"]["analysis_elements"] == ANALYSIS_MIN_DIVISIONS
    # 界面画振型复用同一份结果，不另算
    assert session.modal_view is not None
    assert session.modal_view[3].frequencies[0] == pytest.approx(f1, rel=1e-6)


def test_buckling_of_a_single_member_column_matches_euler():
    """两端铰支压杆 Pcr = π²EI_min/L²。一个单元时偏高 21.6%，这里不加任何节点。"""
    length, load = 8.0, 1.0e4
    session = Session(model=_beam(
        length, nodal_loads=[{"node": 2, "load": [-load, 0, 0, 0, 0, 0]}]))
    got = session.buckling_analysis(num_modes=2)
    assert got.ok, got.payload

    euler = math.pi ** 2 * E * min(IY, IZ) / length ** 2
    assert got.payload["critical_factor"] * load == pytest.approx(euler, rel=1e-2)
    assert got.payload["critical_factor"] * load >= euler * (1 - 1e-9), "应从上方逼近"
    assert got.payload["most_compressed_member"] == 1, "报物理杆件编号，不报加密单元编号"
    assert "mesh_warning" not in got.payload


def test_pdelta_on_a_single_member_cantilever_matches_the_exact_solution():
    """δ = H/(P·k)·(tan kL − kL)，k = √(P/EI)。不加节点，靠默认加密。"""
    length, axial, lateral = 3.0, 2.0e5, 1.0e3
    model = _beam(length, supports=[{"node": 1, "fix": [1] * 6}],
                  nodal_loads=[{"node": 2, "load": [-axial, 0, lateral, 0, 0, 0]}])
    session = Session(model=model)
    got = session.solve_model(analysis="pdelta", increments=10, tolerance=1e-10)
    assert got.ok, got.payload

    # 水平杆件局部 y 朝上（全局 +Z），z 向侧力是绕局部 z 的弯曲，用 Iz
    k = math.sqrt(axial / (E * IZ))
    exact = lateral / (axial * k) * (math.tan(k * length) - k * length)
    tip = _node_u(session.frame, session.solution, 2)[2]
    assert tip == pytest.approx(exact, rel=1e-3)


def test_eigen_results_do_not_depend_on_the_unit_system():
    si = _beam(8.0, nodal_loads=[{"node": 2, "load": [-1e4, 0, 0, 0, 0, 0]}])
    mm = convert_model(deepcopy(si), "N-mm-MPa")
    modal_si = Session(model=si).modal_analysis(num_modes=2).payload
    modal_mm = Session(model=mm).modal_analysis(num_modes=2).payload
    assert modal_mm["modes"][0]["frequency_Hz"] == pytest.approx(
        modal_si["modes"][0]["frequency_Hz"], rel=1e-9)
    lam_si = Session(model=si).buckling_analysis().payload["critical_factor"]
    lam_mm = Session(model=mm).buckling_analysis().payload["critical_factor"]
    assert lam_mm == pytest.approx(lam_si, rel=1e-6)


# --------------------------------------------------------------- 工具与清理

def test_seed_mesh_tool_accepts_a_set_name_and_can_clear():
    session = Session(model=_portal())
    assert session.define_set(name="柱", member_ids=[1, 3]).ok

    got = session.seed_mesh(divisions=6, members="柱")
    assert got.ok, got.payload
    assert got.payload["member_divisions"] == {1: 6, 3: 6}
    assert session.model["mesh"] == {"members": {"1": 6, "3": 6}}

    assert session.seed_mesh(divisions=3).ok
    assert session.model["mesh"]["divisions"] == 3

    assert session.seed_mesh(clear=True, members=[1]).ok
    assert session.model["mesh"]["members"] == {"3": 6}
    assert session.seed_mesh(clear=True).ok
    assert "mesh" not in session.model

    assert not session.seed_mesh(divisions=0).ok
    assert not session.seed_mesh(divisions=4, members=[99]).ok


def test_deleting_a_member_prunes_its_seed():
    """不清的话下一次校验会因悬空布种拒绝整个模型，报错却不指向删杆那一步。"""
    session = Session(model=_portal())
    assert session.seed_mesh(divisions=4, members=[2]).ok
    assert session.remove_members([2]).ok
    assert "members" not in (session.model.get("mesh") or {})
    assert not [e for e in session.validation_errors() if "mesh" in e]
