"""弯扭屈曲（翘曲自由度 + 弯矩耦合）的回归。

真值来自经典薄壁梁稳定理论（Timoshenko & Gere《弹性稳定理论》第 6 章；
Trahair《Flexural-Torsional Buckling of Structures》1993）：

* 叉支座简支梁受均匀弯矩：Mcr = (π/L)·√(EIy·GJ·(1 + π²EIw/(GJ·L²)))；
* 同时受轴压 P：M² = r0²·(Py − P)·(Pθ − P)，Py = π²EIy/L²，
  Pθ = (GJ + π²EIw/L²)/r0²，r0² = (Iy + Iz)/A；
* 均布荷载作用在剪心：Mcr ≈ 1.13 × 均匀弯矩值（等效弯矩系数 C1）。
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("gmsh", reason="翘曲常数由截面网格求得，需要 Gmsh")

import sections as S  # noqa: E402
import section_solver as ss  # noqa: E402
from agent import Session  # noqa: E402
from buckling import continuous_warping_groups  # noqa: E402
from units import convert_model  # noqa: E402

E, NU, LENGTH, MOMENT = 2.06e11, 0.3, 8.0, 1.0e5
G = E / (2 * (1 + NU))
SECTION = S.i_section("H", 0.4, 0.2, 0.008, 0.013)


def _beam(n: int, *, section=SECTION, axial=0.0, moments=(MOMENT, -MOMENT), udl=None) -> dict:
    """沿全局 X 的简支梁，分成 n 根物理杆件；两端叉支座（不能横移、不能扭转）。"""
    loads = {"name": "LC", "nodal_loads": [
        {"node": 1, "load": [0, 0, 0, 0, moments[0], 0]},
        {"node": n + 1, "load": [-axial, 0, 0, 0, moments[1], 0]}]}
    if udl is not None:
        loads = {"name": "LC", "member_loads": [{"member": k + 1, "w": [0, 0, -udl]}
                                                for k in range(n)]}
    return {"units": "N-m-Pa",
            "materials": [{"name": "S", "E": E, "nu": NU, "density": 7850.0}],
            "sections": [section],
            "nodes": [{"id": k + 1, "x": LENGTH * k / n, "y": 0, "z": 0} for k in range(n + 1)],
            "members": [{"id": k + 1, "i": k + 1, "j": k + 2, "section": section["name"],
                         "material": "S"} for k in range(n)],
            "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                         {"node": n + 1, "fix": [0, 1, 1, 1, 0, 0]}],
            "load_cases": [loads]}


def _factor(model: dict, **kw) -> float:
    result = Session(model=model).buckling_analysis(case="LC", num_modes=1,
                                                    lateral_torsional=True, **kw)
    assert result.ok, result.payload
    return result.payload["critical_factor"]


def _iw() -> float:
    return ss.solve_section(SECTION["geometry"])["Iw"]


def _uniform_moment_mcr(iw: float) -> float:
    return math.pi / LENGTH * math.sqrt(
        E * SECTION["Iy"] * G * SECTION["J"] * (1 + math.pi**2 * E * iw / (G * SECTION["J"] * LENGTH**2)))


def test_uniform_moment_matches_the_classical_critical_moment_from_above():
    """防止受弯钢梁的侧向弯扭失稳算不出来：原屈曲只随轴力变化，纯弯梁直接被拒绝。

    均匀弯矩下与经典 Mcr 一致，且网格加密时从上方单调逼近。
    """
    exact = _uniform_moment_mcr(_iw())
    values = [_factor(_beam(n)) * MOMENT for n in (1, 2, 4)]
    assert values[-1] == pytest.approx(exact, rel=1e-5)
    assert values[0] > values[1] > values[2] > exact


def test_axial_compression_lowers_the_critical_moment_as_the_interaction_equation_says():
    """轴压与弯矩同时作用：(λM)² = r0²(Py − λP)(Pθ − λP)，检验轴力项与 Wagner 项的符号。"""
    P = 2.0e5
    iw = _iw()
    A, Iy, Iz, J = SECTION["A"], SECTION["Iy"], SECTION["Iz"], SECTION["J"]
    r0 = (Iy + Iz) / A
    py = math.pi**2 * E * Iy / LENGTH**2
    pt = (G * J + math.pi**2 * E * iw / LENGTH**2) / r0
    a, b, c = MOMENT**2 - r0 * P**2, r0 * P * (py + pt), -r0 * py * pt
    exact = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert _factor(_beam(4, axial=P)) == pytest.approx(exact, rel=1e-5)


def test_uniformly_loaded_beam_has_the_classical_moment_gradient_factor():
    """均布荷载作用在剪心：跨中弯矩的临界值约为均匀弯矩时的 1.13 倍（C1）。"""
    q = 1.0e4
    c1 = _factor(_beam(4, udl=q)) * q * LENGTH**2 / 8 / _uniform_moment_mcr(_iw())
    assert c1 == pytest.approx(1.13, rel=1e-2)


def test_axially_loaded_column_gives_the_same_factor_with_or_without_warping():
    """没有弯矩时弯扭屈曲路径退化为普通屈曲：弱轴欧拉荷载两条路径相同。"""
    model = _beam(4, axial=1.0e5, moments=(0.0, 0.0))
    plain = Session(model=model).buckling_analysis(case="LC", num_modes=1)
    assert plain.ok
    assert _factor(model) == pytest.approx(plain.payload["critical_factor"], rel=1e-9)


def test_restraining_warping_at_the_supports_raises_the_critical_moment():
    """嵌固端约束翘曲后临界弯矩提高；默认杆端翘曲自由，偏保守。"""
    free = _factor(_beam(4))
    restrained = _factor(_beam(4), warping_restrained_nodes=[1, 5])
    assert restrained > 1.2 * free


def test_lateral_torsional_buckling_is_unit_invariant():
    """米制与毫米制同一根梁，弯扭屈曲因子逐位一致到 1e-9。"""
    model = _beam(2)
    assert _factor(convert_model(model, "N-mm-MPa")) == pytest.approx(_factor(model), rel=1e-9)


def test_offset_shear_centre_is_refused():
    """防止对槽钢拿双轴对称公式凑数：剪心不在形心时需要偏置与 Wagner 项，必须拒绝。"""
    channel = {"name": "C", "A": 1.0, "Iy": 1.0, "Iz": 1.0, "J": 1.0,
               "geometry": {"type": "channel",
                            "dimensions": {"h": .25, "b": .09, "tw": .008, "tf": .012}}}
    channel.update({k: v for k, v in S.i_section("tmp", .25, .09, .008, .012).items()
                    if k in ("Iy", "Iz", "J")})
    import section_geometry as sg
    channel["A"] = sg.profile(channel["geometry"])["area"]
    result = Session(model=_beam(1, section=channel)).buckling_analysis(
        case="LC", lateral_torsional=True)
    assert not result.ok
    assert "剪心偏离形心" in result.payload["error"]


def test_section_without_real_geometry_is_refused_but_round_sections_work():
    """只有 A/I/J 的非圆截面翘曲常数无从得知，拒绝；圆截面不翘曲，Iw=0 直接可算。"""
    bare = {k: SECTION[k] for k in ("A", "Iy", "Iz", "J")} | {"name": "H"}
    refused = Session(model=_beam(1, section=bare)).buckling_analysis(
        case="LC", lateral_torsional=True)
    assert not refused.ok and "没有真实轮廓" in refused.payload["error"]
    tube = {k: v for k, v in S.circular_tube("T", 0.2, 0.01).items() if k != "geometry"}
    assert _factor(_beam(1, section=tube)) > 0


def test_warping_is_continuous_along_straight_runs_and_free_at_corners():
    """翘曲连续性按物理条件判断：直梁拆段处连续，梁柱转角处断开。"""
    from model_compiler import compile_model

    model = _beam(2)
    model["nodes"].append({"id": 4, "x": LENGTH, "y": 0, "z": 3.0})
    model["members"].append({"id": 3, "i": 3, "j": 4, "section": "H", "material": "S"})
    compiled = compile_model(Session(model=model).model, validated=True, min_divisions=4)
    groups = continuous_warping_groups(compiled.analysis_model)
    physical = {e: compiled.mapping.physical_member_id(e) for e in groups}
    beam = {groups[e] for e in groups if physical[e] in (1, 2)}
    column = {groups[e] for e in groups if physical[e] == 3}
    assert sum(physical[e] in (1, 2) for e in groups) == 8      # 两根梁各剖 4 段
    assert len(beam) == 1 and len(column) == 1 and beam.isdisjoint(column)
