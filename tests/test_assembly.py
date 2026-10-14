"""Assembly 模块：独立实例、线性阵列、环形阵列。

防的问题：原来要建多跨、多榀的刚架只能逐个点节点连杆件，或者靠专用的门架生成器；
截面指派、梁方向不会随复制的杆件走，转过角度的副本截面朝向还是原来的方向。
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from agent import Session
from model_io import validate_payload
from units import convert_model

E, IY, IZ = 2.06e11, 2e-5, 1.5e-4


def _materials(s: Session) -> None:
    assert s.define_materials_and_sections(
        [{"name": "Q355", "E": E, "nu": 0.3, "density": 7850.0}],
        [{"name": "H", "A": 8e-3, "Iy": IY, "Iz": IZ, "J": 5e-7}]).ok


def _bay() -> Session:
    """一跨门架：柱 1(1-2)、2(3-4)，梁 3(2-4)，跨度 6、高 4；截面用对象指派。"""
    s = Session()
    assert s.add_nodes([[0, 0, 0], [0, 0, 4], [6, 0, 0], [6, 0, 4]]).ok
    _materials(s)
    assert s.add_members([[1, 2], [3, 4], [2, 4]]).ok
    assert s.define_set(name="一跨", member_ids=[1, 2, 3]).ok
    assert s.create_section_assignment("SA", "一跨", "H", "Q355").ok
    return s


def _three_bays_by_hand() -> Session:
    s = Session()
    assert s.add_nodes([[x, 0, z] for x in (0, 6, 12, 18) for z in (0, 4)]).ok
    _materials(s)
    cols = [[2 * k + 1, 2 * k + 2] for k in range(4)]
    beams = [[2 * k + 2, 2 * k + 4] for k in range(3)]
    assert s.add_members(cols + beams).ok
    assert s.create_section_assignment("SA", [m["id"] for m in s.model["members"]], "H", "Q355").ok
    return s


def _load_and_solve(s: Session, feet, tops) -> np.ndarray:
    assert s.create_boundary_condition("柱脚", feet, preset="ENCASTRE").ok
    assert s.set_load_cases([{"name": "W"}]).ok
    assert s.create_load("风", "W", "concentrated_force", tops, force=[1e4, 0, -2e4]).ok
    got = s.solve_model()
    assert got.ok, got.payload
    return s.solution.cases["W"].U


def _ids_at(s: Session, z: float) -> list[int]:
    return sorted(int(n["id"]) for n in s.model["nodes"] if n["z"] == pytest.approx(z))


def test_linear_pattern_builds_the_same_frame_as_by_hand_and_shares_columns():
    """沿 x 阵列一跨门架共 3 跨：相邻跨共用的柱只留一根，结果与手工建的三跨门架一致。"""
    s = _bay()
    got = s.linear_pattern("Pat", "一跨", [1, 0, 0], 6.0, 3)
    assert got.ok, got.payload
    assert len(s.model["members"]) == 3 + 2 * 2, "每份副本的左柱与上一跨右柱重合，跳过"
    assert all(len(v) == 1 for v in got.payload["skipped_overlapping"].values())
    assert any(f["name"].startswith("SA-") for f in s.model["features"]["section_assignments"])
    u_pattern = _load_and_solve(s, _ids_at(s, 0), _ids_at(s, 4))
    hand = _three_bays_by_hand()
    u_hand = _load_and_solve(hand, _ids_at(hand, 0), _ids_at(hand, 4))

    def by_xyz(session, u):
        idx = session.frame.index_of()
        return {(n["x"], n["z"]): u[6 * idx[int(n["id"])]:6 * idx[int(n["id"])] + 6]
                for n in session.model["nodes"]}

    a, b = by_xyz(s, u_pattern), by_xyz(hand, u_hand)
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-9, atol=1e-15)


def test_radial_pattern_rotates_beam_orientation_with_the_copy():
    """绕 z 每 90° 复制悬臂梁：n1 跟着转，四根悬臂端部挠度都是 PL³/(3E·Iy)。

    n1 不转的话，转 90° 的副本 n1 与杆轴平行，截面朝向无法确定。
    """
    s = Session()
    assert s.add_nodes([[1, 0, 0], [3, 0, 0]]).ok
    _materials(s)
    assert s.add_members([[1, 2]]).ok
    assert s.create_section_assignment("SA", [1], "H", "Q355").ok
    assert s.create_beam_orientation("Ori", [1], [0, 1, 0]).ok
    got = s.radial_pattern("Rad", [1], 4, 90.0, [0, 0, 1])
    assert got.ok, got.payload
    n1 = {f["name"]: f["n1"] for f in s.model["features"]["beam_orientations"]}
    assert n1["Ori-Rad"] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)
    roots = [n for n in s.model["nodes"] if np.hypot(n["x"], n["y"]) == pytest.approx(1.0)]
    tips = [n for n in s.model["nodes"] if np.hypot(n["x"], n["y"]) == pytest.approx(3.0)]
    assert len(roots) == len(tips) == 4
    assert s.create_boundary_condition("根部", [int(n["id"]) for n in roots], preset="ENCASTRE").ok
    assert s.set_load_cases([{"name": "P"}]).ok
    P, L = -5e3, 2.0
    assert s.create_load("端部", "P", "concentrated_force", [int(n["id"]) for n in tips],
                         force=[0, 0, P]).ok
    assert s.solve_model().ok
    idx = s.frame.index_of()
    for tip in tips:
        uz = s.solution.cases["P"].U[6 * idx[int(tip["id"])] + 2]
        assert uz == pytest.approx(P * L**3 / (3 * E * IY), rel=1e-9)


def test_delete_pattern_restores_the_model_and_units_convert():
    """删整个阵列：新杆件、新节点、带过去的截面指派一并撤掉，模型回到阵列之前。"""
    s = _bay()
    before = deepcopy(s.model)
    assert s.linear_pattern("Pat", "一跨", [1, 0, 0], 6.0, 3).ok
    assert s.create_instance("上层", "一跨", translate=[0, 0, 4]).ok
    mm = convert_model(deepcopy(s.model), "N-mm-MPa")
    assert mm["assembly"]["instances"][-1]["transform"]["translate"] == pytest.approx([0, 0, 4000])
    assert not [e for e in validate_payload(mm) if "装配" in e or "assembly" in e]
    assert s.delete_instance("上层").ok
    assert s.delete_instance("Pat").ok
    assert s.model == before
    assert not s.delete_instance("Pat").ok
    assert not s.create_instance("零", "一跨").ok, "零变换的副本与原件完全重合"
