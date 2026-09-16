"""对称性检测（讲义 §3-9 四）。

讲义讲的是**利用**对称性取半结构算；本项目只检测、不利用，理由写在
`src/symmetry.py` 的模块说明里。检测出来的用途是**当校核手段**：
对称结构 + 对称荷载，对称位置的位移必须互为镜像。

这一组里有三条盯的是"检测不出来"这种失败方式——它不报错，只是安静地
少给一个结论：

* 力矩按力的规则镜像（少了那个负号），对称的弯矩荷载会被判成不对称；
* 跨过对称面的杆件镜像后 i、j 调个儿，梯形荷载两端没对调也会判错；
* 而均布荷载**不能**跟着对调，它的 w2 是个占位零，一调就把强度换成了零。
"""

from __future__ import annotations

import numpy as np

from frame3d import solve
from model_compiler import compile_model
from model_io import from_dict
from symmetry import (ANTISYMMETRIC, NEITHER, SYMMETRIC, Plane,
                      check_symmetric_response, detect_symmetry)

MAT = [{"name": "M", "E": 2.06e11, "nu": 0.3}]
SEC = [{"name": "C", "A": 1.2e-2, "Iy": 1.6e-4, "Iz": 1.6e-4, "J": 3.2e-4},
       {"name": "B", "A": 8.0e-3, "Iy": 1.0e-4, "Iz": 6.0e-5, "J": 1.2e-4}]


def _portal(cases) -> dict:
    """单跨门式：柱 1-2、梁 2-3、柱 4-3。关于 x = 3 对称。"""
    return {
        "units": "N-m-Pa", "materials": MAT, "sections": SEC,
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 0, "y": 0, "z": 4.0},
                  {"id": 3, "x": 6.0, "y": 0, "z": 4.0},
                  {"id": 4, "x": 6.0, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "C", "material": "M",
                     "ref_vector": [1, 0, 0]},
                    {"id": 2, "i": 2, "j": 3, "section": "B", "material": "M"},
                    {"id": 3, "i": 4, "j": 3, "section": "C", "material": "M",
                     "ref_vector": [1, 0, 0]}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": 4, "fix": [1, 1, 1, 1, 1, 1]}],
        "load_cases": cases,
    }


def _frame(cases):
    return compile_model(_portal(cases)).analysis_model


def _plane_named(found, name):
    return next((p for p in found["planes"] if p["plane"] == name), None)


# ------------------------------------------------------------------ 结构

def test_a_symmetric_portal_is_recognised():
    found = detect_symmetry(_frame([{"name": "D", "member_loads":
                                     [{"member": 2, "w": [0, 0, -20e3]}]}]))
    p = _plane_named(found, "x = 3")
    assert p is not None and p["structure"]
    assert p["node_pairs"][1] == 4 and p["node_pairs"][2] == 3
    assert p["member_pairs"][1] == 3 and p["member_pairs"][2] == 2


def test_a_different_section_on_one_column_breaks_it():
    payload = _portal([{"name": "D", "member_loads":
                        [{"member": 2, "w": [0, 0, -20e3]}]}])
    payload["members"][2] = {**payload["members"][2], "section": "B"}
    found = detect_symmetry(compile_model(payload).analysis_model)
    assert _plane_named(found, "x = 3") is None
    assert "未检测到" in found["advice"]


def test_a_different_support_breaks_it():
    payload = _portal([{"name": "D", "member_loads":
                        [{"member": 2, "w": [0, 0, -20e3]}]}])
    payload["supports"][1] = {"node": 4, "fix": [1, 1, 1, 0, 0, 0]}
    found = detect_symmetry(compile_model(payload).analysis_model)
    assert _plane_named(found, "x = 3") is None


def test_a_hinge_on_only_one_column_breaks_it():
    payload = _portal([{"name": "D", "member_loads":
                        [{"member": 2, "w": [0, 0, -20e3]}]}])
    payload["members"][0] = {**payload["members"][0],
                             "releases": {"i": ["ry", "rz"]}}
    found = detect_symmetry(compile_model(payload).analysis_model)
    assert _plane_named(found, "x = 3") is None


# ------------------------------------------------------------------ 荷载

def test_uniform_gravity_is_symmetric():
    found = detect_symmetry(_frame([{"name": "D", "member_loads":
                                     [{"member": 2, "w": [0, 0, -20e3]}]}]))
    assert _plane_named(found, "x = 3")["cases"]["D"] == SYMMETRIC


def test_a_single_side_sway_load_is_neither():
    found = detect_symmetry(_frame([{"name": "W", "nodal_loads":
                                     [{"node": 2, "load":
                                       [30e3, 0, 0, 0, 0, 0]}]}]))
    assert _plane_named(found, "x = 3")["cases"]["W"] == NEITHER


def test_two_inward_forces_are_symmetric_and_two_same_way_are_antisymmetric():
    """力垂直于镜面的分量变号：**对着推**才是对称，**同向推**是反对称。

    直觉容易反过来——「两边一样大同一个方向」看着才像对称。
    """
    inward = detect_symmetry(_frame([{"name": "A", "nodal_loads": [
        {"node": 2, "load": [30e3, 0, 0, 0, 0, 0]},
        {"node": 3, "load": [-30e3, 0, 0, 0, 0, 0]}]}]))
    assert _plane_named(inward, "x = 3")["cases"]["A"] == SYMMETRIC

    same_way = detect_symmetry(_frame([{"name": "A", "nodal_loads": [
        {"node": 2, "load": [30e3, 0, 0, 0, 0, 0]},
        {"node": 3, "load": [30e3, 0, 0, 0, 0, 0]}]}]))
    assert _plane_named(same_way, "x = 3")["cases"]["A"] == ANTISYMMETRIC


def test_moments_mirror_the_other_way_round():
    """力矩是轴矢量：垂直镜面的分量**不**变号，平行的变号。

    对 x = 3 这个面，绕 z 的力矩（平行于镜面）镜像后要变号。所以一对
    **反号**的 Mz 才是对称荷载。按力的规则去镜像，这一条会判成反对称——
    不报错，只是结论反了。
    """
    got = detect_symmetry(_frame([{"name": "M", "nodal_loads": [
        {"node": 2, "load": [0, 0, 0, 0, 0, 10e3]},
        {"node": 3, "load": [0, 0, 0, 0, 0, -10e3]}]}]))
    assert _plane_named(got, "x = 3")["cases"]["M"] == SYMMETRIC

    same = detect_symmetry(_frame([{"name": "M", "nodal_loads": [
        {"node": 2, "load": [0, 0, 0, 0, 0, 10e3]},
        {"node": 3, "load": [0, 0, 0, 0, 0, 10e3]}]}]))
    assert _plane_named(same, "x = 3")["cases"]["M"] == ANTISYMMETRIC


def test_the_mirror_rules_themselves():
    plane = Plane(0, 3.0)                       # x = 3
    assert list(plane.mirror_force([2.0, 5.0, 7.0])) == [-2.0, 5.0, 7.0]
    assert list(plane.mirror_moment([2.0, 5.0, 7.0])) == [2.0, -5.0, -7.0]
    assert list(plane.mirror_point(np.array([1.0, 0.0, 0.0]))) == [5.0, 0.0, 0.0]


# ------------------------------------- 跨过对称面的杆件：i、j 调个儿的那一档

def _single_beam(loads) -> dict:
    """一根 0→12 的梁，对称面 x = 6 从它中间穿过——镜像会把 i、j 调个儿。"""
    return {
        "units": "N-m-Pa", "materials": MAT, "sections": SEC,
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 12.0, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "B", "material": "M"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 0]},
                     {"node": 2, "fix": [1, 1, 1, 1, 1, 0]}],
        "load_cases": [{"name": "D", **loads}],
    }


def test_a_uniform_load_on_a_beam_across_the_plane_stays_symmetric():
    """回归：均布的 w2 是个占位零，跟着 i/j 对调会把强度换成零。

    这一条挂掉的表现是"重力荷载被判成不对称"——不报错，只是少给一个结论。
    """
    frame = compile_model(_single_beam(
        {"member_loads": [{"member": 1, "w": [0, 0, -20e3]}]})).analysis_model
    assert _plane_named(detect_symmetry(frame), "x = 6")["cases"]["D"] == SYMMETRIC


def test_a_trapezoid_load_across_the_plane_needs_its_ends_swapped():
    """两端强度相同的梯形（其实就是均布）对称；两端不同的不对称。

    这两条走**物理模型**（from_dict，不剖分）：跨间荷载会触发自动剖分，
    剖分点一旦不在对称面上，分析模型本身就不再对称了——见下一条。
    """
    even = from_dict(_single_beam({"member_spans": [
        {"member": 1, "kind": "trapezoid", "w1": [0, 0, -20e3],
         "w2": [0, 0, -20e3]}]}))
    assert _plane_named(detect_symmetry(even), "x = 6")["cases"]["D"] == SYMMETRIC

    sloped = from_dict(_single_beam({"member_spans": [
        {"member": 1, "kind": "trapezoid", "w1": [0, 0, -20e3],
         "w2": [0, 0, -5e3]}]}))
    assert _plane_named(detect_symmetry(sloped), "x = 6")["cases"]["D"] == NEITHER


def test_a_midspan_point_load_is_symmetric_but_an_offset_one_is_not():
    mid = from_dict(_single_beam({"member_spans": [
        {"member": 1, "kind": "point", "w1": [0, 0, -50e3], "a": 6.0}]}))
    assert _plane_named(detect_symmetry(mid), "x = 6")["cases"]["D"] == SYMMETRIC

    off = from_dict(_single_beam({"member_spans": [
        {"member": 1, "kind": "point", "w1": [0, 0, -50e3], "a": 4.0}]}))
    assert _plane_named(detect_symmetry(off), "x = 6")["cases"]["D"] == NEITHER


def test_an_off_centre_split_point_destroys_the_symmetry_of_the_analysis_model():
    """偏心集中力会在 x=4 生成一个内节点，**分析模型**从此不再几何对称。

    物理模型对称、分析模型不对称——这不是 bug，是剖分的必然结果。写成测试
    是为了记住：对着分析模型查对称性会漏掉一些情形，要查物理模型。
    """
    payload = _single_beam({"member_spans": [
        {"member": 1, "kind": "point", "w1": [0, 0, -50e3], "a": 4.0}]})
    physical = from_dict(payload)
    analysis = compile_model(payload).analysis_model
    assert _plane_named(detect_symmetry(physical), "x = 6") is not None
    assert _plane_named(detect_symmetry(analysis), "x = 6") is None


# ------------------------------------------------------------------ 当校核用

def test_a_symmetric_case_gives_a_mirrored_displacement_field():
    frame = _frame([{"name": "D", "member_loads":
                     [{"member": 2, "w": [0, 0, -20e3]}]}])
    got = check_symmetric_response(frame, solve(frame), "D")
    assert got["ok"] and got["checks"]
    assert got["checks"][0]["kind"] == SYMMETRIC
    assert got["checks"][0]["max_relative_difference"] < 1e-12


def test_the_response_check_actually_catches_a_broken_answer():
    """把解人为改坏一点点，校核必须发现。否则这条校核等于没有。"""
    frame = _frame([{"name": "D", "member_loads":
                     [{"member": 2, "w": [0, 0, -20e3]}]}])
    solution = solve(frame)
    U = solution["D"].U
    U[frame.node_dofs(2)[2]] += 0.05 * max(float(np.abs(U).max()), 1e-9)
    got = check_symmetric_response(frame, solution, "D")
    assert not got["ok"]
    assert got["checks"][0]["worst_pair"] in ((2, 3), (3, 2))


def test_an_asymmetric_case_is_skipped_rather_than_failed():
    frame = _frame([{"name": "W", "nodal_loads":
                     [{"node": 2, "load": [30e3, 0, 0, 0, 0, 0]}]}])
    got = check_symmetric_response(frame, solve(frame), "W")
    assert got["ok"] and got["checks"] == []
    assert "没有可用于校核的对称面" in got["note"]
