"""截面特性计算的验证。

最后一条最关键：把算出来的截面喂回求解器，用悬臂梁的闭合解验证
**强弱轴没有搞反**。Iy/Iz 互换不会让任何公式报错，只会让挠度差出数量级——
这种错误只有闭环验证才抓得住。
"""
import math

import pytest

from frame3d import Frame, Material, Member, Node, Section, solve
from sections import (BUILDERS, DEFAULT_DIMENSIONS, circular_tube, i_section,
                      rectangle, rectangle_torsion_factor, solid_circle)


def test_rectangle_matches_the_closed_form():
    b, h = 0.2, 0.4
    s = rectangle("R", b, h)
    assert s["A"] == pytest.approx(b * h)
    assert s["Iz"] == pytest.approx(b * h ** 3 / 12)      # 强轴
    assert s["Iy"] == pytest.approx(h * b ** 3 / 12)      # 弱轴
    assert s["Iz"] > s["Iy"], "高大于宽时强轴必须是 Iz"


def test_square_has_equal_principal_inertias():
    s = rectangle("S", 0.3, 0.3)
    assert s["Iy"] == pytest.approx(s["Iz"])
    assert rectangle_torsion_factor(1.0) == pytest.approx(0.141)


def test_torsion_factor_interpolates_and_saturates():
    assert rectangle_torsion_factor(2.0) == pytest.approx(0.229)
    assert rectangle_torsion_factor(1.75) == pytest.approx((0.196 + 0.229) / 2)
    assert rectangle_torsion_factor(500.0) == pytest.approx(1 / 3)   # 趋近薄板
    assert rectangle_torsion_factor(0.5) == pytest.approx(0.141)     # 端点夹住


def test_i_section_matches_the_closed_form():
    h, b, tw, tf = 0.4, 0.2, 0.010, 0.016
    s = i_section("H", h, b, tw, tf)
    hw = h - 2 * tf
    assert s["A"] == pytest.approx(2 * b * tf + hw * tw)
    assert s["Iz"] == pytest.approx((b * h ** 3 - (b - tw) * hw ** 3) / 12)
    assert s["Iy"] == pytest.approx((2 * tf * b ** 3 + hw * tw ** 3) / 12)
    assert s["J"] == pytest.approx((2 * b * tf ** 3 + hw * tw ** 3) / 3)


def test_i_section_is_far_stiffer_about_the_strong_axis():
    s = i_section("H", 0.4, 0.2, 0.010, 0.016)
    assert s["Iz"] / s["Iy"] > 10


def test_i_section_rejects_impossible_dimensions():
    with pytest.raises(ValueError, match="翼缘太厚"):
        i_section("H", 0.03, 0.2, 0.01, 0.02)
    with pytest.raises(ValueError, match="腹板厚"):
        i_section("H", 0.4, 0.01, 0.02, 0.016)


def test_tube_is_doubly_symmetric_and_closed():
    s = circular_tube("T", 0.3, 0.010)
    assert s["Iy"] == pytest.approx(s["Iz"])
    assert s["J"] == pytest.approx(2 * s["Iz"]), "闭口圆截面 J 等于极惯性矩"
    with pytest.raises(ValueError, match="壁厚太大"):
        circular_tube("T", 0.02, 0.02)


def test_solid_circle_matches_the_closed_form():
    d = 0.25
    s = solid_circle("C", d)
    assert s["A"] == pytest.approx(math.pi * d ** 2 / 4)
    assert s["Iz"] == pytest.approx(math.pi * d ** 4 / 64)


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_every_preset_builds_a_usable_section(kind):
    builder, keys, _labels = BUILDERS[kind]
    values = DEFAULT_DIMENSIONS[kind]
    assert len(values) == len(keys)
    s = builder("X", *values)
    base = {"name", "A", "Iy", "Iz", "J", "cy", "cz", "circular"}
    # 剪应力 τ=V·S/(I·b) 要的两组几何，每种截面都必须给全，
    # 否则折算应力算不了而调用方只会拿到一句"缺几何"。
    shear = {"Sz", "bz", "Sy", "by"}
    # 腹板边缘那一对只有明确区分腹板与翼缘的截面才有；
    # 矩形和圆没有腹板，留空是对的，不能硬凑一个数出来。
    web = {"S_flange", "c_web"}
    assert set(s) in (base | shear, base | shear | web), sorted(set(s))
    assert all(s[k] > 0 for k in (base | shear | web) & set(s) - {"name",
                                                                  "circular"})
    assert isinstance(s["circular"], bool)


def test_computed_section_reproduces_the_cantilever_closed_form():
    """闭环：算出的 Iz 喂回求解器，端部挠度必须等于 PL³/(3 E Iz)。

    Iy 与 Iz 互换不会让任何公式报错，只会让挠度差出数量级——只有这条抓得住。
    """
    E, L, P = 2.1e11, 4.0, 10e3
    props = rectangle("R", 0.2, 0.4)
    f = Frame()
    f.nodes[1] = Node(1, 0.0, 0.0, 0.0)
    f.nodes[2] = Node(2, L, 0.0, 0.0)
    f.members[1] = Member(1, 1, 2, "R", "STEEL")
    f.sections["R"] = Section("R", props["A"], props["Iy"], props["Iz"], props["J"])
    f.materials["STEEL"] = Material("STEEL", E, 0.3)
    f.supports[1] = (1, 1, 1, 1, 1, 1)
    f.nodal_loads[2] = (0.0, 0.0, -P, 0.0, 0.0, 0.0)      # 竖向力 -> 强轴弯曲
    sol = solve(f)
    assert sol.U[f.node_dofs(2)[2]] == pytest.approx(
        -P * L ** 3 / (3 * E * props["Iz"]), rel=1e-10)

    f.nodal_loads[2] = (0.0, -P, 0.0, 0.0, 0.0, 0.0)      # 侧向力 -> 弱轴弯曲
    sol = solve(f)
    assert sol.U[f.node_dofs(2)[1]] == pytest.approx(
        -P * L ** 3 / (3 * E * props["Iy"]), rel=1e-10)
