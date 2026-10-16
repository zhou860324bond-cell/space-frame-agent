"""Timoshenko 有效剪切面积的闭合解、文献公式与单位不变性。

防的真实问题：截面对话框曾对所有形状一律取 Ay=Az=5A/6。这只对实心矩形成立；
默认 300×150×8×12 工字钢强轴剪切面积真实约为腹板面积，5A/6 大了 2.2 倍，
短跨、深梁的剪切变形被低估，而且不报错。

真值全部来自经典理论或文献，不来自本项目另一份实现：
Cowper (1966) 的剪切系数闭合式；Oasys GSA 手册“Thin walled sections”给出的
薄壁工字形、槽形、T 形剪切面积公式 As = F·d·t/6；手推的薄壁等边角钢 5at/6。
"""

from __future__ import annotations

import math

import pytest

from frame3d import Frame, Material, Member, Node, Section, solve
from section_geometry import STEEL_POISSON, scale_geometry, shear_areas

E = 2.06e11
NU = 0.3
G = E / (2 * (1 + NU))

I_DIMS = {"h": 0.30, "b": 0.15, "tw": 0.008, "tf": 0.012}
ALL_SHAPES = [
    {"type": "rectangle", "dimensions": {"b": 0.2, "h": 0.3}},
    {"type": "circle", "dimensions": {"d": 0.05}},
    {"type": "tube", "dimensions": {"d": 0.168, "t": 0.008}},
    {"type": "box", "dimensions": {"b": 0.2, "h": 0.3, "t": 0.008}},
    {"type": "i", "dimensions": dict(I_DIMS)},
    {"type": "channel", "dimensions": {"h": 0.25, "b": 0.09, "tw": 0.008, "tf": 0.012}},
    {"type": "tee", "dimensions": {"h": 0.20, "b": 0.15, "tw": 0.008, "tf": 0.012}},
    {"type": "angle", "dimensions": {"b": 0.075, "t": 0.008}},
]


def _gsa(f_value, d, t):
    """GSA 手册薄壁剪切面积 As = F·d·t/6，d 为翼缘中线间距（T 形为翼缘中线到腹板端）。"""
    return f_value * d * t / 6


def test_rectangle_is_five_sixths_at_zero_poisson_and_cowper_otherwise():
    """实心矩形：ν=0 时为 Timoshenko 的 5/6，ν=0.3 时为 Cowper 10(1+ν)/(12+11ν)。"""
    g = {"type": "rectangle", "dimensions": {"b": 0.2, "h": 0.3}}
    assert shear_areas(g, nu=0.0)["Ay"] == pytest.approx(5 / 6 * 0.06, rel=1e-13)
    got = shear_areas(g, nu=0.3)
    assert got["Ay"] == got["Az"] == pytest.approx(10 * 1.3 / 15.3 * 0.06, rel=1e-13)
    assert "Cowper" in got["formula"]


def test_solid_circle_and_tube_follow_cowper():
    """实心圆 6(1+ν)/(7+6ν)；圆管的 Cowper 圆环式在 m→1 时趋于薄壁 2(1+ν)/(4+3ν)。"""
    d = 0.05
    circle = shear_areas({"type": "circle", "dimensions": {"d": d}}, nu=NU)
    assert circle["Ay"] == pytest.approx(6 * 1.3 / 8.8 * math.pi * d**2 / 4, rel=1e-13)

    D, t = 0.168, 0.008
    m = (D - 2 * t) / D
    k = 6 * 1.3 * (1 + m**2)**2 / (8.8 * (1 + m**2)**2 + 23.6 * m**2)
    tube = shear_areas({"type": "tube", "dimensions": {"d": D, "t": t}}, nu=NU)
    assert tube["Ay"] == tube["Az"] == pytest.approx(
        k * math.pi * (D**2 - (D - 2 * t)**2) / 4, rel=1e-13)
    thin = shear_areas({"type": "tube", "dimensions": {"d": 1.0, "t": 1e-5}}, nu=NU)
    area = math.pi * (1.0 - (1.0 - 2e-5)**2) / 4
    assert thin["Ay"] / area == pytest.approx(2 * 1.3 / 4.9, rel=1e-4)


def test_i_section_strong_axis_matches_cowper_and_gsa():
    """工字钢强轴：与 Cowper 薄壁工字形公式（ν=0）、GSA 工字形公式逐位一致，

    约为腹板面积，只有 5A/6 的 0.46 倍——旧取法把剪切刚度高估约一倍。
    """
    h, b, tw, tf = I_DIMS.values()
    d = h - tf
    m, n = 2 * b * tf / (d * tw), b / d
    cowper = (10 * (1 + 3 * m)**2
              / (12 + 72 * m + 150 * m**2 + 90 * m**3 + 30 * n**2 * (m + m**2))
              * (2 * b * tf + d * tw))
    lam, beta = d * tw / (b * tf), b * tw / (d * tf)
    gsa = _gsa((6 + lam)**2 / (beta + 6 + 2 * lam + 0.2 * lam**2), d, tw)
    got = shear_areas({"type": "i", "dimensions": dict(I_DIMS)})
    assert got["Ay"] == pytest.approx(cowper, rel=1e-12)
    assert got["Ay"] == pytest.approx(gsa, rel=1e-12)
    area = 2 * b * tf + (h - 2 * tf) * tw
    assert 5 * area / 6 / got["Ay"] == pytest.approx(2.196, abs=1e-3)
    # 弱轴：两块翼缘各按实心矩形 5/6，腹板沿厚度受剪按薄壁理论不计
    assert got["Az"] == pytest.approx(5 / 6 * 2 * b * tf, rel=1e-12)


def test_cowper_poisson_correction_is_small_for_thin_walled_i():
    """薄壁截面按 ν=0 取剪切系数：对默认工字钢，Cowper 的 ν=0.3 修正只差 0.9%。"""
    h, b, tw, tf = I_DIMS.values()
    d = h - tf
    m, n = 2 * b * tf / (d * tw), b / d
    with_nu = (10 * 1.3 * (1 + 3 * m)**2
               / (12 + 72 * m + 150 * m**2 + 90 * m**3
                  + 0.3 * (11 + 66 * m + 135 * m**2 + 90 * m**3)
                  + 30 * n**2 * (m + m**2) + 1.5 * n**2 * (8 * m + 9 * m**2))
               * (2 * b * tf + d * tw))
    got = shear_areas({"type": "i", "dimensions": dict(I_DIMS)})["Ay"]
    assert abs(with_nu / got - 1) < 0.01


def test_channel_and_tee_match_gsa_thin_walled_formulas():
    """槽钢强轴对 GSA A1 式（翼缘自腹板中线伸出 b−tw/2），T 型钢对 B2 式。"""
    h, b, tw, tf = 0.25, 0.09, 0.008, 0.012
    d, flange = h - tf, b - tw / 2
    lam, beta = d * tw / (flange * tf), flange * tw / (d * tf)
    gsa = _gsa((6 + lam)**2 / (4 * beta + 6 + 2 * lam + 0.2 * lam**2), d, tw)
    got = shear_areas({"type": "channel", "dimensions": {"h": h, "b": b, "tw": tw, "tf": tf}})
    assert got["Ay"] == pytest.approx(gsa, rel=1e-12)

    h, b = 0.20, 0.15
    d = h - tf / 2
    lam, beta = d * tw / (b * tf), b * tw / (d * tf)
    gsa = _gsa((4 + lam)**2 / (0.5 * beta + 3.2 + 1.4 * lam + 0.2 * lam**2), d, tw)
    got = shear_areas({"type": "tee", "dimensions": {"h": h, "b": b, "tw": tw, "tf": tf}})
    assert got["Ay"] == pytest.approx(gsa, rel=1e-12)
    assert got["Az"] == pytest.approx(5 / 6 * b * tf, rel=1e-12)


def test_equal_angle_matches_hand_derived_closed_form():
    """薄壁等边角钢沿肢方向：计入惯性积的剪力流积分得 As = 5·a·t/6，a 为肢中线长。"""
    b, t = 0.075, 0.008
    got = shear_areas({"type": "angle", "dimensions": {"b": b, "t": t}})
    assert got["Ay"] == got["Az"] == pytest.approx(5 / 6 * (b - t / 2) * t, rel=1e-12)


def test_box_reduces_to_cowper_square_tube():
    """等厚方管：Cowper 薄壁箱形式 ν=0 时 k = 20/48，As = k·A。"""
    b, t = 0.2, 0.004
    got = shear_areas({"type": "box", "dimensions": {"b": b, "h": b, "t": t}})
    area = b * b - (b - 2 * t)**2
    assert got["Ay"] == got["Az"] == pytest.approx(20 / 48 * area, rel=1e-12)


def test_root_radius_and_polygon():
    """根部圆角不进剪力流（结果与直角相同）；多边形没有闭合解，明确拒绝并给出下一步。"""
    plain = shear_areas({"type": "i", "dimensions": dict(I_DIMS)})
    rounded = shear_areas({"type": "i", "dimensions": {**I_DIMS, "r": 0.013}})
    assert rounded["Ay"] == plain["Ay"]
    with pytest.raises(ValueError, match="Ay/Az"):
        shear_areas({"type": "polygon", "outer": [[0, 0], [1, 0], [0, 1]]})


@pytest.mark.parametrize("geometry", ALL_SHAPES, ids=lambda g: g["type"])
def test_shear_area_is_unit_invariant(geometry):
    """同一截面存成米与毫米：剪切面积正好差 10⁶ 倍，公式说明逐字相同。"""
    si = shear_areas(geometry)
    mm = shear_areas(scale_geometry(geometry, 1000.0))
    assert mm["Ay"] == pytest.approx(si["Ay"] * 1e6, rel=1e-12)
    assert mm["Az"] == pytest.approx(si["Az"] * 1e6, rel=1e-12)
    assert mm["formula"] == si["formula"]
    assert 0 < si["Ay"] and 0 < si["Az"]


def _cantilever(length, sec):
    f = Frame()
    f.nodes[1] = Node(1, 0, 0, 0)
    f.nodes[2] = Node(2, length, 0, 0)
    f.materials["M"] = Material("M", E, NU, 7850)
    f.sections["S"] = sec
    f.members[1] = Member(1, 1, 2, "S", "M")
    f.supports[1] = (1, 1, 1, 1, 1, 1)
    return f


def test_short_deep_i_cantilever_tip_deflection_includes_shear_term():
    """0.6 m 长的 300 高工字钢悬臂：端部挠度 = PL³/(3EI) + PL/(G·As)。

    防 Ay/Az 与内核局部轴配反：竖向荷载（局部 y）必须配强轴 Iz 与腹板剪切面积 Ay，
    水平荷载（局部 z）配 Iy 与翼缘剪切面积 Az。剪切项占竖向挠度的三成以上，
    5A/6 会把它低估一半多。
    """
    from sections import i_section

    length, force = 0.6, -50e3
    props = i_section("I", I_DIMS["h"], I_DIMS["b"], I_DIMS["tw"], I_DIMS["tf"])
    areas = shear_areas(props["geometry"], nu=STEEL_POISSON)
    sec = Section("S", props["A"], props["Iy"], props["Iz"], props["J"],
                  Ay=areas["Ay"], Az=areas["Az"])
    f = _cantilever(length, sec)
    f.nodal_loads[2] = (0, force, force, 0, 0, 0)
    u = solve(f).U[f.node_dofs(2)]

    shear_y = force * length / (G * areas["Ay"])
    vertical = force * length**3 / (3 * E * props["Iz"]) + shear_y
    horizontal = force * length**3 / (3 * E * props["Iy"]) + force * length / (G * areas["Az"])
    assert u[2] == pytest.approx(vertical, rel=1e-11)
    assert u[1] == pytest.approx(horizontal, rel=1e-11)
    assert shear_y / vertical > 0.3
