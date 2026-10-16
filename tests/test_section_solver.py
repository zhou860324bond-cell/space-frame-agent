"""截面特性有限元求解（J、Iw、剪心）的回归。

真值来自经典弹性理论：矩形与等边三角形的 Saint-Venant 扭转精确解、开口薄壁截面的
剪心与翘曲常数公式（Timoshenko & Gere《弹性稳定理论》第 5 章；Pilkey 2002 第 6 章）。
薄壁公式本身忽略了交接处与壁厚效应，对照时按其适用精度给容差，并写明偏差方向。
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("gmsh", reason="截面特性求解需要 Gmsh 划分截面网格")

import section_geometry as sg  # noqa: E402
import section_solver as ss  # noqa: E402


def _rectangle_J(a: float, b: float) -> float:
    """矩形 Saint-Venant 扭转常数的精确级数解（a ≥ b）。"""
    a, b = max(a, b), min(a, b)
    series = sum(math.tanh(n*math.pi*a/(2*b)) / n**5 for n in range(1, 400, 2))
    return a*b**3/3 * (1 - 192/math.pi**5 * b/a * series)


def test_rectangle_torsion_constant_matches_the_exact_series_from_above():
    """防止 J 只靠近似系数表：实心矩形与 Saint-Venant 级数解一致，且位移元从上方单调逼近。"""
    geometry = {"type": "rectangle", "dimensions": {"b": 0.1, "h": 0.3}}
    exact = _rectangle_J(0.3, 0.1)
    values = [ss.solve_section(geometry, mesh_size=s)["J"] for s in (0.02, 0.01, 0.005)]
    assert values[-1] == pytest.approx(exact, rel=2e-6)
    assert values[0] > values[1] > values[2] > exact


def test_equilateral_triangle_torsion_constant_matches_the_closed_form():
    """等边三角形 J = √3·a⁴/80（Saint-Venant 精确解），多边形轮廓无近似。"""
    a = 0.2
    geometry = {"type": "polygon", "outer": [[0, 0], [a, 0], [a/2, a*math.sqrt(3)/2]]}
    assert ss.solve_section(geometry)["J"] == pytest.approx(math.sqrt(3)*a**4/80, rel=2e-6)


def test_mesh_integrals_reproduce_the_exact_profile_properties():
    """网格积分出的 A、Iy、Iz 必须与轮廓的精确值一致，否则 J、Iw 的口径就不可信。"""
    geometry = {"type": "channel", "dimensions": {"h": .25, "b": .09, "tw": .008, "tf": .012}}
    got = ss.solve_section(geometry)
    exact = sg.section_properties(geometry)
    for key in ("A", "Iy", "Iz"):
        assert got[key] == pytest.approx(exact[key], rel=1e-10), key


def test_doubly_symmetric_i_section_shear_centre_and_warping_constant():
    """工字钢剪心在形心；Iw 与薄壁公式 If·h0²/2 = tf·b³·h0²/24 相差不到 0.5%。

    J 比薄壁公式 Σbt³/3 略小：薄壁公式把翼缘全长按窄条计，忽略了自由端与交接处的影响。
    """
    b, h, tw, tf = 0.2, 0.4, 0.008, 0.013
    got = ss.solve_section({"type": "i", "dimensions": {"b": b, "h": h, "tw": tw, "tf": tf}})
    assert got["shear_center"] == pytest.approx([0.0, 0.0], abs=1e-6*h)
    h0 = h - tf
    assert got["Iw"] == pytest.approx(tf*b**3*h0**2/24, rel=5e-3)
    thin = (2*b*tf**3 + (h - 2*tf)*tw**3) / 3
    assert 0.95*thin < got["J"] < thin


def test_channel_shear_centre_lies_behind_the_web():
    """防止剪心与形心混为一谈：槽钢剪心在腹板背后，距腹板中线 e = 3b'²tf/(6b'tf + h'tw)。"""
    h, b, tw, tf = 0.25, 0.09, 0.008, 0.012
    geometry = {"type": "channel", "dimensions": {"h": h, "b": b, "tw": tw, "tf": tf}}
    got = ss.solve_section(geometry)
    centroid_z = sg.profile(geometry)["centroid"][1]
    bp, hp = b - tw/2, h - tf
    e = 3*bp**2*tf / (6*bp*tf + hp*tw)
    y_sc, z_sc = got["shear_center"]
    assert y_sc == pytest.approx(0.0, abs=1e-6*h)            # 关于 y 轴（腹板平面）对称
    assert z_sc + centroid_z < 0.0                           # 绝对坐标在腹板外侧（z<0）
    assert z_sc + centroid_z == pytest.approx(tw/2 - e, rel=2e-2)
    thin_iw = tf*bp**3*hp**2/12 * (3*bp*tf + 2*hp*tw) / (6*bp*tf + hp*tw)
    assert got["Iw"] == pytest.approx(thin_iw, rel=3e-2)


@pytest.mark.parametrize("kind, dims, corner, secondary", [
    # 次翘曲：板条以端点为极点 b³t³/36，以中点为极点 b³t³/144（Kollbrunner & Basler）
    ("angle", {"b": .1, "t": .008}, lambda d: (d["t"]/2, d["t"]/2),
     lambda d: 2 * (d["b"] - d["t"]/2)**3 * d["t"]**3 / 36),
    ("tee", {"h": .2, "b": .15, "tw": .008, "tf": .012},
     lambda d: (d["h"] - d["tf"]/2, d["b"]/2),
     lambda d: d["b"]**3*d["tf"]**3/144 + (d["h"] - d["tf"]/2)**3*d["tw"]**3/36),
])
def test_two_plate_sections_shear_at_the_plate_intersection(kind, dims, corner, secondary):
    """角钢、T 型钢由两块板组成：剪心在两板中线交点，主翘曲为零，只剩厚度方向的次翘曲。"""
    geometry = {"type": kind, "dimensions": dims}
    got = ss.solve_section(geometry)
    centroid = sg.profile(geometry)["centroid"]
    target = corner(dims)
    size = max(dims.values())
    assert got["shear_center"][0] + centroid[0] == pytest.approx(target[0], abs=0.01*size)
    assert got["shear_center"][1] + centroid[1] == pytest.approx(target[1], abs=0.01*size)
    assert got["Iw"] == pytest.approx(secondary(dims), rel=5e-2)


def test_section_solver_is_unit_invariant():
    """米制与毫米制得到同一截面：J、Iw、剪心只差长度比例的 4、6、1 次方。"""
    geometry = {"type": "i", "dimensions": {"b": .2, "h": .4, "tw": .008, "tf": .013, "r": .016}}
    in_m = ss.solve_section(geometry)
    in_mm = ss.solve_section(sg.scale_geometry(geometry, 1000.0))
    assert in_mm["J"] == pytest.approx(in_m["J"] * 1e12, rel=1e-9)
    assert in_mm["Iw"] == pytest.approx(in_m["Iw"] * 1e18, rel=1e-9)
    assert in_mm["shear_center"] == pytest.approx(
        [v * 1000 for v in in_m["shear_center"]], abs=1e-6)


def test_root_fillets_stiffen_torsion():
    """根部圆角增加交接处材料，J 必须比直角轮廓大；这是近似公式表达不了的。"""
    dims = {"b": .2, "h": .4, "tw": .008, "tf": .013}
    sharp = ss.solve_section({"type": "i", "dimensions": dims})["J"]
    rounded = ss.solve_section({"type": "i", "dimensions": {**dims, "r": .016}})["J"]
    assert rounded > 1.05 * sharp
