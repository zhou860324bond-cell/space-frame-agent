"""按尺寸算截面特性。

让用户直接填 A、Iy、Iz、J 四个数既难填又容易填错——填 b、h 或型钢的四个尺寸，
特性由公式算，是更可靠的做法。

**轴的约定要和内核一致**：局部 y 朝上（水平杆件），平面内弯曲绕局部 z，
所以对高 h 宽 b 的矩形，`Iz = b·h³/12` 是强轴，`Iy = h·b³/12` 是弱轴。
搞反会让梁的挠度差出一个数量级。

单位一律用米，与 N-m-Pa 一致。
"""

from __future__ import annotations

import math
from typing import Any

# 矩形抗扭系数 β：J = β·b³·h，b 取短边。经典表值，两端分别趋近 0.141 与 1/3。
_BETA_TABLE = [(1.0, 0.141), (1.5, 0.196), (2.0, 0.229), (2.5, 0.249),
               (3.0, 0.263), (4.0, 0.281), (6.0, 0.299), (10.0, 0.312)]


def rectangle_torsion_factor(ratio: float) -> float:
    """按长短边比插值取 β。比值超出表格范围就取端点，趋近薄板的 1/3。"""
    if ratio <= _BETA_TABLE[0][0]:
        return _BETA_TABLE[0][1]
    if ratio >= _BETA_TABLE[-1][0]:
        return 1.0 / 3.0
    for (r0, b0), (r1, b1) in zip(_BETA_TABLE, _BETA_TABLE[1:]):
        if r0 <= ratio <= r1:
            t = (ratio - r0) / (r1 - r0)
            return b0 + t * (b1 - b0)
    return 1.0 / 3.0


def _check(name: str, value: float) -> float:
    v = float(value)
    if not (v > 0.0) or not math.isfinite(v):
        raise ValueError(f"{name} 必须是大于 0 的有限值，收到 {value!r}")
    return v


def rectangle(name: str, width: float, height: float) -> dict[str, Any]:
    """实心矩形。width 沿局部 z，height 沿局部 y（即弯曲平面内的高度）。"""
    b = _check("width", width)
    h = _check("height", height)
    short, long_ = min(b, h), max(b, h)
    return {
        "name": name,
        "A": b * h,
        "Iy": h * b ** 3 / 12.0,                 # 弱轴：绕局部 y
        "Iz": b * h ** 3 / 12.0,                 # 强轴：绕局部 z
        "J": rectangle_torsion_factor(long_ / short) * short ** 3 * long_,
    }


def i_section(name: str, height: float, flange_width: float,
              web_thickness: float, flange_thickness: float) -> dict[str, Any]:
    """双轴对称工字形 / H 型钢。腹板沿局部 y，即弯曲平面内。"""
    h = _check("height", height)
    b = _check("flange_width", flange_width)
    tw = _check("web_thickness", web_thickness)
    tf = _check("flange_thickness", flange_thickness)
    if 2.0 * tf >= h:
        raise ValueError(f"翼缘太厚：2×{tf} 已经不小于截面高 {h}")
    if tw >= b:
        raise ValueError(f"腹板厚 {tw} 不应不小于翼缘宽 {b}")

    hw = h - 2.0 * tf                              # 腹板净高
    return {
        "name": name,
        "A": 2.0 * b * tf + hw * tw,
        # 弱轴：两翼缘绕自身中轴 + 腹板
        "Iy": (2.0 * tf * b ** 3 + hw * tw ** 3) / 12.0,
        # 强轴：外框减去两侧挖空
        "Iz": (b * h ** 3 - (b - tw) * hw ** 3) / 12.0,
        # 开口薄壁自由扭转：J ≈ Σ(1/3)·长边·厚³
        "J": (2.0 * b * tf ** 3 + hw * tw ** 3) / 3.0,
    }


def circular_tube(name: str, outer_diameter: float, thickness: float) -> dict[str, Any]:
    """圆管。双轴对称，Iy = Iz；闭口截面 J = 2I（极惯性矩）。"""
    d = _check("outer_diameter", outer_diameter)
    t = _check("thickness", thickness)
    if 2.0 * t >= d:
        raise ValueError(f"壁厚太大：2×{t} 已经不小于外径 {d}")
    di = d - 2.0 * t
    inertia = math.pi * (d ** 4 - di ** 4) / 64.0
    return {"name": name, "A": math.pi * (d ** 2 - di ** 2) / 4.0,
            "Iy": inertia, "Iz": inertia, "J": 2.0 * inertia}


def solid_circle(name: str, diameter: float) -> dict[str, Any]:
    d = _check("diameter", diameter)
    inertia = math.pi * d ** 4 / 64.0
    return {"name": name, "A": math.pi * d ** 2 / 4.0,
            "Iy": inertia, "Iz": inertia, "J": 2.0 * inertia}


BUILDERS = {
    "矩形": (rectangle, ("width", "height"), ("宽 b (m)", "高 h (m)")),
    "工字形 / H 型钢": (i_section,
                        ("height", "flange_width", "web_thickness", "flange_thickness"),
                        ("截面高 h (m)", "翼缘宽 b (m)", "腹板厚 tw (m)", "翼缘厚 tf (m)")),
    "圆管": (circular_tube, ("outer_diameter", "thickness"), ("外径 d (m)", "壁厚 t (m)")),
    "实心圆": (solid_circle, ("diameter",), ("直径 d (m)",)),
}

DEFAULT_DIMENSIONS = {
    "矩形": (0.2, 0.4),
    "工字形 / H 型钢": (0.4, 0.2, 0.010, 0.016),
    "圆管": (0.3, 0.010),
    "实心圆": (0.25,),
}
