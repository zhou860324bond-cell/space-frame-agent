"""变截面杆件：两端截面之间按几何尺寸线性插值。

门式刚架的楔形梁柱（GB 51022）腹板高度沿杆线性变化。杆件写 ``section``（i 端）
和 ``section_j``（j 端），两端截面须有同一类型的真实轮廓；编译时把杆件剖成若干
段，每段取段中点处线性插值的轮廓，按轮廓精确计算 A、Iy、Iz。于是静力、模态、
屈曲、P-Δ、弯扭屈曲都在同一套等截面单元上进行，不必为变截面另写一套单元。

近似与适用范围（写进结果说明，不静默）：

* 阶梯近似：段内截面取中点值，柔度按中点法积分，误差随段数平方下降。默认段数
  见 :data:`TAPER_MIN_DIVISIONS`，可用布种加密。
* J 与可选的剪切面积、极端纤维距离、剪应力静矩在两端值之间线性插值（腹板变高时
  这些量本就近似线性；J 由翼缘控制、几乎不变）。圆截面的 J 按 2I 精确计算。
"""

from __future__ import annotations

import math
from typing import Any

from section_geometry import profile, section_properties

#: 变截面杆件的最少分段数。阶梯近似的柔度误差 ∝ 1/n²。标定：腹板高 600→300 的
#: 楔形工字钢悬臂梁（Iz 比 4.8，门式刚架常见上限附近），8 段端部挠度误差 0.49%，
#: 16 段 0.12%（tests/test_taper.py 对照精确积分）。线性分析多几段几乎不增加计算量，
#: 取 16；更陡的变截面会超出这个精度，届时用布种加密，结果说明里写明段数。
TAPER_MIN_DIVISIONS = 16

# 两端线性插值的可选数值字段
_LINEAR_FIELDS = ("J", "Ay", "Az", "cy", "cz", "Sz", "bz", "Sy", "by", "S_flange", "c_web")


def tapered_members(data: dict[str, Any]) -> dict[int, tuple[str, str]]:
    """杆件编号 → (i 端截面名, j 端截面名)，只含真正变截面的杆件。"""
    return {int(m["id"]): (m["section"], m["section_j"])
            for m in data.get("members") or []
            if m.get("section_j") and m.get("section_j") != m.get("section")}


def check(data: dict[str, Any]) -> list[str]:
    """变截面杆件的语义检查：两端截面存在、都有同类型真实轮廓、尺寸项相同。"""
    sections = {s["name"]: s for s in data.get("sections") or []}
    errors = []
    for member_id, (name_i, name_j) in sorted(tapered_members(data).items()):
        missing = [n for n in (name_i, name_j) if n not in sections]
        if missing:
            errors.append(f"变截面杆件 {member_id} 引用了不存在的截面 {missing}")
            continue
        gi, gj = sections[name_i].get("geometry"), sections[name_j].get("geometry")
        if gi is None or gj is None:
            errors.append(f"变截面杆件 {member_id} 的两端截面都必须有真实轮廓尺寸"
                          f"（{name_i!r} 与 {name_j!r}），才能沿杆插值；请在「创建截面」中按尺寸定义。")
            continue
        if gi["type"] != gj["type"] or gi["type"] == "polygon":
            errors.append(f"变截面杆件 {member_id} 两端截面类型须相同且为参数截面"
                          f"（{gi['type']} 与 {gj['type']}）；多边形轮廓的顶点无法一一对应插值。")
            continue
        if set(gi.get("dimensions", {})) != set(gj.get("dimensions", {})):
            errors.append(f"变截面杆件 {member_id} 两端截面的尺寸项不一致"
                          f"（{sorted(gi.get('dimensions', {}))} 与 {sorted(gj.get('dimensions', {}))}），"
                          "例如一端有根部圆角 r 另一端没有。")
    return errors


def interpolated_geometry(gi: dict[str, Any], gj: dict[str, Any], ratio: float) -> dict[str, Any]:
    dims = {k: (1 - ratio) * float(gi["dimensions"][k]) + ratio * float(gj["dimensions"][k])
            for k in gi["dimensions"]}
    return {"type": gi["type"], "dimensions": dims}


def section_at(sec_i: dict[str, Any], sec_j: dict[str, Any], ratio: float,
               name: str) -> dict[str, Any]:
    """沿杆相对位置 ``ratio``（0 为 i 端）处的截面记录，字段与模型截面相同。"""
    geometry = interpolated_geometry(sec_i["geometry"], sec_j["geometry"], ratio)
    out: dict[str, Any] = {"name": name, "geometry": geometry}
    for key in _LINEAR_FIELDS:
        if sec_i.get(key) is not None and sec_j.get(key) is not None:
            out[key] = (1 - ratio) * float(sec_i[key]) + ratio * float(sec_j[key])
    if geometry["type"] in ("circle", "tube"):
        d = geometry["dimensions"]["d"]
        inner = d - 2 * geometry["dimensions"]["t"] if geometry["type"] == "tube" else 0.0
        inertia = math.pi * (d**4 - inner**4) / 64
        out.update(A=profile(geometry)["area"], Iy=inertia, Iz=inertia, J=2 * inertia,
                   circular=True)
    else:
        exact = section_properties(geometry)
        out.update(A=exact["A"], Iy=exact["Iy"], Iz=exact["Iz"])
        if sec_i.get("circular") and sec_j.get("circular"):
            out["circular"] = True
    return out
