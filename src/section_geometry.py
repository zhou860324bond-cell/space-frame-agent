"""截面真实轮廓：尺寸、孔洞、形心与实体生成共用，长度沿用模型单位。"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


TYPES = ("rectangle", "box", "circle", "tube", "i", "channel", "angle", "tee", "polygon")


def descriptor(kind, **dimensions):
    return {"type": kind, "dimensions": dimensions}


def scale_geometry(geometry, factor):
    out = deepcopy(geometry)
    if "dimensions" in out:
        out["dimensions"] = {k: float(v) * factor for k, v in out["dimensions"].items()}
    if "outer" in out:
        out["outer"] = (np.asarray(out["outer"], dtype=float) * factor).tolist()
    if "holes" in out:
        out["holes"] = [(np.asarray(p, dtype=float) * factor).tolist() for p in out["holes"]]
    return out


def _area_centroid(points):
    p = np.asarray(points, dtype=float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = 0.5 * cross.sum()
    if abs(area) < 1e-14 * max(float(np.ptp(p, axis=0).max())**2, 1e-30):
        raise ValueError("截面轮廓面积为零，请检查顶点顺序。")
    return abs(float(area)), ((p + q) * cross[:, None]).sum(axis=0) / (6 * area)


def contains(loop, point):
    p = np.asarray(loop)
    x, y = point
    a, b = p, np.roll(p, -1, axis=0)
    crossings = (a[:, 1] > y) != (b[:, 1] > y)
    ids = np.flatnonzero(crossings)
    return bool(np.count_nonzero(x < a[ids, 0] + (y-a[ids, 1]) *
                                (b[ids, 0]-a[ids, 0]) / (b[ids, 1]-a[ids, 1])) % 2)


def _intersects(a, b, c, d):
    def cross(p, q, r):
        u, v = q-p, r-p
        return u[0]*v[1]-u[1]*v[0]
    tol = 1e-12*max(np.linalg.norm(b-a), np.linalg.norm(d-c), 1e-15)**2
    ab = (cross(a, b, c), cross(a, b, d))
    cd = (cross(c, d, a), cross(c, d, b))
    if ab[0]*ab[1] < 0 and cd[0]*cd[1] < 0:
        return True
    for value, point, start, end in ((ab[0], c, a, b), (ab[1], d, a, b),
                                      (cd[0], a, c, d), (cd[1], b, c, d)):
        if abs(value) <= tol and np.dot(point-start, point-end) <= tol:
            return True
    return False


def _loops_touch(first, second):
    return any(_intersects(a, b, c, d)
               for a, b in zip(first, np.roll(first, -1, axis=0), strict=True)
               for c, d in zip(second, np.roll(second, -1, axis=0), strict=True))


def _validate_loop(loop):
    p = np.asarray(loop, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("截面轮廓需要至少三个有限的[y,z]顶点。")
    if np.any(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1) == 0):
        raise ValueError("截面轮廓存在重复相邻顶点，请删除重复点。")
    # 非相邻边相交或接触都无法定义简单截面。
    for i in range(len(p)):
        for j in range(i+2, len(p)):
            if i == 0 and j == len(p)-1:
                continue
            a, b, c, d = p[i], p[(i+1) % len(p)], p[j], p[(j+1) % len(p)]
            if _intersects(a, b, c, d):
                raise ValueError("截面轮廓自交，请按边界顺序重新排列顶点。")
    _area_centroid(p)
    return p


def profile(geometry, circle_points=48):
    """返回以材料形心为原点的外轮廓、孔洞及最小几何尺度。"""
    kind = geometry["type"]
    if kind not in TYPES:
        raise ValueError(f"不支持的截面轮廓类型 {kind!r}。")
    d = {k: float(v) for k, v in geometry.get("dimensions", {}).items()}
    if any(not math.isfinite(v) or v <= 0 for v in d.values()):
        raise ValueError("截面几何尺寸必须为正的有限值。")
    holes = []
    if kind in ("circle", "tube"):
        radius = d["d"] / 2
        phi = np.linspace(0, 2*np.pi, circle_points, endpoint=False)
        outer = radius * np.column_stack((np.cos(phi), np.sin(phi)))
        feature = radius / 5
        if kind == "tube":
            feature = d["t"]
            if feature >= radius:
                raise ValueError("圆管壁厚必须小于半径。")
            holes = [outer * (radius-feature)/radius]
    elif kind == "polygon":
        outer = geometry["outer"]
        holes = geometry.get("holes", [])
        feature = d.get("feature_size", 0.1 * float(np.ptp(np.asarray(outer), axis=0).min()))
    else:
        b, h = d["b"], d.get("h", d["b"])
        if kind in ("rectangle", "box"):
            outer = [[0, 0], [h, 0], [h, b], [0, b]]
            feature = min(b, h)/10
            if kind == "box":
                t = d["t"]
                if 2*t >= min(b, h):
                    raise ValueError("矩形管壁厚必须小于短边的一半。")
                holes = [[[t,t], [h-t,t], [h-t,b-t], [t,b-t]]]
                feature = t
        elif kind == "angle":
            t = d["t"]
            if t >= b:
                raise ValueError("角钢厚度必须小于边宽。")
            outer = [[0,0], [b,0], [b,t], [t,t], [t,b], [0,b]]
            feature = t
        else:
            tw, tf = d["tw"], d["tf"]
            if tw >= b or (tf >= h if kind == "tee" else 2*tf >= h):
                raise ValueError("翼缘或腹板厚度超出截面尺寸。")
            left, right = (b-tw)/2, (b+tw)/2
            if kind == "i":
                outer = [[0,0],[tf,0],[tf,left],[h-tf,left],[h-tf,0],[h,0],
                         [h,b],[h-tf,b],[h-tf,right],[tf,right],[tf,b],[0,b]]
            elif kind == "channel":
                outer = [[0,0],[h,0],[h,b],[h-tf,b],[h-tf,tw],[tf,tw],[tf,b],[0,b]]
            else:
                outer = [[0,left],[h-tf,left],[h-tf,0],[h,0],[h,b],[h-tf,b],
                         [h-tf,right],[0,right]]
            feature = min(tw, tf)
    outer = _validate_loop(outer)
    holes = [_validate_loop(p) for p in holes]
    for hole in holes:
        if _loops_touch(outer, hole) or not all(contains(outer, v) for v in hole):
            raise ValueError("截面孔洞超出外轮廓，请检查孔洞坐标。")
    for i, hole in enumerate(holes):
        for other in holes[i+1:]:
            if _loops_touch(hole, other) or contains(hole, other[0]) or contains(other, hole[0]):
                raise ValueError("截面孔洞相交、接触或嵌套，请分别定义互不重叠的孔洞。")
    area, center = _area_centroid(outer)
    first_moment = area * center
    for hole in holes:
        a, c = _area_centroid(hole)
        area -= a
        first_moment -= a*c
    if area <= 0:
        raise ValueError("截面孔洞面积不小于外轮廓面积。")
    center = first_moment/area
    if kind in ("circle", "tube"):
        area = math.pi * ((d["d"]/2)**2 - (d["d"]/2-d["t"])**2) if kind == "tube" else math.pi*(d["d"]/2)**2
    return {"type": kind, "outer": (outer-center).tolist(),
            "holes": [(p-center).tolist() for p in holes],
            "feature_size": float(feature), "area": float(area),
            "centroid": center.tolist(), "dimensions": d}


def material_probe(p):
    """选外边中点向材料内侧移动的点，用于识别CAD切面。"""
    if p["type"] in ("circle", "tube"):
        r = p["dimensions"]["d"]/2
        return [r-p["dimensions"]["t"]/2 if p["type"] == "tube" else r/2, 0.0]
    loop = np.asarray(p["outer"])
    for a, b in zip(loop, np.roll(loop, -1, axis=0), strict=True):
        tangent = b-a
        n = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        for sign in (1, -1):
            point = (a+b)/2 + sign*n*p["feature_size"]*0.1
            if contains(loop, point) and not any(contains(h, point) for h in p["holes"]):
                return point.tolist()
    raise ValueError("无法找到截面材料内部取样点，请检查轮廓与孔洞。")
