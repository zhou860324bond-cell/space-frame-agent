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


#: 可带根部圆角 r 的轮廓：热轧型钢在腹板与翼缘（或两肢）交接的内凹角处有圆角。
ROOT_FILLET_TYPES = ("i", "channel", "angle", "tee")

#: 每 90° 圆弧的折线段数。轮廓面积按圆弧解析值给出，折线只影响实体网格表面：
#: 4 段弦高约 0.02r。实测 HN400×200×8×13（r=16）三杆节点默认网格：直角轮廓
#: 15.5 万单元，4 段多 3%，8 段多 10%（短折线边在整条短臂表面强制加密）。
FILLET_SEGMENTS_PER_QUARTER = 4


def _concave(loop):
    """外轮廓各顶点是否为内凹角（材料内角大于 180°）。"""
    p = np.asarray(loop, dtype=float)
    prev, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    a, b = p - prev, nxt - p
    turn = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
    q = np.roll(p, -1, axis=0)
    orientation = np.sign((p[:, 0]*q[:, 1] - q[:, 0]*p[:, 1]).sum())
    return turn * orientation < 0


def _round_concave_corners(loop, radius):
    """把外轮廓的每个内凹角倒成半径 radius 的圆弧，返回新轮廓与补入的精确面积。

    热轧型钢表的截面面积含这些根部圆角，直角轮廓会比表值小 2%~4%。圆弧与两条
    边相切；夹角 α 处补入面积 r²/tan(α/2) − r²(π−α)/2，90° 时为 (1−π/4)r²。
    """
    p = np.asarray(loop, dtype=float)
    concave = _concave(p)
    n = len(p)
    tangent = np.zeros(n)
    units = []
    for i in range(n):
        u1, u2 = p[i-1] - p[i], p[(i+1) % n] - p[i]
        u1, u2 = u1/np.linalg.norm(u1), u2/np.linalg.norm(u2)
        units.append((u1, u2))
        if concave[i]:
            alpha = math.acos(max(-1.0, min(1.0, float(u1 @ u2))))
            tangent[i] = radius / math.tan(alpha/2)
    for i in range(n):
        j = (i+1) % n
        if tangent[i] + tangent[j] >= np.linalg.norm(p[j]-p[i]) * (1 - 1e-9):
            raise ValueError(f"根部圆角半径 r={radius:g} 过大，圆弧超出相邻边长；"
                             "请核对 r 与腹板、翼缘尺寸是否用同一长度单位。")
    out, filled = [], 0.0
    for i in range(n):
        if not concave[i]:
            out.append(p[i])
            continue
        u1, u2 = units[i]
        alpha = math.acos(max(-1.0, min(1.0, float(u1 @ u2))))
        bisector = (u1 + u2) / np.linalg.norm(u1 + u2)
        center = p[i] + bisector * radius / math.sin(alpha/2)
        t1, t2 = p[i] + tangent[i]*u1, p[i] + tangent[i]*u2
        a1 = math.atan2(*(t1 - center)[::-1])
        a2 = math.atan2(*(t2 - center)[::-1])
        sweep = (a2 - a1 + math.pi) % (2*math.pi) - math.pi
        # 减去容差再取整：90° 圆弧算出 8.0000001 段时不能多分一段，否则对称截面的
        # 几个圆弧折线不同，形心偏离对称轴。
        count = max(2, math.ceil(FILLET_SEGMENTS_PER_QUARTER * abs(sweep) / (math.pi/2) - 1e-6))
        for k in range(count + 1):
            angle = a1 + sweep * k / count
            out.append(center + radius*np.array([math.cos(angle), math.sin(angle)]))
        filled += radius**2 / math.tan(alpha/2) - radius**2 * (math.pi - alpha) / 2
    return np.asarray(out), filled


def profile(geometry, circle_points=48):
    """返回以材料形心为原点的外轮廓、孔洞及最小几何尺度。

    工字形、槽钢、角钢、T 型钢可在 dimensions 里给根部圆角半径 ``r``，所有内凹角
    按该半径倒圆；不给 r 即直角轮廓。``concave_corners`` 为内凹角个数，供面积
    不符时提示补 r。
    """
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
    if "r" in d and kind not in ROOT_FILLET_TYPES:
        raise ValueError(f"根部圆角 r 只适用于工字形、槽钢、角钢和 T 型钢轮廓，{kind} 轮廓不接受 r。")
    outer = _validate_loop(outer)
    concave_corners = int(_concave(outer).sum())
    exact_area = None
    if "r" in d:
        sharp_area, _ = _area_centroid(outer)
        outer, filled = _round_concave_corners(outer, d["r"])
        outer = _validate_loop(outer)
        exact_area = sharp_area + filled
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
    if exact_area is not None:
        area = exact_area
    return {"type": kind, "outer": (outer-center).tolist(),
            "holes": [(p-center).tolist() for p in holes],
            "concave_corners": concave_corners,
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


def _polygon_moments(loop):
    """多边形对原点的 A、∫y dA、∫z dA、∫y² dA、∫z² dA（格林公式，逆时针为正）。"""
    p = np.asarray(loop, dtype=float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0]*q[:, 1] - q[:, 0]*p[:, 1]
    sign = 1.0 if cross.sum() > 0 else -1.0
    y0, z0, y1, z1 = p[:, 0], p[:, 1], q[:, 0], q[:, 1]
    return sign * np.array([cross.sum() / 2,
                            (cross * (y0 + y1)).sum() / 6,
                            (cross * (z0 + z1)).sum() / 6,
                            (cross * (y0**2 + y0*y1 + y1**2)).sum() / 12,
                            (cross * (z0**2 + z0*z1 + z1**2)).sum() / 12])


def section_properties(geometry):
    """按真实轮廓精确计算 A 与对形心的 ``Iy = ∫z²dA``、``Iz = ∫y²dA``。

    y 为轮廓第一坐标（沿截面高度 h），与内核强轴约定一致。直线边用格林公式；
    根部圆角按"正方形减四分之一圆"的解析式逐个加上：每个圆角面积 (1−π/4)r²，
    形心距角点沿两肢各 e = r(10−3π)/(12−3π)，对过角点的肢线惯性矩
    (1−5π/16)r⁴。圆角要求内凹角为 90° 且两肢平行于 y、z 轴，本模块的型钢轮廓都满足。
    圆形截面另有闭合公式，这里不处理。
    """
    if geometry["type"] in ("circle", "tube"):
        raise ValueError("圆形截面请用闭合公式计算截面特性。")
    dims = dict(geometry.get("dimensions", {}))
    radius = float(dims.pop("r", 0.0))
    sharp = profile({**geometry, "dimensions": dims})
    total = _polygon_moments(sharp["outer"])
    for hole in sharp["holes"]:
        total -= _polygon_moments(hole)
    if radius > 0.0:
        rounded = profile(geometry)          # 校验 r 能放得下
        assert rounded["concave_corners"] == sharp["concave_corners"]
        outer = np.asarray(sharp["outer"], dtype=float)
        a = (1 - math.pi/4) * radius**2
        e = radius * (10 - 3*math.pi) / (12 - 3*math.pi)
        own = (1 - 5*math.pi/16) * radius**4 - a * e**2
        for i in np.flatnonzero(_concave(outer)):
            corner = outer[i]
            u1 = outer[i-1] - corner
            u2 = outer[(i+1) % len(outer)] - corner
            u1, u2 = u1/np.linalg.norm(u1), u2/np.linalg.norm(u2)
            if abs(u1 @ u2) > 1e-9 or min(abs(u1).max(), abs(u2).max()) < 1 - 1e-9:
                raise ValueError("根部圆角的解析特性只适用于两肢平行于坐标轴的 90° 内凹角。")
            c = corner + e * (u1 + u2)
            total += np.array([a, a*c[0], a*c[1], own + a*c[0]**2, own + a*c[1]**2])
    area, sy, sz, iyy, izz = total
    cy, cz = sy/area, sz/area
    return {"A": float(area), "Iy": float(izz - area*cz**2), "Iz": float(iyy - area*cy**2),
            "centroid": [float(cy), float(cz)]}


#: 剪切面积默认泊松比（钢）。截面不携带材料，而 Cowper 剪切系数与 ν 有关；
#: ν 在 0.25~0.35 之间变化时实心矩形的系数只变 ±0.4%。
STEEL_POISSON = 0.3


def _cowper_rectangle(nu):
    """Cowper (1966) 实心矩形：k = 10(1+ν)/(12+11ν)，ν=0 时即 Timoshenko 的 5/6。"""
    return 10 * (1 + nu) / (12 + 11 * nu)


def _cowper_hollow_circle(nu, m):
    """Cowper (1966) 圆环，m = 内径/外径：
    k = 6(1+ν)(1+m²)² / [(7+6ν)(1+m²)² + (20+12ν)m²]。m=0 为实心圆 6(1+ν)/(7+6ν)，
    m→1 为薄壁圆管 2(1+ν)/(4+3ν)。"""
    s = (1 + m * m) ** 2
    return 6 * (1 + nu) * s / ((7 + 6 * nu) * s + (20 + 12 * nu) * m * m)


def _thin_box(h, b, t):
    """等厚薄壁矩形管沿 h 受剪的剪切面积，中线尺寸 h、b。

    Cowper (1966) 薄壁箱形公式取 ν=0（与开口薄壁的剪力流能量法同一假定）：
    m = b·t₁/(h·t)，n = b/h，k = 10(1+3m)² / [12+72m+150m²+90m³+30n²(m+m²)]，
    As = k·2(b+h)t。等厚时 m = n。
    """
    m = n = b / h
    k = 10 * (1 + 3 * m) ** 2 / (12 + 72 * m + 150 * m * m + 90 * m ** 3
                                  + 30 * n * n * (m + m * m))
    return k * 2 * (b + h) * t


def _midline_segments(kind, d):
    """开口薄壁截面的中线：[(起点[y,z], 终点[y,z], 壁厚)]，在交汇点断开成树。"""
    if kind == "angle":
        b, t = d["b"], d["t"]
        c = (t / 2, t / 2)
        return [(c, (b, t / 2), t), (c, (t / 2, b), t)]
    h, b, tw, tf = d["h"], d["b"], d["tw"], d["tf"]
    if kind == "i":
        y = (h - tf) / 2
        return [((-y, 0), (y, 0), tw),
                ((y, 0), (y, -b / 2), tf), ((y, 0), (y, b / 2), tf),
                ((-y, 0), (-y, -b / 2), tf), ((-y, 0), (-y, b / 2), tf)]
    if kind == "channel":
        y, z = (h - tf) / 2, tw / 2
        return [((-y, z), (y, z), tw), ((y, z), (y, b), tf), ((-y, z), (-y, b), tf)]
    y = h - tf / 2                                     # T 型钢：翼缘中线高度
    return [((0, 0), (y, 0), tw), ((y, 0), (y, -b / 2), tf), ((y, 0), (y, b / 2), tf)]


def _open_thin_walled(segments):
    """开口薄壁截面按 Jourawski 剪力流求剪切面积 (Ay, Az)。

    剪力 V 引起的剪力流 q(s) 由截面切开一侧的面积矩给出，剪切应变能
    ∫q²/(2Gt)ds 与 V²/(2G·As) 相等，于是 As = V² / ∫q²/t ds。中线、壁厚按
    薄壁理论计（与 Cowper 薄壁公式同一理想化），截面为树形时切开任一点都把截面
    分成两块、两块面积矩等值反号，所以不必选起点。主轴不与 y、z 重合时
    （角钢）计入惯性积。q 沿每段是二次式，3 点 Gauss 积分精确。
    """
    segs = [(np.asarray(a, float), np.asarray(b, float), float(t)) for a, b, t in segments]
    lengths = [float(np.linalg.norm(b - a)) for a, b, _ in segs]
    area = sum(t * L for (_, _, t), L in zip(segs, lengths, strict=True))
    centre = sum(t * L * (a + b) / 2 for (a, b, t), L in zip(segs, lengths, strict=True)) / area
    segs = [(a - centre, b - centre, t) for a, b, t in segs]
    # 线积分：∫y²、∫z²、∫yz（壁厚方向的 t³ 项按薄壁理论略去）
    iyy = izz = pyz = 0.0
    for (a, b, t), L in zip(segs, lengths, strict=True):
        w = t * L / 3
        iyy += w * (a[0]**2 + a[0]*b[0] + b[0]**2)
        izz += w * (a[1]**2 + a[1]*b[1] + b[1]**2)
        pyz += w * (a[0]*a[1] + (a[0]*b[1] + a[1]*b[0]) / 2 + b[0]*b[1])
    det = iyy * izz - pyz * pyz

    def key(p):
        return tuple(np.round(p / max(lengths), 9))
    ends = [(key(a), key(b)) for a, b, _ in segs]
    if len({p for e in ends for p in e}) != len(segs) + 1:
        raise ValueError("薄壁剪力流只适用于开口（树形）截面。")

    def side_moment(i, node):
        """从第 i 段的 node 端出发、不经过第 i 段能到达的各段面积矩 (∫y, ∫z)。"""
        seen, stack, total = {i}, [node], np.zeros(2)
        while stack:
            p = stack.pop()
            for j, (pa, pb) in enumerate(ends):
                if j not in seen and p in (pa, pb):
                    seen.add(j)
                    a, b, t = segs[j]
                    total += t * lengths[j] * (a + b) / 2
                    stack.append(pb if p == pa else pa)
        return total

    gauss = ((0.5 - math.sqrt(0.15), 5 / 18), (0.5, 8 / 18), (0.5 + math.sqrt(0.15), 5 / 18))
    energy = np.zeros(2)
    for i, ((a, b, t), L) in enumerate(zip(segs, lengths, strict=True)):
        base = side_moment(i, ends[i][0])
        for u, w in gauss:
            qy, qz = base + t * L * u * (a + (b - a) * u / 2)
            flow_y = (izz * qy - pyz * qz) / det       # 单位 Vy
            flow_z = (iyy * qz - pyz * qy) / det       # 单位 Vz
            energy += w * L * np.array([flow_y**2, flow_z**2]) / t
    return float(1 / energy[0]), float(1 / energy[1])


def shear_areas(geometry, nu=STEEL_POISSON):
    """Timoshenko 梁的有效剪切面积 ``Ay``（对应 Vy，沿截面高度 h）与 ``Az``（沿宽度 b）。

    只有实心矩形的 As = 5A/6；薄壁截面的剪力主要由平行于剪力的板件承担，工字钢
    强轴 As 接近腹板面积，按 5A/6 会把剪切刚度高估约一倍。取法：

    * 实心矩形、实心圆、圆管：Cowper (1966) 闭合式，与 ν 有关，默认 ν=0.3；
    * 矩形管：Cowper 薄壁箱形公式（ν=0）；
    * 工字钢、槽钢、T 型钢、角钢：薄壁剪力流能量法（``_open_thin_walled``），
      工字钢强轴与 Cowper 薄壁工字形公式 ν=0 时逐位一致。

    根部圆角 r 不计入（剪力流按中线计）。返回 ``{"Ay", "Az", "formula"}``，
    ``formula`` 是写给用户看的公式说明。多边形截面没有闭合解，明确拒绝。

    参考：G. R. Cowper, The shear coefficient in Timoshenko's beam theory,
    J. Appl. Mech. 33 (1966) 335–340。
    """
    kind = geometry["type"]
    if kind == "polygon":
        raise ValueError("多边形截面没有剪切面积闭合解：请直接填写 Ay/Az，或关闭剪切变形。")
    p = profile(geometry)
    d = p["dimensions"]
    if kind == "rectangle":
        k = _cowper_rectangle(nu)
        area = k * p["area"]
        return {"Ay": area, "Az": area,
                "formula": f"Cowper 实心矩形 k=10(1+ν)/(12+11ν)={k:.4f}（ν={nu:g}），As=k·A"}
    if kind in ("circle", "tube"):
        m = (d["d"] - 2 * d["t"]) / d["d"] if kind == "tube" else 0.0
        k = _cowper_hollow_circle(nu, m)
        area = k * p["area"]
        text = ("Cowper 实心圆 k=6(1+ν)/(7+6ν)" if kind == "circle" else
                f"Cowper 圆环 k=6(1+ν)(1+m²)²/[(7+6ν)(1+m²)²+(20+12ν)m²]，m=内径/外径={m:.4f}，")
        return {"Ay": area, "Az": area, "formula": f"{text}k={k:.4f}（ν={nu:g}），As=k·A"}
    if kind == "box":
        h, b, t = d["h"] - d["t"], d["b"] - d["t"], d["t"]
        return {"Ay": _thin_box(h, b, t), "Az": _thin_box(b, h, t),
                "formula": "Cowper 薄壁箱形 k=10(1+3m)²/[12+72m+150m²+90m³+30n²(m+m²)]"
                           "（ν=0，按中线尺寸），As=k·A"}
    ay, az = _open_thin_walled(_midline_segments(kind, d))
    return {"Ay": ay, "Az": az,
            "formula": "薄壁剪力流能量法 As=V²/∫q²/t ds，q=V·S/I（按中线计，未计根部圆角）"}
