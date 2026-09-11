"""三维视图里的约束与荷载符号。

参照项目 FEM-Python 有一整个 `symbols.py` 干这件事，这是它的界面比我们
"像 CAE"的主要原因之一：不画符号的话，三维视图只是一堆线，看不出
哪里是支座、荷载往哪个方向加。加上符号之后，**建模阶段就能一眼看出
荷载方向加反了没有**——那是最常见也最难从数字上察觉的错误。

两个取舍写在前面：

* **不是每个受约束的节点都画符号。** 平面刚架的面外自由度是代码自动约束的，
  每个节点都有；全画出来满屏都是符号，真正的柱脚反而看不见了。
  所以只画**约束住两个及以上平动方向**的节点，其余在图注里说明有多少个。
* **箭头长度按相对大小归一**，不按绝对值。荷载量级跨好几个数量级时
  （节点 kN 与线荷载 kN/m 混在一起），按绝对值画会让小的那类彻底消失。
"""

from __future__ import annotations

import numpy as np

import viz_theme as T
from frame3d import span_loads_of
from span_loads import POINT

# 只有约束住这么多个平动方向的节点才画支座符号
_MIN_FIXED_TRANSLATIONS = 2


def _go():
    """延迟导入 plotly。

    这个模块里真正被两端共用的是 `model_size` 和 `classify_support`——
    纯几何，只用 numpy。plotly 只有网页端那三个 `*_traces` 函数要，
    可桌面端 `desktop/scene.py` 也 import 本模块，于是**桌面端被迫依赖
    一个它一行都不用的绘图后端**。

    平时看不出来，打包成 exe 时就暴露了：排掉 plotly（网页端那条路径
    桌面版用不到，它还会拖进 pyarrow、tornado 一大串）之后，程序一启动就
    `ModuleNotFoundError: No module named 'plotly'`，栈顶指向 viz_symbols，
    而那一行 import 看上去理所当然。

    改成延迟导入，两端各取所需：桌面端不再碰 plotly，网页端行为不变。
    """
    import plotly.graph_objects as go
    return go


def model_size(frame) -> float:
    """模型的特征尺寸，符号大小按它取，跟结构一起缩放。"""
    coords = np.array([n.xyz for n in frame.nodes.values()])
    if len(coords) < 2:
        return 1.0
    return float(np.max(coords.max(axis=0) - coords.min(axis=0))) or 1.0


def classify_support(mask) -> str:
    """按约束模式分类：固接 / 铰接 / 部分约束。"""
    trans = sum(mask[:3])
    rot = sum(mask[3:])
    if trans == 3 and rot == 3:
        return "固接"
    if trans == 3:
        return "铰接"
    return "部分约束"


def _box(p: np.ndarray, r: float):
    """节点下方的一个小方框，代表固接。"""
    x, y, z = p
    lo, hi = z - 2 * r, z
    corners = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    def seg(a, b):
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])

    for k in range(4):
        a = (*corners[k], lo)
        b = (*corners[(k + 1) % 4], lo)
        seg(a, b)
        seg((*corners[k], hi), a)
    # 底部斜纹，工程图里固接的惯用画法
    for t in np.linspace(-r, r, 4):
        seg((x + t, y - r, lo), (x + t - r * 0.6, y - r, lo - r * 0.6))
    return xs, ys, zs


def _cone(p: np.ndarray, r: float):
    """节点下方的一个三角锥，代表铰接。"""
    x, y, z = p
    tip = (x, y, z)
    base = [(x + r * np.cos(a), y + r * np.sin(a), z - 2 * r)
            for a in np.linspace(0, 2 * np.pi, 5)[:-1]]
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    def seg(a, b):
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])

    for k, b in enumerate(base):
        seg(tip, b)
        seg(b, base[(k + 1) % len(base)])
    return xs, ys, zs


def support_traces(frame, size: float | None = None) -> tuple[list, dict]:
    """支座符号。返回 (traces, 说明)。"""
    r = 0.018 * (size or model_size(frame))
    groups: dict[str, list] = {"固接": [[], [], []], "铰接": [[], [], []],
                               "部分约束": [[], [], []]}
    drawn = 0
    skipped = 0
    for nid, mask in frame.supports.items():
        if sum(mask[:3]) < _MIN_FIXED_TRANSLATIONS:
            skipped += 1
            continue
        kind = classify_support(mask)
        p = frame.nodes[nid].xyz
        xs, ys, zs = _box(p, r) if kind == "固接" else _cone(p, r)
        for bucket, part in zip(groups[kind], (xs, ys, zs)):
            bucket.extend(part)
        drawn += 1

    go = _go()
    # 深色视口上要用亮灰，原来那套浅底色号在这里几乎看不见
    colors = {"固接": T.VIEWPORT_INK, "铰接": T.VIEWPORT_INK,
              "部分约束": T.VIEWPORT_INK_MUTED}
    traces = []
    for kind, (xs, ys, zs) in groups.items():
        if not xs:
            continue
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines", name=f"支座·{kind}",
            line=dict(color=colors[kind], width=3), hoverinfo="skip"))
    return traces, {"drawn": drawn, "skipped": skipped}


def _cone_trace(points, vectors, name, color, ref):
    if not points:
        return None
    go = _go()
    pts = np.array(points)
    vec = np.array(vectors)
    return go.Cone(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        u=vec[:, 0], v=vec[:, 1], w=vec[:, 2],
        anchor="tip", sizemode="absolute", sizeref=ref,
        colorscale=[[0, color], [1, color]], showscale=False,
        name=name, showlegend=True, hoverinfo="skip")


def load_traces(frame, case: str, size: float | None = None,
                arrows_per_member: int = 3) -> tuple[list, dict]:
    """荷载符号：节点力用一支箭头，杆间均布用一排箭头，集中力单独一支。

    **箭头长度按相对大小归一**——荷载量级常跨好几个数量级，按绝对值画的话
    小的那类会彻底消失，看图的人会以为它们不存在。
    """
    load_case = frame.load_cases.get(case)
    if load_case is None:
        return [], {"nodal": 0, "member": 0}

    span = size or model_size(frame)
    nodal_pts, nodal_vec, spans_pts, spans_vec = [], [], [], []

    for nid, load in load_case.nodal_loads.items():
        f = np.asarray(load[:3], dtype=float)
        if np.linalg.norm(f) > 0 and nid in frame.nodes:
            nodal_pts.append(frame.nodes[nid].xyz)
            nodal_vec.append(f)

    for mid, member in frame.members.items():
        pi, pj = frame.nodes[member.i].xyz, frame.nodes[member.j].xyz
        for item in span_loads_of(load_case, mid):
            if item.kind == POINT:
                L = float(np.linalg.norm(pj - pi))
                t = (item.a / L) if L else 0.0
                spans_pts.append(pi + t * (pj - pi))
                spans_vec.append(np.asarray(item.w1, dtype=float))
                continue
            v1, v2 = item.ends()
            for t in np.linspace(0.15, 0.85, arrows_per_member):
                spans_pts.append(pi + t * (pj - pi))
                spans_vec.append((1 - t) * v1 + t * v2)

    traces = []
    for pts, vec, name, color in (
            (nodal_pts, nodal_vec, "节点荷载", T.VIEWPORT_HIGHLIGHT),
            (spans_pts, spans_vec, "杆间荷载", T.VIEWPORT_HIGH)):
        if not pts:
            continue
        mag = np.linalg.norm(np.array(vec), axis=1)
        peak = float(mag.max()) or 1.0
        # 归一到 [0.35, 1]：太小的箭头看不见，但也不能让它们和最大的一样长
        scaled = [v / peak * (0.35 + 0.65 * m / peak) * span * 0.10
                  for v, m in zip(vec, mag)]
        t = _cone_trace(pts, scaled, name, color, ref=1.0)
        if t is not None:
            traces.append(t)
    return traces, {"nodal": len(nodal_pts), "member": len(spans_pts)}
