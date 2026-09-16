"""对称性检测（讲义 §3-9 四）。

讲义讲的是**利用**对称性：取半结构、半杆件、半节点荷载算，工作量减半。
本项目**只检测、不利用**，这是一个有意的取舍，理由写在下面。

检测什么
--------
逐个坐标平面（x=c、y=c、z=c）判断：

* **结构对称**：节点集镜像后与自身重合；每根杆件的镜像仍是一根杆件，
  且截面、材料、端部释放一致；支座的镜像仍是同样的支座。
* **荷载对称/反对称**：逐工况判断。镜像后荷载与原荷载相同即**对称**，
  差一个负号即**反对称**，都不是就是**不对称**。

力和力矩的镜像规则不一样，这是最容易写错的地方：

    力（极矢量）  ：垂直于镜面的分量**变号**，平行于镜面的分量不变
    力矩（轴矢量）：垂直于镜面的分量**不变**，平行于镜面的分量**变号**

力矩是赝矢量，镜像下的行为与力正好相反。把力矩当力一样镜像，
对称的结构会被判成不对称——而且不报错，只是"检测不出来"。

为什么只检测不利用
------------------
取半结构要在对称面上补一套边界条件（对称：面外平动与绕面内轴的转动为零；
反对称：反过来），再把跨越对称面的杆件按"半杆件"处理、把落在对称面上的
节点荷载减半。这套边界条件写错**不报错**，只会给出一个看着合理的错答案，
而且错在哪一根杆上很难看出来。

现代做法是直接算全模型：本项目 600 自由度的框架整解只要 0.1 秒，
省下的那一半算力没有意义，而引入的错误风险是实打实的。

所以对称性在这里的价值不是省算力，而是**当校核手段**：结构和荷载都对称时，
对称位置的位移和内力必须成镜像。这一条能查出装配、坐标转换、荷载等效里
一大类错误，而且不需要任何外部参照。:func:`check_symmetric_response`
就是干这个的。

限制（写出来，不假装没有）
--------------------------
* 只查三个坐标平面。斜的对称面不查——刚架里极少见，且候选面无从枚举。
* 端部释放按"端对端"比较，不考虑镜像后局部 y/z 轴对调导致的 ry/rz 互换。
  这只会让个别斜杆被判成不对称，**偏保守**，不会把不对称判成对称。
* 初应变、支座沉降参与判断；荷载组合不单独判（它是各工况的线性叠加）。
* **查物理模型，不查分析模型。** 跨间集中力会在荷载位置生成内节点，剖分点
  只要不落在对称面上，分析模型的几何就不再对称了——物理上明明对称的结构会
  查不出对称面。检测请传 ``model_io.from_dict`` 出来的物理 Frame；
  :func:`check_symmetric_response` 是例外，它要对着解，只能用分析模型
  （所幸能解的工况通常剖分点也对称）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from frame3d import Frame, LoadCase, span_loads_of
from span_loads import POINT, TRAPEZOID

AXES = ("x", "y", "z")
SYMMETRIC = "对称"
ANTISYMMETRIC = "反对称"
NEITHER = "不对称"

_TOL = 1e-9


@dataclass(frozen=True)
class Plane:
    """一个与坐标轴垂直的镜面。``axis`` 取 0/1/2，``offset`` 是该轴的坐标。"""

    axis: int
    offset: float

    @property
    def name(self) -> str:
        return f"{AXES[self.axis]} = {self.offset:g}"

    def mirror_point(self, p: np.ndarray) -> np.ndarray:
        q = np.array(p, dtype=float)
        q[self.axis] = 2.0 * self.offset - q[self.axis]
        return q

    def mirror_force(self, v: Iterable[float]) -> np.ndarray:
        """力：垂直于镜面的分量变号。"""
        w = np.array(list(v), dtype=float)
        w[self.axis] *= -1.0
        return w

    def mirror_moment(self, v: Iterable[float]) -> np.ndarray:
        """力矩是轴矢量：垂直于镜面的分量**不**变号，其余两个变号。"""
        w = -np.array(list(v), dtype=float)
        w[self.axis] *= -1.0
        return w

    def mirror_wrench(self, six: Iterable[float]) -> np.ndarray:
        """节点荷载 / 沉降的六分量：前三个按力、后三个按力矩。"""
        s = np.array(list(six), dtype=float)
        return np.concatenate([self.mirror_force(s[:3]),
                               self.mirror_moment(s[3:])])


# ------------------------------------------------------------------ 结构

def candidate_planes(frame: Frame, tol: float = _TOL) -> list[Plane]:
    """三个候选镜面，每轴一个。

    节点集若关于 ``axis = c`` 对称，则 c 必然是该轴坐标的中点——所以候选面
    不需要搜索，一个轴只有一个可能。
    """
    if not frame.nodes:
        return []
    pts = np.array([[n.x, n.y, n.z] for n in frame.nodes.values()], dtype=float)
    out = []
    for axis in range(3):
        lo, hi = float(pts[:, axis].min()), float(pts[:, axis].max())
        if hi - lo <= tol:
            continue                    # 整个结构压在一个平面上，谈不上镜像
        out.append(Plane(axis, 0.5 * (lo + hi)))
    return out


def _node_mirror_map(frame: Frame, plane: Plane, tol: float) -> dict[int, int] | None:
    """节点 → 其镜像节点。任何一个节点找不到镜像就返回 None。"""
    ids = list(frame.nodes)
    pts = np.array([[frame.nodes[i].x, frame.nodes[i].y, frame.nodes[i].z]
                    for i in ids], dtype=float)
    scale = max(float(np.abs(pts).max()), 1.0)
    eps = tol * scale
    mapping: dict[int, int] = {}
    for k, nid in enumerate(ids):
        target = plane.mirror_point(pts[k])
        d = np.linalg.norm(pts - target, axis=1)
        j = int(np.argmin(d))
        if d[j] > eps:
            return None
        mapping[nid] = ids[j]
    return mapping


def _member_mirror_map(frame: Frame, nodes: dict[int, int]
                       ) -> tuple[dict[int, int], dict[int, bool]] | None:
    """杆件 → 其镜像杆件，以及该镜像是否把两端调了个个儿。"""
    by_ends: dict[tuple[int, int], int] = {}
    for m in frame.members.values():
        by_ends[(m.i, m.j)] = m.id
    members: dict[int, int] = {}
    flipped: dict[int, bool] = {}
    for m in frame.members.values():
        a, b = nodes[m.i], nodes[m.j]
        if (a, b) in by_ends:
            other, flip = by_ends[(a, b)], False
        elif (b, a) in by_ends:
            other, flip = by_ends[(b, a)], True
        else:
            return None
        peer = frame.members[other]
        if (peer.section, peer.material) != (m.section, m.material):
            return None
        # 端释放：镜像若调了端，就拿对面的另一端比
        mine = (set(m.releases_i), set(m.releases_j))
        theirs = ((set(peer.releases_j), set(peer.releases_i)) if flip
                  else (set(peer.releases_i), set(peer.releases_j)))
        if mine != theirs:
            return None
        members[m.id] = other
        flipped[m.id] = flip
    return members, flipped


def _supports_match(frame: Frame, nodes: dict[int, int]) -> bool:
    """支座镜像。约束是"某方向不许动"，镜像只换方向不换有无，故按位比较。"""
    for nid, mask in frame.supports.items():
        peer = frame.supports.get(nodes[nid])
        if peer is None or tuple(peer) != tuple(mask):
            return False
    return len(frame.supports) == len({nodes[n] for n in frame.supports})


# ------------------------------------------------------------------ 荷载

def _case_symmetry(frame: Frame, case: LoadCase, plane: Plane,
                   nodes: dict[int, int], members: dict[int, int],
                   flipped: dict[int, bool], tol: float) -> str:
    """一个工况相对该镜面是对称、反对称，还是都不是。"""
    for sign, verdict in ((1.0, SYMMETRIC), (-1.0, ANTISYMMETRIC)):
        if _matches(frame, case, plane, nodes, members, flipped, sign, tol):
            return verdict
    return NEITHER


def _close(a, b, tol: float) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(float(np.abs(a).max(initial=0.0)),
                float(np.abs(b).max(initial=0.0)), 1.0)
    return bool(np.all(np.abs(a - b) <= tol * scale))


def _matches(frame: Frame, case: LoadCase, plane: Plane, nodes, members,
             flipped, sign: float, tol: float) -> bool:
    for nid, load in case.nodal_loads.items():
        peer = case.nodal_loads.get(nodes[nid], (0.0,) * 6)
        if not _close(sign * plane.mirror_wrench(load), peer, tol):
            return False
    for nid, given in case.settlements.items():
        peer = case.settlements.get(nodes[nid], (0.0,) * 6)
        if not _close(sign * plane.mirror_wrench(given), peer, tol):
            return False
    for mid in frame.members:
        other = members[mid]
        mine = span_loads_of(case, mid)
        theirs = span_loads_of(case, other)
        if len(mine) != len(theirs):
            return False
        for item, peer in zip(mine, theirs, strict=True):
            if item.kind != peer.kind:
                return False
            w1 = sign * plane.mirror_force(item.w1)
            w2 = sign * plane.mirror_force(item.w2)
            a = item.a
            if flipped[mid]:
                # 镜像把 i、j 调了个个儿：**梯形**两端的强度要跟着对调，
                # **集中力**的位置要从另一端量。漏掉这一步，对称的梯形荷载
                # 会被判成不对称——不报错，只是"检测不出来"。
                #
                # 均布不能跟着调：它的 w2 是个没用到的占位零，
                # 一调就把强度换成了零，本来对称的重力荷载会被判成不对称。
                if item.kind == TRAPEZOID:
                    w1, w2 = w2, w1
                elif item.kind == POINT:
                    a = _member_length(frame, mid) - item.a
            if not _close(w1, peer.w1, tol) or not _close(w2, peer.w2, tol):
                return False
            if abs(a - peer.a) > tol * max(abs(a), abs(peer.a), 1.0):
                return False
        mine_strain = case.member_strains.get(mid, 0.0)
        peer_strain = case.member_strains.get(other, 0.0)
        if abs(sign * mine_strain - peer_strain) > tol * max(
                abs(mine_strain), abs(peer_strain), 1.0):
            return False
    return True


def _member_length(frame: Frame, mid: int) -> float:
    m = frame.members[mid]
    return float(np.linalg.norm(frame.nodes[m.j].xyz - frame.nodes[m.i].xyz))


# ------------------------------------------------------------------ 对外

def detect_symmetry(frame: Frame, tol: float = _TOL) -> dict[str, Any]:
    """检测结构与各工况的对称性。**只报告，不改模型、不解半结构。**"""
    planes: list[dict[str, Any]] = []
    for plane in candidate_planes(frame, tol):
        nodes = _node_mirror_map(frame, plane, tol)
        if nodes is None:
            continue
        pair = _member_mirror_map(frame, nodes)
        if pair is None or not _supports_match(frame, nodes):
            continue
        members, flipped = pair
        cases = {name: _case_symmetry(frame, case, plane, nodes, members,
                                      flipped, tol)
                 for name, case in frame.load_cases.items()}
        planes.append({
            "plane": plane.name, "axis": AXES[plane.axis],
            "offset": plane.offset, "structure": True, "cases": cases,
            "node_pairs": nodes, "member_pairs": members,
            "usable": [n for n, v in cases.items() if v != NEITHER],
        })
    return {
        "planes": planes,
        "symmetric": bool(planes),
        "advice": _advice(planes),
    }


def _advice(planes: list[dict[str, Any]]) -> str:
    if not planes:
        return "未检测到关于坐标平面的对称性。"
    parts = []
    for p in planes:
        usable = p["usable"]
        if usable:
            kinds = "、".join(f"{n}（{p['cases'][n]}）" for n in usable)
            parts.append(f"结构关于 {p['plane']} 对称，工况 {kinds} 也对称，"
                         "可用对称位置的位移与内力互为镜像来校核结果")
        else:
            parts.append(f"结构关于 {p['plane']} 对称，但没有工况对称或反对称，"
                         "不能用来校核")
    return "；".join(parts) + "。本程序直接求解全结构，不做半结构简化。"


def check_symmetric_response(frame: Frame, solution, case: str | None = None,
                             tol: float = 1e-6) -> dict[str, Any]:
    """把对称性当校核手段用：对称结构 + 对称荷载 ⇒ 位移必须互为镜像。

    这一条查得出装配、坐标转换、等效节点荷载里一大类错误，**而且不需要任何
    外部参照**——不用商软、不用手算、不用金标准，结构自己就是自己的对照组。
    """
    name = case or solution.primary
    found = detect_symmetry(frame)
    out: list[dict[str, Any]] = []
    for p in found["planes"]:
        verdict = p["cases"].get(name)
        if verdict not in (SYMMETRIC, ANTISYMMETRIC):
            continue
        sign = 1.0 if verdict == SYMMETRIC else -1.0
        plane = Plane(AXES.index(p["axis"]), p["offset"])
        U = solution[name].U
        worst, where = 0.0, None
        scale = max(float(np.abs(U).max(initial=0.0)), 1e-30)
        for nid, peer in p["node_pairs"].items():
            a = U[frame.node_dofs(nid)]
            b = U[frame.node_dofs(peer)]
            diff = float(np.max(np.abs(sign * plane.mirror_wrench(a) - b)))
            if diff > worst:
                worst, where = diff, (nid, peer)
        out.append({"plane": p["plane"], "case": name, "kind": verdict,
                    "max_relative_difference": worst / scale,
                    "worst_pair": where, "ok": worst / scale <= tol})
    return {"case": name, "checks": out,
            "ok": all(c["ok"] for c in out) if out else True,
            "note": ("没有可用于校核的对称面" if not out else
                     "对称位置的位移互为镜像即通过")}
