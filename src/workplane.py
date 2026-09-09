"""工作平面：把视口里的一次点击，变成一个确定的三维坐标。

**这是三维里画图和二维里画图的全部差别。**

二维界面上点一下就是一个点。三维视口里点一下给到的是一条射线——
从相机出发穿过屏幕上那个像素，射线上每一点在屏幕上都是同一个位置。
「用户想要哪一个」这件事，屏幕上的信息根本不足以回答。

CAD 的通行答案是工作平面：先说清楚「我在哪个平面上画」，
射线与那个平面的交点就是唯一答案。Abaqus 的 sketch plane、
FEM-Python 的 XY/XZ/YZ 选择器，都是这一件事。

平面选好之后还有两级吸附，顺序不能反：

1. **先吸已有节点。** 画梁时点柱顶，要的是「连到那个节点」，
   不是「在柱顶附近新建一个点」。差一点点就是两个不同节点，
   结构图上连着、矩阵里散着。
2. **再吸网格。** 没有节点可吸时落到网格交点上，
   免得得到 3.0000000001 这种坐标。

顺序反过来的话，柱顶如果不在网格点上（比如层高 3.6 而网格 1.0），
点柱顶会被拉到最近的网格点上，永远连不上那根柱。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 三个基本工作平面。**只给这三个**——任意斜平面在界面上说不清楚，
# 而空间刚架的建模面基本都是正交的
WORK_PLANES = ("XY", "XZ", "YZ")

# 每个平面的法向，也就是「被平面固定住的那个坐标」
NORMAL_AXIS = {"XY": 2, "XZ": 1, "YZ": 0}

# 平面上那两个坐标轴的名字，界面上显示用
PLANE_AXES = {"XY": ("X", "Y"), "XZ": ("X", "Z"), "YZ": ("Y", "Z")}

# 点到已有节点多近就算「点的是它」。**按屏幕上的观感定，不按模型尺度定**——
# 这个值太小，用户觉得吸不住；太大，想在柱子旁边新建点却总被拉过去
NODE_SNAP = 0.25


class OffPlane(ValueError):
    """射线与工作平面平行，交不出点来。"""


@dataclass(frozen=True)
class WorkPlane:
    """一个工作平面：平面的朝向，加上它沿法向偏移多少。

    `offset` 就是被固定住的那个坐标的值。在 XZ 平面上画一榀刚架，
    `offset` 是这一榀的 y 坐标——拉伸成多榀时，每一榀就是一个 offset。
    """

    kind: str = "XZ"
    offset: float = 0.0
    grid: float = 0.5              # 网格间距（米），0 表示不吸网格

    def __post_init__(self) -> None:
        if self.kind not in WORK_PLANES:
            raise ValueError(f"工作平面只能是 {WORK_PLANES}，收到 {self.kind!r}")

    @property
    def axis(self) -> int:
        return NORMAL_AXIS[self.kind]

    @property
    def normal(self) -> tuple[float, float, float]:
        n = [0.0, 0.0, 0.0]
        n[self.axis] = 1.0
        return tuple(n)

    @property
    def free_axes(self) -> tuple[int, int]:
        """平面内的两个坐标轴下标。"""
        return tuple(k for k in (0, 1, 2) if k != self.axis)

    def label(self) -> str:
        a, b = PLANE_AXES[self.kind]
        fixed = "XYZ"[self.axis]
        return f"{a}{b} 平面（{fixed} = {self.offset:g} m）"

    # --- 射线 → 平面上的点 ---

    def intersect(self, origin, direction) -> tuple[float, float, float]:
        """射线与平面的交点。

        `direction` 不必是单位向量。射线与平面平行时抛 `OffPlane`——
        **这时候不能猜**：随便返回一个点，用户会得到一个莫名其妙的
        远处节点，而且完全不知道为什么。
        """
        o = [float(v) for v in origin]
        d = [float(v) for v in direction]
        k = self.axis
        if abs(d[k]) < 1e-12:
            raise OffPlane(
                f"视线与{self.kind}平面几乎平行，交不出点。把视角转开一些再画")
        t = (self.offset - o[k]) / d[k]
        hit = [o[i] + t * d[i] for i in range(3)]
        hit[k] = self.offset            # 消掉浮点误差，保证严格落在平面上
        return tuple(hit)

    def project(self, point) -> tuple[float, float, float]:
        """把任意点压到平面上。拾取到的是模型表面上的点时用这个。"""
        p = [float(v) for v in point]
        p[self.axis] = self.offset
        return tuple(p)

    # --- 吸附 ---

    def snap_to_grid(self, point) -> tuple[float, float, float]:
        if not self.grid or self.grid <= 0:
            return tuple(float(v) for v in point)
        p = list(float(v) for v in point)
        for k in self.free_axes:
            p[k] = round(p[k] / self.grid) * self.grid
        p[self.axis] = self.offset
        return tuple(p)

    def snap(self, point, nodes=None,
             node_tol: float = NODE_SNAP) -> tuple[tuple[float, float, float],
                                                   int | None, str]:
        """定位一次点击。返回 (坐标, 吸到的节点编号或 None, 说明)。

        **先节点后网格**，理由见模块开头。返回的说明直接显示在状态栏，
        用户得知道自己这一下到底吸到了什么——吸附是隐形的，
        不说的话，画错了也看不出错在哪。
        """
        p = self.project(point)
        best_id, best_d = None, node_tol
        for n in nodes or []:
            d = math.dist(p, (float(n["x"]), float(n["y"]), float(n["z"])))
            if d <= best_d:
                best_id, best_d = int(n["id"]), d
        if best_id is not None:
            n = next(x for x in nodes if int(x["id"]) == best_id)
            return ((float(n["x"]), float(n["y"]), float(n["z"])), best_id,
                    f"吸附到节点 {best_id}")
        snapped = self.snap_to_grid(p)
        if self.grid and self.grid > 0:
            return snapped, None, f"吸附到 {self.grid:g} m 网格"
        return snapped, None, "自由坐标"

    # --- 网格线，画给用户看 ---

    def grid_lines(self, half_size: float = 12.0
                   ) -> list[tuple[tuple[float, float, float],
                                   tuple[float, float, float]]]:
        """平面上的网格线段。**画出来的网格才是能瞄准的网格**——
        看不见网格却在吸附，用户只会觉得坐标莫名其妙地变了。"""
        if not self.grid or self.grid <= 0:
            return []
        u, v = self.free_axes
        steps = int(half_size / self.grid)
        lines = []
        for i in range(-steps, steps + 1):
            t = i * self.grid
            for fixed, moving in ((u, v), (v, u)):
                a = [0.0, 0.0, 0.0]
                b = [0.0, 0.0, 0.0]
                a[self.axis] = b[self.axis] = self.offset
                a[fixed] = b[fixed] = t
                a[moving], b[moving] = -half_size, half_size
                lines.append((tuple(a), tuple(b)))
        return lines


def plane_through(nodes, kind: str = "XZ") -> float:
    """一组节点在某个平面法向上的公共坐标；不一致时取最常见的那个。

    用来在选中已有构件后**把工作平面对到它所在的那一榀上**，
    省掉用户手填 offset——填错一个数，画出来的东西在另一榀上，
    而三维视图里很难一眼看出来。
    """
    k = NORMAL_AXIS[kind]
    key = "xyz"[k]
    counts: dict[float, int] = {}
    for n in nodes or []:
        v = round(float(n[key]), 9)
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return 0.0
    return max(counts.items(), key=lambda kv: (kv[1], -abs(kv[0])))[0]
