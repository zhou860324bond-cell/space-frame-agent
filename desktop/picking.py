"""屏幕空间拾取与多选：Abaqus 式「点选、Shift 加选、Ctrl 减选、拖框选」。

原来的拾取先要求点中 VTK 表面，再在世界坐标里按「模型尺寸的 10%」找最近的
对象：缩放之后容差跟着模型走而不是跟着手走，大模型里容易选到邻居，细杆又
常常点不中；一次也只能选一个。这里改为在**屏幕像素**里判定，和用户眼睛看到的
距离一致，并维护一个可以增减的选择集。

纯 numpy，不依赖显卡：投影由视口在有渲染器时提供，测试直接喂二维坐标。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 点选容差（逻辑像素）。和鼠标指针的视觉误差同一量级，与模型尺寸、缩放无关。
CLICK_TOLERANCE_PX = 10.0


def project(matrix: np.ndarray, size: tuple[float, float], xyz: np.ndarray) -> np.ndarray:
    """世界坐标 → 屏幕坐标（左上角为原点，单位与 ``size`` 相同）。

    ``matrix`` 是相机的组合投影矩阵（VTK ``GetCompositeProjectionTransformMatrix``），
    ``size`` 是视口宽高。相机背后的点给 NaN，不参与拾取。
    """
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    homo = np.hstack([xyz, np.ones((len(xyz), 1))]) @ np.asarray(matrix, float).T
    w = homo[:, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = homo[:, :3] / w
    width, height = size
    out = np.column_stack([(ndc[:, 0] + 1.0) * 0.5 * width,
                           (1.0 - ndc[:, 1]) * 0.5 * height])
    out[(w[:, 0] <= 0)] = np.nan
    return out


def nearest_point(points: np.ndarray, click: tuple[float, float],
                  tolerance: float = CLICK_TOLERANCE_PX) -> int | None:
    """离点击处最近、且在容差内的点的下标；都太远返回 None。"""
    if len(points) == 0:
        return None
    d = np.hypot(points[:, 0] - click[0], points[:, 1] - click[1])
    d = np.where(np.isnan(d), np.inf, d)
    k = int(np.argmin(d))
    return k if d[k] <= tolerance else None


def _segment_distance(segments: np.ndarray, click: tuple[float, float]) -> np.ndarray:
    a, b = segments[:, 0], segments[:, 1]
    ab = b - a
    ap = np.asarray(click, float) - a
    length2 = np.einsum("ij,ij->i", ab, ab)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(np.einsum("ij,ij->i", ap, ab) / length2, 0.0, 1.0)
    t = np.where(length2 > 0, t, 0.0)
    nearest = a + t[:, None] * ab
    d = np.hypot(*(np.asarray(click, float) - nearest).T)
    return np.where(np.isnan(d), np.inf, d)


def nearest_segment(segments: np.ndarray, click: tuple[float, float],
                    tolerance: float = CLICK_TOLERANCE_PX) -> int | None:
    """离点击处最近的线段（杆件在屏幕上的投影）；都太远返回 None。"""
    if len(segments) == 0:
        return None
    d = _segment_distance(np.asarray(segments, float), click)
    k = int(np.argmin(d))
    return k if d[k] <= tolerance else None


def _normal_rect(rect) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def points_in_rect(points: np.ndarray, rect) -> np.ndarray:
    """完全落在框里的点（布尔掩码）。"""
    x0, y0, x1, y1 = _normal_rect(rect)
    p = np.asarray(points, float)
    with np.errstate(invalid="ignore"):
        return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)


def segments_in_rect(segments: np.ndarray, rect) -> np.ndarray:
    """两个端点都在框里的线段——Abaqus 拖框默认只选完全框住的对象。"""
    s = np.asarray(segments, float)
    return points_in_rect(s[:, 0], rect) & points_in_rect(s[:, 1], rect)


@dataclass
class Selection:
    """一类对象（节点或杆件）的有序选择集。"""

    kind: str | None = None
    ids: list[int] = field(default_factory=list)

    def apply(self, kind: str, picked: list[int], mode: str = "replace") -> None:
        """按修饰键合并一次拾取：replace 替换、add 加选（Shift）、remove 减选（Ctrl）。

        换了对象类型时一律重新开始：节点和杆件不能混在一个区域里。
        """
        picked = [int(k) for k in picked]
        if kind != self.kind or mode == "replace":
            self.kind, self.ids = kind, list(dict.fromkeys(picked))
        elif mode == "add":
            self.ids = list(dict.fromkeys([*self.ids, *picked]))
        elif mode == "remove":
            gone = set(picked)
            self.ids = [k for k in self.ids if k not in gone]
        else:
            raise ValueError(f"未知的选择方式 {mode!r}")

    def clear(self) -> None:
        self.kind, self.ids = None, []

    def __bool__(self) -> bool:
        return bool(self.ids)
