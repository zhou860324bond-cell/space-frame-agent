"""路径曲线（Abaqus 的 XY Data from path）：一串首尾相连的杆件上，内力沿路径的分布。

单杆内力图一次只看一根杆；看一整跨连续梁、一根贯通多层的柱，要在几根杆之间来回
切换，峰值在哪一根、跨过节点时怎么变都拼不起来。这里把选中的杆件按连接关系排成
一条路径，横轴是沿路径的累计长度。

符号约定照各杆件自身的局部轴（i→j）：路径方向与杆件方向相反时，横坐标倒着排，
数值不做换号——轴力受拉为正与方向无关；剪力、扭矩、弯矩的正负随局部轴而定，
图注写明这一点，不替用户做可能出错的换算。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QVBoxLayout

from .dialog_styles import DialogHeader, HintLabel, button_box, style_dialog
from .diagram_panel import COMPONENTS, unit_system
from . import theme


def order_path(members: dict[int, tuple[int, int]], ids: list[int]
               ) -> list[tuple[int, bool]]:
    """把杆件排成一条首尾相连的路径：[(杆件号, 是否反向), …]。

    ``members`` 是 杆件号 → (i 节点, j 节点)。连不成一条不分叉的链时抛 ValueError，
    并说清是断开还是分叉——路径曲线的横轴是累计长度，分叉没有意义。
    """
    ids = list(dict.fromkeys(int(k) for k in ids))
    if not ids:
        raise ValueError("没有选中杆件")
    degree: dict[int, int] = {}
    for mid in ids:
        for node in members[mid]:
            degree[node] = degree.get(node, 0) + 1
    branching = sorted(n for n, d in degree.items() if d > 2)
    if branching:
        raise ValueError(f"所选杆件在节点 {branching} 处分叉，路径必须是一条不分叉的链")
    ends = sorted(n for n, d in degree.items() if d == 1)
    start = ends[0] if ends else members[ids[0]][0]
    remaining = set(ids)
    out: list[tuple[int, bool]] = []
    node = start
    while remaining:
        nxt = next((m for m in sorted(remaining) if node in members[m]), None)
        if nxt is None:
            raise ValueError(f"所选杆件不连续：从节点 {start} 出发走到节点 {node} 就断了，"
                             f"杆件 {sorted(remaining)} 接不上")
        i, j = members[nxt]
        reversed_ = node == j
        out.append((nxt, reversed_))
        node = i if reversed_ else j
        remaining.discard(nxt)
    return out


def path_series(frame, solution, mapping, ordered: list[tuple[int, bool]], case: str,
                component: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """沿路径的 (累计长度, 数值, 各杆分界位置)。数值单位与内核相同（N、N·长度）。"""
    from internal_forces import physical_member_diagram
    xs, ys, joints = [], [], []
    offset = 0.0
    for mid, reversed_ in ordered:
        d = physical_member_diagram(frame, solution, mapping, mid, case, stations=41)
        values = d.component(component)
        x = d.x[::-1] if reversed_ else d.x
        v = values[::-1] if reversed_ else values
        x = (d.length - x) if reversed_ else x
        xs.append(offset + x)
        ys.append(v)
        offset += d.length
        joints.append(offset)
    return np.concatenate(xs), np.concatenate(ys), joints[:-1]


class PathPlotDialog(QDialog):
    """路径曲线窗口：选分量与工况，画沿路径的分布并标出峰值。"""

    def __init__(self, window, ordered: list[tuple[int, bool]]):
        super().__init__(window)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.window, self.ordered = window, ordered
        self.setWindowTitle("路径曲线")
        layout = QVBoxLayout(self)
        chain = " → ".join(str(m) for m, _ in ordered)
        layout.addWidget(DialogHeader("路径曲线", f"杆件 {chain}；横轴为沿路径的累计长度"))
        form = QFormLayout()
        self.component = QComboBox(self)
        for label, key in COMPONENTS.items():
            self.component.addItem(label, key)
        self.case = QComboBox(self)
        self.case.addItems(list(window.session.solution.all_results()))
        if window.case and self.case.findText(window.case) >= 0:
            self.case.setCurrentText(window.case)
        form.addRow("分量", self.component)
        form.addRow("工况", self.case)
        layout.addLayout(form)
        self.figure = Figure(figsize=(7, 3.2), facecolor=theme.PANEL)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(HintLabel("正负号按各杆件自身局部轴（i→j）；路径与杆件反向时横坐标倒排、"
                                   "数值不换号。轴力受拉为正与方向无关。", self))
        box = button_box(self, ok="关闭")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self.component.currentIndexChanged.connect(self.draw)
        self.case.currentIndexChanged.connect(self.draw)
        style_dialog(self, 760, 460)
        self.draw()

    def draw(self) -> dict[str, Any]:
        session = self.window.session
        units = unit_system(session.model)
        key = self.component.currentData()
        moment = key in {"T", "My", "Mz", "M"}
        scale = units.moment_scale if moment else units.force_scale
        unit = units.moment_unit if moment else units.force_unit
        x, y, joints = path_series(session.frame, session.solution,
                                   session.compilation.mapping, self.ordered,
                                   self.case.currentText(), key)
        x = x * units.length_to_m
        y = y * scale
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, color=theme.ACCENT, linewidth=1.6)
        ax.fill_between(x, y, 0, color=theme.ACCENT, alpha=0.12)
        for j in joints:
            ax.axvline(j * units.length_to_m, color=theme.BORDER, linewidth=0.8)
        k = int(np.argmax(np.abs(y))) if len(y) else 0
        if len(y):
            ax.annotate(f"{y[k]:.3g} {unit}", (x[k], y[k]), textcoords="offset points",
                        xytext=(6, 6), fontsize=9)
        ax.set_xlabel("沿路径距离 (m)")
        ax.set_ylabel(f"{self.component.currentText()} ({unit})")
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        return {"peak": float(y[k]) if len(y) else 0.0, "points": len(x)}
