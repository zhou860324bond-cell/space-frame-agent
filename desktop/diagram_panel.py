"""单杆内力图面板。

三维云图看的是**全局分布**——哪根杆内力大、正负怎么分。
但要判断一根梁"跨中够不够、支座够不够"，需要的是**沿杆长的那条曲线**：
它在哪儿过零、峰值在哪个位置、不同组合的包络有多宽。这两件事互相替代不了，
所以云图之外还得有这块。

配色和 `plot3d` 那套静态图一致——桌面端、网页端、报告插图共用一套约定，
不让三个地方各画各的。

**包络只对单根杆件有意义。** 逐点取所有组合的上下界，
再标出每一点是谁在控制；放在全结构上画就是一团糊。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from . import theme

_UNITS_READY = False


def unit_system(model: dict):
    """取模型的单位制，顺便把 matplotlib 的中文字体配好。

    默认字体里没有汉字，图例上的"各组合包络"会变成一排方框——
    **一个中文界面里出现方框，比没有图例更难看**。
    """
    global _UNITS_READY
    if not _UNITS_READY:
        import matplotlib
        for family in ("Noto Sans CJK SC", "Noto Sans CJK JP",
                       "Microsoft YaHei", "SimHei", "PingFang SC",
                       "WenQuanYi Zen Hei"):
            try:
                from matplotlib import font_manager
                font_manager.findfont(family, fallback_to_default=False)
            except Exception:                   # noqa: BLE001
                continue
            matplotlib.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False   # 负号别变方框
            break
        _UNITS_READY = True

    import units
    return units.of(model)          # units.of 就是为这个准备的，别自己解析名字

# 分量：显示名 → 内部名。**要写清楚是绕哪个轴**，
# 光看 My / Mz 两个字母分不出哪个是平面内弯矩，选错了看半天图都是白看
COMPONENTS = {
    "合剪力 |V|（空间杆件）": "V",
    "合弯矩 |M|（空间杆件）": "M",
    "轴力 N（受拉为正）": "N",
    "剪力 Vy": "Vy",
    "剪力 Vz": "Vz",
    "扭矩 T": "T",
    "弯矩 My（绕局部 y）": "My",
    "弯矩 Mz（平面内，最常用）": "Mz",
}

MOMENTS = {"T", "My", "Mz", "M"}

SIGN_CONVENTIONS = {
    "V": "|V| = sqrt(Vy^2 + Vz^2)，无正负方向",
    "M": "|M| = sqrt(My^2 + Mz^2)，无正负方向",
    "N": "+N：沿局部 x 轴受拉",
    "Vy": "+Vy：沿局部 y 轴",
    "Vz": "+Vz：沿局部 z 轴",
    "T": "+T：绕局部 x 轴，按右手定则",
    "My": "+My：绕局部 y 轴，按右手定则",
    "Mz": "+Mz：绕局部 z 轴，按右手定则",
}


def diagram_features(x, y) -> dict:
    """提取专业读图需要的极值、过零点和集中荷载跳跃位置。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not len(x):
        return {"max": None, "min": None, "zeros": [], "jumps": []}
    peak = float(np.abs(y).max(initial=0.0))
    tolerance = max(peak * 1e-10, 1e-12)
    length = float(np.ptp(x))
    close = max(length * 1e-7, 1e-12)
    zeros: list[float] = []
    if peak > 0.0:
        for index in range(len(x) - 1):
            x1, x2, y1, y2 = x[index], x[index + 1], y[index], y[index + 1]
            if abs(y1) <= tolerance:
                zeros.append(float(x1))
            elif y1 * y2 < 0.0 and x2 - x1 > close:
                zeros.append(float(x1 - y1 * (x2 - x1) / (y2 - y1)))
        if abs(y[-1]) <= tolerance:
            zeros.append(float(x[-1]))
    jumps = [float(0.5 * (x[i] + x[i + 1]))
             for i in range(len(x) - 1)
             if x[i + 1] - x[i] <= close
             and abs(y[i + 1] - y[i]) > tolerance]

    def unique(values: list[float]) -> list[float]:
        out: list[float] = []
        for value in values:
            if not out or abs(value - out[-1]) > close:
                out.append(value)
        return out

    i_max, i_min = int(np.argmax(y)), int(np.argmin(y))
    return {
        "max": (float(x[i_max]), float(y[i_max])),
        "min": (float(x[i_min]), float(y[i_min])),
        "zeros": unique(sorted(zeros)),
        "jumps": unique(sorted(jumps)),
    }


class DiagramPanel(QWidget):
    """沿杆长的内力曲线。"""

    locate = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        self.case: str | None = None

        box = QVBoxLayout(self)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("分量"))
        self.component = QComboBox(self)
        self.component.addItems(list(COMPONENTS))
        self.component.setCurrentText("弯矩 Mz（平面内，最常用）")
        self.component.currentTextChanged.connect(self.redraw)
        row.addWidget(self.component)

        row.addWidget(QLabel("杆件"))
        self.member = QComboBox(self)
        self.member.setMinimumWidth(90)
        self.member.currentTextChanged.connect(self._on_member)
        row.addWidget(self.member)

        self.envelope = QCheckBox("叠加各组合包络", self)
        self.envelope.setToolTip(
            "逐点取所有工况与组合的上下界。同一根杆上跨中与支座"
            "常由不同组合控制，包络能一眼看出这一点")
        self.envelope.stateChanged.connect(self.redraw)
        row.addWidget(self.envelope)

        self.btn_png = QPushButton("导出 PNG…", self)
        self.btn_png.clicked.connect(self.export_png)
        row.addWidget(self.btn_png)
        row.addStretch(1)
        box.addLayout(row)

        self.canvas = _Canvas(self)
        box.addWidget(self.canvas, 1)

        self.caption = QLabel("尚无分析结果")
        self.caption.setWordWrap(True)
        self.caption.setProperty("panel", "hint")
        box.addWidget(self.caption)

    # --- 数据 ---

    def attach(self, session, case: str | None) -> None:
        """换模型或换工况时调。**杆件下拉要重填**，
        否则会留着上一个模型的编号，选中一个不存在的杆。"""
        self.session = session
        self.case = case
        frame = getattr(session, "frame", None)
        current = self.member.currentText()
        self.member.blockSignals(True)
        self.member.clear()
        if frame is not None:
            self.member.addItems([str(m) for m in sorted(frame.members)])
            if current in {str(m) for m in frame.members}:
                self.member.setCurrentText(current)
        self.member.blockSignals(False)
        self.redraw()

    def _on_member(self, text: str) -> None:
        if text.isdigit():
            self.locate.emit("member", int(text))
        self.redraw()

    def current_member(self) -> int | None:
        text = self.member.currentText()
        return int(text) if text.isdigit() else None

    def series(self) -> dict | None:
        """算出要画的曲线。**纯数据，不碰画布**——这样它能单独测。"""
        from internal_forces import member_diagram

        session, mid = self.session, self.current_member()
        if session is None or mid is None or session.solution is None:
            return None
        comp = COMPONENTS[self.component.currentText()]
        frame, solution = session.frame, session.solution
        case = self.case or solution.primary
        try:
            d = member_diagram(frame, solution, mid, case, stations=61)
        except Exception:                       # noqa: BLE001
            return None
        # **换算走 units.py 那一套，不要自己再写一份。**
        # `member_diagram` 给的是模型内部单位（N-m-Pa 下就是 N、N·m）；
        # 直接标成 kN·m 会差一千倍——而图照样画得出来，只是数全错了。
        # 这个 bug 我犯过：包络带宽显示成 70787 kN·m，实际是 70.8。
        system = unit_system(session.model)
        scale = system.moment_scale if comp in MOMENTS else system.force_scale
        unit = system.moment_unit if comp in MOMENTS else system.force_unit

        # 横坐标统一以物理米显示；MM 模型内部的 d.x 是毫米，不能只改轴标签。
        out = {"x": np.asarray(d.x) * system.length_to_m,
               "y": np.asarray(d.component(comp)) * scale,
               "component": comp, "case": case, "member": mid,
               "unit": unit, "scale": scale, "x_unit": "m",
               "axis": f"local x: N{frame.members[mid].i} -> N{frame.members[mid].j}",
               "sign_convention": SIGN_CONVENTIONS[comp]}
        out["features"] = diagram_features(out["x"], out["y"])

        if self.envelope.isChecked():
            cases = list(solution.all_results())
            stack = []
            for name in cases:
                try:
                    other = member_diagram(frame, solution, mid, name,
                                           stations=61)
                except Exception:               # noqa: BLE001
                    continue
                stack.append(np.asarray(other.component(comp)))
            if stack:
                block = np.vstack(stack) * scale
                out["upper"] = block.max(axis=0)
                out["lower"] = block.min(axis=0)
                # 每一点由谁控制。**这才是包络真正有用的地方**：
                # 跨中和支座常常不是同一个组合说了算
                out["governs_max"] = [cases[i] for i in block.argmax(axis=0)]
                out["governs_min"] = [cases[i] for i in block.argmin(axis=0)]
        return out

    # --- 绘制 ---

    def redraw(self) -> None:
        data = self.series()
        self.canvas.draw_series(data)
        if data is None:
            self.caption.setText(
                "尚无分析结果，或当前杆件无内力数据。请先求解。")
            return
        y = data["y"]
        peak = float(np.abs(y).max()) if y.size else 0.0
        at = float(data["x"][int(np.abs(y).argmax())]) if y.size else 0.0
        text = (f"杆件 {data['member']}　工况 {data['case']}　"
                f"{data['component']} 峰值 {peak:.3f} {data['unit']}"
                f" 位于 x = {at:.3f} m")
        if "governs_max" in data:
            names = []
            for n in data["governs_max"] + data["governs_min"]:
                if n not in names:
                    names.append(n)
            text += f"　在某处控制的组合：{'、'.join(names)}"
        features = data["features"]
        text += (f"　{data['axis']}；{data['sign_convention']}；"
                 f"过零点 {len(features['zeros'])} 个，跳跃点 "
                 f"{len(features['jumps'])} 个。")
        self.caption.setText(text)

    def export_png(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        data = self.series()
        if data is None:
            QMessageBox.information(self, "无可导出的内力图",
                                    "请先求解并选择一根杆件。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出内力图",
            f"{data['component']}_杆件{data['member']}.png", "PNG 图片 (*.png)")
        if path:
            self.canvas.figure.savefig(path, dpi=200,
                                       facecolor=self.canvas.figure.get_facecolor())


class _Canvas(QWidget):
    """matplotlib 画布。和视口分开，因为它是二维的，也不需要 OpenGL。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 2.6), facecolor=theme.PANEL)
        self.canvas = FigureCanvasQTAgg(self.figure)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self._style()

    def _style(self) -> None:
        ax = self.ax
        ax.set_facecolor(theme.PANEL)
        for side in ax.spines.values():
            side.set_color(theme.BORDER)
        ax.tick_params(colors=theme.INK_MUTED, labelsize=8)
        ax.xaxis.label.set_color(theme.INK_MUTED)
        ax.yaxis.label.set_color(theme.INK_MUTED)
        ax.grid(True, color=theme.BORDER, linewidth=0.5, alpha=0.6)

    def draw_series(self, data: dict | None) -> None:
        self.ax.clear()
        self._style()
        if data is None:
            self.canvas.draw_idle()
            return
        x, y = data["x"], data["y"]
        self.ax.axhline(0.0, color=theme.INK_MUTED, linewidth=1.0)
        if "upper" in data:
            self.ax.fill_between(x, data["lower"], data["upper"],
                                 color=theme.ACCENT, alpha=0.18,
                                 label="各组合包络")
            self.ax.plot(x, data["upper"], color=theme.ACCENT,
                         linewidth=0.9, alpha=0.7)
            self.ax.plot(x, data["lower"], color=theme.ACCENT,
                         linewidth=0.9, alpha=0.7)
        self.ax.fill_between(x, 0.0, y, where=y >= 0.0,
                             color=theme.ACCENT, alpha=0.30, interpolate=True)
        self.ax.fill_between(x, 0.0, y, where=y < 0.0,
                             color=theme.LOAD, alpha=0.24, interpolate=True)
        self.ax.plot(x, y, color=theme.INK, linewidth=1.6,
                     label=data["case"])
        features = data.get("features", {})
        for label, item in (("max", features.get("max")),
                            ("min", features.get("min"))):
            if item is not None:
                self.ax.scatter(*item, s=18, color=theme.INK, zorder=4)
                self.ax.annotate(f"{item[1]:+.3g}", item, xytext=(4, 5),
                                 textcoords="offset points", fontsize=7,
                                 color=theme.INK_MUTED)
        zeros = features.get("zeros", [])
        if zeros:
            self.ax.scatter(zeros, np.zeros(len(zeros)), s=16,
                            facecolors=theme.PANEL_ALT, edgecolors=theme.INK,
                            zorder=4, label="过零点")
        for index, at in enumerate(features.get("jumps", [])):
            self.ax.axvline(at, color=theme.WARN, linestyle="--",
                            linewidth=0.9, alpha=0.8,
                            label="跳跃点" if index == 0 else None)
        self.ax.set_xlabel(
            f"{data.get('axis', 'local x')} ({data.get('x_unit', 'm')})")
        self.ax.set_ylabel(f"signed {data['component']} ({data['unit']})")
        self.ax.text(0.01, 0.98, data.get("sign_convention", ""),
                     transform=self.ax.transAxes, va="top", ha="left",
                     fontsize=7, color=theme.INK_MUTED)
        legend = self.ax.legend(fontsize=8, facecolor=theme.PANEL_ALT,
                                edgecolor=theme.BORDER)
        for text in legend.get_texts():
            text.set_color(theme.INK_MUTED)
        self.figure.tight_layout()
        self.canvas.draw_idle()
