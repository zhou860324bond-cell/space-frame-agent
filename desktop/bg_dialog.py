"""视口背景设置弹窗。

功能区"视图 → 视口背景"弹出本对话框：
- 预设分三组：深色纯色 / 渐变背景 / 浅色纯色，色板网格点选、实时预览；
- 渐变支持四种方向（垂直 / 水平 / 径向 / 对角），与 VTK 背景一一对应；
- 自定义：两个颜色块任选起止色，配方向下拉，一键应用；看中某个预设
  渐变后也可直接换方向；另支持自定义纯色；
- 网格地面开关；确定保留，取消恢复打开前状态。

弹窗只负责"选什么"，真正渲染在 viewport：
set_background_theme / set_custom_solid / set_custom_gradient /
apply_background_value。本文件不直接碰 VTK。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QGridLayout, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout, QWidget)

from . import theme
from .dialog_styles import DialogHeader, DialogSection, HintLabel, style_dialog

# 预设分组：(组名, [(key, 中文名), ...])，key 必须在 viewport.Viewport.BG_THEMES 内
BG_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("纯色 · 深色", [
        ("dark", "经典深色"),
        ("ink", "墨黑"),
        ("charcoal", "炭灰"),
        ("navy", "藏青"),
    ]),
    ("渐变背景", [
        ("deep", "深蓝·垂直"),
        ("midnight", "午夜·垂直"),
        ("teal", "青蓝·垂直"),
        ("plum", "暮紫·垂直"),
        ("horizon", "钢蓝·水平"),
        ("spotlight", "聚光·径向"),
        ("corner_glow", "对角晕染"),
        ("abaqus", "Abaqus 经典"),
    ]),
    ("纯色 · 浅色", [
        ("light", "浅灰"),
        ("white", "纯白"),
        ("paper", "米白"),
        ("sky", "浅蓝"),
    ]),
]

# 渐变方向：(mode, 中文名)
GRAD_CHOICES = [("vertical", "垂直（上→下）"),
                ("horizontal", "水平（左→右）"),
                ("radial", "径向（中心→四周）"),
                ("corner", "对角（中心→角）")]


def value_to_qss(value) -> str:
    """把背景值（纯色 str 或 (mode, base, second)）转成色板用的 QSS。"""
    if isinstance(value, tuple) and len(value) == 3:
        mode, base, second = value
        if mode == "vertical":          # 顶=second，底=base
            return (f"qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                    f"stop:0 {second},stop:1 {base})")
        if mode == "horizontal":        # 左=base，右=second
            return (f"qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                    f"stop:0 {base},stop:1 {second})")
        if mode == "corner":            # 中心偏左上
            return (f"qradialgradient(cx:0.18,cy:0.18,radius:1.15,"
                    f"stop:0 {base},stop:1 {second})")
        return (f"qradialgradient(cx:0.5,cy:0.5,radius:0.78,"
                f"stop:0 {base},stop:1 {second})")
    return str(value)


class ColorChip(QPushButton):
    """可点击的小色块：点击弹取色器，更新自身底色并发出颜色。"""

    color_changed = Signal(str)

    def __init__(self, color: str, title: str, parent=None):
        super().__init__(parent)
        self._color = color
        self._title = title
        self.setFixedSize(34, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("点击选择颜色")
        self._paint()
        self.clicked.connect(self._pick)

    def _paint(self) -> None:
        self.setStyleSheet(
            f"QPushButton{{background:{self._color}; "
            f"border:1px solid {theme.BORDER_LIGHT}; border-radius:2px;}}"
            f"QPushButton:hover{{border:1px solid {theme.ACCENT};}}")

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self, self._title)
        if c.isValid():
            self._color = c.name()
            self._paint()
            self.color_changed.emit(self._color)

    def color(self) -> str:
        return self._color


class SwatchCell(QWidget):
    """一个预设：上方色块（可选中），下方名字。"""

    picked = Signal(str)

    def __init__(self, key: str, name: str, value, parent=None):
        super().__init__(parent)
        self.key = key
        self.btn = QPushButton()
        self.btn.setCheckable(True)
        self.btn.setFixedHeight(46)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        qss_bg = value_to_qss(value)
        self.btn.setStyleSheet(f"""
            QPushButton {{
                background: {qss_bg};
                border: 1px solid {theme.BORDER};
                border-radius: 3px;
            }}
            QPushButton:checked {{
                border: 2px solid {theme.ACCENT};
            }}
            QPushButton:hover {{
                border: 2px solid {theme.ACCENT_DIM};
            }}
        """)
        self.btn.clicked.connect(lambda: self.picked.emit(self.key))

        lbl = QLabel(name)
        lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lbl.setStyleSheet(
            f"color:{theme.INK_MUTED}; font-size:8pt; background:transparent;")

        box = QVBoxLayout(self)
        box.setContentsMargins(2, 2, 2, 2)
        box.setSpacing(3)
        box.addWidget(self.btn)
        box.addWidget(lbl)


class BackgroundDialog(QDialog):
    """视口背景设置弹窗。"""

    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        # 记下打开时状态，取消时恢复
        self._saved_theme = getattr(viewport, "bg_theme", "dark")
        self._saved_custom = getattr(viewport, "custom_bg", None)
        self._saved_grid = getattr(viewport, "show_grid_floor", False)
        self._last_grad_value = None      # 最近一次渐变值，用于"只换方向"

        self.setWindowTitle("视口背景设置")
        style_dialog(self, 486, 588)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 12)
        root.setSpacing(0)
        root.addWidget(DialogHeader(
            "视口背景设置",
            "选择背景配色与渐变方式，点击色板实时预览，确定后保留"))

        body = QVBoxLayout()
        body.setContentsMargins(14, 10, 14, 0)
        body.setSpacing(8)

        palettes = getattr(viewport, "BG_THEMES", {})
        self._cells: dict[str, SwatchCell] = {}

        for group_title, items in BG_GROUPS:
            sec = DialogSection(group_title)
            grid = QGridLayout()
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(6)
            cols = 4
            for i, (key, name) in enumerate(items):
                value = palettes.get(key, "#1b2027")
                cell = SwatchCell(key, name, value, self)
                cell.picked.connect(self._pick_preset)
                grid.addWidget(cell, i // cols, i % cols)
                self._cells[key] = cell
            sec.addLayout(grid)
            body.addWidget(sec)

        # 自定义颜色
        custom_sec = DialogSection("自定义颜色与渐变")
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(self._mini_label("起始色"))
        self.chip_a = ColorChip("#2a3644", "选择起始色", self)
        row1.addWidget(self.chip_a)
        row1.addWidget(self._mini_label("结束色"))
        self.chip_b = ColorChip("#14181e", "选择结束色", self)
        row1.addWidget(self.chip_b)
        row1.addSpacing(4)
        row1.addWidget(self._mini_label("方向"))
        self.cmb_mode = QComboBox()
        for mode, label in GRAD_CHOICES:
            self.cmb_mode.addItem(label, mode)
        self.cmb_mode.setToolTip("渐变方向")
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        row1.addWidget(self.cmb_mode, 1)
        self.btn_apply_grad = QPushButton("应用渐变")
        self.btn_apply_grad.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply_grad.clicked.connect(self._apply_custom_gradient)
        row1.addWidget(self.btn_apply_grad)
        custom_sec.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.btn_solid = QPushButton("自定义纯色…")
        self.btn_solid.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_solid.clicked.connect(self._pick_solid)
        row2.addWidget(self.btn_solid)
        self.lbl_custom = QLabel("")
        self.lbl_custom.setStyleSheet(
            f"color:{theme.INK_DIM}; font-size:8.5pt; background:transparent;")
        row2.addWidget(self.lbl_custom, 1)
        custom_sec.addLayout(row2)
        body.addWidget(custom_sec)

        # 网格地面
        floor_sec = DialogSection("参考地面")
        self.chk_floor = QCheckBox("显示 z=0 网格地面（范围随模型自动调整）")
        self.chk_floor.setChecked(self._saved_grid)
        self.chk_floor.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_floor.toggled.connect(self._on_grid)
        floor_sec.addWidget(self.chk_floor)
        floor_sec.addWidget(HintLabel("浅色背景下网格线自动加深，深色背景下自动变浅"))
        body.addWidget(floor_sec)
        body.addStretch(1)
        root.addLayout(body, 1)

        btn_box = QVBoxLayout()
        btn_box.setContentsMargins(14, 8, 14, 0)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self._reject)
        btn_box.addWidget(btns)
        root.addLayout(btn_box)

        self._mark_current()

    @staticmethod
    def _mini_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{theme.INK_MUTED}; font-size:8.5pt; background:transparent;")
        return lbl

    # --- 选择逻辑 ---

    def _uncheck_all(self) -> None:
        for cell in self._cells.values():
            cell.btn.setChecked(False)

    def _mark_current(self) -> None:
        cur = self._saved_theme
        if cur in self._cells:
            self._cells[cur].btn.setChecked(True)
            value = self.viewport.BG_THEMES.get(cur)
            if isinstance(value, tuple):
                self._last_grad_value = value

    def _pick_preset(self, key: str) -> None:
        self._uncheck_all()
        self._cells[key].btn.setChecked(True)
        self.lbl_custom.setText("")
        value = self.viewport.BG_THEMES.get(key)
        if isinstance(value, tuple):
            self._last_grad_value = value
            for i, (mode, _) in enumerate(GRAD_CHOICES):
                if mode == value[0]:
                    self.cmb_mode.blockSignals(True)
                    self.cmb_mode.setCurrentIndex(i)
                    self.cmb_mode.blockSignals(False)
                    break
        self.viewport.set_background_theme(key)

    def _on_mode_changed(self, _index: int) -> None:
        """只换方向：当前是渐变（预设或自定义）时，用原两色按新方向重应用。"""
        value = self._last_grad_value
        if not (isinstance(value, tuple) and len(value) == 3):
            return
        mode = self.cmb_mode.currentData()
        c1, c2 = self._user_colors(value)
        self._uncheck_all()
        self.viewport.set_custom_gradient(c1, c2, mode)
        self._last_grad_value = self.viewport.custom_bg

    @staticmethod
    def _user_colors(value: tuple) -> tuple[str, str]:
        """把内部 (mode, base, second) 还原成用户语义的 (起始, 结束)。"""
        mode, base, second = value
        if mode == "vertical":      # 用户：起始=顶=second，结束=底=base
            return second, base
        return base, second

    def _apply_custom_gradient(self) -> None:
        c1, c2 = self.chip_a.color(), self.chip_b.color()
        mode = self.cmb_mode.currentData()
        self._uncheck_all()
        self.viewport.set_custom_gradient(c1, c2, mode)
        self._last_grad_value = self.viewport.custom_bg
        self.lbl_custom.setText(f"自定义渐变：{c1} → {c2}")

    def _pick_solid(self) -> None:
        color = QColorDialog.getColor(QColor("#1b2027"), self, "选择背景纯色")
        if not color.isValid():
            return
        hexc = color.name()
        self._uncheck_all()
        self._last_grad_value = None
        self.viewport.set_custom_solid(hexc)
        self.lbl_custom.setText(f"自定义纯色：{hexc}")

    def _on_grid(self, on: bool) -> None:
        self.viewport.toggle_grid_floor(bool(on))

    def _reject(self) -> None:
        # 恢复打开前状态
        if self._saved_custom is not None:
            self.viewport.apply_background_value(self._saved_custom)
        else:
            self.viewport.set_background_theme(self._saved_theme)
        self.viewport.toggle_grid_floor(self._saved_grid)
        self.reject()
