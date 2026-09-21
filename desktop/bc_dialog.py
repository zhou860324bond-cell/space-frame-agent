"""Abaqus 式边界条件对话框。

对应 Abaqus Load 模块 → Create BC → Displacement/Rotation：
- 选择对象（节点）
- 勾选要约束的自由度：U1, U2, U3, UR1, UR2, UR3
- 预设：固定端、铰接、滚动

美化版：统一标题栏、分组区域、自由度矩阵布局。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox,
                               QGridLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout)

from . import dialog_styles, theme
from .dialog_styles import DialogHeader, DialogSection, FormRow, HintLabel, style_dialog

DOFS = [
    ("U1", "X 向平动"),
    ("U2", "Y 向平动"),
    ("U3", "Z 向平动"),
    ("UR1", "绕 X 转动"),
    ("UR2", "绕 Y 转动"),
    ("UR3", "绕 Z 转动"),
]

PRESETS = {
    "固定端": [1, 1, 1, 1, 1, 1],
    "铰接": [1, 1, 1, 0, 0, 0],
    "X 向滚动": [0, 1, 1, 0, 0, 0],
    "Z 向滚动": [1, 1, 0, 0, 0, 0],
    "自由": [0, 0, 0, 0, 0, 0],
}


class BCDialog(QDialog):
    """边界条件对话框 —— 勾选自由度。"""

    def __init__(self, node_id: int, current_fix: list[int] | None = None,
                 current_name: str | None = None, parent=None,
                 current_spring: list[float] | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"边界条件 — 节点 {node_id}")
        style_dialog(self, 440, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(0)

        # 标题栏
        header = DialogHeader(
            f"边界条件 — 节点 {node_id}",
            "设置节点的约束自由度，勾选表示该方向被约束"
        )
        layout.addWidget(header)

        # 内容区域
        content = QVBoxLayout()
        content.setContentsMargins(16, 12, 16, 0)
        content.setSpacing(12)

        # 基本信息分组
        basic_section = DialogSection("基本信息")
        self.txt_name = QLineEdit(current_name or f"BC-Node-{node_id}")
        self.txt_name.setMinimumHeight(28)
        basic_section.addLayout(FormRow("名称", self.txt_name, 80))
        basic_section.addLayout(FormRow("分析步", QLabel("Initial（基础约束）"), 80))
        content.addWidget(basic_section)

        # 预设分组
        preset_section = DialogSection("快速预设")
        preset_grid = QGridLayout()
        preset_grid.setSpacing(6)
        for i, name in enumerate(PRESETS):
            btn = QPushButton(name)
            btn.setMinimumHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, n=name: self._apply_preset(n))
            preset_grid.addWidget(btn, i // 3, i % 3)
        preset_section.addLayout(preset_grid)
        content.addWidget(preset_section)

        # 自由度分组
        dof_section = DialogSection("自由度约束（勾选表示约束）")
        dof_grid = QGridLayout()
        dof_grid.setSpacing(8)
        self.checks: list[QCheckBox] = []
        self.springs: list[QDoubleSpinBox] = []
        for i, (code, desc) in enumerate(DOFS):
            # 自由度代码标签
            code_label = QLabel(code)
            code_label.setStyleSheet(
                f"color: {theme.ACCENT}; font-weight: 700; font-size: 10pt; "
                f"background: transparent; min-width: 36px;"
            )
            code_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dof_grid.addWidget(code_label, i, 0)

            # 描述标签
            desc_label = QLabel(desc)
            desc_label.setStyleSheet(
                f"color: {theme.INK_MUTED}; font-size: 9pt; background: transparent;"
            )
            dof_grid.addWidget(desc_label, i, 1)

            # 勾选框
            cb = QCheckBox()
            cb.setMinimumHeight(24)
            cb.setCursor(Qt.CursorShape.PointingHandCursor)
            dof_grid.addWidget(cb, i, 2)
            self.checks.append(cb)

            # 弹簧刚度。**真实支座几乎总在"完全固定"和"完全自由"之间**——
            # 桩基、弹性地基、橡胶支座都是。以前这里只有勾选框，等于逼用户
            # 在两个极端里挑一个。
            box = QDoubleSpinBox()
            box.setRange(0.0, 1.0e15)
            box.setDecimals(1)
            box.setSingleStep(1.0e6)
            box.setMinimumHeight(24)
            box.setToolTip(
                "支承刚度，0 表示该方向没有弹簧。"
                + ("平动：力/长度（N-m-Pa 下是 N/m）" if i < 3
                   else "转动：力·长度/弧度（N·m/rad）"))
            # 勾了刚性约束，弹簧就用不上了：刚性会把自由度整个划掉。
            # 直接禁用比让用户填完再被拒绝要好。
            cb.toggled.connect(
                lambda checked, w=box: (w.setEnabled(not checked),
                                        w.setValue(0.0) if checked else None))
            dof_grid.addWidget(box, i, 3)
            self.springs.append(box)

        dof_section.addLayout(dof_grid)
        dof_section.addWidget(HintLabel(
            "U1/U2/U3 = 平动自由度，UR1/UR2/UR3 = 转动自由度。"
            "右侧一栏是弹性支承刚度，留 0 表示不设弹簧；"
            "勾了约束的方向弹簧会被自动禁用——刚性约束会让弹簧完全失效。"))
        content.addWidget(dof_section)

        layout.addLayout(content, 1)

        # 按钮
        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(16, 8, 16, 0)
        btns = dialog_styles.button_box(self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btn_layout.addWidget(btns)
        layout.addLayout(btn_layout)

        # 设置当前值
        if current_fix:
            for cb, val in zip(self.checks, current_fix, strict=True):
                cb.setChecked(bool(val))
        if current_spring:
            for box, val in zip(self.springs, current_spring, strict=True):
                if box.isEnabled():
                    box.setValue(float(val))

    def _apply_preset(self, name: str):
        fix = PRESETS[name]
        for cb, val in zip(self.checks, fix, strict=True):
            cb.setChecked(bool(val))

    def get_fix(self) -> list[int]:
        return [1 if cb.isChecked() else 0 for cb in self.checks]

    def get_spring(self) -> list[float] | None:
        """六个方向的支承刚度；全零时返回 None（表示不设弹簧）。"""
        values = [float(box.value()) for box in self.springs]
        return values if any(values) else None

    def get_name(self) -> str:
        return self.txt_name.text().strip()
