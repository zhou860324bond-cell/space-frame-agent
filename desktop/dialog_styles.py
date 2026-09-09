"""对话框统一样式辅助模块。

走专业工程软件（Abaqus/CAE）的对话框语言：
- 顶部只有一条朴素标题 + 一行说明，不做渐变大横幅；
- 分组用经典的细边框 GroupBox（标题压在边框线上），不用填色卡片；
- 表单标签右对齐、定宽，保证输入框左缘整齐。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QVBoxLayout,
                               QWidget)

from . import theme


class DialogHeader(QWidget):
    """对话框顶部标题区：标题 + 一行说明，纯色、细底边，不做横幅。"""

    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.setFixedHeight(48 if description else 34)
        self.setStyleSheet(
            f"background: {theme.PANEL}; "
            f"border-bottom: 1px solid {theme.BORDER};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(1)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {theme.INK}; font-size: 10pt; font-weight: 600; "
            f"background: transparent; border: none;")
        layout.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet(
                f"color: {theme.INK_MUTED}; font-size: 8pt; "
                f"background: transparent; border: none;")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)


class DialogSection(QFrame):
    """经典 GroupBox 式分组：透明底、1px 细边框、标题压在边框线上。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: transparent;
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS_SM};
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 12, 10, 10)
        outer.setSpacing(6)

        # 标题用与面板同色的底，把上边框线遮出一个缺口——QGroupBox 的标准做法
        title_label = QLabel(f"  {title}  ")
        title_label.setStyleSheet(
            f"color: {theme.INK_MUTED}; font-size: 8.5pt; "
            f"background: {theme.PANEL}; border: none;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # 让标题上移压到边框线上
        outer.addWidget(title_label)

        self.content = QVBoxLayout()
        self.content.setContentsMargins(2, 0, 2, 0)
        self.content.setSpacing(6)
        outer.addLayout(self.content)

    def addWidget(self, widget):
        self.content.addWidget(widget)

    def addLayout(self, layout):
        self.content.addLayout(layout)


class FormRow(QHBoxLayout):
    """表单行：标签右对齐、定宽，控件占满剩余，保证输入框左缘对齐。"""

    def __init__(self, label: str, widget, label_width: int = 100):
        super().__init__()
        self.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(label_width)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet(
            f"color: {theme.INK_MUTED}; font-size: 9pt; border: none;")
        self.addWidget(lbl)
        self.addWidget(widget, 1)


class HintLabel(QLabel):
    """提示文字：小号、灰色、无背景。"""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"color: {theme.INK_DIM}; font-size: 8pt; "
            f"padding: 2px 4px; background: transparent; border: none;")
        self.setWordWrap(True)


def style_dialog(dialog, min_width: int = 420, min_height: int = 300):
    """给对话框应用统一、紧凑的专业样式。"""
    dialog.setMinimumSize(min_width, min_height)
    dialog.setStyleSheet(f"""
        QDialog {{
            background: {theme.PANEL};
        }}
        QDialogButtonBox {{
            background: transparent;
            padding: 6px 0;
        }}
        QDialogButtonBox QPushButton {{
            min-width: 76px;
            padding: 5px 18px;
        }}
    """)
