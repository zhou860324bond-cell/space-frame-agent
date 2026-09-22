"""对话框统一样式辅助模块。

走专业工程软件（Abaqus/CAE）的对话框语言：
- 顶部只有一条朴素标题 + 一行说明，不做渐变大横幅；
- 分组用经典的细边框 GroupBox（标题压在边框线上），不用填色卡片；
- 表单标签右对齐、定宽，保证输入框左缘整齐。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialogButtonBox, QFrame, QHBoxLayout, QLabel,
                               QSizePolicy, QVBoxLayout, QWidget)

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


def emphasis(text: str) -> str:
    """把 `**重点**` 变成真正的加粗，顺带堵住 Markdown 星号漏进界面。

    这个项目的中文说明一律是 Markdown 风格写的——docstring、工具描述、
    返回值里的 note 全是 `**这样**`。同一句话复制到界面上时，星号会**原样
    显示**：用户看到的是「漏写不等于撤销」外面挂着四个星号。不会崩、不会
    报错，只是难看且显得业余，所以没人会专门去测它。

    `result_rows.py` 早就为此写过一行 `replace("**", "")`——那是把重点抹掉。
    这里改成真的加粗：作者想强调的那一处，界面上也确实被强调。
    """
    from html import escape

    parts = escape(str(text)).split("**")
    # 偶数段是正文、奇数段是被星号夹住的内容；星号数量不成对时原样退回，
    # 猜一个"大概是想加粗哪里"只会得到更奇怪的结果。
    if len(parts) % 2 == 0:
        return escape(str(text))
    return "".join(p if i % 2 == 0 else f"<b>{p}</b>"
                   for i, p in enumerate(parts))


class HintLabel(QLabel):
    """提示文字：小号、灰色、无背景。`**重点**` 会被渲染成加粗。

    **自动换行的 QLabel 默认会被压扁。** 它的 sizeHint 是按"排成一行"算的，
    布局照那个高度分配空间，于是折行之后最后一两行被切掉——切掉的往往正是
    那句"选错不会报错"。实测规范组合少 9px、面荷载少 26px。

    所以这里把 heightForWidth 接进尺寸策略，并在每次改变宽度后据实抬高
    minimumHeight。提示文字是这个项目交付能力边界的主要位置，被切掉等于
    没写。
    """

    def __init__(self, text: str, parent=None):
        super().__init__(emphasis(text), parent)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setStyleSheet(
            f"color: {theme.INK_DIM}; font-size: 8pt; "
            f"padding: 2px 4px; background: transparent; border: none;")
        self.setWordWrap(True)
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        policy.setVerticalPolicy(QSizePolicy.Policy.MinimumExpanding)
        self.setSizePolicy(policy)

    def setText(self, text: str) -> None:      # noqa: N802 — Qt 的命名
        super().setText(emphasis(text))
        self._fit()

    def resizeEvent(self, event):              # noqa: N802 — Qt 的命名
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        width = self.width()
        if width > 0:
            self.setMinimumHeight(self.heightForWidth(width))


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


def button_box(parent=None, ok: str = "确定", cancel: str = "取消"
               ) -> QDialogButtonBox:
    """确定 / 取消按钮条。

    **走这个入口，不要各对话框自己 new 一个。** Qt 的标准按钮默认取系统语言的
    文案，机器语言不是中文时就是 "OK / Cancel"——于是同一个程序里，材料对话框
    写着「确定」，截面对话框写着 "OK"。这种不一致一眼就能看见，
    而且每加一个对话框就多一次机会漏掉。

    需要别的动作名（"应用""写回模型"）时传 ``ok``，语言仍然是一致的。
    """
    box = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel, parent=parent)
    box.button(QDialogButtonBox.StandardButton.Ok).setText(ok)
    box.button(QDialogButtonBox.StandardButton.Cancel).setText(cancel)
    return box


def close_box(parent=None, ok: str = "关闭") -> QDialogButtonBox:
    """只有一个按钮的按钮条（查看类弹窗）。同样是为了文案一致。"""
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, parent=parent)
    box.button(QDialogButtonBox.StandardButton.Ok).setText(ok)
    return box
