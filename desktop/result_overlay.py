"""视口结果摘要使用 Qt 字体，中文说明与缩放保持可读。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu,
                               QToolButton, QVBoxLayout, QWidget, QWidgetAction)

from . import i18n
from .qt_style import ElidedLabel


class ResultOverlay(QFrame):
    """数值与量程常驻，完整定义通过说明菜单展开。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewportResultCard")
        self.active = False
        box = QVBoxLayout(self)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(3)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.title = ElidedLabel(self)
        self.title.setObjectName("resultOverlayTitle")
        header.addWidget(self.title, 1)
        self.info_button = QToolButton(self)
        self.info_button.setText(i18n.tr("说明"))
        self.info_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.info_menu = QMenu(self)
        panel = QWidget(self.info_menu)
        details_box = QVBoxLayout(panel)
        details_box.setContentsMargins(12, 12, 12, 12)
        self.details = QLabel(panel)
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setMinimumWidth(240)
        self.details.setMaximumWidth(370)
        details_box.addWidget(self.details)
        action = QWidgetAction(self.info_menu)
        action.setDefaultWidget(panel)
        self.info_menu.addAction(action)
        self.info_button.setMenu(self.info_menu)
        header.addWidget(self.info_button)
        box.addLayout(header)
        self.peak = ElidedLabel(self)
        self.range = ElidedLabel(self)
        self.status = ElidedLabel(self)
        self.status.setObjectName("resultOverlayStatus")
        for label in (self.title, self.peak, self.range, self.status):
            label.setProperty("i18nSelfManaged", True)
        for label in (self.peak, self.range, self.status):
            box.addWidget(label)
        self.reset()

    def set_content(self, *, title: str, peak: str, range_text: str,
                    status: str, details: str, light: bool = False) -> None:
        self.active = True
        for label, text in ((self.title, title), (self.peak, peak),
                            (self.range, range_text), (self.status, status)):
            label.setText(text)
            label.setToolTip(text)
            label.setVisible(bool(text))
        self.details.setText(details)
        self.info_button.setToolTip(details)
        self.info_button.setText(i18n.tr("说明"))
        self.info_button.setVisible(bool(details))
        self.set_light(light)
        self.adjustSize()
        self.show()
        self.raise_()

    def set_light(self, light: bool) -> None:
        """切换背景时同步文字配色，不重建模型和结果。"""
        surface = "rgba(248,250,253,240)" if light else "rgba(27,37,49,235)"
        ink = "#253444" if light else "#edf3fa"
        muted = "#526b83" if light else "#b7c6d6"
        self.setStyleSheet(f"""
            QFrame#viewportResultCard {{ background:{surface}; border:1px solid {muted}; border-radius:6px; }}
            QFrame#viewportResultCard QLabel {{ background:transparent; color:{ink}; font-size:9.5pt; }}
            QLabel#resultOverlayTitle {{ font-size:10pt; font-weight:600; }}
            QLabel#resultOverlayStatus {{ color:{muted}; font-size:9pt; }}
            QFrame#viewportResultCard QToolButton {{ color:{ink}; background:transparent; border:0; padding:2px 4px; }}
        """)
        self.info_menu.setStyleSheet("QMenu { background:#ffffff; border:1px solid #b4c1ce; } QLabel { color:#253444; background:transparent; font-size:10pt; }")

    def reset(self) -> None:
        self.active = False
        self.info_menu.close()
        self.hide()
