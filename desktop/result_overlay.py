"""视口结果摘要使用 Qt 字体，中文说明与缩放保持可读。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu,
                               QToolButton, QVBoxLayout, QWidget, QWidgetAction)

from . import i18n, theme
from .qt_style import ElidedLabel


class ResultOverlay(QFrame):
    """数值与量程常驻，完整定义通过说明菜单展开。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewportResultCard")
        self.active = False
        box = QVBoxLayout(self)
        box.setContentsMargins(theme.SPACE_MD, theme.SPACE_SM + 2, theme.SPACE_MD, theme.SPACE_MD)
        box.setSpacing(theme.SPACE_XS)
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
        card = theme.viewport_card(light)
        surface = "rgba({},{},{},{})".format(*card["surface"])
        edge = "rgb({},{},{})".format(*card["edge"][:3])
        ink, muted = card["ink"], card["muted"]
        self.setStyleSheet(f"""
            QFrame#viewportResultCard {{ background:{surface}; border:1px solid {edge};
                                        border-radius:{theme.RADIUS_SM}; }}
            QFrame#viewportResultCard QLabel {{ background:transparent; border:0; color:{ink};
                                               font-size:{theme.FONT_BODY}; }}
            QLabel#resultOverlayTitle {{ font-size:{theme.FONT_STRONG}; font-weight:600; }}
            QLabel#resultOverlayStatus {{ color:{muted}; font-size:{theme.FONT_CAPTION}; }}
            QFrame#viewportResultCard QToolButton {{ color:{muted}; background:transparent;
                                                    border:0; padding:2px 4px; }}
            QFrame#viewportResultCard QToolButton:hover {{ color:{ink}; }}
        """)
        self.info_menu.setStyleSheet(
            f"QMenu {{ background:{theme.PANEL_ALT}; border:1px solid {theme.BORDER_LIGHT}; }}"
            f" QLabel {{ color:{theme.INK}; background:transparent; font-size:{theme.FONT_BODY}; }}")

    def reset(self) -> None:
        self.active = False
        self.info_menu.close()
        self.hide()
