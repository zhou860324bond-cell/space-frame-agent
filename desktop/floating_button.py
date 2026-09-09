"""Compact viewport-edge launcher for the AI assistant dock.

专业风格：扁平钢蓝实心圆，悬停只略微提亮、按下略微压暗，
不做渐变、高光圈层。对话气泡图标用白色线条。
"""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from . import theme


class FloatingAgentButton(QWidget):
    """Small edge tab; hidden while the assistant dock is already visible."""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = False
        self._pressed = False
        self.setFixedSize(QSize(42, 72))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setToolTip("AI 助手（点击展开/收起）")

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self._pressed = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._pressed:
            self._pressed = False
            self.update()
            self.clicked.emit()

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)

            if self._pressed:
                fill = QColor(theme.ACCENT_DIM)
            elif self._hover:
                fill = QColor(theme.ACCENT_HOVER)
            else:
                fill = QColor(theme.PANEL_RAISED)
            p.setBrush(fill)
            p.setPen(QPen(QColor(theme.ACCENT), 1))
            p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 6, 6)
            font = QFont(self.font())
            font.setBold(True)
            font.setPointSizeF(9)
            p.setFont(font)
            p.setPen(QColor("#ffffff"))
            p.drawText(QRect(0, 8, self.width(), 24),
                       Qt.AlignmentFlag.AlignCenter, "AI")
            p.setPen(QColor(theme.INK_MUTED))
            p.drawText(QRect(0, 34, self.width(), 28),
                       Qt.AlignmentFlag.AlignCenter, ">")
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            p.end()
