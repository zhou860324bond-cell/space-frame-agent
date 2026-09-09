"""视口左下角的坐标系指示器。

用 QPainter 画 XYZ 三轴，颜色遵循工程惯例：X红、Y绿、Z蓝。
固定在视口左下角，不随相机旋转——它是"世界坐标朝哪"的路标，
不是当前视角的指南针。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPoint, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class AxisIndicator(QWidget):
    """左下角 XYZ 三轴指示器。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 80)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)

            cx = self.width() // 2
            cy = self.height() // 2
            axis_len = 25

            # 背景圆：极淡的深色底，细边框
            p.setBrush(QColor(27, 32, 39, 150))
            p.setPen(QPen(QColor(61, 66, 72), 1))
            p.drawEllipse(QPoint(cx, cy), 32, 32)

            font = QFont("Segoe UI", 7, QFont.Weight.Bold)
            p.setFont(font)

            # X 轴（工程红，向右下）
            x_end = QPoint(cx + axis_len, cy + axis_len // 3)
            p.setPen(QPen(QColor("#d15b5b"), 2))
            p.drawLine(cx, cy, x_end.x(), x_end.y())
            p.setPen(QColor("#d15b5b"))
            p.drawText(QRectF(x_end.x() - 8, x_end.y() - 8, 16, 16),
                       Qt.AlignmentFlag.AlignCenter, "X")

            # Y 轴（沉稳绿，向左下）
            y_end = QPoint(cx - axis_len // 2, cy + axis_len)
            p.setPen(QPen(QColor("#62a86a"), 2))
            p.drawLine(cx, cy, y_end.x(), y_end.y())
            p.setPen(QColor("#62a86a"))
            p.drawText(QRectF(y_end.x() - 8, y_end.y() - 4, 16, 16),
                       Qt.AlignmentFlag.AlignCenter, "Y")

            # Z 轴（钢蓝，向上）
            z_end = QPoint(cx, cy - axis_len)
            p.setPen(QPen(QColor("#4f8fcf"), 2))
            p.drawLine(cx, cy, z_end.x(), z_end.y())
            p.setPen(QColor("#4f8fcf"))
            p.drawText(QRectF(z_end.x() - 8, z_end.y() - 12, 16, 16),
                       Qt.AlignmentFlag.AlignCenter, "Z")

            # 原点
            p.setBrush(QColor("#c8ccd2"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPoint(cx, cy), 2, 2)

        finally:
            p.end()
