"""视口左下角的坐标系指示器。

用 QPainter 画 XYZ 三轴，颜色与杆件局部轴一致：X 红、Y 绿、Z 蓝。

**它跟着相机转。** 原来这里画的是一个固定朝向的三轴，旁边还有 VTK 自带的
会转的小三轴：两个指示器同时出现、方向还不一致，用户不知道该信哪个。
现在只留这一个，由视口在每次渲染前把相机朝向传进来。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRegion
from PySide6.QtWidgets import QWidget

from . import theme

#: 等轴测默认朝向下三根轴在屏幕上的方向（右、上、朝向观察者），相机信息
#: 还没传进来时先用它画，不出现一个空圈。
_ISOMETRIC = np.array([[0.7071, -0.7071, 0.0],
                       [-0.4082, -0.4082, 0.8165],
                       [0.5774, 0.5774, 0.5774]])
AXIS_LENGTH = 26


class AxisIndicator(QWidget):
    """左下角 XYZ 三轴指示器，随相机旋转。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 80)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # 不用半透明背景：原生子窗口在 OpenGL 视口上会整个不显示（见
        # viewport_legend._Card），改成不透明圆盘，圆形靠遮罩裁出
        # 内缩 1 px：150% 缩放下遮罩换算到物理像素会多出一行，露出一条没画到的白线
        self.setMask(QRegion(self.rect().adjusted(1, 1, -1, -1), QRegion.RegionType.Ellipse))
        self._rotation = _ISOMETRIC.copy()
        self._light = False

    def set_rotation(self, rotation) -> bool:
        """传入相机视图矩阵左上 3×3（行：屏幕右、屏幕上、朝向观察者）。

        返回是否真的变了——视口每次渲染都会调，只有变了才重画。
        """
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        if np.allclose(rotation, self._rotation, atol=1e-4):
            return False
        self._rotation = rotation
        self.update()
        return True

    def set_light(self, light: bool) -> None:
        self._light = bool(light)
        self.update()

    def axis_tip(self, axis: str) -> QPointF:
        """某根轴末端在本控件里的像素位置（导出截图的检查用）。"""
        k = "xyz".index(axis)
        centre = QPointF(self.width() / 2, self.height() / 2)
        return centre + QPointF(self._rotation[0, k], -self._rotation[1, k]) * AXIS_LENGTH

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            colours = theme.viewport_card(self._light)
            centre = QPointF(self.width() / 2, self.height() / 2)
            # 先铺满遮罩内的整块，遮罩边缘不抗锯齿，不能留没画到的缝
            p.fillRect(self.rect(), QColor(*colours["surface"][:3]))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(*colours["edge"]), 1))
            radius = self.width() / 2 - 2
            p.drawEllipse(centre, radius, radius)

            font = QFont()
            font.setPointSizeF(float(theme.FONT_CAPTION.rstrip("pt")))
            font.setBold(True)
            p.setFont(font)
            # 远的先画、近的后画，朝向观察者的轴压在上面
            for k in sorted(range(3), key=lambda k: self._rotation[2, k]):
                axis = "xyz"[k]
                colour = QColor(theme.LOCAL_AXES[axis])
                direction = QPointF(self._rotation[0, k], -self._rotation[1, k])
                tip = centre + direction * AXIS_LENGTH
                pen = QPen(colour, 2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(centre, tip)
                label = centre + direction * (AXIS_LENGTH + 8)
                p.drawText(QRectF(label.x() - 8, label.y() - 8, 16, 16),
                           Qt.AlignmentFlag.AlignCenter, axis.upper())

            p.setBrush(QColor(colours["muted"]))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(centre, 2, 2)
        finally:
            p.end()
