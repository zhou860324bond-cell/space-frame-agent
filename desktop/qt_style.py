"""自绘几个样式表画不出来的原生控件。

样式表能改颜色和边框，画不了**形状**。于是勾选框的"勾"只能用一张图片，
或者干脆不画——我们以前是后者：选中态是一个纯色方块，没有对勾。更糟的是
单选钮走的是同一条规则，被画成了和勾选框一模一样的方块，**用户分不出
「多选」和「单选」**，而这是两种完全不同的交互承诺。

数字框的上下按钮同理：没有样式表规则时走原生绘制，在我们这套浅色 chrome 上
显示成一个残缺的角标。

这些都不会报错，只是看着不对。QProxyStyle 让我们接管这几个图元的绘制，
其余一概交还给原生样式——**不重写整套样式**，那会把 Qt 的键盘焦点、高 DPI
缩放和无障碍行为一起接管过来，代价远大于收益。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QProxyStyle, QStyle, QStyleOption, QWidget

from . import theme

#: 指示器边长。比 Qt 默认的 13px 大一点——13px 下中文界面里的对勾只有几个
#: 像素，看不清是勾还是点。
INDICATOR = 16
BORDER_WIDTH = 1.6


class FrameStyle(QProxyStyle):
    """接管勾选框、单选钮与数字框箭头的绘制，其余交还原生样式。"""

    def pixelMetric(self, metric, option=None, widget=None) -> int:
        if metric in (QStyle.PixelMetric.PM_IndicatorWidth,
                      QStyle.PixelMetric.PM_IndicatorHeight,
                      QStyle.PixelMetric.PM_ExclusiveIndicatorWidth,
                      QStyle.PixelMetric.PM_ExclusiveIndicatorHeight):
            return INDICATOR
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(self, element: QStyle.PrimitiveElement,
                      option: QStyleOption, painter: QPainter,
                      widget: QWidget | None = None) -> None:
        if element in (QStyle.PrimitiveElement.PE_IndicatorCheckBox,
                       QStyle.PrimitiveElement.PE_IndicatorItemViewItemCheck):
            self._checkbox(option, painter)
            return
        if element == QStyle.PrimitiveElement.PE_IndicatorRadioButton:
            self._radio(option, painter)
            return
        if element in (QStyle.PrimitiveElement.PE_IndicatorSpinUp,
                       QStyle.PrimitiveElement.PE_IndicatorSpinDown,
                       QStyle.PrimitiveElement.PE_IndicatorArrowUp,
                       QStyle.PrimitiveElement.PE_IndicatorArrowDown):
            up = element in (QStyle.PrimitiveElement.PE_IndicatorSpinUp,
                             QStyle.PrimitiveElement.PE_IndicatorArrowUp)
            self._chevron(option, painter, up=up)
            return
        super().drawPrimitive(element, option, painter, widget)

    # --- 各图元 -----------------------------------------------------------

    @staticmethod
    def _box(option: QStyleOption) -> QRectF:
        """指示器的方框，居中并留出画边框的余量。"""
        side = min(INDICATOR, option.rect.width(), option.rect.height())
        left = option.rect.x() + (option.rect.width() - side) / 2
        top = option.rect.y() + (option.rect.height() - side) / 2
        return QRectF(left + 1.0, top + 1.0, side - 2.0, side - 2.0)

    @staticmethod
    def _states(option: QStyleOption) -> tuple[bool, bool, bool, bool, bool]:
        state = option.state
        return (bool(state & QStyle.StateFlag.State_Enabled),
                bool(state & QStyle.StateFlag.State_MouseOver),
                bool(state & QStyle.StateFlag.State_HasFocus),
                bool(state & QStyle.StateFlag.State_On),
                bool(state & QStyle.StateFlag.State_NoChange))

    def _colors(self, enabled: bool, hovered: bool, focused: bool,
                marked: bool) -> tuple[QColor, QColor]:
        if not enabled:
            return (QColor(theme.BORDER),
                    QColor(theme.BORDER_LIGHT if marked else theme.PANEL_ALT))
        if marked:
            return (QColor(theme.ACCENT),
                    QColor(theme.ACCENT_HOVER if hovered else theme.ACCENT_DIM))
        return (QColor(theme.ACCENT if hovered or focused else theme.BORDER_LIGHT),
                QColor(theme.SELECTION if hovered else theme.PANEL_ALT))

    def _checkbox(self, option: QStyleOption, painter: QPainter) -> None:
        enabled, hovered, focused, checked, partial = self._states(option)
        border, fill = self._colors(enabled, hovered, focused, checked or partial)
        box = self._box(option)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(border, BORDER_WIDTH))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, 2.5, 2.5)

        if checked or partial:
            mark = QPen(QColor(theme.PANEL_ALT if enabled else theme.INK_DIM), 2.0)
            mark.setCapStyle(Qt.PenCapStyle.RoundCap)
            mark.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(mark)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if checked:
                # 对勾按方框比例画，指示器尺寸变了也不用重新调点
                path = QPainterPath(QPointF(box.left() + box.width() * 0.22,
                                            box.top() + box.height() * 0.53))
                path.lineTo(QPointF(box.left() + box.width() * 0.43,
                                    box.top() + box.height() * 0.74))
                path.lineTo(QPointF(box.left() + box.width() * 0.79,
                                    box.top() + box.height() * 0.28))
                painter.drawPath(path)
            else:
                # 半选：一条横杠。表格里"部分子项选中"就是这个状态，
                # 画成整勾会让人以为全选了。
                painter.drawLine(
                    QPointF(box.left() + box.width() * 0.24, box.center().y()),
                    QPointF(box.left() + box.width() * 0.76, box.center().y()))
        painter.restore()

    def _radio(self, option: QStyleOption, painter: QPainter) -> None:
        """**画成圆的。** 方的单选钮和勾选框长得一样，那是在骗用户。"""
        enabled, hovered, focused, checked, _ = self._states(option)
        border, fill = self._colors(enabled, hovered, focused, False)
        box = self._box(option)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(
            QColor(theme.ACCENT) if checked and enabled else border,
            BORDER_WIDTH))
        painter.setBrush(fill)
        painter.drawEllipse(box)
        if checked:
            radius = box.width() * 0.26
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.ACCENT_DIM if enabled
                                    else theme.BORDER_LIGHT))
            painter.drawEllipse(box.center(), radius, radius)
        painter.restore()

    @staticmethod
    def _chevron(option: QStyleOption, painter: QPainter, up: bool) -> None:
        """数字框与下拉框的小箭头。

        画成两笔的 chevron 而不是实心三角：实心三角在 5px 尺度上会糊成一个
        点，而这套界面里数字框到处都是。
        """
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        rect = QRectF(option.rect)
        if rect.width() <= 2 or rect.height() <= 2:
            return
        width = min(7.0, rect.width() - 2)
        height = min(4.0, rect.height() - 2)
        cx, cy = rect.center().x(), rect.center().y()
        top = cy - height / 2
        bottom = cy + height / 2

        pen = QPen(QColor(theme.INK_MUTED if enabled else theme.BORDER), 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(cx - width / 2, bottom if up else top))
        path.lineTo(QPointF(cx, top if up else bottom))
        path.lineTo(QPointF(cx + width / 2, bottom if up else top))
        painter.drawPath(path)
        painter.restore()


def install(app) -> FrameStyle:
    """把自绘样式装到 QApplication 上。返回它，便于测试直接拿来画。

    **不要写成 ``FrameStyle(app.style())``。** 那个构造函数会接管基样式的
    所有权，而紧接着的 ``setStyle`` 又会删掉应用当前的样式——代理手里就成了
    野指针，之后随便画点什么都可能访问违例。实测全量测试跑到 3% 崩掉。

    默认构造的 QProxyStyle 代理的是应用当前样式，且由 Qt 自己管生命周期。
    """
    style = FrameStyle()
    app.setStyle(style)
    return style
