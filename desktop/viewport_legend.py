"""视口右上角的色标与符号图例，用 Qt 绘制。

原来色标交给 VTK 画，三个问题绕不开：

* VTK 的默认字体没有中文字形，标题只能写成 ``sigma``、``utilization``，
  和左上角结果摘要里的「截面正应力 σ」各说各话；
* 刻度字号按物理像素算，150% 缩放下只有正文的三分之二大；
* 标题是另一个独立的文字对象，量名一长（``M (dominant plane)``）就伸出视口。

这里色标的颜色**直接取自云图映射器的查找表**，不另配一套颜色——色标和
杆件上的颜色由同一张表决定，不会对不上。
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QWidget

from . import theme

PAD = theme.SPACE_MD
BAR_WIDTH = 14
SWATCH = 12
#: 色标与图例的固定宽度。相机适应窗口时给它预留这一条，切换显示模式
#: 时模型不会因为色标出现或消失而左右跳。
WIDTH = 124


def _rgb(colour) -> QColor:
    """接受 0–1 浮点或 0–255 整数的 RGB(A) 元组。"""
    values = list(colour)[:3]
    if all(isinstance(v, float) for v in values) and max(values) <= 1.0:
        values = [round(v * 255) for v in values]
    return QColor(*[int(v) for v in values])


def tick_text(value: float) -> str:
    """刻度数字：三位有效数字，负号用真正的减号，零不写成 -0。"""
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.3g}".replace("-", "−")


class _Card(QWidget):
    """浮层底板：跟随视口明暗配色，不接鼠标，旋转缩放照常穿透到视口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.active = False
        self._light = False
        self.hide()

    def set_light(self, light: bool) -> None:
        self._light = bool(light)
        self.update()

    def _palette(self) -> dict:
        return theme.viewport_card(self._light)

    def _paint_card(self, painter: QPainter) -> None:
        colours = self._palette()
        painter.setPen(QPen(QColor(*colours["edge"]), 1))
        painter.setBrush(QColor(*colours["surface"]))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)

    @staticmethod
    def _font(size: str, bold: bool = False) -> QFont:
        font = QFont()
        font.setPointSizeF(float(size.rstrip("pt")))
        font.setBold(bold)
        return font

    def reset(self) -> None:
        self.active = False
        self.hide()


class ColorScale(_Card):
    """竖直色标：标题、色带（连续或分级）、刻度或分档说明。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = ""
        self.colors: list[QColor] = []
        self.clim = (0.0, 1.0)
        self.banded = False
        self.annotations: list[str] = []
        self.ticks: list[float] = []
        self.setFixedWidth(WIDTH)

    def set_scale(self, title: str, colors, clim, *, banded: bool,
                  ticks=None, annotations=None) -> None:
        """``colors`` 从低到高排列；分级时每个颜色是一档。"""
        self.title = title
        self.colors = [_rgb(c) for c in colors]
        low, high = float(clim[0]), float(clim[1])
        self.clim = (low, high) if high > low else (low, low + 1.0)
        self.banded = bool(banded)
        self.annotations = list(annotations or [])
        if ticks is not None:
            self.ticks = [float(t) for t in ticks]
        elif self.banded:
            n = len(self.colors)
            self.ticks = [low + (high - low) * i / n for i in range(n + 1)]
        else:
            self.ticks = [low + (high - low) * i / 6 for i in range(7)]
        self.active = True
        self.show()
        self.raise_()
        self.update()

    def preferred_height(self) -> int:
        """色带不必拉满整个视口：十几级的色标每级 22 像素足够读。"""
        metrics = QFontMetrics(self._font(theme.FONT_CAPTION))
        steps = max(len(self.colors) if self.banded else 8, 5)
        return PAD * 3 + self._title_height() + steps * max(22, metrics.height() + 6)

    def _title_height(self) -> int:
        return QFontMetrics(self._font(theme.FONT_BODY, bold=True)).height()

    def bar_rect(self) -> QRectF:
        top = PAD + self._title_height() + PAD + 4
        return QRectF(PAD, top, BAR_WIDTH, max(20, self.height() - top - PAD - 4))

    def paintEvent(self, _event) -> None:
        if not self.colors:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._paint_card(painter)
            colours = self._palette()
            ink, muted = QColor(colours["ink"]), QColor(colours["muted"])
            painter.setFont(self._font(theme.FONT_BODY, bold=True))
            painter.setPen(ink)
            title_rect = QRectF(PAD, PAD, self.width() - 2 * PAD, self._title_height())
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             painter.fontMetrics().elidedText(
                                 self.title, Qt.TextElideMode.ElideRight, int(title_rect.width())))
            bar = self.bar_rect()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            if self.banded:
                n = len(self.colors)
                for i, colour in enumerate(self.colors):
                    top = bar.bottom() - bar.height() * (i + 1) / n
                    painter.fillRect(QRectF(bar.left(), top, bar.width(), bar.height() / n + 0.5), colour)
            else:
                gradient = QLinearGradient(bar.bottomLeft(), bar.topLeft())
                last = max(1, len(self.colors) - 1)
                step = max(1, len(self.colors) // 48)
                for i in range(0, len(self.colors), step):
                    gradient.setColorAt(i / last, self.colors[i])
                gradient.setColorAt(1.0, self.colors[-1])
                painter.fillRect(bar, gradient)
            painter.setPen(QPen(QColor(*colours["edge"]), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(bar)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setFont(self._font(theme.FONT_CAPTION))
            metrics = painter.fontMetrics()
            text_left = bar.right() + 6
            text_width = self.width() - text_left - PAD + 2
            if self.annotations:
                n = len(self.annotations)
                for i, text in enumerate(self.annotations):
                    y = bar.bottom() - bar.height() * (i + 0.5) / n
                    painter.setPen(ink)
                    painter.drawText(QRectF(text_left, y - metrics.height() / 2, text_width, metrics.height()),
                                     Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
                return
            low, high = self.clim
            # 刻度太密时隔一个标一个，宁可少标也不叠字
            spacing = bar.height() / max(1, len(self.ticks) - 1)
            stride = max(1, math.ceil((metrics.height() + 2) / max(spacing, 1e-6)))
            last_index = len(self.ticks) - 1
            shown = list(range(0, last_index, stride))
            if shown and last_index - shown[-1] < stride and len(shown) > 1:
                shown.pop()                         # 两端必标，挤的是中间那个
            for i in [*shown, last_index]:
                value = self.ticks[i]
                y = bar.bottom() - bar.height() * (value - low) / (high - low)
                painter.setPen(QPen(muted, 1))
                painter.drawLine(QPointF(bar.right(), y), QPointF(bar.right() + 3, y))
                painter.setPen(ink)
                painter.drawText(QRectF(text_left, y - metrics.height() / 2, text_width, metrics.height()),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 tick_text(value))
        finally:
            painter.end()


class SymbolKey(_Card):
    """模型视图的符号图例：杆件、支座、荷载各是什么颜色。

    支座和荷载靠形状区分，但形状是要学的；没有图例，用户只能猜那个小锥子
    是铰接还是滚动。**符号系统必须自带说明**，否则它只是好看，不是可读。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: list[tuple[str, QColor]] = []
        self.setFixedWidth(WIDTH)

    def set_entries(self, entries) -> None:
        self.entries = [(str(label), QColor(colour)) for label, colour in entries]
        self.active = bool(self.entries)
        self.setVisible(self.active)
        if self.active:
            self.setFixedHeight(self.preferred_height())
            self.raise_()
        self.update()

    def _row(self) -> int:
        return max(SWATCH, QFontMetrics(self._font(theme.FONT_CAPTION)).height()) + 6

    def preferred_height(self) -> int:
        return PAD * 2 + self._row() * len(self.entries)

    def paintEvent(self, _event) -> None:
        if not self.entries:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._paint_card(painter)
            colours = self._palette()
            painter.setFont(self._font(theme.FONT_CAPTION))
            metrics = painter.fontMetrics()
            row = self._row()
            for i, (label, colour) in enumerate(self.entries):
                top = PAD + i * row
                swatch = QRectF(PAD, top + (row - SWATCH) / 2, SWATCH, SWATCH)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(colour)
                painter.drawRoundedRect(swatch, 2, 2)
                painter.setPen(QColor(colours["ink"]))
                text = QRectF(swatch.right() + 6, top, self.width() - swatch.right() - 6 - PAD, row)
                painter.drawText(text, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(text.width())))
        finally:
            painter.end()


def lookup_colors(actor) -> list[tuple]:
    """从云图映射器的查找表读出颜色，低到高排列。"""
    mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else getattr(actor, "mapper", None)
    table = mapper.GetLookupTable() if mapper is not None else None
    if table is None or not hasattr(table, "GetNumberOfTableValues"):
        return []
    count = int(table.GetNumberOfTableValues())
    return [tuple(table.GetTableValue(i)[:3]) for i in range(count)]
