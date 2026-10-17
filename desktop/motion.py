"""界面动效：流光状态文字、用时格式与「减少动效」开关。

后台在干活时，界面上原来只有一行静止的「思考中…」或状态栏里一闪而过的
「求解中…」。几十秒没有变化，「还在跑」和「已经卡死」看起来一模一样。

流光的做法借鉴 OpenAI Codex CLI 终端界面的状态行（codex-rs/tui/src/shimmer.rs，
Apache-2.0）：一条亮带按固定周期从左扫到右，带内亮度按余弦平滑衰减。这里只按
它公开的算法在 Qt 里重写，没有复制它的代码。参数沿用它的取值：周期 2 秒、
两侧各留 10 个字符的空白、亮带半宽 5 个字符、约 30 帧/秒重画。

「减少动效」打开后不再逐帧重画：流光变成静止文字，转圈图标停住，用时仍按秒
刷新——远程桌面和对动画敏感的用户需要这一项。
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy

from . import theme

#: 流光扫过一遍的时间（秒）
SWEEP_SECONDS = 2.0
#: 文字两侧的空白（字符数）：亮带从文字外面进来、扫出去，起止不突兀
SWEEP_PADDING = 10
#: 亮带半宽（字符数）
BAND_HALF_WIDTH = 5.0
#: 亮度向高亮色混合的上限：留一成底色，最亮处也不发白
BLEND_LIMIT = 0.9
#: 重画间隔（毫秒），约 30 帧/秒
FRAME_MS = 32


def format_elapsed(seconds: float) -> str:
    """把已用时间写成「12 秒」「1 分 05 秒」「2 小时 03 分 09 秒」。

    原来一律写「125 s」：过了一分钟就得在心里除以 60，也看不出是秒还是分。
    """
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} 小时 {minutes:02d} 分 {secs:02d} 秒"
    if minutes:
        return f"{minutes} 分 {secs:02d} 秒"
    return f"{secs} 秒"


def shimmer_levels(count: int, elapsed: float) -> list[float]:
    """第 i 个字符此刻的亮度（0~1）。

    亮带中心按 ``elapsed`` 在「字符数 + 两侧空白」的范围里匀速移动；与中心
    相距 d 个字符时亮度为 ``(1 + cos(πd / 半宽)) / 2``，出了亮带为 0。
    """
    if count <= 0:
        return []
    span = count + 2 * SWEEP_PADDING
    center = int((elapsed % SWEEP_SECONDS) / SWEEP_SECONDS * span)
    levels = []
    for i in range(count):
        distance = abs(i + SWEEP_PADDING - center)
        if distance <= BAND_HALF_WIDTH:
            levels.append(0.5 * (1.0 + math.cos(math.pi * distance / BAND_HALF_WIDTH)))
        else:
            levels.append(0.0)
    return levels


def blend(base: QColor, highlight: QColor, level: float) -> QColor:
    """按亮度把底色混向高亮色。"""
    t = max(0.0, min(1.0, level)) * BLEND_LIMIT
    return QColor(round(base.red() + (highlight.red() - base.red()) * t),
                  round(base.green() + (highlight.green() - base.green()) * t),
                  round(base.blue() + (highlight.blue() - base.blue()) * t))


class _Motion(QObject):
    """全局的「减少动效」状态；切换时通知所有在动的小部件。"""

    changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.reduced = False


_motion: _Motion | None = None


def motion() -> _Motion:
    global _motion
    if _motion is None:
        _motion = _Motion()
    return _motion


def reduced() -> bool:
    return motion().reduced


def set_reduced(on: bool) -> None:
    state = motion()
    if state.reduced == bool(on):
        return
    state.reduced = bool(on)
    state.changed.emit(state.reduced)


class ShimmerLabel(QLabel):
    """单行状态文字：标题带流光，后面的说明（用时、怎么停止）用暗色静止显示。

    放不下时在末尾省略，完整文字放进悬停提示。只在可见、有标题、没有减少
    动效时逐帧重画；否则计时器停着，不白占主线程。继承 QLabel 是为了沿用
    样式表里的字号（如 ``chat="caption"``），文字仍由 paintEvent 自己画。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._header = ""
        self._detail = ""
        self._started = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self.update)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        motion().changed.connect(self._sync_timer)

    def header(self) -> str:
        return self._header

    def detail(self) -> str:
        return self._detail

    def set_text(self, header: str, detail: str = "") -> None:
        if header != self._header:
            # 标题换了就从头扫：新一步开始了，看得出来
            self._started = time.monotonic()
        self._header, self._detail = header, detail
        full = f"{header}　{detail}" if header and detail else header
        self.setText(full)
        self.setToolTip(full)
        self._sync_timer()
        self.update()

    def clear(self) -> None:
        self.set_text("", "")

    def animating(self) -> bool:
        return self._timer.isActive()

    def minimumSizeHint(self):  # noqa: N802
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def _sync_timer(self, *_args) -> None:
        if self.isVisible() and self._header and not reduced():
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
            self.update()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event):  # noqa: N802
        super().hideEvent(event)
        self._timer.stop()

    def paintEvent(self, _event):  # noqa: N802
        text = self.text()
        if not text:
            return
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        g.setFont(self.font())
        metrics = QFontMetricsF(self.font())
        shown = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(0.0, self.width() - 2.0))
        lit = min(len(self._header), len(shown))
        rect = QRectF(self.rect())
        baseline = (rect.height() + metrics.ascent() - metrics.descent()) / 2
        dim = QColor(theme.INK_DIM)
        if reduced():
            levels = [0.0] * lit
            base = QColor(theme.INK_MUTED)
        else:
            levels = shimmer_levels(len(self._header), time.monotonic() - self._started)[:lit]
            base = dim
        highlight = QColor(theme.INK)
        x = 1.0
        for i, char in enumerate(shown):
            g.setPen(blend(base, highlight, levels[i]) if i < lit else dim)
            g.drawText(x, baseline, char)
            x += metrics.horizontalAdvance(char)
        g.end()
