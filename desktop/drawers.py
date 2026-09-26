"""抽屉：贴在视口边缘、叠在视口上方的面板。

为什么不是停靠，也不是自由浮窗
------------------------------
* **停靠**每开一个面板就把视口挤小一次，看结果时尤其明显——表格一出来，
  模型只剩半屏。
* **自由浮窗**（上一版的做法）不挤视口，但九个小窗各飘各的：会压在功能区
  和快捷栏上面把按钮挡住（实拍里模型树正好盖住了「侧视/顶视」），拖走了
  找不回来，主窗口一挪它们还留在原地。

抽屉取两者之长：**位置由视口决定**（永远在视口矩形之内，碰不到工具栏），
**尺寸不影响视口**（叠在上方，关掉即还原），并且跟着主窗口移动、最小化。
同一侧只有一个抽屉，多个面板在抽屉里用页签切换，而不是摞成一堆窗。

为什么是顶层 Tool 窗而不是视口的子控件
------------------------------------
三维视口是 VTK 的原生 OpenGL 窗口。在 Windows 上，普通 Qt 子控件**画不到
原生窗口上面**——上一版右缘那颗 AI 召唤按钮就是这么"消失"的。只有顶层窗
能盖住它，所以抽屉是无边框的 Tool 窗（归主窗口所有，不占任务栏），由
`DrawerHost` 负责把它钉在视口的对应边上。

入口在视口**外面**的两条窄栏（`SideRail`）上：它们是主窗口布局里的普通
控件，宽度固定，开关抽屉时视口尺寸一个像素都不变。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QStackedWidget,
                               QTabBar, QToolButton, QVBoxLayout, QWidget)

from . import theme

LEFT, RIGHT, BOTTOM = "left", "right", "bottom"

# 抽屉占视口的比例上限。**上限比默认值更要紧**：拖大到盖住整个视口的
# 抽屉，等于回到了"看结果就看不见模型"。
_MAX_SIDE_RATIO = 0.45
_MAX_BOTTOM_RATIO = 0.5
_MIN_SIDE = 220
_MIN_BOTTOM = 140
GRIP = 6


@dataclass
class _Page:
    key: str
    title: str
    widget: QWidget
    handle: "PanelHandle"


class PanelHandle(QObject):
    """一个面板在抽屉里的"把手"。

    保留旧 QDockWidget 被用到的那几个方法（setVisible/isVisible/raise_/
    visibilityChanged/windowTitle），主窗口各处的调用就不用逐个改写。

    **可见性是逻辑状态**，不是 Qt 窗口状态：主窗口最小化时抽屉会被系统
    藏起来，但面板在用户看来仍然"开着"，还原后要原样回来。
    """

    visibilityChanged = Signal(bool)

    def __init__(self, drawer: "Drawer", key: str, title: str):
        super().__init__(drawer)
        self._drawer = drawer
        self.key = key
        self._title = title

    def windowTitle(self) -> str:
        return self._title

    def isVisible(self) -> bool:
        return self._drawer.is_open() and self._drawer.current_key() == self.key

    def isVisibleTo(self, _ancestor=None) -> bool:
        return self.isVisible()

    def isHidden(self) -> bool:
        return not self.isVisible()

    def setVisible(self, on: bool) -> None:
        if on:
            self._drawer.open_page(self.key)
        elif self.isVisible():
            self._drawer.close()

    def show(self) -> None:
        self.setVisible(True)

    def hide(self) -> None:
        self.setVisible(False)

    def raise_(self) -> None:
        # 抽屉里只有"当前页"一说，没有叠放次序
        if self._drawer.is_open():
            self._drawer.open_page(self.key)


class PanelWindow(QDialog):
    """放在独立窗口里的面板（非模态，归主窗口所有）。

    给"做一件完整的事、需要宽画布"的面板用——手绘草图识别就是：选图、
    核对覆盖标注、标定尺度、确认加载，一次走完就关。塞进抽屉页签，要么
    抽屉被它撑宽，要么画布被压窄；和常驻参考的面板挤在一起也说不通。

    对外接口与 PanelHandle 一致（setVisible/isVisible/raise_/
    visibilityChanged/windowTitle），主窗口里的调用不用分两套。
    """

    visibilityChanged = Signal(bool)

    def __init__(self, title: str, widget: QWidget, parent: QWidget,
                 size: tuple[int, int] = (560, 720)):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        area = QScrollArea(self)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(widget)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(area)
        self.resize(*size)

    def showEvent(self, event):                   # noqa: N802  Qt 回调名
        super().showEvent(event)
        self.visibilityChanged.emit(True)

    def hideEvent(self, event):                   # noqa: N802
        super().hideEvent(event)
        self.visibilityChanged.emit(False)


class _EdgeGrip(QWidget):
    """抽屉内侧那条可拖动的边。无边框窗没有系统的缩放把手，得自己做。"""

    def __init__(self, drawer: "Drawer"):
        super().__init__(drawer)
        self._drawer = drawer
        self._origin: QPoint | None = None
        self._start = 0
        horizontal = drawer.side == BOTTOM
        self.setCursor(Qt.CursorShape.SizeVerCursor if horizontal
                       else Qt.CursorShape.SizeHorCursor)
        if horizontal:
            self.setFixedHeight(GRIP)
        else:
            self.setFixedWidth(GRIP)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.globalPosition().toPoint()
            self._start = self._drawer.extent

    def mouseMoveEvent(self, event):
        if self._origin is None:
            return
        delta = event.globalPosition().toPoint() - self._origin
        side = self._drawer.side
        change = {LEFT: delta.x(), RIGHT: -delta.x(), BOTTOM: -delta.y()}[side]
        self._drawer.set_extent(self._start + change)

    def mouseReleaseEvent(self, _event):
        self._origin = None

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setPen(QPen(QColor(theme.BORDER), 1))
        if self._drawer.side == BOTTOM:
            p.drawLine(0, 0, self.width(), 0)
        elif self._drawer.side == LEFT:
            p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        else:
            p.drawLine(0, 0, 0, self.height())
        p.end()


class Drawer(QFrame):
    """一侧的抽屉。页签切换多个面板，右上角一个关闭。"""

    opened_changed = Signal(bool)
    page_changed = Signal(str)

    def __init__(self, host: "DrawerHost", side: str, extent: int):
        super().__init__(host.window,
                         Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("drawer")
        self.setProperty("side", side)
        self.host = host
        self.side = side
        self.extent = extent
        self._open = False
        self._pages: list[_Page] = []

        self.tabs = QTabBar(self)
        self.tabs.setObjectName("drawerTabs")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        self.tabs.currentChanged.connect(self._on_tab)
        self.title = QLabel(self)
        self.title.setObjectName("drawerTitle")
        self.close_button = QToolButton(self)
        self.close_button.setObjectName("drawerClose")
        self.close_button.setText("×")
        self.close_button.setToolTip("收起（Esc）")
        self.close_button.setAutoRaise(True)
        self.close_button.clicked.connect(self.close)
        self.stack = QStackedWidget(self)

        header = QHBoxLayout()
        header.setContentsMargins(10, 4, 4, 0)
        header.setSpacing(6)
        header.addWidget(self.title)
        header.addWidget(self.tabs, 1)
        header.addWidget(self.close_button)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)
        body.addLayout(header)
        body.addWidget(self.stack, 1)

        # 焦点在抽屉里（比如正在 AI 输入框里打字）时，主窗口的 Esc 快捷键
        # 收不到——抽屉是另一个顶层窗。这里再挂一个，Esc 在哪都能收起。
        escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        escape.setContext(Qt.ShortcutContext.WindowShortcut)
        escape.activated.connect(self._escape)

        grip = _EdgeGrip(self)
        outer = QVBoxLayout(self) if side == BOTTOM else QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        if side == RIGHT or side == BOTTOM:
            outer.addWidget(grip)
            outer.addLayout(body, 1)
        else:
            outer.addLayout(body, 1)
            outer.addWidget(grip)

    # --- 页 ---

    def add_page(self, key: str, title: str, widget: QWidget) -> PanelHandle:
        handle = PanelHandle(self, key, title)
        self._pages.append(_Page(key, title, widget, handle))
        # **每页都包一层滚动区。** QStackedWidget 的最小尺寸取所有页里最大的
        # 那个，而截面优化页要 1200×390——不包的话，Qt 会无视我们给的位置，
        # 把整个底部抽屉撑到盖住右抽屉、越出视口下沿（实拍如此）。
        # 包了之后抽屉严格待在分给它的矩形里，内容放不下就在里面滚。
        if not isinstance(widget, QScrollArea):
            area = QScrollArea(self)
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            area.setWidget(widget)
            widget = area
        self.stack.addWidget(widget)
        self.tabs.blockSignals(True)
        self.tabs.addTab(title)
        self.tabs.blockSignals(False)
        self._sync_header()
        return handle

    def keys(self) -> list[str]:
        return [p.key for p in self._pages]

    def page(self, key: str) -> _Page:
        for p in self._pages:
            if p.key == key:
                return p
        raise KeyError(key)

    def current_key(self) -> str | None:
        index = self.stack.currentIndex()
        return self._pages[index].key if 0 <= index < len(self._pages) else None

    def _sync_header(self) -> None:
        """抽屉顶上只写当前页的标题，不放页签。

        切页由两侧窄栏负责——顶上再放一排页签是重复，而且左抽屉四页在
        290 px 里放不下，「建模过程」被挤到滚动箭头后面看不见（实拍如此）。
        页签控件留着但不显示：它仍是"第几页"的记录者，语言切换也照常翻它。
        """
        from .i18n import tr

        self.tabs.setVisible(False)
        self.title.setVisible(True)
        key = self.current_key()
        page = next((p for p in self._pages if p.key == key), None)
        if page is None and self._pages:
            page = self._pages[0]
        if page is not None:
            self.title.setText(tr(page.title))

    def _on_tab(self, index: int) -> None:
        if not 0 <= index < len(self._pages):
            return
        before = self.current_key()
        self.stack.setCurrentIndex(index)
        after = self._pages[index].key
        if before != after:
            self._emit_page_visibility(before, after)
            self._sync_header()
            self.page_changed.emit(after)

    def _emit_page_visibility(self, before: str | None, after: str | None) -> None:
        if not self._open:
            return
        for p in self._pages:
            if p.key == before:
                p.handle.visibilityChanged.emit(False)
            if p.key == after:
                p.handle.visibilityChanged.emit(True)

    # --- 开合 ---

    def is_open(self) -> bool:
        return self._open

    def open_page(self, key: str) -> None:
        index = self.keys().index(key)
        before = self.current_key() if self._open else None
        if self.tabs.currentIndex() != index:
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(index)
            self.tabs.blockSignals(False)
        self.stack.setCurrentIndex(index)
        was_open = self._open
        self._open = True
        self.host.relayout()
        if not was_open:
            self.opened_changed.emit(True)
            self.page(key).handle.visibilityChanged.emit(True)
        elif before != key:
            self._emit_page_visibility(before, key)
        self._sync_header()
        if before != key:
            self.page_changed.emit(key)

    def close(self) -> bool:                       # noqa: A003  与 QWidget.close 同名是刻意的
        if not self._open:
            return True
        current = self.current_key()
        self._open = False
        super().hide()
        self.host.relayout()
        self.opened_changed.emit(False)
        if current is not None:
            self.page(current).handle.visibilityChanged.emit(False)
        return True

    def toggle_page(self, key: str) -> None:
        if self._open and self.current_key() == key:
            self.close()
        else:
            self.open_page(key)

    # --- 尺寸 ---

    def set_extent(self, value: int) -> None:
        self.extent = self._clamp(int(value))
        self.host.relayout()

    def _clamp(self, value: int) -> int:
        area = self.host.viewport_rect()
        if self.side == BOTTOM:
            top = max(_MIN_BOTTOM, int(area.height() * _MAX_BOTTOM_RATIO))
            return max(_MIN_BOTTOM, min(value, top))
        top = max(_MIN_SIDE, int(area.width() * _MAX_SIDE_RATIO))
        return max(_MIN_SIDE, min(value, top))

    def closeEvent(self, event):                  # noqa: N802  Qt 回调名
        """Alt+F4 之类走的是 Qt 底层的关闭，绕过上面的 close()：窗是藏了，
        抽屉却仍记着"开着"——窄栏按钮还亮着，主窗一挪它又冒出来。
        统一收口到 close()。"""
        event.ignore()
        self.close()

    def _escape(self) -> None:
        self.close()
        self.host.window.activateWindow()


class DrawerHost(QObject):
    """管三个抽屉的位置：左、右贴视口两侧全高，底部夹在两者之间。"""

    def __init__(self, window: QWidget, viewport: QWidget):
        super().__init__(window)
        self.window = window
        self.viewport = viewport
        self.drawers: dict[str, Drawer] = {}
        # 抽屉开合后通知视口：叠在视口上的小部件要让开被盖住的那几条边
        self.on_insets = None
        window.installEventFilter(self)
        viewport.installEventFilter(self)

    def drawer(self, side: str, extent: int) -> Drawer:
        if side not in self.drawers:
            d = Drawer(self, side, extent)
            self.drawers[side] = d
        return self.drawers[side]

    def viewport_rect(self) -> QRect:
        """视口在屏幕上的矩形。抽屉永远待在它里面。"""
        top_left = self.viewport.mapToGlobal(QPoint(0, 0))
        return QRect(top_left, self.viewport.size())

    def geometry_for(self, side: str) -> QRect:
        area = self.viewport_rect()
        d = self.drawers[side]
        extent = d._clamp(d.extent)
        if side == LEFT:
            return QRect(area.left(), area.top(), extent, area.height())
        if side == RIGHT:
            return QRect(area.right() - extent + 1, area.top(), extent, area.height())
        # 底部抽屉让开左右两个已打开的抽屉，四个角不会叠在一起
        left = area.left()
        right = area.right()
        if LEFT in self.drawers and self.drawers[LEFT].is_open():
            left = self.geometry_for(LEFT).right() + 1
        if RIGHT in self.drawers and self.drawers[RIGHT].is_open():
            right = self.geometry_for(RIGHT).left() - 1
        return QRect(left, area.bottom() - extent + 1,
                     max(0, right - left + 1), extent)

    def free_rect(self) -> QRect:
        """视口里没被抽屉盖住的那一块（屏幕坐标）。"""
        area = self.viewport_rect()
        left, right, bottom = area.left(), area.right(), area.bottom()
        if LEFT in self.drawers and self.drawers[LEFT].is_open():
            left = self.geometry_for(LEFT).right() + 1
        if RIGHT in self.drawers and self.drawers[RIGHT].is_open():
            right = self.geometry_for(RIGHT).left() - 1
        if BOTTOM in self.drawers and self.drawers[BOTTOM].is_open():
            bottom = self.geometry_for(BOTTOM).top() - 1
        return QRect(QPoint(left, area.top()), QPoint(right, bottom))

    def insets(self) -> tuple[int, int, int]:
        """(左, 右, 下) 各被抽屉盖住多少像素（视口坐标，未打开为 0）。"""
        area = self.viewport_rect()
        free = self.free_rect()
        return (free.left() - area.left(), area.right() - free.right(),
                area.bottom() - free.bottom())

    def relayout(self) -> None:
        shown = self.window.isVisible() and not self.window.isMinimized()
        for side in (LEFT, RIGHT, BOTTOM):
            d = self.drawers.get(side)
            if d is None:
                continue
            if d.is_open() and shown:
                d.setGeometry(self.geometry_for(side))
                if not d.isVisible():
                    d.show()
                d.raise_()
            elif d.isVisible():
                QFrame.hide(d)
        if self.on_insets is not None:
            self.on_insets(*self.insets())

    def close_all(self) -> bool:
        """Esc 用：收起最后一个打开的抽屉。返回是否真收了东西。"""
        for side in (BOTTOM, RIGHT, LEFT):
            d = self.drawers.get(side)
            if d is not None and d.is_open():
                d.close()
                return True
        return False

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show,
                    QEvent.Type.Hide, QEvent.Type.WindowStateChange):
            self.relayout()
        return False


class SideRail(QFrame):
    """视口外侧的一条窄栏，放抽屉的开关。

    宽度固定——这条栏本身永远不变宽，所以开关抽屉不会让视口跳动。
    """

    WIDTH = 46

    def __init__(self, side: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sideRail")
        self.setProperty("side", side)
        self.setFixedWidth(self.WIDTH)
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(3, 6, 3, 6)
        self._box.setSpacing(4)
        self._tail = QVBoxLayout()
        self._tail.setSpacing(4)
        self._box.addStretch(1)
        self._box.addLayout(self._tail)
        self._count = 0
        self.buttons: dict[str, QToolButton] = {}

    def add_button(self, key: str, text: str, tip: str, *, primary: bool = False,
                   at_end: bool = False) -> QToolButton:
        b = QToolButton(self)
        b.setObjectName("railButton")
        b.setProperty("primary", primary)
        b.setText(text)
        b.setToolTip(tip)
        b.setCheckable(True)
        b.setAutoRaise(True)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFixedSize(QSize(self.WIDTH - 6, 56 if primary else 50))
        if at_end:
            self._tail.addWidget(b)
        else:
            self._box.insertWidget(self._count, b)
            self._count += 1
        self.buttons[key] = b
        return b

    def add_separator(self) -> None:
        line = QFrame(self)
        line.setObjectName("railSeparator")
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        self._box.insertWidget(self._count, line)
        self._count += 1

