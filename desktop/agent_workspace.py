"""停靠在视口两侧的固定面板（模型树、属性、AI 助手），提供与抽屉面板一致的显隐入口。

抽屉面板（drawers.PanelHandle）盖在视口上面；这里的面板是布局里的真实一栏，
打开时把视口挤窄而不是遮住一块。两者对外接口相同（show / hide / isVisible /
visibilityChanged），主窗口和测试不必区分面板放在哪里。
"""
from PySide6.QtCore import QObject, Signal


class WorkspaceHandle(QObject):
    visibilityChanged = Signal(bool)

    def __init__(self, widget, parent=None, title: str = "AI 助手", visible: bool = True):
        super().__init__(parent)
        self.widget = widget
        self._title = title
        self._visible = visible
        widget.setVisible(visible)

    def windowTitle(self):
        return self._title

    def setWindowTitle(self, title):
        self._title = title

    def isVisible(self):
        return self._visible

    def isHidden(self):
        return not self._visible

    def isVisibleTo(self, _parent=None):
        return self._visible

    def setVisible(self, visible):
        visible = bool(visible)
        changed = visible != self._visible
        self._visible = visible
        self.widget.setVisible(visible)
        if changed:
            self.visibilityChanged.emit(visible)

    def show(self):
        self.setVisible(True)

    def hide(self):
        self.setVisible(False)

    def raise_(self):
        self.show()
        self.widget.setFocus()


class PaneHandle(WorkspaceHandle):
    """停靠栏里的一格（例如模型栏下半部的属性）：显示它时连同所在的栏一起显示。"""

    def __init__(self, widget, container: WorkspaceHandle, parent=None, title: str = ""):
        super().__init__(widget, parent, title)
        self.container = container
        container.visibilityChanged.connect(lambda _on: self.visibilityChanged.emit(self.isVisible()))

    def isVisible(self):
        return self._visible and self.container.isVisible()

    def isHidden(self):
        return not self.isVisible()

    def isVisibleTo(self, _parent=None):
        return self.isVisible()

    def setVisible(self, visible):
        if visible:
            self.container.show()
        super().setVisible(visible)
