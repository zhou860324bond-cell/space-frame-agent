"""固定并排的对话与三维视口，提供与其他面板一致的显隐入口。"""
from PySide6.QtCore import QObject, Signal


class WorkspaceHandle(QObject):
    visibilityChanged = Signal(bool)

    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self.widget = widget
        self._visible = True

    def windowTitle(self):
        return "AI 助手"

    def setWindowTitle(self, title):
        self.widget.setWindowTitle(title)

    def isVisible(self):
        return self._visible

    def isHidden(self):
        return not self._visible

    def isVisibleTo(self, _parent=None):
        return self._visible

    def setVisible(self, visible):
        self._visible = bool(visible)
        self.widget.setVisible(visible)
        self.visibilityChanged.emit(self._visible)

    def show(self):
        self.setVisible(True)

    def hide(self):
        self.setVisible(False)

    def raise_(self):
        self.show()
        self.widget.setFocus()
