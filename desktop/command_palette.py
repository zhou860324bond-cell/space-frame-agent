"""命令面板（Ctrl+Shift+P）：按名字找到任意命令，找不到就把这句话交给 AI 助手。

一百多个命令分散在十一个模块页里，知道要做什么却不知道按钮在哪一页，是
"界面不贴合人的操作"里最常见的一种。命令面板给的是第二条路：直接说名字。
它不新增任何功能，列表就是主窗口的 ``actions_by_name``——和功能区、右键菜单
用同一批 QAction，所以灰掉的命令在这里同样灰着，不会绕过前置条件。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QVBoxLayout)

from . import i18n

_ASK_AI = "__ask_ai__"


def _matches(query: str, *texts: str) -> bool:
    """每个空格分开的词都要出现（不分大小写），中文按子串匹配。"""
    haystack = " ".join(t for t in texts if t).lower()
    return all(word in haystack for word in query.lower().split())


class CommandPalette(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("命令面板")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setObjectName("commandPalette")
        self.resize(520, 420)
        box = QVBoxLayout(self)
        box.setContentsMargins(10, 10, 10, 10)
        box.setSpacing(6)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("输入命令名，例如：截面指派、支座沉降、求解……")
        self.search.textChanged.connect(self._fill)
        self.search.returnPressed.connect(self._run_current)
        box.addWidget(self.search)
        self.items = QListWidget(self)
        self.items.itemActivated.connect(self._run)
        box.addWidget(self.items, 1)
        hint = QLabel("Enter 执行 · ↑↓ 选择 · Esc 关闭；灰色的命令当前不可用", self)
        hint.setProperty("panel", "hint")
        box.addWidget(hint)
        self._fill("")
        # 视口坐标系下居中靠上，像主流编辑器的命令面板
        geo = window.geometry()
        self.move(geo.center().x() - self.width() // 2, geo.top() + 90)
        self.search.setFocus()

    def _commands(self):
        from .commands import COMMANDS
        actions = self.window.actions_by_name
        for command in COMMANDS:
            action = actions.get(command.name)
            if action is not None:
                yield command, action

    def _fill(self, text: str) -> None:
        self.items.clear()
        query = text.strip()
        for command, action in self._commands():
            label = command.label
            if query and not _matches(query, label, i18n.ZH_EN.get(label, ""), command.tip,
                                      command.name):
                continue
            shortcut = f"　{command.shortcut}" if command.shortcut else ""
            item = QListWidgetItem(f"{label}{shortcut}")
            item.setToolTip(command.tip)
            item.setData(Qt.ItemDataRole.UserRole, command.name)
            if not action.isEnabled():
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.items.addItem(item)
        if query:
            ask = QListWidgetItem(f"问 AI 助手：{query}")
            ask.setData(Qt.ItemDataRole.UserRole, _ASK_AI)
            ask.setToolTip("没有找到想要的命令时，把这句话发给 AI 助手")
            self.items.addItem(ask)
        for row in range(self.items.count()):
            if self.items.item(row).flags() & Qt.ItemFlag.ItemIsEnabled:
                self.items.setCurrentRow(row)
                break

    def keyPressEvent(self, event):  # noqa: N802  Qt 的命名
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            self.items.keyPressEvent(event)
            return
        super().keyPressEvent(event)

    def _run_current(self) -> None:
        item = self.items.currentItem()
        if item is not None:
            self._run(item)

    def _run(self, item: QListWidgetItem) -> None:
        if not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
        if name == _ASK_AI:
            if not self.window.chat_dock.isVisible():
                self.window.toggle_agent_panel()
            self.window.chat.send(self.search.text().strip())
            return
        self.window.actions_by_name[name].trigger()
