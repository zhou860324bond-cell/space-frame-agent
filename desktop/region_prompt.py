"""Abaqus 式提示区：命令要一个区域时，提示选什么、选了几个，并给出完成 / 使用集合 / 取消。

Abaqus/CAE 的区域选择都走同一条路：点命令 → 提示区写「选择要施加边界条件的
区域」→ 在视口里点选、Shift 加选、Ctrl 减选、拖框，或者改用集合 → 点「完成」→
弹出编辑器。原来这里是先选中**一个**对象再点命令，选多个只能重复整套流程。

控制器只管流程：拾取本身在视口（viewport.start_region_pick），结果以区域字典
交给调用方——``{"nodes": [...]}`` / ``{"members": [...]}`` / ``{"set": 名字}``，
正是 session_features 接受的写法。
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QHBoxLayout, QInputDialog, QLabel, QPushButton, QWidget

from . import theme

_NOUN = {"node": "节点", "member": "杆件"}
_REGION_KEY = {"node": "nodes", "member": "members"}


class RegionPrompt(QWidget):
    """状态栏里的选择提示与按钮。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)
        self.count = QLabel(self)
        self.count.setStyleSheet(f"color:{theme.ACCENT}; font-weight:600;")
        self.done = QPushButton("完成", self)
        self.use_set = QPushButton("使用集合…", self)
        self.cancel = QPushButton("取消", self)
        for widget in (self.count, self.done, self.use_set, self.cancel):
            layout.addWidget(widget)
        self.hide()


class RegionPickController:
    """一次区域选择的生命周期：开始、计数、完成、改用集合、取消。"""

    def __init__(self, window):
        self.window = window
        self.prompt = RegionPrompt(window)
        window.statusBar().addPermanentWidget(self.prompt)
        self.prompt.done.clicked.connect(self.finish)
        self.prompt.use_set.clicked.connect(self.choose_set)
        self.prompt.cancel.clicked.connect(self.cancel)
        window.viewport.region_changed.connect(self._on_changed)
        self.kind: str | None = None
        self.label = ""
        self._callback: Callable[[dict], None] | None = None

    @property
    def active(self) -> bool:
        return self.kind is not None

    def start(self, kind: str, label: str, callback: Callable[[dict], None],
              initial: list[int] | None = None) -> None:
        """开始选择。``initial`` 是命令前已经选中的对象，作为默认区域带进来。"""
        if self.active:
            self.cancel()
        self.kind, self.label, self._callback = kind, label, callback
        self.window.set_prompt(f"{label}：选择{_NOUN[kind]}（点选；Shift 加选，Ctrl 减选；"
                               "拖框多选），或使用集合；选好后点「完成」，Esc 取消")
        self.prompt.show()
        self.window.viewport.start_region_pick(kind, initial)

    def _on_changed(self, kind: str, ids: list) -> None:
        if self.active:
            self.prompt.count.setText(f"已选 {len(ids)} 个{_NOUN[self.kind]}")
            self.prompt.done.setEnabled(bool(ids))

    def finish(self) -> None:
        if not self.active:
            return
        ids = list(self.window.viewport.region_selection.ids)
        if not ids:
            self.window.statusBar().showMessage(
                f"还没有选中{_NOUN[self.kind]}：请在视口中选择，或使用集合。", 5000)
            return
        self._deliver({_REGION_KEY[self.kind]: ids})

    def choose_set(self) -> None:
        if not self.active:
            return
        key = _REGION_KEY[self.kind]
        sets = self.window.session.model.get("sets") or {}
        names = sorted(name for name, entry in sets.items() if entry.get(key))
        if not names:
            self.window.statusBar().showMessage(
                f"还没有包含{_NOUN[self.kind]}的集合：请先在「命名集合」里创建，或直接在视口中选择。",
                6000)
            return
        name, ok = QInputDialog.getItem(self.window, "使用集合",
                                        f"选择{_NOUN[self.kind]}集合（之后集合变化，对象跟着变）：",
                                        names, 0, False)
        if ok and name:
            self._deliver({"set": name})

    def cancel(self) -> None:
        if not self.active:
            return
        self._reset()
        self.window.set_prompt("已取消选择")

    def _deliver(self, region: dict) -> None:
        callback = self._callback
        self._reset()
        if callback is not None:
            callback(region)

    def _reset(self) -> None:
        self.kind, self._callback = None, None
        self.prompt.hide()
        self.window.viewport.stop_region_pick()
