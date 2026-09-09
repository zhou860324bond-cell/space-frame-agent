"""Problem-driven review panel for multimodal v2 drafts."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QVBoxLayout, QWidget)


_PRIORITY = {
    "work_plane_unconfirmed": 0, "scale_unknown": 1, "scale_conflict": 2,
    "perspective": 3, "topology": 4, "intersection_unknown": 5,
    "low_confidence": 6, "load_incomplete": 7, "merge_collision": 8,
}
_CONFIRMABLE = {"work_plane_unconfirmed", "perspective", "low_confidence"}


class IssuePanel(QWidget):
    issue_selected = Signal(str, list)
    resolution_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._issues: dict[str, dict] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel("阻断 0 · 警告 0")
        self.list = QListWidget()
        self.message = QLabel("没有待处理问题")
        self.message.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addWidget(self.list)
        layout.addWidget(self.message)
        row = QHBoxLayout()
        self.btn_confirm = QPushButton("确认此项")
        self.btn_connect = QPushButton("连接")
        self.btn_cross = QPushButton("跨越")
        self.btn_reuse = QPushButton("复用现有节点")
        for button in (self.btn_confirm, self.btn_connect, self.btn_cross,
                       self.btn_reuse):
            row.addWidget(button)
        layout.addLayout(row)
        self.list.currentItemChanged.connect(self._selection_changed)
        self.btn_confirm.clicked.connect(lambda: self._request("confirm"))
        self.btn_connect.clicked.connect(lambda: self._request("connect"))
        self.btn_cross.clicked.connect(lambda: self._request("cross"))
        self.btn_reuse.clicked.connect(lambda: self._request("reuse"))
        self._set_actions(None)

    def set_draft(self, draft: dict) -> None:
        current = self.current_issue_id()
        issues = [item for item in draft.get("issues") or []
                  if item.get("status") == "open"]
        issues.sort(key=lambda item: (
            item.get("severity") != "blocking",
            _PRIORITY.get(str(item.get("category")), 99), str(item.get("id"))))
        self._issues = {str(item["id"]): item for item in issues}
        blockers = sum(item.get("severity") == "blocking" for item in issues)
        self.summary.setText(f"阻断 {blockers} · 警告 {len(issues) - blockers}")
        self.list.clear()
        for issue in issues:
            marker = "⛔" if issue.get("severity") == "blocking" else "⚠"
            item = QListWidgetItem(f"{marker} {issue.get('message', issue['id'])}")
            item.setData(256, str(issue["id"]))
            self.list.addItem(item)
        target = next((row for row in range(self.list.count())
                       if self.list.item(row).data(256) == current), 0)
        if self.list.count():
            self.list.setCurrentRow(target)
        else:
            self.message.setText("没有待处理问题")
            self._set_actions(None)

    def current_issue_id(self) -> str | None:
        item = self.list.currentItem()
        return None if item is None else str(item.data(256))

    def current_issue(self) -> dict | None:
        identifier = self.current_issue_id()
        return self._issues.get(identifier) if identifier else None

    def _selection_changed(self, current, previous) -> None:
        issue = self.current_issue()
        if issue is None:
            return
        self.message.setText(str(issue.get("message", "")))
        self._set_actions(str(issue.get("category")))
        self.issue_selected.emit(str(issue["id"]), list(issue.get("entity_refs") or []))

    def _set_actions(self, category: str | None) -> None:
        self.btn_confirm.setVisible(category in _CONFIRMABLE)
        intersection = category == "intersection_unknown"
        self.btn_connect.setVisible(intersection)
        self.btn_cross.setVisible(intersection)
        self.btn_reuse.setVisible(category == "merge_collision")

    def _request(self, action: str) -> None:
        identifier = self.current_issue_id()
        if identifier:
            self.resolution_requested.emit(identifier, action)
