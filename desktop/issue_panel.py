"""Problem-driven review panel for multimodal v2 drafts."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QVBoxLayout, QWidget)

from . import glyphs


_PRIORITY = {
    "work_plane_unconfirmed": 0, "scale_unknown": 1, "scale_conflict": 2,
    "perspective": 3, "topology": 4, "intersection_unknown": 5,
    "low_confidence": 6, "load_incomplete": 7, "merge_collision": 8,
}
_CONFIRMABLE = {"work_plane_unconfirmed", "perspective", "low_confidence"}


def confidence_band(entity: dict) -> str:
    """识别置信度只描述图像提取质量，不代表结构可靠性。"""
    if entity.get("verified") or entity.get("source") == "user":
        return "verified"
    value = entity.get("confidence")
    if not isinstance(value, (int, float)):
        return "unknown"
    if float(value) >= 0.90:
        return "high"
    if float(value) >= 0.75:
        return "medium"
    return "low"


class IssuePanel(QWidget):
    issue_selected = Signal(str, list)
    resolution_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._issues: dict[str, dict] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel("阻断 0 · 警告 0")
        confidence_row = QHBoxLayout()
        self.confidence_summary = QLabel("低置信度 0")
        self.btn_next_low = QPushButton("审核下一个")
        self.btn_next_low.clicked.connect(self._next_low_confidence)
        confidence_row.addWidget(self.confidence_summary)
        confidence_row.addStretch(1)
        confidence_row.addWidget(self.btn_next_low)
        self.list = QListWidget()
        self.message = QLabel("没有待处理问题")
        self.message.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addLayout(confidence_row)
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
        entity_confidence = {}
        for entity in draft.get("entities") or []:
            target = entity.get("target") or {}
            if "node" in target:
                ref = f"node:{target['node']}"
            elif "member" in target:
                ref = f"member:{target['member']}"
            else:
                continue
            entity_confidence[ref] = entity.get("confidence")
        blockers = sum(item.get("severity") == "blocking" for item in issues)
        self.summary.setText(f"阻断 {blockers} · 警告 {len(issues) - blockers}")
        low_count = sum(item.get("category") == "low_confidence" for item in issues)
        self.confidence_summary.setText(f"低置信度待审核 {low_count}")
        self.btn_next_low.setEnabled(low_count > 0)
        self.list.clear()
        for issue in issues:
            marker = (glyphs.BLOCK if issue.get("severity") == "blocking" else glyphs.WARN)
            badge = ""
            if issue.get("category") == "low_confidence":
                values = [entity_confidence.get(str(ref))
                          for ref in issue.get("entity_refs") or []]
                confidence = next((value for value in values
                                   if isinstance(value, (int, float))), None)
                badge = (f"[低 {float(confidence):.0%}] "
                         if confidence is not None else "[低/未知] ")
            item = QListWidgetItem(
                f"{marker} {badge}{issue.get('message', issue['id'])}")
            item.setData(256, str(issue["id"]))
            self.list.addItem(item)
        target = next((row for row in range(self.list.count())
                       if self.list.item(row).data(256) == current), 0)
        if self.list.count():
            self.list.setCurrentRow(target)
        else:
            self.message.setText("没有待处理问题")
            self._set_actions(None)

    def _next_low_confidence(self) -> None:
        rows = [row for row in range(self.list.count())
                if self._issues.get(str(self.list.item(row).data(256)), {}).get(
                    "category") == "low_confidence"]
        if not rows:
            return
        current = self.list.currentRow()
        target = next((row for row in rows if row > current), rows[0])
        self.list.setCurrentRow(target)

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
