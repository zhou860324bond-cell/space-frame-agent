"""Problem-driven review panel for multimodal v2 drafts."""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QVBoxLayout, QWidget)

from . import glyphs


_PRIORITY = {
    "work_plane_unconfirmed": 0, "scale_unknown": 1, "scale_conflict": 2,
    "perspective": 3, "topology": 4, "pixel_topology": 4, "intersection_unknown": 5,
    "low_confidence": 6, "load_incomplete": 7, "merge_collision": 8,
}
_CONFIRMABLE = {"work_plane_unconfirmed", "perspective", "low_confidence"}


def low_confidence_group(issues: list[dict], identifier: str) -> list[dict]:
    """合并同对象提示，以及已被明确位置说明覆盖的界面置信度提示；原记录不删。"""
    current = next((item for item in issues if str(item.get("id")) == identifier), None)
    if current is None:
        return []
    refs = set(current.get("entity_refs") or [])
    if current.get("category") != "low_confidence" or not refs:
        return [current]
    return [item for item in issues if item.get("status") == "open"
            and item.get("category") == "low_confidence"
            and item.get("severity") == current.get("severity")
            and (item is current or (len(refs) == 1 and set(item.get("entity_refs") or []) == refs)
                 or (item.get("review_generated") is True and len(item.get("entity_refs") or []) == 1
                     and set(item["entity_refs"]) <= refs))]


def confidence_band(entity: dict) -> str:
    """识别置信度只描述图像提取质量，不代表结构可靠性。"""
    if entity.get("verified") or entity.get("source") == "user":
        return "verified"
    value = entity.get("confidence")
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or not 0 <= value <= 1):
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
        self.btn_confirm.clicked.connect(lambda: self._request(self._confirm_action))
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
        groups = []
        shown = set()
        for issue in issues:
            if str(issue["id"]) in shown:
                continue
            group = [i for i in low_confidence_group(issues, str(issue["id"])) if str(i["id"]) not in shown]
            groups.append(group)
            shown.update(str(item["id"]) for item in group)
        self._groups = {str(group[0]["id"]): group for group in groups}
        entity_confidence = {}
        for entity in draft.get("entities") or []:
            target = entity.get("target") or {}
            if "node" in target:
                ref = f"node:{target['node']}"
            elif "member" in target:
                ref = f"member:{target['member']}"
            elif "support" in target:
                ref = f"support:{target['support'].get('node')}:{target['support'].get('name')}"
            elif "load" in target:
                load = target["load"]
                ref = f"load:{load.get('case')}:{load.get('collection')}:{load.get('name')}"
            else:
                continue
            entity_confidence[ref] = entity.get("confidence")
        blockers = sum(item.get("severity") == "blocking" for item in issues)
        self.summary.setText(f"阻断 {blockers} · 警告 {len(issues) - blockers}")
        low_count = sum(item.get("category") == "low_confidence" for item in issues)
        low_groups = sum(group[0].get("category") == "low_confidence" for group in groups)
        self.confidence_summary.setText(f"低置信度待审核 {low_groups}" +
                                       (f" · 记录 {low_count}" if low_groups != low_count else ""))
        self.btn_next_low.setEnabled(low_count > 0)
        self.list.clear()
        for group in groups:
            issue = group[0]
            marker = (glyphs.BLOCK if issue.get("severity") == "blocking" else glyphs.WARN)
            badge = ""
            if issue.get("category") == "low_confidence":
                values = [entity_confidence.get(str(ref))
                          for ref in issue.get("entity_refs") or []]
                confidence = next((value for value in values
                                   if isinstance(value, (int, float)) and not isinstance(value, bool)
                                   and math.isfinite(float(value)) and 0 <= value <= 1), None)
                badge = (f"[低 {float(confidence):.0%}] "
                         if confidence is not None else "[低/未知] ")
            summary = ("像素拓扑候选与草稿不一致：请核对节点位置和杆件连接"
                       if issue.get("category") == "pixel_topology" else issue.get("message", issue["id"]))
            item = QListWidgetItem(
                f"{marker} {badge}{summary}" +
                (f"（{len(group)} 条记录）" if len(group) > 1 else ""))
            item.setData(256, str(issue["id"]))
            self.list.addItem(item)
        target = next((row for row in range(self.list.count())
                       if any(str(i["id"]) == current for i in
                              self._groups[self.list.item(row).data(256)])), 0)
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

    def issue_group(self, identifier: str) -> list[dict]:
        return next((group for group in self._groups.values()
                     if any(str(item["id"]) == identifier for item in group)), [])

    def _selection_changed(self, current, previous) -> None:
        issue = self.current_issue()
        if issue is None:
            return
        group = self._groups[str(issue["id"])]
        self.message.setText("\n".join(f"{n}. {i.get('message', '')}" for n, i in enumerate(group, 1))
                             if len(group) > 1 else str(issue.get("message", "")))
        self._set_actions(str(issue.get("category")))
        self.issue_selected.emit(str(issue["id"]), list(issue.get("entity_refs") or []))

    def _set_actions(self, category: str | None) -> None:
        current = self.current_issue() or {}
        unbound = category == "load_incomplete" and current.get("unbound_symbols_only") is True and bool(current.get("observations"))
        self._confirm_action = "ignore" if unbound else "confirm"
        self.btn_confirm.setText("忽略符号" if unbound else "确认此项")
        self.btn_confirm.setToolTip("明确这些未绑定符号无需施加为外荷载，保留原观察记录；其他荷载问题仍需处理。" if unbound else "核对当前问题涉及的对象后确认。")
        self.btn_confirm.setVisible(category in _CONFIRMABLE or unbound)
        if category == "pixel_topology":
            self._confirm_action = "keep_topology"
            self.btn_confirm.setText("保留当前拓扑")
            self.btn_confirm.setToolTip("仅当已核对原图、确认像素候选不适用时使用；保留候选和人工记录，其他阻断仍需处理。")
            self.btn_confirm.setVisible(True)
        intersection = category == "intersection_unknown"
        self.btn_connect.setVisible(intersection)
        self.btn_cross.setVisible(intersection)
        self.btn_reuse.setVisible(category == "merge_collision")

    def _request(self, action: str) -> None:
        identifier = self.current_issue_id()
        if identifier:
            self.resolution_requested.emit(identifier, action)
