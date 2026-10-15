"""主/子 Agent 的真实任务窗口，与原工程保持隔离直到用户审阅应用。"""
from __future__ import annotations

import json
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QDialog, QFrame, QHBoxLayout, QHeaderView,
                               QLabel, QPlainTextEdit, QPushButton, QScrollArea,
                               QTextBrowser, QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from agent_task import AgentTask, TaskBudget
from model_profiles import redact
from . import dialog_styles, theme

STATUS = {"pending": "待开始", "running": "执行中", "completed": "已完成", "ready": "候选待审阅",
          "draft": "候选待补充", "waiting_input": "等待补充", "blocked": "检查未通过",
          "failed": "请求失败", "cancelled": "已停止", "budget_exhausted": "预算已用完",
          "stale": "基线已失效", "applied": "候选已应用", "needs_continuation": "待继续执行"}
ROLE = {"modeler": "建模 Agent", "reviewer": "检查 Agent", "manager": "主 Agent"}
STATUS_COLOUR = {"completed": theme.SUCCESS, "applied": theme.SUCCESS, "running": theme.ACCENT,
                 "blocked": theme.ERROR, "failed": theme.ERROR, "budget_exhausted": theme.WARN,
                 "stale": theme.WARN, "waiting_input": theme.WARN, "cancelled": theme.INK_DIM}
COLLECTION = {"nodes": "节点", "members": "杆件", "materials": "材料", "sections": "截面",
              "supports": "支座", "load_cases": "工况", "combos": "组合", "sets": "集合"}


class _OfflineManager:
    """离线仅对已有模型分派实际检查，不声称理解自由建模需求。"""
    def __init__(self, session):
        self.session = session

    def complete(self, messages, tools):
        last = messages[-1]
        if last["role"] == "user":
            if not self.session.model.get("nodes"):
                return {"tool_calls": [{"id": "missing", "name": "request_input", "arguments": {
                    "questions": [{"prompt": "离线协作只检查已有模型，请先建模或配置在线主模型。", "choices": []}]}}]}
            return {"tool_calls": [
                {"id": "plan", "name": "set_task_plan", "arguments": {"steps": [
                    {"id": "review", "title": "校验与按需求求解当前模型", "role": "reviewer"}]}},
                {"id": "review", "name": "delegate_agent", "arguments": {
                    "step_id": "review", "instruction": last["content"], "source": ""}}]}
        return {"content": "离线检查子任务已结束。请查看步骤状态和实际工具证据；原工程模型保持不变。"}


class _OfflineReviewer:
    def complete(self, messages, tools):
        last = messages[-1]
        prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        name = None
        args = {}
        if last["role"] == "user":
            name = "validate_model"
        elif last["role"] == "tool":
            result = json.loads(last["content"])
            preceding = next(m for m in reversed(messages) if m.get("tool_calls"))
            previous = preceding["tool_calls"][0]["function"]["name"]
            if not result.get("ok"):
                return {"content": "检查未通过，请按工具返回的问题补充或修改模型。"}
            if previous == "validate_model" and any(word in prompt for word in ("求解", "位移", "计算")):
                name = "solve_model"
            elif previous == "solve_model":
                name, args = "query_results", {"what": "max_displacement"}
        if name:
            return {"tool_calls": [{"id": name, "name": name, "arguments": args}]}
        return {"content": "所请求的检查与结果读取已完成，数值请查看实际工具证据。"}


class AgentTaskDialog(QDialog):
    def __init__(self, chat):
        super().__init__(chat)
        self.chat, self.runner, self.task = chat, chat.runner, None
        self._running = False
        self.setWindowTitle("Agent 协作任务")
        screen = self.screen().availableGeometry()
        self.resize(min(720, int(screen.width() * .85)), min(720, int(screen.height() * .85)))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(dialog_styles.DialogHeader(
            "Agent 协作任务", "主 Agent 分派任务，子 Agent 在独立候选上工作；原模型只有在审阅并应用后才改变。"))
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(12, 12, 12, 8)
        body.setSpacing(8)
        area.setWidget(content)
        outer.addWidget(area, 1)

        # 角色与预算只在任务开始前可改，三个下拉排成一行，不各占一整行
        options = QHBoxLayout()
        options.setSpacing(6)
        self.role_models = {}
        for role in ("modeler", "reviewer"):
            combo = QComboBox()
            combo.addItem("跟随主模型", "")
            for profile in chat.profiles:
                combo.addItem(f"{profile.label} · {profile.model}", profile.id)
            options.addWidget(QLabel(ROLE[role]))
            options.addWidget(combo, 1)
            options.addSpacing(6)
            self.role_models[role] = combo
        self.limit = QComboBox()
        for value in (16, 32, 64):
            self.limit.addItem(f"{value} 次模型往返", value)
        self.limit.setCurrentIndex(1)
        options.addWidget(QLabel("预算"))
        options.addWidget(self.limit)
        body.addLayout(options)

        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("说明任务，或补充上一步缺少的条件…")
        self.prompt.setMinimumHeight(64)
        self.prompt.setMaximumHeight(110)
        self.prompt.setPlainText(chat.entry.toPlainText())
        body.addWidget(self.prompt)

        self.status = QLabel("尚未开始。")
        self.status.setWordWrap(True)
        self.status.setProperty("panel", "hint")
        body.addWidget(self.status)
        self.steps = QTreeWidget()
        self.steps.setHeaderLabels(["步骤", "状态"])
        self.steps.setRootIsDecorated(False)
        self.steps.setMinimumHeight(90)
        self.steps.setMaximumHeight(150)
        header = self.steps.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        body.addWidget(self.steps)
        self.questions = QLabel()
        self.questions.setWordWrap(True)
        self.questions.setVisible(False)
        body.addWidget(self.questions)
        self.choice_box = QWidget()
        self.choice_layout = QVBoxLayout(self.choice_box)
        self.choice_layout.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.choice_box)
        self.reply = QTextBrowser()
        self.reply.setOpenLinks(False)
        self.reply.setPlaceholderText("任务完成后，主 Agent 的结论与实际工具结果显示在这里。")
        self.reply.setMinimumHeight(90)
        body.addWidget(self.reply, 1)

        # 工具证据是审计用的，默认收起；展开后逐行列出调用，不再整段倾倒 JSON
        self.evidence_toggle = QToolButton()
        self.evidence_toggle.setText("实际工具证据")
        self.evidence_toggle.setCheckable(True)
        self.evidence_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.evidence_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.evidence_toggle.setProperty("chat", "tool")
        self.evidence_toggle.setToolTip(
            "任务与子 Agent 上下文保留于本次运行，停止后可补充需求继续；检查与求解串行执行，"
            "审计保存到本地任务目录。知识库与跨运行恢复尚未接入。")
        body.addWidget(self.evidence_toggle)
        self.evidence = QPlainTextEdit()
        self.evidence.setReadOnly(True)
        self.evidence.setMinimumHeight(110)
        self.evidence.setVisible(False)
        self.evidence_toggle.toggled.connect(self._toggle_evidence)
        body.addWidget(self.evidence)

        footer = QWidget()
        footer.setObjectName("dialogFooter")
        row = QHBoxLayout(footer)
        row.setContentsMargins(12, 8, 12, 12)
        self.new_button = QPushButton("新建任务")
        self.stop_button = QPushButton("停止")
        self.candidate = QComboBox()
        self.candidate.addItem("尚无建模候选", "")
        self.candidate.setMinimumWidth(120)
        self.preview_button = QPushButton("审阅候选…")
        self.start_button = QPushButton("开始任务")
        for widget in (self.new_button, self.stop_button):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.candidate)
        row.addWidget(self.preview_button)
        row.addSpacing(8)
        row.addWidget(self.start_button)
        outer.addWidget(footer)
        dialog_styles.set_primary(self.start_button, self.new_button, self.stop_button, self.preview_button)
        self.new_button.clicked.connect(self.new_task)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.preview_button.clicked.connect(self.preview)
        self.runner.busy_changed.connect(self._busy)
        self._busy(self.runner.busy)

    def _toggle_evidence(self, shown):
        self.evidence.setVisible(shown)
        self.evidence_toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def _busy(self, busy):
        for widget in (self.prompt, self.start_button, self.new_button, self.choice_box, self.candidate):
            widget.setEnabled(not busy)
        for combo in (*self.role_models.values(), self.limit):
            combo.setEnabled(not busy and self.task is None)
        self.preview_button.setEnabled(not busy and bool(self.candidate.currentData()))
        self.stop_button.setEnabled(busy and self._running)

    def _build_task(self):
        import credentials
        from configured_provider import create_provider
        from object_context import context_message
        selected = self.chat.model_selector.currentData()
        profiles = {p.id: replace(p) for p in self.chat.profiles}
        role_ids = {r: c.currentData() or selected for r, c in self.role_models.items()}
        legacy_key = credentials.load_api_key() or ""
        def factory(role, sandbox):
            connection = role_ids[role]
            if connection == "offline":
                if role == "modeler":
                    raise ValueError("离线模式不支持自由建模，请配置在线模型")
                return _OfflineReviewer()
            if connection == "legacy":
                from agent import DeepSeekProvider
                return DeepSeekProvider(api_key=legacy_key)
            if connection not in profiles:
                raise ValueError("子Agent连接已不存在，请重新选择模型")
            return create_provider(profiles[connection])
        for connection in role_ids.values():
            if connection in profiles:
                profiles[connection].validate()
        provider = _OfflineManager(self.chat.session) if selected == "offline" else self.chat._make_provider()
        context = context_message(self.chat.session.model, self.chat.references, self.chat.reference_revision)
        return AgentTask(self.chat.session, provider, factory,
                         budget=TaskBudget(max_model_calls=self.limit.currentData()),
                         secret_keys=[p.api_key for p in profiles.values()] + [legacy_key], context=context)

    def start(self):
        if self.runner.busy or not self.prompt.toPlainText().strip():
            return
        text = self.prompt.toPlainText().strip()
        try:
            if self.task is None:
                self.task = self._build_task()
        except Exception as exc:  # noqa: BLE001 配置错误留在窗口，不锁住输入
            self._failed(type(exc).__name__, str(exc))
            return
        self.prompt.clear()
        self._running = True
        self.status.setText("主 Agent 正在组织任务，原工程保持不变。")
        self.runner.submit(lambda report: self.task.ask(text, on_event=report),
                           on_done=self._done, on_failed=self._failed, on_progress=self._event)

    def _event(self, _name, record, _ok):
        self._render(record["state"])

    def _render(self, data):
        self.steps.clear()
        children = {c["step_id"]: c for c in data["children"]}
        for step in data["steps"]:
            state = children.get(step["id"], {}).get("status", "pending")
            item = QTreeWidgetItem(self.steps, [f"{step['title']}　{ROLE[step['role']]}",
                                                STATUS.get(state, "待开始")])
            item.setForeground(1, QColor(STATUS_COLOUR.get(state, theme.INK_MUTED)))
        self.status.setText(f"{STATUS.get(data['status'], data['status'])}　·　模型往返 "
                            f"{data['budget']['model_calls']}/{data['budget']['max_model_calls']}　·　"
                            f"工具 {data['budget']['tool_calls']}/{data['budget']['max_tool_calls']}")
        self.evidence.setPlainText("\n".join(
            _evidence_line(e) for e in data["events"] if e["event"] == "tool"))

    def _done(self, data):
        self._running = False
        self._render(data)
        self.reply.setPlainText(data["reply"])
        numbers = [event for event in data["events"] if event.get("tool") == "query_results"
                   and event.get("ok") and event.get("arguments", {}).get("what") == "max_displacement"
                   and "magnitude_mm" in event.get("result", {})]
        titles = {step["id"]: step["title"] for step in data["steps"]}
        for event in numbers:
            value = event["result"]
            child = next(c for c in data["children"] if c["step_id"] == event["step_id"])
            if event["model_hash"] != child["model_hash"]:
                continue
            source = f"候选「{titles.get(child['source'], child['source'])}」" if child["source"] else "原模型"
            # 显示取 4 位有效数字；完整值保留在工具证据与审计文件中
            self.reply.setPlainText(self.reply.toPlainText() +
                                    f"\n\n实际工具结果（{ROLE.get(child.get('role'), '子 Agent')}，{source}）："
                                    f"工况 {value['case']} 最大节点位移 {value['magnitude_mm']:.4g} mm，"
                                    f"位于节点 {value.get('node')}。")
        if data.get("audit_error"):
            self.status.setText(self.status.text() + "\n" + data["audit_error"])
        self.questions.setText("\n".join(q["prompt"] for q in data["questions"]))
        self.questions.setVisible(bool(data["questions"]))
        while self.choice_layout.count():
            item = self.choice_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for question in data["questions"]:
            for choice in question["choices"]:
                button = QPushButton(choice)
                button.clicked.connect(lambda _checked=False, q=question["prompt"], c=choice:
                                       self.prompt.appendPlainText(f"{q}：{c}"))
                self.choice_layout.addWidget(button)
        self.candidate.clear()
        for child in self.task.candidates():
            self.candidate.addItem(titles.get(child.step_id, child.step_id), child.step_id)
        if not self.candidate.count():
            self.candidate.addItem("尚无建模候选", "")
        self.start_button.setText("继续任务")
        self._busy(False)

    def _failed(self, kind, message):
        self._running = False
        keys = [p.api_key for p in self.chat.profiles]
        detail = redact(message, keys)[:220] or kind
        self.status.setText(f"任务未完成：{detail}")
        self._busy(False)

    def stop(self):
        if self.task is not None and self._running:
            self.task.cancel()
            self.runner.cancel()
            self.status.setText("正在停止，子任务草稿保留，原模型未被子Agent改动。")

    def new_task(self):
        if self.runner.busy:
            return
        if self.task is not None:
            self.task.close()
        self.task = None
        self.steps.clear()
        self.evidence.clear()
        self.reply.clear()
        self.questions.clear()
        self.questions.setVisible(False)
        while self.choice_layout.count():
            item = self.choice_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.candidate.clear()
        self.candidate.addItem("尚无建模候选", "")
        self.start_button.setText("开始任务")
        self.status.setText("尚未开始。")
        self._busy(False)

    def preview(self):
        if self.runner.busy or self.task is None or not self.candidate.currentData():
            return
        try:
            preview = self.task.preview(self.candidate.currentData())
        except ValueError as exc:
            self._failed("候选不可用", str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("候选变更审阅")
        dialog.resize(560, 420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("应用将整体写入该候选，旧分析结果失效；可通过撤销退回。"))
        details = QPlainTextEdit(_diff_text(preview))
        details.setReadOnly(True)
        layout.addWidget(details)
        buttons = dialog_styles.button_box(dialog, ok="应用候选", cancel="取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            try:
                applied = self.task.apply(preview["step_id"], preview["candidate_hash"], confirmed=True)
                if not applied.ok:
                    raise ValueError(str(applied.payload))
                self.chat.model_changed.emit(True)
                self.chat._refresh_references()
                self._render(self.task.snapshot())
                self.start_button.setEnabled(False)
                self.preview_button.setEnabled(False)
            except ValueError as exc:
                self._failed("候选未应用", str(exc))
        dialog.deleteLater()

    def shutdown(self):
        self.hide()
        if self.task is not None:
            self.task.close()


def _evidence_line(event):
    """一次工具调用一行：角色、工具、参数摘要、成败与模型版本。"""
    args = "，".join(f"{k}={v}" for k, v in (event.get("arguments") or {}).items())
    mark = "成功" if event.get("ok") else "失败"
    return (f"[{ROLE.get(event.get('role'), event.get('role', ''))}] {event.get('tool')}({args}) "
            f"{mark} · 模型版本 {str(event.get('model_hash', ''))[:10]}")


def _diff_text(preview):
    """候选差异按对象类别逐行列出；完整差异仍附在后面供核对。"""
    diff = preview.get("diff", {})
    lines = []
    for key, change in (diff.get("collections") or {}).items():
        parts = [f"{change['before']} → {change['after']}"]
        for label, field in (("新增", "added"), ("删除", "removed"), ("修改", "changed")):
            if change.get(field):
                ids = [str(i) for i in change[field]]
                parts.append(f"{label} {'、'.join(ids[:12])}" + (" 等" if len(ids) > 12 else ""))
        lines.append(f"{COLLECTION.get(key, key)}：" + "；".join(parts))
    for key in (diff.get("fields") or {}):
        lines.append(f"{COLLECTION.get(key, key)}：已修改")
    if not lines:
        lines.append("候选与原模型相同。")
    errors = preview.get("validation_errors") or []
    if errors:
        lines += ["", "校验提示："] + [f"· {e}" for e in errors]
    lines += ["", f"候选版本 {preview.get('candidate_hash', '')[:12]}", "",
              "完整差异：", json.dumps(diff, ensure_ascii=False, indent=2)]
    return "\n".join(lines)
