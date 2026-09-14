"""Agent 对话面板。

这个项目的立论是"自然语言驱动结构计算"。桌面端只能点按钮的话，
最正式的那个界面恰恰否定了项目本身——所以这块面板不是锦上添花，
是**主张的兑现处**。

有一个设计决定值得说清楚：**工具调用要摆在明面上。**

```
你：24 米跨门式刚架，柱脚铰接，屋面恒载 8 kN/m
────────────────────────────────────────
    ⚙ define_materials_and_sections
    ⚙ generate_portal_frame   spans=[24.0] eave_height=7.5
    ⚙ set_load_cases
    ⚙ solve_model
Agent：已建好并求解……最大竖向位移 32.7 mm
```

因为整个项目最容易被质疑的一句是：**"这个数是不是大模型编的？"**
把调用链摊开，答案就是自明的——模型只决定调什么，数从求解器出来。
藏起来反而显得心虚。

面板本身不算不画：`Conversation` 改的是同一个 `Session`，改完发信号，
主窗口去重画。所以对话和工具栏操作是**同一份模型**，不会各改各的。
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from PySide6.QtCore import QElapsedTimer, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QPlainTextEdit, QPushButton,
                               QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from . import theme

# 离线时用的脚本。**不是假装在跟大模型说话**——它照样把真工具跑一遍，
# 出来的数还是求解器算的。没有密钥、没有网的场合（答辩现场很常见）
# 靠它演示完整链路
from .demo_script import DEMO_PROMPT, demo_provider


_AGENT_PROFILES = {
    "fast": {"label": "快速", "temperature": 0.0,
             "keep_turns": 3, "context_chars": 12_000},
    "balanced": {"label": "均衡", "temperature": 0.18,
                 "keep_turns": 6, "context_chars": 24_000},
    "explore": {"label": "探索", "temperature": 0.45,
                "keep_turns": 6, "context_chars": 24_000},
}

_TOOL_LABELS = {
    "define_materials_and_sections": "定义材料与截面",
    "generate_frame": "生成规则框架",
    "edit_member": "修改杆件",
    "edit_node": "修改节点",
    "define_set": "创建对象集合",
    "list_sets": "读取对象集合",
    "generate_portal_frame": "生成门式刚架",
    "generate_bent": "生成平面排架",
    "extrude_bents": "扩展为空间结构",
    "select_members": "按条件选择杆件",
    "add_bracing": "添加结构支撑",
    "raise_nodes": "调整节点标高",
    "retaper": "修改构件截面",
    "add_nodes": "添加节点",
    "add_members": "添加杆件",
    "set_load_cases": "设置荷载工况",
    "set_supports": "设置边界约束",
    "assign_properties": "指派材料与截面",
    "remove_members": "移除杆件",
    "set_model": "写入完整模型",
    "add_load_case": "新建荷载工况",
    "set_nodal_load": "设置节点荷载",
    "set_member_load": "设置杆件荷载",
    "set_member_span_load": "设置杆件局部荷载",
    "set_prescribed_displacement": "设置给定位移",
    "set_member_strain": "设置初始应变",
    "set_units": "转换模型单位制",
    "add_self_weight": "施加结构自重",
    "validate_model": "检查分析模型",
    "preview_analysis_mesh": "预览分析网格",
    "solve_model": "运行结构求解",
    "query_results": "读取计算结果",
    "plot_results": "生成结果图",
    "query_diagram": "提取杆件内力",
    "query_envelope": "计算结果包络",
    "check_strength": "执行强度校核",
    "check_symmetry": "检查结构对称性",
    "check_numbering": "检查模型编号",
    "diagnose_supports": "诊断约束条件",
    "buckling_analysis": "执行屈曲分析",
    "modal_analysis": "执行模态分析",
    "sweep": "执行参数扫描",
    "write_report": "生成分析报告",
    "export_learning_trace": "导出操作记录",
    "analyze_joint_solid": "分析节点实体",
    "solve_with_abaqus": "调用 Abaqus 求解",
    "compare_solvers": "对比求解器结果",
    "preview_change": "预演模型修改",
    "apply_preview": "应用已确认修改",
}

_TOOL_STAGES = {
    "定义模型": {
        "define_materials_and_sections", "generate_frame", "edit_member",
        "edit_node", "define_set", "generate_portal_frame", "generate_bent",
        "extrude_bents", "select_members", "add_bracing", "raise_nodes",
        "retaper", "add_nodes", "add_members", "assign_properties",
        "remove_members", "set_model", "set_units"},
    "设置分析条件": {
        "set_load_cases", "set_supports", "add_load_case", "set_nodal_load",
        "set_member_load", "set_member_span_load",
        "set_prescribed_displacement", "set_member_strain", "add_self_weight"},
    "检查模型": {
        "validate_model", "preview_analysis_mesh", "diagnose_supports",
        "check_symmetry", "check_numbering", "preview_change", "apply_preview"},
    "运行分析": {
        "solve_model", "buckling_analysis", "modal_analysis", "sweep",
        "analyze_joint_solid", "solve_with_abaqus", "compare_solvers"},
    "读取结果": {
        "query_results", "query_diagram", "query_envelope", "check_strength",
        "plot_results"},
    "生成交付": {"write_report", "export_learning_trace"},
}


def _tool_label(name: str) -> str:
    return _TOOL_LABELS.get(name, "执行扩展工程操作")


def _tool_stage(name: str) -> str:
    for stage, names in _TOOL_STAGES.items():
        if name in names:
            return stage
    return "工程处理"


@dataclass(frozen=True)
class AgentActivityEvent:
    """Agent 面板消费的工程事件；内部函数名不进入显示合同。"""

    step: int
    stage: str
    action: str
    state: str
    detail: str = ""


class PromptEdit(QPlainTextEdit):
    """多行输入框。

    原来用的是 `QLineEdit`——**单行，根本打不了换行**。而描述一个结构
    往往要写两三句：几何一句、荷载一句、要问什么一句。憋在一行里
    既难写也难改，这是个功能缺陷，不是观感问题。

    键位按聊天软件的通行约定，不自己发明：

        Enter / Ctrl+Enter   发送
        Shift+Enter          换行

    高度随内容长，但封顶——输入框吃掉半个面板的话，
    上面的对话记录就看不见了。
    """

    submitted = Signal()

    MAX_LINES = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        # 占位文字要短。写全"Enter 发送，Shift+Enter 换行"会被面板宽度裁掉，
        # 裁掉一半的提示比没有提示更让人困惑。键位说明放进 tooltip
        self.setPlaceholderText("描述结构，或对上一轮结果提要求…")
        self.setToolTip("Enter 发送　Shift+Enter 换行")
        self.setTabChangesFocus(True)
        self.document().documentLayout().documentSizeChanged.connect(
            lambda *_: self._fit_height())
        self._fit_height()

    def _fit_height(self) -> None:
        line = self.fontMetrics().lineSpacing()
        rows = min(max(1, int(self.document().size().height())), self.MAX_LINES)
        self.setFixedHeight(int(line * rows) + 14)

    # --- 与 QLineEdit 兼容的几个别名 ---
    #
    # 这个类是从 `QLineEdit` 换过来的，调用方（预填按钮、测试）用的是
    # `setText / text / selectAll / hasSelectedText`。**做成直接替代品**
    # 比让每个调用方去记"这个框换成多行了"更省事，也少一类改漏的机会。

    def setText(self, text: str) -> None:
        self.setPlainText(text)

    def text(self) -> str:
        return self.toPlainText()

    def selectAll(self) -> None:
        cursor = self.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        self.setTextCursor(cursor)

    def hasSelectedText(self) -> bool:
        return self.textCursor().hasSelection()

    def clear(self) -> None:
        self.setPlainText("")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if enter and not shift:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class ChatPanel(QWidget):
    """多轮对话面板。"""

    # 模型被改动了，主窗口据此重画；bool 表示这一轮是否动过模型
    model_changed = Signal(bool)
    busy_changed = Signal(bool)

    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self.conversation = None
        self._offline = False
        # 本轮已经流式显示了几条工具调用。收尾时据此决定要不要补显示，
        # 否则联网失败退回离线的那种路径会把调用链打印两遍
        self._streamed = 0
        self._streamed_text = ""
        self._activity_step = 0
        self._workspace_kind: str | None = None
        self._workspace_id: int | None = None
        self._workspace_mode = "模型"
        self._busy_phase = ""
        self._elapsed = QElapsedTimer()
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(250)
        self._busy_timer.timeout.connect(self._update_busy_status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("FrameLab Agent", self)
        title.setObjectName("agentTitle")
        header.addWidget(title)
        self.lbl_status = QLabel()
        self.lbl_status.setObjectName("agentConnectionStatus")
        self.lbl_status.setProperty("panel", "hint")
        header.addWidget(self.lbl_status)
        header.addStretch(1)
        self.profile = QComboBox(self)
        self.profile.setObjectName("agentProfile")
        for key, profile in _AGENT_PROFILES.items():
            self.profile.addItem(profile["label"], key)
        self.profile.setCurrentIndex(self.profile.findData("balanced"))
        self.profile.setToolTip(
            "快速：缩短历史上下文，表达最稳定。\n"
            "均衡：保留更多上下文，并增加轻微表达变化。\n"
            "探索：提高方案与表达多样性；求解器数值始终不变。")
        self.profile.currentIndexChanged.connect(self._profile_changed)
        header.addWidget(self.profile)
        self.btn_connection = QToolButton(self)
        self.btn_connection.setObjectName("agentHeaderAction")
        self.btn_connection.setText("连接设置")
        self.btn_connection.setCheckable(True)
        header.addWidget(self.btn_connection)
        self.btn_new_chat = QToolButton(self)
        self.btn_new_chat.setObjectName("agentHeaderAction")
        self.btn_new_chat.setText("新对话")
        self.btn_new_chat.setToolTip("清空聊天上下文，但保留当前结构模型和计算结果")
        self.btn_new_chat.clicked.connect(self._new_conversation)
        header.addWidget(self.btn_new_chat)
        layout.addLayout(header)

        # 多模型降级配置
        from PySide6.QtWidgets import QCheckBox, QLineEdit
        self.router_widget = QWidget(self)
        self.router_row = QHBoxLayout(self.router_widget)
        self.router_row.setContentsMargins(0, 0, 0, 0)
        self.chk_router = QCheckBox("备用模型降级")
        self.chk_router.setToolTip("主模型失败时自动切换到备用模型")
        self.chk_router.toggled.connect(self._toggle_router_config)
        self.router_row.addWidget(self.chk_router)
        self.router_row.addStretch()
        self.router_widget.hide()
        layout.addWidget(self.router_widget)

        self.router_panel = QFrame(self)
        self.router_panel.setProperty("panel", "config")
        router_grid = QGridLayout(self.router_panel)
        router_grid.setContentsMargins(8, 7, 8, 7)
        router_grid.setHorizontalSpacing(6)
        router_grid.setVerticalSpacing(5)
        self.txt_bk_model = QLineEdit("gpt-4o-mini")
        self.txt_bk_model.setPlaceholderText("备用模型名")
        self.txt_bk_url = QLineEdit("https://api.openai.com/v1")
        self.txt_bk_url.setPlaceholderText("备用 base_url")
        self.txt_bk_key = QLineEdit()
        self.txt_bk_key.setPlaceholderText("备用密钥")
        self.txt_bk_key.setEchoMode(QLineEdit.EchoMode.Password)
        for row, (label, field) in enumerate((
                ("模型", self.txt_bk_model), ("地址", self.txt_bk_url),
                ("密钥", self.txt_bk_key))):
            router_grid.addWidget(QLabel(label), row, 0)
            router_grid.addWidget(field, row, 1)
        self.router_panel.setVisible(False)
        layout.addWidget(self.router_panel)
        self.btn_connection.toggled.connect(self._toggle_connection_settings)

        # 模型调用状态
        self.lbl_call_status = QLabel("")
        self.lbl_call_status.setStyleSheet(f"color:{theme.INK_MUTED}; font-size:8pt;")
        layout.addWidget(self.lbl_call_status)

        self.view = QTextBrowser(self)
        self.view.setOpenExternalLinks(False)
        self.view.setObjectName("assistantConversation")
        self._show_welcome()
        layout.addWidget(self.view, 1)

        # 正文到一段显示一段；完成后再固化进上方历史区。
        self.live_response = QFrame(self)
        self.live_response.setObjectName("agentLiveResponse")
        self.live_response.setMaximumHeight(180)
        live_layout = QVBoxLayout(self.live_response)
        live_layout.setContentsMargins(10, 8, 10, 8)
        live_layout.setSpacing(3)
        live_title = QLabel("FrameLab Agent", self.live_response)
        live_title.setProperty("role", "agent")
        live_layout.addWidget(live_title)
        self.lbl_stream = QLabel(self.live_response)
        self.lbl_stream.setWordWrap(True)
        self.lbl_stream.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        live_layout.addWidget(self.lbl_stream)
        self.live_response.hide()
        layout.addWidget(self.live_response)

        suggestion_caption = QLabel("下一步建议", self)
        suggestion_caption.setObjectName("agentSuggestionCaption")
        layout.addWidget(suggestion_caption)
        suggestions = QHBoxLayout()
        suggestions.setSpacing(4)
        self.suggestion_buttons: list[QPushButton] = []
        for _ in range(3):
            button = QPushButton(self)
            button.setProperty("role", "suggestion")
            button.clicked.connect(lambda _=False, b=button:
                                   self._use_suggestion(b.property("prompt") or ""))
            self.suggestion_buttons.append(button)
            suggestions.addWidget(button)
        suggestions.addStretch(1)
        layout.addLayout(suggestions)

        row = QHBoxLayout()
        self.entry = PromptEdit(self)
        self.entry.submitted.connect(self.send)
        self.btn_send = QPushButton("发送", self)
        self.btn_send.setProperty("role", "primary")
        self.btn_send.clicked.connect(self.send)
        self.btn_stop = QPushButton("停止", self)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("中止本轮后续工具调用。已执行的调用不会回滚，如需退回请使用撤销（Ctrl+Z）")
        self.btn_stop.clicked.connect(self.stop)
        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(self.btn_send)
        col.addWidget(self.btn_stop)
        row.addWidget(self.entry, 1)
        row.addLayout(col)
        layout.addLayout(row)

        self._refresh_status()
        self.refresh_suggestions()

    # --- 后端 ---

    def _show_welcome(self) -> None:
        self.view.setHtml(
            f'<div style="margin:18px 14px;color:{theme.INK};">'
            f'<div style="font-size:16px;font-weight:600;color:{theme.ACCENT_DIM};">'
            '结构建模助手</div>'
            f'<p style="color:{theme.INK_MUTED};line-height:1.55;">'
            '描述结构几何、材料、约束和荷载。我会显示实际工具调用，'
            '计算结果始终来自确定性求解器。</p>'
            f'<p style="color:{theme.INK_DIM};font-size:12px;">'
            '可从下方示例开始，或直接输入你的工程要求。</p></div>')

    def _use_suggestion(self, text: str) -> None:
        self.entry.setPlainText(text)
        self.entry.setFocus()

    def refresh_suggestions(self) -> None:
        """根据确定性的模型阶段预测下一步，不额外调用大模型。"""
        from workflow import inspect_workflow
        from .agent_guidance import next_suggestions

        status = inspect_workflow(self.session)
        items = next_suggestions(status, self._workspace_kind,
                                 self._workspace_id, self._workspace_mode)
        for button, (label, prompt) in zip(self.suggestion_buttons, items):
            button.setText(label)
            button.setProperty("prompt", prompt)
            button.setToolTip(prompt)
            button.show()
        for button in self.suggestion_buttons[len(items):]:
            button.hide()

    def set_workspace_selection(self, kind: str | None,
                                identifier: int | None) -> None:
        """让建议感知用户正在操作的工程对象。"""
        self._workspace_kind = kind if kind in {"node", "member"} else None
        self._workspace_id = identifier if self._workspace_kind else None
        self.refresh_suggestions()

    def set_workspace_mode(self, mode: str) -> None:
        """让已求解阶段的建议感知用户当前正在看的结果视图。"""
        self._workspace_mode = str(mode or "模型")
        self.refresh_suggestions()

    def _refresh_status(self) -> None:
        """说清楚现在接的是谁。**任何情况下都不显示密钥本身**，
        只显示来源和指纹——够回答"是不是换了把钥匙"，且不可还原。"""
        import credentials

        key = credentials.load_api_key()
        if key:
            self._offline = False
            self.lbl_status.setText("在线")
            self.lbl_status.setProperty("connection", "online")
            self.lbl_status.setToolTip(
                f"DeepSeek 在线模型；密钥来源 {credentials.source()}；"
                f"密钥指纹 {credentials.fingerprint(key)}。界面不会显示密钥原文。")
        else:
            self._offline = True
            self.lbl_status.setText("离线演示")
            self.lbl_status.setProperty("connection", "offline")
            self.lbl_status.setToolTip(
                f"开启自由对话：在 {credentials.where_to_put_it()} 中写入自己的 "
                "API key，保存为 UTF-8 后重启程序。")
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)

    def _toggle_connection_settings(self, checked: bool) -> None:
        self.router_widget.setVisible(checked)
        self.router_panel.setVisible(checked and self.chk_router.isChecked())

    def _profile_settings(self) -> dict:
        return _AGENT_PROFILES[self.profile.currentData() or "balanced"]

    def _profile_changed(self, *_args) -> None:
        settings = self._profile_settings()
        if self.conversation is None:
            return
        self.conversation.keep_turns = settings["keep_turns"]
        self.conversation.context_chars = settings["context_chars"]
        provider = getattr(self.conversation, "provider", None)
        setter = getattr(provider, "set_temperature", None)
        if callable(setter):
            setter(settings["temperature"])
        self.lbl_call_status.setText(
            f"已切换到{settings['label']}模式，当前模型与对话历史保持不变")

    def _new_conversation(self) -> None:
        if self.runner.busy:
            return
        if self.conversation is not None:
            self.conversation.reset_history()
        self.view.clear()
        self._show_welcome()
        self.refresh_suggestions()
        self.lbl_call_status.setText("已开始新对话，当前结构模型与计算结果已保留")

    def _toggle_router_config(self, checked):
        self.router_panel.setVisible(checked and self.btn_connection.isChecked())
        # 配置变了，下次发送时重建 conversation
        self.conversation = None

    def _ensure_conversation(self):
        if self.conversation is not None:
            return self.conversation
        from conversation import Conversation

        import credentials
        key = credentials.load_api_key()
        settings = self._profile_settings()
        if key:
            if self.chk_router.isChecked():
                from model_router import ModelConfig, ModelRouter
                configs = [
                    ModelConfig(name="deepseek-v4-flash",
                                base_url="https://api.deepseek.com",
                                api_key=key, priority=1,
                                temperature=settings["temperature"]),
                    ModelConfig(name=self.txt_bk_model.text().strip(),
                                base_url=self.txt_bk_url.text().strip(),
                                api_key=self.txt_bk_key.text().strip() or key,
                                priority=2,
                                temperature=settings["temperature"]),
                ]
                provider = ModelRouter(configs)
            else:
                from agent import DeepSeekProvider
                provider = DeepSeekProvider(
                    api_key=key, temperature=settings["temperature"])
        else:
            provider = demo_provider(self.session)
        self.conversation = Conversation(
            provider, session=self.session,
            keep_turns=settings["keep_turns"],
            context_chars=settings["context_chars"])
        return self.conversation

    # --- 交互 ---

    def send(self) -> None:
        text = self.entry.toPlainText().strip()
        if not text or self.runner.busy:
            return
        self.entry.clear()
        self._append_user(text)
        self._set_busy(True)

        conversation = self._ensure_conversation()
        before = self._model_signature()

        self._streamed = 0
        self._streamed_text = ""
        self._activity_step = 0
        self.lbl_stream.clear()
        self.live_response.hide()
        ok = self.runner.submit(
            lambda report: conversation.ask(
                text, on_tool=report,
                on_text=lambda delta: report("__assistant_delta__", delta, True)),
            on_done=lambda r: self._replied(r, before),
            on_failed=self._failed,
            on_progress=self._tool_done)
        if not ok:                      # 理论上到不了，busy 已经拦过
            self._set_busy(False)

    def _tool_done(self, name: str, args, ok: bool) -> None:
        """一个工具跑完就显示一行。**这个回调在主线程里执行**——
        Runner 内部走信号投递，不是工作线程直接调过来的。"""
        if name == "__assistant_delta__":
            self._streamed_text += str(args)
            self.lbl_stream.setText(_clean_agent_text(self._streamed_text))
            self.live_response.show()
            return
        if self._streamed_text:
            # 有些模型会先流出一句操作说明，再发工具调用。把这句固化成独立
            # 消息，随后最终答复从空白流式卡片重新开始，避免两段文字粘连。
            self._append_agent(self._streamed_text)
            self._streamed_text = ""
            self.lbl_stream.clear()
            self.live_response.hide()
        self._busy_phase = f"正在{_tool_label(name)}"
        self._update_busy_status()
        self._streamed += 1
        self._append_tool(name, args, ok)

    def _replied(self, result, before) -> None:
        # 流式已经逐条显示过了就不要再重复一遍。
        # 联网失败退回离线、或者回调没走到时 `_streamed` 会是 0，那时补上
        if not self._streamed:
            for name, args in result.tool_calls:
                self._append_tool(name, args)
        self._append_agent(result.reply or "（本轮模型未返回文字说明）")
        self.live_response.hide()
        self.lbl_stream.clear()
        self._streamed_text = ""
        if result.stopped_by_limit:
            self._append_note("本轮已达到工具调用上限，任务可能未完成。"
                              "建议将要求拆分为多次描述后重试。")
        self._set_busy(False)
        self.model_changed.emit(self._model_signature() != before)
        self.refresh_suggestions()
        self._update_call_status(result)

    def _update_call_status(self, result=None):
        """显示本轮端到端分解；模型慢和工具慢不再混成一个“正在运行”。"""
        if self.conversation is None:
            return
        parts = []
        metrics = getattr(result, "metrics", {}) if result is not None else {}
        if metrics:
            total = metrics.get("total_ms", 0.0) / 1000
            if metrics.get("fast_path"):
                parts.append(f"本轮 {total:.2f} 秒，本地快速路径")
            else:
                model = metrics.get("provider_ms", 0.0) / 1000
                tool = metrics.get("tool_ms", 0.0) / 1000
                calls = int(metrics.get("provider_calls", 0))
                parts.append(
                    f"本轮 {total:.1f} 秒，模型等待 {model:.1f} 秒，"
                    f"工具 {tool:.2f} 秒，{calls} 次模型往返")
                selected = int(metrics.get("tool_schema_count", 0))
                available = int(metrics.get("all_tool_count", 0))
                if selected and available and selected < available:
                    parts.append(f"工具上下文 {selected}/{available}")
        provider = getattr(self.conversation, "provider", None)
        if provider is None:
            # Conversation 可能把 provider 存在别的属性名
            provider = getattr(self.conversation, "_provider", None)
        from model_router import ModelRouter
        if isinstance(provider, ModelRouter) and provider.call_history:
            last = provider.call_history[-1]
            fallback = "，已使用备用模型" if last.was_fallback else ""
            status = "成功" if last.success else \
                f"失败，{last.error[:30] if last.error else '原因未知'}"
            parts.append(f"最近模型：{last.model}，{status}{fallback}")
        self.lbl_call_status.setText("；".join(parts))

    def _failed(self, kind: str, message: str) -> None:
        """网络超时、密钥无效、脚本用尽……都走这里。

        **要把原因说出来**：只显示"失败了"，用户既不知道该重试还是该改配置，
        也没法判断是自己的问题还是程序的问题。
        """
        hint = ""
        if "Timeout" in kind or "timed out" in message:
            hint = "　通常为网络原因，可重试；无网络时可使用离线演示模式。"
        elif "401" in message or "Unauthorized" in message:
            hint = "　请检查 API 密钥是否有效或已过期。"
        self._append_note(f"本轮请求失败：{kind}: {message[:300]}{hint}")
        self.lbl_call_status.setText("请求失败，请查看上方说明")
        self.live_response.hide()
        self.lbl_stream.clear()
        self._streamed_text = ""
        self._set_busy(False)

    def stop(self) -> None:
        """放弃这一轮剩下的工具调用。

        **不回滚已经执行的调用**，也不该回滚——它们已经改了模型，
        假装没发生过反而更糟。撤销是另一件事，用 Ctrl+Z。
        这里只是别让它继续往下跑。
        """
        if self.conversation is not None:
            self.conversation.cancel()
        self._append_note("已请求停止。当前这一步跑完就停，"
                          "已经执行的工具调用不会回滚（要撤销请按 Ctrl+Z）。")
        self.btn_stop.setEnabled(False)

    def _set_busy(self, busy: bool) -> None:
        self.entry.setEnabled(not busy)
        self.entry.setReadOnly(busy)
        self.btn_send.setEnabled(not busy)
        self.btn_send.setText("发送")
        self.btn_stop.setEnabled(busy)
        self.profile.setEnabled(not busy)
        self.btn_new_chat.setEnabled(not busy)
        if busy:
            self._busy_phase = "正在理解问题"
            self._elapsed.start()
            self._busy_timer.start()
            self._update_busy_status()
        else:
            self._busy_timer.stop()
        self.busy_changed.emit(busy)

    def _update_busy_status(self) -> None:
        if not self._elapsed.isValid():
            return
        seconds = self._elapsed.elapsed() / 1000
        self.lbl_call_status.setText(f"{self._busy_phase}，已用时 {seconds:.1f} 秒")

    def _model_signature(self):
        """粗粒度的模型指纹，用来判断这一轮有没有动过模型。

        比对整个模型字典太重，而且没必要——界面只需要知道"要不要重画"。
        """
        m = self.session.model
        return (len(m.get("nodes") or []), len(m.get("members") or []),
                len(m.get("supports") or []), len(m.get("load_cases") or []),
                len(self.session.history))

    def load_demo_prompt(self) -> None:
        self.entry.setPlainText(DEMO_PROMPT)

    # --- 渲染 ---

    def _html(self, html: str) -> None:
        self.view.append(html)
        self.view.moveCursor(QTextCursor.MoveOperation.End)

    def _append_user(self, text: str) -> None:
        self._html(
            f'<div style="margin:10px 2px 6px 24px;padding:8px 10px;'
            f'background-color:{theme.SELECTION};border:1px solid {theme.BORDER};">'
            f'<div style="font-size:11px;color:{theme.INK_MUTED};">你</div>'
            f'<div style="color:{theme.INK};line-height:1.45;">{_esc(text)}</div>'
            '</div>')

    def _append_tool(self, name: str, args: dict, ok: bool = True) -> None:
        """工具调用摊开显示——这是"数不是编的"最直接的证据。

        失败的调用也要显示，而且要标出来。**模型自己改正错误是常事**，
        藏起失败的那几步，用户就看不出它试过什么、为什么改了主意。
        """
        brief = _brief_args(args)
        self._activity_step += 1
        event = AgentActivityEvent(
            step=self._activity_step, stage=_tool_stage(name),
            action=_tool_label(name), state="完成" if ok else "失败",
            detail=brief)
        if event.step == 1:
            self._html(
                f'<div style="margin:8px 18px 3px;color:{theme.INK_DIM};'
                f'font-size:11px;font-weight:600;">执行过程</div>')
        colour = theme.INK_MUTED if ok else theme.WARN
        detail = (f'<div style="margin-top:2px;color:{theme.INK_DIM};">'
                  f'阶段：{_esc(event.stage)}'
                  + (f'；参数：{_esc(event.detail)}' if event.detail else "")
                  + '</div>')
        self._html(
            f'<div style="margin:2px 18px;padding:4px 7px;color:{colour};'
            f'background-color:{theme.PANEL};border-left:2px solid {colour};'
            f'font-size:11px;">步骤 {event.step}，{event.state}：'
            f'{_esc(event.action)}'
            f'{detail}</div>')

    def _append_agent(self, text: str) -> None:
        text = _clean_agent_text(text)
        self._html(
            f'<div style="margin:7px 22px 11px 2px;padding:8px 10px;'
            f'background-color:{theme.PANEL_ALT};border:1px solid {theme.BORDER};">'
            f'<div style="font-size:11px;font-weight:600;color:{theme.ACCENT};">'
            'FrameLab Agent</div>'
            f'<div style="margin-top:3px;color:{theme.INK};line-height:1.5;">'
            f'{_esc(text)}</div></div>')

    def _append_note(self, text: str) -> None:
        self._html(f'<p style="margin:4px 0;color:{theme.WARN};">'
                   f'{_esc(text)}</p>')


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("\n", "<br>"))


def _clean_agent_text(text: str) -> str:
    """去掉模型常见的装饰符号，并把 Markdown 项目转成普通编号。"""
    value = str(text).replace("**", "").replace("`", "")
    value = value.replace("→", "到").replace("✓", "通过").replace("✗", "未通过")
    value = value.replace("⚠️", "注意：").replace("⚠", "注意：").replace("●", "")
    value = value.replace("✅", "").replace("❌", "")
    value = re.sub(r"[\U0001F300-\U0001FAFF]", "", value)
    lines, item_number = [], 0
    for raw in value.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", raw)
        if re.match(r"^\s*[-*•]\s+", line):
            item_number += 1
            line = re.sub(r"^\s*[-*•]\s+", f"{item_number}. ", line)
        elif line.strip():
            item_number = 0
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _brief_args(args: dict, limit: int = 96) -> str:
    """参数摘要。整段 JSON 会把面板刷屏，而读者只想知道"传了什么量级"。"""
    if not args:
        return ""
    labels = {
        "spans": "跨度", "storeys": "层高", "bays": "开间",
        "eave_height": "檐口高度", "ridge_rise": "屋脊高度",
        "materials": "材料", "sections": "截面", "nodes": "节点",
        "members": "杆件", "cases": "工况", "combos": "组合",
        "case": "工况", "what": "结果类型", "analysis": "分析类型",
        "coordinates": "坐标", "ids": "编号",
        "column_section": "柱截面", "beam_section": "梁截面",
        "rafter_section": "屋面梁截面", "section": "截面",
        "material": "材料", "base": "柱脚", "beam_load": "梁线荷载",
        "node": "节点", "member": "杆件", "name": "名称",
        "step": "分析步", "load": "荷载分量", "w": "分布荷载",
        "fix": "约束自由度", "mu_y": "y 向计算长度系数",
        "mu_z": "z 向计算长度系数",
    }
    value_labels = {
        "pinned": "铰接", "fixed": "固定", "linear": "线性静力",
        "pdelta": "二阶效应", "material": "材料非线性",
        "max_displacement": "最大位移", "reactions": "支座反力",
        "member_forces": "杆件内力",
    }
    bits = []
    for key, value in args.items():
        label = labels.get(key, key.replace("_", " "))
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], dict):
                bits.append(f"{label}：{len(value)} 项")
            else:
                bits.append(f"{label}：{list(value)[:4]}"
                            + (" 等" if len(value) > 4 else ""))
        elif isinstance(value, dict):
            bits.append(f"{label}：{len(value)} 项")
        else:
            shown = value_labels.get(str(value), value)
            bits.append(f"{label}：{shown}")
    out = "；".join(bits)
    return out if len(out) <= limit else out[:limit - 3] + "..."
