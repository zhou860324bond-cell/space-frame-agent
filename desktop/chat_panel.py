"""Agent 对话面板。

这个项目的立论是"自然语言驱动结构计算"。桌面端只能点按钮的话，
最正式的那个界面恰恰否定了项目本身——所以这块面板不是锦上添花，
是**主张的兑现处**。

有一个设计决定值得说清楚：**工具调用要摆在明面上。**

```
你：24 米跨门式刚架，柱脚铰接，屋面恒载 8 kN/m
────────────────────────────────────────
    ◆ define_materials_and_sections
    ◆ generate_portal_frame   spans=[24.0] eave_height=7.5
    ◆ set_load_cases
    ◆ solve_model
Agent：已建好并求解……最大竖向位移 32.7 mm
```

因为整个项目最容易被质疑的一句是：**"这个数是不是大模型编的？"**
把调用链摊开，答案就是自明的——模型只决定调什么，数从求解器出来。
藏起来反而显得心虚。

面板本身不算不画：`Conversation` 改的是同一个 `Session`，改完发信号，
主窗口去重画。所以对话和工具栏操作是**同一份模型**，不会各改各的。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QLayout, QMenu,
                               QPlainTextEdit, QPushButton, QSizePolicy, QTextBrowser,
                               QToolButton, QVBoxLayout, QWidget)

from . import theme

# 离线时用的脚本。**不是假装在跟大模型说话**——它照样把真工具跑一遍，
# 出来的数还是求解器算的。没有密钥、没有网的场合（答辩现场很常见）
# 靠它演示完整链路
from .demo_script import DEMO_PROMPT, demo_provider
from . import glyphs


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
        self.setPlaceholderText("描述工程需求…")
        self.setToolTip("Enter 发送　Shift+Enter 换行")
        self.setTabChangesFocus(True)
        self.document().documentLayout().documentSizeChanged.connect(
            lambda *_: self._fit_height())
        self._fit_height()

    MIN_LINES = 2

    def _fit_height(self) -> None:
        # 至少两行：一行高的输入框在输入卡片里显得局促，也容易误以为不能换行。
        # 额外高度要盖住样式表的上下内边距与文档边距，否则一行字也会冒出滚动条
        line = self.fontMetrics().lineSpacing()
        rows = min(max(self.MIN_LINES, int(self.document().size().height())), self.MAX_LINES)
        margin = 2 * self.document().documentMargin() + 2 * self.frameWidth() + 10
        self.setFixedHeight(int(line * rows + margin))

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
    locate_requested = Signal(str, int)

    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self.conversation = None
        self._offline = False
        # 本轮已经流式显示了几条工具调用。收尾时据此决定要不要补显示，
        # 否则联网失败退回离线的那种路径会把调用链打印两遍
        self._streamed = 0
        self._chat_busy = False
        self._provider_dirty = False
        self._dialogs = []
        self._task_dialog = None
        self.references = []
        self.reference_revision = ""
        self._selection = None
        self._chats = []
        self._current_chat = 0
        from model_profiles import ProfileStore
        self._config_error = ""
        try:
            self.profiles, self.selected_profile = ProfileStore().load()
        except ValueError as exc:
            self.profiles, self.selected_profile = [], ""
            self._config_error = str(exc)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        # 第一行：标题 + 会话 / 协作任务 / 连接设置。原来这些各占一行，
        # 对话区上方压了六层控件；收进一行工具按钮后，对话本身才是主体。
        header = QHBoxLayout()
        header.setSpacing(2)
        self.title_label = QLabel("工程对话")
        self.title_label.setObjectName("chatTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.btn_history = _tool_button("对话", "切换本次运行中的对话；所有对话共用当前工程模型，历史不写入磁盘")
        self.btn_history.clicked.connect(self._history_menu)
        self.btn_new_chat = _tool_button("新建", "新建对话（保留当前对话，可从「对话」切回）")
        self.btn_new_chat.clicked.connect(self.new_chat)
        self.btn_tasks = _tool_button("协作任务", "主 Agent 分派建模与检查子任务，候选在审阅应用前不修改原工程")
        self.btn_tasks.clicked.connect(self.open_task_dialog)
        self.btn_settings = _tool_button("连接设置", "配置模型地址、密钥与备用连接")
        self.btn_settings.clicked.connect(self._settings_menu)
        for button in (self.btn_history, self.btn_new_chat, self.btn_tasks, self.btn_settings):
            header.addWidget(button)
        layout.addLayout(header)

        # 第二行：当前模型。备用连接只在请求失败时才用，放进「连接设置」菜单，
        # 两个并排的下拉框读起来像两个必填项。
        self.model_selector = QComboBox()
        self.model_selector.setToolTip("切换模型保留对话历史与当前工程；运行期间不可切换")
        self.model_selector.setMinimumWidth(0)
        self.model_selector.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._populate_models()
        self.model_selector.currentIndexChanged.connect(self._switch_model)
        layout.addWidget(self.model_selector)
        # 备用连接与会话列表仍是真实的下拉数据源，只是不常驻界面
        self.backup_selector = QComboBox(self)
        self.backup_selector.setToolTip("仅在模型请求失败时启用所选连接；各提供商使用自己的密钥")
        self._populate_backup()
        self.backup_selector.currentIndexChanged.connect(self._switch_model)
        self.backup_selector.hide()
        self.chat_selector = QComboBox(self)
        self.chat_selector.addItem("新对话")
        self.chat_selector.currentIndexChanged.connect(self._switch_chat)
        self.chat_selector.hide()

        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setProperty("panel", "hint")
        layout.addWidget(self.lbl_status)

        self.view = QTextBrowser(self)
        self.view.setOpenExternalLinks(False)
        self.view.setOpenLinks(False)
        self.view.setObjectName("assistantConversation")
        self._show_welcome()
        layout.addWidget(self.view, 1)

        # 模型调用状态：没有内容时不占行
        self.lbl_call_status = QLabel("")
        self.lbl_call_status.setProperty("panel", "hint")
        self.lbl_call_status.setStyleSheet("font-size:8pt;")
        self.lbl_call_status.hide()
        layout.addWidget(self.lbl_call_status)

        suggestions = QHBoxLayout()
        suggestions.setContentsMargins(0, 0, 0, 0)
        suggestions.setSpacing(4)
        for label, prompt in (
                ("门式刚架", "创建一榀 24 米门式刚架，并说明还缺哪些工程参数"),
                ("检查模型", "检查当前模型的材料、约束、荷载和稳定性问题")):
            button = QPushButton(label, self)
            button.setProperty("role", "suggestion")
            button.setToolTip(prompt)
            button.clicked.connect(
                lambda _=False, value=prompt: self._use_suggestion(value))
            suggestions.addWidget(button)
        suggestions.addStretch(1)
        self.suggestions = QWidget(self)
        self.suggestions.setLayout(suggestions)
        layout.addWidget(self.suggestions)

        # 输入卡片：输入框在上，引用对象与发送在同一条底栏。引用是低频操作，
        # 用文字按钮即可，不该和「发送」一样是三个大按钮。
        composer = QFrame(self)
        composer.setObjectName("chatComposer")
        card = QVBoxLayout(composer)
        card.setContentsMargins(1, 1, 1, 1)
        card.setSpacing(0)
        self.entry = PromptEdit(composer)
        self.entry.setObjectName("chatEntry")
        self.entry.submitted.connect(self.send)
        card.addWidget(self.entry)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 6)
        bar.setSpacing(2)
        self.btn_objects = _tool_button("引用对象", "从模型中选择节点或杆件，随本轮请求发送")
        self.btn_objects.clicked.connect(self.pick_objects)
        self.btn_attach = _tool_button("加入所选", "把视口中当前选中的节点或杆件加入引用")
        self.btn_attach.setEnabled(False)
        self.btn_attach.clicked.connect(self.attach_selection)
        self.lbl_context = QLabel()
        self.lbl_context.setProperty("panel", "hint")
        self.lbl_context.setMinimumWidth(0)
        self.lbl_context.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.btn_clear = _tool_button("清除", "清除本轮引用的对象")
        self.btn_clear.clicked.connect(self.clear_references)
        self.btn_stop = QPushButton("停止", composer)
        self.btn_stop.setEnabled(False)
        self.btn_stop.hide()
        self.btn_stop.setToolTip("中止本轮后续工具调用。已执行的调用不会回滚，如需退回请使用撤销（Ctrl+Z）")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_send = QPushButton("发送", composer)
        self.btn_send.setProperty("role", "primary")
        self.btn_send.clicked.connect(self.send)
        for widget in (self.btn_objects, self.btn_attach):
            bar.addWidget(widget)
        bar.addSpacing(4)
        bar.addWidget(self.lbl_context, 1)
        for widget in (self.btn_clear, self.btn_stop, self.btn_send):
            bar.addWidget(widget)
        card.addLayout(bar)
        layout.addWidget(composer)

        self._refresh_status()
        self._refresh_references()
        self.runner.busy_changed.connect(self.set_task_busy)
        self._save_chat()

    def minimumSizeHint(self):
        return QSize(320, 240)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 小屏幕保留模型、输入与引用，辅助说明按可用空间收起。
        self.lbl_status.setVisible(self.height() >= 420)
        self.suggestions.setVisible(self.height() >= 420 and not self._has_messages())
        self.title_label.setVisible(self.width() >= 380)

    def _has_messages(self):
        return bool(self.conversation is not None and self.conversation.history)

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

    def _populate_models(self):
        import credentials
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItem("离线演示 · 确定性预设", "offline")
        if credentials.load_api_key():
            self.model_selector.addItem("DeepSeek · 现有密钥", "legacy")
        for p in self.profiles:
            suffix = "" if p.api_key else " · 请配置密钥"
            self.model_selector.addItem(f"{p.label} · {p.model}{suffix}", p.id)
        selected = self.selected_profile or ("legacy" if credentials.load_api_key() else "offline")
        self.model_selector.setCurrentIndex(max(0, self.model_selector.findData(selected)))
        self.model_selector.blockSignals(False)

    def _refresh_status(self):
        import credentials
        from model_profiles import redact
        selected = self.model_selector.currentData()
        key = ""
        if selected == "legacy":
            key = credentials.load_api_key() or ""
            description = f"DeepSeek · 密钥来源 {credentials.source()}"
        elif selected != "offline":
            profile = next((p for p in self.profiles if p.id == selected), None)
            key = profile.api_key if profile else ""
            description = f"{profile.label} · {profile.protocol}" if profile else "配置不存在"
        else:
            description = "离线演示：意图为预设脚本，工具调用与求解真实执行。自由对话请在「连接设置」中配置模型。"
        self._offline = selected == "offline"
        suffix = f" · 密钥指纹 {credentials.fingerprint(key)}" if key else ""
        self.lbl_status.setText(redact(description + suffix, [p.api_key for p in self.profiles]))
        if self._config_error:
            self.lbl_status.setText(self._config_error)

    def _populate_backup(self):
        previous = self.backup_selector.currentData()
        self.backup_selector.blockSignals(True)
        self.backup_selector.clear()
        self.backup_selector.addItem("不切换", "")
        for profile in self.profiles:
            self.backup_selector.addItem(f"备用 · {profile.label} · {profile.model}", profile.id)
        self.backup_selector.setCurrentIndex(max(0, self.backup_selector.findData(previous)))
        self.backup_selector.blockSignals(False)

    def _single_provider(self, selected):
        if selected == "offline":
            return demo_provider(self.session)
        if selected == "legacy":
            import credentials
            from agent import DeepSeekProvider
            return DeepSeekProvider(api_key=credentials.load_api_key())
        from configured_provider import create_provider
        profile = next((p for p in self.profiles if p.id == selected), None)
        if profile is None:
            raise ValueError("模型连接不存在，请打开模型连接设置")
        return create_provider(profile)

    def _make_provider(self):
        if self._config_error:
            raise ValueError(self._config_error)
        selected = self.model_selector.currentData()
        backup = self.backup_selector.currentData()
        if selected == "offline" or not backup or selected == backup:
            return self._single_provider(selected)
        from configured_provider import FallbackProvider
        import credentials
        return FallbackProvider([(self.model_selector.currentText(), lambda: self._single_provider(selected)),
                                 (self.backup_selector.currentText(), lambda: self._single_provider(backup))],
                                [p.api_key for p in self.profiles] + [credentials.load_api_key() or ""])

    def _ensure_conversation(self):
        from conversation import Conversation
        if self.conversation is None:
            self.conversation = Conversation(self._make_provider(), session=self.session)
        elif self._provider_dirty:
            replacement = self._make_provider()
            self._close_provider(self.conversation.provider)
            self.conversation.provider = replacement
        self._provider_dirty = False
        return self.conversation

    def _switch_model(self, _index):
        if self.runner.busy or self._chat_busy:
            return
        self.selected_profile = self.model_selector.currentData()
        if self.selected_profile == "offline":
            self._config_error = ""
        self._provider_dirty = True
        self._refresh_status()
        self.lbl_call_status.clear()
        self.lbl_call_status.hide()

    def open_settings(self):
        if self.runner.busy or self._chat_busy:
            return
        from .model_settings import ModelSettings
        dialog = ModelSettings(self.runner, self.profiles, self.selected_profile, self)
        dialog.profiles_changed.connect(self._profiles_changed)
        # 保留所有者，隐藏期间后台检测的回调仍可安全访问窗口。
        self._dialogs.append(dialog)
        dialog.show()

    def open_task_dialog(self):
        if self.runner.busy or self._chat_busy:
            return
        if self._task_dialog is None:
            from .agent_task_dialog import AgentTaskDialog
            self._task_dialog = AgentTaskDialog(self)
            self._dialogs.append(self._task_dialog)
        self._task_dialog.show()
        self._task_dialog.raise_()

    def _profiles_changed(self, profiles, selected):
        self.profiles, self.selected_profile = profiles, selected
        self._config_error = ""
        self._populate_models()
        self._populate_backup()
        self._provider_dirty = True
        self._refresh_status()

    def _save_chat(self):
        record = (self.conversation, self.view.toHtml(), self.entry.toPlainText(),
                  list(self.references), self.reference_revision)
        if self._current_chat < len(self._chats):
            self._chats[self._current_chat] = record
        else:
            self._chats.append(record)

    def _history_menu(self):
        if self.runner.busy or self._chat_busy:
            return
        menu = QMenu(self)
        for index in range(self.chat_selector.count()):
            action = menu.addAction(self.chat_selector.itemText(index))
            action.setCheckable(True)
            action.setChecked(index == self._current_chat)
            action.triggered.connect(lambda _checked, i=index: self.chat_selector.setCurrentIndex(i))
        menu.exec(self.btn_history.mapToGlobal(self.btn_history.rect().bottomLeft()))
        menu.deleteLater()

    def _settings_menu(self):
        if self.runner.busy or self._chat_busy:
            return
        menu = QMenu(self)
        menu.addAction("模型连接设置…", self.open_settings)
        backup = menu.addMenu("请求失败时切换到")
        for index in range(self.backup_selector.count()):
            action = backup.addAction(self.backup_selector.itemText(index).removeprefix("备用 · "))
            action.setCheckable(True)
            action.setChecked(index == self.backup_selector.currentIndex())
            action.triggered.connect(lambda _checked, i=index: self.backup_selector.setCurrentIndex(i))
        menu.exec(self.btn_settings.mapToGlobal(self.btn_settings.rect().bottomLeft()))
        menu.deleteLater()

    def new_chat(self):
        if self.runner.busy or self._chat_busy:
            return
        self._save_chat()
        self.chat_selector.addItem(f"新对话 {self.chat_selector.count() + 1}")
        self.chat_selector.setCurrentIndex(self.chat_selector.count() - 1)

    def _switch_chat(self, index):
        if self.runner.busy or self._chat_busy or index == self._current_chat:
            return
        self._save_chat()
        self._current_chat = index
        if index < len(self._chats):
            self.conversation, html, draft, self.references, self.reference_revision = self._chats[index]
            self.view.setHtml(html)
            self.entry.setPlainText(draft)
        else:
            self.conversation = None
            self.references, self.reference_revision = [], ""
            self.entry.clear()
            self._show_welcome()
        self._provider_dirty = True
        self._refresh_references()
        self.suggestions.setVisible(not self._has_messages() and self.height() >= 420)

    def reset_session(self, session):
        self.shutdown()
        self._task_dialog = None
        self.session = session
        self.conversation = None
        self._chats = []
        self._current_chat = 0
        self._selection = None
        self.chat_selector.blockSignals(True)
        self.chat_selector.clear()
        self.chat_selector.addItem("新对话")
        self.chat_selector.blockSignals(False)
        self.entry.clear()
        self._show_welcome()
        self.clear_references()
        self.suggestions.setVisible(self.height() >= 420)
        self._save_chat()

    def set_selection(self, kind, ident):
        self._selection = (kind, ident) if kind in ("node", "member") else None
        self.btn_attach.setEnabled(bool(self._selection) and not self.runner.busy)

    def attach_selection(self):
        if self.runner.busy or self._chat_busy or not self._selection:
            return
        from object_context import model_revision
        current = model_revision(self.session.model)
        if self.reference_revision != current:
            self.references = []
        if self._selection not in self.references:
            if len(self.references) >= 50:
                self._append_note("每轮最多引用 50 个对象，请清除部分引用后重试。")
                return
            self.references.append(self._selection)
        self.reference_revision = current
        self._refresh_references()

    def pick_objects(self):
        if self.runner.busy or self._chat_busy:
            return
        from .object_picker import ObjectPicker
        from object_context import model_revision
        revision = model_revision(self.session.model)
        refs = self.references if self.reference_revision == revision else []
        dialog = ObjectPicker(self.session.model, refs, self)
        dialog.locate.connect(self.locate_requested.emit)
        if dialog.exec():
            references = dialog.references()
            if len(references) > 50:
                self._append_note("每轮最多引用 50 个对象，请缩小选择范围。")
            else:
                self.references, self.reference_revision = references, revision
                self._refresh_references()
        dialog.deleteLater()

    def clear_references(self):
        self.references, self.reference_revision = [], ""
        self._refresh_references()

    def _refresh_references(self):
        from object_context import model_revision
        if not self.references:
            self.lbl_context.setText("未引用对象")
        elif self.reference_revision != model_revision(self.session.model):
            self.lbl_context.setText("引用已过期：模型已修改，请重新选择")
        else:
            names = [f"{'节点' if k == 'node' else '杆件'} {i}" for k, i in self.references]
            self.lbl_context.setText("、".join(names[:3]) +
                                     (f" 等 {len(names)} 项" if len(names) > 3 else ""))
        self.lbl_context.setToolTip(self.lbl_context.text())
        self.btn_clear.setVisible(bool(self.references))

    # --- 交互 ---

    def send(self) -> None:
        text = self.entry.toPlainText().strip()
        if not text or self.runner.busy:
            return
        try:
            from object_context import context_message
            context = context_message(self.session.model, self.references, self.reference_revision)
            conversation = self._ensure_conversation()
            conversation.object_context = context
        except Exception as exc:  # noqa: BLE001 配置错误也必须恢复输入并保护密钥
            self._failed(type(exc).__name__, str(exc))
            return
        self.entry.clear()
        self.suggestions.hide()
        self._append_user(text)
        if not conversation.history:
            self.chat_selector.setItemText(self._current_chat, text[:20])
        self._set_busy(True)

        before = self._model_signature()

        self._streamed = 0
        ok = self.runner.submit(
            lambda report: conversation.ask(text, on_tool=report),
            on_done=lambda r: self._replied(r, before),
            on_failed=self._failed,
            on_progress=self._tool_done)
        if not ok:                      # 理论上到不了，busy 已经拦过
            self._set_busy(False)

    def _tool_done(self, name: str, args, ok: bool) -> None:
        """一个工具跑完就显示一行。**这个回调在主线程里执行**——
        Runner 内部走信号投递，不是工作线程直接调过来的。"""
        self._streamed += 1
        self._append_tool(name, args, ok)

    def _replied(self, result, before) -> None:
        # 流式已经逐条显示过了就不要再重复一遍。
        # 联网失败退回离线、或者回调没走到时 `_streamed` 会是 0，那时补上
        if not self._streamed:
            for name, args in result.tool_calls:
                self._append_tool(name, args)
        self._append_agent(result.reply or "（本轮模型未返回文字说明）")
        if result.stopped_by_limit:
            self._append_note("本轮已达到工具调用上限，任务可能未完成。"
                              "建议将要求拆分为多次描述后重试。")
        self._set_busy(False)
        self.model_changed.emit(self._model_signature() != before)
        self._update_call_status(result)
        self._refresh_references()
        self._save_chat()

    def _update_call_status(self, result=None):
        """显示本轮端到端分解；模型慢和工具慢不再混成一个“正在运行”。"""
        if self.conversation is None:
            return
        parts = []
        metrics = getattr(result, "metrics", {}) if result is not None else {}
        if metrics:
            total = metrics.get("total_ms", 0.0) / 1000
            if metrics.get("fast_path"):
                parts.append(f"本轮 {total:.2f}s · 本地快速路径")
            else:
                model = metrics.get("provider_ms", 0.0) / 1000
                tool = metrics.get("tool_ms", 0.0) / 1000
                calls = int(metrics.get("provider_calls", 0))
                parts.append(
                    f"本轮 {total:.1f}s · 模型等待 {model:.1f}s · "
                    f"工具 {tool:.2f}s · {calls} 次模型往返")
        provider = getattr(self.conversation, "provider", None)
        if provider is None:
            # Conversation 可能把 provider 存在别的属性名
            provider = getattr(self.conversation, "_provider", None)
        from model_router import ModelRouter
        from configured_provider import FallbackProvider
        if isinstance(provider, (ModelRouter, FallbackProvider)) and provider.call_history:
            last = provider.call_history[-1]
            fallback = f" {glyphs.WARN}降级" if last.was_fallback else ""
            status = (glyphs.TICK if last.success else f"{glyphs.CROSS}{last.error[:30] if last.error else ''}")
            parts.append(f"最近模型：{last.model} {status}{fallback}")
        self.lbl_call_status.setText("　|　".join(parts))
        self.lbl_call_status.setVisible(bool(parts))

    def _failed(self, kind: str, message: str) -> None:
        """网络超时、密钥无效、脚本用尽……都走这里。

        **要把原因说出来**：只显示"失败了"，用户既不知道该重试还是该改配置，
        也没法判断是自己的问题还是程序的问题。
        """
        from model_profiles import redact
        import credentials
        message = redact(message, [p.api_key for p in self.profiles] + [credentials.load_api_key() or ""])
        hint = ""
        if "Timeout" in kind or "timed out" in message:
            hint = "　通常为网络原因，可重试；无网络时可使用离线演示模式。"
        elif "401" in message or "Unauthorized" in message:
            hint = "　请检查 API 密钥是否有效或已过期。"
        self._append_note(f"本轮请求失败：{kind}: {message[:300]}{hint}")
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
        self._chat_busy = busy
        self.set_task_busy(bool(busy or self.runner.busy))
        self.entry.setReadOnly(busy)
        self.btn_send.setText("思考中…" if busy else "发送")
        self.btn_stop.setEnabled(busy)
        self.btn_stop.setVisible(busy)
        self.busy_changed.emit(busy)

    def set_task_busy(self, busy):
        idle = not (busy or self._chat_busy)
        for widget in (self.model_selector, self.backup_selector, self.btn_settings, self.btn_tasks,
                       self.btn_history, self.chat_selector, self.btn_new_chat, self.btn_objects,
                       self.btn_clear, self.entry, self.btn_send):
            widget.setEnabled(idle)
        self.btn_attach.setEnabled(idle and bool(self._selection))

    def _model_signature(self):
        """粗粒度的模型指纹，用来判断这一轮有没有动过模型。

        比对整个模型字典太重，而且没必要——界面只需要知道"要不要重画"。
        """
        m = self.session.model
        return (len(m.get("nodes") or []), len(m.get("members") or []),
                len(m.get("supports") or []), len(m.get("load_cases") or []),
                len(self.session.history))

    def shutdown(self):
        """退出或换工程时关闭配置窗口与闲置网络连接。"""
        for dialog in self._dialogs:
            if hasattr(dialog, "shutdown"):
                dialog.shutdown()
            else:
                dialog.hide()
        conversations = [self.conversation] + [item[0] for item in self._chats]
        for conversation in conversations:
            self._close_provider(getattr(conversation, "provider", None))

    @staticmethod
    def _close_provider(provider):
        from configured_provider import FallbackProvider
        if isinstance(provider, FallbackProvider):
            provider.close()
        client = getattr(provider, "_client", None)
        if client is not None:
            client.close()

    def load_demo_prompt(self) -> None:
        self.entry.setPlainText(DEMO_PROMPT)

    # --- 渲染 ---

    def _html(self, html: str) -> None:
        self.view.append(html)
        self.view.moveCursor(QTextCursor.MoveOperation.End)

    def _append_user(self, text: str) -> None:
        self._html(f'<table width="100%" cellspacing="0" cellpadding="12" '
                   f'bgcolor="{theme.PANEL}"><tr><td><b>你</b><br>{_esc(text)}</td></tr></table><br>')

    def _append_tool(self, name: str, args: dict, ok: bool = True) -> None:
        """工具调用摊开显示——这是"数不是编的"最直接的证据。

        失败的调用也要显示，而且要标出来。**模型自己改正错误是常事**，
        藏起失败的那几步，用户就看不出它试过什么、为什么改了主意。
        """
        brief = _brief_args(args)
        colour = theme.INK_MUTED if ok else theme.WARN
        mark = glyphs.BUSY if ok else glyphs.CROSS
        from model_profiles import redact
        brief = redact(brief, [p.api_key for p in self.profiles])
        self._html(f'<p style="margin:1px 0 1px 18px;color:{colour};'
                   f'font-family:Consolas,monospace;font-size:11px;">'
                   f'{mark} {_esc(name)}　{_esc(brief)}</p>')

    def _append_agent(self, text: str) -> None:
        from model_profiles import redact
        text = redact(text, [p.api_key for p in self.profiles])
        from PySide6.QtGui import QTextDocument
        import re
        document = QTextDocument()
        document.setMarkdown(text, QTextDocument.MarkdownFeature.MarkdownDialectGitHub)
        html = re.sub(r'<img\b[^>]*>', '', document.toHtml(), flags=re.IGNORECASE)
        self._html(f'<p style="color:{theme.ACCENT};font-weight:600;">工程助手</p>' + html + '<br>')

    def _append_note(self, text: str) -> None:
        self._html(f'<p style="margin:4px 0;color:{theme.WARN};">'
                   f'{_esc(text)}</p>')


def _tool_button(text: str, tip: str) -> QToolButton:
    """面板内的低频操作：扁平文字按钮，悬停才显出边框。"""
    button = QToolButton()
    button.setText(text)
    button.setToolTip(tip)
    button.setProperty("chat", "tool")
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    return button


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("\n", "<br>"))


def _brief_args(args: dict, limit: int = 96) -> str:
    """参数摘要。整段 JSON 会把面板刷屏，而读者只想知道"传了什么量级"。"""
    if not args:
        return ""
    bits = []
    for key, value in args.items():
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], dict):
                bits.append(f"{key}=[{len(value)} 项]")
            else:
                bits.append(f"{key}={list(value)[:4]}"
                            + ("…" if len(value) > 4 else ""))
        elif isinstance(value, dict):
            bits.append(f"{key}={{{len(value)} 项}}")
        else:
            bits.append(f"{key}={value}")
    out = " ".join(bits)
    return out if len(out) <= limit else out[:limit - 1] + "…"
