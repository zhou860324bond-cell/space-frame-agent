"""Agent 对话面板。

这个项目的立论是"自然语言驱动结构计算"。桌面端只能点按钮的话，
最正式的那个界面恰恰否定了项目本身——所以这块面板不是锦上添花，
是**主张的兑现处**。

有一个设计决定值得说清楚：**工具调用要摆在明面上。**

```
你：24 米跨门式刚架，柱脚铰接，屋面恒载 8 kN/m
────────────────────────────────────────
    √ 生成门式刚架   spans=[24.0] eave_height=7.5   0.02 s
    √ 设置工况
    √ 求解                                           0.31 s
工程助手：已建好并求解……最大竖向位移 32.7 mm
数值依据：3 次成功的工具调用
```

因为整个项目最容易被质疑的一句是：**"这个数是不是大模型编的？"**
把调用链摊开，答案就是自明的——模型只决定调什么，数从求解器出来。

面板按主流 Agent 桌面端的方式组织：回答流式显示；每次工具调用是一张可展开的
卡片（参数、结果、耗时、定位到视口），连续几次收成一张写明调用链的卡片；多步
任务有计划清单；每轮改了模型就给出改动摘要，可以保留或整轮撤销；批量修改要在
卡片上确认；对话落盘，重启后在「对话记录」里找回；输入框支持 @ 引用杆件节点、
/ 命令和拖入图纸。

外观也按这一类软件来：顶栏只有模型名和四个图标按钮，记录区不加边框，用户
消息靠右成气泡，发送与停止是输入卡片右下角同一个位置的圆形按钮，Esc 停止、
空输入框里按 ↑ 取回上一条消息。

面板本身不算不画：`Conversation` 改的是同一个 `Session`，改完发信号，
主窗口去重画。所以对话和工具栏操作是**同一份模型**，不会各改各的。
"""

from __future__ import annotations

import re
import time
from copy import deepcopy

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, QStringListModel, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPen, QPolygonF, QTextCursor
from PySide6.QtWidgets import (QApplication, QComboBox, QCompleter, QFrame,
                               QHBoxLayout, QInputDialog, QLabel, QLayout,
                               QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QSizePolicy, QStackedWidget, QToolButton,
                               QVBoxLayout, QWidget)

from . import glyphs, icons, theme
from .chat_widgets import Avatar, TranscriptView, diff_lines, entry_text, tool_label

# 离线时用的脚本。**不是假装在跟大模型说话**——它照样把真工具跑一遍，
# 出来的数还是求解器算的。没有密钥、没有网的场合（答辩现场很常见）
# 靠它演示完整链路
from .demo_script import DEMO_PROMPT, demo_provider

# 斜杠命令：要么映射成对话里的确定性快捷语句（走 Conversation 的本地快速路径，
# 工具卡片照常显示），要么是面板自己的操作。
SLASH_COMMANDS = {
    "/求解": ("求解模型", "运行求解"),
    "/检查": ("检查模型", "校验模型能否求解"),
    "/位移": ("最大位移", "查询最大位移"),
    "/反力": ("查看反力", "查询支座反力"),
    "/下一步": ("下一步", "按当前状态给出下一步建议"),
    "/报告": ("生成计算书，并概述主要结论", "生成计算书"),
    "/新建": (None, "新建对话"),
    "/历史": (None, "打开对话记录"),
    "/帮助": (None, "列出全部命令"),
}
_MENTION = re.compile(r"@(节点|杆件)\s*(\d+)")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


class PromptEdit(QPlainTextEdit):
    """多行输入框，带 @ 引用与 / 命令补全。

    键位按聊天软件的通行约定，不自己发明：

        Enter / Ctrl+Enter   发送
        Shift+Enter          换行

    高度随内容长，但封顶——输入框吃掉半个面板的话，
    上面的对话记录就看不见了。
    """

    submitted = Signal()
    mention_chosen = Signal(str, int)
    # 空输入框里按 ↑：取回上一条发出的消息（不撤销那一轮，只是把文字放回来）
    recall_requested = Signal()

    MAX_LINES = 6
    MIN_LINES = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        # 占位文字要短；键位说明放进 tooltip
        self.setPlaceholderText("描述工程需求，@ 引用杆件，/ 使用命令…")
        self.setToolTip("Enter 发送　Shift+Enter 换行　↑ 取回上一条　@ 引用节点或杆件　/ 命令")
        self.setTabChangesFocus(True)
        self.document().documentLayout().documentSizeChanged.connect(
            lambda *_: self._fit_height())
        self._fit_height()
        self.objects_provider = lambda: []
        self._model = QStringListModel(self)
        self.completer = QCompleter(self._model, self)
        self.completer.setWidget(self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.completer.activated.connect(self._insert_completion)
        self._token_start = -1
        # 中文经输入法提交，走的是 inputMethodEvent 而不是 keyPressEvent；
        # 只在按键里刷新补全的话，键入「@杆」时列表不会跟着过滤。跟随文字变化刷新。
        self.textChanged.connect(self._text_changed)

    def _text_changed(self) -> None:
        if self.hasFocus():
            self._update_completer()

    def _fit_height(self) -> None:
        # 至少两行：一行高的输入框在输入卡片里显得局促，也容易误以为不能换行。
        # 额外高度要盖住样式表的上下内边距与文档边距，否则一行字也会冒出滚动条
        line = self.fontMetrics().lineSpacing()
        rows = min(max(self.MIN_LINES, int(self.document().size().height())), self.MAX_LINES)
        margin = 2 * self.document().documentMargin() + 2 * self.frameWidth() + 10
        self.setFixedHeight(int(line * rows + margin))

    # --- 与 QLineEdit 兼容的几个别名 ---

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

    # --- 补全 ---

    def _current_token(self) -> tuple[int, str]:
        cursor = self.textCursor()
        before = self.toPlainText()[:cursor.position()]
        match = re.search(r"(^|\s)([@/][^\s@/]*)$", before)
        if not match:
            return -1, ""
        token = match.group(2)
        if token.startswith("/") and before[:match.start(2)].strip():
            return -1, ""             # 命令只认行首
        return match.start(2), token

    def _update_completer(self) -> None:
        start, token = self._current_token()
        popup = self.completer.popup()
        if start < 0:
            popup.hide()
            return
        if token.startswith("@"):
            items = [f"@{name}" for name in self.objects_provider()]
        else:
            items = [f"{cmd}　{tip}" for cmd, (_, tip) in SLASH_COMMANDS.items()]
        if not items:
            popup.hide()
            return
        self._token_start = start
        self._model.setStringList(items)
        self.completer.setCompletionPrefix(token)
        if self.completer.completionCount() == 0:
            popup.hide()
            return
        rect = self.cursorRect()
        rect.setWidth(min(320, max(220, self.width() - rect.left())))
        self.completer.complete(rect)

    def _insert_completion(self, text: str) -> None:
        if self._token_start < 0:
            return
        value = text.split("　")[0]
        cursor = self.textCursor()
        cursor.setPosition(self._token_start)
        cursor.setPosition(self.textCursor().position(), QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(value + " ")
        self.setTextCursor(cursor)
        self._token_start = -1
        match = _MENTION.fullmatch(value)
        if match:
            self.mention_chosen.emit("node" if match.group(1) == "节点" else "member",
                                     int(match.group(2)))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        popup = self.completer.popup()
        if popup.isVisible() and event.key() in (
                Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape,
                Qt.Key.Key_Tab, Qt.Key.Key_Up, Qt.Key.Key_Down):
            event.ignore()             # 交给补全弹窗选择
            return
        enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if enter and not shift:
            self.submitted.emit()
            return
        if event.key() == Qt.Key.Key_Up and not self.toPlainText():
            self.recall_requested.emit()
            return
        super().keyPressEvent(event)


class ChatPanel(QWidget):
    """多轮对话面板。"""

    # 模型被改动了，主窗口据此重画；bool 表示这一轮是否动过模型
    model_changed = Signal(bool)
    busy_changed = Signal(bool)
    locate_requested = Signal(str, int)
    # 拖入了图纸图片：主窗口交给草图识别
    image_dropped = Signal(str)

    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.setObjectName("chatPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.session = session
        self.runner = runner
        self.conversation = None
        self._offline = False
        # 本轮已经流式显示了几条工具调用
        self._streamed = 0
        self._chat_busy = False
        self._provider_dirty = False
        self._dialogs = []
        self._task_dialog = None
        self.references = []
        self.reference_revision = ""
        self._selection = None
        self._turn = None              # 进行中回合的上下文
        self._text_buffer = ""
        self._think_buffer = ""
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(40)
        self._flush_timer.timeout.connect(self._flush_text)
        # 每秒刷新"已用时间"：大模型一次请求可能几十秒，看着秒数在走才知道没有卡住
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        from chat_store import ChatStore, new_record
        from model_profiles import ProfileStore
        self.store = ChatStore()
        self.record = new_record()
        # 本次运行中打开过的对话：id -> (conversation, record)
        self._open: dict[str, tuple] = {}
        self._config_error = ""
        try:
            self.profiles, self.selected_profile = ProfileStore().load()
        except ValueError as exc:
            self.profiles, self.selected_profile = [], ""
            self._config_error = str(exc)
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget(self)
        outer.addWidget(self.pages)
        chat_page = QWidget()
        self.pages.addWidget(chat_page)
        layout = QVBoxLayout(chat_page)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(6)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        # 顶栏：左边是当前模型（兼作标题），右边四个图标按钮。原来是「工程对话」
        # 大标题加四个文字按钮一行、整宽的模型下拉框再一行，读起来像一张表单。
        header = QHBoxLayout()
        header.setSpacing(2)
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("chatModel")
        self.model_selector.setToolTip("当前对话使用的模型。切换模型保留对话历史与当前工程；运行期间不可切换")
        self.model_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_selector.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.model_selector.setMinimumWidth(0)
        self._populate_models()
        self.model_selector.currentIndexChanged.connect(self._model_picked)
        header.addWidget(self.model_selector)
        header.addStretch(1)
        self.btn_history = _icon_button("history", "对话记录", "查看、搜索、重命名以前的对话；记录保存在本机")
        self.btn_history.clicked.connect(self.show_history)
        self.btn_new_chat = _icon_button("new_chat", "新建", "新建对话（当前对话已自动保存）")
        self.btn_new_chat.clicked.connect(self.new_chat)
        self.btn_tasks = _icon_button("agents", "协作任务",
                                      "主 Agent 分派建模与检查子任务，候选在审阅应用前不修改原工程")
        self.btn_tasks.clicked.connect(self.open_task_dialog)
        self.btn_settings = _icon_button("connection", "连接设置", "配置模型地址、密钥与备用连接")
        self.btn_settings.clicked.connect(self._settings_menu)
        for button in (self.btn_history, self.btn_new_chat, self.btn_tasks, self.btn_settings):
            header.addWidget(button)
        layout.addLayout(header)
        # 备用连接与本次运行打开过的对话仍是真实的下拉数据源，只是不常驻界面
        self.backup_selector = QComboBox(self)
        self.backup_selector.setToolTip("仅在模型请求失败时启用所选连接；各提供商使用自己的密钥")
        self._populate_backup()
        self.backup_selector.currentIndexChanged.connect(self._switch_model)
        self.backup_selector.hide()
        self.chat_selector = QComboBox(self)
        self.chat_selector.addItem("新对话", self.record["id"])
        self.chat_selector.currentIndexChanged.connect(self._switch_chat)
        self.chat_selector.hide()

        # 连接说明只留一行小字，放在输入卡片下面当脚注（主流对话软件提示
        # "回答可能有误"的位置），完整说明放在悬停提示里。
        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setProperty("chat", "footnote")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.view = TranscriptView(self)
        self.view.set_welcome(self._welcome())
        self.view.action.connect(self._entry_action)
        self.view.locate.connect(self.locate_requested)
        layout.addWidget(self.view, 1)

        self.lbl_call_status = QLabel("")
        self.lbl_call_status.setProperty("panel", "hint")
        self.lbl_call_status.setStyleSheet("font-size:8pt;")
        self.lbl_call_status.hide()
        layout.addWidget(self.lbl_call_status)

        # 后续建议：随模型阶段、待确认修改和引用对象变化，点一下直接发送
        self.suggestion_row = QHBoxLayout()
        self.suggestion_row.setContentsMargins(0, 0, 0, 0)
        self.suggestion_row.setSpacing(4)
        self.suggestions = QWidget(self)
        self.suggestions.setObjectName("chatSuggestions")
        self.suggestions.setLayout(self.suggestion_row)
        layout.addWidget(self.suggestions)
        self.followups = []

        # 输入卡片：输入框在上，引用对象与发送在同一条底栏
        composer = QFrame(self)
        composer.setObjectName("chatComposer")
        card = QVBoxLayout(composer)
        card.setContentsMargins(1, 1, 1, 1)
        card.setSpacing(0)
        self.entry = PromptEdit(composer)
        self.entry.setObjectName("chatEntry")
        self.entry.submitted.connect(self.send)
        self.entry.objects_provider = self._mention_names
        self.entry.mention_chosen.connect(self._add_reference)
        self.entry.recall_requested.connect(self._recall_last)
        self.entry.installEventFilter(self)
        self.composer = composer
        card.addWidget(self.entry)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 6)
        bar.setSpacing(4)
        # 引用入口是按钮：带图标和描边。原来与右侧"未引用对象"同为扁平灰字，
        # 三段并排看不出哪段能点；状态改成小号说明字，和按钮区分开。
        self.btn_objects = _tool_button("引用对象", "从模型中选择节点或杆件，随本轮请求发送；也可以在输入框里键入 @",
                                        "pick_member")
        self.btn_objects.clicked.connect(self.pick_objects)
        self.btn_attach = _tool_button("加入所选", "把视口中当前选中的节点或杆件加入引用",
                                       "pick_node")
        self.btn_attach.setEnabled(False)
        self.btn_attach.clicked.connect(self.attach_selection)
        for button in (self.btn_objects, self.btn_attach):
            button.setProperty("chat", "chip")
        self.lbl_context = QLabel()
        self.lbl_context.setProperty("chat", "caption")
        self.lbl_context.setMinimumWidth(0)
        self.lbl_context.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.btn_clear = _tool_button("清除", "清除本轮引用的对象")
        self.btn_clear.clicked.connect(self.clear_references)
        # 发送与停止在同一个位置：空闲时是蓝色上箭头，运行中换成方块停止键
        self.btn_stop = RoundButton("stop", composer)
        self.btn_stop.setEnabled(False)
        self.btn_stop.hide()
        self.btn_stop.setToolTip("停止（Esc）：中止本轮后续工具调用。已执行的调用不会回滚，可在改动卡片上撤销本轮")
        self.btn_stop.clicked.connect(self.stop)
        self.btn_send = RoundButton("send", composer)
        self.btn_send.setToolTip("发送（Enter）")
        self.btn_send.clicked.connect(self.send)
        for widget in (self.btn_objects, self.btn_attach):
            bar.addWidget(widget)
        bar.addSpacing(4)
        bar.addWidget(self.lbl_context, 1)
        for widget in (self.btn_clear, self.btn_stop, self.btn_send):
            bar.addWidget(widget)
        card.addLayout(bar)
        layout.addWidget(composer)
        layout.addWidget(self.lbl_status)

        self.pages.addWidget(self._build_history_page())

        self._refresh_status()
        self._refresh_references()
        self.runner.busy_changed.connect(self.set_task_busy)
        self._open[self.record["id"]] = (None, self.record)

    def minimumSizeHint(self):
        return QSize(320, 240)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 小屏幕保留模型、输入与引用，辅助说明按可用空间收起。
        self.lbl_status.setVisible(self.height() >= 420)
        self.suggestions.setVisible(self.height() >= 300 and bool(self.followups))
        # 模型名再长也给右侧四个图标留位
        self.model_selector.setMaximumWidth(max(120, self.width() - 4 * 30 - 28))

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.entry and event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            # Qt 样式表没有 :focus-within，输入框获得焦点时由这里给整张输入卡片描蓝边
            self.composer.setProperty("focused", event.type() == QEvent.Type.FocusIn)
            self.composer.style().unpolish(self.composer)
            self.composer.style().polish(self.composer)
        return super().eventFilter(watched, event)

    def _recall_last(self) -> None:
        last = next((e["text"] for e in reversed(self.record["entries"]) if e["type"] == "user"), "")
        if last:
            self.entry.setPlainText(last)
            self.entry.moveCursor(QTextCursor.MoveOperation.End)

    def _has_messages(self):
        return bool(self.view.entries)

    # --- 欢迎与状态 ---

    def _welcome(self) -> QWidget:
        """空对话时的欢迎区：在记录区里居中，头像、一句问候、三行用法。

        上一版是贴在记录区顶端的一张边框卡片，下面一大片空白；主流对话软件的
        空状态是居中的问候，开始输入后让位给消息。
        """
        from PySide6.QtWidgets import QGridLayout
        box = QFrame()
        box.setObjectName("chatWelcome")
        column = QVBoxLayout(box)
        column.setContentsMargins(8, 0, 8, 0)
        column.setSpacing(12)
        column.addStretch(3)
        column.addWidget(Avatar(40), 0, Qt.AlignmentFlag.AlignHCenter)
        title = QLabel("想建一个什么样的结构？")
        title.setProperty("chat", "welcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        column.addWidget(title)
        rows = QGridLayout()
        rows.setContentsMargins(0, 4, 0, 4)
        rows.setHorizontalSpacing(8)
        rows.setVerticalSpacing(6)
        for row, (name, text) in enumerate((
                ("chat", "描述结构几何、材料、约束和荷载，助手调用建模工具完成"),
                ("pick_member", "键入 @ 引用杆件或节点，键入 / 查看命令"),
                ("sketch_ai", "把图纸图片拖进来，交给草图识别"))):
            mark = QLabel()
            mark.setPixmap(icons.icon(name).pixmap(
                QSize(16, 16), QApplication.instance().devicePixelRatio()))
            mark.setAlignment(Qt.AlignmentFlag.AlignTop)
            rows.addWidget(mark, row, 0)
            label = QLabel(text)
            label.setWordWrap(True)
            label.setProperty("chat", "welcomeTip")
            rows.addWidget(label, row, 1)
        rows.setColumnStretch(1, 1)
        column.addLayout(rows)
        note = QLabel("每次工具调用都会列出参数与结果，数值全部来自确定性求解器。")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setProperty("chat", "caption")
        column.addWidget(note)
        column.addStretch(4)
        return box

    def _show_welcome(self) -> None:
        self.view.load([])

    def _use_suggestion(self, text: str) -> None:
        self.entry.setPlainText(text)
        self.entry.moveCursor(QTextCursor.MoveOperation.End)
        self.entry.setFocus()
        if text == "/":
            self.entry._update_completer()

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
        # 添加入口直接列在下拉框里：没配过模型时下拉框只有「离线演示」一项，
        # 用户不知道还能选别的，而入口藏在顶栏连接图标的菜单里。
        self.model_selector.insertSeparator(self.model_selector.count())
        from .model_settings import PROVIDERS
        for index, name in enumerate(PROVIDERS):
            self.model_selector.addItem(f"添加 {name} 连接…", f"add:{index}")
        if self.profiles:
            self.model_selector.addItem("管理模型连接…", "manage")
        selected = self.selected_profile or ("legacy" if credentials.load_api_key() else "offline")
        self.model_selector.setCurrentIndex(max(0, self.model_selector.findData(selected)))
        self.model_selector.blockSignals(False)

    def _model_picked(self, index):
        """下拉框里选了「添加…」「管理…」：退回原来的模型，打开连接设置。"""
        data = self.model_selector.itemData(index)
        if not (isinstance(data, str) and (data.startswith("add:") or data == "manage")):
            self._switch_model(index)
            return
        self.model_selector.blockSignals(True)
        self.model_selector.setCurrentIndex(max(0, self.model_selector.findData(
            self.selected_profile or "offline")))
        self.model_selector.blockSignals(False)
        self.open_settings(int(data[4:]) if data.startswith("add:") else None)

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
            description = "离线演示：意图为预设脚本，工具调用与求解真实执行"
        self._offline = selected == "offline"
        suffix = f" · 密钥指纹 {credentials.fingerprint(key)}" if key else ""
        self.lbl_status.setText(redact(description + suffix, self._secrets()))
        self.lbl_status.setToolTip("自由对话请在「连接设置」中配置模型。" if self._offline else "")
        if self._config_error:
            self.lbl_status.setText(self._config_error)

    def _secrets(self) -> list[str]:
        import credentials
        return [p.api_key for p in self.profiles] + [credentials.load_api_key() or ""]

    def _populate_backup(self):
        previous = self.backup_selector.currentData()
        self.backup_selector.blockSignals(True)
        self.backup_selector.clear()
        self.backup_selector.addItem("不切换", "")
        for profile in self.profiles:
            self.backup_selector.addItem(f"备用 · {profile.label} · {profile.model}", profile.id)
        self.backup_selector.setCurrentIndex(max(0, self.backup_selector.findData(previous)))
        self.backup_selector.blockSignals(False)

    # --- 模型连接 ---

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
        return FallbackProvider([(self.model_selector.currentText(), lambda: self._single_provider(selected)),
                                 (self.backup_selector.currentText(), lambda: self._single_provider(backup))],
                                self._secrets())

    def _ensure_conversation(self):
        from conversation import Conversation
        if self.conversation is None:
            self.conversation = Conversation(self._make_provider(), session=self.session, plan_tool=True)
            self.conversation.import_history(self.record.get("exchanges", []))
        elif self._provider_dirty:
            replacement = self._make_provider()
            self._close_provider(self.conversation.provider)
            self.conversation.provider = replacement
        self._provider_dirty = False
        self._open[self.record["id"]] = (self.conversation, self.record)
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

    def open_settings(self, preset: int | None = None):
        """``preset`` 给出提供商序号时，窗口直接以该提供商模板新建连接。"""
        if self.runner.busy or self._chat_busy:
            return
        from .model_settings import ModelSettings
        dialog = ModelSettings(self.runner, self.profiles, self.selected_profile, self, preset=preset)
        dialog.profiles_changed.connect(self._profiles_changed)
        # 保留所有者，隐藏期间后台检测的回调仍可安全访问窗口。
        self._dialogs.append(dialog)
        dialog.show()

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

    # --- 对话记录 ---

    def _build_history_page(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(12, 8, 12, 12)
        column.setSpacing(6)
        top = QHBoxLayout()
        back = _tool_button("‹ 返回对话", "回到当前对话")
        back.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        top.addWidget(back)
        top.addStretch(1)
        title = QLabel("对话记录")
        title.setObjectName("chatTitle")
        top.addWidget(title)
        column.addLayout(top)
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("搜索标题或消息内容…")
        self.history_search.setClearButtonEnabled(True)
        self.history_search.textChanged.connect(self._fill_history)
        column.addWidget(self.history_search)
        self.history_list = QListWidget()
        self.history_list.setObjectName("chatHistory")
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self._history_context)
        self.history_list.itemActivated.connect(
            lambda item: self.open_chat(item.data(Qt.ItemDataRole.UserRole)))
        column.addWidget(self.history_list, 1)
        hint = QLabel("双击打开；右键可重命名或删除。记录只保存在本机，不随工程文件分享。")
        hint.setWordWrap(True)
        hint.setProperty("panel", "hint")
        column.addWidget(hint)
        return page

    def show_history(self):
        if self.runner.busy or self._chat_busy:
            return
        self._save_chat()
        self.history_search.clear()
        self._fill_history()
        self.pages.setCurrentIndex(1)
        self.history_search.setFocus()

    def _fill_history(self, *_):
        import time
        self.history_list.clear()
        today = time.localtime()
        group = None
        for item in self.store.search(self.history_search.text()):
            label = day_group(time.localtime(item["updated"]), today)
            if label != group:
                group = label
                head = QListWidgetItem(label)
                head.setFlags(Qt.ItemFlag.NoItemFlags)
                font = head.font()
                font.setBold(True)
                head.setFont(font)
                head.setForeground(QColor(theme.INK_DIM))
                self.history_list.addItem(head)
            stamp = time.strftime("%m-%d %H:%M", time.localtime(item["updated"]))
            current = "　（当前）" if item["id"] == self.record["id"] else ""
            row = QListWidgetItem(f"{item['title']}{current}\n{stamp} · {item['messages']} 条消息 · {item['preview']}")
            row.setData(Qt.ItemDataRole.UserRole, item["id"])
            row.setToolTip(item["preview"])
            self.history_list.addItem(row)
        if not self.history_list.count():
            empty = QListWidgetItem("还没有保存的对话" if not self.history_search.text()
                                    else "没有匹配的对话")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.history_list.addItem(empty)

    def _history_context(self, point):
        item = self.history_list.itemAt(point)
        chat_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not chat_id:
            return
        menu = QMenu(self)
        menu.addAction("打开", lambda: self.open_chat(chat_id))
        menu.addAction("重命名…", lambda: self.rename_chat(chat_id))
        menu.addSeparator()
        menu.addAction("删除…", lambda: self.delete_chat(chat_id))
        menu.exec(self.history_list.viewport().mapToGlobal(point))
        menu.deleteLater()

    def rename_chat(self, chat_id, title=None):
        if title is None:
            current = next((i["title"] for i in self.store.list() if i["id"] == chat_id), "")
            title, ok = QInputDialog.getText(self, "重命名对话", "对话名称", text=current)
            if not ok:
                return
        try:
            self.store.rename(chat_id, title)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "重命名失败", f"{exc}。请换一个名称后重试。")
            return
        if chat_id == self.record["id"]:
            self.record["title"] = " ".join(str(title).split())[:40]
        self._sync_selector_titles()
        self._fill_history()

    def delete_chat(self, chat_id, confirm=True):
        if confirm and QMessageBox.question(
                self, "删除对话", "删除后无法恢复，工程模型不受影响。确定删除这段对话？"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.store.delete(chat_id)
        if chat_id in self._open:
            conversation, _ = self._open.pop(chat_id)
            self._close_provider(getattr(conversation, "provider", None))
            index = self.chat_selector.findData(chat_id)
            if chat_id == self.record["id"]:
                self._start_record()
            if index >= 0:
                self.chat_selector.blockSignals(True)
                self.chat_selector.removeItem(index)
                self.chat_selector.setCurrentIndex(self.chat_selector.findData(self.record["id"]))
                self.chat_selector.blockSignals(False)
        self._fill_history()

    def open_chat(self, chat_id):
        if not chat_id or self.runner.busy or self._chat_busy:
            return
        index = self.chat_selector.findData(chat_id)
        if index < 0:
            try:
                record = self.store.load(chat_id)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "无法打开对话", f"{exc}。文件可能已损坏，可在列表中删除它。")
                return
            self._open[chat_id] = (None, record)
            self.chat_selector.addItem(record.get("title") or "对话", chat_id)
            index = self.chat_selector.count() - 1
        self.pages.setCurrentIndex(0)
        if index == self.chat_selector.currentIndex():
            return
        self.chat_selector.setCurrentIndex(index)
        if not self.record.get("restored_note") and self.record.get("entries"):
            self.record["restored_note"] = True
            self._add_entry({"type": "note", "level": "info",
                             "text": "已恢复这段对话。继续追问时模型能看到前文，"
                                     "但当前工程模型可能与当时不同。"})

    def _sync_selector_titles(self):
        for i in range(self.chat_selector.count()):
            chat_id = self.chat_selector.itemData(i)
            record = self._open.get(chat_id, (None, None))[1]
            if record is not None:
                self.chat_selector.setItemText(i, record.get("title") or "新对话")

    def _save_chat(self):
        """有内容才落盘；空白的新对话不在列表里留一条空记录。"""
        self.record["draft"] = self.entry.toPlainText()
        self.record["references"] = [list(r) for r in self.references]
        self.record["reference_revision"] = self.reference_revision
        export = getattr(self.conversation, "export_history", None)
        if export is not None:
            self.record["exchanges"] = export()
        self._open[self.record["id"]] = (self.conversation, self.record)
        if not self.record["entries"]:
            return
        try:
            self.store.save(self.record, self._secrets())
        except OSError as exc:
            self.lbl_call_status.setText(f"对话未能保存：{exc}")
            self.lbl_call_status.show()

    def _start_record(self):
        from chat_store import new_record
        self.record = new_record()
        self.conversation = None
        self._open[self.record["id"]] = (None, self.record)
        self.references, self.reference_revision = [], ""
        self.entry.clear()
        self.view.load(self.record["entries"])
        self._refresh_references()

    def new_chat(self):
        if self.runner.busy or self._chat_busy:
            return
        self._save_chat()
        self.pages.setCurrentIndex(0)
        from chat_store import new_record
        record = new_record()
        self._open[record["id"]] = (None, record)
        self.chat_selector.addItem("新对话", record["id"])
        self.chat_selector.setCurrentIndex(self.chat_selector.count() - 1)

    def _switch_chat(self, index):
        chat_id = self.chat_selector.itemData(index)
        if self.runner.busy or self._chat_busy or chat_id == self.record["id"] or chat_id is None:
            return
        self._save_chat()
        conversation, record = self._open.get(chat_id, (None, None))
        if record is None:
            return
        self.record, self.conversation = record, conversation
        self.view.load(record["entries"])
        self.entry.setPlainText(record.get("draft", ""))
        self.references = [tuple(r) for r in record.get("references", [])]
        self.reference_revision = record.get("reference_revision", "")
        self._provider_dirty = True
        self._refresh_references()

    def reset_session(self, session):
        self._save_chat()
        self.shutdown()
        self._task_dialog = None
        self.session = session
        self._selection = None
        self._open = {}
        self._start_record()
        self.chat_selector.blockSignals(True)
        self.chat_selector.clear()
        self.chat_selector.addItem("新对话", self.record["id"])
        self.chat_selector.blockSignals(False)
        self.pages.setCurrentIndex(0)

    # --- 引用对象 ---

    def set_selection(self, kind, ident):
        self._selection = (kind, ident) if kind in ("node", "member") else None
        self.btn_attach.setEnabled(bool(self._selection) and not self.runner.busy)

    def _mention_names(self) -> list[str]:
        model = self.session.model
        names = [f"杆件{m['id']}" for m in model.get("members") or [] if isinstance(m, dict) and "id" in m]
        names += [f"节点{n['id']}" for n in model.get("nodes") or [] if isinstance(n, dict) and "id" in n]
        return names[:2000]

    def _add_reference(self, kind, ident) -> bool:
        from object_context import model_revision
        current = model_revision(self.session.model)
        if self.reference_revision != current:
            self.references = []
        if (kind, ident) not in self.references:
            if len(self.references) >= 50:
                self._append_note("每轮最多引用 50 个对象，请清除部分引用后重试。")
                return False
            self.references.append((kind, ident))
        self.reference_revision = current
        self._refresh_references()
        return True

    def attach_selection(self):
        if self.runner.busy or self._chat_busy or not self._selection:
            return
        self._add_reference(*self._selection)

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

    def _reference_names(self) -> list[str]:
        return [f"{'节点' if k == 'node' else '杆件'} {i}" for k, i in self.references]

    def _refresh_references(self):
        from object_context import model_revision
        if not self.references:
            self.lbl_context.setText("未引用对象")
        elif self.reference_revision != model_revision(self.session.model):
            self.lbl_context.setText("引用已过期：模型已修改，请重新选择")
        else:
            names = self._reference_names()
            self.lbl_context.setText("、".join(names[:3]) +
                                     (f" 等 {len(names)} 项" if len(names) > 3 else ""))
        self.lbl_context.setToolTip(self.lbl_context.text())
        self.btn_clear.setVisible(bool(self.references))
        self.refresh_suggestions()

    def refresh_suggestions(self) -> None:
        """按当前进度换一组后续建议；运行中不给建议，避免误点发出第二个请求。"""
        from followups import suggest
        from object_context import model_revision
        if self._chat_busy or self.runner.busy:
            items = []
        else:
            refs = self.references if self.reference_revision == model_revision(self.session.model) else []
            last = next((e['text'] for e in reversed(self.record['entries']) if e['type'] == 'user'), '')
            items = suggest(self.session, refs, last)
        if items != self.followups:
            self.followups = items
            while self.suggestion_row.count():
                item = self.suggestion_row.takeAt(0)
                if item.widget():
                    # 先藏起来再延迟删除：移出布局的旧按钮在真正删除前仍留在原位，
                    # 实拍里被新一组按钮压住，只露出半个字。
                    item.widget().hide()
                    item.widget().deleteLater()
            for followup in items:
                button = QPushButton(followup.label, self.suggestions)
                button.setProperty('role', 'suggestion')
                button.setToolTip(f'发送：{followup.prompt}')
                button.clicked.connect(lambda _=False, text=followup.prompt: self.send(text))
                self.suggestion_row.addWidget(button)
            self.suggestion_row.addStretch(1)
        self.suggestions.setVisible(bool(items) and self.height() >= 300)

    # --- 拖入图纸 ---

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(u.toLocalFile().lower().endswith(IMAGE_SUFFIXES) for u in urls):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(IMAGE_SUFFIXES):
                self.image_dropped.emit(path)
                self._append_note(f"已把图纸 {path.rsplit('/', 1)[-1]} 交给草图识别：请在识别面板里核对标注、"
                                  "标定尺度后确认加载，识别结果需要人工复核。", level="info")
                event.acceptProposedAction()
                return

    # --- 发送 ---

    def _slash(self, text: str) -> str | None:
        """处理 / 命令。返回要发给对话的文字；面板自己处理完则返回 None。"""
        command = text.split()[0]
        if command not in SLASH_COMMANDS:
            self._append_note(f"没有命令 {command}。键入 / 查看全部命令。")
            return None
        prompt, _ = SLASH_COMMANDS[command]
        if prompt:
            rest = text[len(command):].strip()
            return f"{prompt}{'：' + rest if rest else ''}"
        self.entry.clear()
        if command == "/新建":
            self.new_chat()
        elif command == "/历史":
            self.show_history()
        else:
            self._append_note("可用命令：" + "；".join(f"{c} {tip}" for c, (_, tip) in SLASH_COMMANDS.items()),
                              level="info")
        return None

    def send(self, text: str | None = None) -> None:
        typed = text is None
        text = (self.entry.toPlainText() if typed else text).strip()
        if not text or self.runner.busy or self._chat_busy:
            return
        if text.startswith("/"):
            text = self._slash(text)
            if text is None:
                return
        for word, ident in _MENTION.findall(text):
            if not self._add_reference("node" if word == "节点" else "member", int(ident)):
                return
        try:
            from object_context import context_message
            context = context_message(self.session.model, self.references, self.reference_revision)
            conversation = self._ensure_conversation()
            conversation.object_context = context
        except Exception as exc:  # noqa: BLE001 配置错误也必须恢复输入并保护密钥
            self._failed(type(exc).__name__, str(exc))
            return
        if typed:
            self.entry.clear()
        if not self.record["entries"]:
            self.record["title"] = text[:20]
            self._sync_selector_titles()
        self._add_entry({"type": "user", "text": text, "refs": self._reference_names()})
        index = self._add_entry({"type": "assistant", "segments": [], "plan": [],
                                 "status": "streaming", "activity": "思考中…"})
        self._turn = {"index": index, "user": text, "before": deepcopy(self.session.model),
                      "cursor": self.session.history.cursor, "signature": self._model_signature(),
                      "pending": (self.session.pending_change or {}).get("preview_id"),
                      "round_text": None, "round_thinking": None, "started": time.monotonic()}
        self._clock.start()
        self._set_busy(True)
        self._streamed = 0
        ok = self.runner.submit(
            lambda report: conversation.ask(
                text, on_event=lambda kind, data: report(kind, data, True)),
            on_done=self._replied,
            on_failed=self._failed,
            on_progress=self._on_event)
        if not ok:                      # 理论上到不了，busy 已经拦过
            self._set_busy(False)

    # --- 流式事件（主线程） ---

    def _assistant(self) -> dict | None:
        if self._turn is None:
            return None
        return self.record["entries"][self._turn["index"]]

    def _on_event(self, kind: str, data, _ok: bool) -> None:
        entry = self._assistant()
        if entry is None:
            return
        segments = entry["segments"]
        if kind == "text":
            if self._turn["round_text"] is None:
                segments.append({"kind": "text", "text": ""})
                self._turn["round_text"] = len(segments) - 1
            self._text_buffer += data["delta"]
            entry["activity"] = "正在回答…"
            if not self._flush_timer.isActive():
                self._flush_timer.start()
            return
        if kind == "reasoning":
            if self._turn["round_thinking"] is None:
                segments.append({"kind": "thinking", "text": ""})
                self._turn["round_thinking"] = len(segments) - 1
            self._think_buffer += data["delta"]
            entry["activity"] = "正在思考…"
            if not self._flush_timer.isActive():
                self._flush_timer.start()
            return
        self._flush_text()
        if kind == "round":
            self._turn["round_text"] = None
            self._turn["round_thinking"] = None
            entry["activity"] = "思考中…"
        elif kind == "tool_prepare":
            entry["activity"] = f"正在生成调用：{tool_label(data['name'])}…"
        elif kind == "tool_start":
            segments.append({"kind": "tool", "id": data.get("id", ""), "name": data["name"],
                             "args": data.get("args") or {}, "state": "running"})
            entry["activity"] = f"正在执行：{tool_label(data['name'])}"
        elif kind == "tool_end":
            seg = next((s for s in reversed(segments) if s["kind"] == "tool"
                        and s["name"] == data["name"] and s.get("state") == "running"), None)
            if seg is None:
                seg = {"kind": "tool", "name": data["name"], "args": data.get("args") or {}}
                segments.append(seg)
            seg.update(state="ok" if data["ok"] else "failed", ms=data.get("ms"),
                       summary=data.get("summary", ""))
            self._streamed += 1
            entry["activity"] = "思考中…"
        elif kind == "plan":
            entry["plan"] = data["steps"]
        self.view.refresh(self._turn["index"])

    def _tick(self) -> None:
        entry = self._assistant()
        if entry is None or entry.get("status") != "streaming":
            self._clock.stop()
            return
        entry["elapsed"] = int(time.monotonic() - self._turn["started"])
        self.view.refresh(self._turn["index"])

    def _flush_text(self) -> None:
        entry = self._assistant()
        if entry is None or not (self._text_buffer or self._think_buffer):
            return
        position = self._turn["round_text"]
        if position is not None and self._text_buffer:
            entry["segments"][position]["text"] += self._text_buffer
        thinking = self._turn["round_thinking"]
        if thinking is not None and self._think_buffer:
            seg = entry["segments"][thinking]
            # 思考过程只为让用户看到进展，落盘保留最后 4000 字即可
            seg["text"] = (seg["text"] + self._think_buffer)[-4000:]
        self._text_buffer = self._think_buffer = ""
        self.view.refresh(self._turn["index"])

    def _replied(self, result) -> None:
        self._flush_text()
        entry = self._assistant()
        reply = result.reply or "（本轮模型未返回文字说明）"
        from model_profiles import redact
        reply = redact(reply, self._secrets())
        if entry is not None:
            position = self._turn["round_text"]
            if position is None:
                entry["segments"].append({"kind": "text", "text": reply})
            elif entry["segments"][position]["text"].strip() != reply.strip():
                entry["segments"][position]["text"] = reply
            entry["status"] = "done"
            self._update_call_status(result)
            entry["metrics"] = self.lbl_call_status.text()
            self.view.refresh(self._turn["index"])
        if result.stopped_by_limit:
            self._append_note("本轮已达到工具调用上限或已停止，任务可能未完成。"
                              "建议将要求拆分为多次描述后重试。")
        self._review_changes()
        self._turn = None
        self._set_busy(False)
        self.model_changed.emit(self._model_signature() != self._last_signature)
        self.view.refresh_actions()
        self._refresh_references()
        self._save_chat()

    def _review_changes(self) -> None:
        """本轮改了模型就给出改动卡片；有待确认的批量修改就给出确认卡片。"""
        from change_preview import diff_models, model_digest
        turn = self._turn
        self._last_signature = turn["signature"]
        diff = diff_models(turn["before"], self.session.model)
        if diff["changed"]:
            self._add_entry({"type": "change", "title": "本轮修改了模型",
                             "lines": diff_lines(diff) or ["模型参数已修改"],
                             "cursor": turn["cursor"], "after": model_digest(self.session.model),
                             "status": "pending"})
        pending = self.session.pending_change
        if pending and pending.get("preview_id") != turn["pending"]:
            self._add_entry({"type": "approval",
                             "title": f"待确认：{tool_label(pending['tool'])}（预演 {pending['preview_id']}）",
                             "lines": diff_lines(pending.get("diff") or {}) or ["将修改已有对象"],
                             "preview_id": pending["preview_id"], "status": "pending"})

    def _entry_action(self, name: str, index: int) -> None:
        entries = self.record["entries"]
        if not 0 <= index < len(entries):
            return
        entry = entries[index]
        if name == "copy":
            QApplication.clipboard().setText(entry_text(entry))
            return
        if self.runner.busy or self._chat_busy:
            return
        if name == "keep":
            entry["status"] = "kept"
        elif name == "revert":
            entry["status"] = "reverted" if self._revert(entry) else "stale"
        elif name in ("approve", "reject"):
            pending = self.session.pending_change
            if not pending or pending.get("preview_id") != entry.get("preview_id"):
                entry["status"] = "expired"
            else:
                entry["status"] = "approved" if name == "approve" else "rejected"
                self.view.refresh(index)
                self.send(f"确认 {entry['preview_id']}" if name == "approve" else "取消")
                return
        elif name in ("regenerate", "edit"):
            self._rewind(index, name)
            return
        self.view.refresh(index)
        self._save_chat()

    def _revert(self, change: dict) -> bool:
        """整轮撤销：模型自本轮结束后没再动过才允许，否则会连带抹掉用户后来的手工修改。"""
        from change_preview import model_digest
        if model_digest(self.session.model) != change.get("after"):
            return False
        if not self.session.goto_step(change["cursor"]):
            return False
        self.model_changed.emit(True)
        self._refresh_references()
        return True

    def _rewind(self, index: int, mode: str) -> None:
        """重新生成 / 编辑重发：撤回最后一轮（含其模型改动），再重问或放回输入框。"""
        entries = self.record["entries"]
        user_index = max((i for i in range(len(entries)) if entries[i]["type"] == "user"
                          and i <= index), default=None)
        if user_index is None:
            return
        for change in (e for e in entries[user_index:] if e["type"] == "change"
                       and e.get("status") == "pending"):
            if not self._revert(change):
                self._append_note("这一轮之后模型又被修改过，不能直接重来。请先用 Ctrl+Z 退回，"
                                  "或在新的消息里说明要怎么改。")
                return
        text = entries[user_index]["text"]
        del entries[user_index:]
        if self.conversation is not None:
            self.conversation.drop_last_exchange()
        self.view.load(entries)
        self._save_chat()
        if mode == "regenerate":
            self.send(text)
        else:
            self.entry.setPlainText(text)
            self.entry.moveCursor(QTextCursor.MoveOperation.End)
            self.entry.setFocus()

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
        from model_router import ModelRouter
        from configured_provider import FallbackProvider
        if isinstance(provider, (ModelRouter, FallbackProvider)) and provider.call_history:
            last = provider.call_history[-1]
            fallback = f" {glyphs.WARN}降级" if last.was_fallback else ""
            status = (glyphs.TICK if last.success else f"{glyphs.CROSS}{last.error[:30] if last.error else ''}")
            parts.append(f"最近模型：{last.model} {status}{fallback}")
        self.lbl_call_status.setText("　|　".join(parts))

    def _failed(self, kind: str, message: str) -> None:
        """网络超时、密钥无效、脚本用尽……都走这里。**要把原因说出来。**"""
        from model_profiles import redact
        message = redact(message, self._secrets())
        hint = ""
        if "Timeout" in kind or "timed out" in message:
            hint = "　通常为网络原因，可重试；无网络时可使用离线演示模式。"
        elif "401" in message or "Unauthorized" in message:
            hint = "　请检查 API 密钥是否有效或已过期。"
        self._flush_text()
        entry = self._assistant()
        if entry is not None:
            entry["status"] = "done"
            self.view.refresh(self._turn["index"])
            self._review_changes()
            self._turn = None
            self.model_changed.emit(self._model_signature() != self._last_signature)
        self._append_note(f"本轮请求失败：{kind}: {message[:300]}{hint}")
        self._set_busy(False)
        self.view.refresh_actions()
        self._save_chat()

    def stop(self) -> None:
        """放弃这一轮剩下的工具调用。**不回滚已经执行的调用**；改动卡片上可以整轮撤销。"""
        if self.conversation is not None:
            self.conversation.cancel()
        entry = self._assistant()
        if entry is not None:
            entry["activity"] = "正在停止，当前这一步完成后结束…"
            self.view.refresh(self._turn["index"])
        self.btn_stop.setEnabled(False)

    def _set_busy(self, busy: bool) -> None:
        refocus = self.entry.hasFocus() or self.btn_stop.hasFocus()
        self._chat_busy = busy
        self.set_task_busy(bool(busy or self.runner.busy))
        self.entry.setReadOnly(busy)
        self.btn_stop.setEnabled(busy)
        self.btn_stop.setVisible(busy)
        self.btn_send.setVisible(not busy)
        # 焦点交给停止键，按 Esc 就能停；结束后回到输入框，接着打下一句
        if refocus:
            (self.btn_stop if busy else self.entry).setFocus()
        self.busy_changed.emit(busy)

    def set_task_busy(self, busy):
        idle = not (busy or self._chat_busy)
        for widget in (self.model_selector, self.backup_selector, self.btn_settings, self.btn_tasks,
                       self.btn_history, self.chat_selector, self.btn_new_chat, self.btn_objects,
                       self.btn_clear, self.entry, self.btn_send):
            widget.setEnabled(idle)
        self.btn_attach.setEnabled(idle and bool(self._selection))
        if hasattr(self, 'followups'):
            self.refresh_suggestions()

    def _model_signature(self):
        """粗粒度的模型指纹，用来判断这一轮有没有动过模型。"""
        m = self.session.model
        return (len(m.get("nodes") or []), len(m.get("members") or []),
                len(m.get("supports") or []), len(m.get("load_cases") or []),
                len(self.session.history))

    _last_signature = None

    def shutdown(self):
        """退出或换工程时保存对话，关闭配置窗口与闲置网络连接。"""
        try:
            self._save_chat()
        except Exception:  # noqa: BLE001 退出时保存失败不能阻止关闭
            pass
        for dialog in self._dialogs:
            if hasattr(dialog, "shutdown"):
                dialog.shutdown()
            else:
                dialog.hide()
        for conversation, _ in list(self._open.values()):
            self._close_provider(getattr(conversation, "provider", None))
        self._close_provider(getattr(self.conversation, "provider", None))

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

    # --- 记录 ---

    def _add_entry(self, entry: dict) -> int:
        # 视图与当前对话共用同一个 entries 列表，追加一次即可
        if self.view.entries is not self.record["entries"]:
            self.view.load(self.record["entries"])
        return self.view.append(entry)

    def _append_user(self, text: str) -> None:
        self._add_entry({"type": "user", "text": text, "refs": []})

    def _append_agent(self, text: str) -> None:
        from model_profiles import redact
        self._add_entry({"type": "assistant", "status": "done", "plan": [],
                         "segments": [{"kind": "text", "text": redact(text, self._secrets())}]})

    def _append_tool(self, name: str, args: dict, ok: bool = True) -> None:
        self._add_entry({"type": "assistant", "status": "done", "plan": [], "segments": [
            {"kind": "tool", "name": name, "args": args, "state": "ok" if ok else "failed"}]})

    def _append_note(self, text: str, level: str = "warn") -> None:
        self._add_entry({"type": "note", "level": level, "text": text})


def day_group(stamp, today) -> str:
    """对话记录按更新日期分组：今天、昨天、近 7 天、更早。"""
    import time
    midnight = (lambda t: time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1)))
    days = round((midnight(today) - midnight(stamp)) / 86400)
    if days <= 0:
        return "今天"
    if days == 1:
        return "昨天"
    return "近 7 天" if days < 7 else "更早"


def _icon_button(icon: str, name: str, tip: str) -> QToolButton:
    """顶栏的图标按钮：只显示图标，名称和用途写在悬停提示里。"""
    button = QToolButton()
    button.setText(name)
    button.setIcon(icons.icon(icon))
    button.setIconSize(QSize(18, 18))
    button.setToolTip(f"{name}：{tip}")
    button.setProperty("chat", "icon")
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class RoundButton(QPushButton):
    """输入卡片右下角的圆形按钮：``send`` 是蓝底白色上箭头，``stop`` 是描边圆里一个方块。

    自己画而不用文字：「发送」「思考中…」宽度不同，运行时按钮变宽，把引用
    状态挤走；图形按钮尺寸固定。停止键获得焦点时按 Esc 即停止。
    """

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("chatSend" if kind == "send" else "chatStop")
        self.setText("发送" if kind == "send" else "停止")
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def event(self, event):
        # 主窗口也有 Esc 快捷键；停止键有焦点时先把 Esc 留给自己
        if (self.kind == "stop" and event.type() == QEvent.Type.ShortcutOverride
                and event.key() == Qt.Key.Key_Escape):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event):  # noqa: N802
        if self.kind == "stop" and event.key() == Qt.Key.Key_Escape:
            if self.isEnabled():
                self.click()
            return
        super().keyPressEvent(event)

    def enterEvent(self, event):  # noqa: N802
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):  # noqa: N802
        super().leaveEvent(event)
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(1, 1, self.width() - 2, self.height() - 2)
        enabled = self.isEnabled()
        hover = enabled and self.underMouse()
        c = rect.center()
        if self.kind == "send":
            g.setPen(Qt.PenStyle.NoPen)
            fill = (theme.ACCENT if hover else theme.ACCENT_DIM) if enabled else theme.PANEL_HOVER
            g.setBrush(QColor(fill))
            g.drawEllipse(rect)
            pen = QPen(QColor(theme.ON_ACCENT if enabled else theme.INK_DIM), 2.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            g.setPen(pen)
            g.drawLine(QPointF(c.x(), c.y() + 6), QPointF(c.x(), c.y() - 6))
            g.drawPolyline(QPolygonF([QPointF(c.x() - 5, c.y() - 1), QPointF(c.x(), c.y() - 6),
                                      QPointF(c.x() + 5, c.y() - 1)]))
        else:
            g.setPen(QPen(QColor(theme.ACCENT if hover else theme.BORDER_LIGHT), 1.5))
            g.setBrush(QColor(theme.SELECTION if hover else theme.SURFACE))
            g.drawEllipse(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            g.setPen(Qt.PenStyle.NoPen)
            g.setBrush(QColor(theme.INK if enabled else theme.INK_DIM))
            g.drawRoundedRect(QRectF(c.x() - 5, c.y() - 5, 10, 10), 2, 2)
        if self.hasFocus():
            g.setPen(QPen(QColor(theme.ACCENT), 1.2, Qt.PenStyle.DotLine))
            g.setBrush(Qt.BrushStyle.NoBrush)
            g.drawEllipse(rect)
        g.end()


def _tool_button(text: str, tip: str, icon: str = "") -> QToolButton:
    """面板内的低频操作：扁平文字按钮，悬停才显出边框；给了图标就图文并排。"""
    button = QToolButton()
    button.setText(text)
    button.setToolTip(tip)
    button.setProperty("chat", "tool")
    if icon:
        button.setIcon(icons.icon(icon))
        button.setIconSize(QSize(16, 16))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    else:
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    return button
