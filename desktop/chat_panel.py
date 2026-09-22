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

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QTextBrowser,
                               QVBoxLayout, QWidget)

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setProperty("panel", "hint")
        layout.addWidget(self.lbl_status)

        # 多模型降级配置
        from PySide6.QtWidgets import QCheckBox, QLineEdit
        self.router_row = QHBoxLayout()
        self.router_row.setContentsMargins(0, 0, 0, 0)
        self.chk_router = QCheckBox("备用模型降级")
        self.chk_router.setToolTip("主模型失败时自动切换到备用模型")
        self.chk_router.toggled.connect(self._toggle_router_config)
        self.router_row.addWidget(self.chk_router)
        self.router_row.addStretch()
        layout.addLayout(self.router_row)

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

        # 模型调用状态
        self.lbl_call_status = QLabel("")
        self.lbl_call_status.setStyleSheet(f"color:{theme.INK_MUTED}; font-size:8pt;")
        layout.addWidget(self.lbl_call_status)

        self.view = QTextBrowser(self)
        self.view.setOpenExternalLinks(False)
        self.view.setObjectName("assistantConversation")
        self._show_welcome()
        layout.addWidget(self.view, 1)

        suggestions = QHBoxLayout()
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

    def _refresh_status(self) -> None:
        """说清楚现在接的是谁。**任何情况下都不显示密钥本身**，
        只显示来源和指纹——够回答"是不是换了把钥匙"，且不可还原。"""
        import credentials

        key = credentials.load_api_key()
        if key:
            self._offline = False
            self.lbl_status.setText(
                f"大模型：DeepSeek　密钥来源 {credentials.source()}"
                f"　指纹 {credentials.fingerprint(key)}")
        else:
            self._offline = True
            self.lbl_status.setText(
                "未配置 API 密钥，当前为离线演示模式："
                "对话按预设意图分派，但工具调用与求解均为实际执行。\n"
                f"要开启自由对话：新建文本文件 {credentials.where_to_put_it()}，"
                "里面只写你自己的 API key 一行，存成 UTF-8，然后重开程序。")

    def _toggle_router_config(self, checked):
        self.router_panel.setVisible(checked)
        # 配置变了，下次发送时重建 conversation
        self.conversation = None

    def _ensure_conversation(self):
        if self.conversation is not None:
            return self.conversation
        from conversation import Conversation

        import credentials
        key = credentials.load_api_key()
        if key:
            if self.chk_router.isChecked():
                from model_router import ModelConfig, ModelRouter
                configs = [
                    ModelConfig(name="deepseek-v4-flash",
                                base_url="https://api.deepseek.com",
                                api_key=key, priority=1),
                    ModelConfig(name=self.txt_bk_model.text().strip(),
                                base_url=self.txt_bk_url.text().strip(),
                                api_key=self.txt_bk_key.text().strip() or key,
                                priority=2),
                ]
                provider = ModelRouter(configs)
            else:
                from agent import DeepSeekProvider
                provider = DeepSeekProvider(api_key=key)
        else:
            provider = demo_provider(self.session)
        self.conversation = Conversation(provider, session=self.session)
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
        if isinstance(provider, ModelRouter) and provider.call_history:
            last = provider.call_history[-1]
            fallback = f" {glyphs.WARN}降级" if last.was_fallback else ""
            status = (glyphs.TICK if last.success else f"{glyphs.CROSS}{last.error[:30] if last.error else ''}")
            parts.append(f"最近模型：{last.model} {status}{fallback}")
        self.lbl_call_status.setText("　|　".join(parts))

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
        self.btn_send.setText("思考中…" if busy else "发送")
        self.btn_stop.setEnabled(busy)
        self.busy_changed.emit(busy)

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
        self._html(f'<p style="margin:10px 0 2px 0;color:{theme.INK};">'
                   f'<b>你</b>　{_esc(text)}</p>')

    def _append_tool(self, name: str, args: dict, ok: bool = True) -> None:
        """工具调用摊开显示——这是"数不是编的"最直接的证据。

        失败的调用也要显示，而且要标出来。**模型自己改正错误是常事**，
        藏起失败的那几步，用户就看不出它试过什么、为什么改了主意。
        """
        brief = _brief_args(args)
        colour = theme.INK_MUTED if ok else theme.WARN
        mark = glyphs.BUSY if ok else glyphs.CROSS
        self._html(f'<p style="margin:1px 0 1px 18px;color:{colour};'
                   f'font-family:Consolas,monospace;font-size:11px;">'
                   f'{mark} {_esc(name)}　{_esc(brief)}</p>')

    def _append_agent(self, text: str) -> None:
        self._html(f'<p style="margin:2px 0 10px 0;color:{theme.INK};">'
                   f'<b style="color:{theme.ACCENT};">Agent</b>　'
                   f'{_esc(text)}</p>')

    def _append_note(self, text: str) -> None:
        self._html(f'<p style="margin:4px 0;color:{theme.WARN};">'
                   f'{_esc(text)}</p>')


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
