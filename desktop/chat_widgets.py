"""对话记录的结构化显示：消息、工具调用卡片、计划清单、改动审阅与确认卡片。

原来的对话区是一个 QTextBrowser，每条消息拼一段 HTML 追加进去：工具调用只剩
一行灰字，看不到参数和结果；也没法在某条消息上放「撤销本轮」「重新生成」这类
按钮。这里改成一列独立控件，每条记录对应 ``entries`` 里的一个字典——界面从数据
重绘，数据原样写进 ``chat_store``，重启后看到的和当时一模一样。

记录类型（``entry["type"]``）：
- ``user``       用户消息，``text``、``refs``（引用的对象名）
- ``assistant``  助手回合，``segments``（按发生顺序的文字段与工具卡片）、
                 ``plan``、``status``、``metrics``
- ``change``     本轮对模型的改动摘要，可保留或撤销
- ``approval``   待用户确认的批量修改（Session 的预演闸门）
- ``note``       提示，``level`` 为 ``info`` / ``warn``
"""
from __future__ import annotations

import json
import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)


TOOL_LABELS = {
    "define_materials_and_sections": "定义材料与截面", "generate_frame": "生成规则框架",
    "edit_member": "修改杆件", "edit_node": "修改节点", "define_set": "定义集合",
    "list_sets": "列出集合", "generate_bent": "生成排架", "extrude_bents": "拉伸排架",
    "select_members": "选择杆件", "add_bracing": "添加支撑", "raise_nodes": "抬升节点",
    "retaper": "调整变截面", "generate_portal_frame": "生成门式刚架", "add_nodes": "添加节点",
    "add_members": "添加杆件", "set_supports": "设置支座",
    "list_boundary_conditions": "列出边界条件", "delete_boundary_condition": "删除边界条件",
    "assign_properties": "指派截面", "remove_members": "删除杆件", "set_model": "整体替换模型",
    "add_load_case": "新建工况", "set_nodal_load": "施加节点荷载", "set_member_load": "施加杆件荷载",
    "set_member_span_load": "施加跨间荷载", "set_prescribed_displacement": "给定位移",
    "set_member_strain": "施加杆件应变", "set_load_cases": "设置工况",
    "export_learning_trace": "导出建模过程", "set_units": "设置单位",
    "generate_live_patterns": "生成活载布置", "generate_combinations": "生成荷载组合",
    "apply_area_load": "施加面荷载", "add_self_weight": "添加自重", "validate_model": "校验模型",
    "preview_change": "预演修改", "apply_preview": "应用已确认的修改",
    "preview_analysis_mesh": "预览分析网格", "response_spectrum_analysis": "反应谱分析",
    "set_member_connection": "设置杆端连接", "add_step": "添加分析步", "list_steps": "列出分析步",
    "delete_step": "删除分析步", "solve_steps": "按分析步求解", "define_amplitude": "定义幅值",
    "list_amplitudes": "列出幅值", "delete_amplitude": "删除幅值", "solve_model": "求解",
    "solve_with_abaqus": "Abaqus 求解", "analyze_joint_solid": "节点实体分析",
    "compare_solvers": "求解器对比", "query_results": "查询结果", "plot_results": "绘制结果",
    "query_diagram": "查询内力图", "sweep": "参数扫描", "query_envelope": "查询包络",
    "modal_analysis": "模态分析", "buckling_analysis": "屈曲分析", "write_report": "生成计算书",
    "check_strength": "强度验算", "check_symmetry": "对称性检查", "check_numbering": "编号检查",
    "diagnose_supports": "支座诊断",
}

COLLECTIONS = {"nodes": "节点", "members": "杆件", "materials": "材料", "sections": "截面",
               "supports": "支座", "load_cases": "工况", "combos": "组合", "sets": "集合",
               "units": "单位制", "steps": "分析步", "amplitudes": "幅值"}
# 模型文件的内部字段，变了对用户没有意义
_HIDDEN_FIELDS = {"schema_version"}

PLAN_MARK = {"pending": "○", "in_progress": "◐", "done": "✓"}


_BOLD = re.compile(r"\*\*([^*\n]+?)\*\*")
_CODE = re.compile(r"(`[^`\n]*`)")
_IMG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)


def markdown_ready(text: str) -> str:
    """修正中文 Markdown 的加粗。

    CommonMark 规定：结束的 ``**`` 前面是标点时，后面必须是空白或标点才算闭合。
    中文里「**结论。**后面接正文」很常见，于是界面上露出两对星号。这里把
    ``**…**`` 直接换成 ``<b>…</b>``（Qt 的 Markdown 支持内联 HTML），行内代码不动；
    顺带去掉图片标签，模型回答不应从网络加载图片。
    """
    parts = _CODE.split(_IMG.sub("", text))
    return "".join(p if p.startswith("`") else _BOLD.sub(r"<b>\1</b>", p) for p in parts)


def tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, name)


def diff_lines(diff: dict) -> list[str]:
    """把 ``change_preview.diff_models`` 的结果写成几行中文。"""
    lines = []
    for key, change in (diff.get("collections") or {}).items():
        parts = [f"{change['before']} → {change['after']}"]
        for label, field in (("新增", "added"), ("删除", "removed"), ("修改", "changed")):
            ids = [str(i) for i in change.get(field) or []]
            if ids:
                parts.append(f"{label} {'、'.join(ids[:10])}" + (" 等" if len(ids) > 10 else ""))
        lines.append(f"{COLLECTIONS.get(key, key)}：" + "；".join(parts))
    for key in (diff.get("fields") or {}):
        if key not in _HIDDEN_FIELDS:
            lines.append(f"{COLLECTIONS.get(key, key)}：已修改")
    return lines


def _round(value):
    """显示用：浮点取 6 位有效数字，0.014679999999999999 写成 0.01468。"""
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def format_args(args: dict, per_key: int = 240) -> str:
    """参数逐项一行；过长的值截断，完整参数在审计记录里。"""
    if not args:
        return "（无）"
    lines = []
    for key, value in args.items():
        text = json.dumps(_round(value), ensure_ascii=False, separators=(", ", ": "))
        lines.append(f"{key}: {text if len(text) <= per_key else text[:per_key - 1] + '…'}")
    return "\n".join(lines)


def brief_args(args: dict, limit: int = 80) -> str:
    """参数摘要。整段 JSON 会把卡片标题撑爆，标题只放量级。"""
    if not args:
        return ""
    bits = []
    for key, value in args.items():
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], dict):
                bits.append(f"{key}=[{len(value)} 项]")
            else:
                bits.append(f"{key}={list(value)[:4]}" + ("…" if len(value) > 4 else ""))
        elif isinstance(value, dict):
            bits.append(f"{key}={{{len(value)} 项}}")
        else:
            bits.append(f"{key}={value}")
    out = " ".join(bits)
    return out if len(out) <= limit else out[:limit - 1] + "…"


def object_refs(args: dict) -> list[tuple[str, int]]:
    """从工具参数里找出可以在视口定位的节点、杆件编号（最多 6 个）。"""
    found: list[tuple[str, int]] = []

    def add(kind, value):
        if isinstance(value, bool):
            return
        if isinstance(value, int) and (kind, value) not in found:
            found.append((kind, value))
        elif isinstance(value, list):
            for item in value[:6]:
                add(kind, item)

    for key, value in (args or {}).items():
        lowered = key.lower()
        if "member" in lowered:
            add("member", value)
        elif "node" in lowered:
            add("node", value)
    return found[:6]


def entry_text(entry: dict) -> str:
    """一条记录的纯文字，用于复制、搜索和测试断言。"""
    kind = entry.get("type")
    if kind == "user":
        return entry.get("text", "")
    if kind == "assistant":
        parts = []
        if entry.get("plan"):
            parts += [f"{PLAN_MARK.get(s['status'], '○')} {s['title']}" for s in entry["plan"]]
        for seg in entry.get("segments", []):
            if seg["kind"] == "text":
                parts.append(seg["text"])
            elif seg["kind"] == "thinking":
                continue            # 思考过程不进复制与检索的正文
            else:
                parts.append(f"{tool_label(seg['name'])} {seg['name']} {brief_args(seg.get('args') or {})}")
        return "\n".join(p for p in parts if p)
    if kind in ("change", "approval"):
        return "\n".join([entry.get("title", "")] + list(entry.get("lines", [])))
    return entry.get("text", "")


def _label(text: str = "", role: str = "", rich: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setTextFormat(Qt.TextFormat.MarkdownText if rich else Qt.TextFormat.PlainText)
    if role:
        label.setProperty("chat", role)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(0)
    return label


def _link_button(text: str, tip: str = "") -> QToolButton:
    button = QToolButton()
    button.setText(text)
    button.setToolTip(tip)
    button.setProperty("chat", "tool")
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    return button


class ToolCard(QFrame):
    """一次工具调用。标题行一眼看清做了什么、成没成、用了多久；展开看参数与结果。"""

    locate = Signal(str, int)

    def __init__(self, segment: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("toolCard")
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.header = QToolButton(self)
        self.header.setObjectName("toolCardHeader")
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.ArrowType.RightArrow)
        # 标题文字可能很长；不让它的理想宽度撑宽整列（否则出现横向溢出）
        self.header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.header.setMinimumWidth(0)
        self.header.toggled.connect(self._toggle)
        box.addWidget(self.header)
        self.body = QWidget(self)
        body = QVBoxLayout(self.body)
        body.setContentsMargins(22, 2, 8, 6)
        body.setSpacing(3)
        self.args_label = _label(role="mono")
        self.result_label = _label(role="mono")
        self.refs_row = QHBoxLayout()
        self.refs_row.setSpacing(2)
        body.addWidget(self.args_label)
        body.addWidget(self.result_label)
        body.addLayout(self.refs_row)
        self.body.setVisible(False)
        box.addWidget(self.body)
        self.update_segment(segment)

    def _toggle(self, shown):
        self.body.setVisible(shown)
        self.header.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def update_segment(self, seg: dict) -> None:
        state = seg.get("state", "running")
        mark = {"running": "…", "ok": "✓", "failed": "✕"}[state]
        took = f"　{seg['ms'] / 1000:.2f} s" if seg.get("ms") is not None else ""
        self.header.setText(f"{mark}  {tool_label(seg['name'])}　{brief_args(seg.get('args') or {}, 48)}{took}")
        self.header.setToolTip(f"{seg['name']}（点击展开参数与结果）")
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        args = seg.get("args") or {}
        self.args_label.setText("参数\n" + format_args(args))
        summary = seg.get("summary")
        self.result_label.setText("结果\n" + summary if summary else "结果\n（执行中）")
        while self.refs_row.count():
            item = self.refs_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for kind, ident in object_refs(args):
            name = "节点" if kind == "node" else "杆件"
            button = _link_button(f"定位{name} {ident}", "在视口中选中并居中显示")
            button.clicked.connect(lambda _=False, k=kind, i=ident: self.locate.emit(k, i))
            self.refs_row.addWidget(button)
        self.refs_row.addStretch(1)


class ThinkingCard(QFrame):
    """模型的思考过程。运行中展开、实时滚动；本轮结束后收起，需要时再点开。

    推理模型每轮先想十几秒才开始调用工具，这段时间只显示「思考中」会让人以为
    卡住了。思考内容只供阅读，不进入回答正文、不回传模型，也不是数值依据。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("thinkingCard")
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.header = QToolButton(self)
        self.header.setObjectName("toolCardHeader")
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.header.toggled.connect(self._toggle)
        box.addWidget(self.header)
        self.body = _label("", "thinking")
        self.body.setContentsMargins(22, 0, 8, 6)
        box.addWidget(self.body)
        self._user_choice = None
        self.header.clicked.connect(lambda checked: setattr(self, "_user_choice", checked))

    def _toggle(self, shown):
        self.body.setVisible(shown)
        self.header.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def update_segment(self, seg: dict, live: bool) -> None:
        text = seg.get("text", "")
        self.header.setText(("正在思考…" if live else "思考过程") + f"　{len(text)} 字")
        # 运行中显示最后几行，跟着思考进度滚动；结束后显示全文
        self.body.setText(text[-600:] if live else text)
        shown = live if self._user_choice is None else self._user_choice
        self.header.blockSignals(True)
        self.header.setChecked(shown)
        self.header.blockSignals(False)
        self._toggle(shown)


class PlanCard(QFrame):
    def __init__(self, steps: list, parent=None):
        super().__init__(parent)
        self.setObjectName("planCard")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(10, 6, 10, 6)
        self.box.setSpacing(2)
        self.set_steps(steps)

    def set_steps(self, steps: list) -> None:
        while self.box.count():
            item = self.box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        done = sum(1 for s in steps if s["status"] == "done")
        self.box.addWidget(_label(f"计划　{done}/{len(steps)}", "caption"))
        for step in steps:
            row = _label(f"{PLAN_MARK.get(step['status'], '○')}  {step['title']}",
                         "planDone" if step["status"] == "done" else
                         "planActive" if step["status"] == "in_progress" else "")
            self.box.addWidget(row)


class EntryWidget(QFrame):
    """一条记录。``action`` 发出 (动作名, 记录序号)，由面板决定怎么做。"""

    action = Signal(str, int)
    locate = Signal(str, int)

    def __init__(self, index: int, entry: dict, parent=None):
        super().__init__(parent)
        self.index = index
        self.entry = entry
        self.setProperty("entry", entry["type"])
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(10, 8, 10, 8)
        self.box.setSpacing(4)
        self._tool_cards: dict[int, ToolCard] = {}
        self._text_labels: dict[int, QLabel] = {}
        self.plan_card = None
        self.actions_row = None
        self._build()

    # --- 构建 ---

    def _actions(self, items) -> None:
        row = QHBoxLayout()
        row.setSpacing(2)
        if self.entry["type"] == "user":
            row.addStretch(1)
        self.buttons = {}
        for name, text, tip in items:
            button = _link_button(text, tip)
            button.clicked.connect(lambda _=False, n=name: self.action.emit(n, self.index))
            row.addWidget(button)
            self.buttons[name] = button
        if self.entry["type"] != "user":
            row.addStretch(1)
        self.actions_row = row
        self.box.addLayout(row)

    def _build(self) -> None:
        kind = self.entry["type"]
        if kind == "user":
            refs = self.entry.get("refs") or []
            if refs:
                self.box.addWidget(_label("引用：" + "、".join(refs), "caption"))
            self.box.addWidget(_label(self.entry.get("text", ""), "user"))
            self._actions([("copy", "复制", "复制这条消息"),
                           ("edit", "编辑", "撤回这一轮，把文字放回输入框修改后重发")])
        elif kind == "assistant":
            head = QHBoxLayout()
            speaker = _label("工程助手", "speaker")
            speaker.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            head.addWidget(speaker)
            self.status_label = _label("", "caption")
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            head.addWidget(self.status_label, 1)
            self.box.addLayout(head)
            self.content = QVBoxLayout()
            self.content.setSpacing(4)
            self.box.addLayout(self.content)
            self.footer = _label("", "caption")
            self.box.addWidget(self.footer)
            self._actions([("copy", "复制", "复制这条回答"),
                           ("regenerate", "重新生成", "撤回这一轮的改动并重新提问")])
            self.refresh()
        elif kind in ("change", "approval"):
            self.box.addWidget(_label(self.entry.get("title", ""), "speaker"))
            self.lines = _label("\n".join(self.entry.get("lines", [])), "mono")
            self.box.addWidget(self.lines)
            self.state_label = _label("", "caption")
            self.box.addWidget(self.state_label)
            if kind == "change":
                self._actions([("keep", "保留", "保留本轮改动"),
                               ("revert", "撤销本轮", "把模型退回到这一轮开始之前")])
            else:
                self._actions([("approve", "确认应用", "授权应用预演中的批量修改"),
                               ("reject", "取消", "放弃这次修改")])
                self.buttons["approve"].setProperty("chat", "primaryLink")
            self.refresh()
        else:
            self.box.addWidget(_label(self.entry.get("text", ""),
                                      "warn" if self.entry.get("level") == "warn" else "caption"))

    # --- 刷新 ---

    def refresh(self) -> None:
        kind = self.entry["type"]
        if kind == "assistant":
            self._refresh_assistant()
        elif kind in ("change", "approval"):
            status = self.entry.get("status", "pending")
            text = {"kept": "已保留", "reverted": "已撤销，模型已退回到本轮之前",
                    "approved": "已确认，正在应用", "rejected": "已取消",
                    "stale": "模型此后又有修改，请用 Ctrl+Z 逐步退回",
                    "expired": "预演已失效"}.get(status, "")
            self.state_label.setText(text)
            self.state_label.setVisible(bool(text))
            for button in self.buttons.values():
                button.setVisible(status == "pending")

    def _refresh_assistant(self) -> None:
        entry = self.entry
        plan = entry.get("plan") or []
        if plan:
            if self.plan_card is None:
                self.plan_card = PlanCard(plan, self)
                self.content.insertWidget(0, self.plan_card)
            else:
                self.plan_card.set_steps(plan)
        offset = 1 if self.plan_card is not None else 0
        live_index = len(entry.get("segments", [])) - 1 if entry.get("status") == "streaming" else -1
        for i, seg in enumerate(entry.get("segments", [])):
            if seg["kind"] == "thinking":
                card = self._tool_cards.get(i)
                if card is None:
                    card = ThinkingCard(self)
                    self._tool_cards[i] = card
                    self.content.insertWidget(i + offset, card)
                card.update_segment(seg, i == live_index)
            elif seg["kind"] == "text":
                label = self._text_labels.get(i)
                if label is None:
                    label = _label("", "assistant", rich=True)
                    self._text_labels[i] = label
                    self.content.insertWidget(i + offset, label)
                label.setText(markdown_ready(seg["text"]))
            else:
                card = self._tool_cards.get(i)
                if card is None:
                    card = ToolCard(seg, self)
                    card.locate.connect(self.locate)
                    self._tool_cards[i] = card
                    self.content.insertWidget(i + offset, card)
                else:
                    card.update_segment(seg)
        status = entry.get("status", "done")
        activity = entry.get("activity", "")
        if entry.get("elapsed"):
            activity += f"　{entry['elapsed']} s"
        self.status_label.setText(activity if status == "streaming" else "")
        tools = [s for s in entry.get("segments", []) if s["kind"] == "tool"]
        footer = []
        if tools:
            ok = sum(1 for s in tools if s.get("state") == "ok")
            failed = len(tools) - ok
            if ok:
                footer.append(f"数值依据：{ok} 次成功的工具调用" +
                              (f"（另有 {failed} 次失败）" if failed else "") + "，展开卡片查看参数与结果")
            else:
                footer.append(f"本轮工具调用均未成功（{failed} 次），回答中没有来自求解器的新数值")
        if entry.get("metrics"):
            footer.append(entry["metrics"])
        self.footer.setText("\n".join(footer))
        self.footer.setVisible(bool(footer) and status != "streaming")
        if self.actions_row is not None:
            for button in self.buttons.values():
                button.setVisible(status != "streaming")


class TranscriptView(QScrollArea):
    """对话记录列表。数据在面板的 ``entries`` 里，这里只负责显示。"""

    action = Signal(str, int)
    locate = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assistantConversation")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner = QWidget()
        self.inner.setObjectName("transcriptInner")
        self.column = QVBoxLayout(self.inner)
        self.column.setContentsMargins(6, 6, 6, 6)
        self.column.setSpacing(6)
        self.column.addStretch(1)
        self.setWidget(self.inner)
        self.entries: list[dict] = []
        self.widgets: list[QWidget] = []
        self.welcome = None
        self._stick = True
        self.verticalScrollBar().rangeChanged.connect(self._follow)
        self.verticalScrollBar().valueChanged.connect(self._track)

    def _track(self, value):
        bar = self.verticalScrollBar()
        self._stick = value >= bar.maximum() - 24

    def _follow(self, _low, high):
        # 用户往上翻看旧消息时不要把他拽回底部；停在底部时跟随新内容
        if self._stick:
            self.verticalScrollBar().setValue(high)

    def set_welcome(self, widget: QWidget | None) -> None:
        if self.welcome is not None:
            self.welcome.deleteLater()
        self.welcome = widget
        if widget is not None:
            self.column.insertWidget(0, widget)
            widget.setVisible(not self.entries)

    def load(self, entries: list[dict]) -> None:
        for widget in self.widgets:
            widget.deleteLater()
        self.widgets = []
        self.entries = entries
        for index, entry in enumerate(entries):
            self._make(index, entry)
        if self.welcome is not None:
            self.welcome.setVisible(not entries)
        self._stick = True
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()))

    def _make(self, index: int, entry: dict) -> QWidget:
        widget = EntryWidget(index, entry, self.inner)
        widget.action.connect(self.action)
        widget.locate.connect(self.locate)
        self.column.insertWidget(self.column.count() - 1, widget)
        self.widgets.append(widget)
        return widget

    def append(self, entry: dict) -> int:
        self.entries.append(entry)
        index = len(self.entries) - 1
        self._make(index, entry)
        if self.welcome is not None:
            self.welcome.setVisible(False)
        self.refresh_actions()
        return index

    def refresh(self, index: int) -> None:
        if 0 <= index < len(self.widgets):
            self.widgets[index].refresh()

    def refresh_actions(self) -> None:
        """「编辑」「重新生成」只对最后一轮有效：更早的回合之后还有别的改动。"""
        last_user = max((i for i, e in enumerate(self.entries) if e["type"] == "user"), default=-1)
        last_reply = max((i for i, e in enumerate(self.entries) if e["type"] == "assistant"), default=-1)
        for i, widget in enumerate(self.widgets):
            buttons = getattr(widget, "buttons", {})
            if "edit" in buttons:
                buttons["edit"].setVisible(i == last_user)
            if "regenerate" in buttons:
                buttons["regenerate"].setVisible(
                    i == last_reply and self.entries[i].get("status") != "streaming")

    def toPlainText(self) -> str:  # noqa: N802 — 与 QTextBrowser 同名，旧调用方不用改
        return "\n".join(entry_text(e) for e in self.entries)
