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

外观按主流对话软件：用户消息是靠右的气泡、助手回答通栏无底色；状态用画出来的
标记（转动的圆弧、绿勾、红叉）而不是字符；连续几次工具调用收进一张可展开的
卡片，标题写明调用链；消息操作是悬停才显出的小图标。
"""
from __future__ import annotations

import json
import re

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                               QToolButton, QToolTip, QVBoxLayout, QWidget)

from . import glyphs, icons, theme

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

PLAN_MARK = {"pending": "○", "in_progress": glyphs.BUSY, "done": glyphs.TICK}


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


def _passive(widget: QWidget) -> QWidget:
    """卡片标题行里的小部件不接鼠标，点击落到整行按钮上。"""
    widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    return widget


def _link_button(text: str, tip: str = "") -> QToolButton:
    button = QToolButton()
    button.setText(text)
    button.setToolTip(tip)
    button.setProperty("chat", "chip")
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    return button


def _action_button(icon_name: str, tip: str) -> QToolButton:
    """消息下方的图标操作（复制、编辑、重新生成）。文字放进提示，不占一行。"""
    button = QToolButton()
    button.setIcon(icons.icon(icon_name))
    button.setIconSize(QSize(16, 16))
    button.setToolTip(tip)
    button.setProperty("chat", "action")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setFixedSize(24, 24)
    return button


def _push(text: str, tip: str, role: str = "") -> QPushButton:
    button = QPushButton(text)
    button.setToolTip(tip)
    if role:
        button.setProperty("role", role)
    return button


def _color(name: str) -> QColor:
    return QColor(getattr(theme, name))


class ElidedLabel(QLabel):
    """单行标签，放不下时在末尾省略。完整文字在悬停提示里。

    QLabel 本身不省略；不处理的话，一行长参数要么把卡片撑宽出现横向滚动，
    要么被硬切掉半个字。
    """

    def __init__(self, text: str = "", role: str = "", parent=None):
        super().__init__(parent)
        self._full = ""
        if role:
            self.setProperty("chat", role)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 — 覆盖 Qt 方法
        self._full = text
        self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight,
                                              max(0, self.width()))
        super().setText(shown)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def sizeHint(self):  # noqa: N802
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 2,
                     super().sizeHint().height())

    def minimumSizeHint(self):  # noqa: N802
        return QSize(0, super().minimumSizeHint().height())


class StatusIcon(QWidget):
    """16 像素的状态标记，全部用 QPainter 画。

    原来用 ``√ × …`` 这类字符拼在标题前面，字号一变就和文字对不齐，运行中
    也只是一个不动的省略号。现在：运行中是一段转动的圆弧，成功是绿底白勾，
    失败是红底白叉，计划里待办是空心圈、进行中是蓝色实心点。
    """

    STATES = ("running", "ok", "failed", "partial", "pending", "active", "info", "warn")

    def __init__(self, state: str = "pending", size: int = 16, parent=None):
        super().__init__(parent)
        self._state = ""
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._spin)
        self.setFixedSize(size, size)
        _passive(self)
        self.set_state(state)

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        if state == "running":
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _spin(self) -> None:
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        side = min(self.width(), self.height())
        r = QRectF(1.5, 1.5, side - 3, side - 3)
        state = self._state
        if state == "running":
            g.setPen(QPen(_color("PANEL_HOVER"), 2))
            g.drawEllipse(r)
            pen = QPen(_color("ACCENT"), 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            g.setPen(pen)
            g.drawArc(r, -self._angle * 16, 100 * 16)
        elif state in ("ok", "failed", "partial", "info", "warn"):
            fill = {"ok": "SUCCESS", "failed": "ERROR", "partial": "WARN",
                    "info": "ACCENT", "warn": "WARN"}[state]
            g.setPen(Qt.PenStyle.NoPen)
            g.setBrush(_color(fill))
            if state == "warn":
                g.drawPolygon(QPolygonF([QPointF(side / 2, 1), QPointF(side - 0.5, side - 1.5),
                                         QPointF(0.5, side - 1.5)]))
            else:
                g.drawEllipse(r)
            pen = QPen(QColor(theme.ON_ACCENT), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            g.setPen(pen)
            s = side / 16.0
            if state == "ok":
                g.drawPolyline(QPolygonF([QPointF(4.6 * s, 8.3 * s), QPointF(7 * s, 10.6 * s),
                                          QPointF(11.4 * s, 5.8 * s)]))
            elif state == "failed":
                g.drawLine(QPointF(5.5 * s, 5.5 * s), QPointF(10.5 * s, 10.5 * s))
                g.drawLine(QPointF(10.5 * s, 5.5 * s), QPointF(5.5 * s, 10.5 * s))
            elif state in ("partial", "warn"):
                top = 5.5 if state == "warn" else 4.2
                g.drawLine(QPointF(8 * s, top * s), QPointF(8 * s, 9.2 * s))
                g.drawPoint(QPointF(8 * s, 11.8 * s))
            else:
                g.drawLine(QPointF(8 * s, 7.2 * s), QPointF(8 * s, 11.6 * s))
                g.drawPoint(QPointF(8 * s, 4.6 * s))
        elif state == "active":
            g.setPen(QPen(_color("ACCENT"), 1.6))
            g.setBrush(Qt.BrushStyle.NoBrush)
            g.drawEllipse(r)
            g.setPen(Qt.PenStyle.NoPen)
            g.setBrush(_color("ACCENT"))
            g.drawEllipse(r.center(), side / 5, side / 5)
        else:
            g.setPen(QPen(_color("BORDER_LIGHT"), 1.6))
            g.setBrush(Qt.BrushStyle.NoBrush)
            g.drawEllipse(r)
        g.end()


class TypingDots(QWidget):
    """三个依次起伏的圆点：请求已发出、模型还没返回第一个字。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0
        self._timer = QTimer(self)
        self._timer.setInterval(140)
        self._timer.timeout.connect(self._step)
        self.setFixedSize(36, 16)

    def _step(self) -> None:
        self._phase = (self._phase + 1) % 6
        self.update()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event):  # noqa: N802
        super().hideEvent(event)
        self._timer.stop()

    def paintEvent(self, _event):  # noqa: N802
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        g.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            lit = self._phase % 3 == i
            color = _color("ACCENT" if lit else "BORDER_LIGHT")
            g.setBrush(color)
            g.drawEllipse(QPointF(6 + i * 11, 8 - (2 if lit else 0)), 3.2, 3.2)
        g.end()


class Avatar(QWidget):
    """助手头像：蓝底圆上画一榀白色门式刚架，和软件做的事一致。"""

    def __init__(self, size: int = 22, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, _event):  # noqa: N802
        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        side = min(self.width(), self.height())
        g.setPen(Qt.PenStyle.NoPen)
        g.setBrush(_color("ACCENT_DIM"))
        g.drawEllipse(QRectF(0, 0, side, side))
        pen = QPen(QColor(theme.ON_ACCENT), max(1.4, side / 14))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        g.setPen(pen)
        s = side / 22.0
        g.drawPolyline(QPolygonF([QPointF(6 * s, 16 * s), QPointF(6 * s, 9.5 * s),
                                  QPointF(11 * s, 6 * s), QPointF(16 * s, 9.5 * s),
                                  QPointF(16 * s, 16 * s)]))
        g.end()


def _disclosure(shown: bool) -> str:
    return glyphs.DISCLOSE_OPEN if shown else glyphs.DISCLOSE_CLOSED


class _CardHeader(QToolButton):
    """可展开卡片的整行标题：状态标记、名称、摘要、耗时、展开箭头。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolCardHeader")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 标题文字可能很长；不让它的理想宽度撑宽整列（否则出现横向溢出）
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)
        self.status = StatusIcon("running", 16, self)
        self.name = _passive(QLabel(self))
        self.name.setProperty("chat", "toolName")
        self.detail = _passive(ElidedLabel("", "toolArgs", self))
        self.took = _passive(QLabel(self))
        self.took.setProperty("chat", "toolTime")
        self.chevron = _passive(QLabel(_disclosure(False), self))
        self.chevron.setProperty("chat", "toolTime")
        row.addWidget(self.status)
        row.addWidget(self.name)
        row.addWidget(self.detail, 1)
        row.addWidget(self.took)
        row.addWidget(self.chevron)
        self.setFixedHeight(max(28, self.fontMetrics().height() + 12))
        self.toggled.connect(lambda shown: self.chevron.setText(_disclosure(shown)))


class ToolCard(QFrame):
    """一次工具调用。标题行一眼看清做了什么、成没成、用了多久；展开看参数与结果。"""

    locate = Signal(str, int)

    def __init__(self, segment: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("toolCard")
        box = QVBoxLayout(self)
        # 留出 1 像素，子控件不压住卡片的描边
        box.setContentsMargins(1, 1, 1, 1)
        box.setSpacing(0)
        self.header = _CardHeader(self)
        self.header.toggled.connect(self._toggle)
        box.addWidget(self.header)
        self.body = QWidget(self)
        self.body.setObjectName("toolBody")
        body = QVBoxLayout(self.body)
        body.setContentsMargins(8, 0, 8, 8)
        body.setSpacing(4)
        detail = QFrame(self.body)
        detail.setObjectName("toolDetail")
        inner = QVBoxLayout(detail)
        inner.setContentsMargins(8, 6, 8, 6)
        inner.setSpacing(6)
        self.args_label = _label(role="mono")
        self.result_label = _label(role="mono")
        inner.addWidget(self.args_label)
        inner.addWidget(self.result_label)
        body.addWidget(detail)
        self.refs_row = QHBoxLayout()
        self.refs_row.setSpacing(4)
        body.addLayout(self.refs_row)
        self.body.setVisible(False)
        box.addWidget(self.body)
        self.update_segment(segment)

    def _toggle(self, shown):
        self.body.setVisible(shown)

    def update_segment(self, seg: dict) -> None:
        state = seg.get("state", "running")
        args = seg.get("args") or {}
        self.header.status.set_state(state)
        self.header.name.setText(tool_label(seg["name"]))
        self.header.detail.setText(brief_args(args, 160))
        self.header.took.setText(f"{seg['ms'] / 1000:.2f} s" if seg.get("ms") is not None else "")
        self.header.setToolTip(f"{seg['name']}　{brief_args(args, 200)}\n点击展开参数与结果")
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        self.args_label.setText("参数\n" + format_args(args))
        summary = seg.get("summary")
        self.result_label.setText("结果\n" + summary if summary else "结果\n（执行中）")
        while self.refs_row.count():
            item = self.refs_row.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        for kind, ident in object_refs(args):
            name = "节点" if kind == "node" else "杆件"
            button = _link_button(f"定位{name} {ident}", "在视口中选中并居中显示")
            button.clicked.connect(lambda _=False, k=kind, i=ident: self.locate.emit(k, i))
            self.refs_row.addWidget(button)
        self.refs_row.addStretch(1)


class ToolGroup(QFrame):
    """连续的几次工具调用收在一张卡片里。

    一轮建模动辄五六次调用，每次一张卡片会把回答正文挤到屏幕外。收起时标题
    写明调用链（「生成门式刚架 → 设置工况 → 求解」）、成败与总耗时，仍然一眼
    看得到数值从哪来；展开后每次调用照旧可以看参数与结果、定位到视口。
    运行中自动展开，跟着进度走；本轮结束后收起，用户点开过就按用户的来。
    """

    locate = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolGroup")
        box = QVBoxLayout(self)
        box.setContentsMargins(1, 1, 1, 1)
        box.setSpacing(0)
        self.header = _CardHeader(self)
        self.header.toggled.connect(self._toggle)
        self.header.clicked.connect(lambda checked: setattr(self, "_user_choice", checked))
        box.addWidget(self.header)
        self.body = QWidget(self)
        self.body.setObjectName("toolBody")
        self.column = QVBoxLayout(self.body)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(0)
        box.addWidget(self.body)
        self.cards: dict[int, ToolCard] = {}
        self._user_choice = None

    def _toggle(self, shown):
        self.body.setVisible(shown)

    def update_run(self, items: list[tuple[int, dict]], live: bool) -> None:
        for index, seg in items:
            card = self.cards.get(index)
            if card is None:
                card = ToolCard(seg, self.body)
                card.locate.connect(self.locate)
                self.cards[index] = card
                self.column.addWidget(card)
            else:
                card.update_segment(seg)
        segs = [seg for _, seg in items]
        states = [s.get("state", "running") for s in segs]
        running = "running" in states
        failed = states.count("failed")
        if running:
            state = "running"
            current = next(s for s in segs if s.get("state", "running") == "running")
            title = f"正在执行第 {states.index('running') + 1} 步：{tool_label(current['name'])}"
        else:
            state = "failed" if failed == len(segs) else "partial" if failed else "ok"
            title = f"{len(segs)} 次工具调用" + (f"（{failed} 次失败）" if failed else "")
        chain = " → ".join(tool_label(s["name"]) for s in segs)
        self.header.status.set_state(state)
        self.header.name.setText(title)
        self.header.detail.setText(chain)
        total = sum(s.get("ms") or 0 for s in segs)
        self.header.took.setText(f"{total / 1000:.2f} s" if not running and total else "")
        self.header.setToolTip(chain + "\n点击展开每次调用的参数与结果")
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)
        shown = (running and live) if self._user_choice is None else self._user_choice
        self.header.blockSignals(True)
        self.header.setChecked(shown)
        self.header.blockSignals(False)
        self.header.chevron.setText(_disclosure(shown))
        self._toggle(shown)


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
        box.setSpacing(2)
        self.header = _CardHeader(self)
        self.header.toggled.connect(self._toggle)
        box.addWidget(self.header)
        self.body = QFrame(self)
        self.body.setObjectName("thinkingBody")
        inner = QVBoxLayout(self.body)
        inner.setContentsMargins(12, 0, 8, 4)
        inner.setSpacing(0)
        self.text = _label("", "thinking")
        inner.addWidget(self.text)
        box.addWidget(self.body)
        self._user_choice = None
        self.header.clicked.connect(lambda checked: setattr(self, "_user_choice", checked))

    def _toggle(self, shown):
        self.body.setVisible(shown)

    def update_segment(self, seg: dict, live: bool) -> None:
        text = seg.get("text", "")
        self.header.status.set_state("running" if live else "info")
        self.header.name.setText("正在思考…" if live else "思考过程")
        self.header.detail.setText(f"{len(text)} 字")
        # 运行中显示最后几行，跟着思考进度滚动；结束后显示全文
        self.text.setText(text[-600:] if live else text)
        shown = live if self._user_choice is None else self._user_choice
        self.header.blockSignals(True)
        self.header.setChecked(shown)
        self.header.blockSignals(False)
        self.header.chevron.setText(_disclosure(shown))
        self._toggle(shown)


class PlanCard(QFrame):
    """多步任务的计划清单：进度条加逐项状态。"""

    STATE = {"done": "ok", "in_progress": "active"}

    def __init__(self, steps: list, parent=None):
        super().__init__(parent)
        self.setObjectName("planCard")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(12, 8, 12, 8)
        self.box.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        title = QLabel("执行计划")
        title.setProperty("chat", "cardTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.count = QLabel()
        self.count.setProperty("chat", "caption")
        head.addWidget(self.count)
        self.box.addLayout(head)
        self.progress = QProgressBar(self)
        self.progress.setObjectName("planProgress")
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.box.addWidget(self.progress)
        self.rows = QVBoxLayout()
        self.rows.setSpacing(4)
        self.box.addLayout(self.rows)
        self.set_steps(steps)

    def set_steps(self, steps: list) -> None:
        while self.rows.count():
            item = self.rows.takeAt(0)
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    child = layout.takeAt(0).widget()
                    if child is not None:
                        child.hide()
                        child.deleteLater()
        done = sum(1 for s in steps if s["status"] == "done")
        self.count.setText(f"{done}/{len(steps)}")
        self.progress.setRange(0, max(1, len(steps)))
        self.progress.setValue(done)
        for step in steps:
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(StatusIcon(self.STATE.get(step["status"], "pending"), 14))
            row.addWidget(_label(step["title"],
                                 "planDone" if step["status"] == "done" else
                                 "planActive" if step["status"] == "in_progress" else ""), 1)
            self.rows.addLayout(row)


class EntryWidget(QFrame):
    """一条记录。``action`` 发出 (动作名, 记录序号)，由面板决定怎么做。

    排版按主流对话软件：用户消息是靠右的气泡，助手回答不加底色、通栏显示；
    复制、编辑、重新生成是消息下方的小图标，鼠标移上来才显出，最后一条回答
    常显。改动与待确认卡片是带按钮的独立卡片，提示是一条窄的说明条。
    """

    action = Signal(str, int)
    locate = Signal(str, int)

    BUBBLE_RATIO = 0.85

    def __init__(self, index: int, entry: dict, parent=None):
        super().__init__(parent)
        self.index = index
        self.entry = entry
        self.setProperty("entry", entry["type"])
        self.box = QVBoxLayout(self)
        card = entry["type"] in ("change", "approval")
        self.box.setContentsMargins(*((12, 8, 12, 8) if card else (0, 0, 0, 0)))
        self.box.setSpacing(4)
        self._tool_cards: dict[int, QWidget] = {}
        self._text_labels: dict[int, QLabel] = {}
        self._blocks: dict[int, QWidget] = {}
        self.plan_card = None
        self.actions_row = None
        self.actions_bar = None
        self._effect = None
        self._hover = False
        self._pinned = False
        self.buttons: dict = {}
        self._build()

    # --- 构建 ---

    def _actions(self, items) -> None:
        """消息下方的图标操作条；不在悬停状态时透明，但仍占位，避免内容跳动。"""
        self.actions_bar = QWidget(self)
        self.actions_bar.setObjectName("chatActions")
        row = QHBoxLayout(self.actions_bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        if self.entry["type"] == "user":
            row.addStretch(1)
        for name, icon_name, tip in items:
            button = _action_button(icon_name, tip)
            button.clicked.connect(lambda _=False, n=name, b=button: self._clicked(n, b))
            row.addWidget(button)
            self.buttons[name] = button
        if self.entry["type"] != "user":
            row.addStretch(1)
        self.actions_row = row
        self._effect = QGraphicsOpacityEffect(self.actions_bar)
        self.actions_bar.setGraphicsEffect(self._effect)
        self._update_actions()
        self.box.addWidget(self.actions_bar)

    def _clicked(self, name: str, button: QToolButton) -> None:
        self.action.emit(name, self.index)
        if name == "copy":
            QToolTip.showText(button.mapToGlobal(button.rect().bottomLeft()), "已复制到剪贴板", button)

    def _card_buttons(self, items) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)
        for name, text, tip, role in items:
            button = _push(text, tip, role)
            button.clicked.connect(lambda _=False, n=name: self.action.emit(n, self.index))
            row.addWidget(button)
            self.buttons[name] = button
        row.addStretch(1)
        self.actions_row = row
        self.box.addLayout(row)

    def _build(self) -> None:
        kind = self.entry["type"]
        if kind == "user":
            refs = self.entry.get("refs") or []
            if refs:
                ref = QLabel("引用 " + "、".join(refs))
                ref.setProperty("chat", "refs")
                ref.setAlignment(Qt.AlignmentFlag.AlignRight)
                ref.setWordWrap(True)
                ref.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                self.box.addWidget(ref)
            row = QHBoxLayout()
            row.setSpacing(0)
            row.addStretch(1)
            self.bubble = QFrame(self)
            self.bubble.setProperty("chat", "bubble")
            inner = QVBoxLayout(self.bubble)
            inner.setContentsMargins(12, 8, 12, 8)
            inner.setSpacing(0)
            self.text_label = _label(self.entry.get("text", ""), "user")
            inner.addWidget(self.text_label)
            row.addWidget(self.bubble)
            self.box.addLayout(row)
            self._actions([("copy", "copy", "复制这条消息"),
                           ("edit", "edit", "撤回这一轮，把文字放回输入框修改后重发")])
            self._fit_bubble()
        elif kind == "assistant":
            head = QHBoxLayout()
            head.setSpacing(8)
            head.addWidget(Avatar(22, self))
            speaker = QLabel("工程助手")
            speaker.setProperty("chat", "speaker")
            head.addWidget(speaker)
            self.busy_mark = StatusIcon("running", 14, self)
            head.addWidget(self.busy_mark)
            self.status_label = _label("", "caption")
            head.addWidget(self.status_label, 1)
            self.box.addLayout(head)
            self.content = QVBoxLayout()
            self.content.setSpacing(8)
            self.box.addLayout(self.content)
            self.typing = TypingDots(self)
            self.box.addWidget(self.typing)
            self.footer = _label("", "caption")
            self.box.addWidget(self.footer)
            self._actions([("copy", "copy", "复制这条回答"),
                           ("regenerate", "regenerate", "撤回这一轮的改动并重新提问")])
            self.refresh()
        elif kind in ("change", "approval"):
            head = QHBoxLayout()
            head.setSpacing(8)
            head.addWidget(StatusIcon("info" if kind == "change" else "warn", 16, self))
            head.addWidget(_label(self.entry.get("title", ""), "cardTitle"), 1)
            self.box.addLayout(head)
            lines = list(self.entry.get("lines", []))
            self.lines = _label("\n".join(lines), "mono")
            self.box.addWidget(self.lines)
            self._more = None
            if len(lines) > 4:
                # 一轮建模动辄七八类对象都在变，全部摊开会把卡片拉得比回答还长
                self.lines.setText("\n".join(lines[:4]))
                self._more = _link_button(f"展开全部 {len(lines)} 项", "显示本轮全部改动明细")
                self._more.setProperty("chat", "link")
                self._more.clicked.connect(self._expand_lines)
                self.box.addWidget(self._more, 0, Qt.AlignmentFlag.AlignLeft)
            self.state_label = _label("", "caption")
            self.box.addWidget(self.state_label)
            if kind == "change":
                self._card_buttons([("keep", "保留", "保留本轮改动", ""),
                                    ("revert", "撤销本轮", "把模型退回到这一轮开始之前", "")])
            else:
                self._card_buttons([("approve", "确认应用", "授权应用预演中的批量修改", "primary"),
                                    ("reject", "取消", "放弃这次修改", "")])
            self.refresh()
        else:
            warn = self.entry.get("level") == "warn"
            pill = QFrame(self)
            pill.setProperty("chat", "notePill")
            pill.setProperty("level", "warn" if warn else "info")
            row = QHBoxLayout(pill)
            row.setContentsMargins(8, 6, 8, 6)
            row.setSpacing(8)
            mark = StatusIcon("warn" if warn else "info", 14, pill)
            row.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(_label(self.entry.get("text", ""), "noteWarn" if warn else "note"), 1)
            self.box.addWidget(pill)

    def _expand_lines(self) -> None:
        self.lines.setText("\n".join(self.entry.get("lines", [])))
        self._more.hide()

    def _fit_bubble(self) -> None:
        """气泡宽度随文字：短句贴着文字收窄，长段落最多占一栏的 85%。"""
        if not hasattr(self, "bubble"):
            return
        metrics = self.text_label.fontMetrics()
        text = self.entry.get("text", "") or " "
        natural = max(metrics.horizontalAdvance(line) for line in text.splitlines() or [text])
        limit = int(max(120, self.width()) * self.BUBBLE_RATIO)
        self.bubble.setFixedWidth(max(48, min(limit, natural + 24 + 6)))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._fit_bubble()

    # --- 悬停显示操作 ---

    def set_pinned(self, pinned: bool) -> None:
        self._pinned = pinned
        self._update_actions()

    def enterEvent(self, event):  # noqa: N802
        super().enterEvent(event)
        self._hover = True
        self._update_actions()

    def leaveEvent(self, event):  # noqa: N802
        super().leaveEvent(event)
        self._hover = False
        self._update_actions()

    def _update_actions(self) -> None:
        if self._effect is not None:
            self._effect.setOpacity(1.0 if (self._hover or self._pinned) else 0.0)

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

    def _place(self, key: int, widget: QWidget) -> None:
        """按段落顺序追加；段落只会在末尾增加，所以新块总是放到最后。"""
        old = self._blocks.get(key)
        if old is widget:
            return
        if old is not None:
            position = self.content.indexOf(old)
            self.content.insertWidget(position, widget)
            old.hide()
            old.deleteLater()
        else:
            self.content.addWidget(widget)
        self._blocks[key] = widget

    def _refresh_assistant(self) -> None:
        entry = self.entry
        plan = entry.get("plan") or []
        if plan:
            if self.plan_card is None:
                self.plan_card = PlanCard(plan, self)
                self.content.insertWidget(0, self.plan_card)
            else:
                self.plan_card.set_steps(plan)
        segments = entry.get("segments", [])
        streaming = entry.get("status") == "streaming"
        live_index = len(segments) - 1 if streaming else -1
        i = 0
        while i < len(segments):
            seg = segments[i]
            if seg["kind"] == "thinking":
                card = self._tool_cards.get(i)
                if card is None:
                    card = ThinkingCard(self)
                    self._tool_cards[i] = card
                    self._place(i, card)
                card.update_segment(seg, i == live_index)
                i += 1
            elif seg["kind"] == "text":
                label = self._text_labels.get(i)
                if label is None:
                    label = _label("", "assistant", rich=True)
                    label.setOpenExternalLinks(False)
                    self._text_labels[i] = label
                    self._place(i, label)
                label.setText(markdown_ready(seg["text"]))
                i += 1
            else:
                end = i
                while end < len(segments) and segments[end]["kind"] == "tool":
                    end += 1
                run = list(range(i, end))
                if len(run) == 1:
                    card = self._tool_cards.get(i)
                    if card is None:
                        card = ToolCard(seg, self)
                        card.locate.connect(self.locate)
                        self._tool_cards[i] = card
                        self._place(i, card)
                    else:
                        card.update_segment(seg)
                else:
                    group = self._blocks.get(i)
                    if not isinstance(group, ToolGroup):
                        group = ToolGroup(self)
                        group.locate.connect(self.locate)
                        self._place(i, group)
                    group.update_run([(k, segments[k]) for k in run], streaming)
                    self._tool_cards.update(group.cards)
                i = end
        activity = entry.get("activity", "")
        if entry.get("elapsed"):
            activity += f"　{entry['elapsed']} s"
        self.status_label.setText(activity if streaming else "")
        self.busy_mark.setVisible(streaming)
        self.busy_mark.set_state("running" if streaming else "pending")
        self.typing.setVisible(streaming and not segments)
        tools = [s for s in segments if s["kind"] == "tool"]
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
        self.footer.setVisible(bool(footer) and not streaming)
        if self.actions_bar is not None:
            self.actions_bar.setVisible(not streaming)


class TranscriptView(QScrollArea):
    """对话记录列表。数据在面板的 ``entries`` 里，这里只负责显示。"""

    action = Signal(str, int)
    locate = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assistantConversation")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner = QWidget()
        self.inner.setObjectName("transcriptInner")
        self.column = QVBoxLayout(self.inner)
        self.column.setContentsMargins(12, 12, 12, 12)
        self.column.setSpacing(16)
        self.column.addStretch(1)
        self.setWidget(self.inner)
        self.entries: list[dict] = []
        self.widgets: list[QWidget] = []
        self.welcome = None
        self._stick = True
        # 往上翻看旧消息时，右下角出现「回到底部」按钮
        self.btn_bottom = QToolButton(self)
        self.btn_bottom.setObjectName("chatScrollDown")
        self.btn_bottom.setText(glyphs.CHEVRON_DOWN)
        self.btn_bottom.setToolTip("回到最新消息")
        self.btn_bottom.setFixedSize(28, 28)
        self.btn_bottom.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_bottom.clicked.connect(self.scroll_to_bottom)
        self.btn_bottom.hide()
        self.verticalScrollBar().rangeChanged.connect(self._follow)
        self.verticalScrollBar().valueChanged.connect(self._track)

    def _track(self, value):
        bar = self.verticalScrollBar()
        self._stick = value >= bar.maximum() - 24
        self._update_bottom_button()

    def _follow(self, _low, high):
        # 用户往上翻看旧消息时不要把他拽回底部；停在底部时跟随新内容
        if self._stick:
            self.verticalScrollBar().setValue(high)
        self._update_bottom_button()

    def _update_bottom_button(self) -> None:
        show = not self._stick and self.verticalScrollBar().maximum() > 0
        self.btn_bottom.setVisible(show)
        if show:
            self.btn_bottom.raise_()

    def scroll_to_bottom(self) -> None:
        self._stick = True
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        self._update_bottom_button()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        port = self.viewport().geometry()
        self.btn_bottom.move(port.center().x() - self.btn_bottom.width() // 2,
                             port.bottom() - self.btn_bottom.height() - 12)

    def set_welcome(self, widget: QWidget | None) -> None:
        if self.welcome is not None:
            self.welcome.deleteLater()
        self.welcome = widget
        if widget is not None:
            # 空对话时说明卡片占满可视区并居中，不是贴在顶上一小条
            self.column.insertWidget(0, widget, 8)
            widget.setVisible(not self.entries)

    def load(self, entries: list[dict]) -> None:
        for widget in self.widgets:
            widget.hide()
            widget.deleteLater()
        self.widgets = []
        self.entries = entries
        for index, entry in enumerate(entries):
            self._make(index, entry)
        if self.welcome is not None:
            self.welcome.setVisible(not entries)
        self.refresh_actions()
        self._stick = True
        QTimer.singleShot(0, self.scroll_to_bottom)

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
        # 自己发出的消息总要看到，不管之前翻到了哪里
        if entry["type"] == "user":
            self._stick = True
        self.refresh_actions()
        return index

    def refresh(self, index: int) -> None:
        if 0 <= index < len(self.widgets):
            self.widgets[index].refresh()

    def refresh_actions(self) -> None:
        """「编辑」「重新生成」只对最后一轮有效：更早的回合之后还有别的改动。

        最后一条回答的操作常显，其余消息鼠标移上去才显出。
        """
        last_user = max((i for i, e in enumerate(self.entries) if e["type"] == "user"), default=-1)
        last_reply = max((i for i, e in enumerate(self.entries) if e["type"] == "assistant"), default=-1)
        for i, widget in enumerate(self.widgets):
            buttons = getattr(widget, "buttons", {})
            if "edit" in buttons:
                buttons["edit"].setVisible(i == last_user)
            if "regenerate" in buttons:
                buttons["regenerate"].setVisible(
                    i == last_reply and self.entries[i].get("status") != "streaming")
            if hasattr(widget, "set_pinned"):
                widget.set_pinned(i == last_reply)

    def toPlainText(self) -> str:  # noqa: N802 — 与 QTextBrowser 同名，旧调用方不用改
        return "\n".join(entry_text(e) for e in self.entries)
