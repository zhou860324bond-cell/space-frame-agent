"""属性面板：选中一个对象，就地改它。

**这块是补一个缺口，不是加一个功能。**

拾取早就做好了——点中一根杆，状态栏会说它多长、什么截面。然后就没了：
看得见，改不了。选中和表格编辑各自都能用，中间缺的就是这一块把它们接起来。

三条约束：

1. **面板不自己改模型字典。** 走 `Session.edit_member` / `edit_node`，
   那两个带 `@_records`，所以手工改的和 Agent 改的进同一条建模过程、
   同一条撤销链。自己改字典的话，用户改完按 Ctrl+Z 什么都不会发生。
2. **改完才提交，不是边打边提交。** 坐标输入框每敲一个字符就提交，
   会在建模过程里堆出一串垃圾步骤，撤销也变得没法用。
   所以只在编辑完成（回车或失焦）时提交一次。
3. **提交失败要把界面退回去。** 截面名打错了，模型没变，
   面板上却显示着新值——那是界面在说谎。
"""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox,
                               QFormLayout, QFrame, QLabel,
                               QLineEdit, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from . import i18n
from .qt_style import ElidedLabel


# 六个局部自由度，顺序与模型里的 fix 数组一致
DOF = ("ux", "uy", "uz", "rx", "ry", "rz")

# 端部释放常用的两种。**只放常用的**——六个复选框摆出来，
# 用户既不知道该勾哪个，也很容易勾出一个机构
RELEASE_PRESETS = (("刚接", []), ("平面内铰接", ["rz"]), ("两向铰接", ["ry", "rz"]))

# 支座常用形式。同理，六个格子让用户自己勾很容易勾出机构
SUPPORT_PRESETS = (("自由", []),
                   ("铰接", [1, 1, 1, 0, 0, 0]),
                   ("固接", [1, 1, 1, 1, 1, 1]),
                   ("滑动（仅竖向）", [0, 0, 1, 0, 0, 0]))


class PropertiesPanel(QWidget):
    """选中对象的属性，可编辑。"""

    # 改动已提交，主窗口据此重画并刷新状态
    edited = Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.kind: str | None = None
        self.ident: int | None = None
        self._filling = False          # 填表期间屏蔽信号，见 _fill

        box = QVBoxLayout(self)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(8)

        self.title = QLabel("未选中对象")
        self.title.setProperty("panel", "title")
        box.addWidget(self.title)

        self.hint = QLabel("在视口中点选一个节点或杆件，或在结果表中点一行。")
        self.hint.setWordWrap(True)
        self.hint.setProperty("panel", "hint")
        box.addWidget(self.hint)

        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setProperty("ribbon", "sep")
        box.addWidget(line)

        self.form_host = QWidget(self)
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(6)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        box.addWidget(self.form_host)

        self.advanced_button = QToolButton(self)
        self.advanced_button.setText(i18n.tr("高级参数"))
        self.advanced_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_button.setCheckable(True)
        self.advanced_button.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        box.addWidget(self.advanced_button)
        self.advanced_host = QWidget(self)
        advanced_box = QVBoxLayout(self.advanced_host)
        advanced_box.setContentsMargins(0, 0, 0, 0)
        advanced_box.setSpacing(8)
        self.advanced_hint = QLabel("端部约束、梁方向、刚域偏移与计算长度系数。")
        self.advanced_hint.setWordWrap(True)
        self.advanced_hint.setProperty("panel", "hint")
        advanced_box.addWidget(self.advanced_hint)
        self.advanced_form = QFormLayout()
        self.advanced_form.setContentsMargins(0, 0, 0, 0)
        self.advanced_form.setSpacing(6)
        self.advanced_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        advanced_box.addLayout(self.advanced_form)
        box.addWidget(self.advanced_host)
        self.advanced_host.hide()

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setProperty("panel", "hint")
        box.addWidget(self.status)
        box.addStretch(1)

        self._widgets: dict[str, Any] = {}
        self._errors: dict[str, QLabel] = {}
        self._field_labels: dict[str, str] = {}
        self._active_form = self.form
        self._error_key = None
        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self._reveal_error)
        self.clear()

    # --- 显示 ---

    def clear(self) -> None:
        self.kind = self.ident = None
        self._clear_form()
        self.title.setText("未选中对象")
        self.hint.setText("在视口中点选一个节点或杆件，或在结果表中点一行。")
        self.hint.setVisible(True)
        self.status.setText("")
        self.advanced_button.hide()
        self.advanced_button.setChecked(False)
        self.advanced_host.hide()

    def _clear_form(self) -> None:
        self._error_timer.stop()
        self._error_key = None
        filling = self._filling
        self._filling = True
        try:
            for form in (self.form, self.advanced_form):
                while form.rowCount():
                    form.removeRow(0)
        finally:
            self._filling = filling
        self._widgets = {}
        self._errors = {}
        self._field_labels = {}
        self._active_form = self.form

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_host.setVisible(checked and self.kind == "member")
        self.advanced_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def show_object(self, kind: str, ident: int) -> None:
        """显示某个对象的属性。kind 是 "node" 或 "member"。"""
        try:
            frame = self.session.frame or self.session.preview_frame()
        except ValueError:
            self.clear()
            return
        if (kind, ident) != (self.kind, self.ident):
            self.advanced_button.setChecked(False)
        self.kind, self.ident = kind, ident
        self.advanced_button.setVisible(kind == "member")
        self.advanced_host.setVisible(kind == "member" and self.advanced_button.isChecked())
        self.hint.setVisible(False)
        self.status.setText("")
        self._clear_form()
        if kind == "member":
            self._fill_member(frame, ident)
        else:
            self._fill_node(frame, ident)

    def refresh(self) -> None:
        """模型变了之后重新读一遍。选中的对象可能已经不存在了。"""
        if self.kind is None or self.ident is None:
            return
        alive = (self.session.model.get("members")
                 if self.kind == "member" else self.session.model.get("nodes"))
        if not any(int(e["id"]) == self.ident for e in alive or []):
            self.clear()                # 选中的对象被删了
            return
        self.show_object(self.kind, self.ident)

    # --- 杆件 ---

    def _fill_member(self, frame, mid: int) -> None:
        import numpy as np

        entry = next((m for m in self.session.model.get("members") or []
                      if int(m["id"]) == mid), None)
        if entry is None or mid not in frame.members:
            self.clear()
            return
        m = frame.members[mid]
        a, b = frame.nodes[m.i].xyz, frame.nodes[m.j].xyz
        length = float(np.linalg.norm(b - a))
        length_unit = "mm" if self.session.model.get("units") == "N-mm-MPa" else "m"

        self.title.setText(f"杆件 {mid}")
        self._read_only("端点", f"节点 {m.i} → 节点 {m.j}")
        self._read_only("长度", f"{length:.4f} {length_unit}")

        sections = [s["name"] for s in self.session.model.get("sections") or []]
        materials = [x["name"] for x in self.session.model.get("materials") or []]
        self._combo("section", "截面", sections, entry.get("section"))
        self._combo("material", "材料", materials, entry.get("material"))
        self._active_form = self.advanced_form
        ref = entry.get("ref_vector")
        text = ", ".join(f"{float(v):g}" for v in ref) if ref else ""
        self._text("ref_vector", "梁方向（局部 y）", text,
                   "例如 0, 0, 1；留空使用自动方向")
        for end in ("i", "j"):
            offset = entry.get(f"offset_{end}")
            value = ", ".join(f"{float(v):g}" for v in offset) if offset else ""
            self._text(f"offset_{end}", f"{end} 端刚域偏移（全局）", value,
                       "例如 0.2, 0, 0；留空表示无刚域")

        # 计算长度系数：欧拉校核用。不给就由杆端释放推定，而推定**只对无侧移
        # 结构成立**——有侧移的框架柱 μ>1、悬臂柱 2.0。所以这一栏必须在界面上
        # 能填，否则有侧移的框架永远只能拿到一个偏不安全的判定。
        for key, label in (("mu_y", "计算长度系数 μy"),
                           ("mu_z", "计算长度系数 μz")):
            value = entry.get(key)
            self._text(key, label, "" if value is None else f"{float(value):g}",
                       "留空按杆端约束推定（仅无侧移适用）；两端铰 1、"
                       "一端固一端铰 0.7、两端固 0.5、悬臂 2")

        rel = entry.get("releases") or {}
        for end in ("i", "j"):
            current = list(rel.get(end) or [])
            names = [n for n, v in RELEASE_PRESETS]
            match = next((n for n, v in RELEASE_PRESETS
                          if sorted(v) == sorted(current)), None)
            if match is None:               # 自定义组合：原样列出来，不改成预设
                match = "、".join(current)
                names = names + [match]
            self._combo(f"release_{end}", f"{end} 端约束", names, match)

    # --- 节点 ---

    def _fill_node(self, frame, nid: int) -> None:
        entry = next((n for n in self.session.model.get("nodes") or []
                      if int(n["id"]) == nid), None)
        if entry is None:
            self.clear()
            return
        length_unit = "mm" if self.session.model.get("units") == "N-mm-MPa" else "m"
        self.title.setText(f"节点 {nid}")
        for key, label in (("x", "X 坐标"), ("y", "Y 坐标"), ("z", "Z 坐标")):
            self._number(key, f"{label} ({length_unit})", float(entry[key]))

        fix = next((s["fix"] for s in self.session.model.get("supports") or []
                    if int(s["node"]) == nid), None)
        names = [n for n, v in SUPPORT_PRESETS]
        match = next((n for n, v in SUPPORT_PRESETS
                      if (v or None) == (list(fix) if fix else None)), None)
        if match is None and fix:
            match = "自定义：" + "".join(
                d for d, on in zip(DOF, fix, strict=True) if on) or "自定义"
            names = names + [match]
        self._combo("support", "约束", names, match or "自由")

        # 自定义约束时把六个格子摆出来，但**只在需要时**——
        # 平时六个复选框摆着，用户很容易勾出一个机构
        if match and match.startswith("自定义"):
            self._read_only("固定方向",
                            "、".join(d for d, on in zip(DOF, fix, strict=True) if on) or "无")

    # --- 控件 ---

    def _read_only(self, label: str, text: str) -> None:
        value = ElidedLabel(text)
        value.setToolTip(text)
        value.setProperty("panel", "value")
        self.form.addRow(label, value)

    def _add_field(self, key: str, label: str, widget: QWidget) -> None:
        """每个输入旁保留独立的错误行，仍使用工具校验与撤销链。"""
        advanced = self._active_form is self.advanced_form
        if advanced:
            field = QWidget(self.advanced_host)
            layout = QVBoxLayout(field)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            layout.addWidget(QLabel(label, field))
            layout.addWidget(widget)
            self.advanced_form.addRow(field)
        else:
            self.form.addRow(label, widget)
        self._widgets[key] = widget
        self._field_labels[key] = label
        message = QLabel(self)
        message.setWordWrap(True)
        message.setTextFormat(Qt.TextFormat.PlainText)
        message.setProperty("fieldError", True)
        if advanced:
            layout.addWidget(message)
        else:
            self.form.addRow(message)
        message.hide()
        self._errors[key] = message

    def _clear_error(self, key: str) -> None:
        field = self._widgets.get(key)
        if field is not None:
            field.setProperty("invalid", False)
            field.style().unpolish(field)
            field.style().polish(field)
        if key in self._errors:
            self._errors[key].hide()

    def _fail(self, key: str, message: str) -> None:
        """拒绝提交后恢复真实值，并将原因与下一步放到出错字段旁。"""
        self.status.setText(message)
        self._reload()
        if key in self._errors:
            if self.advanced_host.isAncestorOf(self._widgets[key]):
                self.advanced_button.setChecked(True)
            self._errors[key].setText(message + "\n已恢复原值，请修正后重新输入。")
            self._errors[key].show()
            field = self._widgets[key]
            field.setProperty("invalid", True)
            field.style().unpolish(field)
            field.style().polish(field)
            self._error_key = key
            # 等滚动区完成新表单的高度计算，再定位到字段旁的原因。
            self._error_timer.start(0)

    def _reveal_error(self) -> None:
        error = self._errors.get(self._error_key)
        if error is None or error.isHidden():
            return
        area = self.parentWidget()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parentWidget()
        if area is not None:
            self.advanced_host.layout().activate()
            self.layout().activate()
            point = error.mapTo(self, QPoint(error.width() // 2, error.height() // 2))
            area.ensureVisible(point.x(), point.y(), 8,
                               min(error.height() // 2 + 12, area.viewport().height() // 2))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # 换行和滚动条可能再次改变表单高度，最终尺寸下重新定位当前错误。
        if getattr(self, "_error_key", None) is not None:
            self._error_timer.start(0)

    def _combo(self, key: str, label: str, options: list[str],
               current: str | None) -> None:
        box = QComboBox(self)
        box.addItems(options)
        if current and current in options:
            box.setCurrentText(current)
        box.currentTextChanged.connect(
            lambda text, k=key: self._commit(k, text))
        self._add_field(key, label, box)

    def _number(self, key: str, label: str, value: float) -> None:
        box = QDoubleSpinBox(self)
        box.setRange(-1e6, 1e6)
        box.setDecimals(4)
        box.setSingleStep(0.1)
        box.setValue(value)
        box.setKeyboardTracking(False)      # **只在编辑完成时发信号**，
        box.editingFinished.connect(        # 否则每敲一个字符就提交一步
            lambda k=key, w=box: self._commit(k, w.value()))
        self._add_field(key, label, box)
        box.valueChanged.connect(lambda _value, k=key: self._clear_error(k))

    def _text(self, key: str, label: str, value: str, placeholder: str = "") -> None:
        """一个按回车或失焦提交的短文本字段。"""
        box = QLineEdit(value, self)
        box.setPlaceholderText(placeholder)
        box.editingFinished.connect(lambda k=key, w=box: self._commit(k, w.text()))
        self._add_field(key, label, box)
        box.textEdited.connect(lambda _text, k=key: self._clear_error(k))

    # --- 提交 ---

    def _commit(self, key: str, value: Any) -> None:
        if self._filling or self.ident is None:
            return
        self._clear_error(key)
        kwargs: dict[str, Any] = {}
        if self.kind == "member":
            if key in ("section", "material"):
                kwargs[key] = value
            elif key == "ref_vector":
                raw = str(value).strip()
                if not raw:
                    kwargs["ref_vector"] = None
                else:
                    try:
                        vector = [float(v.strip()) for v in raw.split(",")]
                    except ValueError:
                        self._fail(key, "未生效：梁方向应为三个逗号分隔的数字，例如 0, 0, 1。")
                        return
                    if len(vector) != 3:
                        self._fail(key, "未生效：梁方向必须包含三个分量，例如 0, 0, 1。")
                        return
                    if not all(math.isfinite(v) for v in vector):
                        self._fail(key, "未生效：梁方向必须为有限数字，例如 0, 0, 1。")
                        return
                    kwargs["ref_vector"] = vector
            elif key.startswith("offset_"):
                raw = str(value).strip()
                if not raw:
                    kwargs[key] = None
                else:
                    try:
                        vector = [float(v.strip()) for v in raw.split(",")]
                    except ValueError:
                        self._fail(key, "未生效：刚域偏移应为三个逗号分隔的数字，例如 0.2, 0, 0。")
                        return
                    if len(vector) != 3:
                        self._fail(key, "未生效：刚域偏移必须包含三个全局分量，例如 0.2, 0, 0。")
                        return
                    if not all(math.isfinite(v) for v in vector):
                        self._fail(key, "未生效：刚域偏移必须为有限数字，例如 0.2, 0, 0。")
                        return
                    kwargs[key] = vector
            elif key in ("mu_y", "mu_z"):
                raw = str(value).strip()
                if not raw:
                    kwargs[key] = None          # 清空 = 回到按杆端约束推定
                else:
                    try:
                        factor = float(raw)
                    except ValueError:
                        self._fail(key, "未生效：计算长度系数应为一个正数，例如 1、0.7、2。")
                        return
                    if not (math.isfinite(factor) and factor > 0):
                        self._fail(key, "未生效：计算长度系数必须为大于零的有限数字，例如 1、0.7、2。")
                        return
                    kwargs[key] = factor
            elif key.startswith("release_"):
                end = key[-1]
                preset = next((v for n, v in RELEASE_PRESETS if n == value), None)
                if preset is None:
                    return              # 自定义组合，面板不改它
                kwargs[f"releases_{end}"] = list(preset)
            result = self.session.edit_member(member_id=self.ident, **kwargs)
        else:
            if key in ("x", "y", "z"):
                kwargs[key] = float(value)
            elif key == "support":
                preset = next((v for n, v in SUPPORT_PRESETS if n == value), None)
                if preset is None:
                    return
                kwargs["fix"] = list(preset)
            result = self.session.edit_node(node_id=self.ident, **kwargs)

        if not result.ok:
            # **提交失败要把界面退回去。** 模型没变而面板显示着新值，
            # 就是界面在说谎——用户会以为改成功了
            detail = (result.payload.get("error")
                      or "；".join(str(e) for e in result.payload.get("errors", [])))
            hint = result.payload.get("hint")
            self._fail(key, f"未生效：{detail}" + (f"\n{hint}" if hint else ""))
            return

        changed = result.payload.get("changed") or {}
        if not changed:
            self.status.clear()
            return                      # 值没变，别在建模过程里堆空步骤
        message = f"已修改{self._field_labels.get(key, key)}。原有分析结果已失效，请重新求解。"
        self.edited.emit()
        # **发完信号再写状态。** 主窗口收到信号会调 refresh()，
        # 而 refresh 会重建整个表单——先写就被冲掉了，用户改完看不到确认
        self.status.setText(message)

    def _reload(self) -> None:
        """把控件退回模型里的真实值，**但把说明留在屏幕上**。

        `show_object` 会清空状态栏。退回时顺手把刚写的错误一起清掉的话，
        用户只会看见值自己跳了回去，没有任何解释——比不退回更让人困惑。
        """
        message = self.status.text()
        self._filling = True
        try:
            if self.kind and self.ident is not None:
                self.show_object(self.kind, self.ident)
        finally:
            self._filling = False
        self.status.setText(message)
