"""杆件属性指派对话框。

当前内核把截面几何和材料分别存放在杆件上，因此一次“属性指派”必须同时
明确两者。只改截面会让几何优先创建的草稿继续缺材料，却在界面上看似完成。
"""

from __future__ import annotations

import re

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QVBoxLayout, QWidget)

from . import dialog_styles


class AssignmentDialog(QDialog):
    def __init__(self, member_ids: list[int], sections: list[str],
                 materials: list[str], current_section: str = "",
                 current_material: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("属性指派")
        self.resize(360, 190)

        layout = QVBoxLayout(self)
        target = (f"杆件 {member_ids[0]}" if len(member_ids) == 1
                  else f"{len(member_ids)} 根杆件")
        layout.addWidget(QLabel(f"目标：{target}"))
        form = QFormLayout()
        self.cmb_section = QComboBox(self)
        self.cmb_section.addItems(sections)
        if current_section in sections:
            self.cmb_section.setCurrentText(current_section)
        self.cmb_material = QComboBox(self)
        self.cmb_material.addItems(materials)
        if current_material in materials:
            self.cmb_material.setCurrentText(current_material)
        form.addRow("截面", self.cmb_section)
        form.addRow("材料", self.cmb_material)
        layout.addLayout(form)

        hint = QLabel("两项会同时写入所选杆件；之后仍可在属性面板中单独修改。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = dialog_styles.button_box(self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_assignment(self) -> tuple[str, str]:
        return self.cmb_section.currentText(), self.cmb_material.currentText()


class MemberTarget(QWidget):
    """复用内核的几何筛选及命名集合，显示实际批量目标。"""

    def __init__(self, session, selected_id=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.selected_id = selected_id
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scope = QComboBox(self)
        for label, value in (("当前选中杆件", "selected"), ("全部杆件", "all"),
                             ("水平杆件", "horizontal"), ("竖向杆件", "vertical"),
                             ("斜杆件", "inclined"), ("输入杆件编号", "ids")):
            self.scope.addItem(label, value)
        for name, entry in (session.model.get("sets") or {}).items():
            if entry.get("members"):
                self.scope.addItem(f"集合：{name}", ("set", name))
        self.ids = QLineEdit(self)
        self.ids.setPlaceholderText("例如：1, 3, 5（不接受不存在的编号）")
        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        layout.addWidget(self.scope)
        layout.addWidget(self.ids)
        layout.addWidget(self.summary)
        self.scope.currentIndexChanged.connect(self.refresh)
        self.ids.textChanged.connect(self.refresh)
        self.refresh()

    def member_ids(self):
        scope = self.scope.currentData()
        known = {int(m["id"]) for m in self.session.model.get("members") or []}
        if scope == "selected":
            ids = [self.selected_id] if self.selected_id is not None else []
        elif scope == "all":
            ids = sorted(known)
        elif scope == "ids":
            parts = re.split(r"[,，\s]+", self.ids.text().strip())
            if not parts or any(not value.isdecimal() for value in parts):
                raise ValueError("请输入逗号分隔的杆件编号。")
            ids = sorted({int(value) for value in parts})
        elif isinstance(scope, tuple):
            ids = self.session._expand_members(scope[1])
        else:
            result = self.session.select_members(orientation=scope)
            if not result.ok:
                raise ValueError(str(result.payload.get("error")))
            ids = result.payload["member_ids"]
        if not ids or set(ids) - known:
            raise ValueError("没有有效目标，请选择杆件、切换范围或修正编号。")
        return ids

    def refresh(self, *_):
        self.ids.setVisible(self.scope.currentData() == "ids")
        try:
            ids = self.member_ids()
            sample = "、".join(map(str, ids[:12])) + ("…" if len(ids) > 12 else "")
            self.summary.setText(f"将作用于 {len(ids)} 根杆件：{sample}")
        except ValueError as exc:
            self.summary.setText(str(exc))


class BatchAssignmentDialog(AssignmentDialog):
    def __init__(self, session, selected_id=None, parent=None):
        member = next((m for m in session.model.get("members") or []
                       if m["id"] == selected_id), {})
        super().__init__([selected_id] if selected_id else [],
                         [s["name"] for s in session.model.get("sections") or []],
                         [m["name"] for m in session.model.get("materials") or []],
                         member.get("section", ""), member.get("material", ""), parent)
        self.setWindowTitle("批量属性指派")
        self.layout().takeAt(0).widget().deleteLater()
        self.target = MemberTarget(session, selected_id, self)
        self.layout().insertWidget(0, self.target)

    def accept(self):
        try:
            self.target.member_ids()
        except ValueError as exc:
            QMessageBox.warning(self, "属性指派未提交", str(exc))
            return
        super().accept()


class MemberSetupDialog(QDialog):
    """显式设置梁参考方向或杆端各方向的力学连接。"""

    def __init__(self, session, selected_id=None, *, orientation=False, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.setWindowTitle("梁截面方向" if orientation else "杆端连接")
        self.resize(480, 380)
        layout = QVBoxLayout(self)
        self.target = MemberTarget(session, selected_id, self)
        layout.addWidget(self.target)
        self.options = {}
        form = QFormLayout()
        if orientation:
            self.auto = QCheckBox("恢复自动局部轴", self)
            self.vector = QLineEdit(self)
            member = next((m for m in session.model.get("members") or []
                           if m["id"] == selected_id), {})
            self.vector.setText(", ".join(map(str, member.get("ref_vector") or [])))
            self.vector.setPlaceholderText("局部 y 轴参考方向，例如 0, 0, 1")
            self.auto.toggled.connect(self.vector.setDisabled)
            form.addRow(self.auto)
            form.addRow("全局参考向量", self.vector)
            hint = "参考方向投影后形成局部 y 轴，不能与杆轴平行。请复核截面强弱轴及局部载荷方向。"
        else:
            self.end = QComboBox(self)
            for label, value in (("i 端", "i"), ("j 端", "j"), ("两端", "both")):
                self.end.addItem(label, value)
            form.addRow("连接位置", self.end)
            for dof, label in (("rx", "扭转 rx"), ("ry", "弯曲 ry"), ("rz", "弯曲 rz")):
                row = QWidget(self)
                box = QHBoxLayout(row)
                box.setContentsMargins(0, 0, 0, 0)
                mode = QComboBox(row)
                for caption, value in (("保持原值", "keep"), ("刚接", "rigid"),
                                       ("释放（理想铰）", "release"), ("转动弹簧", "spring")):
                    mode.addItem(caption, value)
                stiffness = QDoubleSpinBox(row)
                stiffness.setDecimals(6)
                stiffness.setRange(0, 1e18)
                stiffness.setSuffix(" N·mm/rad" if session.model.get("units") == "N-mm-MPa"
                                    else " N·m/rad")
                stiffness.setEnabled(False)
                mode.currentIndexChanged.connect(
                    lambda _, m=mode, s=stiffness: s.setEnabled(m.currentData() == "spring"))
                box.addWidget(mode)
                box.addWidget(stiffness, 1)
                self.options[dof] = (mode, stiffness)
                form.addRow(label, row)
            hint = "按局部轴分别定义连接；未修改方向保留原值。螺栓数量不能直接决定刚接或铰接，弹簧刚度需有计算或试验依据。"
        layout.addLayout(form)
        note = QLabel(hint, self)
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = dialog_styles.button_box(self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def changes(self):
        if self.orientation:
            if self.auto.isChecked():
                return {"ref_vector": None}
            try:
                vector = [float(s) for s in re.split(r"[,，\s]+", self.vector.text().strip())]
            except ValueError as exc:
                raise ValueError("请填写三个有限数字，或勾选恢复自动局部轴。") from exc
            if len(vector) != 3:
                raise ValueError("参考方向需要三个分量。")
            return {"ref_vector": vector}
        rotations = {}
        for dof, (mode, stiffness) in self.options.items():
            value = mode.currentData()
            if value != "keep":
                if value == "spring" and stiffness.value() <= 0:
                    raise ValueError("转动弹簧必须填写正数刚度；零刚度请选释放。")
                rotations[dof] = {"mode": value, "stiffness": stiffness.value()}
        if not rotations:
            raise ValueError("请至少修改一个转动方向，或取消本次操作。")
        return {"end": self.end.currentData(), "rotations": rotations}

    def _accept(self):
        try:
            self.target.member_ids()
            self.changes()
        except ValueError as exc:
            QMessageBox.warning(self, "设置未提交", str(exc))
            return
        self.accept()


class SetDialog(QDialog):
    """创建或覆盖显式命名集合，保存在当前模型中。"""

    def __init__(self, session, selected_id=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("命名集合")
        self.resize(480, 440)
        layout = QVBoxLayout(self)
        listing = QPlainTextEdit(self)
        listing.setReadOnly(True)
        listing.setMaximumHeight(120)
        listing.setPlainText("\n".join(
            f"{name}：杆件 {entry.get('members', [])}；节点 {entry.get('nodes', [])}"
            for name, entry in (session.model.get("sets") or {}).items()) or "尚无集合")
        layout.addWidget(listing)
        self.name = QLineEdit(self)
        self.name.setPlaceholderText("例如：顶层梁；同名提交会覆盖原集合")
        layout.addWidget(self.name)
        self.include_members = QCheckBox("包含杆件", self)
        self.include_members.setChecked(True)
        layout.addWidget(self.include_members)
        self.target = MemberTarget(session, selected_id, self)
        self.include_members.toggled.connect(self.target.setEnabled)
        layout.addWidget(self.target)
        self.nodes = QLineEdit(self)
        self.nodes.setPlaceholderText("可选节点编号，例如：1, 2, 3")
        layout.addWidget(self.nodes)
        note = QLabel("集合只命名已有对象，不自动粘结节点、创建连接或施加载荷。", self)
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = dialog_styles.button_box(self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def definition(self):
        definition = {"name": self.name.text().strip()}
        if not definition["name"]:
            raise ValueError("请填写集合名称。")
        if self.include_members.isChecked():
            definition["member_ids"] = self.target.member_ids()
        if self.nodes.text().strip():
            parts = re.split(r"[,，\s]+", self.nodes.text().strip())
            if any(not value.isdecimal() for value in parts):
                raise ValueError("节点编号需用逗号分隔。")
            ids = sorted({int(value) for value in parts})
            known = {int(n["id"]) for n in self.session.model.get("nodes") or []}
            if set(ids) - known:
                raise ValueError("节点编号不存在，请修正后提交。")
            definition["node_ids"] = ids
        if "member_ids" not in definition and "node_ids" not in definition:
            raise ValueError("请至少选择一根杆件或一个节点。")
        return definition

    def _accept(self):
        try:
            self.definition()
        except ValueError as exc:
            QMessageBox.warning(self, "集合未提交", str(exc))
            return
        self.accept()
