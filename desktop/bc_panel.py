"""边界条件面板：支座 + 荷载的人工设置。

类似 Abaqus/CAE 的 Load 模块：选中节点/杆件，直接设置支座约束和荷载，
不需要通过 AI 对话。所有操作走 Session 的 @_records 方法，支持撤销。
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from . import theme

# 支座预设：(名称, fix数组)
SUPPORT_PRESETS = [
    ("固定端", [1, 1, 1, 1, 1, 1]),
    ("铰接", [1, 1, 1, 0, 0, 0]),
    ("滚动（X向可动）", [0, 1, 1, 0, 0, 0]),
    ("滚动（Z向可动）", [1, 1, 0, 0, 0, 0]),
    ("自由", [0, 0, 0, 0, 0, 0]),
]


class BCPanel(QWidget):
    """边界条件面板：支座 + 荷载。"""

    changed = Signal()  # 边界条件改变时发出，主窗口据此重画

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node: int | None = None
        self._current_member: int | None = None
        self._filling = False

        box = QVBoxLayout(self)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(8)

        # 标题
        title = QLabel("边界条件")
        title.setStyleSheet(f"font-weight:650; font-size:10pt; color:{theme.INK};")
        box.addWidget(title)

        flow = QLabel("支座约束作用于 Initial；给定位移和荷载作用于当前分析步。")
        flow.setWordWrap(True)
        flow.setStyleSheet(f"color:{theme.INK_MUTED};")
        box.addWidget(flow)

        # 工况选择（放在边界条件之后，避免看起来像支座属于某个荷载工况）
        case_row = QHBoxLayout()
        case_row.addWidget(QLabel("当前分析步"))
        self.cmb_case = QComboBox()
        self.cmb_case.currentTextChanged.connect(self._on_case_changed)
        case_row.addWidget(self.cmb_case)
        self.btn_new_case = QPushButton("新建")
        self.btn_new_case.setFixedWidth(50)
        self.btn_new_case.clicked.connect(self._new_case)
        case_row.addWidget(self.btn_new_case)
        case_row.addStretch()

        # 选中对象提示
        self.lbl_selection = QLabel("未选中对象。在视口中用「选择节点」或「选择杆件」选中后，在这里设置边界条件。")
        self.lbl_selection.setWordWrap(True)
        self.lbl_selection.setStyleSheet(f"color:{theme.INK_MUTED}; font-size:8pt;")
        box.addWidget(self.lbl_selection)

        # 支座设置
        self.group_support = QGroupBox("边界条件（Initial）")
        support_layout = QVBoxLayout(self.group_support)
        self.txt_bc_name = QLineEdit("BC-1")
        self.txt_bc_name.setPlaceholderText("边界条件名称")
        support_layout.addWidget(QLabel("名称"))
        support_layout.addWidget(self.txt_bc_name)
        self.cmb_support = QComboBox()
        for name, _ in SUPPORT_PRESETS:
            self.cmb_support.addItem(name)
        support_layout.addWidget(QLabel("支座类型"))
        support_layout.addWidget(self.cmb_support)
        self.btn_apply_support = QPushButton("应用到选中节点")
        self.btn_apply_support.clicked.connect(self._apply_support)
        self.btn_clear_support = QPushButton("删除支座")
        self.btn_clear_support.clicked.connect(self._clear_support)
        support_buttons = QHBoxLayout()
        support_buttons.addWidget(self.btn_apply_support)
        support_buttons.addWidget(self.btn_clear_support)
        support_layout.addLayout(support_buttons)
        box.addWidget(self.group_support)
        box.addLayout(case_row)

        # 分析步边界条件：支座沉降/主动位移。它和 Initial 支座不是一回事。
        self.group_settlement = QGroupBox("给定位移（当前分析步）")
        settlement_layout = QFormLayout(self.group_settlement)
        self.txt_settlement_name = QLineEdit("Displacement-1")
        settlement_layout.addRow("名称", self.txt_settlement_name)
        self.settlement_spins = [self._make_spin(-1e9, 1e9, 0) for _ in range(6)]
        for label, spin in zip(("U1", "U2", "U3", "UR1", "UR2", "UR3"),
                               self.settlement_spins):
            settlement_layout.addRow(label, spin)
        settlement_note = QLabel("只在 Initial 中已约束的自由度上生效；可用于支座沉降或主动位移。")
        settlement_note.setWordWrap(True)
        settlement_layout.addRow(settlement_note)
        settlement_btn_row = QHBoxLayout()
        self.btn_apply_settlement = QPushButton("应用")
        self.btn_apply_settlement.clicked.connect(self._apply_settlement)
        self.btn_clear_settlement = QPushButton("删除")
        self.btn_clear_settlement.clicked.connect(self._clear_settlement)
        settlement_btn_row.addWidget(self.btn_apply_settlement)
        settlement_btn_row.addWidget(self.btn_clear_settlement)
        settlement_layout.addRow(settlement_btn_row)
        box.addWidget(self.group_settlement)

        # 节点荷载
        self.group_nodal = QGroupBox("节点集中力")
        nodal_layout = QFormLayout(self.group_nodal)
        self.txt_nodal_name = QLineEdit("CF-1")
        nodal_layout.addRow("名称", self.txt_nodal_name)
        self.spn_fx = self._make_spin(-1e9, 1e9, 0)
        self.spn_fy = self._make_spin(-1e9, 1e9, 0)
        self.spn_fz = self._make_spin(-1e9, 1e9, 0)
        self.spn_mx = self._make_spin(-1e9, 1e9, 0)
        self.spn_my = self._make_spin(-1e9, 1e9, 0)
        self.spn_mz = self._make_spin(-1e9, 1e9, 0)
        self._nodal_layout = nodal_layout
        nodal_layout.addRow("Fx (N)", self.spn_fx)
        nodal_layout.addRow("Fy (N)", self.spn_fy)
        nodal_layout.addRow("Fz (N)", self.spn_fz)
        nodal_layout.addRow("Mx", self.spn_mx)
        nodal_layout.addRow("My", self.spn_my)
        nodal_layout.addRow("Mz", self.spn_mz)
        nodal_btn_row = QHBoxLayout()
        self.btn_apply_nodal = QPushButton("应用")
        self.btn_apply_nodal.clicked.connect(self._apply_nodal_load)
        self.btn_clear_nodal = QPushButton("删除")
        self.btn_clear_nodal.clicked.connect(self._clear_nodal_load)
        self.btn_copy_nodal = QPushButton("复制")
        self.btn_copy_nodal.clicked.connect(self._copy_nodal_load)
        nodal_btn_row.addWidget(self.btn_apply_nodal)
        nodal_btn_row.addWidget(self.btn_copy_nodal)
        nodal_btn_row.addWidget(self.btn_clear_nodal)
        nodal_layout.addRow(nodal_btn_row)
        box.addWidget(self.group_nodal)

        # 杆件荷载
        self.group_member = QGroupBox("杆件均布荷载")
        member_layout = QFormLayout(self.group_member)
        self.txt_member_name = QLineEdit("Line-1")
        member_layout.addRow("名称", self.txt_member_name)
        self.spn_wx = self._make_spin(-1e6, 1e6, 0)
        self.spn_wy = self._make_spin(-1e6, 1e6, 0)
        self.spn_wz = self._make_spin(-1e6, 1e6, 0)
        self._member_layout = member_layout
        member_layout.addRow("wx", self.spn_wx)
        member_layout.addRow("wy", self.spn_wy)
        member_layout.addRow("wz", self.spn_wz)
        member_btn_row = QHBoxLayout()
        self.btn_apply_member = QPushButton("应用")
        self.btn_apply_member.clicked.connect(self._apply_member_load)
        self.btn_clear_member = QPushButton("删除")
        self.btn_clear_member.clicked.connect(self._clear_member_load)
        self.btn_copy_member = QPushButton("复制")
        self.btn_copy_member.clicked.connect(self._copy_member_load)
        member_btn_row.addWidget(self.btn_apply_member)
        member_btn_row.addWidget(self.btn_copy_member)
        member_btn_row.addWidget(self.btn_clear_member)
        member_layout.addRow(member_btn_row)
        box.addWidget(self.group_member)

        box.addStretch()

        self.refresh()

    def _make_spin(self, min_val, max_val, default):
        sp = QDoubleSpinBox()
        sp.setRange(min_val, max_val)
        sp.setValue(default)
        sp.setDecimals(3)
        sp.setSingleStep(1.0)
        return sp

    def refresh(self):
        """从当前模型刷新工况列表和选中对象的边界条件。"""
        self._refresh_units()
        self._filling = True
        # 刷新工况列表
        current = self.cmb_case.currentText()
        self.cmb_case.blockSignals(True)
        self.cmb_case.clear()
        cases = self.session.model.get("load_cases", [])
        for c in cases:
            self.cmb_case.addItem(c.get("name", ""))
        if current and self.cmb_case.findText(current) >= 0:
            self.cmb_case.setCurrentText(current)
        elif cases:
            self.cmb_case.setCurrentIndex(0)
        self.cmb_case.blockSignals(False)
        self._filling = False

        # 刷新选中对象的边界条件
        self._load_selection_values()

    def _refresh_units(self):
        millimetres = self.session.model.get("units") == "N-mm-MPa"
        moment = "N·mm" if millimetres else "N·m"
        line = "N/mm" if millimetres else "N/m"
        length = "mm" if millimetres else "m"
        for spin in (self.spn_mx, self.spn_my, self.spn_mz):
            spin.setSuffix(f" {moment}")
        for spin in (self.spn_wx, self.spn_wy, self.spn_wz):
            spin.setSuffix(f" {line}")
        for spin in (self.spn_fx, self.spn_fy, self.spn_fz):
            spin.setSuffix(" N")
        for spin in self.settlement_spins[:3]:
            spin.setSuffix(f" {length}")
        for spin in self.settlement_spins[3:]:
            spin.setSuffix(" rad")

    def set_selection(self, kind: str | None, ident: int | None):
        """主窗口选中对象后调用，更新面板显示。"""
        if kind == "node":
            self._current_node = ident
            self._current_member = None
            self.lbl_selection.setText(f"当前选中：节点 {ident}")
        elif kind == "member":
            self._current_node = None
            self._current_member = ident
            self.lbl_selection.setText(f"当前选中：杆件 {ident}")
        else:
            self._current_node = None
            self._current_member = None
            self.lbl_selection.setText("未选中对象。在视口中用「选择节点」或「选择杆件」选中后，在这里设置边界条件。")
        self._load_selection_values()

    def _load_selection_values(self):
        """加载当前选中对象的边界条件值到输入框。"""
        if self._filling:
            return
        self._filling = True
        case_name = self.cmb_case.currentText()
        case = next((c for c in self.session.model.get("load_cases", [])
                      if c.get("name") == case_name), None)

        # 节点：支座 + 节点荷载
        if self._current_node is not None:
            self.group_support.setEnabled(True)
            self.group_settlement.setEnabled(True)
            self.group_nodal.setEnabled(True)
            self.group_member.setEnabled(False)
            # 支座
            support = next((s for s in self.session.model.get("supports", [])
                            if int(s["node"]) == self._current_node), None)
            if support:
                self.txt_bc_name.setText(str(support.get("name") or
                                             f"BC-Node-{self._current_node}"))
                fix = support["fix"]
                for i, (name, preset) in enumerate(SUPPORT_PRESETS):
                    if preset == fix:
                        self.cmb_support.setCurrentIndex(i)
                        break
                else:
                    self.cmb_support.setCurrentIndex(0)
            else:
                self.txt_bc_name.setText(f"BC-Node-{self._current_node}")
                self.cmb_support.setCurrentIndex(len(SUPPORT_PRESETS) - 1)  # 自由
            # 节点荷载
            load = [0.0] * 6
            self.txt_nodal_name.setText(f"CF-Node-{self._current_node}")
            if case:
                entry = next((item for item in case.get("nodal_loads", [])
                              if int(item["node"]) == self._current_node), None)
                if entry:
                    load = entry["load"]
                    self.txt_nodal_name.setText(str(entry.get("name") or
                                                   f"CF-Node-{self._current_node}"))
            for sp, val in zip([self.spn_fx, self.spn_fy, self.spn_fz,
                                self.spn_mx, self.spn_my, self.spn_mz], load):
                sp.setValue(float(val))
            # 当前分析步的给定位移
            settlement = [0.0] * 6
            self.txt_settlement_name.setText(
                f"Displacement-Node-{self._current_node}")
            if case:
                entry = next((item for item in case.get("settlements", [])
                              if int(item["node"]) == self._current_node), None)
                if entry:
                    settlement = entry["d"]
                    self.txt_settlement_name.setText(str(
                        entry.get("name") or f"Displacement-Node-{self._current_node}"))
            for spin, value in zip(self.settlement_spins, settlement):
                spin.setValue(float(value))
        # 杆件：杆件荷载
        elif self._current_member is not None:
            self.group_support.setEnabled(False)
            self.group_settlement.setEnabled(False)
            self.group_nodal.setEnabled(False)
            self.group_member.setEnabled(True)
            load = [0.0] * 3
            self.txt_member_name.setText(f"Line-Member-{self._current_member}")
            if case:
                entry = next((item for item in case.get("member_loads", [])
                              if int(item["member"]) == self._current_member), None)
                if entry:
                    load = entry["w"]
                    self.txt_member_name.setText(str(entry.get("name") or
                                                    f"Line-Member-{self._current_member}"))
            for sp, val in zip([self.spn_wx, self.spn_wy, self.spn_wz], load):
                sp.setValue(float(val))
        else:
            self.group_support.setEnabled(False)
            self.group_settlement.setEnabled(False)
            self.group_nodal.setEnabled(False)
            self.group_member.setEnabled(False)
        self._filling = False

    def _on_case_changed(self, _text):
        if not self._filling:
            self._load_selection_values()

    def _new_case(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "新建工况", "工况名称：")
        if ok and name.strip():
            result = self.session.add_load_case(name.strip())
            if result.ok:
                self.refresh()
                self.cmb_case.setCurrentText(name.strip())
                self.changed.emit()
            else:
                self._show_error("新建分析步失败", result.payload)

    @staticmethod
    def _error_text(payload: dict) -> str:
        if payload.get("error"):
            return str(payload["error"])
        errors = payload.get("errors") or payload.get("warnings") or []
        if isinstance(errors, str):
            return errors
        return "\n".join(str(item) for item in errors) or "操作未完成。"

    def _show_error(self, title: str, payload: dict) -> None:
        QMessageBox.warning(self, title, self._error_text(payload))

    def _apply_support(self):
        if self._current_node is None:
            return
        _, fix = SUPPORT_PRESETS[self.cmb_support.currentIndex()]
        result = self.session.set_supports(
            [self._current_node], fix, name=self.txt_bc_name.text().strip())
        if result.ok:
            self._load_selection_values()
            self.changed.emit()
        else:
            self._show_error("边界条件设置失败", result.payload)

    def _clear_support(self):
        if self._current_node is None:
            return
        result = self.session.set_supports(
            [self._current_node], [0] * 6,
            name=self.txt_bc_name.text().strip())
        if result.ok:
            self._load_selection_values()
            self.changed.emit()
        else:
            self._show_error("删除支座失败", result.payload)

    def _apply_settlement(self):
        if self._current_node is None:
            return
        values = [spin.value() for spin in self.settlement_spins]
        result = self.session.set_prescribed_displacement(
            self._current_node, values, self.cmb_case.currentText(),
            self.txt_settlement_name.text().strip())
        if result.ok:
            self.changed.emit()
        else:
            self._show_error("给定位移设置失败", result.payload)

    def _clear_settlement(self):
        if self._current_node is None:
            return
        result = self.session.set_prescribed_displacement(
            self._current_node, [0.0] * 6, self.cmb_case.currentText(),
            self.txt_settlement_name.text().strip())
        if result.ok:
            self._load_selection_values()
            self.changed.emit()
        else:
            self._show_error("删除给定位移失败", result.payload)

    def _apply_nodal_load(self):
        if self._current_node is None:
            return
        load = [self.spn_fx.value(), self.spn_fy.value(), self.spn_fz.value(),
                self.spn_mx.value(), self.spn_my.value(), self.spn_mz.value()]
        result = self.session.set_nodal_load(self._current_node, load,
                                               self.cmb_case.currentText(),
                                               self.txt_nodal_name.text().strip())
        if result.ok:
            self.changed.emit()
        else:
            self._show_error("节点荷载设置失败", result.payload)

    def _clear_nodal_load(self):
        if self._current_node is None:
            return
        result = self.session.set_nodal_load(self._current_node, [0]*6,
                                               self.cmb_case.currentText(),
                                               self.txt_nodal_name.text().strip())
        if result.ok:
            self._load_selection_values()
            self.changed.emit()
        else:
            self._show_error("删除节点荷载失败", result.payload)

    def _copy_nodal_load(self):
        if self._current_node is None:
            return
        from PySide6.QtWidgets import QInputDialog
        default = (self.txt_nodal_name.text().strip() or
                   f"CF-Node-{self._current_node}") + "-Copy"
        name, ok = QInputDialog.getText(self, "复制节点荷载", "新名称：",
                                        text=default)
        if not ok or not name.strip():
            return
        values = [self.spn_fx.value(), self.spn_fy.value(), self.spn_fz.value(),
                  self.spn_mx.value(), self.spn_my.value(), self.spn_mz.value()]
        result = self.session.set_nodal_load(
            self._current_node, values, self.cmb_case.currentText(), name.strip())
        if result.ok:
            self.changed.emit()
        else:
            self._show_error("复制节点荷载失败", result.payload)

    def _apply_member_load(self):
        if self._current_member is None:
            return
        load = [self.spn_wx.value(), self.spn_wy.value(), self.spn_wz.value()]
        result = self.session.set_member_load(self._current_member, load,
                                                self.cmb_case.currentText(),
                                                self.txt_member_name.text().strip())
        if result.ok:
            self.changed.emit()
        else:
            self._show_error("杆件荷载设置失败", result.payload)

    def _clear_member_load(self):
        if self._current_member is None:
            return
        result = self.session.set_member_load(self._current_member, [0]*3,
                                                self.cmb_case.currentText(),
                                                self.txt_member_name.text().strip())
        if result.ok:
            self._load_selection_values()
            self.changed.emit()
        else:
            self._show_error("删除杆件荷载失败", result.payload)

    def _copy_member_load(self):
        if self._current_member is None:
            return
        from PySide6.QtWidgets import QInputDialog
        default = (self.txt_member_name.text().strip() or
                   f"Line-Member-{self._current_member}") + "-Copy"
        name, ok = QInputDialog.getText(self, "复制杆件荷载", "新名称：",
                                        text=default)
        if not ok or not name.strip():
            return
        values = [self.spn_wx.value(), self.spn_wy.value(), self.spn_wz.value()]
        result = self.session.set_member_load(
            self._current_member, values, self.cmb_case.currentText(), name.strip())
        if result.ok:
            self.changed.emit()
        else:
            self._show_error("复制杆件荷载失败", result.payload)
