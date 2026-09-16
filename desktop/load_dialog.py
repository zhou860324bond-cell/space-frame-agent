"""Abaqus 式载荷对话框。

对应 Abaqus Load 模块 → Create Load：
- Concentrated force：集中力（作用在节点上），CF1/CF2/CF3, CM1/CM2/CM3
- Line load：杆件满跨均布、梯形分布，以及杆中任意位置集中力

所有杆件荷载分量都按全局坐标输入；位置 ``a`` 从杆件 i 端量起。
"""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, 
                               QDoubleSpinBox, QFormLayout, QGroupBox,
                               QLabel, QLineEdit, QStackedWidget, QVBoxLayout, QWidget)

from . import dialog_styles


NODE_LOAD_TYPES = [("集中力（节点）", "concentrated")]
MEMBER_LOAD_TYPES = [
    ("均布线载荷（杆件满跨）", "line"),
    ("梯形线载荷（i 端 → j 端）", "trapezoid"),
    ("集中力（杆中指定位置）", "member_point"),
]


class ConcentratedForceWidget(QWidget):
    """集中力输入。"""

    def __init__(self, parent=None, moment_unit: str = "N·m"):
        super().__init__(parent)
        form = QFormLayout(self)
        self.spn_cf1 = self._make_spin(" N")
        self.spn_cf2 = self._make_spin(" N")
        self.spn_cf3 = self._make_spin(" N")
        self.spn_cm1 = self._make_spin(f" {moment_unit}")
        self.spn_cm2 = self._make_spin(f" {moment_unit}")
        self.spn_cm3 = self._make_spin(f" {moment_unit}")
        form.addRow("CF1 (X 向力)", self.spn_cf1)
        form.addRow("CF2 (Y 向力)", self.spn_cf2)
        form.addRow("CF3 (Z 向力)", self.spn_cf3)
        form.addRow("CM1 (绕 X 弯矩)", self.spn_cm1)
        form.addRow("CM2 (绕 Y 弯矩)", self.spn_cm2)
        form.addRow("CM3 (绕 Z 弯矩)", self.spn_cm3)

    def _make_spin(self, suffix):
        sp = QDoubleSpinBox()
        sp.setRange(-1e9, 1e9)
        sp.setValue(0)
        sp.setDecimals(2)
        sp.setSuffix(suffix)
        return sp

    def get_load(self) -> list[float]:
        return [self.spn_cf1.value(), self.spn_cf2.value(), self.spn_cf3.value(),
                self.spn_cm1.value(), self.spn_cm2.value(), self.spn_cm3.value()]


class LineLoadWidget(QWidget):
    """线载荷输入。"""

    def __init__(self, parent=None, line_unit: str = "N/m"):
        super().__init__(parent)
        self.line_unit = line_unit
        form = QFormLayout(self)
        self.spn_p1 = self._make_spin()
        self.spn_p2 = self._make_spin()
        self.spn_p3 = self._make_spin()
        form.addRow("P1 (X 向均布力)", self.spn_p1)
        form.addRow("P2 (Y 向均布力)", self.spn_p2)
        form.addRow("P3 (Z 向均布力)", self.spn_p3)

    def _make_spin(self):
        sp = QDoubleSpinBox()
        sp.setRange(-1e6, 1e6)
        sp.setValue(0)
        sp.setDecimals(2)
        sp.setSuffix(f" {self.line_unit}")
        return sp

    def get_load(self) -> list[float]:
        return [self.spn_p1.value(), self.spn_p2.value(), self.spn_p3.value()]


class TrapezoidLoadWidget(QWidget):
    """梯形线载荷：分别输入杆件 i、j 端的全局强度。"""

    def __init__(self, parent=None, line_unit: str = "N/m"):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        start = QGroupBox("i 端强度 w1（全局坐标）")
        start_form = QFormLayout(start)
        end = QGroupBox("j 端强度 w2（全局坐标）")
        end_form = QFormLayout(end)
        self.start_spins = [self._make_spin(line_unit) for _ in range(3)]
        self.end_spins = [self._make_spin(line_unit) for _ in range(3)]
        for label, spin in zip(("X", "Y", "Z"), self.start_spins):
            start_form.addRow(label, spin)
        for label, spin in zip(("X", "Y", "Z"), self.end_spins):
            end_form.addRow(label, spin)
        layout.addWidget(start)
        layout.addWidget(end)

    @staticmethod
    def _make_spin(unit: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(2)
        spin.setSuffix(f" {unit}")
        return spin

    def get_load(self) -> list[float]:
        return [spin.value() for spin in self.start_spins]

    def get_end_load(self) -> list[float]:
        return [spin.value() for spin in self.end_spins]


class MemberPointLoadWidget(QWidget):
    """杆中集中力：全局力分量 + 从 i 端量起的位置。"""

    def __init__(self, parent=None, length_unit: str = "m"):
        super().__init__(parent)
        form = QFormLayout(self)
        self.force_spins = [self._make_spin("N") for _ in range(3)]
        for label, spin in zip(("P1 (X 向力)", "P2 (Y 向力)", "P3 (Z 向力)"),
                               self.force_spins):
            form.addRow(label, spin)
        self.spn_a = self._make_spin(length_unit, maximum=1e9)
        self.spn_a.setMinimum(0.0)
        form.addRow("位置 a（距 i 端）", self.spn_a)
        note = QLabel("a 必须位于杆长范围内；分量按全局坐标输入。")
        note.setWordWrap(True)
        form.addRow(note)

    @staticmethod
    def _make_spin(unit: str, maximum: float = 1e9) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-maximum, maximum)
        spin.setDecimals(3)
        spin.setSuffix(f" {unit}")
        return spin

    def get_load(self) -> list[float]:
        return [spin.value() for spin in self.force_spins]

    def get_position(self) -> float:
        return self.spn_a.value()


class GravityWidget(QWidget):
    """重力输入。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.spn_g1 = self._make_spin()
        self.spn_g2 = self._make_spin()
        self.spn_g3 = self._make_spin()
        self.spn_g3.setValue(-9.81)  # 默认 Z 向向下
        form.addRow("G1 (X 向加速度)", self.spn_g1)
        form.addRow("G2 (Y 向加速度)", self.spn_g2)
        form.addRow("G3 (Z 向加速度)", self.spn_g3)
        form.addRow(QLabel("重力 = 密度 × 加速度，自动施加到所有杆件"))

    def _make_spin(self):
        sp = QDoubleSpinBox()
        sp.setRange(-100, 100)
        sp.setValue(0)
        sp.setDecimals(3)
        sp.setSuffix(" m/s²")
        return sp

    def get_load(self) -> list[float]:
        return [self.spn_g1.value(), self.spn_g2.value(), self.spn_g3.value()]


class LoadDialog(QDialog):
    """载荷对话框。"""

    def __init__(self, target_type: str = "node", target_id: int = 0,
                 parent=None, units: str = "N-m-Pa",
                 cases: list[str] | None = None,
                 current_case: str | None = None):
        super().__init__(parent)
        self.target_type = target_type
        self.target_id = target_id
        self.setWindowTitle(f"创建载荷 — {target_type} {target_id}")
        self.resize(360, 320)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.txt_name = QLineEdit(
            f"CF-Node-{target_id}" if target_type == "node"
            else f"Line-Member-{target_id}")
        form.addRow("名称", self.txt_name)
        self.cmb_case = QComboBox()
        self.cmb_case.addItems(cases or ["Load-1"])
        if current_case and self.cmb_case.findText(current_case) >= 0:
            self.cmb_case.setCurrentText(current_case)
        form.addRow("分析步", self.cmb_case)

        self.cmb_type = QComboBox()
        choices = NODE_LOAD_TYPES if target_type == "node" else MEMBER_LOAD_TYPES
        for label, key in choices:
            self.cmb_type.addItem(label, key)
        self.cmb_type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("载荷类型", self.cmb_type)
        layout.addLayout(form)

        # 堆叠输入区
        self.stack = QStackedWidget()
        millimetres = units == "N-mm-MPa"
        self.cf_widget = ConcentratedForceWidget(
            moment_unit="N·mm" if millimetres else "N·m")
        self.ll_widget = LineLoadWidget(
            line_unit="N/mm" if millimetres else "N/m")
        self.trap_widget = TrapezoidLoadWidget(
            line_unit="N/mm" if millimetres else "N/m")
        self.point_widget = MemberPointLoadWidget(
            length_unit="mm" if millimetres else "m")
        self.stack.addWidget(self.cf_widget)
        self.stack.addWidget(self.ll_widget)
        self.stack.addWidget(self.trap_widget)
        self.stack.addWidget(self.point_widget)
        layout.addWidget(self.stack)

        btns = dialog_styles.button_box(self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._on_type_changed(0)

    def _on_type_changed(self, _index):
        load_type = self.get_load_type()
        if load_type == "concentrated":
            self.stack.setCurrentWidget(self.cf_widget)
        elif load_type == "line":
            self.stack.setCurrentWidget(self.ll_widget)
        elif load_type == "trapezoid":
            self.stack.setCurrentWidget(self.trap_widget)
        elif load_type == "member_point":
            self.stack.setCurrentWidget(self.point_widget)

    def get_load_type(self) -> str:
        return str(self.cmb_type.currentData())

    def get_load(self) -> list[float]:
        w = self.stack.currentWidget()
        return w.get_load()

    def get_end_load(self) -> list[float] | None:
        if self.get_load_type() != "trapezoid":
            return None
        return self.trap_widget.get_end_load()

    def get_position(self) -> float | None:
        if self.get_load_type() != "member_point":
            return None
        return self.point_widget.get_position()

    def get_name(self) -> str:
        return self.txt_name.text().strip()

    def get_case(self) -> str:
        return self.cmb_case.currentText().strip()
