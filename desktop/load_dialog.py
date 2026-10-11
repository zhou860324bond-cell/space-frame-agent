"""Abaqus 式载荷对话框。

对应 Abaqus Load 模块 → Create Load：
- Concentrated force：集中力（作用在节点上），CF1/CF2/CF3, CM1/CM2/CM3
- Line load：杆件满跨均布、梯形分布，以及杆中任意位置集中力

所有杆件荷载分量都按全局坐标输入；位置 ``a`` 从杆件 i 端量起。
"""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, 
                               QDoubleSpinBox, QFormLayout, QGroupBox,
                               QLabel, QLineEdit, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from . import dialog_styles


NODE_LOAD_TYPES = [("集中力（节点）", "concentrated")]
MEMBER_LOAD_TYPES = [
    ("均布线载荷（整杆或指定区间）", "line"),
    ("梯形线载荷（起点 → 终点）", "trapezoid"),
    ("集中力（杆中指定位置）", "member_point"),
]


class LoadPositionWidget(QWidget):
    """用沿杆距离或原始全局坐标定义作用位置及受载区间。"""

    def __init__(self, model, target_type, target_id, length_unit, parent=None):
        super().__init__(parent)
        self.model, self.target_type, self.target_id = model, target_type, target_id
        self.length = None
        form = QFormLayout(self)
        self.mode = QComboBox(self)
        if target_type == "node":
            self.mode.addItem("当前节点", "full")
        else:
            self.mode.addItem("整根杆件", "full")
            self.mode.addItem("距 i 端距离", "distance")
            if model:
                import numpy as np
                from coordinates import model_member_ends
                self.ends = model_member_ends(model, target_id)
                self.length = float(np.linalg.norm(self.ends[1] - self.ends[0]))
        if model:
            self.mode.addItem("全局坐标", "coordinate")
        form.addRow("位置输入", self.mode)
        self.start = QDoubleSpinBox(self)
        self.end = QDoubleSpinBox(self)
        for spin in (self.start, self.end):
            spin.setRange(0, self.length if self.length is not None else 1e9)
            spin.setDecimals(6)
            spin.setSuffix(f" {length_unit}")
        self.end.setValue(self.length or 0)
        self.start_coordinate = QLineEdit(self)
        self.end_coordinate = QLineEdit(self)
        for edit in (self.start_coordinate, self.end_coordinate):
            edit.setPlaceholderText(f"X, Y, Z（{length_unit}，原始全局坐标）")
        if target_type == "member" and model:
            self.start_coordinate.setText(", ".join(f"{v:.12g}" for v in self.ends[0]))
            self.end_coordinate.setText(", ".join(f"{v:.12g}" for v in self.ends[1]))
        elif model:
            node = next(n for n in model["nodes"] if n["id"] == target_id)
            self.start_coordinate.setText(", ".join(str(node[key]) for key in "xyz"))
        form.addRow("起点 a / 作用位置", self.start)
        form.addRow("终点 b", self.end)
        form.addRow("起点 / 作用点坐标", self.start_coordinate)
        form.addRow("终点坐标", self.end_coordinate)
        self.note = QLabel("集中力只使用作用位置；分布荷载需要起点和终点。", self)
        self.note.setWordWrap(True)
        form.addRow(self.note)
        self.load_type = "line" if target_type == "member" else "concentrated"
        self.mode.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self, *_args):
        point = self.load_type in {"member_point", "concentrated"}
        if self.load_type == "member_point" and self.mode.currentData() == "full":
            self.mode.setCurrentIndex(self.mode.findData("distance"))
        mode = self.mode.currentData()
        self.start.setEnabled(mode == "distance")
        self.end.setEnabled(mode == "distance" and not point)
        self.start_coordinate.setEnabled(mode == "coordinate")
        self.end_coordinate.setEnabled(mode == "coordinate" and not point)
        form = self.layout()
        form.setRowVisible(self.end, not point)
        form.setRowVisible(self.end_coordinate, not point)
        form.setRowVisible(self.start, self.target_type == "member")

    def get_range(self):
        from coordinates import parse_coordinate, station_on_segment

        mode = self.mode.currentData()
        if mode == "full":
            return None, None
        if mode == "distance":
            a, b = self.start.value(), self.end.value()
        else:
            a = station_on_segment(parse_coordinate(self.start_coordinate.text()), *self.ends)
            b = (a if self.load_type == "member_point" else
                 station_on_segment(parse_coordinate(self.end_coordinate.text()), *self.ends))
        if self.load_type == "member_point":
            if self.length is not None and not 0 <= a <= self.length:
                raise ValueError("作用位置超出杆长；请检查距 i 端距离。")
            return a, None
        if not a < b or (self.length is not None and b > self.length):
            raise ValueError("分布荷载必须满足 0 ≤ 起点 a < 终点 b ≤ 杆长；请调整位置。")
        return a, b

    def get_node(self):
        if self.mode.currentData() != "coordinate":
            return self.target_id
        import numpy as np
        from coordinates import parse_coordinate

        point = parse_coordinate(self.start_coordinate.text())
        matches = [n["id"] for n in self.model.get("nodes", [])
                   if np.linalg.norm(np.asarray([n[k] for k in "xyz"]) - point)
                   <= (1e-5 if self.start.suffix().strip() == "mm" else 1e-8)]
        if len(matches) != 1:
            raise ValueError("坐标未唯一对应已有节点；请检查坐标，杆中位置请使用杆件集中力。")
        return matches[0]


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
        start = QGroupBox("起点强度 w1（全局坐标）")
        start_form = QFormLayout(start)
        end = QGroupBox("终点强度 w2（全局坐标）")
        end_form = QFormLayout(end)
        self.start_spins = [self._make_spin(line_unit) for _ in range(3)]
        self.end_spins = [self._make_spin(line_unit) for _ in range(3)]
        for label, spin in zip(("X", "Y", "Z"), self.start_spins, strict=True):
            start_form.addRow(label, spin)
        for label, spin in zip(("X", "Y", "Z"), self.end_spins, strict=True):
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
                               self.force_spins, strict=True):
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
                 current_case: str | None = None, model: dict | None = None):
        super().__init__(parent)
        self.target_type = target_type
        self.target_id = target_id
        self.setWindowTitle(f"创建载荷 — {target_type} {target_id}")
        self.resize(480, 480)

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

        self.position = LoadPositionWidget(
            model, target_type, target_id, "mm" if millimetres else "m", self)
        fields = QWidget(self)
        fields_layout = QVBoxLayout(fields)
        fields_layout.addWidget(self.position)
        fields_layout.addWidget(self.stack)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(120)
        self.scroll.setWidget(fields)
        layout.addWidget(self.scroll, 1)
        # 保留旧部件接口；真实创建流程统一使用下面的定位区域。
        self.point_widget.spn_a.valueChanged.connect(self.position.start.setValue)
        self.point_widget.layout().labelForField(self.point_widget.spn_a).hide()
        self.point_widget.spn_a.hide()
        self.error = QLabel(self)
        self.error.setWordWrap(True)
        layout.addWidget(self.error)

        btns = dialog_styles.button_box(self)
        btns.accepted.connect(self._accept)
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
        if hasattr(self, "position"):
            self.position.load_type = load_type
            self.position.refresh()

    def _accept(self):
        try:
            if self.target_type == "node":
                self.get_target_id()
            else:
                self.get_range()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()

    def get_range(self):
        return self.position.get_range()

    def get_target_id(self):
        return self.position.get_node() if self.target_type == "node" else self.target_id

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
        return self.position.get_range()[0]

    def get_name(self) -> str:
        return self.txt_name.text().strip()

    def get_case(self) -> str:
        return self.cmb_case.currentText().strip()
