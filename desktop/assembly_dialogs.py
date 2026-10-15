"""Assembly 模块的对话框：实例（平移 / 旋转）、线性阵列、环形阵列，以及实例管理器。

对话框只收集参数；确定后由窗口调 session 的 create_instance / linear_pattern /
radial_pattern / delete_instance，撤销与校验都在那里。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView,
                               QLineEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .dialog_styles import DialogHeader, HintLabel, button_box, style_dialog

MODES = {"instance": ("创建实例", "把所选杆件复制一份：先绕轴旋转，再平移。重合的节点直接共用。"),
         "linear": ("线性阵列", "沿一个或两个方向等距复制；数量含原件，相邻副本重合的杆件只留一根。"),
         "radial": ("环形阵列", "绕轴每隔一个角度复制一份；数量含原件，梁方向随副本旋转。")}


class _Vector(QWidget):
    """三个分量的输入。"""

    def __init__(self, values=(0.0, 0.0, 0.0), parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.boxes = []
        for v in values:
            box = QDoubleSpinBox(self)
            box.setRange(-1e9, 1e9)
            box.setDecimals(4)
            box.setValue(float(v))
            row.addWidget(box)
            self.boxes.append(box)

    def value(self) -> list[float]:
        return [b.value() for b in self.boxes]


class InstanceDialog(QDialog):
    """实例 / 线性阵列 / 环形阵列的参数。``values()`` 返回 session 工具的关键字参数。"""

    def __init__(self, mode: str, name: str, region_text: str, parent=None):
        super().__init__(parent)
        title, hint = MODES[mode]
        self.mode = mode
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader(title, f"{hint}\n部件：{region_text}"))
        form = QFormLayout()
        self.name = QLineEdit(name, self)
        form.addRow("名称", self.name)
        if mode == "instance":
            self.translate = _Vector((0.0, 0.0, 0.0), self)
            self.axis = _Vector((0.0, 0.0, 1.0), self)
            self.angle = self._number(-360.0, 360.0, 0.0)
            self.point = _Vector((0.0, 0.0, 0.0), self)
            self.translate.setToolTip("沿全局 x、y、z 的平移量，模型长度单位")
            form.addRow("平移量", self.translate)
            form.addRow("旋转轴方向", self.axis)
            form.addRow("转角（度）", self.angle)
            form.addRow("旋转中心", self.point)
        elif mode == "linear":
            self.direction1 = _Vector((1.0, 0.0, 0.0), self)
            self.spacing1 = self._number(-1e9, 1e9, 6.0)
            self.number1 = self._count(2)
            self.direction2 = _Vector((0.0, 1.0, 0.0), self)
            self.spacing2 = self._number(-1e9, 1e9, 0.0)
            self.number2 = self._count(1)
            form.addRow("方向 1", self.direction1)
            form.addRow("间距 1", self.spacing1)
            form.addRow("数量 1（含原件）", self.number1)
            form.addRow("方向 2", self.direction2)
            form.addRow("间距 2", self.spacing2)
            form.addRow("数量 2（含原件）", self.number2)
        else:
            self.number = self._count(4)
            self.angle = self._number(-360.0, 360.0, 90.0)
            self.axis = _Vector((0.0, 0.0, 1.0), self)
            self.point = _Vector((0.0, 0.0, 0.0), self)
            form.addRow("数量（含原件）", self.number)
            form.addRow("角度间隔（度）", self.angle)
            form.addRow("旋转轴方向", self.axis)
            form.addRow("旋转中心", self.point)
        layout.addLayout(form)
        self.hint = HintLabel("", self)
        layout.addWidget(self.hint)
        box = button_box(self)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        style_dialog(self, 460, 320)

    def _number(self, low: float, high: float, value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox(self)
        box.setRange(low, high)
        box.setDecimals(4)
        box.setValue(value)
        return box

    def _count(self, value: int) -> QSpinBox:
        box = QSpinBox(self)
        box.setRange(1, 500)
        box.setValue(value)
        return box

    def _accept(self) -> None:
        if not self.name.text().strip():
            self.hint.setText("请给实例起个名字。")
            return
        self.accept()

    def values(self) -> dict[str, Any]:
        name = self.name.text().strip()
        if self.mode == "instance":
            return {"name": name, "translate": self.translate.value(),
                    "rotate_axis": self.axis.value(), "rotate_angle": self.angle.value(),
                    "rotate_point": self.point.value()}
        if self.mode == "linear":
            return {"name": name, "direction1": self.direction1.value(),
                    "spacing1": self.spacing1.value(), "number1": self.number1.value(),
                    "direction2": self.direction2.value(), "spacing2": self.spacing2.value(),
                    "number2": self.number2.value()}
        return {"name": name, "number": self.number.value(), "angle_deg": self.angle.value(),
                "axis": self.axis.value(), "point": self.point.value()}


def describe_transform(transform: dict[str, Any]) -> str:
    bits = []
    if any(transform.get("translate") or []):
        bits.append("平移 (" + ", ".join(f"{v:g}" for v in transform["translate"]) + ")")
    rotate = transform.get("rotate")
    if rotate:
        bits.append(f"绕 ({', '.join(f'{v:g}' for v in rotate['axis'])}) 转 {rotate['angle_deg']:g}°")
    return "，".join(bits) or "原位"


class InstanceManagerDialog(QDialog):
    """实例管理器：列出实例（所属阵列、变换、杆件数），可删除实例或整个阵列。"""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("实例管理器")
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader("实例管理器",
                                      "独立实例：副本是普通杆件，可单独修改；删除实例会撤掉它新建的杆件与节点。"))
        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(["名称", "阵列", "变换", "杆件", "新节点"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.delete = QPushButton("删除实例", self)
        self.delete.clicked.connect(lambda: self._delete(pattern=False))
        self.delete_pattern = QPushButton("删除整个阵列", self)
        self.delete_pattern.clicked.connect(lambda: self._delete(pattern=True))
        row.addWidget(self.delete)
        row.addWidget(self.delete_pattern)
        layout.addLayout(row)
        close = button_box(self, ok="关闭")
        close.accepted.connect(self.accept)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        style_dialog(self, 680, 360)
        self.refresh()

    def refresh(self) -> None:
        rows = self.window.session.list_instances().payload["instances"]
        self.rows = rows
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            cells = [row["name"], row.get("pattern") or "", describe_transform(row["transform"]),
                     str(row["members"]), str(row["new_nodes"])]
            for c, text in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(text))
        if rows and self.table.currentRow() < 0:
            self.table.selectRow(0)

    def _delete(self, pattern: bool) -> None:
        r = self.table.currentRow()
        if r < 0 or r >= len(self.rows):
            return
        row = self.rows[r]
        name = row.get("pattern") if pattern and row.get("pattern") else row["name"]
        self.window.delete_instance_ui(name)
        self.refresh()
