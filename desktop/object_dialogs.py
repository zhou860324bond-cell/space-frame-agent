"""Abaqus 式对象对话框：创建（名称、分析步/工况、类型）→ 选区域 → 编辑数值；以及管理器。

对应 Abaqus/CAE 的 Create Boundary Condition / Create Load 两段式对话框、Assign
Section 对话框，和 Boundary Condition / Load / Section Assignment Manager。管理器
对每类对象都是同一组按钮：创建、编辑、复制、改名、删除、抑制、恢复。

对话框只收集输入，不改模型：确定之后由窗口调 session 的对象工具
（create_boundary_condition 等），撤销、校验、再生成都在那里统一完成。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .dialog_styles import DialogHeader, HintLabel, button_box, style_dialog

DOF_LABELS = ("U1", "U2", "U3", "UR1", "UR2", "UR3")

#: 创建边界条件时的类型（Abaqus 的 Types for Selected Step）→ 预设名或 None（自定义）
BC_TYPES = (("位移/转角（逐个勾选）", None), ("固接 ENCASTRE", "ENCASTRE"),
            ("铰接 PINNED", "PINNED"), ("X 对称 XSYMM", "XSYMM"),
            ("Y 对称 YSYMM", "YSYMM"), ("Z 对称 ZSYMM", "ZSYMM"),
            ("弹性支承", "SPRING"))
LOAD_TYPES = (("集中力", "concentrated_force"), ("力矩", "moment"),
              ("线荷载", "line_load"), ("重力", "gravity"),
              ("给定位移（支座沉降）", "settlement"))
LOAD_REGION = {"concentrated_force": "node", "moment": "node",
               "line_load": "member", "gravity": "member", "settlement": "node"}
LOAD_FIELD = {"concentrated_force": "force", "moment": "moment", "line_load": "w",
              "gravity": "g", "settlement": "d"}
KIND_LABEL = {"section_assignments": "截面指派", "beam_orientations": "梁方向",
              "bcs": "边界条件", "loads": "荷载", "constraints": "约束"}
#: Interaction 模块的约束类型（Abaqus 的 Create Constraint 列表里梁模型用得上的几种）
CONSTRAINT_TYPES = (("MPC 刚性连杆 BEAM", "beam"), ("MPC 绑定 TIE", "tie"),
                    ("MPC 铰接 PIN", "pin"), ("运动耦合 Coupling", "coupling"))
CONSTRAINT_HINT = {
    "beam": "从节点随主节点作刚体运动（含刚臂）：偏心连接、加载点、设备质量点。",
    "tie": "从节点全部自由度与主节点相等：用于坐标重合的两层节点。",
    "pin": "只连平动、转动各自独立：重合节点处的铰。",
    "coupling": "所选自由度随主节点作刚体运动；只选 U1 U2 UR3 即水平刚性楼板。",
}


def next_name(prefix: str, existing: set[str]) -> str:
    """Abaqus 式默认名：BC-1、BC-2……取第一个没被占用的。"""
    k = 1
    while f"{prefix}-{k}" in existing:
        k += 1
    return f"{prefix}-{k}"


class CreateObjectDialog(QDialog):
    """第一段：名称、分析步或工况、类型。确定后进入区域选择。"""

    def __init__(self, title: str, name: str, scope_label: str, scopes: list[str],
                 types: tuple[tuple[str, Any], ...], parent=None, editable_scope: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader(title, "确定后在视口中选择区域，或改用集合。"))
        form = QFormLayout()
        self.name = QLineEdit(name, self)
        self.scope = QComboBox(self)
        self.scope.addItems(scopes or [])
        # 荷载的工况可以就地新建：输入一个还没有的名字即可，不必先去别处建工况
        self.scope.setEditable(editable_scope)
        self.kind = QComboBox(self)
        for label, value in types:
            self.kind.addItem(label, value)
        form.addRow("名称", self.name)
        # 约束不属于分析步也不属于工况（Abaqus 的 Constraint 是模型级对象），没有这一行
        self.has_scope = scopes is not None
        if self.has_scope:
            form.addRow(scope_label, self.scope)
        else:
            self.scope.hide()
        form.addRow("类型", self.kind)
        layout.addLayout(form)
        box = button_box(self, ok="继续…")
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        style_dialog(self, 380, 220)

    def _accept(self) -> None:
        if not self.name.text().strip():
            self.name.setFocus()
            return
        if self.has_scope and not self.scope.currentText().strip():
            self.scope.setFocus()
            return
        self.accept()

    def values(self) -> tuple[str, str, Any]:
        return (self.name.text().strip(), self.scope.currentText().strip(),
                self.kind.currentData())


class EditBCDialog(QDialog):
    """第二段：逐个自由度勾选约束，弹性支承填刚度。"""

    def __init__(self, name: str, fix: list[int], spring: list[float] | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"编辑边界条件 {name}")
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader(f"编辑边界条件 {name}",
                                      "勾选的方向位移为零；弹性支承填刚度，同一方向不能两者都有。"))
        grid = QGridLayout()
        self.checks: list[QCheckBox] = []
        self.springs: list[QDoubleSpinBox] = []
        spring = spring or [0.0] * 6
        for k, label in enumerate(DOF_LABELS):
            box = QCheckBox(label, self)
            box.setChecked(bool(fix[k]))
            value = QDoubleSpinBox(self)
            value.setRange(0.0, 1e18)
            value.setDecimals(3)
            value.setValue(float(spring[k]))
            value.setToolTip("弹性支承刚度：平动 力/长度，转动 力·长度/弧度；0 表示没有弹簧")
            grid.addWidget(box, k, 0)
            grid.addWidget(QLabel("弹簧刚度", self), k, 1)
            grid.addWidget(value, k, 2)
            self.checks.append(box)
            self.springs.append(value)
        layout.addLayout(grid)
        self.hint = HintLabel("", self)
        layout.addWidget(self.hint)
        box = button_box(self)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        style_dialog(self, 420, 360)

    def _accept(self) -> None:
        clash = [DOF_LABELS[k] for k in range(6)
                 if self.checks[k].isChecked() and self.springs[k].value() > 0]
        if clash:
            self.hint.setText(f"{'、'.join(clash)} 既固定又给了弹簧刚度，只能取一个：固定会让弹簧失效。")
            return
        if not any(c.isChecked() for c in self.checks) and not any(
                s.value() > 0 for s in self.springs):
            self.hint.setText("至少约束一个方向或给一个弹簧刚度，否则这个边界条件不起作用。")
            return
        self.accept()

    def values(self) -> tuple[list[int], list[float] | None]:
        fix = [int(c.isChecked()) for c in self.checks]
        spring = [s.value() for s in self.springs]
        return fix, (spring if any(spring) else None)


class EditConstraintDialog(QDialog):
    """约束的编辑对话框：类型、主节点、（运动耦合的）被约束自由度。"""

    def __init__(self, name: str, constraint_type: str, control: int, dofs: list[int] | None,
                 nodes: list[int], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"编辑约束 {name}")
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader(f"编辑约束 {name}",
                                      "从节点区域跟随主节点；从节点上不能再有支座。"))
        form = QFormLayout()
        self.kind = QComboBox(self)
        for label, value in CONSTRAINT_TYPES:
            self.kind.addItem(label, value)
        self.kind.setCurrentIndex([v for _, v in CONSTRAINT_TYPES].index(constraint_type))
        self.control = QComboBox(self)
        for nid in nodes:
            self.control.addItem(f"节点 {nid}", nid)
        self.control.setCurrentIndex(max(0, nodes.index(control) if control in nodes else 0))
        form.addRow("类型", self.kind)
        form.addRow("主节点", self.control)
        layout.addLayout(form)
        self.type_hint = HintLabel("", self)
        layout.addWidget(self.type_hint)
        grid = QGridLayout()
        self.checks: list[QCheckBox] = []
        dofs = dofs or [1] * 6
        for k, label in enumerate(DOF_LABELS):
            box = QCheckBox(label, self)
            box.setChecked(bool(dofs[k]))
            grid.addWidget(box, k // 3, k % 3)
            self.checks.append(box)
        layout.addLayout(grid)
        floor = QPushButton("水平刚性楼板（U1 U2 UR3）", self)
        floor.clicked.connect(lambda: [c.setChecked(k in (0, 1, 5))
                                       for k, c in enumerate(self.checks)])
        layout.addWidget(floor)
        self.floor = floor
        self.hint = HintLabel("", self)
        layout.addWidget(self.hint)
        box = button_box(self)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self.kind.currentIndexChanged.connect(lambda *_: self._sync())
        self._sync()
        style_dialog(self, 420, 360)

    def _sync(self) -> None:
        coupling = self.kind.currentData() == "coupling"
        for box in self.checks:
            box.setEnabled(coupling)
        self.floor.setEnabled(coupling)
        self.type_hint.setText(CONSTRAINT_HINT[self.kind.currentData()])

    def _accept(self) -> None:
        if self.kind.currentData() == "coupling" and not any(c.isChecked() for c in self.checks):
            self.hint.setText("运动耦合至少要选一个自由度，否则这个约束不起作用。")
            return
        self.accept()

    def values(self) -> tuple[str, int, list[int] | None]:
        kind = self.kind.currentData()
        dofs = [int(c.isChecked()) for c in self.checks] if kind == "coupling" else None
        return kind, int(self.control.currentData()), dofs


class EditLoadDialog(QDialog):
    """第二段：荷载的三个全局分量。"""

    UNITS = {"concentrated_force": "力", "moment": "力×长度",
             "line_load": "力/长度", "gravity": "加速度", "settlement": "长度"}

    def __init__(self, name: str, load_type: str, vector: list[float], unit_system: str,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"编辑荷载 {name}")
        layout = QVBoxLayout(self)
        system = "N、m" if unit_system == "N-m-Pa" else "N、mm"
        layout.addWidget(DialogHeader(f"编辑荷载 {name}",
                                      f"全局坐标分量，单位：{self.UNITS[load_type]}（{system}）。"))
        form = QFormLayout()
        self.values_ = []
        for k, axis in enumerate(("X", "Y", "Z")):
            spin = QDoubleSpinBox(self)
            spin.setRange(-1e18, 1e18)
            spin.setDecimals(6)
            spin.setValue(float(vector[k]))
            form.addRow(f"{axis} 分量", spin)
            self.values_.append(spin)
        layout.addLayout(form)
        self.hint = HintLabel(
            "支座沉降是强加位移：只在 Initial 中已约束的方向生效，向下沉降填负的 Z 分量。"
            if load_type == "settlement" else "", self)
        layout.addWidget(self.hint)
        box = button_box(self)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        style_dialog(self, 380, 260)

    def _accept(self) -> None:
        if not any(s.value() for s in self.values_):
            self.hint.setText("三个分量都是 0，这个荷载不起作用；请填写数值。")
            return
        self.accept()

    def vector(self) -> list[float]:
        return [s.value() for s in self.values_]


class OrientationDialog(QDialog):
    """梁方向：截面局部 y 轴（n1）的近似方向，全局分量。"""

    PRESETS = (("全局 X", (1.0, 0.0, 0.0)), ("全局 Y", (0.0, 1.0, 0.0)),
               ("全局 Z", (0.0, 0.0, 1.0)))

    def __init__(self, name: str, n1: list[float], count_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("梁方向")
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader("梁方向（n1）", count_text + "　截面高度方向沿 n1 在截面内的投影。"))
        form = QFormLayout()
        self.name = QLineEdit(name, self)
        form.addRow("名称", self.name)
        self.values_ = []
        for k, axis in enumerate(("X", "Y", "Z")):
            spin = QDoubleSpinBox(self)
            spin.setRange(-1.0e6, 1.0e6)
            spin.setDecimals(4)
            spin.setValue(float(n1[k]))
            form.addRow(f"n1 的 {axis} 分量", spin)
            self.values_.append(spin)
        layout.addLayout(form)
        row = QHBoxLayout()
        for label, vec in self.PRESETS:
            button = QPushButton(label, self)
            button.clicked.connect(lambda _=False, v=vec: [s.setValue(x) for s, x in
                                                            zip(self.values_, v, strict=True)])
            row.addWidget(button)
        layout.addLayout(row)
        self.hint = HintLabel("", self)
        layout.addWidget(self.hint)
        box = button_box(self)
        box.accepted.connect(self._accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        style_dialog(self, 380, 300)

    def _accept(self) -> None:
        if not any(s.value() for s in self.values_):
            self.hint.setText("n1 不能是零向量：请给出截面高度方向的近似方向。")
            return
        self.accept()

    def values(self) -> tuple[str, list[float]]:
        return self.name.text().strip(), [s.value() for s in self.values_]


class AssignSectionDialog(QDialog):
    """截面指派：选截面与材料（区域已在视口选好）。"""

    def __init__(self, name: str, sections: list[str], materials: list[str],
                 count_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("截面指派")
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader("截面指派", count_text))
        form = QFormLayout()
        self.name = QLineEdit(name, self)
        self.section = QComboBox(self)
        self.section.addItems(sections)
        self.material = QComboBox(self)
        self.material.addItems(materials)
        form.addRow("名称", self.name)
        form.addRow("截面", self.section)
        form.addRow("材料", self.material)
        layout.addLayout(form)
        box = button_box(self)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        style_dialog(self, 380, 220)

    def values(self) -> tuple[str, str, str]:
        return self.name.text().strip(), self.section.currentText(), self.material.currentText()


def describe_region(feature: dict[str, Any], expanded: dict[str, Any]) -> str:
    region = feature["region"]
    if "set" in region:
        text = f"集合「{region['set']}」"
    else:
        text = "选取"
    ids = expanded.get("nodes") or expanded.get("members") or []
    noun = "节点" if "nodes" in expanded else "杆件"
    text += f"　{len(ids)} 个{noun}"
    if expanded.get("problem"):
        text += f"　⚠ {expanded['problem']}"
    return text


def describe_values(kind: str, feature: dict[str, Any]) -> str:
    if kind == "section_assignments":
        return f"{feature['section']} / {feature['material']}"
    if kind == "beam_orientations":
        return "n1 = (" + ", ".join(f"{v:g}" for v in feature["n1"]) + ")"
    if kind == "constraints":
        label = dict((v, k) for k, v in CONSTRAINT_TYPES)[feature["type"]]
        text = f"{label}，主节点 {feature['control']}"
        if feature["type"] == "coupling":
            text += "（" + " ".join(DOF_LABELS[k] for k, v in enumerate(feature.get("dofs") or [1] * 6)
                                    if v) + "）"
        return text
    if kind == "bcs":
        fixed = [DOF_LABELS[k] for k, v in enumerate(feature["fix"]) if v]
        text = "固定 " + ("、".join(fixed) if fixed else "无")
        if feature.get("spring"):
            text += "；弹性支承"
        return text
    label = dict((v, k) for k, v in LOAD_TYPES)[feature["type"]]
    vec = feature.get(LOAD_FIELD[feature["type"]]) or []
    return f"{label} ({', '.join(f'{v:g}' for v in vec)})"


def step_matrix(model: dict[str, Any], kind: str, features: list[dict]) -> tuple[list[str], dict]:
    """管理器的分析步矩阵（Abaqus 管理器的 Created / Propagated 列）。

    返回 (列名, {对象名: [每列的状态文字]})。边界条件在创建它的分析步「创建」，之后
    仍生效的步「传播」；荷载属于工况，工况在哪一步生效它就在哪一步「生效」。没有
    声明分析步时，各工况独立求解，只有一列。
    """
    import steps as step_module
    declared = model.get("steps") or []
    if not declared:
        if kind == "bcs":
            return ["Initial"], {f["name"]: ["创建" if (f.get("step") or "Initial") == "Initial"
                                             else "（分析步不存在）"] for f in features}
        return ["独立工况"], {f["name"]: ["生效"] for f in features}
    names = ["Initial"] + [s["name"] for s in declared]
    try:
        effective = step_module.resolve(step_module.from_payload(model), model)
    except ValueError as exc:
        return names, {f["name"]: [f"分析步有误：{exc}"] + [""] * (len(names) - 1)
                       for f in features}
    table = {}
    import features as feature_module
    for feature in features:
        cells = []
        if kind == "bcs":
            # 与求解同源：状态来自对象层的 bc_states（创建 / 传播 / 修改 / 失活）
            source = next((f for f in (model.get("features") or {}).get("bcs") or []
                           if f["name"] == feature["name"]), feature)
            cells = [status for _step, _values, status in feature_module.bc_states(model, source)]
        else:
            cells.append("")                                  # 荷载不进 Initial 步
            for eff in effective:
                amp = eff.loads.get(feature["case"])
                cells.append(f"生效（{amp}）" if amp else "")
        table[feature["name"]] = cells
    return names, table


class ObjectManagerDialog(QDialog):
    """一类对象的管理器：Abaqus 的 … Manager。按钮动作交给窗口执行。"""

    COLUMNS = {"section_assignments": ("名称", "区域", "截面 / 材料"),
               "beam_orientations": ("名称", "区域", "方向"),
               "bcs": ("名称", "区域", "约束"),
               "constraints": ("名称", "从节点区域", "类型与主节点"),
               "loads": ("名称", "区域", "荷载", "工况")}

    def __init__(self, window, kind: str):
        super().__init__(window)
        self.window, self.kind = window, kind
        label = KIND_LABEL[kind]
        self.setWindowTitle(f"{label}管理器")
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader(f"{label}管理器",
                                      "对象施加在区域上；区域是集合时，集合变化对象随之更新。"))
        self.table = QTableWidget(0, len(self.COLUMNS[kind]) + 1, self)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemDoubleClicked.connect(lambda *_: self._run("edit"))
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.buttons: dict[str, QPushButton] = {}
        keys = (("create", "创建…"), ("edit", "编辑…"), ("region", "编辑区域…"),
                ("copy", "复制…"), ("rename", "改名…"), ("delete", "删除"),
                ("suppress", "抑制"), ("resume", "恢复"))
        if kind == "bcs":
            # Abaqus 的边界条件管理器：选中某一步的格子，编辑即在该步修改，另有失活 / 激活
            keys += (("deactivate", "失活"), ("activate", "激活"))
        for key, text in keys:
            button = QPushButton(text, self)
            button.clicked.connect(lambda _=False, k=key: self._run(k))
            row.addWidget(button)
            self.buttons[key] = button
        layout.addLayout(row)
        close = button_box(self, ok="关闭")
        close.accepted.connect(self.accept)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.table.currentCellChanged.connect(lambda *_: self._sync_buttons())
        style_dialog(self, 720, 380)
        self.refresh()

    def selected_name(self) -> str | None:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.text() if item else None

    def selected_step(self) -> str | None:
        """当前格子所在的分析步列（边界条件管理器）；不在步列上时为 None。"""
        col = self.table.currentColumn()
        first = len(self.COLUMNS[self.kind])
        if self.kind != "bcs" or col < first or col >= first + len(self.steps):
            return None
        return self.steps[col - first]

    def refresh(self) -> None:
        listed = self.window.session.list_features(self.kind).payload["features"][self.kind]
        current, column = self.selected_name(), self.table.currentColumn()
        steps, matrix = ([], {})
        if self.kind in ("bcs", "loads"):
            steps, matrix = step_matrix(self.window.session.model, self.kind, listed)
        self.steps = steps
        headers = [*self.COLUMNS[self.kind], *steps, "状态"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(listed))
        for row, feature in enumerate(listed):
            cells = [feature["name"], describe_region(feature, feature),
                     describe_values(self.kind, feature)]
            if self.kind == "loads":
                cells.append(feature["case"])
            cells += matrix.get(feature["name"], [])
            cells.append("已抑制" if feature["suppressed"] else "正常")
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, col, item)
            if feature["name"] == current:
                self.table.setCurrentCell(row, max(column, 0))
        if self.table.currentRow() < 0 and listed:
            self.table.selectRow(0)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        has = self.selected_name() is not None
        for key, button in self.buttons.items():
            button.setEnabled(key == "create" or has)

    def _run(self, action: str) -> None:
        self.window.manage_object(self.kind, action, self.selected_name(), manager=self,
                                  step=self.selected_step())
        self.refresh()
