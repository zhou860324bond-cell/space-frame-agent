"""分析步管理器与幅值曲线对话框。

原来的「分析步」按钮是两个 QInputDialog：选线性/P-Δ/材料，再问增量数，
设一个 `self.analysis_type`。它跟 `src/steps.py` 里的 Step 对象**毫无关系**
——不能增删、不能排序、看不到传播。求解内核已经支持跨步传播与失活，
界面却够不到，这是这个项目反复栽的那个坑：能力只交付了一半。

这两个对话框把那条通路接上。设计上只有一处需要特别照顾：

**传播是隐式的，所以界面必须把「实际生效」摆在最显眼的地方。** 用户在
第三步只写了一行「失活活载」，实际在算的却是第一步那几个荷载减掉活载。
只显示声明的话，这件事就只能靠人对着两份自己比——而比错了不会有任何提示，
结果照样算得出来。所以列表里声明与生效并排，选中一行还会展开这一步相对
上一步的变化。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from . import theme
from .dialog_styles import (DialogHeader, DialogSection, FormRow, HintLabel,
                            button_box, close_box, style_dialog)

#: 分析类型的显示名 ↔ 内核名。材料非线性不在其中，理由见 src/steps.py：
#: 一个分析步里常有几个工况同时生效，而材料非线性不能事后叠加。
ANALYSES = [("线性静力", "linear"), ("P-Delta 二阶弹性", "pdelta")]

#: 内核名 → 显示名。表格里显示 "linear" 是把实现细节漏给用户看。
ANALYSES_BY_KERNEL = [(kernel, label) for label, kernel in ANALYSES]

#: 内置幅值曲线的一句话说明，直接显示在下拉框旁边——RAMP 和 STEP
#: 的区别是这个对话框里最容易选错的一处。
BUILTIN_NOTES = {
    "RAMP": "0→1 线性斜坡：荷载随分析步均匀增长",
    "STEP": "全程恒为 1：一开始就是全值，不参与放大（重力配它）",
}


def _loads_text(loads: dict) -> str:
    """把 {工况: 幅值} 写成人看的形式。

    直接 str() 一个 dict 会在表格里显示成 `{'DL': 'RAMP', 'LL': ...}`——
    引号和花括号占掉本来就不宽的列，真正要看的工况名反而被省略号吃掉。
    """
    return "，".join(f"{case}={amp}" for case, amp in sorted(loads.items()))


def _nodes_text(nodes) -> str:
    return "、".join(str(n) for n in nodes)


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.Stretch)
    return table


class AmplitudeDialog(QDialog):
    """幅值曲线管理器（Abaqus 的 Amplitude Manager）。"""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("幅值曲线")
        style_dialog(self, 560, 480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.addWidget(DialogHeader(
            "幅值曲线",
            "荷载在分析步内怎么随伪时间变化。改变的是加载路径，不是弹性解的终点"))

        body = QVBoxLayout()
        body.setContentsMargins(14, 10, 14, 0)
        body.setSpacing(10)

        listing = DialogSection("已有曲线")
        self.table = _table(["名称", "类型", "表 (伪时间, 系数)"])
        self.table.setMinimumHeight(160)
        listing.addWidget(self.table)
        body.addWidget(listing)

        editor = DialogSection("新建自定义曲线")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：前段加满")
        editor.addLayout(FormRow("名称", self.name_edit))
        self.points_edit = QLineEdit("0,0; 0.3,1; 1,1")
        editor.addLayout(FormRow("表", self.points_edit))
        editor.addWidget(HintLabel(
            "格式：伪时间,系数; 伪时间,系数; …　时间必须严格递增，"
            "表外取端点值不外推"))
        row = QHBoxLayout()
        row.addStretch(1)
        add = QPushButton("新建")
        add.clicked.connect(self._add)
        remove = QPushButton("删除选中")
        remove.clicked.connect(self._remove)
        row.addWidget(remove)
        row.addWidget(add)
        editor.addLayout(row)
        body.addWidget(editor)

        layout.addLayout(body)
        box = close_box(self)
        box.accepted.connect(self.accept)
        layout.addWidget(box)

        self._reload()

    def _reload(self) -> None:
        payload = self.session.list_amplitudes().payload
        rows = [(name, "内置", note)
                for name, note in payload["builtin"].items()]
        rows += [(name, "自定义",
                  "; ".join(f"{t:g},{v:g}" for t, v in points))
                 for name, points in sorted(payload["defined"].items())]
        self.table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                if values[1] == "内置":
                    item.setForeground(Qt.GlobalColor.gray)
                self.table.setItem(r, c, item)

    def _add(self) -> None:
        points = []
        for chunk in self.points_edit.text().split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(",")
            if len(parts) != 2:
                QMessageBox.warning(self, "格式不对",
                                    f"「{chunk}」不是「时间,系数」的形式")
                return
            try:
                points.append([float(parts[0]), float(parts[1])])
            except ValueError:
                QMessageBox.warning(self, "格式不对", f"「{chunk}」里有不是数的东西")
                return
        result = self.session.define_amplitude(self.name_edit.text(),
                                               points=points)
        if not result.ok:
            QMessageBox.warning(self, "没能新建", str(
                result.payload.get("error") or result.payload.get("errors")))
            return
        self.name_edit.clear()
        self._reload()

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        if self.table.item(row, 1).text() == "内置":
            QMessageBox.information(self, "删不掉",
                                    f"{name} 是内置曲线，删不掉")
            return
        result = self.session.delete_amplitude(name)
        if not result.ok:
            QMessageBox.warning(self, "没能删除",
                                str(result.payload.get("error")))
            return
        self._reload()


class StepManagerDialog(QDialog):
    """分析步管理器（Abaqus 的 Step Manager）。

    列表里**声明**与**实际生效**并排。传播是隐式的，只给声明的话，
    "这一步到底在算什么"就只能靠人对着前面几步自己推。
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("分析步")
        style_dialog(self, 720, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.addWidget(DialogHeader(
            "分析步",
            "荷载与边界条件在步之间传播：上一步有而这一步没提的会自动沿用"))

        body = QVBoxLayout()
        body.setContentsMargins(14, 10, 14, 0)
        body.setSpacing(10)

        listing = DialogSection("分析步（按顺序执行）")
        self.table = _table(["分析步", "分析类型", "声明的变化", "实际生效的荷载",
                             "实际生效的支座"])
        self.table.setMinimumHeight(170)
        self.table.itemSelectionChanged.connect(self._show_changes)
        listing.addWidget(self.table)
        self.changes = HintLabel("")
        self.changes.setWordWrap(True)
        listing.addWidget(self.changes)
        body.addWidget(listing)

        editor = DialogSection("新建分析步")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：撤活载")
        editor.addLayout(FormRow("名称", self.name_edit))

        self.analysis_box = QComboBox()
        for label, _ in ANALYSES:
            self.analysis_box.addItem(label)
        editor.addLayout(FormRow("分析类型", self.analysis_box))

        self.increments = QSpinBox()
        self.increments.setRange(1, 1000)
        self.increments.setValue(10)
        editor.addLayout(FormRow("载荷增量数", self.increments))

        activate = QHBoxLayout()
        self.case_box = QComboBox()
        self.amp_box = QComboBox()
        activate.addWidget(QLabel("施加工况"))
        activate.addWidget(self.case_box, 1)
        activate.addWidget(QLabel("幅值"))
        activate.addWidget(self.amp_box, 1)
        editor.addLayout(activate)

        self.deactivate_cases = QLineEdit()
        self.deactivate_cases.setPlaceholderText("工况名，逗号分隔；留空表示不失活")
        editor.addLayout(FormRow("失活工况", self.deactivate_cases))
        self.deactivate_nodes = QLineEdit()
        self.deactivate_nodes.setPlaceholderText("节点号，逗号分隔；留空表示不拆")
        editor.addLayout(FormRow("拆除支座", self.deactivate_nodes))
        editor.addWidget(HintLabel(
            "只写这一步新建或改写的东西。**漏写不等于撤销**——上一步有而这里"
            "没提的会继续生效，要让它消失必须填进上面两栏。"))

        row = QHBoxLayout()
        row.addStretch(1)
        remove = QPushButton("删除选中")
        remove.clicked.connect(self._remove)
        add = QPushButton("新建")
        add.clicked.connect(self._add)
        row.addWidget(remove)
        row.addWidget(add)
        editor.addLayout(row)
        body.addWidget(editor)

        layout.addLayout(body)

        box = button_box(self, ok="求解全部分析步", cancel="关闭")
        box.accepted.connect(self._solve)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        self._reload()

    # --- 数据 ---

    def _reload(self) -> None:
        self.case_box.clear()
        self.case_box.addItems(
            [c["name"] for c in (self.session.model.get("load_cases") or [])]
            or ["default"])
        self.amp_box.clear()
        amplitudes = self.session.list_amplitudes().payload
        self.amp_box.addItems(list(amplitudes["builtin"])
                              + sorted(amplitudes["defined"]))

        payload = self.session.list_steps().payload
        declared = payload.get("declared") or []
        effective = payload.get("effective") or []
        self._effective = effective
        self.table.setRowCount(len(declared))
        for r, item in enumerate(declared):
            eff = effective[r] if r < len(effective) else {}
            declared_bits = []
            if item.get("loads"):
                declared_bits.append("施加 " + _loads_text(item["loads"]))
            if item.get("deactivate_loads"):
                declared_bits.append(
                    "失活 " + "、".join(item["deactivate_loads"]))
            if item.get("supports"):
                declared_bits.append(
                    "改支座 " + _nodes_text(item["supports"]))
            if item.get("deactivate_supports"):
                declared_bits.append(
                    "拆支座 " + _nodes_text(item["deactivate_supports"]))
            cells = [item["name"],
                     dict(ANALYSES_BY_KERNEL).get(item.get("analysis",
                                                           "linear"), "—"),
                     "；".join(declared_bits) or "（无，全部沿用）",
                     _loads_text(eff.get("loads") or {}),
                     _nodes_text(eff.get("supports") or [])]
            for c, text in enumerate(cells):
                cell = QTableWidgetItem(str(text))
                # 列宽有限，生效的荷载一多就会被省略号吃掉；悬停给全文，
                # 否则"到底传下来了什么"又变成看不见的东西。
                cell.setToolTip(str(text))
                self.table.setItem(r, c, cell)
        if payload.get("resolve_error"):
            self.changes.setText(f"结算不出来：{payload['resolve_error']}")
            self.changes.setStyleSheet(f"color: {theme.ERROR}; font-size: 8pt;")

    def _show_changes(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(getattr(self, "_effective", [])):
            return
        changes = self._effective[row].get("changes") or []
        self.changes.setStyleSheet(f"color: {theme.INK_DIM}; font-size: 8pt;")
        self.changes.setText("相对上一步的变化：" + "；".join(changes))

    # --- 动作 ---

    def _add(self) -> None:
        def split(text):
            return [x.strip() for x in text.split(",") if x.strip()]

        try:
            nodes = [int(x) for x in split(self.deactivate_nodes.text())]
        except ValueError:
            QMessageBox.warning(self, "格式不对", "拆除支座那栏只能填节点号")
            return
        result = self.session.add_step(
            self.name_edit.text(),
            analysis=ANALYSES[self.analysis_box.currentIndex()][1],
            loads={self.case_box.currentText(): self.amp_box.currentText()},
            deactivate_loads=split(self.deactivate_cases.text()),
            deactivate_supports=nodes,
            increments=self.increments.value())
        if not result.ok:
            QMessageBox.warning(self, "没能新建", str(
                result.payload.get("error") or result.payload.get("errors")))
            return
        self.name_edit.clear()
        self.deactivate_cases.clear()
        self.deactivate_nodes.clear()
        self._reload()

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        result = self.session.delete_step(name)
        if not result.ok:
            QMessageBox.warning(
                self, "没能删除",
                f"{result.payload.get('error')}\n\n"
                "删掉中间一步会让后面的失活失去对象，那时候整串分析步结算不出来。")
            return
        self._reload()

    def _solve(self) -> None:
        result = self.session.solve_steps()
        if not result.ok:
            QMessageBox.warning(self, "求解失败", str(
                result.payload.get("error") or result.payload))
            return
        lines = [f"{row['step']}（{row['analysis']}）："
                 f"最大挠度 {row['max_deflection_mm']} mm，"
                 f"平衡 {'通过' if row['equilibrium_ok'] else '未通过'}"
                 for row in result.payload["steps"]]
        QMessageBox.information(
            self, "分析步求解完成",
            "\n".join(lines)
            + f"\n\n会话里留的是「{result.payload['inspecting']}」的结果，"
              "后处理都按它来。\n\n" + result.payload["limitation"])
        self.accept()
