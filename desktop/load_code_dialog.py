"""规范组合、活载最不利布置、面荷载导荷，以及边界条件管理器。

这四样求解侧早就有了，界面一直够不到——本轮清的就是这种"能力只交付了
一半"的缺口。它们的共同点是**算错了看不出来**，所以每个对话框都把关键的
那句提醒摆在面板里，而不是留在文档或工具描述里：

* 规范组合：漏一个组合完全看不出来，包络只在给定的组合里取极值；
* 活载布置：只算满布会把跨中弯矩算小 43%；
* 面荷载：单向板还是双向板，导出来的线荷载差一倍；
* 边界条件：改错了最难看出来，所以要能整张表摆出来核对。
"""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout,
                               QHeaderView, QLineEdit, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from .dialog_styles import (DialogHeader, DialogSection, FormRow, HintLabel,
                            button_box, close_box, style_dialog)

#: 荷载类别 → 在 generate_combinations 里的参数名
KINDS = [("恒载", "dead"), ("活载", "live"), ("风载", "wind"),
         ("雪载", "snow"), ("吊车", "crane")]

#: 导荷方式。**选错不会报错，只是总重差一截。** 两种双向板下「受荷宽度」
#: 填的是板的短跨 Lx，不是从属宽度——这两个数在单向板下往往不一样。
LOAD_PATHS = [("单向板（沿一个方向传，宽度=从属宽度）", "one_way"),
              ("双向板·短边梁（三角形，宽度=短跨 Lx）", "two_way_short"),
              ("双向板·长边梁（梯形，宽度=短跨 Lx）", "two_way_long")]

PATTERNS = [("满布", "full"), ("奇数跨", "odd"), ("偶数跨", "even"),
            ("相邻两跨", "adjacent")]


def _table(headers: list[str], stretch: bool = True) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    if stretch:
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
    return table


def _fill(table: QTableWidget, rows: list[list[str]]) -> None:
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            item = QTableWidgetItem(str(text))
            item.setToolTip(str(text))
            table.setItem(r, c, item)


def _case_names(session) -> list[str]:
    return [str(c["name"]) for c in (session.model.get("load_cases") or [])
            if c.get("name")]


def _members_from(text: str) -> list[int] | str:
    """解析杆件输入：编号列表或集合名。集合名原样交给会话层去展开。"""
    raw = text.strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("，", ",").split(",") if p.strip()]
    if all(p.lstrip("-").isdigit() for p in parts):
        return [int(p) for p in parts]
    return raw if len(parts) == 1 else parts


class CombinationDialog(QDialog):
    """按规范生成荷载组合。"""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("按规范生成荷载组合")
        style_dialog(self, 620, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.addWidget(DialogHeader(
            "按规范生成荷载组合",
            "每个可变荷载轮流当控制荷载。漏一个组合完全看不出来"))

        body = QVBoxLayout()
        body.setContentsMargins(14, 10, 14, 0)
        body.setSpacing(10)

        picker = DialogSection("把工况归到荷载类别")
        cases = _case_names(session)
        self.boxes: dict[str, QComboBox] = {}
        for label, key in KINDS:
            box = QComboBox()
            box.addItem("（无）")
            box.addItems(cases)
            self.boxes[key] = box
            picker.addLayout(FormRow(label, box))
        self.standard = QComboBox()
        self.standard.addItems(["GB50068-2018", "GB50009-2012"])
        picker.addLayout(FormRow("规范", self.standard))
        picker.addWidget(HintLabel(
            "GB50068-2018 用 γG=1.3、γQ=1.5；GB50009-2012 用 1.2/1.4。"
            "有风荷载时会另生成一组 γG=1.0——**风吸把柱子往上拔时恒载是有利的**，"
            "那时候用 1.3 反而不保守。"))
        body.addWidget(picker)

        preview = DialogSection("将要生成的组合")
        self.table = _table(["组合名", "系数", "说明"])
        self.table.setMinimumHeight(170)
        preview.addWidget(self.table)
        body.addWidget(preview)

        layout.addLayout(body)
        box = button_box(self, ok="生成并写入模型")
        box.accepted.connect(self._generate)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _picked(self) -> dict:
        return {key: box.currentText()
                for key, box in self.boxes.items()
                if box.currentText() != "（无）"}

    def _generate(self) -> None:
        picked = self._picked()
        if not picked:
            QMessageBox.information(self, "还没选工况",
                                    "至少把一个工况归到某个荷载类别。")
            return
        result = self.session.generate_combinations(
            standard=self.standard.currentText(), **picked)
        if not result.ok:
            QMessageBox.warning(self, "没能生成", str(
                result.payload.get("error") or result.payload.get("errors")))
            return
        combos = result.payload["combos"]
        _fill(self.table, [[c["name"],
                            " + ".join(f"{v:g}×{k}"
                                       for k, v in sorted(c["factors"].items())),
                            c.get("basis", "")] for c in combos])
        QMessageBox.information(
            self, "已生成", f"生成 {len(combos)} 个组合，已写入模型。\n\n"
            "包络只在这些组合里取极值——少一个就是少一个，"
            "而结果看着完全正常。")
        self.accept()


class LivePatternDialog(QDialog):
    """活载最不利布置。"""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("活载最不利布置")
        style_dialog(self, 600, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.addWidget(DialogHeader(
            "活载最不利布置",
            "只算满布会把跨中弯矩算小 43%"))

        body = QVBoxLayout()
        body.setContentsMargins(14, 10, 14, 0)
        body.setSpacing(10)

        form = DialogSection("参数")
        self.members = QLineEdit()
        self.members.setPlaceholderText("梁的编号（逗号分隔）或集合名")
        form.addLayout(FormRow("梁", self.members))
        self.intensity = QDoubleSpinBox()
        self.intensity.setRange(0.0, 1e9)
        self.intensity.setDecimals(3)
        self.intensity.setValue(10.0)
        self.intensity.setSuffix("  kN/m（向下）")
        form.addLayout(FormRow("活载线荷载", self.intensity))
        self.prefix = QLineEdit("LL")
        form.addLayout(FormRow("工况名前缀", self.prefix))
        form.addWidget(HintLabel(
            "跨的归类由几何推断，返回值里会原样给出归组结果——**要核对**："
            "混进了柱或另一方向的梁时，归出来的跨会很怪。\n"
            "这是规范里的简化做法；严格的最不利位置要画影响线逐点找。"))
        body.addWidget(form)

        listing = DialogSection("生成的工况")
        self.table = _table(["工况", "压哪些跨"])
        self.table.setMinimumHeight(150)
        listing.addWidget(self.table)
        body.addWidget(listing)

        layout.addLayout(body)
        box = button_box(self, ok="生成布置工况")
        box.accepted.connect(self._generate)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _generate(self) -> None:
        members = _members_from(self.members.text())
        if not members:
            QMessageBox.information(self, "还没选梁", "先填要布活载的梁。")
            return
        result = self.session.generate_live_patterns(
            members, [0.0, 0.0, -abs(self.intensity.value()) * 1e3],
            prefix=self.prefix.text().strip() or "LL")
        if not result.ok:
            QMessageBox.warning(self, "没能生成", str(
                result.payload.get("error") or result.payload.get("errors")))
            return
        payload = result.payload
        _fill(self.table, [[case["name"],
                            f"{case['loaded_spans']} 跨　杆件 "
                            + "、".join(str(m) for m in case["members"])]
                           for case in payload["cases"]])
        spans = "；".join(
            f"[{s['range'][0]:g}, {s['range'][1]:g}] ← 杆件 "
            + "、".join(str(m) for m in s["members"])
            for s in payload["spans"])
        QMessageBox.information(
            self, "已生成",
            f"归出 {payload['span_count']} 跨，生成 {payload['count']} 个工况。\n\n"
            f"归组结果：{spans}\n\n"
            "要核对——混进了柱或另一方向的梁时，归出来的跨会很怪。")


class AreaLoadDialog(QDialog):
    """面荷载导成线荷载。"""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("面荷载导荷")
        style_dialog(self, 600, 500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.addWidget(DialogHeader(
            "面荷载导荷", "kN/m² 按板的传力方式导成梁上的线荷载"))

        body = QVBoxLayout()
        body.setContentsMargins(14, 10, 14, 0)
        body.setSpacing(10)

        form = DialogSection("参数")
        self.members = QLineEdit()
        self.members.setPlaceholderText("梁的编号（逗号分隔）或集合名")
        form.addLayout(FormRow("梁", self.members))
        self.q = QDoubleSpinBox()
        self.q.setRange(0.0, 1e9)
        self.q.setDecimals(3)
        self.q.setValue(4.0)
        self.q.setSuffix("  kN/m²")
        form.addLayout(FormRow("面荷载", self.q))
        self.width = QDoubleSpinBox()
        self.width.setRange(0.001, 1e6)
        self.width.setDecimals(3)
        self.width.setValue(3.0)
        self.width.setSuffix("  m")
        self.width_row = FormRow("受荷宽度", self.width)
        form.addLayout(self.width_row)
        self.path = QComboBox()
        for label, _ in LOAD_PATHS:
            self.path.addItem(label)
        form.addLayout(FormRow("传力方式", self.path))
        self.case = QComboBox()
        self.case.addItems(_case_names(session) or ["default"])
        form.addLayout(FormRow("工况", self.case))
        form.addWidget(HintLabel(
            "**单向板与双向板导出来的线荷载差一倍**，而选错不会报错——"
            "双向板沿短跨是三角形分布、沿长跨是梯形，不是简单的 q×宽度。\n"
            "两种双向板下「受荷宽度」填的是板的**短跨 Lx**，不是从属宽度。\n"
            "三角形与梯形都精确生成，不走等效均布——等效均布只保证跨中弯矩"
            "相等，支座附近的剪力是另一回事。"))
        body.addWidget(form)

        listing = DialogSection("导出的线荷载")
        self.table = _table(["杆件", "段数", "形状"])
        self.table.setMinimumHeight(140)
        listing.addWidget(self.table)
        body.addWidget(listing)

        layout.addLayout(body)
        box = button_box(self, ok="施加")
        box.accepted.connect(self._apply)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _apply(self) -> None:
        members = _members_from(self.members.text())
        if not members:
            QMessageBox.information(self, "还没选梁", "先填要导荷的梁。")
            return
        result = self.session.apply_area_load(
            members, self.q.value() * 1e3, self.width.value(),
            load_path=LOAD_PATHS[self.path.currentIndex()][1],
            case_name=self.case.currentText())
        if not result.ok:
            QMessageBox.warning(self, "没能施加", str(
                result.payload.get("error") or result.payload.get("errors")))
            return
        detail = result.payload.get("detail") or []
        _fill(self.table, [[e.get("member"), e.get("pieces"), e.get("note", "")]
                           for e in detail])
        QMessageBox.information(
            self, "已施加",
            f"{result.payload['count']} 根梁导上了线荷载，"
            f"共 {result.payload['entries']} 段（{result.payload['load_path']}）。")


class BCManagerDialog(QDialog):
    """边界条件管理器 —— 相当于 Abaqus 的 BC Manager。

    边界条件是**最容易改错又最难看出来**的一类：填错了不报错，结构照样
    算得出来，只是算的不是你要的那个。整张表摆出来才核对得了。
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("边界条件管理器")
        style_dialog(self, 660, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.addWidget(DialogHeader(
            "边界条件管理器",
            "边界条件改错了不会报错，结构照样算——整张表摆出来才核对得了"))

        body = QVBoxLayout()
        body.setContentsMargins(14, 10, 14, 0)
        body.setSpacing(10)

        listing = DialogSection("全部边界条件")
        self.table = _table(["名称", "节点", "类型", "约束 U1 U2 U3 UR1 UR2 UR3",
                             "含义"])
        self.table.setMinimumHeight(240)
        listing.addWidget(self.table)
        row = QHBoxLayout()
        row.addStretch(1)
        remove = QPushButton("删除选中")
        remove.clicked.connect(self._remove)
        row.addWidget(remove)
        listing.addLayout(row)
        body.addWidget(listing)

        layout.addLayout(body)
        box = close_box(self)
        box.accepted.connect(self.accept)
        layout.addWidget(box)

        self._reload()

    def _reload(self) -> None:
        payload = self.session.list_boundary_conditions().payload
        rows = []
        for item in payload.get("boundary_conditions") or []:
            fix = item.get("fix") or [0] * 6
            rows.append([
                item.get("name", "") or "（未命名）",
                "、".join(str(n) for n in (item.get("nodes") or [])),
                item.get("type", "") or "自定义",
                # ● 被约束、○ 自由。六个 0/1 排一行最容易看错，
                # 而看错了不会报错——结构照样算，只是算的不是你要的那个。
                " ".join("●" if v else "○" for v in fix),
                item.get("means", ""),
            ])
        _fill(self.table, rows)

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        result = self.session.delete_boundary_condition(name)
        if not result.ok:
            QMessageBox.warning(self, "没能删除",
                                str(result.payload.get("error")))
            return
        self._reload()
