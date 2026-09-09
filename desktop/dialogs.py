"""建模对话框：参数化建模、七张表编辑、模型 JSON。

**这三样是从网页版搬回来的。** 迁移到桌面端时我只搬了自然语言这一条入口，
把参数化表单和表格编辑落在了 `gui_app.py` 里——桌面端因此比它要替代的
东西功能更少，而且我用"点了预填到对话框"的按钮把这个洞盖住了，
界面看起来是完整的。**一个看起来完整的残缺品，比明显残缺的糟糕得多。**

设计上守两条：

1. **改动一律先校验再落地。** `Session.set_model` 是故意"先存后验"的
   （Agent 要靠它做增量修补），但人工编辑不能这样：一份改坏的表
   把好模型顶掉，用户就只能靠撤销找回来。所以这里自己先 `validate_payload`。
2. **对话框不碰视口、不碰求解。** 它只产出一个模型字典，交回主窗口。
"""

from __future__ import annotations

import json
import math
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QGroupBox, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from . import theme


def _spins(parent, low, high, value, step=0.1, decimals=2) -> QDoubleSpinBox:
    box = QDoubleSpinBox(parent)
    box.setRange(low, high)
    box.setValue(value)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    return box


class ParametricDialog(QDialog):
    """参数化建模。

    规整框架和门式刚架用这个最快——**一句自然语言要等一个网络往返，
    这里填完就出**，而且结果确定可复现，不联网也能用。
    不规整的地方到「模型表格」里逐行改。

    跨度、层高、开间都填成"逗号分隔的一串"，因为**不等跨是常态**：
    只给一个数和一个数量，就永远做不出边跨小、中跨大的常见布置。
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("参数化建模")
        self.setMinimumWidth(460)
        self.result_kind: str | None = None
        self.params: dict[str, Any] = {}

        box = QVBoxLayout(self)
        self.kind = QComboBox(self)
        self.kind.addItems(["规整框架", "坡屋面门式刚架", "不规则折线刚架"])
        self.kind.currentIndexChanged.connect(self._switch)

        top = QFormLayout()
        top.addRow("结构形式", self.kind)
        box.addLayout(top)

        self.pages = QTabWidget(self)
        self.pages.tabBar().hide()          # 用上面的下拉切，别有两处控制
        self.pages.addTab(self._frame_page(), "框架")
        self.pages.addTab(self._portal_page(), "门式")
        self.pages.addTab(self._bent_page(), "不规则")
        box.addWidget(self.pages)

        self.quick_setup = QCheckBox(
            "生成时同时指派属性、柱脚与荷载（高级快捷方式）", self)
        self.quick_setup.setToolTip(
            "默认按 Part → Property → Step → Load 分阶段建模。只有确认这些定义"
            "都适用于整个生成模型时，才展开并使用快捷指派。")
        box.addWidget(self.quick_setup)
        self.quick_group = QGroupBox("高级快捷指派", self)
        common = QFormLayout(self.quick_group)
        # 这里只能指派已经定义好的属性。允许随手输入一个新名字会让建模
        # 表面成功、到求解时才因“未定义材料/截面”失败，错误暴露得太晚。
        self.material = QComboBox(self)
        self.col_sec = QComboBox(self)
        self.beam_sec = QComboBox(self)
        self._fill_choices()
        common.addRow("材料", self.material)
        common.addRow("柱截面", self.col_sec)
        common.addRow("梁 / 斜梁截面", self.beam_sec)
        self.base = QComboBox(self); self.base.addItems(["固接", "铰接"])
        common.addRow("柱脚", self.base)
        common.addRow("", self.beam_release)
        # 建模只创建几何与指派。荷载属于后续 Load 模块；保留这个快捷项，
        # 但默认不偷偷加一个用户没有要求的 20 kN/m。
        self.beam_load = _spins(self, 0.0, 1e4, 0.0, 1.0, 1)
        self.beam_load.setSuffix("  kN/m")
        self.beam_load.setToolTip(
            "施加在梁 / 斜梁上的竖向均布线荷载，向下为正。填 0 表示暂不加荷载")
        common.addRow("梁上均布荷载", self.beam_load)
        box.addWidget(self.quick_group)
        self.quick_group.setVisible(False)
        self.quick_setup.toggled.connect(self.quick_group.setVisible)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setProperty("panel", "hint")
        box.addWidget(self.hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)
        self._switch()

    # --- 页 ---

    def _frame_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.spans = QLineEdit("6, 6, 6", page)
        self.storeys = QLineEdit("3.6, 3.6", page)
        self.bays = QLineEdit("6", page)
        for w, tip in ((self.spans, "沿 X 方向各跨跨度，米。不等跨直接写不同的数"),
                       (self.storeys, "各层层高，自下而上，米"),
                       (self.bays, "沿 Y 方向各开间宽度，米。留空表示单榀平面刚架")):
            w.setToolTip(tip)
        form.addRow("各跨跨度 (m)", self.spans)
        form.addRow("各层层高 (m)", self.storeys)
        form.addRow("各开间宽度 (m)", self.bays)
        self.beam_release = QCheckBox("梁两端释放平面内弯矩（铰接梁）", page)
        return page

    def _portal_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.p_spans = QLineEdit("24", page)
        self.p_spans.setToolTip("各跨跨度，米。多跨连续门式刚架写成 24, 24")
        self.eave = _spins(self, 1.0, 60.0, 7.5, 0.1)
        self.ridge = _spins(self, 0.0, 20.0, 1.2, 0.1)
        self.ridge.setToolTip("屋脊比檐口高出的高度，米。除以半跨即坡度")
        form.addRow("各跨跨度 (m)", self.p_spans)
        form.addRow("檐口高 (m)", self.eave)
        form.addRow("屋脊升高 (m)", self.ridge)
        return page

    def _bent_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        self.b_profile = QLineEdit("0:7.2, 12:8.4, 24:7.2", page)
        self.b_columns = QLineEdit("0, 6, 12, 18, 24", page)
        self.b_levels = QLineEdit("3.6", page)
        self.b_base_levels = QLineEdit("", page)
        self.b_bays = QLineEdit("6, 6", page)
        self.b_profile.setToolTip(
            "屋面或顶层折线的 x:z 坐标点，米；可写任意折点、单坡、双坡和悬挑")
        self.b_columns.setToolTip("各柱轴线的 X 坐标，米；允许不等跨")
        self.b_levels.setToolTip("中间楼层的绝对标高，米；没有则留空")
        self.b_base_levels.setToolTip("逐柱柱脚标高，米；用于台阶地基，数量必须等于柱数")
        self.b_bays.setToolTip("沿 Y 方向各开间宽度，米；留空保持单榀")
        form.addRow("顶层折线 x:z (m)", self.b_profile)
        form.addRow("柱轴线 X (m)", self.b_columns)
        form.addRow("中间层标高 (m)", self.b_levels)
        form.addRow("逐柱柱脚标高 (m)", self.b_base_levels)
        form.addRow("纵向开间 (m)", self.b_bays)
        return page

    def _fill_choices(self) -> None:
        """材料截面下拉只列当前模型中已经定义的名称。"""
        model = self.session.model
        mats = [m["name"] for m in model.get("materials") or []]
        secs = [s["name"] for s in model.get("sections") or []]
        self.material.addItem("未指派（先建几何）", "")
        self.material.addItems(mats)
        if mats:
            self.material.setCurrentIndex(1)
        for combo, default in ((self.col_sec, "COLUMN"), (self.beam_sec, "RAFTER")):
            combo.addItem("未指派（先建几何）", "")
            combo.addItems(secs)
            if default in secs:
                combo.setCurrentText(default)
            elif secs:
                combo.setCurrentIndex(1)
        if secs and len(secs) >= 2 and self.col_sec.currentText() == self.beam_sec.currentText():
            self.beam_sec.setCurrentText(secs[1])

    def _switch(self) -> None:
        idx = self.kind.currentIndex()
        self.pages.setCurrentIndex(idx)
        hints = (
            "沿 X 方向布跨、沿 Y 方向布开间；开间留空则生成单榀平面刚架。",
            "用跨度、檐口和屋脊快速生成标准门式刚架。",
            "用实际轴线坐标、绝对标高和任意顶层折线生成不等跨、错层、"
            "台阶柱脚及多折点屋面；生成后仍可局部增删杆件。",
        )
        self.hint.setText(hints[idx])

    # --- 取值 ---

    @staticmethod
    def _numbers(text: str, what: str) -> list[float]:
        """把"6, 6, 6"解析成 [6,6,6]。**报错要指出是哪一项**，
        只说"格式错误"用户得自己逐个找。"""
        out = []
        for piece in str(text).replace("，", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                value = float(piece)
            except ValueError:
                raise ValueError(f"{what}里的「{piece}」不是数字")
            if value <= 0:
                raise ValueError(f"{what}里的 {piece} 必须大于 0")
            out.append(value)
        return out

    @staticmethod
    def _coordinates(text: str, what: str) -> list[list[float]]:
        points = []
        for piece in str(text).replace("，", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            fields = piece.replace("：", ":").split(":")
            if len(fields) != 2:
                raise ValueError(f"{what}中的「{piece}」应写成 x:z")
            try:
                point = [float(fields[0]), float(fields[1])]
            except ValueError:
                raise ValueError(f"{what}中的「{piece}」含有非数字")
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"{what}中的「{piece}」必须是有限坐标")
            points.append(point)
        if len(points) < 2:
            raise ValueError(f"{what}至少需要两个坐标点")
        return points

    @staticmethod
    def _positions(text: str, what: str) -> list[float]:
        """解析绝对坐标/标高；与跨度不同，0 和负坐标都是合法的。"""
        out = []
        for piece in str(text).replace("，", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                value = float(piece)
            except ValueError:
                raise ValueError(f"{what}里的「{piece}」不是数字")
            if not (-1e100 < value < 1e100):
                raise ValueError(f"{what}里的坐标必须是有限数")
            out.append(value)
        return out

    def _accept(self) -> None:
        try:
            if self.kind.currentIndex() == 0:
                self.result_kind = "frame"
                self.params = dict(
                    spans=self._numbers(self.spans.text(), "各跨跨度"),
                    storeys=self._numbers(self.storeys.text(), "各层层高"),
                    bays=self._numbers(self.bays.text(), "各开间宽度"))
                if not self.params["spans"]:
                    raise ValueError("至少要有一跨")
                if not self.params["storeys"]:
                    raise ValueError("至少要有一层")
            elif self.kind.currentIndex() == 1:
                self.result_kind = "portal"
                self.params = dict(
                    spans=self._numbers(self.p_spans.text(), "各跨跨度"),
                    eave_height=self.eave.value(),
                    ridge_rise=self.ridge.value())
                if not self.params["spans"]:
                    raise ValueError("至少要有一跨")
            else:
                self.result_kind = "bent"
                self.params = dict(
                    profile=self._coordinates(self.b_profile.text(), "顶层折线"),
                    columns=self._positions(self.b_columns.text(), "柱轴线"),
                    levels=self._positions(self.b_levels.text(), "中间层标高"),
                    bays=self._numbers(self.b_bays.text(), "纵向开间"))
                bases = self._positions(self.b_base_levels.text(), "逐柱柱脚标高")
                if bases:
                    self.params["base_levels"] = bases
                if not self.params["columns"]:
                    raise ValueError("至少要有一根柱")
        except ValueError as exc:
            QMessageBox.warning(self, "参数有误", str(exc))
            return

        # 默认只创建几何。材料/截面、边界和荷载属于不同分析对象，像 Abaqus
        # 一样在各自模块中显式定义，避免“点一次生成”顺手制造一批暗参数。
        if not self.quick_setup.isChecked():
            self.params["material"] = ""
            self.params["column_section"] = ""
            beam_key = ("rafter_section" if self.result_kind == "portal"
                        else "beam_section")
            self.params[beam_key] = ""
            self.params["base"] = "free"
            self.accept()
            return

        known_materials = {m["name"] for m in self.session.model.get("materials") or []}
        known_sections = {s["name"] for s in self.session.model.get("sections") or []}
        for name, widget, label, known in (
                ("material", self.material, "材料", known_materials),
                ("column_section", self.col_sec, "柱截面", known_sections),
                ("beam_section", self.beam_sec, "梁截面", known_sections)):
            value = ("" if widget.currentIndex() == 0
                     else widget.currentText().strip())
            if value and value not in known:
                QMessageBox.warning(self, "参数不完整",
                                    f"请选择已经定义的{label}，或选择“未指派”先建几何。")
                return
            self.params[name] = value
        # 同一个界面字段，两个生成器的参数名不一样：
        # 框架用 beam_section / beam_load，门式用 rafter_section / rafter_load。
        # **这类映射最容易漏**——漏了界面看着好好的，一点确定就报
        # "unexpected keyword argument"，而用户完全不知道发生了什么
        if self.result_kind == "portal":
            self.params["rafter_section"] = self.params.pop("beam_section")
        if self.result_kind == "frame" and self.beam_release.isChecked():
            self.params["beam_release"] = True
        self.params["base"] = ("fixed" if self.base.currentText() == "固接"
                               else "pinned")
        load = self.beam_load.value()
        if load > 0:
            key = "rafter_load" if self.result_kind == "portal" else "beam_load"
            self.params[key] = load * 1000.0            # kN/m → N/m
        self.accept()


class TableDialog(QDialog):
    """七张表编辑。

    这是日常改动用得最多的一条路：改一个截面名、挪一个节点坐标、
    删一根杆——**说一句话的时间够点三下了**，而且自然语言在"改哪一个"
    这件事上并不可靠。

    七张表来自 `model_tables.to_tables`，落回去用 `from_tables`。
    CSV 导入导出也在这里，因为它和表格是同一份数据。
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from model_tables import TABLE_NAMES, to_tables

        self.session = session
        self.setWindowTitle("模型表格")
        self.resize(880, 560)
        self.model: dict[str, Any] | None = None

        box = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.tables: dict[str, QTableWidget] = {}
        data = to_tables(session.model)
        from model_tables import LABELS
        for name in TABLE_NAMES:
            rows = data.get(name) or []
            table = self._make_table(rows, name)
            self.tables[name] = table
            self.tabs.addTab(table, f"{LABELS.get(name, name)}（{len(rows)}）")
        box.addWidget(self.tabs, 1)

        hint = QLabel(
            "直接编辑单元格；行尾留空行可新增，清空整行即删除。"
            "改动点「确定」后才会校验并写回模型，校验不通过不会覆盖当前模型。")
        hint.setWordWrap(True)
        hint.setProperty("panel", "hint")
        box.addWidget(hint)

        row = QHBoxLayout()
        for text, slot in (("增行", self._add_row), ("删除选中行", self._del_row),
                           ("导入 CSV…", self._import_csv),
                           ("导出 CSV…", self._export_csv)):
            b = QPushButton(text, self)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        box.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)

    def _make_table(self, rows: list[dict], name: str) -> QTableWidget:
        from model_tables import COLUMNS

        cols = list(COLUMNS[name])
        table = QTableWidget(len(rows) + 3, len(cols), self)   # 留三行空白好加
        table.setHorizontalHeaderLabels(cols)
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        for r, row in enumerate(rows):
            for c, key in enumerate(cols):
                value = row.get(key)
                table.setItem(r, c, QTableWidgetItem(
                    "" if value is None else str(value)))
        return table

    def _current(self) -> QTableWidget:
        return self.tabs.currentWidget()

    def _add_row(self) -> None:
        t = self._current()
        t.insertRow(t.rowCount())

    def _del_row(self) -> None:
        t = self._current()
        for index in sorted((i.row() for i in t.selectionModel().selectedRows()),
                            reverse=True):
            t.removeRow(index)

    def _rows_of(self, name: str) -> list[dict[str, Any]]:
        from model_tables import COLUMNS

        table = self.tables[name]
        cols = list(COLUMNS[name])
        out = []
        for r in range(table.rowCount()):
            row = {}
            for c, key in enumerate(cols):
                item = table.item(r, c)
                row[key] = item.text().strip() if item else ""
            if any(v for v in row.values()):     # 整行空白就是删除
                out.append(row)
        return out

    def collect(self) -> dict[str, list[dict[str, Any]]]:
        from model_tables import TABLE_NAMES
        return {name: self._rows_of(name) for name in TABLE_NAMES}

    def _accept(self) -> None:
        from model_io import validate_payload
        from model_tables import from_tables

        try:
            model = from_tables(self.session.model, self.collect())
        except Exception as exc:                # noqa: BLE001
            QMessageBox.warning(self, "表格有误", str(exc)[:600])
            return
        errors = validate_payload(model)
        if errors:
            # **先校验再落地。** 一份改坏的表把好模型顶掉的话，
            # 用户就只能靠撤销找回来——那是能避免的麻烦
            QMessageBox.warning(
                self, "模型校验未通过",
                "改动未写回。请修正以下问题后重试：\n\n"
                + "\n".join(f"· {e}" for e in errors[:10]))
            return
        self.model = model
        self.accept()

    # --- CSV ---

    def _import_csv(self) -> None:
        import csv

        name = list(self.tables)[self.tabs.currentIndex()]
        path, _ = QFileDialog.getOpenFileName(
            self, f"导入 {name} 表", "", "CSV 文件 (*.csv)")
        if not path:
            return
        from model_tables import COLUMNS
        cols = list(COLUMNS[name])
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        table = self.tables[name]
        table.setRowCount(len(rows) + 3)
        for r, row in enumerate(rows):
            for c, key in enumerate(cols):
                table.setItem(r, c, QTableWidgetItem(str(row.get(key, ""))))

    def _export_csv(self) -> None:
        import csv

        name = list(self.tables)[self.tabs.currentIndex()]
        path, _ = QFileDialog.getSaveFileName(
            self, f"导出 {name} 表", f"{name}.csv", "CSV 文件 (*.csv)")
        if not path:
            return
        from model_tables import COLUMNS
        cols = list(COLUMNS[name])
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(self._rows_of(name))


class JsonDialog(QDialog):
    """模型 JSON 直接编辑。

    表格覆盖不到的角落（比如某个工具新加的字段）用它。**校验同样在写回之前**。
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("模型 JSON")
        self.resize(760, 620)
        self.model: dict[str, Any] | None = None

        box = QVBoxLayout(self)
        self.edit = QPlainTextEdit(self)
        self.edit.setPlainText(
            json.dumps(session.model, ensure_ascii=False, indent=2))
        font = self.edit.font(); font.setFamily("Consolas"); font.setPointSize(10)
        self.edit.setFont(font)
        box.addWidget(self.edit, 1)

        hint = QLabel("整份模型的原始数据。改完点「确定」，"
                      "会先做结构校验，通过才写回。")
        hint.setWordWrap(True); hint.setProperty("panel", "hint")
        box.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)

    def _accept(self) -> None:
        from model_io import validate_payload

        try:
            model = json.loads(self.edit.toPlainText())
        except json.JSONDecodeError as exc:
            QMessageBox.warning(
                self, "JSON 格式有误",
                f"第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}")
            return
        if not isinstance(model, dict):
            QMessageBox.warning(self, "JSON 格式有误", "顶层必须是一个对象。")
            return
        errors = validate_payload(model)
        if errors:
            QMessageBox.warning(
                self, "模型校验未通过",
                "改动未写回。请修正以下问题后重试：\n\n"
                + "\n".join(f"· {e}" for e in errors[:10]))
            return
        self.model = model
        self.accept()
