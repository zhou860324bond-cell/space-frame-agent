"""两个停靠面板：建模过程时间线、结果表。

放在一起是因为它们回答的是同一类问题——**"这个状态/这个数字是从哪来的"**。
时间线回答"模型怎么变成现在这样"，结果表回答"这个数字出现在哪根杆、哪个工况"。

两个面板都**只显示和转发，不算不改**。点了哪一行由主窗口决定怎么办。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout,
                               QHeaderView, QHBoxLayout, QLabel, QListWidget, QMenu,
                               QListWidgetItem, QPushButton, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QTextBrowser, QToolButton,
                               QVBoxLayout, QWidget, QWidgetAction)

from . import glyphs
from .qt_style import ElidedLabel

STEP_INDEX = Qt.ItemDataRole.UserRole + 1
LOCATE = Qt.ItemDataRole.UserRole + 2
ROW_MARK = Qt.ItemDataRole.UserRole + 3


class TimelinePanel(QWidget):
    """建模过程时间线。

    数据一直都在（`BuildHistory` 每步都存了深拷贝快照），但之前界面上
    只有模型树里孤零零一行"建模过程（3 步）"，点不开——**存了却不给看，
    等于没存**。

    点任意一行就跳回那一步的模型。这也是"可追溯"这个说法在界面上的兑现处：
    与其在报告里写一句"过程可追溯"，不如让人自己拖一遍。
    """

    goto_step = Signal(int)          # 列表下标；-1 表示回到空模型

    def __init__(self, parent=None):
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(4)

        self.hint = QLabel("双击任意一步，可回退至该步骤对应的模型状态")
        self.hint.setProperty("panel", "hint")
        self.hint.setWordWrap(True)
        box.addWidget(self.hint)

        self.list = QListWidget(self)
        self.list.setAlternatingRowColors(False)
        self.list.setWordWrap(True)          # 摘要长了折行，不要横向滚动条
        self.list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setMinimumHeight(150)      # 和模型树并排时不至于被挤成两行
        self.list.itemDoubleClicked.connect(self._on_activate)
        box.addWidget(self.list, 1)

    def _on_activate(self, item: QListWidgetItem) -> None:
        self.goto_step.emit(int(item.data(STEP_INDEX)))

    def rebuild(self, history) -> None:
        self.list.clear()
        head = QListWidgetItem("0　（空模型）")
        head.setData(STEP_INDEX, -1)
        self.list.addItem(head)

        for k in range(len(history)):
            step = history[k]
            mark = glyphs.CROSS if not step.ok else " "
            item = QListWidgetItem(
                f"{step.index + 1}　{mark} {step.summary}\n     {step.changed}")
            item.setData(STEP_INDEX, k)
            if not step.ok:
                item.setForeground(Qt.GlobalColor.gray)
                item.setToolTip(step.error or "该操作未通过校验")
            self.list.addItem(item)

        # 当前所在的那一步高亮，并滚动到可见处——**用户得看得出自己在哪**，
        # 撤销几步之后如果列表没有任何反应，会以为撤销没生效
        row = history.cursor + 1
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)
            self.list.scrollToItem(self.list.item(row))
        n = len(history)
        self.hint.setText(
            f"共 {n} 步，当前位于第 {history.cursor + 1} 步。双击可跳转。"
            if n else "尚无建模操作记录。")


class ResultPanel(QWidget):
    """结果表。

    这块原来是弹一个消息框、把 JSON 塞进"详细信息"里——那是给开发者看的。
    工程结果该是表：能扫、能排序、能点一行就在视口里定位到那根杆。

    **每一列都带单位**，而且位置列写坐标不写编号——编号规则是生成器定的，
    用户并不知道，光给"杆件 27"等于没说。
    """

    locate = Signal(str, int)        # ("member" | "node", 编号)
    display_changed = Signal(object)
    extreme_requested = Signal()
    stress_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(4)

        self.caption = ElidedLabel("尚无分析结果")
        self.caption.setProperty("result", "title")
        box.addWidget(self.caption)
        self.summary = QWidget(self)
        self.summary.setObjectName("resultSummary")
        self.summary_grid = QGridLayout(self.summary)
        self.summary_grid.setContentsMargins(0, 4, 0, 4)
        self.summary_grid.setSpacing(8)
        self.summary_cards = []
        self.summary.hide()
        box.addWidget(self.summary)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("云图量程"))
        self.range_mode = QComboBox(self)
        self.range_mode.addItem("95% 裁剪", 95.0)
        self.range_mode.addItem("满量程", None)
        # 这个下拉框最需要解释，原先却是这一排里唯一没有 tooltip 的：
        # 默认裁剪意味着色标上限**不是**真实峰值（实测一个门式刚架，
        # p95 只有峰值的 58%），照色标读数会低估。不写出来没人猜得到。
        self.range_mode.setToolTip(
            """色标上限取哪个值。

95% 裁剪（默认）：上限取 |值| 的 95 分位，超出部分饱和成极值色。
刚架内力是重尾分布——实测一个两层两跨框架，55% 的样点落在色带最底下
的 20%。满量程画出来，绝大多数构件是同一个颜色，云图等于没有信息。
代价是：色标上限不是真实峰值，照色标读数会低估。

满量程：上限取真实峰值，颜色可以直接对着色标读数，代价是大部分构件
挤在色带低端。要照图读具体数值时用这个。

两种模式下峰值本身都不受影响：「标注极值」的标签给的始终是真实峰值、
以及它所在的杆件与位置。""")
        controls.addWidget(self.range_mode)
        controls.addWidget(QLabel("分级"))
        self.levels = QComboBox(self)
        # 默认连续：沿杆平滑渐变，一眼看出内力怎么沿杆走。分级把合弯矩这类
        # 先降后升的量切成一圈圈色环，整张图像彩色条纹，读不出走势；
        # 要按色标读区间数值时再切到分级。
        self.levels.addItem("连续", 0)
        for n in (6, 8, 10, 12, 16, 20, 24):
            self.levels.addItem(f"{n} 级", n)
        self.levels.setCurrentIndex(0)
        self.levels.setToolTip(
            "连续：沿杆平滑渐变，看内力走势。\n"
            "N 级：切成 N 级色块，每一段颜色对应色标上一个可读的区间，"
            "照图读数时用。")
        controls.addWidget(self.levels)
        controls.addWidget(QLabel("色系"))
        self.palette = QComboBox(self)
        from . import theme as _theme
        for key, label in _theme.CONTOUR_PALETTES.items():
            self.palette.addItem(label, key)
        self.palette.setCurrentIndex(
            max(0, self.palette.findData(_theme.DEFAULT_PALETTE)))
        self.palette.setToolTip(
            "彩虹谱是 Abaqus 的经典色序，相邻两级色差最大、最好分。\n"
            "代价是明度不单调：色盲用户和黑白打印读不出高低顺序，\n"
            "那两种场合用「蓝—灰—红」或「单蓝」。")
        controls.addWidget(self.palette)
        self.shading = QCheckBox("立体", self)
        self.shading.setChecked(True)
        self.shading.setToolTip(
            "给云图打一层柔和的光，圆管看得出是圆的。\n"
            "要严格照色标读数就关掉：打光后同一个数值在向光面和背光面\n"
            "会差出一点色差。")
        controls.addWidget(self.shading)
        controls.addWidget(QLabel("符号"))
        self.sign_mode = QComboBox(self)
        self.sign_mode.addItem("全部", "all")
        self.sign_mode.addItem("仅正值", "positive")
        self.sign_mode.addItem("仅负值", "negative")
        controls.addWidget(self.sign_mode)
        self.overlay_deformed = QCheckBox("叠加变形", self)
        controls.addWidget(self.overlay_deformed)
        self.show_extrema = QCheckBox("标注极值", self)
        self.show_extrema.setChecked(True)
        controls.addWidget(self.show_extrema)
        self.btn_extreme = QPushButton("定位当前分量极值", self)
        self.btn_extreme.clicked.connect(lambda: self.extreme_requested.emit())
        controls.addWidget(self.btn_extreme)
        self.btn_stress = QPushButton("截面正应力", self)
        self.btn_stress.clicked.connect(lambda: self.stress_requested.emit())
        controls.addWidget(self.btn_stress)
        controls.addStretch(1)
        # 显示参数在弹出面板中排列，避免一条长工具栏撑大结果窗口。
        items = [controls.itemAt(index).widget() for index in range(controls.count())]
        while controls.count():
            controls.takeAt(0)
        self.display_menu = QMenu(self)
        self.display_controls = QWidget(self.display_menu)
        grid = QGridLayout(self.display_controls)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        for line, (label, field) in enumerate(((0, 1), (2, 3), (4, 5), (7, 8))):
            grid.addWidget(items[label], line, 0)
            grid.addWidget(items[field], line, 1)
        for line, index in enumerate((6, 9, 10, 11, 12), start=4):
            grid.addWidget(items[index], line, 0, 1, 2)
        display_action = QWidgetAction(self.display_menu)
        display_action.setDefaultWidget(self.display_controls)
        self.display_menu.addAction(display_action)
        for widget in (self.range_mode, self.sign_mode, self.levels, self.palette):
            widget.currentIndexChanged.connect(self._emit_display_options)
        self.shading.stateChanged.connect(self._emit_display_options)
        self.overlay_deformed.stateChanged.connect(self._emit_display_options)
        self.show_extrema.stateChanged.connect(self._emit_display_options)

        self.table = QTableWidget(self)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_select)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("结果筛选"))
        self.row_filter = QComboBox(self)
        for text, mark in (("全部", None), ("超限", "fail"),
                           ("无法判定", "unclear")):
            self.row_filter.addItem(text, mark)
        self.row_filter.currentIndexChanged.connect(self._filter_rows)
        self.table.horizontalHeader().sortIndicatorChanged.connect(self._filter_rows)
        filter_row.addWidget(self.row_filter)
        self.filter_count = QLabel(self)
        filter_row.addWidget(self.filter_count, 1)
        box.addLayout(filter_row)
        filter_row = QHBoxLayout()
        filter_row.addStretch(1)
        self.all_columns = QCheckBox("全部计算列", self)
        self.all_columns.toggled.connect(self._show_columns)
        self.all_columns.hide()
        self._primary_columns = None
        filter_row.addWidget(self.all_columns)
        self.details_button = QToolButton(self)
        self.details_button.setText("计算说明")
        self.details_button.setCheckable(True)
        filter_row.addWidget(self.details_button)
        self.display_button = QToolButton(self)
        self.display_button.setText("显示设置")
        self.display_button.setMenu(self.display_menu)
        self.display_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        filter_row.addWidget(self.display_button)
        box.addLayout(filter_row)
        self.notes = QTextBrowser(self)
        self.notes.setMaximumHeight(140)
        self.notes.setMinimumHeight(56)
        self.notes.hide()
        self.details_button.toggled.connect(self.notes.setVisible)
        box.addWidget(self.notes)
        box.addWidget(self.table, 1)

    def display_options(self) -> dict:
        return {"percentile": self.range_mode.currentData(),
                "sign": self.sign_mode.currentData(),
                "levels": self.levels.currentData(),
                "palette": self.palette.currentData(),
                "shading": self.shading.isChecked(),
                "overlay_deformed": self.overlay_deformed.isChecked(),
                "show_extrema": self.show_extrema.isChecked()}

    def apply_options(self, options: dict) -> None:
        """按保存下来的选项摆好各控件，最后只发一次 display_changed。

        认不出的值（比如旧版本存的色系名）就跳过、保留当前值——设置文件
        不该有本事让界面进入一个下拉框里不存在的状态。
        """
        combos = {"percentile": self.range_mode, "sign": self.sign_mode,
                  "levels": self.levels, "palette": self.palette}
        checks = {"shading": self.shading, "overlay_deformed": self.overlay_deformed,
                  "show_extrema": self.show_extrema}
        widgets = list(combos.values()) + list(checks.values())
        for widget in widgets:
            widget.blockSignals(True)
        try:
            for key, combo in combos.items():
                if key in options:
                    index = combo.findData(options[key])
                    if index >= 0:
                        combo.setCurrentIndex(index)
            for key, box in checks.items():
                if key in options:
                    box.setChecked(bool(options[key]))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._emit_display_options()

    def _emit_display_options(self, *_args) -> None:
        self.display_changed.emit(self.display_options())

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        target = item.data(LOCATE) if item else None
        if target:
            self.locate.emit(target[0], int(target[1]))

    def _filter_rows(self, *_args) -> None:
        """严重度随首列单元格排序，筛选不改变编号与定位关联。"""
        wanted = self.row_filter.currentData()
        visible = 0
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 0)
            show = wanted is None or (cell is not None and cell.data(ROW_MARK) == wanted)
            self.table.setRowHidden(row, not show)
            visible += int(show)
        self.filter_count.setText(f"显示 {visible}/{self.table.rowCount()} 行")

    # 超限用红色；无法判定用琥珀色，避免把缺参数误读为不合格。
    _MARK_TINT = {"fail": "#f7dede", "unclear": "#fbf0da"}

    def _show_columns(self, *_args) -> None:
        for column in range(self.table.columnCount()):
            label = self.table.horizontalHeaderItem(column).text()
            self.table.setColumnHidden(column, self._primary_columns is not None
                                       and not self.all_columns.isChecked()
                                       and label not in self._primary_columns)

    def _show_overview(self, cards: list[dict]) -> None:
        for card in self.summary_cards:
            card.setParent(None)
            card.deleteLater()
        self.summary_cards = []
        for data in cards:
            card = QFrame(self.summary)
            card.setProperty("resultCard", data.get("state", "metric"))
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 8, 12, 8)
            layout.setSpacing(4)
            label = ElidedLabel(str(data["label"]), card)
            label.setProperty("result", "label")
            value = ElidedLabel(str(data["value"]), card)
            value.setProperty("result", "value")
            layout.addWidget(label)
            layout.addWidget(value)
            detail = QToolButton(card)
            target = data.get("target")
            detail.setText(f"定位杆件 {target[1]}" if target else "查看清单")
            detail.setToolTip(str(data.get("detail", "")))
            detail.setProperty("result", "detail")
            detail.setMinimumWidth(0)
            detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            if target := data.get("target"):
                detail.clicked.connect(lambda _=False, target=target: self.locate.emit(*target))
            elif mark := data.get("filter"):
                detail.clicked.connect(lambda _=False, mark=mark: self.row_filter.setCurrentIndex(
                    self.row_filter.findData(mark)))
            else:
                detail.setEnabled(False)
            layout.addWidget(detail)
            card.ensurePolished()
            card.setMinimumHeight(92)
            self.summary_cards.append(card)
        self.summary.setVisible(bool(cards))
        self._layout_overview()

    def _layout_overview(self) -> None:
        columns = 4 if self.width() >= 480 else 2
        for index, card in enumerate(self.summary_cards):
            self.summary_grid.addWidget(card, index // columns, index % columns)
        self.summary.setMinimumHeight(((len(self.summary_cards) + columns - 1) // columns) * 100)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_overview()

    def show_rows(self, title: str, columns: list[str],
                  rows: list[list[Any]],
                  locators: list[tuple[str, int] | None] | None = None,
                  marks: list[str | None] | None = None, *,
                  overview: list[dict] | None = None,
                  primary_columns: set[str] | None = None) -> None:
        """填表。

        `locators[i]` 说明第 i 行对应视口里的哪个对象，可为 None。
        `marks[i]` 是该行的严重度（fail / unclear / pass），用来着色——
        校核表动辄十几行，逐格去读"结论"那一列不现实，超限的行要自己跳出来。
        """
        self.table.setSortingEnabled(False)     # 填的过程中排序会打乱行序
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                cell = QTableWidgetItem()
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # 数值列存成数值再显示，**否则排序会按字符串排**，
                    # "9" 会排在 "10" 后面
                    cell.setData(Qt.ItemDataRole.DisplayRole, value)
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                else:
                    cell.setText("" if value is None else str(value))
                if c == 0 and locators and r < len(locators) and locators[r]:
                    cell.setData(LOCATE, locators[r])
                if c == 0 and marks and r < len(marks):
                    cell.setData(ROW_MARK, marks[r])
                tint = (self._MARK_TINT.get(marks[r])
                        if marks and r < len(marks) else None)
                if tint:
                    cell.setBackground(QColor(tint))
                self.table.setItem(r, c, cell)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._primary_columns = primary_columns
        self.all_columns.setVisible(primary_columns is not None)
        self.all_columns.setChecked(False)
        self._show_columns()
        self._show_overview(overview or [])
        self._set_caption(title, concise=f"强度验算 · {len(rows)} 根杆件" if overview else None)
        self.row_filter.setEnabled(bool(marks))
        if not marks:
            self.row_filter.setCurrentIndex(0)
        self._filter_rows()

    def _set_caption(self, title: str, concise: str | None = None) -> None:
        """标题第一行是结论，其余是限制与警告。

        标题保持单行，完整结论、限制与警告保存在可展开的计算说明中。
        """
        head, *rest = [line for line in str(title).split("\n") if line.strip()] or ["结果"]
        self.caption.setText(concise or head)
        self.caption.setToolTip(head)
        details = "\n\n".join(([head] if concise else []) + rest)
        self.notes.setPlainText(details)
        self.details_button.setVisible(bool(details))
        self.details_button.setChecked(False)

    def show_message(self, title: str, text: str) -> None:
        """没有表可给的时候（比如报错），也要在同一个地方说话，
        不要一会儿弹窗一会儿面板。"""
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self._show_overview([])
        self._primary_columns = None
        self.all_columns.hide()
        self.row_filter.setCurrentIndex(0)
        self.row_filter.setEnabled(False)
        self._filter_rows()
        self._set_caption(f"{title}\n{text}")
        self.details_button.setChecked(bool(text))
