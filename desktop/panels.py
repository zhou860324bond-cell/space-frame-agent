"""两个停靠面板：建模过程时间线、结果表。

放在一起是因为它们回答的是同一类问题——**"这个状态/这个数字是从哪来的"**。
时间线回答"模型怎么变成现在这样"，结果表回答"这个数字出现在哪根杆、哪个工况"。

两个面板都**只显示和转发，不算不改**。点了哪一行由主窗口决定怎么办。
"""

from __future__ import annotations

from html import escape
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QHeaderView, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from . import theme

STEP_INDEX = Qt.ItemDataRole.UserRole + 1
LOCATE = Qt.ItemDataRole.UserRole + 2


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
            mark = "✗" if not step.ok else " "
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

        self.caption = QLabel("尚无分析结果")
        self.caption.setProperty("panel", "hint")
        self.caption.setWordWrap(True)
        box.addWidget(self.caption)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("云图量程"))
        self.range_mode = QComboBox(self)
        self.range_mode.addItem("95% 裁剪", 95.0)
        self.range_mode.addItem("满量程", None)
        controls.addWidget(self.range_mode)
        controls.addWidget(QLabel("分级"))
        self.levels = QComboBox(self)
        for n in (6, 8, 10, 12, 16, 20, 24):
            self.levels.addItem(f"{n} 级", n)
        self.levels.setCurrentText("12 级")
        self.levels.setToolTip(
            "云图分成几级色块。连续渐变只看得出「这边比那边红」，"
            "分级之后每一段颜色对应色标上一个可读的区间。")
        controls.addWidget(self.levels)
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
        box.addLayout(controls)
        for widget in (self.range_mode, self.sign_mode, self.levels):
            widget.currentIndexChanged.connect(self._emit_display_options)
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
        box.addWidget(self.table, 1)

    def display_options(self) -> dict:
        return {"percentile": self.range_mode.currentData(),
                "sign": self.sign_mode.currentData(),
                "levels": self.levels.currentData(),
                "overlay_deformed": self.overlay_deformed.isChecked(),
                "show_extrema": self.show_extrema.isChecked()}

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

    # 严重度 → 底色。红只留给**真的不合格**；"判不了"用琥珀色，
    # 因为把它染成红的会让人以为结构有问题——那正是这一档要避免的误读。
    _MARK_TINT = {"fail": "#f7dede", "unclear": "#fbf0da"}

    def show_rows(self, title: str, columns: list[str],
                  rows: list[list[Any]],
                  locators: list[tuple[str, int] | None] | None = None,
                  marks: list[str | None] | None = None) -> None:
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
                tint = (self._MARK_TINT.get(marks[r])
                        if marks and r < len(marks) else None)
                if tint:
                    cell.setBackground(QColor(tint))
                self.table.setItem(r, c, cell)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._set_caption(title)

    def _set_caption(self, title: str) -> None:
        """标题第一行是结论，其余是限制与警告。

        全用同一种字号铺成一堵墙时，最要紧的那一行反而看不见了。所以第一行
        加粗，后面的话降一号、用弱色——**但一句都不删**：那些话正是这个结果
        最容易被误读的地方。
        """
        head, *rest = [line for line in str(title).split("\n") if line.strip()]
        body = (f'<div style="font-weight:600">{escape(head)}</div>'
                + "".join(
                    f'<div style="color:{theme.INK_MUTED};font-size:11px">'
                    f'{escape(line)}</div>' for line in rest))
        self.caption.setText(body)

    def show_message(self, title: str, text: str) -> None:
        """没有表可给的时候（比如报错），也要在同一个地方说话，
        不要一会儿弹窗一会儿面板。"""
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self._set_caption(f"{title}\n{text}")
