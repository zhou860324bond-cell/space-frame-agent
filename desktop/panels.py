"""两个停靠面板：建模过程时间线、结果表。

放在一起是因为它们回答的是同一类问题——**"这个状态/这个数字是从哪来的"**。
时间线回答"模型怎么变成现在这样"，结果表回答"这个数字出现在哪根杆、哪个工况"。

两个面板都**只显示和转发，不算不改**。点了哪一行由主窗口决定怎么办。
"""

from __future__ import annotations

from html import escape
import math
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QFrame, QGridLayout, QHeaderView, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QToolButton,
                               QVBoxLayout, QWidget)

from . import theme

STEP_INDEX = Qt.ItemDataRole.UserRole + 1
LOCATE = Qt.ItemDataRole.UserRole + 2
NUMERIC_VALUE = Qt.ItemDataRole.UserRole + 3


def _format_result_value(value: int | float) -> str:
    """把计算值显示成紧凑、可比较的工程数字。"""
    number = float(value)
    if not math.isfinite(number):
        return "—"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    magnitude = abs(number)
    if magnitude and (magnitude >= 1_000_000 or magnitude < 0.0001):
        return f"{number:.3e}"
    if magnitude >= 1_000:
        return f"{number:,.2f}".rstrip("0").rstrip(".")
    digits = 3 if magnitude >= 1 else 4
    return f"{number:.{digits}f}".rstrip("0").rstrip(".") or "0"


class _ResultTableItem(QTableWidgetItem):
    """显示格式与排序值分离，千分位不会破坏数值排序。"""

    def __init__(self, value: Any):
        super().__init__()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.setText(_format_result_value(value))
            self.setData(NUMERIC_VALUE, float(value))
            self.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                  | Qt.AlignmentFlag.AlignVCenter)
        else:
            self.setText("—" if value is None or value == "" else str(value))
            self.setTextAlignment(Qt.AlignmentFlag.AlignLeft
                                  | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(NUMERIC_VALUE)
        right = other.data(NUMERIC_VALUE)
        if left is not None and right is not None:
            return float(left) < float(right)
        return super().__lt__(other)


_HEADER_HELP = {
    "杆件": "物理杆件编号。选择一行可在三维视口中定位。",
    "节点": "模型节点编号。选择一行可在三维视口中定位。",
    "截面": "该杆件采用的截面名称或规格。",
    "应力比 σ/[σ]": "计算应力与许用应力之比；不大于 1 表示强度满足。",
    "控制状态": "控制该杆件校核结果的受力状态。",
    "轴力 N (kN)": "杆件轴向内力；正值为受拉，负值为受压。",
    "Euler 临界力 Pcr (kN)": "理想压杆发生 Euler 屈曲时的临界轴力。",
    "稳定利用率 N/Pcr": "轴向压力与 Euler 临界力之比；仅在适用条件满足时使用。",
    "长细比 λ": "构件计算长度与截面回转半径之比。",
    "计算长度系数 μ": "把实际约束条件换算为等效计算长度的系数。",
    "计算长度依据": "计算长度系数的来源；自动推断结果应结合节点约束复核。",
    "校核结论": "通过、超限或暂无法判断；无法判断不等于不合格。",
    "结果量": "结果分量的工程名称和常用符号。",
    "值": "当前工况、当前位置对应的计算值。",
    "单位": "当前数值采用的工程单位。",
    "方向 / 约定": "分量方向、正负号和取值位置的说明。",
    "档位": "网格由粗到细的计算顺序；最后一档是当前最细网格。",
    "全局网格 (mm)": "远离节点热点区的目标单元尺寸。",
    "热点网格 (mm)": "节点相贯线附近局部加密区的目标单元尺寸；收敛判断采用此尺寸。",
    "C3D10 单元数": "该档局部实体子模型的十节点二次四面体单元数量。",
    "最大 Mises (MPa)": "节点热点区域的最大等效应力；奇异几何下只用于观察分布。",
    "控制热点应力 (MPa)": "控制杆臂按 0.4t/1.0t 表面路径线性外推得到的热点应力。",
    "相邻变化率 (%)": "本档热点应力相对上一档的绝对变化率。",
    "变化 / 差异 (%)": "同一序列表示相邻档变化率；双后端对标表示两结果的相对差异。",
    "证据状态": "细化记录、证据不足或满足全部杆臂门禁后的正式发布状态。",
}


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
    trust_requested = Signal()
    convergence_requested = Signal()
    analysis_mesh_requested = Signal()
    history_compare_requested = Signal(str, str)
    history_manage_requested = Signal()
    history_remove_requested = Signal(str)
    history_protection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(4)

        # 首屏先给工程结论，不再用“结果类型 / 数据量”占掉最宝贵的位置。
        # 类型已经写在标题里；工程师真正先找的是极值、控制位置、工况和单位。
        self.summary = QFrame(self)
        self.summary.setObjectName("resultSummary")
        summary_box = QVBoxLayout(self.summary)
        summary_box.setContentsMargins(0, 0, 0, 0)
        summary_box.setSpacing(4)
        headline = QHBoxLayout()
        headline.setSpacing(6)
        self.summary_maximum = self._summary_metric(headline, "最大值")
        self.summary_minimum = self._summary_metric(headline, "最小值")
        self.summary_location = self._summary_metric(
            headline, "控制位置", stretch=2)
        summary_box.addLayout(headline)
        facts = QHBoxLayout()
        facts.setSpacing(6)
        self.summary_case = self._summary_metric(facts, "工况 / 组合")
        self.summary_unit = self._summary_metric(facts, "单位")
        self.summary_status = self._summary_metric(facts, "求解状态")
        self.summary_trust = self._summary_metric(
            facts, "结果可信度", stretch=2)
        summary_box.addLayout(facts)
        box.addWidget(self.summary)
        self._result_row_count = 0
        self._result_marks: list[str | None] = []
        self._engineering_summary: dict | None = None
        self._trust: dict | None = None

        self.caption = QLabel("尚无分析结果")
        self.caption.setObjectName("resultCaption")
        self.caption.setProperty("panel", "hint")
        self.caption.setWordWrap(True)
        self.caption.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.caption)
        self.caption_toggle = QToolButton(self)
        self.caption_toggle.setObjectName("resultCaptionToggle")
        self.caption_toggle.setAutoRaise(True)
        self.caption_toggle.clicked.connect(self._toggle_caption)
        self.caption_toggle.hide()
        box.addWidget(self.caption_toggle, 0, Qt.AlignmentFlag.AlignLeft)

        # 节点实体结果比普通梁结果多一层“这份 ODB/VTU 到底从哪来”的问题。
        # 直接放在结果页上，避免用户为了核对来源再打开一个遮挡视口的窗口。
        self.provenance = QFrame(self)
        self.provenance.setObjectName("resultProvenance")
        provenance_box = QVBoxLayout(self.provenance)
        provenance_box.setContentsMargins(0, 0, 0, 0)
        provenance_box.setSpacing(4)
        provenance_header = QHBoxLayout()
        provenance_header.setContentsMargins(0, 0, 0, 0)
        provenance_title = QLabel("实体结果溯源", self.provenance)
        provenance_title.setProperty("role", "section")
        provenance_header.addWidget(provenance_title)
        provenance_header.addStretch(1)
        self.provenance_status = QLabel("—", self.provenance)
        self.provenance_status.setProperty("role", "resultMetricValue")
        provenance_header.addWidget(self.provenance_status)
        provenance_box.addLayout(provenance_header)
        provenance_row = QHBoxLayout()
        provenance_row.setContentsMargins(0, 0, 0, 0)
        provenance_row.setSpacing(6)
        self.provenance_source = self._provenance_metric(
            provenance_row, "结果来源", stretch=1)
        self.provenance_time = self._provenance_metric(
            provenance_row, "生成时间")
        self.provenance_fingerprint = self._provenance_metric(
            provenance_row, "模型 / 节点指纹", stretch=2)
        self.provenance_mesh = self._provenance_metric(
            provenance_row, "提交网格")
        self.provenance_artifact = self._provenance_metric(
            provenance_row, "主结果文件")
        provenance_box.addLayout(provenance_row)
        self.history_controls = QFrame(self.provenance)
        history_box = QVBoxLayout(self.history_controls)
        history_box.setContentsMargins(8, 3, 8, 3)
        history_box.setSpacing(4)
        self.history_compare_bar = QWidget(self.history_controls)
        history_row = QHBoxLayout(self.history_compare_bar)
        history_row.setContentsMargins(0, 0, 0, 0)
        history_row.setSpacing(6)
        self.history_caption = QLabel("历史对比", self.history_controls)
        history_row.addWidget(self.history_caption)
        self.history_left = QComboBox(self.history_controls)
        self.history_right = QComboBox(self.history_controls)
        self.history_left.setMinimumWidth(230)
        self.history_right.setMinimumWidth(230)
        self.history_left.setToolTip("选择作为版本 A 的实体运行记录。")
        self.history_right.setToolTip("选择作为版本 B 的实体运行记录。")
        history_row.addWidget(self.history_left, 1)
        self.history_between = QLabel("对比", self.history_controls)
        history_row.addWidget(self.history_between)
        history_row.addWidget(self.history_right, 1)
        self.btn_history_compare = QPushButton("比较两次结果", self.history_controls)
        self.btn_history_compare.setToolTip(
            "比较输入指纹、网格计划、控制热点应力、Kt 和主结果文件；"
            "输入不一致时不会把差异归因于求解器。")
        self.btn_history_compare.clicked.connect(self._emit_history_compare)
        self.history_left.currentIndexChanged.connect(
            self._update_history_compare_enabled)
        self.history_right.currentIndexChanged.connect(
            self._update_history_compare_enabled)
        history_row.addWidget(self.btn_history_compare)
        history_row.addStretch(1)
        self.btn_history_manage = QPushButton("管理历史", self.history_controls)
        self.btn_history_manage.setToolTip(
            "查看每次实体运行的磁盘占用，并安全清理不再需要的旧版本。")
        self.btn_history_manage.clicked.connect(
            lambda: self.history_manage_requested.emit())
        history_row.addWidget(self.btn_history_manage)
        history_box.addWidget(self.history_compare_bar)

        self.history_cleanup_bar = QWidget(self.history_controls)
        cleanup_row = QHBoxLayout(self.history_cleanup_bar)
        cleanup_row.setContentsMargins(0, 0, 0, 0)
        cleanup_row.setSpacing(6)
        self.history_cleanup_caption = QLabel(
            "可清理运行", self.history_cleanup_bar)
        cleanup_row.addWidget(self.history_cleanup_caption)
        self.history_cleanup = QComboBox(self.history_controls)
        self.history_cleanup.setMinimumWidth(360)
        self.history_cleanup.setToolTip(
            "仅列出非 latest、且未被版本 A/B 占用的独立运行目录。")
        self.history_cleanup.currentIndexChanged.connect(
            self._update_history_remove_enabled)
        cleanup_row.addWidget(self.history_cleanup, 1)
        self.btn_history_remove = QPushButton("移入回收站", self.history_controls)
        self.btn_history_remove.setToolTip(
            "二次确认后，将所选旧运行的 ODB/VTU、网格及摘要一起移入 Windows 回收站。")
        self.btn_history_remove.clicked.connect(self._emit_history_remove)
        cleanup_row.addWidget(self.btn_history_remove)
        cleanup_note = QLabel(
            "保护：每个后端 latest；版本 A/B 同时受保护", self.history_cleanup_bar)
        cleanup_note.setProperty("panel", "hint")
        cleanup_row.addWidget(cleanup_note)
        history_box.addWidget(self.history_cleanup_bar)
        provenance_box.addWidget(self.history_controls)
        box.addWidget(self.provenance)
        self._provenance: dict | None = None
        self._solid_history_count = 0
        self._solid_cleanup_count = 0
        self._solid_history_entries: list[tuple[str, str]] = []
        self._solid_cleanup_entries: list[tuple[str, str]] = []
        self._history_cleanup_eligible: set[str] = set()
        self.provenance.hide()
        self.history_controls.hide()

        # 云图控制只在云图页面出现。把它放在所有数据表上方，会让强度、
        # 模态等结果看起来像一套尚未收尾的调试器界面。
        self.cloud_controls = QFrame(self)
        self.cloud_controls.setObjectName("resultControls")
        controls = QGridLayout(self.cloud_controls)
        controls.setContentsMargins(10, 7, 10, 7)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)

        section_label = QLabel("云图显示", self.cloud_controls)
        section_label.setProperty("role", "section")
        controls.addWidget(section_label, 0, 0)
        controls.addWidget(QLabel("量程", self.cloud_controls), 0, 1)
        self.range_mode = QComboBox(self)
        self.range_mode.addItem("增强对比（P95）", 95.0)
        self.range_mode.addItem("满量程", None)
        self.range_mode.setCurrentIndex(1)
        self.range_mode.setToolTip(
            "满量程显示真实最小值到最大值。\n"
            "增强对比适合少量峰值压缩了其余颜色时使用；它只改变显示，"
            "不会改变计算结果。")
        controls.addWidget(self.range_mode, 0, 2)
        controls.addWidget(QLabel("色带", self.cloud_controls), 0, 3)
        self.levels = QComboBox(self)
        for n in (6, 8, 10, 12, 16, 20, 24):
            self.levels.addItem(f"{n} 级", n)
        self.levels.setCurrentText("12 级")
        self.levels.setToolTip(
            "控制离散色带数量。12 级适合日常屏幕审阅；"
            "较少色带更容易区分范围，较多色带更接近连续分布。")
        controls.addWidget(self.levels, 0, 4)
        controls.addWidget(QLabel("配色", self.cloud_controls), 0, 5)
        self.palette = QComboBox(self)
        from . import theme as _theme
        for key, label in _theme.CONTOUR_PALETTES.items():
            self.palette.addItem(label, key)
        self.palette.setCurrentIndex(
            max(0, self.palette.findData(_theme.DEFAULT_PALETTE)))
        self.palette.setToolTip(
            "彩虹色便于区分相邻区间；蓝—灰—红适合查看正负号；"
            "单蓝适合黑白打印和连续高低比较。")
        controls.addWidget(self.palette, 0, 6)
        controls.setColumnStretch(7, 1)

        self.shading = QCheckBox("表面明暗", self)
        self.shading.setChecked(True)
        self.shading.setToolTip(
            "用光照增强杆件的空间形体。关闭后颜色与图例严格一致，"
            "更适合按色标读数。")
        controls.addWidget(self.shading, 1, 1, 1, 2)
        controls.addWidget(QLabel("符号", self.cloud_controls), 1, 3)
        self.sign_mode = QComboBox(self)
        self.sign_mode.addItem("全部", "all")
        self.sign_mode.addItem("仅正值", "positive")
        self.sign_mode.addItem("仅负值", "negative")
        self.sign_mode.setToolTip(
            "筛选有正负号的局部内力分量。合力、应力幅值等无符号结果"
            "通常保持为“全部”。")
        controls.addWidget(self.sign_mode, 1, 4)
        self.overlay_deformed = QCheckBox("叠加变形", self)
        self.overlay_deformed.setToolTip(
            "在结果云图上叠加放大后的变形轮廓，便于同时判断分布与变形模式。")
        controls.addWidget(self.overlay_deformed, 1, 5)
        self.show_extrema = QCheckBox("标注极值", self)
        self.show_extrema.setChecked(True)
        self.show_extrema.setToolTip(
            "在三维视口中标记当前结果分量的绝对最大值位置。")
        controls.addWidget(self.show_extrema, 1, 6)
        self.display_scope = QLabel(
            "显示细分：每根杆件 41 个显示点。只影响曲线和色带，不改变分析网格或求解结果。",
            self.cloud_controls)
        self.display_scope.setObjectName("displayScopeNotice")
        self.display_scope.setWordWrap(True)
        controls.addWidget(self.display_scope, 2, 1, 1, 5)
        self.btn_analysis_mesh = QPushButton("查看分析网格", self.cloud_controls)
        self.btn_analysis_mesh.setToolTip(
            "显示真正参与刚度组装的分析节点和分析单元；这与云图显示细分不同。")
        self.btn_analysis_mesh.clicked.connect(
            lambda: self.analysis_mesh_requested.emit())
        controls.addWidget(self.btn_analysis_mesh, 2, 6)
        box.addWidget(self.cloud_controls)

        self.result_actions = QWidget(self)
        actions = QHBoxLayout(self.result_actions)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        actions.addStretch(1)
        self.btn_extreme = QPushButton("定位当前分量极值", self)
        self.btn_extreme.setToolTip(
            "查找当前分量在整个结构中的绝对最大值，并定位到对应杆件和截面位置。")
        self.btn_extreme.clicked.connect(lambda: self.extreme_requested.emit())
        actions.addWidget(self.btn_extreme)
        self.btn_stress = QPushButton("查看截面应力", self)
        self.btn_stress.setToolTip(
            "根据轴力和双向弯矩绘制当前探针截面的正应力分布。")
        self.btn_stress.clicked.connect(lambda: self.stress_requested.emit())
        actions.addWidget(self.btn_stress)
        self.btn_trust = QPushButton("可信度详情", self)
        self.btn_trust.setToolTip(
            "查看模型合法性、静力平衡、分析离散和网格收敛适用性。")
        self.btn_trust.clicked.connect(lambda: self.trust_requested.emit())
        actions.addWidget(self.btn_trust)
        self.btn_convergence = QPushButton("收敛诊断", self)
        self.btn_convergence.setToolTip(
            "统一查看 B31 网格细化、实体热点多档网格和非线性增量收敛；"
            "没有专项记录时会明确说明，不会把云图显示点当成分析网格。")
        self.btn_convergence.clicked.connect(
            lambda: self.convergence_requested.emit())
        actions.addWidget(self.btn_convergence)
        box.addWidget(self.result_actions)
        for widget in (self.range_mode, self.sign_mode, self.levels, self.palette):
            widget.currentIndexChanged.connect(self._emit_display_options)
        self.shading.stateChanged.connect(self._emit_display_options)
        self.overlay_deformed.stateChanged.connect(self._emit_display_options)
        self.show_extrema.stateChanged.connect(self._emit_display_options)

        self.table = QTableWidget(self)
        self.table.setObjectName("resultTable")
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setToolTip(
            "选择一行可在三维视口定位对应节点或杆件；点击表头可排序。")
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.itemSelectionChanged.connect(self._on_select)
        box.addWidget(self.table, 1)
        self.set_context("message")

    def display_options(self) -> dict:
        return {"percentile": self.range_mode.currentData(),
                "sign": self.sign_mode.currentData(),
                "levels": self.levels.currentData(),
                "palette": self.palette.currentData(),
                "shading": self.shading.isChecked(),
                "overlay_deformed": self.overlay_deformed.isChecked(),
                "show_extrema": self.show_extrema.isChecked()}

    def _summary_metric(self, row: QHBoxLayout, caption: str,
                        stretch: int = 0) -> QLabel:
        card = QFrame(self.summary)
        card.setProperty("role", "resultMetric")
        content = QVBoxLayout(card)
        content.setContentsMargins(9, 5, 9, 6)
        content.setSpacing(1)
        label = QLabel(caption, card)
        label.setProperty("role", "resultMetricLabel")
        value = QLabel("—", card)
        value.setProperty("role", "resultMetricValue")
        content.addWidget(label)
        content.addWidget(value)
        row.addWidget(card, stretch)
        return value

    def _provenance_metric(self, row: QHBoxLayout, caption: str,
                           stretch: int = 0) -> QLabel:
        card = QFrame(self.provenance)
        card.setProperty("role", "resultMetric")
        content = QVBoxLayout(card)
        content.setContentsMargins(9, 5, 9, 6)
        content.setSpacing(1)
        label = QLabel(caption, card)
        label.setProperty("role", "resultMetricLabel")
        value = QLabel("—", card)
        value.setProperty("role", "resultMetricValue")
        value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        content.addWidget(label)
        content.addWidget(value)
        row.addWidget(card, stretch)
        return value

    def set_solid_provenance(self, provenance: dict | None) -> None:
        self._provenance = provenance
        if provenance:
            values = {
                "source": self.provenance_source,
                "generated": self.provenance_time,
                "fingerprint": self.provenance_fingerprint,
                "mesh": self.provenance_mesh,
                "artifact": self.provenance_artifact,
            }
            tips = provenance.get("tooltips") or {}
            for key, widget in values.items():
                widget.setText(str(provenance.get(key) or "—"))
                widget.setToolTip(str(tips.get(key) or widget.text()))
            self.provenance_artifact.setProperty(
                "severity", provenance.get("severity") or "unclear")
            self.provenance_artifact.style().unpolish(self.provenance_artifact)
            self.provenance_artifact.style().polish(self.provenance_artifact)
            self.provenance_status.setText(
                str(provenance.get("status") or "需要复核"))
            self.provenance_status.setProperty(
                "severity", provenance.get("severity") or "unclear")
            self.provenance_status.style().unpolish(self.provenance_status)
            self.provenance_status.style().polish(self.provenance_status)
        self._update_action_visibility()

    def set_solid_history(self, entries: list[tuple[str, str]],
                          cleanup_eligible: set[str] | None = None) -> None:
        """填充同页历史选择器；少于两次运行时不占界面空间。"""
        previous_left = self.history_left.currentData()
        previous_right = self.history_right.currentData()
        self.history_left.blockSignals(True)
        self.history_right.blockSignals(True)
        self.history_left.clear()
        self.history_right.clear()
        for identifier, label in entries:
            self.history_left.addItem(label, identifier)
            self.history_right.addItem(label, identifier)
        self._solid_history_count = len(entries)
        self._solid_history_entries = list(entries)
        self._solid_cleanup_entries = list(entries)
        self._solid_cleanup_count = len(entries)
        self._history_cleanup_eligible = (
            set(cleanup_eligible) if cleanup_eligible is not None
            else {identifier for identifier, _label in entries})
        if len(entries) >= 2:
            left_index = self.history_left.findData(previous_left)
            right_index = self.history_right.findData(previous_right)
            self.history_left.setCurrentIndex(left_index if left_index >= 0 else 0)
            self.history_right.setCurrentIndex(right_index if right_index >= 0 else 1)
        self.history_left.blockSignals(False)
        self.history_right.blockSignals(False)
        self._update_history_compare_enabled()
        self._refresh_history_cleanup()
        self._update_action_visibility()

    def set_solid_cleanup_entries(self, entries: list[tuple[str, str]],
                                  cleanup_eligible: set[str]) -> None:
        """管理页可加入不完整运行，但它们不能混进版本 A/B 对比框。"""
        self._solid_cleanup_entries = list(entries)
        self._solid_cleanup_count = len(entries)
        self._history_cleanup_eligible = set(cleanup_eligible)
        self._refresh_history_cleanup()
        self._update_action_visibility()

    def _update_history_compare_enabled(self, *_args) -> None:
        left = self.history_left.currentData()
        right = self.history_right.currentData()
        self.btn_history_compare.setEnabled(
            self._solid_history_count >= 2 and left is not None and left != right)
        self._refresh_history_cleanup()
        if _args:
            self.history_protection_changed.emit()

    def _refresh_history_cleanup(self) -> None:
        """从清理候选中即时排除当前 A/B；latest 已由主窗口预先排除。"""
        current = self.history_cleanup.currentData()
        protected = {self.history_left.currentData(), self.history_right.currentData()}
        self.history_cleanup.blockSignals(True)
        self.history_cleanup.clear()
        for identifier, label in self._solid_cleanup_entries:
            if (identifier in self._history_cleanup_eligible
                    and identifier not in protected):
                self.history_cleanup.addItem(label, identifier)
        index = self.history_cleanup.findData(current)
        if index >= 0:
            self.history_cleanup.setCurrentIndex(index)
        self.history_cleanup.blockSignals(False)
        self._update_history_remove_enabled()

    def _update_history_remove_enabled(self, *_args) -> None:
        self.btn_history_remove.setEnabled(
            self.history_cleanup.currentData() is not None)

    def history_protected_ids(self) -> set[str]:
        return {str(value) for value in (
            self.history_left.currentData(), self.history_right.currentData())
                if value is not None}

    def _emit_history_compare(self) -> None:
        left = self.history_left.currentData()
        right = self.history_right.currentData()
        if left is not None and right is not None and left != right:
            self.history_compare_requested.emit(str(left), str(right))

    def _emit_history_remove(self) -> None:
        identifier = self.history_cleanup.currentData()
        if identifier is not None:
            self.history_remove_requested.emit(str(identifier))

    def _refresh_summary(self) -> None:
        engineering = self._engineering_summary or {}

        def text(key: str, fallback: str) -> str:
            value = engineering.get(key)
            return fallback if value is None or value == "" else str(value)

        self.summary_maximum.setText(text("maximum", "不适用"))
        self.summary_minimum.setText(text("minimum", "不适用"))
        self.summary_location.setText(text(
            "location", "未提供可定位的控制点"))
        self.summary_case.setText(text("case", "—"))
        self.summary_unit.setText(text("unit", "—"))
        failed = self._result_marks.count("fail")
        unclear = self._result_marks.count("unclear")
        passed = self._result_marks.count("pass")
        if engineering.get("solve_status"):
            text = str(engineering["solve_status"])
            severity = str(engineering.get("severity") or "ready")
        elif failed:
            text, severity = f"已完成 · {failed} 项超限", "fail"
        elif unclear:
            text, severity = f"已完成 · {unclear} 项待复核", "unclear"
        elif passed:
            text, severity = "已完成 · 检查通过", "pass"
        elif self._result_row_count:
            text, severity = "已完成", "ready"
        else:
            text, severity = "未生成", "idle"
        self.summary_status.setText(text)
        self.summary_status.setProperty("severity", severity)
        self.summary_status.style().unpolish(self.summary_status)
        self.summary_status.style().polish(self.summary_status)
        if self.context == "solid_storage":
            trust_text, trust_severity = "不适用", "idle"
            trust_tip = "这是磁盘运行目录管理视图，不评价位移、内力或应力可信度。"
        elif self._trust is None:
            trust_text, trust_severity = "尚未检查", "idle"
            trust_tip = "完成求解后显示模型合法性、静力平衡和分析离散检查。"
        else:
            trust_text = str(self._trust.get("label") or "需要复核")
            trust_severity = str(self._trust.get("severity") or "unclear")
            trust_tip = str(self._trust.get("summary") or trust_text)
        self.summary_trust.setText(trust_text)
        self.summary_trust.setProperty("severity", trust_severity)
        self.summary_trust.setToolTip(trust_tip)
        self.summary_trust.style().unpolish(self.summary_trust)
        self.summary_trust.style().polish(self.summary_trust)

    def set_engineering_summary(self, summary: dict | None) -> None:
        """设置顶部工程结论；值由结果生产者显式给出，面板不猜字段语义。"""
        self._engineering_summary = dict(summary) if summary else None
        self._refresh_summary()

    def set_trust_assessment(self, assessment: dict | None) -> None:
        self._trust = assessment
        if assessment:
            self.display_scope.setText(
                f"显示细分：每根杆件 {assessment.get('display_stations', 41)} 个显示点。"
                "只影响曲线和色带，不改变分析网格或求解结果。")
        self._refresh_summary()
        self._update_action_visibility()

    def _update_action_visibility(self) -> None:
        is_contour = self.context == "contour"
        is_probe = self.context == "probe"
        is_storage = self.context == "solid_storage"
        self.provenance.setVisible(
            self._provenance is not None
            and self.context in {"solid_joint", "convergence", "solid_history",
                                 "solid_history_manage", "solid_storage"})
        self.history_controls.setVisible(
            not self.provenance.isHidden()
            and (self._solid_history_count >= 1
                 or (is_storage and self._solid_cleanup_count >= 1)))
        managing = self.context in {"solid_history_manage", "solid_storage"}
        has_comparison = self._solid_history_count >= 2 and not is_storage
        self.history_caption.setVisible(has_comparison)
        self.history_left.setVisible(has_comparison)
        self.history_between.setVisible(has_comparison)
        self.history_right.setVisible(has_comparison)
        self.history_caption.setText("保护版本" if managing else "历史对比")
        self.history_between.setText("与" if managing else "对比")
        self.btn_history_compare.setVisible(has_comparison and not managing)
        self.history_cleanup_bar.setVisible(managing)
        self.btn_history_manage.setText("返回收敛诊断" if managing else "管理历史")
        self.btn_history_manage.setVisible(not is_storage)
        self.result_actions.setVisible(
            not is_storage and (is_contour or is_probe or self._trust is not None))
        self.btn_extreme.setVisible(is_contour)
        self.btn_stress.setVisible(is_contour or is_probe)
        self.btn_trust.setVisible(self._trust is not None)
        self.btn_convergence.setVisible(self._trust is not None)

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

    _MARK_STYLE = {
        "fail": ("#f7dede", theme.ERROR),
        "unclear": ("#fbf0da", theme.WARN),
        "pass": ("#e3f3e9", theme.SUCCESS),
    }

    def set_context(self, context: str) -> None:
        """按当前结果类型只展示有意义的工具，避免控件噪声。"""
        self.context = context
        is_contour = context == "contour"
        self.cloud_controls.setVisible(is_contour)
        self._update_action_visibility()
        self._refresh_summary()

    def show_rows(self, title: str, columns: list[str],
                  rows: list[list[Any]],
                  locators: list[tuple[str, int] | None] | None = None,
                  marks: list[str | None] | None = None,
                  context: str = "table",
                  engineering: dict | None = None) -> None:
        """填表。

        `locators[i]` 说明第 i 行对应视口里的哪个对象，可为 None。
        `marks[i]` 是该行的严重度（fail / unclear / pass），用来着色——
        校核表动辄十几行，逐格去读"结论"那一列不现实，超限的行要自己跳出来。
        """
        self._engineering_summary = dict(engineering) if engineering else None
        self.set_context(context)
        self._result_row_count = len(rows)
        self._result_marks = list(marks or [])
        self.table.setSortingEnabled(False)     # 填的过程中排序会打乱行序
        self.table.clear()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        for c, label in enumerate(columns):
            item = self.table.horizontalHeaderItem(c)
            if item:
                item.setToolTip(_HEADER_HELP.get(label, label))
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                cell = _ResultTableItem(value)
                if c == 0 and locators and r < len(locators) and locators[r]:
                    cell.setData(LOCATE, locators[r])
                    kind, identifier = locators[r]
                    noun = "杆件" if kind == "member" else "节点"
                    cell.setToolTip(f"选择此行，在三维视口定位{noun} {identifier}。")
                elif len(cell.text()) > 36:
                    cell.setToolTip(cell.text())
                self.table.setItem(r, c, cell)
            # 只强调结论单元格，不再把整行涂成警告色。这样密集表格更易扫读，
            # 也不会把“暂无法判断”误看成整行报错。
            mark = marks[r] if marks and r < len(marks) else None
            style = self._MARK_STYLE.get(mark)
            status_cell = self.table.item(r, len(row) - 1) if row else None
            if style and status_cell:
                background, foreground = style
                status_cell.setBackground(QColor(background))
                status_cell.setForeground(QColor(foreground))
                font = status_cell.font()
                font.setBold(True)
                status_cell.setFont(font)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if columns:
            header.setSectionResizeMode(
                len(columns) - 1, QHeaderView.ResizeMode.Stretch)
        self._set_caption(title)
        self._refresh_summary()

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
        self._caption_expanded = False
        self.caption.setMaximumHeight(72 if rest else 16777215)
        self.caption_toggle.setVisible(bool(rest))
        self.caption_toggle.setText(f"展开计算说明（{len(rest)} 条）")

    def _toggle_caption(self) -> None:
        self._caption_expanded = not self._caption_expanded
        self.caption.setMaximumHeight(16777215 if self._caption_expanded else 72)
        self.caption_toggle.setText(
            "收起计算说明" if self._caption_expanded else
            f"展开计算说明（{max(1, self.caption.text().count('<div') - 1)} 条）")

    def show_message(self, title: str, text: str) -> None:
        """没有表可给的时候（比如报错），也要在同一个地方说话，
        不要一会儿弹窗一会儿面板。"""
        self.set_context("message")
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self._result_row_count = 0
        self._result_marks = []
        self._engineering_summary = None
        self._set_caption(f"{title}\n{text}")
        self._refresh_summary()
