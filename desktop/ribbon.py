"""功能区（Ribbon）。

顶部按任务阶段分页：项目 / 建模 / 荷载 / 分析 / 结果 / 视图。这是 CAE
软件的通行组织方式，好处不在好看，在**它把"现在该干什么"写在了界面上**——
一个没用过的人从左往右点一遍，正好就是一次完整的分析流程。

三条设计约束：

1. **按钮不持有状态。** 每个按钮绑的是主窗口的 `QAction`，
   置灰、勾选、快捷键都只在 QAction 上定义一次。功能区里再存一份，
   迟早会出现"菜单是灰的、功能区是亮的"。
2. **图标要画这个软件真正在做的事**，不用通用的齿轮放大镜凑数。
   见 `icons.py`。
3. **只做空间刚架。** 不摆"网格""接触""非线性"这些我们没有的页签。
   摆了就是在骗人，而且答辩时一点就穿。

大按钮（图标在上、文字在下）留给每一页的主操作，小按钮（图标在左）
放次要操作——一眼能看出这一页最该点哪个。
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QSizePolicy, QTabWidget,
                               QToolButton, QVBoxLayout, QWidget)

from . import icons, theme

LARGE = QSize(30, 30)
SMALL = QSize(20, 20)


def _button(action, large: bool) -> QToolButton:
    b = QToolButton()
    b.setDefaultAction(action)             # 状态全在 QAction 上，这里不留副本
    b.setIconSize(LARGE if large else SMALL)
    b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon if large
                         else Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    b.setAutoRaise(True)
    b.setProperty("ribbon", "large" if large else "small")
    if action.objectName() in {"solve", "sketch_ai"}:
        b.setProperty("role", "primary")
    if large:
        b.setMinimumWidth(58)
        b.setMaximumWidth(92)
    return b


class RibbonGroup(QWidget):
    """功能区里的一组。底下带一行组名——CAE 里这行小字很有用，
    它告诉你这几个按钮为什么被放在一起。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setProperty("ribbonGroup", True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 2)
        outer.setSpacing(2)

        self.body = QGridLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setHorizontalSpacing(2)
        self.body.setVerticalSpacing(1)
        outer.addLayout(self.body, 1)

        caption = QLabel(title)
        caption.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        caption.setProperty("ribbon", "caption")
        outer.addWidget(caption)

        self._col = 0
        self._small_row = 0

    def add_large(self, action) -> QToolButton:
        b = _button(action, True)
        if self._small_row:                # 前一列还是小按钮，换列
            self._col += 1
            self._small_row = 0
        self.body.addWidget(b, 0, self._col, 3, 1)
        self._col += 1
        return b

    def add_small(self, action) -> QToolButton:
        """小按钮竖着堆，一列三个，堆满换列。"""
        b = _button(action, False)
        self.body.addWidget(b, self._small_row, self._col,
                            Qt.AlignmentFlag.AlignLeft)
        self._small_row += 1
        if self._small_row >= 3:
            self._small_row = 0
            self._col += 1
        return b


class RibbonPage(QWidget):
    """一个页签的内容：若干组，横着排，右侧留白。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(4, 2, 4, 0)
        self._row.setSpacing(0)
        self._row.addStretch(1)

    def group(self, title: str) -> RibbonGroup:
        g = RibbonGroup(title, self)
        # 插在弹簧前面，组就一直靠左，右边留白
        self._row.insertWidget(self._row.count() - 1, g)
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setProperty("ribbon", "sep")
        self._row.insertWidget(self._row.count() - 1, sep)
        return g


class Ribbon(QTabWidget):
    """整个功能区。

    页签顺序就是分析流程的顺序：**建模 → 加荷载 → 算 → 看结果**。
    这不是随手排的——不熟悉的人从左往右点一遍就能走完一次完整分析。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setProperty("ribbon", "bar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setMaximumHeight(112)
        self.pages: dict[str, RibbonPage] = {}

    def page(self, title: str) -> RibbonPage:
        p = RibbonPage(self)
        self.addTab(p, title)
        self.pages[title] = p
        return p


class QuickBar(QWidget):
    """功能区下面那条**常驻**图标带。

    这是从参照界面里学到的最有价值的一个结构。视角、选择模式、显示模式
    这些东西**每个阶段都要用**——把它们放进某个页签，就意味着你看云图时
    想转个角度得先切到"视图"页再切回来，这个来回是纯浪费。

    除了图标还放两个输入：**工况下拉**（审结果时切得最勤）和
    **变形放大系数**（判断"这个变形是真的大还是被放大了"要靠手动调它，
    CAE 里这是个常驻输入框）。
    """

    case_changed = Signal(str)
    scale_changed = Signal(float)

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.setObjectName("workspaceBar")
        self._window = window
        a = window.actions_by_name
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 3, 8, 3)
        row.setSpacing(2)

        def separator(row) -> None:
            sep = QFrame(self)
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setProperty("toolbar", "sep")
            row.addWidget(sep)

        def section(row, text: str) -> None:
            label = QLabel(text, self)
            label.setProperty("toolbar", "section")
            row.addWidget(label)

        def icon_group(row, *names: str) -> None:
            for n in names:
                b = _button(a[n], False)
                b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
                b.setIconSize(QSize(20, 20))
                row.addWidget(b)
            separator(row)

        section(row, "视角")
        icon_group(row, "front", "side", "top", "iso", "fit")

        section(row, "选择")
        for n in ("pick_node", "pick_member"):
            b = _button(a[n], False)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            b.setIconSize(QSize(16, 16))
            row.addWidget(b)
        separator(row)

        # 人工建模按钮
        section(row, "编辑")
        for n in ("model_node", "model_member", "delete"):
            b = _button(a[n], False)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            b.setIconSize(QSize(16, 16))
            row.addWidget(b)
        separator(row)
        icon_group(row, "labels", "props", "undo", "redo")

        # 精确建模：工作平面 + 网格捕捉 + 精确坐标建点。
        # 三维里直接点坐标会飘，这三个控件把节点"锁"在工作平面和网格上。
        section(row, "工作平面")
        self.plane = QComboBox(self)
        self.plane.addItems(["XY", "XZ", "YZ"])
        self.plane.setFixedWidth(56)
        self.plane.setToolTip(
            "工作平面：新建节点先投影到该平面\n"
            "XY=水平面固定 z，XZ=正立面固定 y，YZ=侧立面固定 x")
        self.plane.currentTextChanged.connect(self._on_plane)
        row.addWidget(self.plane)

        self.offset_label = QLabel("z=")
        row.addWidget(self.offset_label)
        self.offset = QDoubleSpinBox(self)
        self.offset.setRange(-1.0e6, 1.0e6)
        self.offset.setDecimals(3)
        self.offset.setSingleStep(0.5)
        self.offset.setFixedWidth(78)
        self.offset.setToolTip(
            "工作平面位置：XY 填 z，XZ 填 y，YZ 填 x；单位跟随当前模型")
        self.offset.valueChanged.connect(window.set_work_offset)
        row.addWidget(self.offset)

        row.addWidget(QLabel("捕捉"))
        self.snap = QComboBox(self)
        self.snap.setEditable(True)
        self.snap.addItems(["关闭", "0.1", "0.25", "0.5", "1", "2"])
        self.snap.setFixedWidth(66)
        self.snap.setToolTip(
            "网格捕捉间距：工作平面内按此间距对齐坐标\n"
            "选“关闭”则不吸附网格；靠近已有节点始终会自动吸附")
        # 构建期逐项触发会把捕捉设成中间值，先屏蔽，定到“关闭”
        self.snap.blockSignals(True)
        self.snap.setCurrentIndex(0)
        self.snap.blockSignals(False)
        self.snap.currentTextChanged.connect(window.set_snap_size)
        row.addWidget(self.snap)

        self.coord_btn = QToolButton(self)
        self.coord_btn.setText("坐标建点")
        self.coord_btn.setAutoRaise(True)
        self.coord_btn.setToolTip("输入精确 x/y/z 坐标创建节点（鼠标点不准时用）")
        self.coord_btn.clicked.connect(window.create_node_exact)
        row.addWidget(self.coord_btn)
        separator(row)

        section(row, "显示")
        icon_group(row, "model", "deformed", "contour", "modal")

        row.addWidget(QLabel("工况"))
        self.cases = QComboBox(self)
        self.cases.setMinimumWidth(132)
        self.cases.setToolTip("切换显示哪个荷载工况或组合的结果")
        self.cases.currentTextChanged.connect(self._on_case)
        row.addWidget(self.cases)

        row.addWidget(QLabel("放大"))
        self.scale = QComboBox(self)
        self.scale.setEditable(True)
        self.scale.addItems(["自动", "1", "10", "50", "100", "200", "500"])
        self.scale.setToolTip(
            "变形图的放大倍数。判断变形是真的大还是被放大了，靠调这个")
        self.scale.setMinimumWidth(84)
        self.scale.currentTextChanged.connect(self._on_scale)
        row.addWidget(self.scale)

        row.addStretch(1)
        self._filling = False

    def _on_plane(self, plane: str) -> None:
        self.offset_label.setText({"XY": "z=", "XZ": "y=", "YZ": "x="}.get(
            plane, "="))
        self._window.set_work_plane(plane)

    def _on_case(self, text: str) -> None:
        if not self._filling and text:
            self.case_changed.emit(text)

    def _on_scale(self, text: str) -> None:
        """0 表示自动。**看不懂的输入一律回到自动**，不要报错——
        这是个常驻输入框，用户会在里面乱敲，弹窗会很烦人。"""
        if self._filling:
            return
        text = (text or "").strip()
        if text in ("", "自动", "auto"):
            self.scale_changed.emit(0.0)
            return
        try:
            value = float(text)
        except ValueError:
            return
        self.scale_changed.emit(max(0.0, value))

    def set_cases(self, names: list[str], current: str | None) -> None:
        """刷新工况列表。**填的时候要屏蔽信号**，否则 clear() 会发出
        一个空的 currentTextChanged，把当前工况清掉。"""
        self._filling = True
        try:
            self.cases.clear()
            self.cases.addItems(names)
            if current and current in names:
                self.cases.setCurrentText(current)
            self.cases.setEnabled(bool(names))
        finally:
            self._filling = False


def build(window) -> Ribbon:
    """按 Abaqus/CAE 的模块顺序组织功能区：
    Part → Property → Assembly → Step → Load → Mesh → Job → Visualization

    每个模块只做一类事，模块之间有明确顺序——这是 Abaqus 的核心逻辑。
    """
    a = window.actions_by_name
    r = Ribbon(window)

    # ========== 建模 ==========
    p = r.page("建模")
    g = p.group("创建")
    g.add_large(a["sketch"]); g.add_large(a["sketch_ai"])
    g.add_large(a["frame"])
    g = p.group("编辑")
    g.add_small(a["brace"])
    g = p.group("查看")
    g.add_small(a["tables"]); g.add_small(a["json"])

    # ========== 属性 ==========
    p = r.page("属性")
    g = p.group("材料")
    g.add_large(a["material"])
    g = p.group("截面")
    g.add_large(a["beam_section"]); g.add_large(a["section"])
    g = p.group("指派")
    g.add_large(a["assign_section"]); g.add_small(a["hinge"])
    g = p.group("单位")
    g.add_small(a["units"])

    # ========== 载荷 ==========
    p = r.page("载荷")
    g = p.group("边界条件")
    g.add_large(a["create_bc"])
    g = p.group("载荷")
    g.add_large(a["create_load"]); g.add_large(a["gravity"])
    g = p.group("工况")
    g.add_small(a["load"]); g.add_small(a["combo"])

    # ========== 分析 ==========
    p = r.page("分析")
    g = p.group("求解")
    g.add_large(a["solve"]); g.add_small(a["analysis_step"])
    g = p.group("特征值")
    g.add_small(a["buckling"])
    g = p.group("局部实体")
    g.add_large(a["solid_joint"])
    g = p.group("检查")
    g.add_small(a["diagnose"]); g.add_small(a["analysis_mesh"])
    g = p.group("历史")
    g.add_small(a["timeline"])

    # ========== 结果 ==========
    p = r.page("结果")
    g = p.group("显示")
    g.add_large(a["diagram"])
    g.add_small(a["clear_results"])
    g = p.group("查询")
    g.add_small(a["curve"]); g.add_small(a["deflection"])
    g.add_small(a["envelope"]); g.add_small(a["report"])
    # 校核单独成一组：它回答的是"够不够"，与"是多少"不是一回事，
    # 混在查询里会让人以为它只是又一种查询方式。
    g = p.group("校核")
    g.add_large(a["strength"])
    g.add_small(a["symmetry"]); g.add_small(a["bandwidth"])

    # ========== 视图 ==========
    # 注意：标准视角（前/侧/顶/等轴测/适应）和编号标注在下方常驻快捷栏
    # 每个阶段都要用，这里不再重复占位置，只放快捷栏没有的视图设置。
    p = r.page("视图")
    g = p.group("视口背景")
    g.add_large(a["bg_settings"])
    g = p.group("参考显示")
    g.add_large(a["grid_floor"])
    g = p.group("视角管理")
    g.add_small(a["camera"])

    # ========== 项目 ==========
    p = r.page("项目")
    g = p.group("文件")
    g.add_large(a["new"]); g.add_small(a["open"]); g.add_small(a["save"])
    g = p.group("面板")
    g.add_small(a["chat"])
    g = p.group("Agent 学习")
    g.add_small(a["learning_trace"])

    # 项目在最左侧；之后按 Part → Property → Load → Analysis → Results 走。
    r.tabBar().moveTab(r.indexOf(r.pages["项目"]), 0)

    return r


def icon_for(name: str, size: int = 32):
    return icons.icon(name, size)
