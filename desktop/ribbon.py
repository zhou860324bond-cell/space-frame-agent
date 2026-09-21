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
                               QHBoxLayout, QLabel, QSizePolicy,
                               QStackedWidget, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from . import icons

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

    EXPANDED_HEIGHT = 112

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setProperty("ribbon", "bar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setMaximumHeight(self.EXPANDED_HEIGHT)
        self.pages: dict[str, RibbonPage] = {}
        self._collapsed = False

        # 顶部原来是四层：页签 + 功能区 + 快捷栏 + 流程条，一共吃掉约 260px，
        # 在 1000px 高的窗口上是 26%。功能区是其中最高的一层，而它又是**查完
        # 就不用**的那一层——建好模型之后大部分时间在看视口。
        # 所以给它一个收起开关：双击页签，或点右端那个按钮。
        self._toggle = QToolButton(self)
        self._toggle.setAutoRaise(True)
        # 属性名刻意不叫 "ribbon"：那个标记的语义是"动作驱动的命令按钮"，
        # 有测试按它来断言每个按钮背后都挂着 QAction。折叠开关是视图 chrome，
        # 不是命令，借用那个名字会把那条断言变成假红。
        self._toggle.setProperty("ribbonChrome", "collapse")
        self.setCornerWidget(self._toggle, Qt.Corner.TopRightCorner)
        self._toggle.clicked.connect(self.toggle_collapsed)
        self.tabBarDoubleClicked.connect(lambda _index: self.toggle_collapsed())
        self._sync_toggle()

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """收起/展开功能区内容，页签始终留着。

        只收内容不收页签：页签同时是"我在哪个模块"的指示，收掉它就等于
        把当前所处的阶段也藏了，那是另一种反人类。
        """
        self._collapsed = bool(collapsed)
        for name in self.pages:
            self.pages[name].setVisible(not self._collapsed)
        self.setMaximumHeight(self.tabBar().sizeHint().height() + 4
                              if self._collapsed else self.EXPANDED_HEIGHT)
        self._sync_toggle()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def _sync_toggle(self) -> None:
        self._toggle.setText("⌄" if self._collapsed else "⌃")
        self._toggle.setToolTip("展开功能区（也可双击页签）" if self._collapsed
                                else "收起功能区，把高度让给视口（也可双击页签）")

    def page(self, title: str) -> RibbonPage:
        p = RibbonPage(self)
        self.addTab(p, title)
        self.pages[title] = p
        return p


def _fits(widget, sample: str) -> int:
    """能容下 `sample` 的最小控件宽度。**chrome 从 Qt 自己的 sizeHint 反推。**

    写死像素值在这个项目里会裂。工作平面的 z 输入框原来是 `setFixedWidth(78)`，
    而它量程 ±1e6、3 位小数，实拍里 "0.000" 的最后一位被切掉、上下箭头压在
    数字上（显示成 "0.00("）。

    也不能改成硬编码的"边框 2 + padding 14 + 箭头 18"——那同样是猜。样式表
    的 padding、下拉箭头宽度、spinbox 上下按钮的排布（Windows 11 风格是左右
    并排，约 44px，不是上下叠放的 18px）都会变。

    所以这里只做一件事：拿控件**自己**的 sizeHint 减去它据以计算 sizeHint 的
    那段文字宽度，差值就是这个控件在当前样式下的真实 chrome，再加上我们真正
    想显示的 `sample`。换字号、换 DPI、换样式表、切英文之后都成立。
    """
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QAbstractSpinBox, QComboBox

    fm = QFontMetrics(widget.font())
    if isinstance(widget, QAbstractSpinBox):
        # spinbox 的 sizeHint 按量程两端里更长的那个文本算
        widest = max((widget.textFromValue(widget.minimum()),
                      widget.textFromValue(widget.maximum())), key=len)
    elif isinstance(widget, QComboBox):
        widest = max((widget.itemText(i) for i in range(widget.count())),
                     key=lambda t: fm.horizontalAdvance(t), default="")
    else:
        widest = ""
    chrome = widget.sizeHint().width() - fm.horizontalAdvance(widest)
    return int(fm.horizontalAdvance(sample) + max(chrome, 0) + 4)


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

        def separator(into=None) -> None:
            sep = QFrame(self)
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setProperty("toolbar", "sep")
            (into or row).addWidget(sep)

        def section(text: str, into=None) -> None:
            label = QLabel(text, self)
            label.setProperty("toolbar", "section")
            (into or row).addWidget(label)

        def labelled(name: str, into=None, size: int = 16):
            """带文字的工具按钮。

            **纯图标是这条工具栏原来最大的毛病**：18 个 20px 的线框图标挤在
            一起，不逐个悬停认不出任何一个。图标只在含义早已固化时才独立成立
            （撤销/重做那种），其余一律配字。
            """
            b = _button(a[name], False)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setIconSize(QSize(size, size))
            (into or row).addWidget(b)
            return b

        def icon_only(name: str, into=None, size: int = 18):
            b = _button(a[name], False)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            b.setIconSize(QSize(size, size))
            (into or row).addWidget(b)
            return b

        # ---- 左段：任何阶段都要用的东西，固定不动 ----
        section("视角")
        for n in ("front", "side", "top", "iso", "fit"):
            labelled(n)
        separator()
        # 撤销/重做的图标是通用约定，配字反而占地方
        icon_only("undo"); icon_only("redo")
        separator()

        # ---- 中段：随当前功能区页变化 ----
        # 原来这里把"建模用的"和"看结果用的"控件**同时**摆着：选择/建节点/
        # 删除/工作平面/捕捉，和工况/放大，一共十几项常驻。可它们从来不会在
        # 同一时刻被用到——建几何时没有结果可切，审结果时不会去建节点。
        # 按阶段切换之后，同屏可见的控件少一半，而且剩下的都配了字。
        self.context = QStackedWidget(self)
        self.context.setSizePolicy(QSizePolicy.Policy.Maximum,
                                   QSizePolicy.Policy.Preferred)
        row.addWidget(self.context)

        build_page = QWidget(self)
        build_row = QHBoxLayout(build_page)
        build_row.setContentsMargins(0, 0, 0, 0)
        build_row.setSpacing(2)
        section("选择", build_row)
        for n in ("pick_node", "pick_member"):
            labelled(n, build_row)
        separator(build_row)
        section("编辑", build_row)
        for n in ("model_node", "model_member", "delete"):
            labelled(n, build_row)
        separator(build_row)
        icon_only("labels", build_row); icon_only("props", build_row)
        separator(build_row)

        # 精确建模：工作平面 + 网格捕捉 + 精确坐标建点。
        # 三维里直接点坐标会飘，这几个控件把节点"锁"在工作平面和网格上。
        #
        # **只在真的要放点时才出现。** 这一组实测占 425px，是整条工具栏放不下
        # 模式切换的直接原因；而不点节点的时候它一个都用不上——工作平面、捕捉
        # 间距只影响"下一个点落在哪儿"。所以跟着建节点/建杆件的勾选状态显隐。
        self.precise = QWidget(self)
        precise_row = QHBoxLayout(self.precise)
        precise_row.setContentsMargins(0, 0, 0, 0)
        precise_row.setSpacing(2)
        build_row.addWidget(self.precise)
        self.precise.setVisible(False)
        for _name in ("model_node", "model_member"):
            a[_name].toggled.connect(self._sync_precise)
        build_row = precise_row
        section("工作平面", build_row)
        self.plane = QComboBox(self)
        self.plane.addItems(["XY", "XZ", "YZ"])
        self.plane.setFixedWidth(_fits(self.plane, "XZ"))
        self.plane.setToolTip(
            "工作平面：新建节点先投影到该平面\n"
            "XY=水平面固定 z，XZ=正立面固定 y，YZ=侧立面固定 x")
        self.plane.currentTextChanged.connect(self._on_plane)
        build_row.addWidget(self.plane)

        self.offset_label = QLabel("z=")
        build_row.addWidget(self.offset_label)
        self.offset = QDoubleSpinBox(self)
        self.offset.setRange(-1.0e6, 1.0e6)
        self.offset.setDecimals(3)
        self.offset.setSingleStep(0.5)
        # **别写死宽度。** 原来是 setFixedWidth(78)，而这个框量程 ±1e6、
        # 3 位小数——实拍里 "0.000" 的最后一位被切掉、上下箭头压在数字上。
        self.offset.setMinimumWidth(_fits(self.offset, "-99999.999"))
        self.offset.setToolTip(
            "工作平面位置：XY 填 z，XZ 填 y，YZ 填 x；单位跟随当前模型")
        self.offset.valueChanged.connect(window.set_work_offset)
        build_row.addWidget(self.offset)

        build_row.addWidget(QLabel("捕捉"))
        self.snap = QComboBox(self)
        self.snap.setEditable(True)
        self.snap.addItems(["关闭", "0.1", "0.25", "0.5", "1", "2"])
        self.snap.setMinimumWidth(_fits(self.snap, "关闭"))
        self.snap.setToolTip(
            "网格捕捉间距：工作平面内按此间距对齐坐标\n"
            "选“关闭”则不吸附网格；靠近已有节点始终会自动吸附")
        # 构建期逐项触发会把捕捉设成中间值，先屏蔽，定到“关闭”
        self.snap.blockSignals(True)
        self.snap.setCurrentIndex(0)
        self.snap.blockSignals(False)
        self.snap.currentTextChanged.connect(window.set_snap_size)
        build_row.addWidget(self.snap)

        self.coord_btn = QToolButton(self)
        self.coord_btn.setText("坐标建点")
        self.coord_btn.setAutoRaise(True)
        self.coord_btn.setToolTip("输入精确 x/y/z 坐标创建节点（鼠标点不准时用）")
        self.coord_btn.clicked.connect(window.create_node_exact)
        build_row.addWidget(self.coord_btn)
        build_row.addStretch(1)

        result_page = QWidget(self)
        result_row = QHBoxLayout(result_page)
        result_row.setContentsMargins(0, 0, 0, 0)
        result_row.setSpacing(2)
        result_row.addWidget(QLabel("工况"))
        self.cases = QComboBox(self)
        # 工况名是用户自己起的，是**内容不是界面**：真有人把工况叫"全部"，
        # 切英文时也不能把它翻成 "All"。让 i18n 绕开这一个下拉。
        self.cases.setProperty("i18nSelfManaged", True)
        self.cases.setMinimumWidth(132)
        self.cases.setToolTip("切换显示哪个荷载工况或组合的结果")
        self.cases.currentTextChanged.connect(self._on_case)
        result_row.addWidget(self.cases)

        result_row.addWidget(QLabel("变形放大"))
        self.scale = QComboBox(self)
        self.scale.setEditable(True)
        self.scale.addItems(["自动", "1", "10", "50", "100", "200", "500"])
        self.scale.setToolTip(
            "变形图的放大倍数。判断变形是真的大还是被放大了，靠调这个")
        self.scale.setMinimumWidth(_fits(self.scale, "自动"))
        self.scale.currentTextChanged.connect(self._on_scale)
        result_row.addWidget(self.scale)
        result_row.addStretch(1)

        self.context.addWidget(build_page)
        self.context.addWidget(result_page)
        self._context_pages = {"建模": 0, "结果": 1}

        row.addStretch(1)

        # ---- 右段：显示模式。**这是状态不是动作，必须看得出当前在哪个。** ----
        # 原来它和旁边的动作按钮长得一模一样（四个无字小图标），既看不出
        # 自己在看什么，也容易误点。改成一排带字的互斥按钮。
        separator()
        section("显示")
        self.mode_buttons: dict[str, QToolButton] = {}
        for label, name in (("模型", "model"), ("分析网格", "analysis_mesh"),
                            ("变形", "deformed"), ("云图", "contour"),
                            ("模态", "modal")):
            b = labelled(name)
            # 按钮上写 MODES 里的短名。动作本身叫"梁内力云图"（说明书式的
            # 全名，tooltip 里保留），但这排是**并列的状态标签**，
            # 一个 105px 的长名会把整排挤歪。
            b.setText(label)
            b.setProperty("segment", True)
            self.mode_buttons[label] = b
        self._filling = False

    def _sync_precise(self) -> None:
        """建节点/建杆件任一勾上就显示精确建模控件，都不勾就收起来。"""
        a = self._window.actions_by_name
        self.precise.setVisible(any(a[n].isChecked()
                                    for n in ("model_node", "model_member")))

    # 看这几个显示模式时，需要的是工况与放大倍数，不是建节点
    RESULT_MODES = ("变形", "云图", "模态")

    def show_context_for(self, page: str, mode: str | None = None) -> None:
        """切换中段控件。

        规则有先后：**显示模式优先于功能区页**。用户正在看变形图却停在
        "建模"页是很常见的（刚建完就去看结果，页签没跟着动），这时该给的是
        工况和放大倍数，而不是"建节点/建杆件"。只有不在结果模式时，
        才按功能区页判断。
        """
        if mode in self.RESULT_MODES:
            key = "结果"
        elif page in ("结果", "分析"):
            key = "结果"
        else:
            key = "建模"
        index = self._context_pages.get(key)
        if index is not None and index != self.context.currentIndex():
            self.context.setCurrentIndex(index)

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
    # 分析网格是**显示模式**不是一次性动作，它和模型/变形/云图/模态一起
    # 归常驻快捷栏右段的模式分段控件，这里不再重复占位置——
    # 与「视图」页不重复标准视角是同一条约定。
    g.add_small(a["diagnose"])
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
    g.add_small(a["load_labels"])
    g = p.group("视角管理")
    g.add_small(a["camera"])
    g = p.group("语言")
    g.add_small(a["lang"])

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
