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
                               QHBoxLayout, QLabel, QMenu, QSizePolicy,
                               QStackedWidget, QTabWidget, QToolBar, QToolButton,
                               QWidgetAction,
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
    if action.objectName() in {"solve", "frame"}:
        b.setProperty("role", "primary")
    if large:
        b.setMinimumWidth(58)
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

    collapsed_changed = Signal(bool)
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
        for page in self.pages.values():
            page.ensurePolished()
        expanded = max(self.EXPANDED_HEIGHT,
                       max((page.minimumSizeHint().height() for page in self.pages.values()), default=0)
                       + self.tabBar().sizeHint().height() + 4)
        self.setMaximumHeight(self.tabBar().sizeHint().height() + 4
                              if self._collapsed else expanded)
        self._sync_toggle()
        self.collapsed_changed.emit(self._collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def _sync_toggle(self) -> None:
        from . import i18n
        self._toggle.setText(i18n.tr("更多命令" if self._collapsed else "收起命令"))
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


class ContextStack(QStackedWidget):
    """只按当前控件组计算宽度，避免隐藏的建模输入撑大结果工具栏。"""

    def sizeHint(self):
        return self.currentWidget().sizeHint() if self.currentWidget() else super().sizeHint()

    def minimumSizeHint(self):
        return self.currentWidget().minimumSizeHint() if self.currentWidget() else super().minimumSizeHint()


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
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.toolbar = QToolBar(self)
        self.toolbar.setObjectName("workspaceTools")
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        outer.addWidget(self.toolbar)
        row = self.toolbar

        def labelled(name, into, text=None):
            button = _button(a[name], False)
            button.setIconSize(QSize(18, 18))
            if text is not None:
                button.setText(text)
            into.addWidget(button)
            return button

        def popup(text, names, into):
            button = QToolButton(self)
            button.setText(text)
            button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            menu = QMenu(button)
            panel = QWidget(menu)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)
            for name in names:
                item = labelled(name, layout)
                item.clicked.connect(menu.close)
            action = QWidgetAction(menu)
            action.setDefaultWidget(panel)
            menu.addAction(action)
            button.setMenu(menu)
            into.addWidget(button)
            return button, layout

        self.view_button, _ = popup("视角", ("front", "side", "top", "iso"), row)
        self.view_button.setIcon(a["iso"].icon())
        labelled("fit", row)
        row.addSeparator()
        for name in ("undo", "redo"):
            button = labelled(name, row)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        row.addSeparator()

        self.common = QWidget(self)
        common_row = QHBoxLayout(self.common)
        common_row.setContentsMargins(0, 0, 0, 0)
        common_row.setSpacing(4)
        self.common_commands = {
            "项目": ("open", "save"), "建模": ("frame", "sketch"),
            "属性": ("material", "assign_section"),
            "载荷": ("create_bc", "create_load"),
            "分析": ("diagnose", "solve"), "结果": ("strength", "report"),
            "视图": ("bg_settings", "grid_floor"),
        }
        self.common_buttons = {}
        for names in self.common_commands.values():
            for name in names:
                button = labelled(name, common_row)
                button.setVisible(False)
                self.common_buttons[name] = button
        row.addWidget(self.common)
        row.addSeparator()

        self.context = ContextStack(self)
        self.context.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        row.addWidget(self.context)
        build_page = QWidget(self)
        build_row = QHBoxLayout(build_page)
        build_row.setContentsMargins(0, 0, 0, 0)
        build_row.setSpacing(4)
        popup("选择", ("pick_node", "pick_member"), build_row)
        self.edit_button, editor = popup(
            "编辑", ("model_node", "model_member", "delete", "labels", "props"), build_row)

        self.precise = QWidget(self)
        build_row = QHBoxLayout(self.precise)
        build_row.setContentsMargins(4, 6, 4, 6)
        build_row.setSpacing(6)
        editor.addWidget(self.precise)
        self.precise.setVisible(False)
        for name in ("model_node", "model_member"):
            a[name].toggled.connect(self._sync_precise)
        build_row.addWidget(QLabel("工作平面"))
        self.plane = QComboBox(self)
        self.plane.addItems(["XY", "XZ", "YZ"])
        self.plane.setFixedWidth(_fits(self.plane, "XZ"))
        self.plane.setToolTip("XY 固定 z，XZ 固定 y，YZ 固定 x；节点投影到当前工作平面。")
        self.plane.currentTextChanged.connect(self._on_plane)
        build_row.addWidget(self.plane)
        self.offset_label = QLabel("z=")
        build_row.addWidget(self.offset_label)
        self.offset = QDoubleSpinBox(self)
        self.offset.setRange(-1.0e6, 1.0e6)
        self.offset.setDecimals(3)
        self.offset.setSingleStep(0.5)
        self.offset.setMinimumWidth(_fits(self.offset, "-99999.999"))
        self.offset.setToolTip("工作平面的固定坐标，单位与模型长度单位一致。")
        self.offset.valueChanged.connect(window.set_work_offset)
        build_row.addWidget(self.offset)
        build_row.addWidget(QLabel("捕捉"))
        self.snap = QComboBox(self)
        self.snap.setEditable(True)
        self.snap.addItems(["关闭", "0.1", "0.25", "0.5", "1", "2"])
        self.snap.setMinimumWidth(_fits(self.snap, "关闭"))
        self.snap.setCurrentIndex(0)
        self.snap.currentTextChanged.connect(window.set_snap_size)
        build_row.addWidget(self.snap)
        self.coord_btn = QToolButton(self)
        self.coord_btn.setText("坐标建点")
        self.coord_btn.clicked.connect(window.create_node_exact)
        build_row.addWidget(self.coord_btn)

        result_page = QWidget(self)
        result_row = QHBoxLayout(result_page)
        result_row.setContentsMargins(0, 0, 0, 0)
        result_row.setSpacing(6)
        result_row.addWidget(QLabel("工况"))
        self.cases = QComboBox(self)
        self.cases.setProperty("i18nSelfManaged", True)
        self.cases.setMinimumWidth(110)
        self.cases.setMaximumWidth(180)
        self.cases.setToolTip("切换显示哪个荷载工况或组合的结果")
        self.cases.currentTextChanged.connect(self._on_case)
        result_row.addWidget(self.cases)
        result_row.addWidget(QLabel("变形放大"))
        self.scale = QComboBox(self)
        self.scale.setEditable(True)
        self.scale.addItems(["自动", "1", "10", "50", "100", "200", "500"])
        self.scale.setMinimumWidth(_fits(self.scale, "自动"))
        self.scale.setMaximumWidth(110)
        self.scale.setToolTip("变形显示的放大倍数，0 或自动表示自动取值。")
        self.scale.currentTextChanged.connect(self._on_scale)
        result_row.addWidget(self.scale)
        self.context.addWidget(build_page)
        self.context.addWidget(result_page)
        self._context_pages = {"建模": 0, "结果": 1}

        space = QWidget(self)
        space.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row.addWidget(space)
        row.addSeparator()
        self.mode_button, modes = popup("模型", (), row)
        self.mode_button.setProperty("i18nSelfManaged", True)
        self.mode_button.setProperty("segment", True)
        self.mode_buttons = {}
        for label, name in (("模型", "model"), ("分析网格", "analysis_mesh"),
                            ("变形", "deformed"), ("云图", "contour"),
                            ("内力图", "force_diagram"), ("应力比", "utilization"),
                            ("模态", "modal")):
            button = labelled(name, modes, label)
            button.clicked.connect(self.mode_button.menu().close)
            self.mode_buttons[label] = button
        self._filling = False
        window.ribbon.collapsed_changed.connect(
            lambda _value: self.show_context_for(window.ribbon.tabText(window.ribbon.currentIndex()), window.mode))


    def _sync_precise(self) -> None:
        """建节点/建杆件任一勾上就显示精确建模控件，都不勾就收起来。"""
        a = self._window.actions_by_name
        self.precise.setVisible(any(a[n].isChecked()
                                    for n in ("model_node", "model_member")))

    # 看这几个显示模式时，需要的是工况与放大倍数，不是建节点
    RESULT_MODES = ("变形", "云图", "内力图", "应力比", "模态")

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
            self.context.updateGeometry()
        names = self.common_commands.get(page, ())
        for name, button in self.common_buttons.items():
            button.setVisible(name in names and self._window.ribbon.is_collapsed())
        from . import i18n
        current_mode = mode or self._window.mode
        self.mode_button.setText(i18n.tr(current_mode))
        if current_mode in self.mode_buttons:
            self.mode_button.setIcon(self.mode_buttons[current_mode].icon())

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
    g = p.group("规范与导荷")
    g.add_large(a["code_combo"]); g.add_small(a["live_pattern"])
    g.add_small(a["area_load"]); g.add_small(a["bc_manager"])

    # ========== 分析 ==========
    p = r.page("分析")
    g = p.group("求解")
    g.add_large(a["solve"]); g.add_small(a["solve_options"])
    g = p.group("分析步")
    g.add_large(a["step_manager"]); g.add_small(a["amplitude"])
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
