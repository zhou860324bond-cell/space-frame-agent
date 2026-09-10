"""主窗口。

布局照 CAE 的通行做法：顶部模块工具栏、左侧模型树、中间视口、
底部状态栏，右侧一个可停靠的属性/结果面板。

**主窗口只搭骨架和转发动作**，算什么、画什么都在别处。这样它才不会长成
那种两千行的上帝类——参照项目的 `main_window.py` 就是那样，
一个文件塞下了整个应用。
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (QDialog, QDockWidget, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMenu, QMessageBox, QPushButton,
                               QScrollArea, QStatusBar, QTabWidget, QToolBar,
                               QVBoxLayout, QWidget)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import Session                          # noqa: E402

from . import scene, theme                         # noqa: E402
from .chat_panel import ChatPanel                  # noqa: E402
from .model_tree import ModelTree                  # noqa: E402
from .diagram_panel import DiagramPanel             # noqa: E402
from .panels import ResultPanel, TimelinePanel      # noqa: E402
from .sketch_panel import SketchPanel                # noqa: E402
from .section_opt_panel import SectionOptPanel       # noqa: E402
from .bc_panel import BCPanel                          # noqa: E402
from .properties import PropertiesPanel             # noqa: E402
from .floating_button import FloatingAgentButton      # noqa: E402
from . import result_rows                           # noqa: E402
from .viewport import Viewport                     # noqa: E402
from .worker import Runner                         # noqa: E402
from .workflow_bar import WorkflowBar, draft_target  # noqa: E402

def _takes_command(handler) -> bool:
    """处理函数要不要收那条 Command。

    与其让每个处理函数都写一个用不上的参数，不如在这里看一眼签名。
    """
    import inspect
    try:
        return bool(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return False


def _format_combo_factors(factors: dict[str, float]) -> str:
    """把内核使用的 factors 显示成可编辑的工程表达式。"""
    return " + ".join(f"{float(value):g}×{name}" for name, value in factors.items())


def _parse_combo_expression(expression: str, case_names: set[str]) -> dict[str, float]:
    """将 ``1.3×DL + 1.5×LL`` 转为内核要求的 factors。"""
    text = expression.replace("×", "*").replace(" ", "")
    if not text:
        raise ValueError("组合表达式不能为空")
    factors: dict[str, float] = {}
    for term in re.findall(r"[+-]?[^+-]+", text):
        sign = -1.0 if term.startswith("-") else 1.0
        term = term.lstrip("+-")
        if term.count("*") == 0:
            factor, case = 1.0, term
        elif term.count("*") == 1:
            raw_factor, case = term.split("*", 1)
            try:
                factor = float(raw_factor)
            except ValueError as exc:
                raise ValueError(f"系数 {raw_factor!r} 不是数字") from exc
        else:
            raise ValueError(f"项 {term!r} 只能包含一个乘号")
        if not case:
            raise ValueError("每一项都必须指定工况名")
        if case not in case_names:
            raise ValueError(f"工况 {case!r} 不存在，可用：{'、'.join(sorted(case_names))}")
        factors[case] = factors.get(case, 0.0) + sign * factor
    factors = {case: factor for case, factor in factors.items() if factor}
    if not factors:
        raise ValueError("组合系数不能全部为 0")
    return factors


# 视口显示模式。切模式时整个场景重建。
MODES = ("模型", "分析网格", "变形", "云图", "模态")


class MainWindow(QMainWindow):
    def __init__(self, session: Session | None = None):
        super().__init__()
        self.session = session or Session()
        self.result = None
        self.case: str | None = None
        self.mode = "模型"
        # 空间结构里各杆局部轴不同，默认用旋转不变量合弯矩。需要核对有符号
        # 局部分量时再切到 My/Mz；默认直接画 Mz 会让不同方向杆件看起来乱跳色。
        self.component = "M"
        self.scale = 0.0
        self.analysis_type = "linear"
        self.analysis_options = {"increments": 10, "max_iter": 40,
                                 "tolerance": 1e-7}

        self.setWindowTitle("FrameLab Studio — 空间刚架分析")
        self.resize(1440, 900)
        self.setDockNestingEnabled(True)
        self.setTabPosition(Qt.DockWidgetArea.BottomDockWidgetArea,
                            QTabWidget.TabPosition.South)

        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)

        # 求解和大模型调用都得离开界面线程，否则窗口会变"未响应"
        self.runner = Runner(self)

        self.chat = ChatPanel(self.session, self.runner, self)
        self.chat.model_changed.connect(self._on_chat_changed)
        self.chat.busy_changed.connect(self._on_busy)
        chat_dock = QDockWidget("AI 助手", self)
        chat_dock.setObjectName("assistantDock")
        chat_dock.setWidget(self.chat)
        chat_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea
                                   | Qt.DockWidgetArea.LeftDockWidgetArea)
        chat_dock.setMinimumWidth(320)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, chat_dock)
        self.chat_dock = chat_dock
        # 默认显示在右侧
        self.chat_dock.setVisible(True)
        self.chat_dock.raise_()

        self.timeline = TimelinePanel(self)
        self.timeline.goto_step.connect(self.goto_step)
        self.timeline_dock = QDockWidget("建模过程", self)
        self.timeline_dock.setObjectName("timelineDock")
        self.timeline_dock.setWidget(self.timeline)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self.timeline_dock)
        self.timeline_dock.hide()

        self.results = ResultPanel(self)
        self.results.locate.connect(self.locate)
        self.results.display_changed.connect(self._on_result_display_changed)
        self.results.extreme_requested.connect(self.locate_current_extreme)
        self.results.stress_requested.connect(self.show_section_stress)
        self.result_display_options = self.results.display_options()
        self._last_probe = None
        self.results_dock = QDockWidget("结果", self)
        self.results_dock.setObjectName("resultsDock")
        self.results_dock.setWidget(self.results)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                           self.results_dock)
        self.results_dock.hide()
        # 结果面板是配角，不该抢视口的地方。给个偏小的初始高度，
        # 用户想看细节自己拖大
        self.results.setMinimumHeight(120)
        self.results.setMaximumHeight(260)

        self.diagram = DiagramPanel(self)
        self.diagram.locate.connect(self.locate)
        self.diagram_dock = QDockWidget("内力图", self)
        self.diagram_dock.setObjectName("diagramDock")
        self.diagram_dock.setWidget(self.diagram)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                           self.diagram_dock)
        self.diagram_dock.hide()
        # 内力图和结果表放同一处，用页签切——它们都是"看数"，
        # 分成两块会把视口挤没
        self.tabifyDockWidget(self.results_dock, self.diagram_dock)

        # 截面优化：和结果/内力图叠成页签，都在底部
        self.section_opt = SectionOptPanel(self.session, self.runner, self)
        self.section_opt_dock = QDockWidget("截面优化", self)
        self.section_opt_dock.setObjectName("sectionOptDock")
        self.section_opt_dock.setWidget(self.section_opt)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                           self.section_opt_dock)
        self.section_opt_dock.hide()
        self.tabifyDockWidget(self.diagram_dock, self.section_opt_dock)

        self.properties = PropertiesPanel(self.session, self)
        self.properties.edited.connect(self._on_property_edited)
        self.props_dock = QDockWidget("属性", self)
        self.props_dock.setObjectName("propertiesDock")
        self.props_dock.setWidget(self.properties)
        self.props_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea
                                        | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.props_dock)
        # 属性面板默认隐藏，选中对象后自动显示
        self.props_dock.setVisible(False)

        # 手绘草图 → 模型：隐藏，通过 Part 模块的"草图建模"按钮弹出对话框
        self.sketch = SketchPanel(self.session, self.runner, self)
        self.sketch.model_loaded.connect(self._on_sketch_loaded)
        self.sketch_dock = QDockWidget("手绘草图", self)
        self.sketch_dock.setObjectName("sketchDock")
        self.sketch_scroll = QScrollArea(self.sketch_dock)
        self.sketch_scroll.setWidgetResizable(True)
        self.sketch_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sketch_scroll.setWidget(self.sketch)
        self.sketch_dock.setWidget(self.sketch_scroll)
        self.sketch_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea
                                          | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sketch_dock)
        self.sketch_dock.setVisible(False)

        # 边界条件面板：隐藏，通过 Load 模块的"创建边界条件"按钮弹出对话框
        self.bc = BCPanel(self.session, self)
        self.bc.changed.connect(
            lambda: self._after_manual_edit("边界条件或荷载已修改"))
        self.bc_dock = QDockWidget("边界条件", self)
        self.bc_dock.setObjectName("boundaryDock")
        self.bc_scroll = QScrollArea(self.bc_dock)
        self.bc_scroll.setWidgetResizable(True)
        self.bc_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.bc_scroll.setWidget(self.bc)
        self.bc_dock.setWidget(self.bc_scroll)
        self.bc_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea
                                      | Qt.DockWidgetArea.LeftDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.bc_dock)
        self.bc_dock.setVisible(False)

        self.viewport.picked.connect(self._on_picked)
        self.viewport.probed.connect(self.probe_member_result)
        self.viewport.node_created.connect(self._on_node_created)
        self.viewport.member_created.connect(self._on_member_created)
        self.viewport.node_snapped.connect(self._on_node_snapped)
        self.viewport.escape_pressed.connect(self.cancel_interaction)
        self.viewport.context_requested.connect(self._show_viewport_context_menu)

        self.tree = ModelTree(self)
        self.tree.activated_item.connect(self._on_tree_action)
        dock = QDockWidget("模型树", self)
        dock.setObjectName("modelTreeDock")
        dock.setWidget(self.tree)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.tree_dock = dock
        self._tree_auto_hidden = False

        # ===== 最后创建浮动按钮，确保它在所有组件之上 =====
        # 右侧中间圆形按钮 —— 点击召唤/收起 AI 助手对话框
        # 放在视口右侧边缘，对话框显示时也挡不住
        self.agent_button = FloatingAgentButton(self)
        self.agent_button.clicked.connect(self.toggle_agent_panel)
        self.chat_dock.visibilityChanged.connect(self._on_chat_visibility)
        self.agent_button.show()
        self.agent_button.raise_()

        self._build_actions()
        self._build_empty_state()
        self._build_ribbon()
        self._build_menus()
        self._build_statusbar()
        self.refresh()

    # --- 构件 ---

    def _build_empty_state(self) -> None:
        """Offer the three real starting paths inside an otherwise empty viewport."""
        panel = QFrame(self.viewport)
        panel.setObjectName("emptyState")
        # 原来是写死 500×210 再 addStretch，于是标题和按钮之间空出一大片。
        # 空白本身不是问题，**没有内容的空白**才是：它让这张卡片看着像没画完。
        # 改成按内容定高，把省下来的位置用来说清楚下一步会发生什么。
        panel.setFixedWidth(520)
        box = QVBoxLayout(panel)
        box.setContentsMargins(28, 22, 28, 22)
        box.setSpacing(8)
        title = QLabel("创建第一个结构模型", panel)
        title.setProperty("emptyState", "title")
        subtitle = QLabel(
            "从图纸识别、参数化生成或二维草图开始；三条路径最终进入同一模型。",
            panel)
        subtitle.setProperty("emptyState", "subtitle")
        subtitle.setWordWrap(True)
        steps = QLabel(
            "建好几何之后，按顶部「分析流程」往右走：定义材料截面与荷载 → "
            "校验 → 求解 → 看变形、内力与校核结论。",
            panel)
        steps.setProperty("emptyState", "subtitle")
        steps.setWordWrap(True)
        box.addWidget(title)
        box.addWidget(subtitle)
        box.addSpacing(2)
        box.addWidget(steps)
        box.addSpacing(6)
        actions = QHBoxLayout()
        for text, action_name, primary in (
                ("AI 识别图纸", "sketch_ai", True),
                ("参数化框架", "frame", False),
                ("二维草图", "sketch", False)):
            button = QPushButton(text, panel)
            button.setMinimumHeight(34)
            if primary:
                button.setProperty("role", "primary")
            button.clicked.connect(self.actions_by_name[action_name].trigger)
            actions.addWidget(button)
        box.addLayout(actions)
        panel.adjustSize()
        self.empty_state = panel
        self._position_empty_state()

    def _position_empty_state(self) -> None:
        if not hasattr(self, "empty_state"):
            return
        area = self.viewport.rect()
        x = max(12, (area.width() - self.empty_state.width()) // 2)
        y = max(12, (area.height() - self.empty_state.height()) // 2)
        self.empty_state.move(x, y)
        self.empty_state.raise_()

    def _build_actions(self) -> None:
        """按 commands.COMMANDS 建 QAction。

        **状态只存在 QAction 上**：功能区按钮绑的是同一个对象，
        所以置灰、勾选、快捷键都只写一遍，不会出现"菜单是灰的、
        功能区是亮的"这种自相矛盾。
        """
        from . import commands, icons

        self.actions_by_name: dict[str, QAction] = {}
        for cmd in commands.COMMANDS:
            act = QAction(icons.icon(cmd.icon), cmd.label, self)
            act.setObjectName(cmd.name)
            act.setToolTip(cmd.tip)
            act.setStatusTip(cmd.tip)
            if cmd.shortcut:
                act.setShortcut(cmd.shortcut)
            if cmd.checkable:
                act.setCheckable(True)
            # 晚绑定：点的时候才去主窗口上取方法，这样动作表不依赖构造顺序
            act.triggered.connect(
                lambda _=False, c=cmd: self._run_command(c))
            self.actions_by_name[cmd.name] = act
            self.addAction(act)                 # 让快捷键在窗口任何位置都生效

        # Esc 全局退出当前建模/拾取模式（视口自身也会发 escape_pressed，
        # 这里再兜一层，焦点在面板/树上时按 Esc 同样有效；cancel 幂等）
        sc_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        sc_esc.activated.connect(self.cancel_interaction)

        # 常用的几个留成属性，代码里读起来顺一点
        self._sync_selection_actions()
        self.act_new = self.actions_by_name["new"]
        self.act_solve = self.actions_by_name["solve"]
        self.act_generate = self.actions_by_name["frame"]
        self.act_report = self.actions_by_name["report"]
        self.act_chat = self.actions_by_name["chat"]
        self.act_chat.setChecked(True)

        # 四个显示模式做成互斥的一组。**看得见当前在哪个模式**，
        # 比按了之后靠图猜要好——变形图和模型图在小位移下长得很像
        self.mode_actions = {"模型": self.actions_by_name["model"],
                             "分析网格": self.actions_by_name["analysis_mesh"],
                             "变形": self.actions_by_name["deformed"],
                             "云图": self.actions_by_name["contour"],
                             "模态": self.actions_by_name["modal"]}
        group = QActionGroup(self)
        group.setExclusive(True)
        for act in self.mode_actions.values():
            group.addAction(act)
        self.mode_actions[self.mode].setChecked(True)

        # 拾取过滤器也是互斥的一组，但**允许全不选**——不选时回到纯看图，
        # 转视角不会误点中东西
        self.pick_actions = {"node": self.actions_by_name["pick_node"],
                             "member": self.actions_by_name["pick_member"]}
        picks = QActionGroup(self)
        picks.setExclusive(True)
        picks.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional)
        for act in self.pick_actions.values():
            picks.addAction(act)

    def _run_command(self, cmd) -> None:
        """分派。找不到处理函数就说清楚，不要静默无反应——
        **按了没动静是最糟的交互**，用户分不清是没点中还是坏了。"""
        handler = getattr(self, cmd.handler, None)
        if handler is None:
            QMessageBox.information(self, cmd.label,
                                    f"「{cmd.label}」还没接上（缺 {cmd.handler}）。")
            return
        try:
            handler(cmd) if _takes_command(handler) else handler()
        except Exception as e:
            QMessageBox.critical(self, f"{cmd.label} 失败",
                                 f"错误：{type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

    def _build_ribbon(self) -> None:
        from . import ribbon

        self.ribbon = ribbon.build(self)
        self.ribbon.currentChanged.connect(self._on_ribbon_page_changed)
        self.quickbar = ribbon.QuickBar(self, self)
        self.quickbar.case_changed.connect(self.set_case)
        self.quickbar.scale_changed.connect(self.set_scale)
        self.workflow_bar = WorkflowBar(self)
        self.workflow_bar.stage_requested.connect(self._on_workflow_stage)
        holder = QWidget(self)
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(self.ribbon)
        box.addWidget(self.quickbar)
        box.addWidget(self.workflow_bar)
        bar = QToolBar("功能区", self)
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.addWidget(holder)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

    def _on_ribbon_page_changed(self, index: int) -> None:
        """把当前模块的下一步操作写出来，避免用户在功能区里猜流程。"""
        page = self.ribbon.tabText(index)
        hints = {
            "项目": "项目：新建或打开模型；建模过程可随时撤销。",
            "建模": "建模：创建几何后，用“选择节点/杆件”进入对象选择。",
            "属性": "属性：选择杆件，创建或指派材料与截面。",
            "载荷": "载荷：选择节点创建边界条件，或选择杆件/节点施加载荷。",
            "分析": "分析：先执行模型检查，再求解或进行模态、屈曲分析。",
            "结果": "结果：查看变形与内力，并做强度验算、对称性与存储方案校核。",
            "视图": "视图：切换标准视角、选择视口背景、开关网格地面与编号标注。",
        }
        self.set_prompt(hints.get(page, "就绪 | 选择上方功能区模块开始建模"))

    def _show_ribbon_page(self, name: str) -> None:
        page = self.ribbon.pages.get(name)
        if page is not None:
            self.ribbon.setCurrentWidget(page)

    def _on_workflow_stage(self, stage: str) -> None:
        """Translate a workflow intent into an existing, backend-valid action."""
        from workflow import WorkflowPhase, inspect_workflow

        status = inspect_workflow(self.session)
        if stage == "model":
            self._show_ribbon_page("建模")
            return
        if stage == "define":
            _active_stage, page, hint = draft_target(status)
            self._show_ribbon_page(page)
            self.set_prompt(hint)
            return
        if stage == "validate":
            checked = self.session.validate_model()
            if checked.ok:
                self.statusBar().showMessage("模型完整性校验通过，可以开始求解。", 5000)
            else:
                self._show_validation_errors(checked.payload)
            self.workflow_bar.update_state(self.session)
            return
        if stage == "solve":
            if status.phase is WorkflowPhase.READY:
                self._show_ribbon_page("分析")
                self.solve()
            else:
                self._on_workflow_stage("model" if status.phase is WorkflowPhase.EMPTY
                                        else "define")
                self.set_prompt("模型尚未通过校验，请先完成当前阻断项")
            return
        if stage == "results":
            if status.phase is WorkflowPhase.SOLVED:
                self._show_ribbon_page("结果")
                self.show_deformed()
            else:
                self._on_workflow_stage("solve")

    def _build_menus(self) -> None:
        """Build one compact main-menu button from the shared QActions.

        The full menu hierarchy remains intact for documentation and keyboard
        use, but no longer consumes a dedicated row above the ribbon.
        """
        from . import commands
        from PySide6.QtWidgets import QToolButton

        menus = (("文件", ("new", "open", "save", None, "report", "learning_trace")),
                 ("建模", ("sketch", "frame", "portal", None,
                           "model_node", "model_member", "delete", "brace", None,
                           "tables", "json")),
                 ("属性", ("material", "beam_section", "section", None,
                           "assign_section", "units")),
                 ("载荷", ("create_bc", "support", None,
                           "create_load", "gravity", None,
                           "load", "combo")),
                 ("分析", ("solve", None, "modal", "buckling", "solid_joint", None,
                           "diagnose", "analysis_mesh")),
                 ("结果", ("model", "deformed", "contour", "diagram", None,
                           "envelope", "clear_results", "labels")),
                 ("视图", ("iso", "front", "side", "top", "fit", None,
                           "bg_settings", "grid_floor", "labels", "camera", None,
                           "chat", "props")))
        root = QMenu(self)
        root.setTitle("主菜单")
        for title, names in menus:
            menu = root.addMenu(title)
            for name in names:
                if name is None:
                    menu.addSeparator()
                else:
                    menu.addAction(self.actions_by_name[name])
                if title == "文件" and name == "open":
                    # 成熟 CAE 工具都会保留近期模型：反复打开同一工程是日常，
                    # 再走一遍文件选择器只会打断建模节奏。
                    self.recent_menu = menu.addMenu("最近打开")
                    self.recent_menu.aboutToShow.connect(self._populate_recent_menu)
        self.main_menu = root
        self.menu_button = QToolButton(self.ribbon)
        self.menu_button.setObjectName("mainMenuButton")
        self.menu_button.setText("☰  菜单")
        self.menu_button.setToolTip("文件、建模、分析、结果与视图菜单")
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_button.setMenu(root)
        self.ribbon.setCornerWidget(
            self.menu_button, Qt.Corner.TopLeftCorner)
        self.menuBar().hide()

    def _build_statusbar(self) -> None:
        self.setStatusBar(QStatusBar(self))
        self.statusBar().setSizeGripEnabled(False)

        # Abaqus 式提示区：显示当前操作需要做什么（纯文字，不加背景条）
        self.lbl_prompt = QLabel("  就绪 | 选择上方功能区模块开始建模")
        self.lbl_prompt.setMinimumWidth(450)
        self.lbl_prompt.setStyleSheet(
            f"color:{theme.ACCENT}; font-weight:600; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_prompt, 1)

        # 分隔线
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet(f"color: {theme.BORDER};")
        self.statusBar().addPermanentWidget(sep1)

        # 状态信息 - 模型状态
        self.lbl_model = QLabel("模型：空")
        self.lbl_model.setStyleSheet(f"color: {theme.INK_MUTED}; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_model)

        # 状态信息 - 求解状态
        self.lbl_solve = QLabel("求解：未求解")
        self.lbl_solve.setStyleSheet(f"color: {theme.INK_MUTED}; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_solve)

        # 状态信息 - 拾取状态
        self.lbl_pick = QLabel("")
        self.lbl_pick.setStyleSheet(f"color: {theme.WARN}; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_pick)

        # 分隔线
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"color: {theme.BORDER};")
        self.statusBar().addPermanentWidget(sep2)

        # 最近操作
        self.lbl_last = QLabel("最近：尚未建模")
        self.lbl_last.setToolTip("最近一次成功或失败的建模操作")
        self.lbl_last.setStyleSheet(f"color: {theme.INK_DIM}; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_last)

        self._on_ribbon_page_changed(self.ribbon.currentIndex())

    def set_prompt(self, text: str) -> None:
        """设置提示区文字（Abaqus 式操作提示）。"""
        self.lbl_prompt.setText(f"  {text}")

    # --- 状态 ---

    def set_mode(self, name: str) -> None:
        self.mode = name
        if name in {"变形", "云图"} and self.session.solution is not None:
            self.viewport.set_pick_mode("member")
        act = self.mode_actions.get(name)
        if act is not None:
            act.setChecked(True)
        self.redraw()

    def new_model(self) -> None:
        self._replace_session(Session())
        self.result = None
        self.case = None
        self.refresh()

    def _replace_session(self, session: Session) -> None:
        """让所有会修改模型的面板指向同一个新会话。"""
        self.session = session
        self._selected_kind = self._selected_id = None
        self.viewport.set_selection(None, None)
        self.viewport.set_problem_refs([])
        self._last_probe = None
        self.chat.session = session
        self.chat.conversation = None
        self.properties.session = session
        self.properties.clear()
        self.bc.session = session
        self.sketch.session = session
        self.section_opt.session = session
        self.bc.set_selection(None, None)
        self.lbl_pick.setText("")

    def solve(self) -> None:
        """求解。**在后台线程里跑。**

        单跨刚架是毫秒级，但网架、扫参、模态一上来就是秒级，
        阻塞主线程的话窗口会变成"未响应"——用户看到的是程序死了，
        而不是程序在算。
        """
        if not self.session.model.get("nodes"):
            QMessageBox.information(self, "无可求解的模型", "当前没有模型，请先建立模型后再求解。")
            return
        if self.runner.busy:
            return
        self._on_busy(True)
        self.statusBar().showMessage("求解中…")
        ok = self.runner.submit(
                                lambda: self.session.solve_model(
                                    analysis=self.analysis_type,
                                    **self.analysis_options),
                                on_done=self._solved, on_failed=self._solve_failed)
        if not ok:
            self._on_busy(False)

    def _solved(self, result) -> None:
        self.result = result
        self._on_busy(False)
        self.statusBar().clearMessage()
        if result.ok:
            cases = list(result.payload["cases"])
            self.case = self.case if self.case in cases else cases[0]
            self.set_mode("变形")
            # 静默失败检测摘要 + 实验胶囊路径（solve_model 自动跑的两项质检）
            sf = result.payload.get("silent_failures")
            if sf and sf.get("findings"):
                bad = [f for f in sf["findings"] if f["status"] != "pass"]
                passed = len(sf["findings"]) - len(bad)
                if bad:
                    names = "、".join(f["name"] for f in bad[:3])
                    self.statusBar().showMessage(
                        f"静默检测 {passed}/{len(sf['findings'])} 通过，"
                        f"{len(bad)} 项异常：{names}{'…' if len(bad) > 3 else ''}。"
                        f"结果面板可查看详情。", 15000)
                else:
                    self.statusBar().showMessage(
                        f"静默检测全部 {len(sf['findings'])} 项通过", 8000)
            cap = result.payload.get("capsule")
            if cap:
                self.statusBar().showMessage(
                    self.statusBar().currentMessage() + f" | 胶囊：{cap}", 15000)
        else:
            self._show_validation_errors(result.payload)
        self.section_opt.refresh_sections()
        self.refresh()

    def _show_validation_errors(self, payload: dict) -> None:
        """把校验错误摆进结果面板，**每条尽量带上可定位的对象**。

        原来是弹一个消息框，里面写着"节点 17 缺少约束"——可用户在三维视图里
        根本找不到 17 号在哪。摆进表里、点一行就在视口高亮，这条信息才有用。
        """
        import re

        errors = payload.get("errors") or [payload.get("error") or "未给出原因"]
        rows, loc = [], []
        for e in errors:
            text = str(e)
            hit = re.search(r"节点\s*(\d+)", text)
            target = ("node", int(hit.group(1))) if hit else None
            if target is None:
                hit = re.search(r"杆件\s*(\d+)", text)
                target = ("member", int(hit.group(1))) if hit else None
            rows.append([target[1] if target else "", text])
            loc.append(target)
        self.results_dock.setVisible(True)
        self.results.show_rows(
            f"模型校验未通过，共 {len(rows)} 条。点击一行可在视口中定位。",
            ["编号", "问题"], rows, loc)
        hint = payload.get("hint")
        if hint:
            self.statusBar().showMessage(str(hint)[:200], 10000)

    def clear_results(self) -> None:
        """丢弃求解结果，但不改任何建模数据。"""
        if self.result is None and self.session.solution is None:
            self.statusBar().showMessage("当前没有需要清除的计算结果。", 4000)
            return
        self.result = None
        self.case = None
        self.session.solution = None
        self.results.show_message("计算结果已清除", "模型、材料、约束与荷载均未修改，可重新求解。")
        self.set_mode("模型")
        self.refresh()
        self.statusBar().showMessage("计算结果已清除，模型保持不变。", 5000)

    def _solve_failed(self, kind: str, message: str) -> None:
        """求解器自己抛异常。守门拦不住的（比如奇异矩阵）会走到这里。"""
        self._on_busy(False)
        self.statusBar().clearMessage()
        QMessageBox.critical(self, "求解过程出错", f"{kind}: {message[:600]}")

    # --- Agent 面板 ---

    def _on_busy(self, busy: bool) -> None:
        """跑着活儿的时候把会改模型的动作置灰。

        不置灰的话，用户在求解途中点「新建」，后台线程正在读的那个
        Session 就被换掉了——**这类竞态出来的错误现场根本没法复现**。
        """
        idle = not busy
        self.act_new.setEnabled(idle)
        self.act_generate.setEnabled(idle)
        self.act_solve.setEnabled(idle and bool(self.session.model.get("nodes")))
        self.workflow_bar.setEnabled(idle)

    def _on_chat_changed(self, touched_model: bool) -> None:
        """对话改过模型就重画。

        对话和工具栏改的是**同一个 Session**，所以两边永远看着同一份模型，
        不会各改各的——这是当初把 Session 传进面板而不是新建一个的原因。
        """
        if not touched_model:
            return
        self._adopt_session_solution()
        self.refresh()

    def _on_sketch_loaded(self, model: dict) -> None:
        """手绘草图识别成功并加载为当前模型后，刷新界面。"""
        self.result = None
        self.case = None
        self.section_opt.refresh_sections()
        self.refresh()

    def _adopt_session_solution(self) -> None:
        """把 Session 里的求解结果认领到界面上。

        Agent 在对话里调 `solve_model`，结果只落在 Session 上；界面自己那份
        `result` 不跟上的话，就会出现**树里没有"结果"分支、状态栏还写着
        "未求解"，可视口已经画得出变形**——三处状态各说各的，
        比什么都没更新更让人困惑。
        """
        from agent import ToolResult

        if not self.session.solution:
            return
        cases = list(self.session.solution.all_results())
        if not cases:
            return
        self.result = ToolResult(True, {"cases": cases})
        if self.case not in cases:
            self.case = cases[0]
        if self.mode == "模型":
            self.set_mode("变形")

    def _demo_prompt(self) -> None:
        self.chat_dock.setVisible(True)
        self.chat.load_demo_prompt()

    # --- 绘制 ---

    def redraw(self) -> None:
        frame = None
        draft = False
        try:
            frame = self.session.preview_frame()
        except ValueError:
            # 模型还不完整（人工建模最初只点了几个节点，材料/截面/杆件还没齐），
            # 正式装配会报“不合法”。退到宽松草稿帧：有节点就画出来、点得着，
            # 否则用户刚建的节点在视口里不可见，也就无法选中它们去连杆件。
            frame = scene.draft_frame(self.session.model)
            draft = True
            if not frame.nodes:
                self.viewport.clear()
                return
        if not draft and not frame.members:
            self.viewport.clear()
            return

        if self.mode == "分析网格":
            preview = self.session.preview_analysis_mesh()
            if preview.ok:
                self.viewport.show_analysis_mesh(frame, preview.payload)
                return
            self.mode = "模型"
            self.mode_actions["模型"].setChecked(True)

        solved = self.result is not None and self.result.ok
        if self.mode == "模型" or not solved:
            self.viewport.show_model(frame, self.case)
            return
        frame = self.session.frame or frame
        if self.mode == "变形":
            scale = self.scale or self._auto_scale(frame)
            self.viewport.show_deformed(frame, self.session.solution,
                                        self.case, scale)
        elif self.mode == "云图":
            unit = "kN·m" if self.component in ("T", "My", "Mz", "M") else "kN"
            options = self.result_display_options
            self.viewport.show_contour(frame, self.session.solution, self.case,
                                       self.component,
                                       title=f"{self.component} ({unit})",
                                       percentile=options["percentile"],
                                       sign_filter=options["sign"],
                                       overlay_deformed=options["overlay_deformed"],
                                       show_extrema=options["show_extrema"])
        elif self.mode == "模态":
            got = self.session.modal_analysis(num_modes=6)
            if not got.ok:
                QMessageBox.information(self, "无法进行模态分析",
                                        str(got.payload.get("error", ""))[:400])
                self.set_mode("模型")
                return
            from modal import modal
            r = modal(frame, 6)
            f1 = r.frequencies[0]
            self.viewport.show_mode(frame, r.shapes, 0,
                                    f"第 1 阶　{f1:.3f} Hz　周期 {1 / f1:.4f} s")

    def _auto_scale(self, frame) -> float:
        return scene.auto_deformation_scale(frame, self.session.solution,
                                            self.case)

    def refresh(self) -> None:
        self.tree.rebuild(self.session, self.result)
        model = self.session.model
        if model.get("nodes"):
            self.lbl_model.setText(
                f"模型：{len(model['nodes'])} 节点 / {len(model.get('members') or [])} 杆件"
                f"　{model.get('units', 'N-m-Pa')}")
        else:
            self.lbl_model.setText("模型：空")
        self.lbl_solve.setText(self._solve_text())
        cases = (list(self.result.payload["cases"])
                 if self.result is not None and self.result.ok else [])
        self.quickbar.set_cases(cases, self.case)
        if self.diagram_dock.isVisible():
            self.diagram.attach(self.session, self.case)
        if self.props_dock.isVisible():
            self.properties.session = self.session   # 新建模型换过 Session
            self.properties.refresh()
        if self.bc_dock.isVisible():
            self.bc.session = self.session
            self.bc.refresh()
        if len(self.session.history):
            last = self.session.history[len(self.session.history) - 1]
            self.lbl_last.setText(f"最近：第 {last.index + 1} 步　{last.summary}")
        else:
            self.lbl_last.setText("最近：尚未建模")
        self.act_solve.setEnabled(bool(model.get("nodes")))
        self.workflow_bar.update_state(self.session)
        self.redraw()
        has_model = bool(model.get("nodes"))
        self._sync_model_tree(has_model)
        self.empty_state.setVisible(not has_model)
        if self.empty_state.isVisible():
            self._position_empty_state()

    def _sync_model_tree(self, has_model: bool) -> None:
        """Give an empty viewport its width back, then reveal real model data."""
        if not has_model:
            if not self.tree_dock.isHidden():
                self.tree_dock.hide()
            self._tree_auto_hidden = True
        elif self._tree_auto_hidden:
            self.tree_dock.show()
            self.tree_dock.raise_()
            self._tree_auto_hidden = False




    # ------------------------------------------------- 手动建模入口
    #
    # 这三条是从网页版**搬回来**的。迁移到桌面端时只搬了自然语言一条，
    # 另外两条落在了 gui_app.py 里，桌面端因此比它要替代的东西功能更少。
    #
    # 自然语言擅长"从无到有"和大范围改动；手动编辑擅长"改一个数"。
    # 少了后者，这个软件在改一个截面名这件事上比 Excel 还笨。

    def open_parametric(self, cmd=None) -> None:
        """参数化建模对话框。

        点「门式刚架」进来时直接切到门式那一页——按钮上写着什么，
        打开就该是什么，否则用户得自己再选一次。
        """
        from .dialogs import ParametricDialog

        dialog = ParametricDialog(self.session, self)
        if cmd is not None and getattr(cmd, "name", "") == "portal":
            dialog.kind.setCurrentIndex(1)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        params = dict(dialog.params)
        if dialog.result_kind == "frame":
            result = self.session.generate_frame(**params)
        elif dialog.result_kind == "portal":
            result = self.session.generate_portal_frame(**params)
        else:
            bays = params.pop("bays", [])
            result = self.session.generate_bent(**params)
            if result.ok and bays:
                result = self.session.extrude_bents(
                    bays=bays, tie_section=params.get("beam_section") or None)
        if not result.ok:
            QMessageBox.warning(self, "建模未成功", self._explain(result.payload))
            return
        self._after_manual_edit(
            f"已生成模型：{result.payload.get('summary', {}).get('nodes', '?')} 节点")

    def open_tables(self) -> None:
        """七张表编辑。"""
        from .dialogs import TableDialog

        if not self.session.model:
            QMessageBox.information(self, "尚无模型",
                                    "请先建立模型，再进入表格编辑。")
            return
        dialog = TableDialog(self.session, self)
        if dialog.exec() != dialog.DialogCode.Accepted or dialog.model is None:
            return
        result = self.session.set_model(model=dialog.model)
        if not result.ok:
            QMessageBox.warning(self, "表格模型未写回", self._explain(result.payload))
            return
        self._after_manual_edit("模型表格已写回")

    def open_json(self) -> None:
        from .dialogs import JsonDialog

        dialog = JsonDialog(self.session, self)
        if dialog.exec() != dialog.DialogCode.Accepted or dialog.model is None:
            return
        result = self.session.set_model(model=dialog.model)
        if not result.ok:
            QMessageBox.warning(self, "JSON 模型未写回", self._explain(result.payload))
            return
        self._after_manual_edit("模型 JSON 已写回")

    def _on_property_edited(self) -> None:
        """属性面板改完了。走和其它手工改动一样的收尾——**同一条路**，
        不再另写一遍清理，那正是当初三处清理逻辑各不相同的由来。"""
        self._after_manual_edit("属性已修改")
        self.properties.refresh()

    def _after_manual_edit(self, what: str) -> None:
        """手动改完模型的收尾。

        **和撤销走同一条路**：模型变了，原有结果就失效了。
        不清的话，界面会拿着旧模型的位移去画新模型的变形图。
        """
        self.result = None
        self.case = None
        self._last_probe = None
        self.results.show_message(
            what, "模型已变更，原有分析结果已失效并清除，请重新求解。")
        self.set_mode("模型")
        self.refresh()
        self.statusBar().showMessage(what, 5000)

    def _on_result_display_changed(self, options: dict) -> None:
        """结果显示选项只改变视图，不触碰求解数据。"""
        self.result_display_options = dict(options)
        if self.mode == "云图" and self.session.solution is not None:
            self.redraw()

    def probe_member_result(self, member_id: int, point) -> None:
        """将视口点击位置变成可审查的杆件截面结果表。"""
        if self.session.solution is None or self.session.frame is None:
            return
        from .result_inspector import probe_member

        data = probe_member(self.session.frame, self.session.solution,
                            member_id, point, self.case)
        self._last_probe = data
        rows = []
        for component in ("N", "Vy", "Vz", "T", "My", "Mz"):
            unit = (data["moment_unit"] if component in {"T", "My", "Mz"}
                    else data["force_unit"])
            rows.append([component, data["forces"][component], unit,
                         "杆件局部分量"])
        for prefix, values in (("U(global)", data["global_displacement"]),
                               ("U(local)", data["local_displacement"])):
            for axis, value in zip("xyz", values):
                rows.append([f"{prefix}.{axis}", float(value),
                             data["displacement_unit"], "中心线位移"])
        if data["stress"] is not None:
            rows += [["sigma_min", data["stress"]["min"], "MPa", "受压端"],
                     ["sigma_max", data["stress"]["max"], "MPa", "受拉端"]]
        else:
            rows.append(["截面正应力", "不可用", "", data["stress_error"]])
        self.results_dock.setVisible(True)
        self.results_dock.raise_()
        self.results.show_rows(
            f"结果探针：杆件 {member_id}，工况 {data['case']}，"
            f"距 i 端 x={data['x']:.4g}/{data['length']:.4g}",
            ["结果量", "值", "单位", "约定"], rows,
            [("member", member_id)] * len(rows))
        self.viewport.set_result_marker(
            data["point"], f"PROBE M{member_id} x={data['x']:.3g}")

    def locate_current_extreme(self) -> None:
        """定位当前梁内力分量的全结构绝对极值。"""
        if self.session.solution is None or self.session.frame is None:
            QMessageBox.information(self, "尚无分析结果", "请先求解。")
            return
        from .result_inspector import global_extreme

        extreme = global_extreme(self.session.frame, self.session.solution,
                                 self.component, self.case)
        if extreme["member"] is None:
            return
        self.locate("member", extreme["member"])
        self.viewport.set_result_marker(
            extreme["point"],
            f"MAX |{self.component}|={extreme['value']:+.3g} {extreme['unit']}")
        self.results.show_rows(
            f"{self.component} 全结构绝对极值，工况 {extreme['case']}",
            ["杆件", "距 i 端 x", "值", "单位"],
            [[extreme["member"], extreme["x"], extreme["value"],
              extreme["unit"]]], [("member", extreme["member"])])

    def show_section_stress(self) -> None:
        """打开当前探针截面的轴力+双向弯曲正应力图。"""
        if self.session.solution is None or self.session.frame is None:
            QMessageBox.information(self, "尚无分析结果", "请先求解。")
            return
        data = self._last_probe
        selected = getattr(self, "_selected_id", None)
        if data is None or (selected is not None and data["member"] != selected):
            if getattr(self, "_selected_kind", None) != "member":
                QMessageBox.information(
                    self, "请选择杆件", "先在视口中选择或探测一根杆件。")
                return
            from .result_inspector import probe_member
            member = self.session.frame.members[selected]
            midpoint = 0.5 * (self.session.frame.nodes[member.i].xyz
                              + self.session.frame.nodes[member.j].xyz)
            data = probe_member(self.session.frame, self.session.solution,
                                selected, midpoint, self.case)
            self._last_probe = data
        if data["stress"] is None:
            QMessageBox.information(self, "截面正应力不可用", data["stress_error"])
            return
        from .result_inspector import show_stress_dialog
        show_stress_dialog(self, data)

    def run_diagnose(self) -> None:
        """约束诊断。

        刚度矩阵奇异时，光说"矩阵奇异"没用——要指出**是哪几个节点的
        哪几个方向**构成了机构。这一条也是从网页版搬回来的。
        """
        if not self.session.model.get("nodes"):
            QMessageBox.information(self, "尚无模型", "请先建立模型。")
            return
        result = self.session.diagnose_supports()
        self.results_dock.setVisible(True)
        if not result.ok:
            self.results.show_message("约束诊断未完成",
                                      self._explain(result.payload))
            return
        modes = result.payload.get("modes") or []
        if not modes:
            self.results.show_message(
                "约束诊断",
                "未检出刚体模态：约束足以消除全部刚体位移。"
                "若求解仍失败，请检查是否存在零刚度杆件或重复节点。")
            return
        rows, loc = [], []
        for k, mode in enumerate(modes, 1):
            for part in mode.get("participants", [])[:6]:
                rows.append([k, part.get("node"), part.get("direction"),
                             round(float(part.get("amplitude", 0.0)), 4)])
                node = part.get("node")
                loc.append(("node", node) if isinstance(node, int) else None)
        self.results.show_rows(
            f"检出 {len(modes)} 个刚体模态：以下节点方向缺少足够约束。"
            "点击一行可在视口中定位。",
            ["模态", "节点", "方向", "参与幅值"], rows, loc)

    # ------------------------------------------------- 撤销与时间线

    def undo(self) -> None:
        if not self.session.undo():
            self.statusBar().showMessage("已到达建模过程起点，无可撤销的操作", 4000)
            return
        self._after_history_move("已撤销")

    def redo(self) -> None:
        if not self.session.redo():
            self.statusBar().showMessage("已到达建模过程末端，无可重做的操作", 4000)
            return
        self._after_history_move("已重做")

    def goto_step(self, cursor: int) -> None:
        if self.session.goto_step(cursor):
            self._after_history_move(f"已跳到第 {cursor + 1} 步")

    def _after_history_move(self, what: str) -> None:
        """撤销/重做/跳步之后的收尾。

        **必须把界面上的结果一起清掉。** `Session._restore` 已经清了求解结果，
        界面这份 `result` 不跟着清的话，模型退回去了而结果面板还显示着
        旧模型的内力——数字都在，就是对不上，而且看不出来。
        """
        self.result = None
        self.case = None
        self.results.show_message(
            what, "模型状态已变更，原有分析结果已失效并清除，请重新求解。")
        self.set_mode("模型")
        self.refresh()
        self.statusBar().showMessage(
            f"{what}（第 {self.session.history.cursor + 1} 步）", 5000)

    def set_case(self, name: str) -> None:
        """切换显示哪个工况/组合的结果。审结果时这是切得最勤的一个动作。"""
        if name and name != self.case:
            self.case = name
            self.redraw()
            self.lbl_solve.setText(self._solve_text())

    def set_scale(self, value: float) -> None:
        """变形放大系数。0 表示按模型尺寸自动定。"""
        self.scale = float(value)
        if self.mode == "变形":
            self.redraw()

    def show_diagram(self) -> None:
        """打开单杆内力图。

        三维云图看的是全局分布，这里看的是**沿杆长的那条曲线**：
        在哪儿过零、峰值在什么位置、不同组合的包络有多宽。
        判断"这根梁跨中够不够"靠的是后者。
        """
        on = self.actions_by_name["curve"].isChecked()
        self.diagram_dock.setVisible(on)
        if on:
            self.diagram_dock.raise_()
            self.diagram.attach(self.session, self.case)

    def show_timeline(self) -> None:
        on = self.actions_by_name["timeline"].isChecked()
        self.timeline_dock.setVisible(on)
        if on:
            self.timeline.rebuild(self.session.history)

    def export_learning_trace(self) -> None:
        """把人和 Agent 的确定性建模步骤导出成可筛选的学习样本。"""
        result = self.session.export_learning_trace(label="desktop")
        if not result.ok:
            QMessageBox.warning(self, "学习轨迹导出失败", self._explain(result.payload))
            return
        path = result.payload["path"]
        verified = bool(result.payload["verified"])
        state = "已通过完整模型校验，可作为正样本" if verified else \
            "仍是草稿/错误过程，仅用于学习修复，不应作为正确答案"
        self.statusBar().showMessage(f"学习轨迹已保存：{Path(path).name}", 7000)
        QMessageBox.information(
            self, "学习轨迹已导出",
            f"已记录 {result.payload['steps']} 个建模步骤。\n{state}\n\n{path}")

    # ------------------------------------------------- 拾取与标注

    def set_pick_node(self) -> None:
        self._apply_pick_mode()

    def set_pick_member(self) -> None:
        self._apply_pick_mode()

    def _apply_pick_mode(self) -> None:
        mode = next((k for k, a in self.pick_actions.items() if a.isChecked()),
                    None)
        self.viewport.set_pick_mode(mode)
        if mode == "node":
            self.set_prompt("选择节点模式：在视口中点击节点选中，可用于创建边界条件/载荷/查看属性")
        elif mode == "member":
            self.set_prompt("选择杆件模式：在视口中点击杆件选中，可用于截面指派/施加荷载/查看属性")
        else:
            self.set_prompt("就绪 | 选择上方功能区模块开始建模")

    def toggle_labels(self) -> None:
        on = self.actions_by_name["labels"].isChecked()
        self.viewport.set_labels(on)

    # --- 人工建模 ---

    def set_model_node(self) -> None:
        """进入建节点模式：在视口中点击任意位置创建节点。"""
        self._apply_model_mode()

    def set_model_member(self) -> None:
        """进入建杆件模式：先点一个节点，再点另一个节点创建杆件。"""
        self._apply_model_mode()

    def _apply_model_mode(self) -> None:
        mode = next((k for k, a in {
            "node": self.actions_by_name["model_node"],
            "member": self.actions_by_name["model_member"],
        }.items() if a.isChecked()), None)
        # 建模模式和拾取模式互斥
        if mode:
            self.pick_actions["node"].setChecked(False)
            self.pick_actions["member"].setChecked(False)
        self.viewport.set_model_mode(mode)
        if mode == "node":
            self.set_prompt("建节点模式：在视口中点击任意位置创建节点 | 再次点击「建节点」退出")
        elif mode == "member":
            self.set_prompt("建杆件模式：先点击第一个节点，再点击第二个节点创建杆件 | 再次点击「建杆件」退出")
        else:
            self.set_prompt("就绪 | 选择上方功能区模块开始建模")

    def _on_node_snapped(self, nid: int) -> None:
        """建节点时点在了已有节点附近：吸附过去，不重复建节点。"""
        self._selected_kind = "node"
        self._selected_id = nid
        self._describe_selection("node", nid)
        self.statusBar().showMessage(
            f"已吸附到已有节点 {nid}，未重复创建（鼠标靠近已有节点会自动吸附）", 4000)

    def cancel_interaction(self) -> None:
        """Esc：退出当前建模/拾取模式，回到浏览状态。"""
        changed = False
        for name in ("model_node", "model_member", "pick_node", "pick_member"):
            act = self.actions_by_name.get(name)
            if act is not None and act.isChecked():
                act.setChecked(False)
                changed = True
        # 由这两个统一同步视口模式与提示
        self._apply_model_mode()
        self._apply_pick_mode()
        self.viewport.set_selection(None, None)
        if changed:
            self.statusBar().showMessage("已退出当前操作，回到浏览", 3000)

    # --- 精确建模：工作平面 / 网格捕捉 / 精确坐标 ---

    def set_work_plane(self, plane: str) -> None:
        if plane not in ("XY", "XZ", "YZ"):
            return
        self.viewport.set_work_plane(plane)
        ax = {"XY": "z", "XZ": "y", "YZ": "x"}[plane]
        self.statusBar().showMessage(
            f"工作平面设为 {plane}（新建节点固定 {ax}={self.viewport.work_offset:g}）", 3500)

    def set_work_offset(self, value) -> None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        self.viewport.set_work_offset(value)

    def set_snap_size(self, text: str) -> None:
        t = (text or "").strip()
        value = 0.0
        if t and t not in ("关闭", "off", "OFF", "0", "0.0"):
            try:
                value = float(t)
            except ValueError:
                return
        self.viewport.set_snap_size(value)
        if value > 0:
            self.statusBar().showMessage(f"网格捕捉开启：间距 {value:g}", 3000)

    def create_node_exact(self) -> None:
        """输入精确坐标建节点（Abaqus 式坐标输入，弥补鼠标点选不精确）。"""
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                                       QDoubleSpinBox)
        dlg = QDialog(self)
        unit = self.session.units.length
        dlg.setWindowTitle(f"按坐标创建节点（{unit}）")
        form = QFormLayout(dlg)
        spins = {}
        # 工作平面固定轴默认带出平面高度，其余两轴从 0 起
        fixed = {"XY": "z", "XZ": "y", "YZ": "x"}[self.viewport.work_plane]
        defaults = {"x": 0.0, "y": 0.0,
                    "z": self.viewport.work_offset if fixed == "z" else 0.0}
        if fixed == "y":
            defaults["y"] = self.viewport.work_offset
        if fixed == "x":
            defaults["x"] = self.viewport.work_offset
        for axis, label in (("x", "X 坐标"), ("y", "Y 坐标"), ("z", "Z 坐标")):
            sp = QDoubleSpinBox()
            sp.setRange(-1.0e6, 1.0e6)
            sp.setDecimals(6)
            sp.setSingleStep(0.5)
            sp.setSuffix(f" {unit}")
            sp.setValue(defaults[axis])
            if axis == fixed:
                sp.setToolTip(f"当前工作平面 {self.viewport.work_plane} 固定该坐标")
            spins[axis] = sp
            form.addRow(f"{label}：", sp)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        xyz = [float(spins[a].value()) for a in ("x", "y", "z")]
        self._create_node_at(xyz)

    def _create_node_at(self, xyz) -> None:
        """统一处理鼠标与精确输入建点，包括已有坐标复用。"""
        values = [float(value) for value in xyz]
        result = self.session.add_nodes([values])
        if not result.ok:
            self.statusBar().showMessage(
                f"创建节点失败：{self._explain(result.payload)}", 5000)
            return
        node_id = int(result.payload["input_node_ids"][0])
        if result.payload.get("no_change"):
            self._selected_kind, self._selected_id = "node", node_id
            self.viewport.set_selection("node", node_id)
            self._describe_selection("node", node_id)
            self.statusBar().showMessage(
                f"该坐标已有节点 {node_id}，已选中且未重复创建。", 4000)
            return
        self._after_manual_edit(
            f"已创建节点 {node_id}："
            f"({values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f})")

    def _on_node_created(self, x: float, y: float, z: float) -> None:
        """视口中点击创建了节点。"""
        self._create_node_at([x, y, z])

    def _on_member_created(self, ni: int, nj: int) -> None:
        """视口中点击两节点创建了杆件。"""
        from PySide6.QtWidgets import QInputDialog

        sections = [s["name"] for s in self.session.model.get("sections", [])]
        materials = [m["name"] for m in self.session.model.get("materials", [])]
        section = getattr(self, "_active_section", None)
        if sections and section not in sections:
            section, ok = QInputDialog.getItem(
                self, "创建杆件", "选择截面：", sections, 0, False)
            if not ok:
                return
            self._active_section = section
        elif not sections:
            section = ""
        material = getattr(self, "_active_material", None)
        if materials and material not in materials:
            material, ok = QInputDialog.getItem(
                self, "创建杆件", "选择材料：", materials, 0, False)
            if not ok:
                return
            self._active_material = material
        elif not materials:
            material = ""
        result = self.session.add_members([[ni, nj]], section, material)
        if result.ok:
            mid = result.payload["added_member_ids"][-1]
            assigned = section or "未指派截面"
            self._after_manual_edit(f"已创建杆件 {mid}：节点 {ni} → {nj}（{assigned}）")
        else:
            self.statusBar().showMessage(f"创建杆件失败：{self._explain(result.payload)}", 5000)

    def delete_selected(self) -> None:
        """删除当前选中的节点或杆件。"""
        kind = getattr(self, "_selected_kind", None)
        ident = getattr(self, "_selected_id", None)
        if kind is None or ident is None:
            self.statusBar().showMessage("没有选中对象。先用「选择节点」或「选择杆件」选中要删除的对象。", 5000)
            return
        if kind == "node":
            result = self.session.remove_nodes([ident])
        else:
            result = self.session.remove_members([ident])
        if result.ok:
            self._selected_kind = None
            self._selected_id = None
            self._sync_selection_actions()
            self.viewport.set_selection(None, None)
            self.bc.set_selection(None, None)
            self.properties.clear()
            label = "节点" if kind == "node" else "杆件"
            self._after_manual_edit(f"已删除{label} {ident}")
        else:
            self.statusBar().showMessage(f"删除失败：{self._explain(result.payload)}", 5000)

    def locate(self, kind: str, ident: int) -> None:
        """在视口里定位一个对象。结果表和错误清单点行都走这里。

        **这是"这个数字在哪"的落点。** 之前位移只是个数字、报错只是个编号，
        用户在三维视图里找不到它们；有了这条路，报告里"每个数字可溯源"
        才在界面上真正兑现。
        """
        self.viewport.set_selection(kind, ident)
        self._on_picked(kind, ident)
        label = "节点" if kind == "node" else "杆件"
        self.statusBar().showMessage(f"已在视口定位{label} {ident}", 4000)

    def locate_problem(self, refs: list[tuple[str, int]]) -> None:
        """诊断问题可能同时涉及多个对象；保留整组红色问题标记。"""
        self.viewport.set_problem_refs(refs)
        if len(refs) == 1:
            self.locate(*refs[0])
        elif refs:
            self.statusBar().showMessage(
                f"已在视口标出 {len(refs)} 个相关对象", 4000)

    def _viewport_context_menu(self) -> QMenu:
        """按当前选择组装右键菜单，动作直接复用功能区 QAction。"""
        menu = QMenu(self)
        kind = getattr(self, "_selected_kind", None)
        ident = getattr(self, "_selected_id", None)
        if kind in {"node", "member"} and ident is not None:
            label = "节点" if kind == "node" else "杆件"
            title = menu.addAction(f"{label} {ident}")
            title.setEnabled(False)
            menu.addSeparator()
            names = (["create_bc", "create_load"] if kind == "node"
                     else ["assign_section", "hinge", "create_load"])
            for name in names:
                menu.addAction(self.actions_by_name[name])
            menu.addSeparator()
            menu.addAction(self.actions_by_name["delete"])
        else:
            menu.addAction(self.actions_by_name["pick_node"])
            menu.addAction(self.actions_by_name["pick_member"])
            menu.addSeparator()
            menu.addAction(self.actions_by_name["fit"])
            menu.addAction(self.actions_by_name["labels"])
        return menu

    def _show_viewport_context_menu(self, global_pos) -> None:
        self._viewport_context_menu().exec(global_pos)

    def _on_picked(self, kind: str, ident: int) -> None:
        self._selected_kind = kind
        self._selected_id = ident
        if self.viewport.selection != (kind, ident):
            self.viewport.set_selection(kind, ident)
        self._sync_selection_actions()
        self._describe_selection(kind, ident)
        # **选中就显示属性。** 这一步是把"拾取"这条路走完：
        # 之前点中一根杆只能看到它多长，改不了
        self.properties.show_object(kind, ident)
        self.props_dock.show()
        self.props_dock.raise_()
        # 边界条件面板也更新
        self.bc.set_selection(kind, ident)
        # 点中一根杆，内力图就切到它——**这是"在哪"和"多大"接起来的一步**
        if kind == "member" and self.diagram_dock.isVisible():
            self.diagram.member.setCurrentText(str(ident))

    def _describe_selection(self, kind: str, ident: int) -> None:
        """选中之后在状态栏说清楚它是谁、在哪。

        只说"杆件 27"是没用的——**编号规则是生成器定的，用户并不知道**，
        所以必须同时给坐标。
        """
        try:
            frame = self.session.frame or self.session.preview_frame()
        except ValueError:
            return
        length_unit = "mm" if self.session.model.get("units") == "N-mm-MPa" else "m"
        if kind == "node" and ident in frame.nodes:
            x, y, z = frame.nodes[ident].xyz
            self.lbl_pick.setText(
                f"选中：节点 {ident}　({x:.3f}, {y:.3f}, {z:.3f}) {length_unit}")
        elif kind == "member" and ident in frame.members:
            m = frame.members[ident]
            a, b = frame.nodes[m.i].xyz, frame.nodes[m.j].xyz
            import numpy as np
            length = float(np.linalg.norm(b - a))
            self.lbl_pick.setText(
                f"选中：杆件 {ident}　节点 {m.i} → {m.j}　"
                f"长 {length:.3f} {length_unit}　截面 {m.section}")
        else:
            self.lbl_pick.setText("")

    # ------------------------------------------------- CAE 操作

    def generate_frame_dialog(self) -> None:
        """兼容旧调用；规则框架与门式刚架共用同一套已校验界面。"""
        from .commands import BY_NAME
        self.open_parametric(BY_NAME["frame"])

    def generate_portal_dialog(self) -> None:
        """兼容旧调用；进入统一参数化建模对话框的门式页。"""
        from .commands import BY_NAME
        self.open_parametric(BY_NAME["portal"])

    def apply_gravity(self) -> None:
        """直接按材料密度和截面面积计算自重荷载，施加到所有杆件上。"""
        if not self.session.model.get("members"):
            self.statusBar().showMessage("当前没有杆件，无法计算自重。", 5000)
            return
        result = self.session.add_self_weight()
        if not result.ok:
            self.statusBar().showMessage(
                f"施加自重失败：{self._explain(result.payload)}", 8000)
            return
        self._after_manual_edit(
            f"已将自重写入 {result.payload['case']} 工况（{result.payload['members']} 根杆件）")

    def show_bc_panel(self) -> None:
        """打开边界条件面板。"""
        self.bc_dock.setVisible(True)
        self.bc_dock.raise_()
        self.bc.refresh()

    def edit_materials_sections(self) -> None:
        """弹出材料与截面编辑对话框，直接修改模型的材料和截面定义。"""
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QTabWidget,
                                       QTableWidget, QTableWidgetItem, QVBoxLayout)
        dlg = QDialog(self)
        dlg.setWindowTitle("材料与截面")
        dlg.resize(600, 400)
        tabs = QTabWidget()
        unit_name = self.session.model.get("units", "N-m-Pa")
        if unit_name == "N-mm-MPa":
            modulus_unit, density_unit, length_unit = "MPa", "t/mm³", "mm"
        else:
            modulus_unit, density_unit, length_unit = "Pa", "kg/m³", "m"
        # 材料表
        mat_table = QTableWidget()
        mat_table.setColumnCount(4)
        mat_table.setHorizontalHeaderLabels(
            ["名称", f"E ({modulus_unit})", "nu", f"密度 ({density_unit})"])
        materials = self.session.model.get("materials", [])
        mat_table.setRowCount(len(materials))
        for i, m in enumerate(materials):
            mat_table.setItem(i, 0, QTableWidgetItem(m.get("name", "")))
            mat_table.setItem(i, 1, QTableWidgetItem(str(m.get("E", ""))))
            mat_table.setItem(i, 2, QTableWidgetItem(str(m.get("nu", ""))))
            mat_table.setItem(i, 3, QTableWidgetItem(str(m.get("density", ""))))
        tabs.addTab(mat_table, "材料")
        # 截面表
        sec_table = QTableWidget()
        sec_table.setColumnCount(5)
        sec_table.setHorizontalHeaderLabels(
            ["名称", f"A ({length_unit}²)", f"Iy ({length_unit}⁴)",
             f"Iz ({length_unit}⁴)", f"J ({length_unit}⁴)"])
        sections = self.session.model.get("sections", [])
        sec_table.setRowCount(len(sections))
        for i, s in enumerate(sections):
            sec_table.setItem(i, 0, QTableWidgetItem(s.get("name", "")))
            sec_table.setItem(i, 1, QTableWidgetItem(str(s.get("A", ""))))
            sec_table.setItem(i, 2, QTableWidgetItem(str(s.get("Iy", ""))))
            sec_table.setItem(i, 3, QTableWidgetItem(str(s.get("Iz", ""))))
            sec_table.setItem(i, 4, QTableWidgetItem(str(s.get("J", ""))))
        tabs.addTab(sec_table, "截面")
        layout = QVBoxLayout(dlg)
        layout.addWidget(tabs)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        def cell_text(table, row: int, column: int) -> str:
            item = table.item(row, column)
            return item.text().strip() if item is not None else ""

        # 读回材料
        new_materials = []
        new_sections = []
        try:
            for i in range(mat_table.rowCount()):
                name = cell_text(mat_table, i, 0)
                if name:
                    new_materials.append({
                        "name": name,
                        "E": float(cell_text(mat_table, i, 1)),
                        "nu": float(cell_text(mat_table, i, 2)),
                        "density": float(cell_text(mat_table, i, 3) or 0),
                    })
            # 读回截面
            for i in range(sec_table.rowCount()):
                name = cell_text(sec_table, i, 0)
                if name:
                    new_sections.append({
                        "name": name,
                        "A": float(cell_text(sec_table, i, 1)),
                        "Iy": float(cell_text(sec_table, i, 2)),
                        "Iz": float(cell_text(sec_table, i, 3)),
                        "J": float(cell_text(sec_table, i, 4)),
                    })
        except ValueError:
            QMessageBox.warning(
                self, "材料与截面未更新",
                "数值单元格不能为空，并且必须填写有效数字。改动尚未写入。")
            return
        # 和“创建材料/截面”共用同一条 Property 提交路径：空项目允许先
        # 定义属性，已有杆件时则校验全部引用，并且这一步可以 Ctrl+Z。
        result = self.session.define_materials_and_sections(
            new_materials, new_sections)
        if not result.ok:
            QMessageBox.warning(self, "材料与截面未更新", self._explain(result.payload))
            return
        self._after_manual_edit(
            f"已更新材料（{len(new_materials)}种）和截面（{len(new_sections)}种）")

    def add_brace(self) -> None:
        """在当前选中的杆件所在的节间加 X 型支撑。"""
        if getattr(self, "_selected_kind", None) != "member":
            self.statusBar().showMessage("请先选中一根梁或柱，然后在它所在的节间加支撑。", 5000)
            return
        mid = self._selected_id
        members = self.session.model.get("members", [])
        nodes = {int(n["id"]): n for n in self.session.model.get("nodes", [])}
        target = next((m for m in members if int(m["id"]) == mid), None)
        if target is None:
            return
        ni, nj = int(target["i"]), int(target["j"])
        if ni not in nodes or nj not in nodes:
            return
        # 找与这根杆平行且最近的另一根杆，构成节间
        import numpy as np
        v1 = np.array([nodes[nj]["x"]-nodes[ni]["x"], nodes[nj]["y"]-nodes[ni]["y"], nodes[nj]["z"]-nodes[ni]["z"]])
        best = None
        best_dist = float("inf")
        for m in members:
            if int(m["id"]) == mid:
                continue
            i2, j2 = int(m["i"]), int(m["j"])
            if i2 not in nodes or j2 not in nodes:
                continue
            v2 = np.array([nodes[j2]["x"]-nodes[i2]["x"], nodes[j2]["y"]-nodes[i2]["y"], nodes[j2]["z"]-nodes[i2]["z"]])
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            cos = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if cos > 0.9:  # 平行
                dist = np.linalg.norm(np.array([nodes[i2]["x"]-nodes[ni]["x"], nodes[i2]["y"]-nodes[ni]["y"], nodes[i2]["z"]-nodes[ni]["z"]]))
                if dist < best_dist:
                    best_dist = dist
                    best = (i2, j2)
        if best is None:
            self.statusBar().showMessage("找不到平行的杆件构成节间。", 5000)
            return
        i2, j2 = best
        section = target.get("section", "BEAM")
        material = target.get("material", "Q355")
        # 加两根斜杆构成 X
        result = self.session.add_members([[ni, j2], [nj, i2]], section, material)
        if not result.ok:
            self.statusBar().showMessage(f"添加支撑失败：{self._explain(result.payload)}", 5000)
            return
        self._after_manual_edit(f"已在杆件 {mid} 所在节间添加 X 型支撑")

    def edit_units(self) -> None:
        """弹出单位制选择对话框。"""
        from PySide6.QtWidgets import QDialog, QComboBox, QDialogButtonBox, QFormLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("单位制")
        form = QFormLayout(dlg)
        cmb = QComboBox()
        cmb.addItems(["N-m-Pa", "N-mm-MPa"])
        current = self.session.model.get("units", "N-m-Pa")
        idx = cmb.findText(current)
        if idx >= 0:
            cmb.setCurrentIndex(idx)
        form.addRow("单位制", cmb)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        result = self.session.set_units(cmb.currentText())
        if not result.ok:
            QMessageBox.warning(self, "单位制转换失败", self._explain(result.payload))
            return
        self._after_manual_edit(f"已转换单位制：{result.payload['was']} → {result.payload['units']}")

    def edit_load_combos(self) -> None:
        """弹出荷载组合编辑对话框。"""
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QTableWidget,
                                       QTableWidgetItem, QVBoxLayout, QLabel)
        dlg = QDialog(self)
        dlg.setWindowTitle("荷载组合")
        dlg.resize(500, 300)
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("格式：组合名 = 系数1×工况1 + 系数2×工况2 + ..."))
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["组合名", "表达式"])
        combos = self.session.model.get("combos", [])
        table.setRowCount(max(len(combos), 3))
        for i, c in enumerate(combos):
            table.setItem(i, 0, QTableWidgetItem(c.get("name", "")))
            table.setItem(i, 1, QTableWidgetItem(_format_combo_factors(c.get("factors") or {})))
        layout.addWidget(table)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        case_names = {str(case.get("name")) for case in self.session.model.get("load_cases", [])
                      if case.get("name")}
        if not case_names:
            QMessageBox.information(self, "尚无荷载工况", "请先创建至少一个荷载工况，再定义组合。")
            return
        new_combos = []
        for i in range(table.rowCount()):
            name = table.item(i, 0)
            expr = table.item(i, 1)
            if name and name.text().strip() and expr and expr.text().strip():
                try:
                    factors = _parse_combo_expression(expr.text(), case_names)
                except ValueError as exc:
                    QMessageBox.warning(self, "荷载组合格式错误", f"第 {i + 1} 行：{exc}")
                    return
                new_combos.append({"name": name.text().strip(), "factors": factors})
        result = self.session.set_load_cases(
            list(self.session.model.get("load_cases") or []), combos=new_combos)
        if not result.ok:
            QMessageBox.warning(self, "荷载组合校验失败", self._explain(result.payload))
            return
        self._after_manual_edit(f"已更新 {len(new_combos)} 个荷载组合")

    # ------------------------------------------------- Abaqus 式对话框

    def create_material(self) -> None:
        """Abaqus 式：创建/编辑材料。"""
        from .material_dialog import MaterialDialog
        unit_name = self.session.model.get("units", "N-m-Pa")
        dlg = MaterialDialog(parent=self, units=unit_name)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        mat = dlg.get_material()
        from copy import deepcopy

        new_materials = deepcopy(self.session.model.get("materials", []))
        # 检查是否已存在同名材料
        existing = next((m for m in new_materials
                         if m["name"] == mat["name"]), None)
        if existing:
            existing.update(mat)
        else:
            new_materials.append(mat)
        result = self.session.define_materials_and_sections(
            new_materials, list(self.session.model.get("sections", [])))
        if not result.ok:
            QMessageBox.warning(self, "材料未创建", self._explain(result.payload))
            return
        self._after_manual_edit(f"已创建材料：{mat['name']}（E={mat['E']:.3e}, ν={mat['nu']}）")

    def create_section(self) -> None:
        """Abaqus 式：创建/编辑梁截面。"""
        from .section_dialog import SectionDialog
        unit_name = self.session.model.get("units", "N-m-Pa")
        dlg = SectionDialog(parent=self, units=unit_name)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        raw_section = dlg.get_section()
        # ``type`` 和 ``params`` 是截面对话框的 UI 元数据；正式模型只认
        # name / A / Iy / Iz / J。把它们写进去会违反 JSON Schema，直到
        # 下一次求解才暴露出来，用户会误以为是求解器出了问题。
        section_type = raw_section["type"]
        sec = {key: raw_section[key] for key in
               ("name", "A", "Iy", "Iz", "J", "Ay", "Az")
               if key in raw_section}
        from copy import deepcopy

        new_sections = deepcopy(self.session.model.get("sections", []))
        existing = next((s for s in new_sections
                         if s["name"] == sec["name"]), None)
        if existing:
            existing.update(sec)
        else:
            new_sections.append(sec)
        result = self.session.define_materials_and_sections(
            list(self.session.model.get("materials", [])), new_sections)
        if not result.ok:
            QMessageBox.warning(self, "截面未创建", self._explain(result.payload))
            return
        length_unit = "mm" if unit_name == "N-mm-MPa" else "m"
        self._after_manual_edit(
            f"已创建截面：{sec['name']}（{section_type}, "
            f"A={sec['A']:.3e} {length_unit}²）")

    def assign_section(self) -> None:
        """Abaqus 式：选中杆件后一次指派材料与截面。"""
        if getattr(self, "_selected_kind", None) != "member":
            self.statusBar().showMessage("请先选中一根杆件，然后指派属性。", 5000)
            return
        sections = [s["name"] for s in self.session.model.get("sections", [])]
        materials = [m["name"] for m in self.session.model.get("materials", [])]
        if not sections or not materials:
            self.statusBar().showMessage(
                "属性指派需要材料和截面，请先在 Property 模块补齐定义。", 5000)
            return
        mid = self._selected_id
        member = next(m for m in self.session.model.get("members", [])
                      if int(m["id"]) == mid)
        from .assignment_dialog import AssignmentDialog
        dialog = AssignmentDialog(
            [mid], sections, materials, member.get("section", ""),
            member.get("material", ""), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        section, material = dialog.get_assignment()
        result = self.session.assign_properties([mid], section, material)
        if not result.ok:
            QMessageBox.warning(self, "属性指派失败", self._explain(result.payload))
            return
        self._after_manual_edit(
            f"杆件 {mid} 已指派属性：截面 {section}，材料 {material}")

    def create_hinge(self) -> None:
        """给选中杆件创建显式端铰；内核使用杆端自由度释放。"""
        if getattr(self, "_selected_kind", None) != "member":
            self.statusBar().showMessage("请先选择一根杆件，再创建铰接。", 5000)
            return
        from PySide6.QtWidgets import QInputDialog
        end_text, ok = QInputDialog.getItem(
            self, "创建铰接", "铰接位置：", ["i 端", "j 端", "两端"], 0, False)
        if not ok:
            return
        type_text, ok = QInputDialog.getItem(
            self, "创建铰接", "释放类型：",
            ["平面内铰（rz）", "双向弯曲铰（ry, rz）", "空间球铰（rx, ry, rz）"],
            0, False)
        if not ok:
            return
        release = (["rz"] if type_text.startswith("平面内") else
                   ["ry", "rz"] if type_text.startswith("双向") else
                   ["rx", "ry", "rz"])
        member_id = int(self._selected_id)
        entry = next(m for m in self.session.model.get("members", [])
                     if int(m["id"]) == member_id)
        existing = entry.get("releases") or {}
        kwargs = {}
        ends = (("i", "j") if end_text == "两端" else
                ("i",) if end_text.startswith("i") else ("j",))
        for end in ends:
            kwargs[f"releases_{end}"] = sorted(
                set(existing.get(end, ())) | set(release))
        result = self.session.edit_member(member_id, **kwargs)
        if not result.ok:
            QMessageBox.warning(self, "铰接未创建", self._explain(result.payload))
            return
        self._after_manual_edit(
            f"杆件 {member_id} 的 {end_text} 已创建{type_text}；橙色球为杆端释放")

    def edit_analysis_step(self) -> None:
        """Abaqus Step 风格：先选分析过程，再提交作业。"""
        from PySide6.QtWidgets import QInputDialog
        labels = ["线性静力", "P-Delta 二阶弹性", "双线性轴向材料非线性"]
        current = {"linear": 0, "pdelta": 1, "material": 2}.get(
            self.analysis_type, 0)
        choice, ok = QInputDialog.getItem(
            self, "编辑分析步", "分析过程：", labels, current, False)
        if not ok:
            return
        self.analysis_type = {labels[0]: "linear", labels[1]: "pdelta",
                              labels[2]: "material"}[choice]
        if self.analysis_type != "linear":
            increments, ok = QInputDialog.getInt(
                self, "编辑分析步", "载荷增量数：",
                self.analysis_options["increments"], 1, 1000, 1)
            if not ok:
                return
            self.analysis_options["increments"] = increments
        scope = {"linear": "小变形线弹性",
                 "pdelta": "小应变二阶弹性，迭代更新轴力",
                 "material": "双线性轴向塑性；弯曲保持弹性"}[self.analysis_type]
        self.statusBar().showMessage(f"分析步：{choice}（{scope}）", 8000)

    def create_bc(self) -> None:
        """Abaqus 式：选中节点，创建边界条件（勾选自由度）。"""
        if getattr(self, "_selected_kind", None) != "node":
            self.set_prompt("创建边界条件：请先在视口中选中一个节点，然后点击此按钮")
            self.statusBar().showMessage("请先选中一个节点，然后创建边界条件。", 5000)
            return
        from .bc_dialog import BCDialog
        nid = self._selected_id
        current_entry = next((s for s in self.session.model.get("supports", [])
                              if int(s["node"]) == nid), None)
        current = current_entry["fix"] if current_entry else None
        dlg = BCDialog(nid, current,
                       current_name=(current_entry or {}).get("name"), parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        fix = dlg.get_fix()
        bc_name = dlg.get_name()
        if any(fix) and not bc_name:
            QMessageBox.warning(self, "边界条件创建失败", "边界条件名称不能为空。")
            return
        result = self.session.set_supports(
            [nid], fix, name=bc_name or f"BC-Node-{nid}")
        if not result.ok:
            QMessageBox.warning(self, "边界条件创建失败", self._explain(result.payload))
            return
        self.set_prompt(f"已设置节点 {nid} 边界条件：{fix}")
        self._after_manual_edit(f"节点 {nid} 边界条件已更新：{fix}")

    # 这些命令必须先在视口里选中对象才有意义。
    # 键是命令名，值是它接受的选择类型。
    _NEEDS_SELECTION = {
        "create_load": ("node", "member"),
        "create_bc": ("node",),
        "solid_joint": ("node",),
    }

    def _sync_selection_actions(self) -> None:
        """按当前选择开关那些"必须先选中"的按钮。

        原来这些按钮永远可点，点了只在状态栏闪一句五秒后消失的提示。
        用户看到的是"点了没反应"——这正是"很多功能都是摆设"那类抱怨的来源。
        **前置条件应该看得见**：不满足就置灰，并把原因写进 tooltip。
        """
        kind = getattr(self, "_selected_kind", None)
        for name, accepted in self._NEEDS_SELECTION.items():
            action = self.actions_by_name.get(name)
            if action is None:
                continue
            allowed = kind in accepted
            action.setEnabled(allowed)
            what = "节点或杆件" if len(accepted) > 1 else "节点"
            action.setToolTip(action.text() if allowed
                              else f"请先在视口中选中一个{what}")

    def create_load(self) -> None:
        """Abaqus 式：选中对象，创建载荷。"""
        kind = getattr(self, "_selected_kind", None)
        if kind not in ("node", "member"):
            self.set_prompt("创建载荷：请先在视口中选中一个节点或杆件，然后点击此按钮")
            self.statusBar().showMessage("请先选中一个节点或杆件，然后创建载荷。", 5000)
            return
        from .load_dialog import LoadDialog
        ident = self._selected_id
        available = [case["name"] for case in self.session.model.get("load_cases", [])]
        shown_cases = available or ["Load-1"]
        dlg = LoadDialog(kind, ident, parent=self,
                         units=self.session.model.get("units", "N-m-Pa"),
                         cases=shown_cases, current_case=self.case)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        load_type = dlg.get_load_type()
        load = dlg.get_load()
        load_name = dlg.get_name()
        case_name = dlg.get_case()
        if not load_name:
            QMessageBox.warning(self, "施加载荷失败", "载荷名称不能为空。")
            return
        # 确保有工况
        if case_name not in available:
            case_result = self.session.add_load_case(case_name)
            if not case_result.ok:
                QMessageBox.warning(self, "创建荷载工况失败", self._explain(case_result.payload))
                return
        if load_type == "concentrated":
            result = self.session.set_nodal_load(ident, load, case_name, load_name)
            description = f"节点 {ident} 已施加集中力：{load}"
        elif load_type == "line":
            result = self.session.set_member_load(ident, load, case_name, load_name)
            description = f"杆件 {ident} 已施加满跨均布载荷：{load}"
        elif load_type == "trapezoid":
            end_load = dlg.get_end_load()
            result = self.session.set_member_span_load(
                ident, "trapezoid", load, w2=end_load,
                case_name=case_name, name=load_name)
            description = f"杆件 {ident} 已施加梯形载荷：i端 {load} → j端 {end_load}"
        elif load_type == "member_point":
            position = dlg.get_position()
            result = self.session.set_member_span_load(
                ident, "point", load, a=position,
                case_name=case_name, name=load_name)
            description = f"杆件 {ident} 距 i 端 {position:g} 处已施加集中力：{load}"
        else:
            QMessageBox.warning(self, "施加载荷失败", "未识别的荷载类型。")
            return
        if not result.ok:
            QMessageBox.warning(self, "施加载荷失败", self._explain(result.payload))
            return
        self.set_prompt(description)
        self._after_manual_edit(description)

    def open_sketch(self) -> None:
        """打开 2D 草图对话框，画一榀框架后拉伸成 3D 模型。"""
        materials = [m["name"] for m in self.session.model.get("materials", [])]
        sections = [s["name"] for s in self.session.model.get("sections", [])]
        self.set_prompt("草图建模：使用连续画线工具画一榀框架，设置拉伸参数后点击生成模型")
        from .sketch_dialog import SketchDialog
        dlg = SketchDialog(self, materials=materials, sections=sections)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.set_prompt("就绪 | 选择上方功能区模块开始建模")
            return
        data = dlg.get_model_data()
        if not data["nodes"]:
            self.statusBar().showMessage("草图为空，没有生成模型。", 4000)
            self.set_prompt("草图为空：请使用连续画线工具画一榀框架")
            return
        self._generate_from_sketch(data)

    def open_sketch_ai(self) -> None:
        """打开多模态草图识别面板：先识别草稿，再由用户确认加载。"""
        visible = self.sketch_dock.isVisible()
        self.sketch_dock.setVisible(not visible)
        if not visible:
            self.sketch_dock.raise_()
            self.set_prompt("多模态草图识别：选择图片 → 核对覆盖标注 → 标定尺度 → 确认加载")
        else:
            self.set_prompt("就绪 | 选择上方功能区模块开始建模")

    def _generate_from_sketch(self, data: dict) -> None:
        """把 2D 草图转换成 3D 模型，支持沿进深方向拉伸。"""
        try:
            nodes_2d = data["nodes"]
            lines_2d = data["lines"]
            plane = data["plane"]
            bays = data["bays"]
            bay_len = data["bay_len"]
            base = data["base"]

            if len(nodes_2d) < 2:
                raise ValueError("草图至少需要两个节点")
            if not lines_2d:
                raise ValueError("草图还没有杆件，请用连续画线连接节点")
            if bays < 1 or bay_len <= 0:
                raise ValueError("榀数和榀距必须大于 0")
            for n1, n2 in lines_2d:
                if not (0 <= n1 < len(nodes_2d) and 0 <= n2 < len(nodes_2d)):
                    raise ValueError("草图中存在无效的杆件端点")
                if n1 == n2:
                    raise ValueError("草图中存在起点和终点相同的杆件")
                p1, p2 = nodes_2d[n1], nodes_2d[n2]
                if abs(p1[0] - p2[0]) < 1e-9 and abs(p1[1] - p2[1]) < 1e-9:
                    raise ValueError("草图中存在长度为零的杆件，请删除重合节点")

            unit_scale = (1000.0 if self.session.model.get("units") == "N-mm-MPa"
                          else 1.0)

            # 确定 3D 坐标映射
            def to_3d(x2d, y2d, bay_idx):
                if plane == 0:
                    point = (x2d, bay_idx * bay_len, y2d)
                elif plane == 1:
                    point = (bay_idx * bay_len, x2d, y2d)
                else:
                    point = (x2d, y2d, bay_idx * bay_len)
                # 草图尺寸固定按界面标注的米输入，正式模型按当前单位制保存。
                return tuple(value * unit_scale for value in point)

            # 生成节点
            all_nodes = []
            node_map = []
            for b in range(bays):
                bay_map = {}
                for i, (x2d, y2d) in enumerate(nodes_2d):
                    x3d, y3d, z3d = to_3d(x2d, y2d, b)
                    all_nodes.append([x3d, y3d, z3d])
                    bay_map[i] = len(all_nodes)
                node_map.append(bay_map)

            # 生成杆件
            all_members = []
            for b in range(bays):
                for n1, n2 in lines_2d:
                    all_members.append([node_map[b][n1], node_map[b][n2]])
            if bays > 1:
                for b in range(bays - 1):
                    for i in range(len(nodes_2d)):
                        all_members.append([node_map[b][i], node_map[b + 1][i]])

            # 先在候选会话里建完并校验；失败不能覆盖用户当前模型。
            # 草图只替换几何，材料、截面和单位制与参数化生成器保持一致。
            from copy import deepcopy
            seed = {key: deepcopy(self.session.model[key])
                    for key in ("materials", "sections", "units")
                    if key in self.session.model}
            session = Session(model=seed)
            r1 = session.add_nodes(all_nodes)
            if not r1.ok:
                raise RuntimeError(f"添加节点失败: {r1.payload}")
            # add_nodes 会把重合点吸附到同一个节点；草图中的输入序号必须
            # 映射到实际节点编号，否则后续杆件会连到不存在的编号。
            actual_ids = r1.payload["input_node_ids"]
            all_members = [[actual_ids[i - 1], actual_ids[j - 1]]
                           for i, j in all_members]
            sections = [s["name"] for s in session.model.get("sections", [])]
            materials = [m["name"] for m in session.model.get("materials", [])]
            section = data.get("section") or (sections[0] if sections else "")
            material = data.get("material") or (materials[0] if materials else "")
            if section and sections and section not in sections:
                raise ValueError("草图使用的截面已不存在，请重新选择")
            if material and materials and material not in materials:
                raise ValueError("草图使用的材料已不存在，请重新选择")
            r2 = session.add_members(all_members, section, material)
            if not r2.ok:
                raise RuntimeError(f"添加杆件失败: {r2.payload}")

            # 设置柱底约束
            if base == "fixed":
                fix = [1, 1, 1, 1, 1, 1]
            else:
                fix = [1, 1, 1, 0, 0, 0]
            # 草图坐标原点只是画布中心，用户不应当必须把柱脚精确点在 z=0。
            # 以「吸附去重后的真实节点」判定柱底：add_nodes 可能把重合点合并，
            # 输入序号未必等于真实节点 id，必须用真实 id 和真实坐标，否则约束
            # 会加错节点或直接失败。
            real_nodes = session.model["nodes"]
            base_z = min(float(n["z"]) for n in real_nodes)
            base_ids = [int(n["id"]) for n in real_nodes
                        if abs(float(n["z"]) - base_z) < 1e-9]
            for nid in base_ids:
                support = session.edit_node(nid, fix=fix)
                if not support.ok:
                    raise RuntimeError(f"设置柱底约束失败: {support.payload}")

            self._replace_session(session)
            self.result = None
            self.case = None
            n_node = len(session.model["nodes"])
            n_mem = len(session.model["members"])
            self.statusBar().showMessage(
                f"已从草图生成模型：{n_node} 节点 / {n_mem} 杆件"
                f"（{bays} 榀，榀距 {bay_len}m）", 6000)
            self.refresh()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "草图生成失败", f"错误：{str(e)}")
            import traceback
            traceback.print_exc()

    def diagnose_model(self) -> None:
        """AI 辅助模型检查：检查模型完整性，给出修正建议。

        这是人工建模后的 AI 辅助校验——不替代人工，只帮人发现容易漏掉的问题。
        """
        from PySide6.QtWidgets import QMessageBox
        model = self.session.model
        nodes = model.get("nodes", [])
        members = model.get("members", [])
        supports = model.get("supports", [])
        load_cases = model.get("load_cases", [])

        issues = []
        suggestions = []
        locations: list[list[tuple[str, int]]] = []

        # 1. 模型为空
        if not nodes:
            issues.append("模型为空，没有节点")
            suggestions.append("在 Part 模块使用草图建模或规则框架生成模型")
            locations.append([])
            self._show_diagnose_result(issues, suggestions, locations)
            return

        # 2. 没有杆件
        if not members:
            issues.append("模型只有节点，没有杆件")
            suggestions.append("使用草图建模的连续画线工具，或在视口中用建杆件工具连接节点")
            locations.append([("node", int(n["id"])) for n in nodes])

        # 3. 没有支座
        if not supports:
            issues.append("模型没有任何支座约束")
            suggestions.append("在 Load 模块选中柱底节点，使用创建边界条件勾选自由度（固定端全选）")
            base_z = min(float(n.get("z", 0.0)) for n in nodes)
            locations.append([
                ("node", int(n["id"])) for n in nodes
                if abs(float(n.get("z", 0.0)) - base_z) < 1e-9])

        # 4. 没有荷载
        has_load = False
        for lc in load_cases:
            if lc.get("nodal_loads") or lc.get("member_loads"):
                has_load = True
                break
        if not has_load:
            issues.append("模型没有任何荷载")
            suggestions.append("在 Load 模块选中节点/杆件，使用创建载荷施加集中力/线载荷，或使用重力自动计算自重")
            locations.append([])

        # 5. 孤立节点（没有连接任何杆件）
        connected_nodes = set()
        for m in members:
            connected_nodes.add(int(m["i"]))
            connected_nodes.add(int(m["j"]))
        isolated = [n["id"] for n in nodes if int(n["id"]) not in connected_nodes]
        if isolated:
            issues.append(f"有 {len(isolated)} 个孤立节点（没有连接任何杆件）：{isolated[:5]}")
            suggestions.append("检查是否有多余节点，使用删除工具移除，或用建杆件工具连接")
            locations.append([("node", int(nid)) for nid in isolated])

        # 6. 重复杆件
        member_pairs = set()
        duplicates = []
        duplicate_ids = []
        for m in members:
            pair = tuple(sorted([int(m["i"]), int(m["j"])]))
            if pair in member_pairs:
                duplicates.append(pair)
                duplicate_ids.append(int(m["id"]))
            member_pairs.add(pair)
        if duplicates:
            issues.append(f"有 {len(duplicates)} 根重复杆件：{duplicates[:3]}")
            suggestions.append("重复杆件会导致刚度矩阵异常，使用删除工具移除重复的杆件")
            locations.append([("member", mid) for mid in duplicate_ids])

        # 7. 零长度杆件
        zero_len = []
        node_coords = {int(n["id"]): (n["x"], n["y"], n["z"]) for n in nodes}
        for m in members:
            i, j = int(m["i"]), int(m["j"])
            if i in node_coords and j in node_coords:
                x1, y1, z1 = node_coords[i]
                x2, y2, z2 = node_coords[j]
                if ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) < 1e-6:
                    zero_len.append(m["id"])
        if zero_len:
            issues.append(f"有 {len(zero_len)} 根零长度杆件：{zero_len[:3]}")
            suggestions.append("零长度杆件会导致求解失败，检查节点坐标是否正确")
            locations.append([("member", int(mid)) for mid in zero_len])

        # 8. 柱底没有约束（z=0 的节点没有支座）
        base_nodes = [int(n["id"]) for n in nodes if abs(n.get("z", 0)) < 0.01]
        supported_nodes = {int(s["node"]) for s in supports}
        unsupported_base = [n for n in base_nodes if n not in supported_nodes]
        if unsupported_base and base_nodes:
            issues.append(f"有 {len(unsupported_base)} 个底部节点（z≈0）没有支座约束：{unsupported_base[:5]}")
            suggestions.append("底部节点通常需要支座约束，在 Load 模块选中这些节点，使用创建边界条件设置固定端或铰接")
            locations.append([("node", nid) for nid in unsupported_base])

        # 9. 材料/截面
        if not model.get("materials"):
            issues.append("没有定义材料")
            suggestions.append("在 Property 模块使用创建材料定义弹性模量、泊松比、密度")
            locations.append([("member", int(m["id"])) for m in members])
        if not model.get("sections"):
            issues.append("没有定义截面")
            suggestions.append("在 Property 模块使用创建截面选择梁截面类型，输入尺寸")
            locations.append([("member", int(m["id"])) for m in members])

        if not issues:
            QMessageBox.information(self, "模型检查",
                                    "模型检查通过，没有发现明显问题。\n\n"
                                    "可以在 Step 模块提交求解。")
        else:
            self._show_diagnose_result(issues, suggestions, locations)

    def _show_diagnose_result(self, issues: list, suggestions: list,
                              locations: list[list[tuple[str, int]]] | None = None) -> None:
        """显示模型检查结果。"""
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout,
                                       QLabel, QListWidget, QPushButton,
                                       QVBoxLayout)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"模型检查 — 发现 {len(issues)} 个问题")
        dlg.resize(560, 430)
        layout = QVBoxLayout(dlg)
        issue_list = QListWidget(dlg)
        issue_list.addItems([f"{i + 1}. {text}" for i, text in enumerate(issues)])
        layout.addWidget(issue_list)
        detail = QLabel(dlg)
        detail.setWordWrap(True)
        detail.setProperty("panel", "hint")
        layout.addWidget(detail)
        nav = QHBoxLayout()
        previous = QPushButton("上一项", dlg)
        locate = QPushButton("在视口定位", dlg)
        following = QPushButton("下一项", dlg)
        nav.addWidget(previous)
        nav.addWidget(locate)
        nav.addWidget(following)
        layout.addLayout(nav)

        refs_by_row = locations or [[] for _ in issues]

        def select_row(row: int) -> None:
            if not 0 <= row < len(issues):
                return
            detail.setText(f"修正建议：{suggestions[row]}")
            refs = refs_by_row[row] if row < len(refs_by_row) else []
            locate.setEnabled(bool(refs))
            self.viewport.set_problem_refs(refs)
            previous.setEnabled(row > 0)
            following.setEnabled(row + 1 < len(issues))

        issue_list.currentRowChanged.connect(select_row)
        previous.clicked.connect(
            lambda: issue_list.setCurrentRow(max(0, issue_list.currentRow() - 1)))
        following.clicked.connect(
            lambda: issue_list.setCurrentRow(
                min(len(issues) - 1, issue_list.currentRow() + 1)))
        locate.clicked.connect(
            lambda: self.locate_problem(refs_by_row[issue_list.currentRow()]))
        issue_list.setCurrentRow(0)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.exec()

    def toggle_agent_panel(self) -> None:
        """切换 AI 助手对话框显示/隐藏（右侧浮动按钮调用）。"""
        visible = self.chat_dock.isVisible()
        self.chat_dock.setVisible(not visible)
        if not visible:
            self.chat_dock.raise_()
            self.act_chat.setChecked(True)
            self.set_prompt("AI 助手已展开，可以输入自然语言描述模型需求")
        else:
            self.act_chat.setChecked(False)
            self.set_prompt("就绪 | 选择上方功能区模块开始建模")
        # 对话框显示/隐藏后，更新按钮位置
        self._update_agent_button_pos()

    def showEvent(self, event):
        """窗口显示时，确保浮动按钮在最上层并位置正确。"""
        super().showEvent(event)
        if not getattr(self, "_initial_workspace_sized", False):
            # 实测 1680x1000 窗口下，顶部五条横带吃掉 257px（26% 高），
            # 左右停靠区 250+380=630px（38% 宽）——三维视口只剩不到一半窗口。
            # 空项目时模型树里全是"（0）"，右侧对话区也大片空白，
            # 却占着最贵的横向空间。**默认宽度按"够用"取，不按"能塞下"取。**
            self.resizeDocks([self.tree_dock], [200], Qt.Orientation.Horizontal)
            self.resizeDocks([self.chat_dock], [330], Qt.Orientation.Horizontal)
            self._initial_workspace_sized = True
        self._update_agent_button_pos()
        self._position_empty_state()

    def resizeEvent(self, event):
        """窗口大小变化时，更新浮动按钮位置。"""
        super().resizeEvent(event)
        self._update_agent_button_pos()
        self._position_empty_state()

    def _update_agent_button_pos(self):
        """Only show the compact assistant launcher when its dock is closed."""
        if not hasattr(self, 'agent_button'):
            return
        dock_visible = hasattr(self, 'chat_dock') and self.chat_dock.isVisible()
        self.agent_button.setVisible(not dock_visible)
        if dock_visible:
            return
        x = self.width() - self.agent_button.width() - 14
        y = max(180, (self.height() - self.agent_button.height()) // 2)
        self.agent_button.move(x, y)
        self.agent_button.raise_()

    def _on_chat_visibility(self, visible: bool) -> None:
        if hasattr(self, "act_chat"):
            self.act_chat.setChecked(visible)
        self._update_agent_button_pos()

    def toggle_props(self) -> None:
        on = self.actions_by_name["props"].isChecked()
        self.props_dock.setVisible(on)
        if on:
            self.props_dock.raise_()
            self.properties.refresh()

    def toggle_chat(self) -> None:
        """切换 AI 助手对话框显示/隐藏（功能区按钮调用）。"""
        visible = self.act_chat.isChecked()
        self.chat_dock.setVisible(visible)
        if visible:
            self.chat_dock.raise_()
            self.set_prompt("AI 助手已展开，可以输入自然语言描述模型需求")

    # --- 文件 ---

    def open_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "打开模型", "",
                                              "模型 JSON (*.json)")
        if not path:
            return
        self._load_model_path(Path(path))

    def _recent_paths(self) -> list[Path]:
        """返回仍存在的近期模型，最多八个。"""
        settings = QSettings("SpaceFrameAgent", "Desktop")
        raw = settings.value("recent_model_paths", [])
        if isinstance(raw, str):
            raw = [raw]
        paths = []
        for value in raw or []:
            path = Path(str(value))
            if path.is_file() and path not in paths:
                paths.append(path)
        return paths[:8]

    def _remember_recent_path(self, path: Path) -> None:
        settings = QSettings("SpaceFrameAgent", "Desktop")
        paths = [path, *(item for item in self._recent_paths() if item != path)]
        settings.setValue("recent_model_paths", [str(item) for item in paths[:8]])

    def _populate_recent_menu(self) -> None:
        self.recent_menu.clear()
        paths = self._recent_paths()
        if not paths:
            empty = self.recent_menu.addAction("暂无可用的近期模型")
            empty.setEnabled(False)
            return
        for index, path in enumerate(paths, start=1):
            action = self.recent_menu.addAction(f"&{index}  {path.name}")
            action.setToolTip(str(path))
            action.triggered.connect(lambda _=False, item=path: self._load_model_path(item))
        self.recent_menu.addSeparator()
        clear = self.recent_menu.addAction("清除近期记录")
        clear.triggered.connect(
            lambda: QSettings("SpaceFrameAgent", "Desktop").remove("recent_model_paths"))

    def _load_model_path(self, path: Path) -> bool:
        """从明确路径加载模型；文件选择器和“最近打开”共用这条安全路径。"""
        if not path.is_file():
            QMessageBox.warning(self, "文件不存在", f"找不到模型文件：\n{path}")
            return False
        import json
        from model_io import validate_payload
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except Exception as exc:              # noqa: BLE001
            QMessageBox.warning(self, "文件读取失败", f"{type(exc).__name__}: {exc}")
            return False
        errors = validate_payload(payload)
        if errors:
            # **先校验再装进去**：装了再校验的话，一个坏文件会把好模型顶掉
            QMessageBox.warning(self, "模型校验未通过",
                                "\n".join(str(e) for e in errors[:8]))
            return False
        session = Session()
        result = session.set_model(model=payload)
        if not result.ok:
            QMessageBox.warning(self, "模型未载入", self._explain(result.payload))
            return False
        from draft_commit import load_sidecar
        provenance, sidecar_warning = load_sidecar(path, session.model)
        session.multimodal_provenance = provenance
        if session.history.cursor >= 0:
            session.history[session.history.cursor].multimodal_provenance = provenance
        self._replace_session(session)
        self.result = self.case = None
        self.refresh()
        self._remember_recent_path(path)
        if sidecar_warning:
            QMessageBox.warning(self, "多模态追溯未恢复", sidecar_warning)
        self.statusBar().showMessage(f"已载入模型 {Path(path).name}", 5000)
        return True

    def save_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        if not self.session.model.get("nodes"):
            QMessageBox.information(self, "无可保存的模型", "当前没有模型可保存，请先建立模型。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存模型", "模型.json",
                                              "模型 JSON (*.json)")
        if not path:
            return
        import json
        Path(path).write_text(
            json.dumps(self.session.model, ensure_ascii=False, indent=2),
            encoding="utf-8")
        from draft_commit import save_sidecar
        save_sidecar(path, self.session.model, self.session.multimodal_provenance)
        self._remember_recent_path(Path(path))
        self.statusBar().showMessage(f"模型已保存至 {Path(path).name}", 5000)

    # --- 显示 ---

    def show_model(self) -> None:
        """回到模型视图。**求解之后也要能回来**——看完云图想再核对一遍
        支座和荷载画在哪，没有这个按钮就只能新建模型重来。"""
        self.set_mode("模型")

    def show_analysis_mesh(self) -> None:
        """只读预览 Physical Member → Analysis Element 编译结果。"""
        if not self.session.model.get("nodes"):
            QMessageBox.information(
                self, "无可预览的模型", "当前没有模型，请先建立物理杆件。")
            return
        preview = self.session.preview_analysis_mesh()
        if not preview.ok:
            self._show_validation_errors(preview.payload)
            self.set_mode("模型")
            return
        rows, locations = [], []
        for item in preview.payload.get("analysis_element_details", ()):
            physical = int(item["physical_member"])
            rows.append([int(item["element"]), physical,
                         int(item["i"]), int(item["j"])])
            locations.append(("member", physical))
        p = preview.payload
        caption = (f"分析网格：{p['physical_members']} 个物理杆件 → "
                   f"{p['analysis_elements']} 个分析杆段，"
                   f"{p['split_nodes']} 个切分节点。点击一行定位物理杆件。")
        self.results_dock.setVisible(True)
        self.results.show_rows(caption, ["分析杆段", "物理杆件", "节点 i", "节点 j"],
                               rows, locations)
        self.set_mode("分析网格")
        self.statusBar().showMessage(caption, 8000)

    def show_deformed(self) -> None:
        if self._needs_solution():
            self.set_mode("变形")

    def show_contour(self) -> None:
        if self._needs_solution():
            self.results_dock.setVisible(True)
            self.results_dock.raise_()
            self.set_mode("云图")

    def pick_component(self) -> None:
        """选择梁中心线内力结果显示哪个分量。

        菜单里把每个分量是什么写出来——`My` 和 `Mz` 光看字母是分不出
        哪个是平面内弯矩的，而选错了看半天图都是白看。
        """
        labels = (("M", "合弯矩 |M|（空间云图推荐）"),
                  ("V", "合剪力 |V|（空间云图推荐）"),
                  ("N", "轴力 N（受拉为正）"), ("Vy", "剪力 Vy"),
                  ("Vz", "剪力 Vz"), ("T", "扭矩 T"),
                  ("My", "弯矩 My（绕局部 y）"), ("Mz", "弯矩 Mz（平面内，常用）"))
        menu = QMenu(self)
        for key, text in labels:
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(key == self.component)
            act.triggered.connect(lambda _=False, k=key: self._set_component(k))
        anchor = self.actions_by_name["diagram"]
        widgets = [w for w in anchor.associatedObjects()
                   if hasattr(w, "mapToGlobal")]
        menu.exec(widgets[0].mapToGlobal(widgets[0].rect().bottomLeft())
                  if widgets else self.cursor().pos())

    def _set_component(self, key: str) -> None:
        self.component = key
        if self._needs_solution():
            self.set_mode("云图")

    def _needs_solution(self) -> bool:
        """需要结果的动作先过这道闸。

        拦下来时要**把模式退回"模型"**：不退的话按钮勾在"变形"上，
        视口里画的却是模型——界面在说谎。
        """
        if self.result is not None and self.result.ok:
            return True
        QMessageBox.information(self, "尚无分析结果", "尚无分析结果。请先执行求解，再查看该结果视图。")
        self.set_mode("模型")
        return False

    # --- 分析 ---

    def run_modal(self) -> None:
        if self._needs_solution():
            self.set_mode("模态")

    def run_buckling(self) -> None:
        self._analyse("buckling", "线弹性屈曲",
                      lambda: self.session.buckling_analysis(num_modes=5))

    def run_envelope(self) -> None:
        self._analyse("envelope", f"{self.component} 包络",
                      lambda: self.session.query_envelope(
                          component=self.component))

    def run_deflection(self) -> None:
        self._analyse("deflection", "最大挠度",
                      lambda: self.session.query_results(
                          what="max_deflection"))

    def run_solid_joint(self) -> None:
        """对当前节点运行自研 C3D10 局部实体子模型。

        这是分钟级作业，必须复用统一后台 runner。入口还同时卡住三个容易
        造成“按钮点了却不知道为什么不算”的前置条件：选中节点、整体梁模型
        已求解、当前没有别的作业。
        """
        if getattr(self, "_selected_kind", None) != "node":
            self.set_prompt("节点实体：请先用“选择节点”在视口中选中一个节点")
            return
        if self.session.solution is None:
            QMessageBox.information(
                self, "节点实体", "请先求解整体梁模型，再选择节点运行局部实体分析。")
            return
        node_id = int(self._selected_id)
        case = self.case
        title = f"节点 {node_id} 局部实体"
        self._analyse(
            "solid_joint", title,
            lambda: self.session.analyze_joint_solid(node_id=node_id, case=case))

    # --- 校核 ---

    def run_strength_check(self) -> None:
        """逐杆强度与稳定校核。缺许用应力时不算，并直接告诉用户去哪儿补。"""
        self._analyse("strength", "强度验算",
                      lambda: self.session.check_strength())

    def run_symmetry_check(self) -> None:
        self._analyse("symmetry", "对称性",
                      lambda: self.session.check_symmetry(case=self.case))

    def run_numbering_check(self) -> None:
        self._analyse("numbering", "编号与存储",
                      lambda: self.session.check_numbering())

    def write_report(self) -> None:
        self._analyse("generic", "计算书",
                      lambda: self.session.write_report(
                          fmt="both", filename="刚架分析报告"))

    def _analyse(self, kind: str, title: str, fn) -> None:
        """跑一个分析，结果进**结果面板**。

        原来是弹一个消息框、把 JSON 塞进"详细信息"——那是给开发者看的。
        工程结果该是表：能扫、能排序、能点一行在视口里定位。

        照样走后台线程：屈曲、模态在网架规模上是秒级的。
        """
        if not self.session.model.get("nodes"):
            QMessageBox.information(self, title,
                                    "当前没有模型，请先建立模型。")
            return
        if self.runner.busy:
            return
        self._on_busy(True)
        self.statusBar().showMessage(f"{title}计算中…")
        self.results_dock.setVisible(True)
        self.results.show_message(title, "计算中…")

        def done(result) -> None:
            self._on_busy(False)
            self.statusBar().clearMessage()
            if not result.ok:
                self.results.show_message(
                    f"{title}未完成",
                    self._explain(result.payload))
            else:
                caption, cols, rows, loc = result_rows.to_rows(
                    kind, result.payload)
                self.results.show_rows(f"{title}　{caption}", cols, rows, loc,
                                       result_rows.row_marks(kind,
                                                             result.payload))
            self.refresh()

        def failed(kind_: str, message: str) -> None:
            self._on_busy(False)
            self.statusBar().clearMessage()
            self.results.show_message(f"{title}　计算出错",
                                      f"{kind_}：{message[:400]}")

        self.runner.submit(fn, on_done=done, on_failed=failed)

    @staticmethod
    def _explain(payload: dict) -> str:
        """把工具返回的错误摊成人话。

        三段式：**说清是什么问题、为什么、下一步做什么**。
        只给一个异常类名或一句"失败"，用户既不知道该改模型还是该改参数。
        """
        errors = payload.get("errors")
        if errors:
            lines = [f"  · {e}" for e in list(errors)[:10]]
            more = f"\n  …另有 {len(errors) - 10} 条" if len(errors) > 10 else ""
            return "模型校验未通过：\n" + "\n".join(lines) + more
        error = payload.get("error")
        hint = payload.get("hint")
        text = str(error) if error else "未给出原因。"
        return text + (f"\n建议：{hint}" if hint else "")

    # --- 视图 ---

    def view_iso(self) -> None:
        self.viewport.set_view("isometric")

    def view_front(self) -> None:
        self.viewport.set_view("xz")

    def view_side(self) -> None:
        self.viewport.set_view("yz")

    def view_top(self) -> None:
        self.viewport.set_view("xy")

    def view_fit(self) -> None:
        self.viewport.reset_camera()

    # --- 视口背景与网格地面 ---

    def open_bg_settings(self) -> None:
        """打开视口背景设置弹窗（预设主题 / 自定义颜色 / 网格地面）。"""
        from .bg_dialog import BackgroundDialog
        dlg = BackgroundDialog(self.viewport, self)
        dlg.exec()

    def toggle_grid_floor(self) -> None:
        on = not self.viewport.show_grid_floor
        self.viewport.toggle_grid_floor(on)
        self.statusBar().showMessage(
            "网格地面：显示" if on else "网格地面：隐藏", 3000)

    def export_shot(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "保存截图", "界面截图.png",
                                              "PNG 图片 (*.png)")
        if path:
            self.export_screenshot(path)
            self.statusBar().showMessage(f"截图已保存至 {Path(path).name}", 5000)

    # --- 关闭 ---

    def closeEvent(self, event) -> None:
        """退出时按顺序收摊。

        **VTK 的渲染窗口不显式关掉，进程退出时会崩**——用户看到的是
        "程序已停止工作"，而他明明只是点了关闭。这类崩溃不影响任何数据，
        却会让人觉得整个软件不稳。

        顺序要紧：先等后台线程停下，再关渲染窗口。反过来的话，
        工作线程可能正拿着一个已经被销毁的对象。
        """
        try:
            if self.runner.busy:
                self.runner.wait(3000)
        except Exception:                       # noqa: BLE001
            pass
        try:
            self.viewport.shutdown()
        except Exception:                       # noqa: BLE001
            pass
        super().closeEvent(event)

    # --- 导出 ---

    def _solve_text(self) -> str:
        if self.result is None:
            return "求解：未求解"
        if not self.result.ok:
            return "求解：模型校验未通过"
        n = len(self.result.payload["cases"])
        return f"求解：{n} 个工况/组合　当前 {self.case}"

    def export_screenshot(self, path: str) -> str:
        """整窗截图。

        **不能只用 QWidget.grab()**：VTK 的渲染窗口是个带自己 GL 表面的原生子窗口，
        Qt 抓窗口时抓不到它，视口会是一片黑。所以分两次抓再合成——
        外壳用 Qt 抓，视口用 plotter 自己的 screenshot，然后贴回它的位置。
        答辩截图和报告插图都走这条路。
        """
        from PySide6.QtCore import QBuffer, QIODevice
        from PySide6.QtGui import QImage

        chrome = self.grab().toImage()
        shot = self.viewport.plotter.screenshot(return_img=True)
        if shot is not None:
            import numpy as np
            arr = np.ascontiguousarray(shot[:, :, :3])
            h, w, _ = arr.shape
            view = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            geo = self.viewport.geometry()
            top_left = self.viewport.mapTo(self, self.viewport.rect().topLeft())
            from PySide6.QtGui import QPainter
            painter = QPainter(chrome)
            painter.drawImage(top_left, view.scaled(
                geo.width(), geo.height()))
            painter.end()
        chrome.save(path)
        return path

    # --- 树的动作 ---

    def _on_tree_action(self, action: str, payload) -> None:
        if action == "materials":
            self.create_material()
        elif action == "sections":
            self.create_section()
        elif action in {"geometry", "nodes", "members"}:
            self.open_tables()
        elif action == "supports":
            self.show_bc_panel()
        elif action == "support_object" and payload is not None:
            self.viewport.set_selection("node", int(payload))
            self._on_picked("node", int(payload))
            self.show_bc_panel()
        elif action == "loads":
            if payload:
                self.case = str(payload)
                self.refresh()
            self.show_bc_panel()
        elif action == "load_object" and isinstance(payload, dict):
            self.case = str(payload.get("case") or self.case or "")
            kind, ident = str(payload["kind"]), int(payload["id"])
            self.viewport.set_selection(kind, ident)
            self._on_picked(kind, ident)
            self.show_bc_panel()
        elif action == "results" and payload:
            self.case = str(payload)
            self.set_mode("变形")
        elif action == "history":
            self.timeline_dock.setVisible(True)
            self.timeline_dock.raise_()
            self.timeline.rebuild(self.session.history)
