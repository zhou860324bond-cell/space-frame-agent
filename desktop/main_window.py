"""主窗口。

布局照 CAE 的通行做法：顶部模块工具栏、中间视口、两侧与底部抽屉。
主窗口负责装配与建模流程；命令、结果交互和偏好设置分别由 mixin 承担，
实际数值计算留在内核，场景绘制留在视口。
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel,
                               QMainWindow, QMenu, QMessageBox, QPushButton,
                               QScrollArea, QSizePolicy, QSplitter, QStatusBar, QToolBar,
                               QToolButton,
                               QVBoxLayout, QWidget)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import Session                          # noqa: E402

from stress import StressUnavailable                # noqa: E402
from . import scene, theme                         # noqa: E402
from .chat_panel import ChatPanel                  # noqa: E402
from .model_tree import ModelTree, ModelTreePanel  # noqa: E402
from .diagram_panel import DiagramPanel             # noqa: E402
from .panels import ResultPanel, TimelinePanel      # noqa: E402
from .sketch_panel import SketchPanel                # noqa: E402
from .section_opt_panel import SectionOptPanel       # noqa: E402
from .properties import PropertiesPanel             # noqa: E402
from .drawers import (BOTTOM, LEFT, RIGHT, DrawerHost,  # noqa: E402
                      PanelWindow)
from . import dialog_styles                         # noqa: E402
from . import result_rows                           # noqa: E402
from .viewport import Viewport                     # noqa: E402
from .worker import Runner                         # noqa: E402
from .workflow_bar import WorkflowBar, draft_target  # noqa: E402
from . import glyphs
from .window_commands import WindowCommandsMixin
from .window_preferences import WindowPreferencesMixin, _settings
from .window_results import WindowResultsMixin
from .window_objects import WindowObjectsMixin
from .window_jobs import WindowJobsMixin
from .window_assembly import WindowAssemblyMixin
from .window_viz import WindowVizMixin
from .region_prompt import RegionPickController


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
MODES = ("模型", "分析网格", "变形", "云图", "实体云图", "内力图", "应力比", "模态")


class MainWindow(WindowObjectsMixin, WindowAssemblyMixin, WindowJobsMixin, WindowVizMixin,
                 WindowCommandsMixin,
                 WindowPreferencesMixin,
                 WindowResultsMixin, QMainWindow):
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
        self._solid_grid = None
        self._solid_binding = None
        self._solid_info = None
        self._solid_reset_camera = True
        self.analysis_type = "linear"
        self._modeling_reviews = {}
        self._analysis_mesh_key = None
        self.analysis_options = {"increments": 10, "max_iter": 40,
                                 "tolerance": 1e-7}

        self.setWindowTitle("FrameLab Studio — 空间刚架分析")
        self.resize(1440, 900)

        self.viewport = Viewport(self)
        # 中心区是一行三栏：左模型栏（模型树 + 属性）| 视口 | 右 AI 助手。
        # 原来两侧各有一条窄栏，七个按钮开七种抽屉页，其中模型树、属性、边界
        # 三页与功能区、管理器重复；现在模型栏常驻（像 Abaqus 的模型树），
        # AI 助手由页签行右端的开关收放，两侧不再有窄栏。
        center = QWidget(self)
        row = QHBoxLayout(center)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.agent_splitter = QSplitter(Qt.Orientation.Horizontal, center)
        self.agent_splitter.setChildrenCollapsible(False)
        row.addWidget(self.agent_splitter, 1)
        self.agent_splitter.addWidget(self.viewport)
        self.setCentralWidget(center)

        # 求解和大模型调用都得离开界面线程，否则窗口会变"未响应"
        self.runner = Runner(self)
        # 模型每变一次，推迟一会儿写一份自动存档（见 desktop/autosave.py）
        from .autosave import AutoSave
        self.autosave = AutoSave(self)

        # 三个抽屉：左放"模型是什么"，右放"找人帮忙/定义边界"，底放"算出了
        # 什么"。每侧一个抽屉、多页用页签切——不再是九个各自漂着的小窗。
        self.drawers = DrawerHost(self, self.viewport)
        self.drawers.on_insets = self._on_drawer_insets
        self.left_drawer = self.drawers.drawer(LEFT, 290)
        self.right_drawer = self.drawers.drawer(RIGHT, 420)
        self.bottom_drawer = self.drawers.drawer(BOTTOM, 300)

        self.chat = ChatPanel(self.session, self.runner, self)
        self.chat.model_changed.connect(self._on_chat_changed)
        self.chat.busy_changed.connect(self._on_busy)
        # 默认使用用户选择的并排布局，右侧 AI 入口可以恢复完整模型视口。
        from .agent_workspace import PaneHandle, WorkspaceHandle
        self.chat.setMinimumWidth(320)
        self.viewport.setMinimumWidth(250)
        self.agent_splitter.addWidget(self.chat)
        self.chat_dock = WorkspaceHandle(self.chat, self)
        self.chat.locate_requested.connect(self._on_picked)
        self.chat.image_dropped.connect(self._recognize_dropped_image)

        # 手绘草图 → 模型：独立窗口，从建模页的「草图识别」打开。它是一次走完
        # 就关的流程、要宽画布，不和常驻参考的面板挤在抽屉里。
        self.sketch = SketchPanel(self.session, self.runner, self)
        self.sketch.model_loaded.connect(self._on_sketch_loaded)
        self.sketch_dock = PanelWindow("手绘草图识别", self.sketch, self)

        self.tree = ModelTree(self)
        self.tree.activated_item.connect(self._on_tree_action)
        self.tree.feature_action.connect(self._on_feature_action)
        self.tree.feature_selected.connect(self._on_feature_selected)
        # 模型栏：上半模型树、下半选中对象的属性，常驻在视口左侧
        self.model_panel = QSplitter(Qt.Orientation.Vertical, self)
        self.model_panel.setObjectName("modelPanel")
        self.model_panel.setChildrenCollapsible(False)
        self.model_panel.setMinimumWidth(240)
        # 树上方带搜索框：按名称筛选或输入“节点 65”精确定位，与视口共用同一选择
        self.tree_panel = ModelTreePanel(self.tree, self)
        self.model_panel.addWidget(self.tree_panel)
        self.agent_splitter.insertWidget(0, self.model_panel)
        self.agent_splitter.setStretchFactor(0, 0)
        self.agent_splitter.setStretchFactor(1, 1)
        self.agent_splitter.setStretchFactor(2, 0)
        self.agent_splitter.setSizes([280, 860, 400])
        self.tree_dock = WorkspaceHandle(self.model_panel, self, title="模型树", visible=False)
        # 开局视作"自动收起"状态：第一次刷新时若已有模型，就把树展开
        self._tree_auto_hidden = True
        # 用户亲手收起过模型树，就别在下一次刷新时又自作主张地弹出来
        self._tree_dismissed = False
        self._pending_command = None
        # (求解对象, 强度验算 payload)——按求解对象身份判断是否过期
        self._utilization = None
        self._task_busy = False
        self._close_pending = False
        self._utilization_timer = QTimer(self)
        self._utilization_timer.setSingleShot(True)
        self._utilization_timer.timeout.connect(
            lambda: self.show_utilization(interactive=False))
        self._busy_action_states = {}
        self._displayed_solution = None

        self.properties = PropertiesPanel(self.session, self)
        self.properties.edited.connect(self._on_property_edited)
        self.model_panel.addWidget(self._scrolled(self.properties))
        self.model_panel.setStretchFactor(0, 3)
        self.model_panel.setStretchFactor(1, 2)
        self.props_dock = PaneHandle(self.model_panel.widget(1), self.tree_dock, self,
                                     title="属性")

        from .solid_result import SolidResultPanel
        self.solid_panel = SolidResultPanel(self)
        self.solid_panel.changed.connect(self._redraw_solid_result)
        self.solid_panel.file_requested.connect(self._switch_solid_level)
        self.solid_panel.open_requested.connect(self.open_solid_result)
        self.solid_panel.model_requested.connect(self.show_model)
        self.solid_panel.query_requested.connect(self.viewport.set_solid_query)
        self.solid_panel.extreme_requested.connect(self.locate_solid_extreme)
        self.solid_panel.clear_requested.connect(self.clear_solid_query)
        self.solid_dock = self.left_drawer.add_page("solid", "实体结果", self.solid_panel)

        # 原来这里还有一张「边界条件」表单页（逐个节点填 U1…UR3），与载荷页的
        # 创建边界条件、边界条件管理器做同一件事而数据不同，已删除；它独有的
        # 给定位移（支座沉降）移到「创建载荷」的类型里。

        self.timeline = TimelinePanel(self)
        self.timeline.goto_step.connect(self.goto_step)
        self.timeline_dock = self.left_drawer.add_page(
            "timeline", "建模过程", self.timeline)

        self.results = ResultPanel(self)
        self.results.locate.connect(self.locate)
        self.results.display_changed.connect(self._on_result_display_changed)
        self.results.extreme_requested.connect(self.locate_current_extreme)
        self.results.stress_requested.connect(self.show_section_stress)
        self.result_display_options = self.results.display_options()
        self._last_probe = None
        self.results.setMinimumHeight(120)
        self.results_dock = self.bottom_drawer.add_page("results", "结果表", self.results)
        # 明细表默认独立查看，避免打开结果时覆盖模型的下半部。
        self.results_dock.prefer_floating = True

        self.diagram = DiagramPanel(self)
        self.diagram.locate.connect(self.locate)
        self.diagram_dock = self.bottom_drawer.add_page("diagram", "单杆内力图", self.diagram)
        self.diagram_dock.prefer_floating = True

        self.section_opt = SectionOptPanel(self.session, self.runner, self)
        self.section_opt_dock = self.bottom_drawer.add_page(
            "section_opt", "截面优化", self.section_opt)
        self.section_opt_dock.prefer_floating = True

        self.viewport.picked.connect(self._on_picked)
        self.viewport.solid_picked.connect(self.query_solid_cell)
        self.viewport.probed.connect(self.probe_member_result)
        self.viewport.node_created.connect(self._on_node_created)
        self.viewport.member_created.connect(self._on_member_created)
        self.viewport.node_snapped.connect(self._on_node_snapped)
        self.viewport.escape_pressed.connect(self.cancel_interaction)
        self.viewport.context_requested.connect(self._show_viewport_context_menu)
        self.viewport.cleared.connect(self._on_selection_cleared)

        self._build_actions()
        for handle in (self.chat_dock, self.tree_dock, self.props_dock):
            handle.visibilityChanged.connect(lambda _on: self._sync_rails())
        for drawer in (self.left_drawer, self.bottom_drawer):
            for key in drawer.keys():
                drawer.page(key).handle.visibilityChanged.connect(lambda _on: self._sync_rails())
        self._sync_rails()
        self._build_empty_state()
        self._build_ribbon()
        self._build_menus()
        self._build_statusbar()
        # Abaqus 式提示区选择：命令要区域时由它接管视口拾取（见 region_prompt.py）
        self.region_pick = RegionPickController(self)
        self.actions_by_name["render_profiles"].setChecked(self.viewport.render_profiles)
        self.runner.busy_changed.connect(self._on_busy)
        self.refresh()
        self.actions_by_name["stop_task"].setEnabled(False)

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
        box.setContentsMargins(24, 24, 24, 24)
        box.setSpacing(8)
        title = QLabel("创建第一个结构模型", panel)
        title.setProperty("emptyState", "title")
        subtitle = QLabel(
            "推荐从参数化框架开始；图纸识别为实验性功能，需核对识别结果。",
            panel)
        subtitle.setProperty("emptyState", "subtitle")
        subtitle.setWordWrap(True)
        steps = QLabel(
            "建好几何之后，按页签从左往右：属性 → 载荷 → 作业求解 → 结果。"
            "右上角「下一步」随时告诉你该做什么；Ctrl+Shift+P 可搜索任意命令。",
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
                ("AI 识别图纸", "sketch_ai", False),
                ("参数化框架", "frame", True),
                ("二维草图", "sketch", False)):
            button = QPushButton(text, panel)
            button.setMinimumHeight(34)
            if action_name == "sketch_ai":
                button.setToolTip("实验性：识别后请核对拓扑、尺度和边界条件，再确认加载。")
            if primary:
                button.setProperty("role", "primary")
            button.clicked.connect(self.actions_by_name[action_name].trigger)
            actions.addWidget(button)
        box.addLayout(actions)
        panel.adjustSize()
        self.empty_state = panel
        self._position_empty_state()

    def _on_drawer_insets(self, left: int, right: int, bottom: int) -> None:
        self.viewport.set_overlay_insets(left, right, bottom)
        self._position_empty_state()

    def _position_empty_state(self) -> None:
        if not hasattr(self, "empty_state"):
            return
        # 居中在**没被抽屉盖住**的那一块里
        left, right, bottom = (self.drawers.insets() if hasattr(self, "drawers")
                               else (0, 0, 0))
        area = self.viewport.rect().adjusted(left, 0, -right, -bottom)
        x = area.left() + max(12, (area.width() - self.empty_state.width()) // 2)
        y = area.top() + max(12, (area.height() - self.empty_state.height()) // 2)
        self.empty_state.move(x, y)
        self.empty_state.raise_()

    # 全部面板把手（PanelHandle），测试与"全部收起"按它逐个检查。
    PANELS = ("tree_dock", "props_dock", "solid_dock", "timeline_dock", "chat_dock",
              "sketch_dock", "results_dock", "diagram_dock", "section_opt_dock")

    @staticmethod
    def _scrolled(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _on_drawer_page_opened(self, key: str) -> None:
        """面板打开时补一次刷新——关着的面板不跟随模型变化，省掉无用功。"""
        handle = getattr(self, {"tree": "tree_dock", "props": "props_dock",
                                "timeline": "timeline_dock", "chat": "chat_dock",
                                "results": "results_dock", "diagram": "diagram_dock",
                                "section_opt": "section_opt_dock"}.get(key, ""), None)
        if handle is None or not handle.isVisible():
            return
        if key == "props":
            self.properties.session = self.session
            self.properties.refresh()
        elif key == "timeline":
            self.timeline.rebuild(self.session.history)
        elif key == "diagram":
            self.diagram.attach(self.session, self.case)
        elif key == "tree":
            self.tree.rebuild(self.session, self.result)
        elif key == "chat":
            self.set_prompt("AI 助手已展开，可以输入自然语言描述模型需求")

    def _sync_rails(self) -> None:
        """功能区开关动作的勾选状态与面板的真实显隐保持一致。"""
        if not hasattr(self, "actions_by_name"):
            return
        for action_name, handle in (("chat", self.chat_dock), ("props", self.props_dock),
                                    ("tree_panel", self.tree_dock),
                                    ("timeline", self.timeline_dock),
                                    ("curve", self.diagram_dock),
                                    ("results_table", self.results_dock),
                                    ("section_opt_panel", self.section_opt_dock)):
            action = self.actions_by_name.get(action_name)
            if action is not None and action.isChecked() != handle.isVisible():
                action.blockSignals(True)
                action.setChecked(handle.isVisible())
                action.blockSignals(False)

    def _build_ribbon(self) -> None:
        from . import ribbon

        self.ribbon = ribbon.build(self)
        self.ribbon.currentChanged.connect(self._on_ribbon_page_changed)
        self.quickbar = ribbon.QuickBar(self, self)
        self.quickbar.case_changed.connect(self.set_case)
        self.quickbar.scale_changed.connect(self.set_scale)
        self.ribbon.set_collapsed(False)
        # 流程状态与「下一步」原来单独占一整行，和模块页签说的是同一件事。
        # 收成页签行右端的「流程」菜单加一个主按钮；页签本身按状态着色。
        self.workflow_bar = WorkflowBar(self, compact=True)
        self.workflow_bar.stage_requested.connect(self._on_workflow_stage)
        self.workflow_bar.state_changed.connect(self._sync_tab_states)
        self.workflow_bar.next_button.setProperty("role", "primary")
        self.ribbon.add_corner(self.workflow_bar)
        self.agent_button = QToolButton(self)
        self.agent_button.setDefaultAction(self.actions_by_name["chat"])
        self.agent_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.agent_button.setProperty("ribbonChrome", "agent")
        self.agent_button.setAutoRaise(True)
        self.ribbon.add_corner(self.agent_button)
        holder = QWidget(self)
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(self.ribbon)
        box.addWidget(self.quickbar)
        bar = QToolBar("功能区", self)
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.addWidget(holder)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

    def next_page(self, status) -> str:
        """下一步该去的模块页：空模型去建模，草稿去缺项所在页，可求解去作业，已求解去结果。"""
        from workflow import WorkflowPhase
        if status.phase is WorkflowPhase.EMPTY:
            return "建模"
        if status.phase is WorkflowPhase.DRAFT:
            return draft_target(status)[1]
        return "作业" if status.phase is WorkflowPhase.READY else "结果"

    def _sync_tab_states(self, status=None) -> None:
        """在下一步所在的模块页签上标一个绿色箭头，当前就在那一页时不标。

        流程条整行删掉之后，"现在该去哪一页"由页签自己说：只标一个页签，
        用图标而不改页签文字（文字要参与中英切换，动态拼接会被翻译缓存错认）。

        原来标的是蓝色圆点，和当前页的蓝色下划线同色：当前页同时带圆点和
        下划线，求解后「结果」页没选中也带圆点，两种"这里"读起来分不开。
        现在下一步是绿色箭头，当前页只靠下划线，两种含义各用各的记号。
        """
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
        if status is not None:
            self._next_status = status
        status = getattr(self, "_next_status", None)
        if status is None:
            return
        target = self.next_page(status)
        if not hasattr(self, "_next_marker"):
            pixmap = QPixmap(24, 24)            # 按 2 倍像素画，高缩放屏上不发虚
            pixmap.setDevicePixelRatio(2.0)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # 带杆的箭头读作"去那里"；单个实心三角容易被看成"运行"按钮
            pen = QPen(QColor(theme.SUCCESS), 1.8)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(1.5, 6), QPointF(7, 6))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.SUCCESS))
            painter.drawPolygon(QPolygonF([QPointF(6, 2), QPointF(11, 6), QPointF(6, 10)]))
            painter.end()
            self._next_marker = QIcon(pixmap)
        tip = self.workflow_bar.next_button.toolTip()
        current = self.ribbon.currentWidget()
        for title, page in self.ribbon.pages.items():
            index = self.ribbon.indexOf(page)
            marked = title == target and page is not current
            self.ribbon.setTabIcon(index, self._next_marker if marked else QIcon())
            self.ribbon.setTabToolTip(index, f"下一步：{tip}" if title == target else "")

    def _on_ribbon_page_changed(self, index: int) -> None:
        """把当前模块的下一步操作写出来，避免用户在功能区里猜流程。"""
        page = self.ribbon.tabText(index)
        self._sync_tab_states()
        self.quickbar.show_context_for(page, self.mode)
        hints = {
            "项目": "项目：新建或打开模型；建模过程可随时撤销。",
            "建模": "建模：创建几何后，用“选择节点/杆件”进入对象选择。",
            "属性": "属性：选择杆件，创建或指派材料与截面。",
            "装配": "装配：核对端点编号与重复几何，建立可复用的命名集合。",
            "连接": "连接：按局部轴定义杆端刚接、释放或转动弹簧，并复核实际构造。",
            "网格": "网格：预览物理杆件与分析杆段映射；局部实体网格在整体求解后设置。",
            "作业": "作业：先检查模型与约束，再提交独立工况或声明分析步，并查看运行状态。",
            "载荷": "载荷：选择节点创建边界条件，或选择杆件/节点施加载荷。",
            "分析": "分析：选择分析过程，或定义有传播关系的分析步及幅值曲线。",
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
                self._show_ribbon_page("作业")
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
        from PySide6.QtWidgets import QToolButton

        menus = (("文件", ("new", "open", "save", None, "report", "learning_trace")),
                 ("建模", ("modeling_workflow", "sketch", "frame", "portal", None,
                           "model_node", "model_member", "delete", "brace", None,
                           "tables", "json")),
                 ("属性", ("material", "beam_section", "section", None,
                           "assign_section", "batch_assignment", "beam_orientation", "units")),
                 ("装配", ("assembly_check", "model_sets")),
                 ("连接", ("member_connections", "hinge")),
                 ("作业", ("model_check", "support_diagnosis", "solve", "solve_steps", "stop_task")),
                 ("载荷", ("create_bc", "bc_manager", None,
                           "create_load", "gravity", None,
                           "load", "combo")),
                 ("分析", ("solve", None, "modal", "buckling", "solid_joint", None,
                           "diagnose", "analysis_mesh")),
                 ("结果", ("model", "deformed", "contour", "force_diagram",
                           "utilization", "solid_contour", "open_solid_result",
                           "diagram", None,
                           "envelope", "clear_results", "labels")),
                 ("视图", ("iso", "front", "side", "top", "fit", None,
                           "bg_settings", "grid_floor", "labels", "load_labels",
                           "camera", None, "lang", None, "chat", "props")))
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
        self.menu_button.setText(f"{glyphs.MENU}  菜单")
        self.menu_button.setToolTip("文件、建模、分析、结果与视图菜单")
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_button.setMenu(root)
        self.ribbon.setCornerWidget(
            self.menu_button, Qt.Corner.TopLeftCorner)
        self.menuBar().hide()

    def _build_statusbar(self) -> None:
        from .qt_style import ElidedLabel
        self.setStatusBar(QStatusBar(self))
        self.statusBar().setSizeGripEnabled(False)

        # Abaqus 式提示区：显示当前操作需要做什么（纯文字，不加背景条）
        self.lbl_prompt = ElidedLabel("就绪 | 选择上方功能区模块开始建模")
        self.lbl_prompt.setMinimumWidth(0)
        self.lbl_prompt.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_prompt.setStyleSheet(
            f"color:{theme.ACCENT}; font-weight:600; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_prompt, 1)

        # 分隔线
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet(f"color: {theme.BORDER};")
        self.statusBar().addPermanentWidget(sep1)

        # 状态信息 - 模型状态
        self.lbl_model = ElidedLabel("模型：空")
        self.lbl_model.setMaximumWidth(210)
        self.lbl_model.setStyleSheet(f"color: {theme.INK_MUTED}; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_model)

        # 状态信息 - 求解状态
        self.lbl_solve = ElidedLabel("求解：未求解")
        self.lbl_solve.setMaximumWidth(260)
        self.lbl_solve.setStyleSheet(f"color: {theme.INK_MUTED}; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_solve)

        # 状态信息 - 拾取状态
        self.lbl_pick = ElidedLabel("")
        self.lbl_pick.setMaximumWidth(130)
        self.lbl_pick.setStyleSheet(f"color: {theme.WARN}; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_pick)

        # 分隔线
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"color: {theme.BORDER};")
        self.statusBar().addPermanentWidget(sep2)

        # 最近操作
        self.lbl_last = ElidedLabel("最近：尚未建模")
        self.lbl_last.setMaximumWidth(220)
        self.lbl_last.setToolTip("最近一次成功或失败的建模操作")
        self.lbl_last.setStyleSheet(f"color: {theme.INK_DIM}; padding: 1px 6px;")
        self.statusBar().addPermanentWidget(self.lbl_last)
        self.btn_cancel_task = QPushButton("停止任务", self)
        self.btn_cancel_task.setVisible(False)
        self.btn_cancel_task.clicked.connect(self.stop_task)
        self.statusBar().addPermanentWidget(self.btn_cancel_task)

        self._on_ribbon_page_changed(self.ribbon.currentIndex())

    def set_prompt(self, text: str) -> None:
        """设置提示区文字（Abaqus 式操作提示）。"""
        self.lbl_prompt.setText(f"  {text}")

    # --- 状态 ---

    def set_mode(self, name: str, redraw: bool = True) -> None:
        """切换显示模式。紧接着要 refresh() 的调用方传 redraw=False，
        免得同一帧画两遍——大模型上一遍就是几百毫秒。"""
        if self.mode == "实体云图" and name != "实体云图":
            self.viewport.clear()
            self.solid_dock.hide()
        self.mode = name
        if name == "实体云图":
            self.viewport.set_model_mode(None)
            self.viewport.set_pick_mode(None)
            for key in ("model_node", "model_member", "pick_node", "pick_member"):
                self.actions_by_name[key].setChecked(False)
            self.solid_dock.show()
            self.empty_state.hide()
        # 中段控件也要跟着换：看变形图时要的是工况和放大，不是建节点
        if hasattr(self, "quickbar"):
            self.quickbar.show_context_for(
                self.ribbon.tabText(self.ribbon.currentIndex()), name)
        if name in {"变形", "云图", "内力图", "应力比"} and self.session.solution is not None:
            self.viewport.set_pick_mode("member")
        act = self.mode_actions.get(name)
        if act is not None:
            act.setChecked(True)
        if redraw:
            self.redraw()

    def offer_autosave_restore(self) -> bool:
        """启动时：上次有没保存的模型就问要不要恢复。只在程序入口调用，
        不放进构造函数——测试和对话里新建窗口不该弹这个问题。"""
        pending = self.autosave.pending()
        if pending is None or self.session.model.get("nodes"):
            return False
        model = pending["model"]
        count = f"{len(model.get('nodes') or [])} 个节点、{len(model.get('members') or [])} 根杆件"
        answer = QMessageBox.question(
            self, "恢复未保存的模型",
            f"上次关闭时有一个没保存的模型（{count}，{pending.get('saved_at', '')}）。\n"
            "要恢复它吗？选「否」会丢弃这份自动存档。")
        if answer != QMessageBox.StandardButton.Yes:
            self.autosave.discard()
            return False
        # 不走 set_model：它要求模型完整（材料、截面、支座齐全），而自动存档
        # 最该救回来的恰恰是还没建完的半成品——原样恢复，缺什么界面会接着提示
        import copy
        session = Session(model=copy.deepcopy(model))
        self._replace_session(session)
        self.result = self.case = None
        self.refresh()
        self.statusBar().showMessage("已恢复上次未保存的模型，记得保存。", 8000)
        return True

    def new_model(self) -> None:
        self._replace_session(Session())
        self.result = None
        self.case = None
        self.refresh()

    def _replace_session(self, session: Session) -> None:
        """让所有会修改模型的面板指向同一个新会话。"""
        self.session = session
        self._modeling_reviews = {}
        self._analysis_mesh_key = None
        self._selected_kind = self._selected_id = None
        self.viewport.set_selection(None, None)
        self.viewport.set_problem_refs([])
        self._last_probe = None
        self.chat.reset_session(session)
        self.properties.session = session
        self.properties.clear()
        self.sketch.session = session
        self.section_opt.session = session
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

    def solve_declared_steps(self) -> None:
        """声明分析步使用同一个后台任务及结果生效路径。"""
        if self.runner.busy:
            return
        if not self.session.model.get("steps"):
            QMessageBox.information(self, "尚无分析步", "请先在分析步管理器创建步骤；独立工况可使用求解。")
            return
        self._on_busy(True)
        self.statusBar().showMessage("分析步求解中…")
        if not self.runner.submit(self.session.solve_steps, on_done=self._steps_solved,
                                  on_failed=self._solve_failed):
            self._on_busy(False)

    def _steps_solved(self, result) -> None:
        if not result.ok:
            self._solved(result)
            return
        self._solved(self.session.solve_summary)
        self.statusBar().showMessage(
            f"已完成 {result.payload['count']} 个分析步；当前查看 {result.payload['inspecting']}。"
            + result.payload["limitation"], 15000)

    def _solved(self, result) -> None:
        self.result = result
        self.result_frame = None
        self._on_busy(False)
        self.statusBar().clearMessage()
        if result.ok:
            cases = list(result.payload["cases"])
            self.case = self.case if self.case in cases else cases[0]
            self.set_mode("变形", redraw=False)          # 末尾的 refresh 会画
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
                        f"点右侧「结果」查看详情。", 15000)
                else:
                    self.statusBar().showMessage(
                        f"静默检测全部 {len(sf['findings'])} 项通过", 8000)
            # 总览先填好，不弹出——想看数字时点右侧「结果」就在那儿
            caption, cols, rows, loc = result_rows.to_rows("solve", result.payload)
            self.results.show_rows(caption, cols, rows, loc,
                                   result_rows.row_marks("solve", result.payload))
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
        if self.result is None and self.session.solution is None and self._solid_grid is None:
            self.statusBar().showMessage("当前没有需要清除的计算结果。", 4000)
            return
        self.result = None
        self.case = None
        self.session.solution = None
        self._solid_grid = self._solid_binding = self._solid_info = None
        self.clear_solid_query()
        self.solid_panel.caption.setText("实体结果已清除，请重新分析或打开已保存结果。")
        self.solid_panel.levels.clear()
        self.solid_panel.source.clear()
        self.results.show_message("计算结果已清除", "模型、材料、约束与荷载均未修改，可重新求解。")
        self.set_mode("模型")
        self.refresh()
        self.statusBar().showMessage("计算结果已清除，模型保持不变。", 5000)

    def _solve_failed(self, kind: str, message: str) -> None:
        """求解器自己抛异常。守门拦不住的（比如奇异矩阵）会走到这里。"""
        self._on_busy(False)
        self.statusBar().clearMessage()
        if kind == "TaskCancelled":
            self.statusBar().showMessage(message, 8000)
            self.refresh()
            return
        QMessageBox.critical(self, "求解过程出错", f"{kind}: {message[:600]}")

    # --- Agent 面板 ---

    def _on_busy(self, busy: bool) -> None:
        """跑着活儿的时候把会改模型的动作置灰。

        不置灰的话，用户在求解途中点「新建」，后台线程正在读的那个
        Session 就被换掉了——**这类竞态出来的错误现场根本没法复现**。
        """
        busy = bool(busy or self.runner.busy)
        idle = not busy
        self.chat.set_task_busy(busy)
        if busy and not self._task_busy:
            self._busy_action_states = {
                name: action.isEnabled() for name, action in self.actions_by_name.items()
                if name not in self.BUSY_ALLOWED}
            self._pending_command = None
            self.viewport.set_model_mode(None)
        for name, enabled in self._busy_action_states.items():
            self.actions_by_name[name].setEnabled(idle and enabled)
        self._task_busy = busy
        self.act_solve.setEnabled(idle and bool(self.session.model.get("nodes")))
        self.actions_by_name["stop_task"].setEnabled(busy)
        self.btn_cancel_task.setVisible(busy)
        self.btn_cancel_task.setEnabled(busy)
        for widget in (self.workflow_bar, self.quickbar, self.tree, self.properties,
                       self.timeline, self.sketch, self.section_opt,
                       self.diagram, self.results, self.empty_state, self.solid_panel):
            widget.setEnabled(idle)
        if idle and self._close_pending:
            QTimer.singleShot(0, self.close)

    def stop_task(self) -> None:
        """请求停止当前任务，等安全检查点到达后再恢复编辑。"""
        if self.chat.conversation is not None:
            self.chat.conversation.cancel()
        if self.runner.cancel():
            self.btn_cancel_task.setEnabled(False)
            self.actions_by_name["stop_task"].setEnabled(False)
            self.statusBar().showMessage(
                "正在停止任务；将于安全检查点结束，矩阵分解完成前请等待。")

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
            self.set_mode("变形", redraw=False)          # 调用方随后都会 refresh

    def _demo_prompt(self) -> None:
        self.chat_dock.setVisible(True)
        self.chat.load_demo_prompt()

    # --- 绘制 ---

    def _sync_datums(self) -> None:
        self.viewport.datum_points = self.session.datum_points()
        self.viewport.constraint_links = self.session.constraint_links()

    def redraw(self) -> None:
        # 重画会清掉视口（也就停了动画），开关的勾要跟着取消，免得显示「在播」实际没播
        self._sync_datums()
        animate = getattr(self, "actions_by_name", {}).get("animate")
        if animate is not None:
            animate.setChecked(False)
        self._redraw()

    def _sync_element_owner(self, frame) -> None:
        """视口要画的是求解用的分析模型时，告诉它分析杆段属于哪根物理杆件。

        求解过之后，连「模型」显示画的也是分析模型（preview_frame 直接返回它）；
        不同步的话，点选、显示组拿到的是杆段号，隐藏一根柱只隐藏了它的一段。
        """
        compilation = self.session.compilation
        self.viewport.element_owner = (
            {e: compilation.mapping.physical_member_id(e) for e in frame.members}
            if compilation is not None and frame is self.session.frame else {})

    def _redraw(self) -> None:
        self._sync_result_validity()
        if self.mode == "实体云图" and self._solid_grid is not None:
            self.viewport.show_solid_result(
                self._solid_grid, self._solid_info[-1], **self.solid_panel.options(),
                reset_camera=self._solid_reset_camera, query=self.solid_panel.query.isChecked())
            self._solid_reset_camera = False
            return
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
        self._sync_element_owner(frame)

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
        self._sync_element_owner(frame)
        if self.mode == "变形":
            scale = self.scale or self._auto_scale(frame)
            self.viewport.show_deformed(frame, self._frame_solution(),
                                        self.case, scale)
        elif self.mode == "云图":
            options = self.result_display_options
            try:
                self.viewport.show_contour(
                    frame, self.session.solution, self.case, self.component,
                    percentile=options["percentile"],
                    sign_filter=options["sign"],
                    levels=options.get("levels"),
                    overlay_deformed=options["overlay_deformed"],
                    show_extrema=options["show_extrema"])
            except StressUnavailable as exc:
                # 应力云图要截面的极端纤维距离。**明确拒绝并退回内力云图**，
                # 比画一张来路不明的应力图安全；也别把用户卡在一个空视口里。
                QMessageBox.information(self, "无法绘制应力云图", str(exc))
                self.component = "Mz"
                self.viewport.show_contour(
                    frame, self.session.solution, self.case, self.component,
                    percentile=options["percentile"],
                    sign_filter=options["sign"],
                    levels=options.get("levels"),
                    overlay_deformed=options["overlay_deformed"],
                    show_extrema=options["show_extrema"])
        elif self.mode == "应力比":
            cached = self._utilization
            if cached is not None and cached[0] is self.session.solution:
                self.viewport.show_utilization(frame, self.session.compilation.mapping,
                                               cached[1])
            else:
                # 结果换过（重新求解）而验算还是上一次的：先画模型，后台重算
                self.viewport.show_model(frame, self.case)
                self._utilization_timer.start(0)
        elif self.mode == "内力图":
            component = self.component
            if component == scene.STRESS:
                # 应力没有"受拉侧"这回事，也不是一个沿杆可画成形状的内力
                component = "M"
            self.viewport.show_force_diagram(frame, self.session.solution,
                                             self.case, component)
        elif self.mode == "模态":
            # 同一个 Frame 只算一次。原来每次重画（切工况、改放大系数、开关
            # 标注）都把模态分析完整跑**两遍**——校验一遍、画图再一遍。
            #
            # 振型画在 Session 算模态时用的**那个**分析模型上（每杆至少剖成
            # 4 段），不能拿静力的 Frame 再算一遍：剖分不同，图上的频率就和
            # 结果表里的对不上。
            cached = self.session.modal_view
            if cached is None or cached[0] != self.session._model_key():
                got = self.session.modal_analysis(num_modes=6)
                if not got.ok:
                    QMessageBox.information(self, "无法进行模态分析",
                                            str(got.payload.get("error")
                                                or got.payload.get("errors", ""))[:400])
                    self.set_mode("模型")
                    return
                cached = self.session.modal_view
            frame, r = cached[2], cached[3]
            f1 = r.frequencies[0]
            self.viewport.show_mode(frame, r.shapes, 0,
                                    f"第 1 阶　{f1:.3f} Hz　周期 {1 / f1:.4f} s")

    def _auto_scale(self, frame) -> float:
        return scene.auto_deformation_scale(frame, self.session.solution,
                                            self.case)

    def refresh(self) -> None:
        self._sync_result_validity()
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
        if len(self.session.history):
            last = self.session.history[len(self.session.history) - 1]
            self.lbl_last.setText(f"最近：第 {last.index + 1} 步　{last.summary}")
        else:
            self.lbl_last.setText("最近：尚未建模")
        self.act_solve.setEnabled(not self.runner.busy and bool(model.get("nodes")))
        self.workflow_bar.update_state(self.session)
        # 工具栏改了模型，对话面板的后续建议也要跟着换（例如求解后改为问位移）
        self.chat.refresh_suggestions()
        dialog = getattr(self, "_modeling_dialog", None)
        if dialog is not None and dialog.isVisible():
            dialog.refresh()
        self.redraw()
        has_model = bool(model.get("nodes"))
        self._sync_model_tree(has_model)
        self.empty_state.setVisible(not has_model and self.mode != "实体云图")
        if self.empty_state.isVisible():
            self._position_empty_state()
        self.autosave.schedule()

    def _sync_result_validity(self) -> None:
        """模型修改后清空表、探针、曲线与云图，共用会话内容校验。"""
        if self.runner.busy:
            return
        if self._solid_binding is not None and (
                self._solid_binding[0] is not self.session.solution
                or self._solid_binding[1] != self.session.result_identity
                or self.session.result_error()):
            self._solid_grid = self._solid_binding = self._solid_info = None
            self.clear_solid_query()
            self.solid_panel.caption.setText("实体结果已失效，请重新求解并进行节点实体分析。")
            self.solid_panel.levels.clear()
            self.solid_panel.source.clear()
            self.results.show_message("实体结果已失效", "整体求解版本已变化，请重新进行节点实体分析。")
            if self.mode == "实体云图":
                self.set_mode("模型", redraw=False)
        identity = self.result.payload.get("result_identity") if self.result else None
        stale = self.session.solution is not None and self.session.result_error()
        changed = (identity is not None and identity != self.session.result_identity)
        if self._displayed_solution is not self.session.solution:
            self._displayed_solution = self.session.solution
            self._last_probe = None
            self._utilization = None
            self.diagram.attach(self.session, self.case)
            self.viewport.set_result_marker(None)
        if stale or changed or (self.result is not None and self.result.ok
                                and self.session.solution is None):
            self.result = None
            self.case = None
            self._last_probe = None
            self._utilization = None
            if self.mode != "实体云图":
                self.set_mode("模型", redraw=False)
            self.results.show_message("结果已失效", "模型或求解版本已变化，请重新求解。")
            self.diagram.attach(self.session, None)
            self.viewport.set_result_marker(None)

    def _sync_model_tree(self, has_model: bool) -> None:
        """Give an empty viewport its width back, then reveal real model data."""
        if not has_model:
            if not self.tree_dock.isHidden():
                self.tree_dock.hide()
            self._tree_auto_hidden = True
        elif self._tree_auto_hidden:
            self._tree_auto_hidden = False
            if not self._tree_dismissed:
                self.tree_dock.show()




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
        result = self.session.replace_editing_model(model=dialog.model)
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
            self._last_probe = None
            self.viewport.set_result_marker(None)
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
        if mode and self.mode == "实体云图":
            self.set_mode("模型")
        self.viewport.set_pick_mode(mode)
        if mode == "node":
            self.set_prompt("选择节点模式：在视口中点击节点选中，可用于创建边界条件/载荷/查看属性")
        elif mode == "member":
            self.set_prompt("选择杆件模式：在视口中点击杆件选中，可用于截面指派/施加荷载/查看属性")
        else:
            self.set_prompt("就绪 | 选择上方功能区模块开始建模")

    def toggle_render_profiles(self) -> None:
        self.viewport.set_render_profiles(self.actions_by_name["render_profiles"].isChecked())
        self.redraw()

    def toggle_labels(self) -> None:
        on = self.actions_by_name["labels"].isChecked()
        self.viewport.set_labels(on)

    def toggle_load_labels(self) -> None:
        """荷载箭头旁边标不标数值。

        模型一大，几十个数字糊成一片，遮住的正是它们要说明的那根杆件。
        关掉之后箭头还在——方向和分布照样看得见，只是不再有字。
        """
        on = self.actions_by_name["load_labels"].isChecked()
        if self.viewport.set_load_labels(on):
            self.redraw()
        self.statusBar().showMessage(
            "荷载数值：显示" if on else "荷载数值：隐藏（箭头仍在）", 3000)

    def toggle_language(self) -> None:
        """中英文界面切换。

        勾上是英文。翻的只有界面骨架——功能区、菜单、面板标题、状态提示；
        对话框正文和报错仍是中文，这一点在命令提示里就写明了，
        不让用户切过去之后才发现只翻了一半。
        """
        from . import i18n

        want = "en" if self.actions_by_name["lang"].isChecked() else "zh"
        if not i18n.set_language(want):
            return
        self.menu_button.setText(i18n.tr(f"{glyphs.MENU}  菜单"))
        # 抽屉只翻页签与标题、窄栏只翻按钮——抽屉**里面**各面板的正文不翻，
        # 理由同 i18n.retranslate：翻一半比不翻更难用。
        drawers = self.drawers.drawers.values()
        i18n.retranslate(
            (self.ribbon, self.quickbar, self.workflow_bar, self.results,
             self.ribbon.corner, *(d.tabs for d in drawers)),
            self.actions_by_name.values())
        # 抽屉标题随当前页变化，不能交给 retranslate（它把第一次见到的文字
        # 当作原文缓存下来）；让抽屉按当前语言重写一次
        for drawer in drawers:
            drawer._sync_header()
        # 流程条自己翻自己（步骤名后面还挂着算出来的 ✓ / !3），
        # 所以切完语言要让它按当前模型状态重画一次。
        self.workflow_bar.update_state(self.session)
        self.statusBar().showMessage(
            "Interface language: English (dialogs remain in Chinese)"
            if want == "en" else "界面语言：中文", 4000)
        self.redraw()

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
        if mode and self.mode == "实体云图":
            self.set_mode("模型")
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
        if self.viewport._solid_query_enabled:
            self.solid_panel.query.setChecked(False)
            self.statusBar().showMessage("已退出实体点选，可继续旋转和缩放。", 3000)
            return
        changed = False
        region = getattr(self, "region_pick", None)
        if region is not None and region.active:
            region.cancel()
            changed = True
        for name in ("model_node", "model_member", "pick_node", "pick_member"):
            act = self.actions_by_name.get(name)
            if act is not None and act.isChecked():
                act.setChecked(False)
                changed = True
        # 由这两个统一同步视口模式与提示
        self._apply_model_mode()
        self._apply_pick_mode()
        self.viewport.set_selection(None, None)
        if getattr(self, "_pending_command", None):
            self._pending_command = None
            changed = True
        if changed:
            self.statusBar().showMessage("已退出当前操作，回到浏览", 3000)
        elif self.drawers.close_all():
            self.statusBar().showMessage("已收起面板", 2000)

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
        from PySide6.QtWidgets import (QDialog, QFormLayout,
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
        btns = dialog_styles.button_box(dlg)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        xyz = [float(spins[a].value()) for a in ("x", "y", "z")]
        self._create_node_at(xyz)

    def _create_node_at(self, xyz) -> None:
        """统一处理鼠标与精确输入建点，包括已有坐标复用。"""
        if self.runner.busy:
            return
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
        if self.runner.busy:
            return
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
        if kind in {"solid_cell", "solid_point"}:
            # 实体查询表里点行：在实体网格上高亮原单元或节点，结果表保留原查询
            if self._solid_grid is not None:
                if self.mode != "实体云图":
                    self.set_mode("实体云图")
                association = "cell" if kind == "solid_cell" else "point"
                count = (self._solid_grid.n_cells if association == "cell"
                         else self._solid_grid.n_points)
                if 1 <= ident <= count:
                    self.viewport.select_solid(association, ident - 1, focus=True)
                    label = "单元" if association == "cell" else "节点"
                    self.statusBar().showMessage(
                        f"已定位实体{label}序号 {ident}，结果表保留原查询。", 4000)
            return
        if self.mode == "实体云图":
            self.set_mode("模型")
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
        self.chat.set_selection(kind, ident)
        self._selected_kind = kind
        self._selected_id = ident
        if self.viewport.selection != (kind, ident):
            self.viewport.set_selection(kind, ident)
        self._sync_selection_actions()
        self._describe_selection(kind, ident)
        # **选中就显示属性。** 这一步是把"拾取"这条路走完：
        # 之前点中一根杆只能看到它多长，改不了
        self.properties.show_object(kind, ident)
        self.tree.select_object(kind, ident)
        # 建模时点中就把属性摆出来，这是"拾取"这条路的终点；看结果时点选是
        # 在查数（值写进结果表与状态栏），再弹一个抽屉只会把模型又挤掉一块。
        # 左抽屉本来就开着时照常切到属性页。
        if self.mode in ("模型", "分析网格") and not self._tree_dismissed:
            self.props_dock.show()
        # 点中一根杆，内力图就切到它——**这是"在哪"和"多大"接起来的一步**
        if kind == "member" and self.diagram_dock.isVisible():
            self.diagram.member.setCurrentText(str(ident))
        self._resume_pending_command(kind)

    def _on_selection_cleared(self) -> None:
        """视口空白处单击：取消选择，属性区与右键菜单都回到无选择状态。"""
        self._selected_kind = None
        self._selected_id = None
        self.chat.set_selection(None, None)
        self.properties.clear()
        self.lbl_pick.setText("")
        self._sync_selection_actions()

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
        """Abaqus 的 Gravity 载荷：走创建载荷流程，类型预选重力，区域默认全部杆件。

        原来这里直接写逐杆的自重条目，和「创建载荷 → 重力」得到的荷载对象是两种
        东西：一个能在载荷管理器里编辑、抑制，另一个只能去表格里删。
        """
        if not self.session.model.get("members"):
            self.statusBar().showMessage("当前没有杆件，无法计算自重。", 5000)
            return
        self.create_load(load_type="gravity")

    def show_bc_panel(self) -> None:
        """旧入口名：边界条件统一在边界条件管理器里查看与编辑。"""
        self.open_bc_manager()

    def edit_materials_sections(self) -> None:
        """弹出材料与截面编辑对话框，直接修改模型的材料和截面定义。"""
        from PySide6.QtWidgets import (QDialog, QTabWidget,
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
        btns = dialog_styles.button_box(dlg)
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
        from PySide6.QtWidgets import QDialog, QComboBox, QFormLayout
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
        btns = dialog_styles.button_box(dlg)
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
        from PySide6.QtWidgets import (QDialog, QTableWidget,
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
        btns = dialog_styles.button_box(dlg)
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

    def open_modeling_workflow(self) -> None:
        from .modeling_workflow import ModelingWorkflowDialog
        if getattr(self, "_modeling_dialog", None) is None:
            self._modeling_dialog = ModelingWorkflowDialog(self)
        self._modeling_dialog.refresh()
        self._modeling_dialog.show()
        self._modeling_dialog.raise_()
        self._modeling_dialog.activateWindow()

    def check_manual_model(self) -> None:
        self._on_workflow_stage("validate")

    def check_assembly(self) -> None:
        from .modeling_workflow import assembly_issues
        issues = assembly_issues(self.session)
        self.results_dock.show()
        if not self.session.model.get("members"):
            self.results.show_message("装配尚未定义", "请先创建节点和杆件，再检查装配。")
        elif issues:
            self.results.show_rows("装配检查发现问题，请修正后重新检查。",
                                   ["对象", "编号", "原因与下一步"],
                                   [["节点" if kind == "node" else "杆件", ident, text]
                                    for kind, ident, text in issues],
                                   [(kind, ident) for kind, ident, _ in issues])
        else:
            self.results.show_message("装配几何检查完成",
                                      "未检出重合节点、重复杆件或端点错误。请复核相交杆件是否需要共用节点；本项不证明结构稳定。")

    def open_model_sets(self) -> None:
        from .assignment_dialog import SetDialog
        selected = self._selected_id if getattr(self, "_selected_kind", None) == "member" else None
        dialog = SetDialog(self.session, selected, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = self.session.define_set(**dialog.definition())
            if not result.ok:
                QMessageBox.warning(self, "集合未创建", self._explain(result.payload))
                return
            self._after_manual_edit("命名集合已更新，可用于批量属性与连接设置。")

    def assign_member_group(self) -> None:
        from .assignment_dialog import BatchAssignmentDialog
        if not self.session.model.get("materials") or not self.session.model.get("sections"):
            self.statusBar().showMessage("请先创建材料和截面，再进行批量属性指派。", 5000)
            self._show_ribbon_page("属性")
            return
        selected = self._selected_id if getattr(self, "_selected_kind", None) == "member" else None
        dialog = BatchAssignmentDialog(self.session, selected, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        section, material = dialog.get_assignment()
        # 与「截面指派」同一种对象：原来这里写的是逐杆的直接条目，管理器里看不见
        from .object_dialogs import next_name
        name = next_name("SA", self._feature_names("section_assignments"))
        result = self.session.create_section_assignment(name, dialog.target.member_ids(),
                                                        section, material)
        if not result.ok:
            QMessageBox.warning(self, "属性未指派", self._explain(result.payload))
            return
        self._after_manual_edit(f"已创建截面指派「{name}」。")

    def _configure_member_group(self, orientation=False) -> None:
        from .assignment_dialog import MemberSetupDialog
        selected = self._selected_id if getattr(self, "_selected_kind", None) == "member" else None
        dialog = MemberSetupDialog(self.session, selected, orientation=orientation, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = self.session.configure_member_group(dialog.target.member_ids(), **dialog.changes())
        if not result.ok:
            QMessageBox.warning(self, "杆件设置未提交", self._explain(result.payload))
            return
        if not result.payload.get("no_change"):
            self._after_manual_edit("梁截面方向已更新。" if orientation else "杆端连接已更新，请复核机构与传力路径。")

    def set_member_connections(self) -> None:
        self._configure_member_group()

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
        # name / A / Iy / Iz / J 这一套。把它们写进去会违反 JSON Schema，
        # 直到下一次求解才暴露出来，用户会误以为是求解器出了问题。
        #
        # 反过来，这份白名单**漏掉一个键就是静默丢数据**：cy/cz/circular
        # 一丢，界面上建的截面就永远做不了强度验算，而且不报错——
        # 只在点「强度验算」时被拒绝，看着像校核功能坏了。geometry 同理：
        # 漏掉它，非圆截面就没有真实轮廓，节点实体只能拒绝、视口也画不出截面。
        section_type = raw_section["type"]
        sec = {key: raw_section[key] for key in
               ("name", "A", "Iy", "Iz", "J", "Ay", "Az", "cy", "cz", "circular", "geometry")
               if key in raw_section}
        from copy import deepcopy

        new_sections = deepcopy(self.session.model.get("sections", []))
        index = next((i for i, s in enumerate(new_sections)
                      if s["name"] == sec["name"]), None)
        # 同名截面整条替换而不是 update：截面的每个字段都由尺寸推出，旧的
        # geometry、cy/cz、Ay/Az 或剪应力静矩若残留，会与新尺寸对不上。
        if index is not None:
            new_sections[index] = sec
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

    def open_code_combinations(self) -> None:
        """按规范生成荷载组合。"""
        from .load_code_dialog import CombinationDialog

        CombinationDialog(self.session, self).exec()
        self.refresh()

    def open_live_patterns(self) -> None:
        """活载最不利布置。"""
        from .load_code_dialog import LivePatternDialog

        LivePatternDialog(self.session, self).exec()
        self.refresh()

    def open_area_load(self) -> None:
        """面荷载导成线荷载。"""
        from .load_code_dialog import AreaLoadDialog

        AreaLoadDialog(self.session, self).exec()
        self.refresh()

    def open_step_manager(self) -> None:
        """分析步管理器。内核早就支持跨步传播与失活，界面一直够不到。"""
        from .step_dialog import StepManagerDialog

        dialog = StepManagerDialog(self.session, self)
        dialog.exec()
        self.refresh()

    def open_amplitude_manager(self) -> None:
        """幅值曲线管理器。"""
        from .step_dialog import AmplitudeDialog

        AmplitudeDialog(self.session, self).exec()

    def edit_analysis_step(self) -> None:
        """选求解过程和增量数。

        **这不是分析步管理器**——它只设置一次普通求解用哪种过程。真正的
        分析步（可增删、跨步传播、失活）在 open_step_manager 里。
        """
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

    # 这些命令必须先在视口里选中对象才有意义。
    # 键是命令名，值是它接受的选择类型。
    _NEEDS_SELECTION = {
        "solid_joint": ("node",),
    }

    def _sync_selection_actions(self) -> None:
        """这些命令**始终可点**，只把"它要作用在谁身上"写进提示。

        上一版是没选中就置灰。置灰比"点了没反应"好，但仍然要求用户先猜对
        顺序：先去找「选择节点」、点中一个、再回来点命令。实测最常见的抱怨
        恰恰是"很多按钮点不了"。CAE 的通行做法（Abaqus 的提示区、PrePoMax
        的选择模式）是**命令在前、拾取在后**：点了命令，它告诉你去点什么，
        点中之后自动接着走。见 `_require_selection`。
        """
        kind = getattr(self, "_selected_kind", None)
        for name, accepted in self._NEEDS_SELECTION.items():
            action = self.actions_by_name.get(name)
            if action is None:
                continue
            action.setEnabled(not self.runner.busy)
            what = "节点或杆件" if len(accepted) > 1 else "节点"
            action.setToolTip(action.text() if kind in accepted
                              else f"{action.text()}：点击后在视口中点选一个{what}")

    def _require_selection(self, handler: str, accepted: tuple[str, ...],
                           label: str) -> bool:
        """命令需要一个选中对象。有就放行；没有就进入拾取，点中后自动续上。"""
        if getattr(self, "_selected_kind", None) in accepted:
            self._pending_command = None
            return True
        self._pending_command = (handler, accepted, label)
        current = next((k for k, a in self.pick_actions.items() if a.isChecked()), None)
        # 已经在合适的拾取模式里就沿用；否则取最后一个（荷载默认拾杆件）
        mode = current if current in accepted else accepted[-1]
        self.pick_actions[mode].setChecked(True)
        self.viewport.set_pick_mode(mode)
        what = "节点或杆件" if len(accepted) > 1 else "节点"
        self.set_prompt(f"{label}：请在视口中点选一个{what}（Esc 取消）")
        self.statusBar().showMessage(f"{label}：在视口中点选一个{what}", 6000)
        return False

    def _resume_pending_command(self, kind: str) -> None:
        pending = getattr(self, "_pending_command", None)
        if not pending or kind not in pending[1]:
            return
        self._pending_command = None
        self.set_prompt(f"{pending[2]}：已选中，填写参数")
        # 等这次拾取事件走完再弹对话框，否则 VTK 的鼠标状态会卡在按下
        QTimer.singleShot(0, getattr(self, pending[0]))

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

    def _recognize_dropped_image(self, path: str) -> None:
        """对话面板拖入的图纸：打开草图识别面板并载入，后续核对与确认仍由用户完成。"""
        if not self.sketch_dock.isVisible():
            self.open_sketch_ai()
        self.sketch.load_image(path)

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
        except Exception as e:  # noqa: BLE001  同上，出图这一路；错误文本原样进对话框
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
        from PySide6.QtWidgets import (QDialog, QHBoxLayout,
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

        btns = dialog_styles.close_box(dlg)
        btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.exec()

    def toggle_agent_panel(self) -> None:
        """召唤/收起 AI 助手（右侧窄栏的「AI」按钮与 Ctrl+G 共用）。"""
        self.chat_dock.setVisible(not self.chat_dock.isVisible())
        if self.chat_dock.isVisible():
            self._on_drawer_page_opened("chat")
            self.chat.setFocus()
        else:
            self.set_prompt("就绪 | 选择上方功能区模块开始建模")

    def showEvent(self, event):
        super().showEvent(event)
        self.drawers.relayout()
        self._position_empty_state()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_empty_state()

    def toggle_props(self) -> None:
        on = self.actions_by_name["props"].isChecked()
        self.props_dock.setVisible(on)
        if on:
            self._on_drawer_page_opened("props")

    def toggle_tree(self) -> None:
        """模型栏（模型树 + 属性）开关。"""
        on = self.actions_by_name["tree_panel"].isChecked()
        self._tree_dismissed = not on
        self.tree_dock.setVisible(on)
        if on:
            self._on_drawer_page_opened("tree")

    def show_results_table(self) -> None:
        on = self.actions_by_name["results_table"].isChecked()
        self.results_dock.setVisible(on)
        if on:
            self.results_dock.raise_()

    def show_section_opt(self) -> None:
        on = self.actions_by_name["section_opt_panel"].isChecked()
        self.section_opt_dock.setVisible(on)
        if on:
            self.section_opt_dock.raise_()

    def open_command_palette(self) -> None:
        from .command_palette import CommandPalette
        CommandPalette(self).exec()

    def toggle_chat(self) -> None:
        """页签行右端与菜单里的「AI 助手」开关。"""
        if self.act_chat.isChecked() != self.chat_dock.isVisible():
            self.toggle_agent_panel()

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
        settings = _settings()
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
        settings = _settings()
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
            lambda: _settings().remove("recent_model_paths"))

    def _load_model_path(self, path: Path) -> bool:
        """从明确路径加载模型；文件选择器和“最近打开”共用这条安全路径。"""
        if self.runner.busy:
            self.statusBar().showMessage("后台任务执行中；请等待完成或停止任务后再打开模型。")
            return False
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
        self.autosave.mark_saved()
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
        from model_io import atomic_write_text
        from draft_commit import save_sidecar
        try:
            atomic_write_text(path, json.dumps(self.session.model,
                                               ensure_ascii=False, indent=2))
            save_sidecar(path, self.session.model, self.session.multimodal_provenance)
        except OSError as exc:
            QMessageBox.warning(self, "保存未完成",
                                f"文件写入失败：{exc}。请检查目录权限或另选保存位置。")
            return
        self._remember_recent_path(Path(path))
        self.autosave.mark_saved()
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
        self._analysis_mesh_key = self.session._model_key()
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

    def show_solid_contour(self) -> None:
        """切回实体主视口；无缓存时允许打开已有结果，避免重新计算。"""
        self._sync_result_validity()
        if self._solid_grid is None:
            self.open_solid_result()
            self.mode_actions[self.mode].setChecked(True)
            return
        self._solid_reset_camera = self.mode != "实体云图"
        self.set_mode("实体云图")

    def open_solid_result(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "打开实体结果", str(Path("results/native_solid_joint").resolve()),
            "节点实体结果 (*.json *.vtu *.npz)")
        if path:
            self._load_solid_result(path)

    def _load_solid_result(self, path, binding=None) -> bool:
        from .solid_result import read_result

        try:
            grid, summary, paths, index, caption = read_result(path)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            QMessageBox.warning(self, "实体结果无法打开",
                                f"结果文件读取失败：{exc}。请重新选择完整结果或重新进行节点实体分析。")
            return False
        self._solid_grid, self._solid_binding = grid, binding
        # 换档或换文件后单元序号对应的是另一套网格，旧查询必须撤掉
        self.clear_solid_query()
        self._solid_info = (summary, paths, index, caption)
        self.solid_panel.bind(paths, index, caption, saved=binding is None, geometry=summary.get("geometry"))
        if summary:
            title, columns, rows, locators = result_rows.to_rows("solid_joint", summary)
            self.results.show_rows(title, columns, rows,
                                   locators if binding is not None else [None] * len(rows))
        self._solid_reset_camera = True
        self.set_mode("实体云图")
        self.statusBar().showMessage("实体结果已打开，可旋转、缩放并在左侧切换结果量。", 6000)
        return True

    def _switch_solid_level(self, path) -> None:
        if not self._load_solid_result(path, self._solid_binding) and self._solid_info:
            summary, paths, index, caption = self._solid_info
            self.solid_panel.bind(paths, index, caption, saved=self._solid_binding is None, geometry=summary.get("geometry"))

    def _redraw_solid_result(self) -> None:
        if self.mode == "实体云图" and self._solid_grid is not None and not self.runner.busy:
            self.redraw()

    def show_utilization(self, interactive: bool = True) -> None:
        """应力比图。验算按物理构件逐根做，大模型要几秒，放后台跑；
        同一次求解只算一次，从「强度验算」按钮算过的结果也直接复用。

        ``interactive=False`` 是重画时自动补算（比如停在应力比上重新求解了）：
        算不了只在状态栏说一声并退回模型显示，**不弹模态框**——模态框从后台
        回调里弹出来，会和正在跑的事件循环嵌套，实测在测试里直接把进程带崩
        （0xc0000374 堆损坏）；用户也没有主动点什么，不该被一个对话框打断。"""
        if self._close_pending or not self._needs_solution():
            return
        cached = self._utilization
        if cached is not None and cached[0] is self.session.solution:
            self.set_mode("应力比")
            return
        if self.runner.busy:
            return
        solution = self.session.solution
        self._on_busy(True)
        self.statusBar().showMessage("应力比：强度与稳定验算中…")

        def done(result) -> None:
            self._on_busy(False)
            self.statusBar().clearMessage()
            if not result.ok:
                # 最常见的是材料没给许用应力、截面没给 cy/cz——说清楚去哪补
                why = self._explain(result.payload)
                if interactive:
                    QMessageBox.information(self, "无法显示应力比", why)
                else:
                    self.statusBar().showMessage(f"无法显示应力比：{why[:160]}", 10000)
                self.set_mode("模型" if self.mode == "应力比" else self.mode)
                return
            self._remember_strength(solution, result.payload)
            self.set_mode("应力比")

        def failed(kind: str, message: str) -> None:
            self._on_busy(False)
            self.statusBar().clearMessage()
            if kind == "TaskCancelled":
                self.statusBar().showMessage(message, 8000)
                return
            QMessageBox.critical(self, "应力比验算出错", f"{kind}：{message[:400]}")

        if not self.runner.submit(lambda: self.session.check_strength(),
                                  on_done=done, on_failed=failed):
            self._on_busy(False)

    def _remember_strength(self, solution, payload: dict) -> None:
        """记下某次求解的验算结果，并把逐杆表填进结果页（不弹出）。"""
        self._utilization = (solution, payload)
        caption, cols, rows, loc = result_rows.to_rows("strength", payload)
        self.results.show_rows(f"强度验算　{caption}", cols, rows, loc,
                               result_rows.row_marks("strength", payload),
                               overview=result_rows.strength_overview(payload),
                               primary_columns={"杆件", "截面", "应力比 σ/[σ]", "折算比", "N/(φA)/f", "结论"})

    def show_force_diagram(self) -> None:
        """三维内力图。不弹结果抽屉：图本身就是结果，别再挡住它。"""
        if self._needs_solution():
            self.set_mode("内力图")

    def pick_component(self) -> None:
        """选择梁中心线内力结果显示哪个分量。

        菜单里把每个分量是什么写出来——`My` 和 `Mz` 光看字母是分不出
        哪个是平面内弯矩的，而选错了看半天图都是白看。
        """
        labels = ((scene.STRESS, "极端纤维正应力 σ（N/A ± M·c/I，受拉为正）"),
                  ("M", "合弯矩 |M|（空间云图推荐）"),
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
        if key == scene.STRESS:
            # 应力云图要截面的极端纤维距离。**在点下去的那一刻就回答**，
            # 而不是切过去之后留给用户一个空视口。
            frame = self.session.frame or self.session.preview_frame()
            try:
                scene.check_stress_available(frame)
            except StressUnavailable as exc:
                QMessageBox.information(self, "无法绘制应力云图", str(exc))
                return
        self.component = key
        if self._needs_solution():
            # 正在看内力图时换分量就留在内力图；应力只能画云图（它没有受拉侧，
            # 也不是一个能沿杆画成形状的内力）
            keep = self.mode == "内力图" and key != scene.STRESS
            self.set_mode("内力图" if keep else "云图")

    def _needs_solution(self) -> bool:
        """需要结果的动作先过这道闸。

        拦下来时要**把模式退回"模型"**：不退的话按钮勾在"变形"上，
        视口里画的却是模型——界面在说谎。
        """
        if self.runner.busy:
            return False
        self._sync_result_validity()
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
        if not self._require_selection("run_solid_joint", ("node",), "节点实体"):
            return
        if self.session.solution is None:
            QMessageBox.information(
                self, "节点实体", "请先求解整体梁模型，再选择节点运行局部实体分析。")
            return
        node_id = int(self._selected_id)
        case = self.case
        levels = self._ask_joint_levels(node_id)
        if levels is None:
            return
        options = getattr(self, "_joint_analysis_options", {})
        title = f"节点 {node_id} 局部实体（{'收敛判断 3 档' if levels > 1 else '快速 1 档'}）"
        self._analyse(
            "solid_joint", title,
            lambda report: self.session.analyze_joint_solid(
                node_id=node_id, case=case, levels=levels,
                **options,
                progress=lambda text: report(text, None, True)),
            with_progress=True)

    def _ask_joint_levels(self, node_id: int) -> int | None:
        """先说清两种算法的代价与区别，再开算。

        节点实体是分钟级作业（实测 6 杆 D219×8 节点单档约 4 分钟），而且
        单档网格判不了热点应力是否收敛，正式的应力集中系数 Kt 只有多档才给。
        不问一句就开算，用户等了几分钟才发现拿不到想要的那个数。
        """
        from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout
        box = QDialog(self)
        box.setWindowTitle("节点实体")
        box.resize(460, 300)
        layout = QVBoxLayout(box)
        note = QLabel(f"节点{node_id}：默认由整体梁切面位移与转角驱动。\n过渡区域保留远端截面与孔洞，曲线和圆角尺寸须按实际节点核对。")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        backend = QComboBox()
        backend.addItem("自研C3D10", "native")
        backend.addItem("Abaqus细网格", "abaqus")
        form.addRow("求解后端", backend)
        geometry = QComboBox()
        geometry.addItem("一体连续节点（自动）", "transition")
        geometry.addItem("斜接裁切与圆角", "miter")
        geometry.addItem("截面包络填充（原方式）", "envelope")
        geometry.addItem("原始杆件交汇", "intersection")
        geometry.setToolTip("采用一体连续实体假设。柱顶模板保持柱截面，并在交汇处生成局部曲面过渡；匹配两臂使用G2扫掠，其他节点采用斜接。无焊缝、螺栓或接触界面。")
        form.addRow("节点几何", geometry)
        loading = QComboBox()
        loading.addItem("切面位移驱动", "displacement")
        loading.addItem("杆端内力加载", "force")
        loading.setToolTip("切面位移驱动：保留区内不能有整体支座。杆端内力加载：锚固一根杆件，"
                           "其余杆端施加整体梁内力；保留区内的支座反力改由锚固杆件承担，结果中会写明。")
        form.addRow("加载方式", loading)
        levels = QComboBox()
        levels.addItem("快速（1 档）", 1)
        levels.addItem("收敛判断（3 档）", 3)
        form.addRow("网格档位", levels)
        size = QDoubleSpinBox()
        size.setRange(0, 10000)
        size.setDecimals(2)
        size.setSuffix(" mm")
        size.setSpecialValueText("自动（局部细化）")
        size.setToolTip("指定全局网格尺寸；节点连接附近进一步细化。多档时此值为最细档全局尺寸。")
        form.addRow("网格尺寸", size)
        backend.currentIndexChanged.connect(
            lambda: levels.setCurrentIndex(1) if backend.currentData() == "abaqus" else None)
        levels.currentIndexChanged.connect(
            lambda: levels.setCurrentIndex(1) if backend.currentData() == "abaqus" else None)
        layout.addLayout(form)
        warning = QLabel("较细网格增加计算时间和内存。自研规模不足时可选择Abaqus；三档结果稳定后再判断应力。")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        buttons = dialog_styles.button_box(box)
        buttons.accepted.connect(box.accept)
        buttons.rejected.connect(box.reject)
        layout.addWidget(buttons)
        if box.exec() != QDialog.DialogCode.Accepted:
            return None
        count = int(levels.currentData())
        self._joint_analysis_options = dict(
            backend=backend.currentData(), geometry_mode=geometry.currentData(),
            loading_method=loading.currentData(),
            mesh_sizes_mm=([size.value()*1.5**k for k in range(count-1, -1, -1)] if size.value() else None))
        return count

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

    def _analyse(self, kind: str, title: str, fn, with_progress: bool = False) -> None:
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
                if kind == "strength":
                    # 从「强度验算」按钮算过就记下，应力比图直接复用
                    self._utilization = (self.session.solution, result.payload)
                caption, cols, rows, loc = result_rows.to_rows(
                    kind, result.payload)
                self.results.show_rows(f"{title}　{caption}", cols, rows, loc,
                                       result_rows.row_marks(kind,
                                                             result.payload),
                                       overview=result_rows.strength_overview(result.payload)
                                       if kind == "strength" else None,
                                       primary_columns={"杆件", "截面", "应力比 σ/[σ]", "折算比", "N/(φA)/f", "结论"}
                                       if kind == "strength" else None)
                if kind == "solid_joint":
                    directory = (result.payload.get("files") or {}).get("directory")
                    if directory:
                        self._load_solid_result(
                            Path(directory) / "summary.json",
                            (self.session.solution, self.session.result_identity))
            self.refresh()

        def failed(kind_: str, message: str) -> None:
            self._on_busy(False)
            self.statusBar().clearMessage()
            self.results.show_message(f"{title}　已停止" if kind_ == "TaskCancelled"
                                      else f"{title}　计算出错",
                                      f"{kind_}：{message[:400]}")

        def progress(text: str, _args=None, _ok: bool = True) -> None:
            # 分钟级作业要一直看得见走到哪了，不能只有一句"计算中"
            self.statusBar().showMessage(f"{title}：{text}")
            self.results.show_message(title, text)

        self.runner.submit(fn, on_done=done, on_failed=failed,
                           on_progress=progress if with_progress else None)

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
        self._close_pending = True
        self._utilization_timer.stop()
        if self.runner.busy:
            event.ignore()
            self.stop_task()
            return
        try:
            self.autosave.flush()               # 别等计时器，关窗前把最后一版写下来
        except OSError:
            pass
        self._save_preferences()
        self.chat.shutdown()
        # QDialog 不会因父窗口 close() 自动隐藏，退出时同步关闭各独立面板。
        for name in self.PANELS:
            getattr(self, name).hide()
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
        identity = self.session.result_identity or {}
        kind = (identity.get("parameters") or {}).get("analysis", "linear")
        label = {"linear": "线性", "pdelta": "P-Δ", "material": "材料非线性",
                 "step": "非比例加载"}.get(kind, kind)
        self.lbl_solve.setToolTip(
            f"求解编号：{identity.get('solve_id', '')}\n"
            f"模型指纹：{identity.get('model_fingerprint', '')}\n"
            f"求解时间：{identity.get('solved_at', '')}\n"
            f"参数：{identity.get('parameters', {})}")
        return f"求解：{label}　{n} 个工况/组合　当前 {self.case}"

    def export_screenshot(self, path: str) -> str:
        """整窗截图。

        **不能只用 QWidget.grab()**：VTK 的渲染窗口是个带自己 GL 表面的原生子窗口，
        Qt 抓窗口时抓不到它，视口会是一片黑。所以分两次抓再合成——
        外壳用 Qt 抓，视口用 plotter 自己的 screenshot，然后贴回它的位置。
        答辩截图和报告插图都走这条路。
        """
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
            # 抽屉是独立的 Tool 窗口，空模型引导卡片在视口上层；
            # VTK 画面贴回后必须把这些覆盖层再贴一次，否则导出图里会消失。
            for overlay in (self.empty_state, self.viewport.axis_indicator,
                            self.viewport.result_overlay, self.viewport.color_scale,
                            self.viewport.symbol_key, self.left_drawer,
                            self.right_drawer, self.bottom_drawer):
                if overlay.isVisible():
                    painter.drawPixmap(self.mapFromGlobal(overlay.mapToGlobal(overlay.rect().topLeft())),
                                       overlay.grab())
            painter.end()
        chrome.save(path)
        return path

    # --- 树的动作 ---

    def _on_tree_action(self, action: str, payload) -> None:
        if action == "locate_object":
            self.locate(*payload)
            return
        if action == "feature_branch":
            self.open_object_manager(str(payload))
            return
        if action == "instances":
            self.open_instance_manager()
            return
        if action == "sets":
            self.open_model_sets()
            return
        if action == "steps":
            self.open_step_manager()
            return
        if action == "materials":
            self.create_material()
        elif action == "sections":
            self.create_section()
        elif action in {"geometry", "nodes", "members"}:
            self.open_tables()
        elif action in ("supports", "support_object"):
            # 「直接定义」下的支座：管理器看不到它们，先提议转成对象
            if action == "support_object" and payload is not None:
                self.viewport.set_selection("node", int(payload))
                self._on_picked("node", int(payload))
            if self.direct_supports() and not self.convert_direct_supports():
                return
            self.open_bc_manager()
        elif action == "loads":
            # 点某个工况：切换到它；点「荷载」文件夹本身：打开荷载管理器
            if payload:
                self.case = str(payload)
                self.refresh()
            else:
                self.open_load_manager()
        elif action == "load_object" and isinstance(payload, dict):
            self.case = str(payload.get("case") or self.case or "")
            kind, ident = str(payload["kind"]), int(payload["id"])
            self.viewport.set_selection(kind, ident)
            self._on_picked(kind, ident)
            self.open_load_manager()
        elif action == "results" and payload:
            self.case = str(payload)
            self.set_mode("变形")
        elif action == "instance" and payload:
            record = next((r for r in (self.session.model.get("assembly") or {}).get("instances") or []
                           if r["name"] == payload), None)
            if record is not None:
                self.viewport.highlight_region("member", record["members"])
                self.statusBar().showMessage(f"实例「{payload}」：{len(record['members'])} 根杆件", 5000)
        elif action == "result_step":
            self.show_step_result(str(payload))
        elif action == "result_frame":
            self.show_step_result(*payload)
        elif action == "result_mode":
            self.show_step_mode(*payload)
        elif action == "history":
            self.timeline_dock.setVisible(True)
            self.timeline_dock.raise_()
            self.timeline.rebuild(self.session.history)
