"""按CAE模块组织人工建模检查；状态来自当前模型及实际预览，不来自点击次数。"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout)

from . import dialog_styles


@dataclass(frozen=True)
class ModelingStage:
    key: str
    title: str
    page: str
    description: str
    actions: tuple[str, ...]


STAGES = (
    ModelingStage("part", "1 几何 · Part", "建模",
                  "确认单位制；建立节点和杆件中心线，复核坐标与尺寸。",
                  ("units", "sketch", "frame", "tables")),
    ModelingStage("property", "2 属性 · Property", "属性",
                  "定义材料及梁截面，向杆件指派属性，并复核局部轴与截面强弱轴。",
                  ("material", "beam_section", "batch_assignment", "beam_orientation")),
    ModelingStage("assembly", "3 装配 · Assembly", "装配",
                  "当前使用全局坐标下的杆系。检查节点编号与端点连接，并创建可复用集合。几何相交不自动代表连接。",
                  ("assembly_check", "model_sets", "tables")),
    ModelingStage("step", "4 分析步 · Step", "分析",
                  "独立工况使用求解设置；需逐步改变载荷或约束时定义分析步及幅值。步骤各自重解，不继承塑性状态。位移、反力和杆端力固定输出。",
                  ("solve_options", "step_manager", "amplitude")),
    ModelingStage("interaction", "5 连接 · Interaction", "连接",
                  "同一节点编号默认刚接；按局部轴设置释放或有依据的转动弹簧。当前没有螺栓预紧及接触求解。",
                  ("member_connections", "beam_orientation", "props")),
    ModelingStage("load", "6 约束与载荷 · Load", "载荷",
                  "定义Initial约束、独立工况、节点力或杆件荷载及组合；复核方向、单位和分析步实际生效项。",
                  ("create_bc", "create_load", "load", "combo", "bc_manager")),
    ModelingStage("mesh", "7 离散 · Mesh", "网格",
                  "预览整体梁分析杆段与物理杆件的映射。内节点和集中作用点自动剖分；不是实体网格。局部C3D10网格在整体求解后单独设置。",
                  ("mesh_preview",)),
    ModelingStage("job", "8 作业 · Job", "作业",
                  "先做模型检查和机构诊断。独立工况与声明分析步分别提交；后台运行，可停止并查看失败原因。",
                  ("model_check", "support_diagnosis", "solve", "solve_steps", "stop_task")),
    ModelingStage("visualization", "9 结果 · Visualization", "结果",
                  "检查变形、反力平衡及内力；再做局部实体与网格收敛检查，最后导出报告。整体梁结果与实体应力分别显示。",
                  ("deformed", "contour", "curve", "strength", "solid_joint", "report")),
)


def assembly_issues(session):
    """几何阶段独立检查，不要求尚未定义的材料、载荷或约束。"""
    from draft import FrameDraft, SNAP_TOL

    draft = FrameDraft.from_model(session.model)
    issues = []
    tolerance = SNAP_TOL / session.units.length_to_m
    for group in draft.coincident_nodes(tol=tolerance):
        issues.append(("node", group[0], f"节点 {group} 坐标重合但编号不同，请核对并共用同一节点编号。"))
    known = {int(n["id"]) for n in draft.nodes}
    pairs = {}
    for member in draft.members:
        mid = int(member["id"])
        endpoints = (int(member["i"]), int(member["j"]))
        missing = set(endpoints) - known
        if missing:
            issues.append(("member", mid, f"杆件 {mid} 的端点 {sorted(missing)} 不存在，请修正端点引用。"))
            continue
        length = draft.length(mid)
        if length is not None and length <= tolerance:
            issues.append(("member", mid, f"杆件 {mid} 长度为零或过小，请修正节点坐标。"))
        pair = tuple(sorted(endpoints))
        if pair in pairs:
            issues.append(("member", mid, f"杆件 {mid} 与杆件 {pairs[pair]} 使用同一对端点，请核对是否重复。"))
        pairs[pair] = mid
    return issues


def inspect_modeling(window):
    """九阶段状态是定义与复核状态，不等于数值解可靠性结论。"""
    session = window.session
    if not window.runner.busy:
        session.result_error()
    model = session.model
    nodes, members = model.get("nodes") or [], model.get("members") or []
    key = session._model_key()
    reviews = getattr(window, "_modeling_reviews", {})
    rows = {}
    has_geometry = len(nodes) >= 2 and bool(members)
    rows["part"] = ("已定义" if has_geometry else "待定义",
                    f"{len(nodes)} 节点 / {len(members)} 杆件；{model.get('units', 'N-m-Pa')}")
    mats = {m["name"] for m in model.get("materials") or []}
    secs = {s["name"] for s in model.get("sections") or []}
    unassigned = [m["id"] for m in members
                  if m.get("material") not in mats or m.get("section") not in secs]
    rows["property"] = ("待定义" if not members or unassigned else "已指派",
                        f"{len(mats)} 材料 / {len(secs)} 截面；未指派杆件 {unassigned[:12]}"
                        if unassigned or not members else "全部杆件有材料与截面；仍需复核梁方向。")
    issues = assembly_issues(session)
    rows["assembly"] = ("需修正" if issues else "已复核" if reviews.get("assembly") == key else "待复核",
                        issues[0][2] if issues else f"未检出重合节点、重复杆件或端点错误；{len(model.get('sets') or {})} 个集合。")
    steps = model.get("steps") or []
    step_note = f"{len(steps)} 个声明分析步" if steps else "独立工况；" + {
        "linear": "线性静力", "pdelta": "P-Delta 二阶弹性",
        "material": "双线性轴向材料非线性"}.get(window.analysis_type, window.analysis_type)
    step_error = None
    if steps:
        import steps as step_module
        try:
            step_module.resolve(step_module.from_payload(model), model)
        except ValueError as exc:
            step_error = str(exc)
    rows["step"] = ("需修正" if step_error else "已定义", step_error or step_note)
    released = sum(bool(m.get("releases")) for m in members)
    springs = sum(bool(m.get("connections")) for m in members)
    rows["interaction"] = ("已复核" if reviews.get("interaction") == key else "待复核",
                           f"{released} 根含释放 / {springs} 根含连接弹簧；其余共享节点按刚接。")
    cases = model.get("load_cases") or []
    rows["load"] = ("已定义" if model.get("supports") and cases else "待定义",
                    f"{len(model.get('supports') or [])} 支座 / {len(cases)} 工况 / {len(model.get('combos') or [])} 组合；空工况不会自动补载荷。")
    rows["mesh"] = ("已预览" if getattr(window, "_analysis_mesh_key", None) == key else "待预览",
                    "整体梁网格；模型变化后须重新预览。")
    errors = [] if window.runner.busy else session.validation_errors()
    solved = session.solution is not None and session.result_db is not None
    rows["job"] = ("运行中" if window.runner.busy else "需修正" if errors else "已求解" if solved else "可提交",
                   window.statusBar().currentMessage() if window.runner.busy else
                   f"完整性校验未通过，共 {len(errors)} 项；请修正上方待定义项，再执行完整性校验查看对象位置。"
                   if errors else "声明分析步须使用提交分析步；普通求解只计算独立工况与组合。")
    rows["visualization"] = ("可查看" if solved else "待求解",
                             "检查平衡与警告后判读结果；单次实体求解不能证明应力收敛。")
    return rows


class ModelingWorkflowDialog(QDialog):
    """非模态检查清单，允许打开真实编辑器并返回继续复核。"""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("刚架建模全流程")
        self.resize(820, 620)
        layout = QVBoxLayout(self)
        note = QLabel("按 Part → Property → Assembly → Step → Interaction → Load → Mesh → Job → Visualization 组织。状态表示当前定义与人工复核，不代替工程验证。", self)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(len(STAGES), 3, self)
        self.table.setHorizontalHeaderLabels(["步骤", "状态", "当前模型"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setWordWrap(True)
        layout.addWidget(self.table, 1)
        self.detail = QLabel(self)
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.detail)
        self.action_host = QHBoxLayout()
        layout.addLayout(self.action_host)
        self.review = QPushButton("确认本步骤已复核", self)
        self.review.clicked.connect(self.mark_reviewed)
        layout.addWidget(self.review)
        close = dialog_styles.close_box(self)
        close.accepted.connect(self.accept)
        layout.addWidget(close)
        self.table.currentCellChanged.connect(self.select_stage)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        self.table.selectRow(0)

    def showEvent(self, event):
        super().showEvent(event)
        available = self.screen().availableGeometry()
        self.resize(min(self.width(), available.width() - 24),
                    min(self.height(), available.height() - 60))
        QTimer.singleShot(0, self.table.resizeRowsToContents)
        self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def refresh(self):
        # 分步求解会暂时切换Session模型；运行中保持最后一份定义，不读取中间配置。
        if self.window.runner.busy and hasattr(self, "_rows"):
            rows = dict(self._rows)
            rows["job"] = ("运行中", self.window.statusBar().currentMessage())
        else:
            rows = inspect_modeling(self.window)
            self._rows = rows
        for row, stage in enumerate(STAGES):
            for col, value in enumerate((stage.title, *rows[stage.key])):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()
        self.review.setEnabled(not self.window.runner.busy)

    def select_stage(self, row, *_):
        if row < 0:
            return
        stage = STAGES[row]
        self.detail.setText(stage.description)
        self.window._show_ribbon_page(stage.page)
        while self.action_host.count():
            item = self.action_host.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        for name in stage.actions:
            button = QToolButton(self)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setDefaultAction(self.window.actions_by_name[name])
            self.action_host.addWidget(button)
        self.action_host.addStretch(1)
        self.review.setVisible(stage.key in ("assembly", "interaction"))

    def mark_reviewed(self):
        stage = STAGES[self.table.currentRow()]
        if self.window.runner.busy:
            return
        if stage.key == "assembly" and assembly_issues(self.window.session):
            self.window.check_assembly()
            return
        if not self.window.session.model.get("members"):
            self.detail.setText("尚未创建杆件，请先建立几何再复核连接。")
            return
        self.window._modeling_reviews[stage.key] = self.window.session._model_key()
        self.refresh()
