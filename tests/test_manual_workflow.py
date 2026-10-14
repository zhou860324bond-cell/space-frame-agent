"""人工CAE流程的真实遗漏、批量事务和后台步骤结果回归。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent import Session


def frame_session():
    session = Session()
    assert session.add_nodes([[0, 0, 0], [2, 0, 0], [0, 2, 0]]).ok
    assert session.define_materials_and_sections(
        [{"name": "M", "E": 2e11, "nu": .3}],
        [{"name": "S", "A": .01, "Iy": 1e-5, "Iz": 2e-5, "J": 1e-5}]).ok
    assert session.add_members([[1, 2], [1, 3]], "S", "M").ok
    assert session.set_supports([1], fix=[1] * 6).ok
    assert session.set_load_cases([{"name": "F", "nodal_loads": [
        {"node": 2, "load": [0, 1000, 0, 0, 0, 0]}]}]).ok
    return session


def test_batch_direction_failure_preserves_model_results_and_history():
    """第二根杆的平行参考方向错误，不能留下第一根已改、旧结果失效的半事务。"""
    session = frame_session()
    assert session.solve_model().ok
    original = deepcopy(session.model)
    solution = session.solution
    result_identity = deepcopy(session.result_identity)
    result = session.configure_member_group([1, 2], ref_vector=[0, 1, 0])
    assert not result.ok and result.payload["member"] == 2
    assert session.model == original
    assert session.solution is solution and session.result_identity == result_identity
    assert not session.history[-1].ok


def test_connection_modes_replace_only_requested_rotations_in_one_transaction():
    """改回刚接须清除旧弹簧和释放，且未选方向的释放与弹簧保持原样。"""
    session = frame_session()
    assert session.configure_member_group([1, 2], end="j", rotations={
        "rz": {"mode": "release"}, "ry": {"mode": "spring", "stiffness": 1e6}}).ok
    count = len(session.history)
    assert session.configure_member_group([1, 2], end="j", rotations={
        "rz": {"mode": "spring", "stiffness": 2e6}}).ok
    assert len(session.history) == count + 1
    for member in session.model["members"]:
        assert not member.get("releases")
        assert member["connections"]["j"] == {"ry": 1e6, "rz": 2e6}
    assert session.configure_member_group([1, 2], end="j", rotations={
        "rz": {"mode": "rigid"}}).ok
    assert all(m["connections"]["j"] == {"ry": 1e6} for m in session.model["members"])


@pytest.mark.parametrize("stiffness", [0, -1, float("nan"), float("inf")])
def test_invalid_group_spring_stiffness_does_not_write(stiffness):
    """零、负数和非有限刚度不能进入人工连接设置，避免求解时才出错。"""
    session = frame_session()
    before = deepcopy(session.model)
    assert not session.configure_member_group([1], rotations={
        "rz": {"mode": "spring", "stiffness": stiffness}}).ok
    assert session.model == before


def test_manual_rotational_spring_matches_cantilever_closed_form():
    """人工输入的转动弹簧必须实际进入刚度阵，位移增加量按PL²/k校核。"""
    session = frame_session()
    assert session.configure_member_group([1], ref_vector=[0, 1, 0], end="i",
                                          rotations={"rz": {"mode": "spring", "stiffness": 1e7}}).ok
    assert session.solve_model().ok
    actual = session.result_db.field("F", "U").value(2)[1]
    expected = 1000 * 2**3 / (3 * 2e11 * 2e-5) + 1000 * 2**2 / 1e7
    assert actual == pytest.approx(expected, rel=1e-10)


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_batch_target_reuses_named_sets_and_rejects_unknown_ids(qt_app):
    """批量界面应展开真实集合，不应把不存在的编号静默过滤后部分提交。"""
    from desktop.assignment_dialog import MemberTarget
    session = frame_session()
    assert session.define_set("梁组", member_ids=[1, 2]).ok
    target = MemberTarget(session)
    target.scope.setCurrentIndex(target.scope.findText("集合：梁组"))
    assert target.member_ids() == [1, 2]
    target.scope.setCurrentIndex(target.scope.findData("ids"))
    target.ids.setText("1, 99")
    with pytest.raises(ValueError):
        target.member_ids()


def test_batch_assignment_can_select_a_group_without_viewport_selection(qt_app):
    """几何先建立时没有视口单杆选择，批量指派仍应能显式选择全部杆件。"""
    from desktop.assignment_dialog import BatchAssignmentDialog
    dialog = BatchAssignmentDialog(frame_session())
    dialog.target.scope.setCurrentIndex(dialog.target.scope.findData("all"))
    assert dialog.target.member_ids() == [1, 2]
    assert dialog.get_assignment() == ("S", "M")


def test_connection_dialog_preserves_unedited_directions_and_requires_stiffness(qt_app):
    """打开连接窗口不能自动改成刚接，也不能把默认零值当作有效弹簧刚度。"""
    from desktop.assignment_dialog import MemberSetupDialog
    dialog = MemberSetupDialog(frame_session(), 1)
    mode, stiffness = dialog.options["rz"]
    mode.setCurrentIndex(mode.findData("spring"))
    with pytest.raises(ValueError):
        dialog.changes()
    stiffness.setValue(1e7)
    assert dialog.changes() == {"end": "i", "rotations": {
        "rz": {"mode": "spring", "stiffness": 1e7}}}


def test_geometry_check_works_before_materials_and_finds_disconnected_coincidence(qt_app):
    """装配检查不能因未创建载荷被阻止；不同编号的重合端点应指出真实位置。"""
    from desktop.modeling_workflow import assembly_issues
    session = Session(model={"units": "N-m-Pa", "nodes": [
        {"id": 1, "x": 0, "y": 0, "z": 0},
        {"id": 2, "x": 1, "y": 0, "z": 0},
        {"id": 3, "x": 1, "y": 0, "z": 0}], "members": [
        {"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 2, "j": 1},
        {"id": 3, "i": 3, "j": 99}]})
    issues = assembly_issues(session)
    assert {target[:2] for target in issues} == {("node", 2), ("member", 2), ("member", 3)}


def inspection_window(session):
    return SimpleNamespace(session=session, runner=SimpleNamespace(busy=False),
                           analysis_type="linear", _modeling_reviews={}, _analysis_mesh_key=None)


def test_nine_stage_review_and_mesh_preview_expire_on_any_model_change(qt_app):
    """修改模型后，旧网格预览与人工确认不能继续显示已完成；有定义不等于已复核。"""
    from desktop.modeling_workflow import STAGES, inspect_modeling
    window = inspection_window(frame_session())
    rows = inspect_modeling(window)
    assert len(rows) == len(STAGES) == 9
    assert rows["interaction"][0] == "待复核"
    window._modeling_reviews = {key: window.session._model_key()
                                for key in ("assembly", "interaction")}
    window._analysis_mesh_key = window.session._model_key()
    rows = inspect_modeling(window)
    assert rows["assembly"][0] == rows["interaction"][0] == "已复核"
    assert rows["mesh"][0] == "已预览"
    window.session.model["nodes"][1]["x"] = 3
    rows = inspect_modeling(window)
    assert rows["interaction"][0] == "待复核" and rows["mesh"][0] == "待预览"


def test_declared_step_errors_are_visible_before_submission(qt_app):
    """声明分析步引用不存在的工况时，不能因独立工况完整便显示步骤已就绪。"""
    from desktop.modeling_workflow import inspect_modeling
    window = inspection_window(frame_session())
    window.session.model["steps"] = [{"name": "错误步", "analysis": "linear",
                                      "loads": {"不存在": "RAMP"}}]
    assert inspect_modeling(window)["step"][0] == "需修正"


def test_step_manager_submits_to_parent_runner_instead_of_blocking_ui(qt_app, monkeypatch):
    """分析步管理器原先在界面线程同步求解；有主窗口时必须委托后台入口。"""
    from PySide6.QtWidgets import QWidget
    from desktop.step_dialog import StepManagerDialog
    parent = QWidget()
    submitted = []
    parent.solve_declared_steps = lambda: submitted.append(True)
    session = frame_session()
    monkeypatch.setattr(session, "solve_steps", lambda: pytest.fail("不能在界面线程求解"))
    dialog = StepManagerDialog(session, parent)
    dialog._solve()
    assert submitted == [True]


def test_step_background_completion_publishes_current_solution_summary():
    """分步求解的返回没有cases字段，后台完成必须发布实际保留步的求解摘要。"""
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from desktop.main_window import MainWindow
    session = frame_session()
    assert session.add_step("第一步", loads={"F": "RAMP"}).ok
    assert session.add_step("第二步").ok
    result = session.solve_steps()
    assert result.ok
    published = []
    messages = []
    window = SimpleNamespace(session=session, _solved=published.append,
                             statusBar=lambda: SimpleNamespace(showMessage=lambda text, *_: messages.append(text)))
    MainWindow._steps_solved(window, result)
    assert published == [session.solve_summary]
    assert published[0].payload["cases"]["第二步"]["equilibrium_ok"]
    assert "第二步" in messages[0]


def test_open_workflow_does_not_read_intermediate_models_during_step_job(qt_app, monkeypatch):
    """分步后台求解临时改Session.model，清单计时刷新不得读取该中间状态。"""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QWidget
    from desktop import modeling_workflow
    from desktop.commands import COMMANDS
    window = QWidget()
    window.session = frame_session()
    window.runner = SimpleNamespace(busy=False)
    window.analysis_type = "linear"
    window._modeling_reviews = {}
    window._show_ribbon_page = lambda _: None
    window.actions_by_name = {cmd.name: QAction(cmd.label, window) for cmd in COMMANDS}
    window.statusBar = lambda: SimpleNamespace(currentMessage=lambda: "分析步求解中")
    dialog = modeling_workflow.ModelingWorkflowDialog(window)
    dialog.show()
    qt_app.processEvents()
    available = dialog.screen().availableGeometry()
    assert dialog.height() <= available.height() - 60
    original = dialog._rows
    window.runner.busy = True
    monkeypatch.setattr(modeling_workflow, "inspect_modeling",
                        lambda _: pytest.fail("不能读取求解中的临时模型"))
    dialog.refresh()
    assert dialog._rows is original
    assert dialog.table.item(7, 1).text() == "运行中"
    assert not dialog.review.isEnabled()


def test_step_definition_and_group_edit_are_reversible():
    """新增人工批量设置与分析步声明必须进入同一撤销链，不能只刷新界面。"""
    session = frame_session()
    before = deepcopy(session.model)
    assert session.configure_member_group([1, 2], end="both", rotations={
        "rz": {"mode": "release"}}).ok
    assert session.history.model_at(session.history.cursor - 1) == before
    connection_model = deepcopy(session.model)
    assert session.add_step("静力步", loads={"F": "RAMP"}).ok
    assert session.history.model_at(session.history.cursor - 1) == connection_model


def test_geometry_table_edits_commit_before_properties_and_loads(qt_app):
    """几何阶段表格原先因空材料、支座和载荷被阻止，应允许合法半成品编辑。"""
    from PySide6.QtWidgets import QTableWidgetItem
    from desktop.dialogs import TableDialog
    session = Session()
    assert session.add_nodes([[0, 0, 0], [2, 0, 0]]).ok
    assert session.add_members([[1, 2]]).ok
    dialog = TableDialog(session)
    dialog.tables["nodes"].setItem(1, 1, QTableWidgetItem("3"))
    dialog._accept()
    assert dialog.model is not None
    assert dialog.model["nodes"][1]["x"] == 3
    assert session.replace_editing_model(dialog.model).ok
    assert session.model["members"][0]["section"] == ""
    assert session.validation_errors()


@pytest.mark.parametrize("mutation", ["missing_endpoint", "nan", "unknown_section", "spring_release"])
def test_draft_edit_rejects_invalid_existing_data_without_partial_write(mutation):
    """允许缺定义不能放行错误引用、非有限数或释放弹簧冲突，更不能覆盖好模型。"""
    session = frame_session()
    before = deepcopy(session.model)
    candidate = deepcopy(before)
    candidate["supports"] = []
    if mutation == "missing_endpoint":
        candidate["members"][0]["j"] = 99
    elif mutation == "nan":
        candidate["nodes"][0]["x"] = float("nan")
    elif mutation == "unknown_section":
        candidate["members"][0]["section"] = "不存在"
    else:
        candidate["members"][0]["releases"] = {"i": ["rz"]}
        candidate["members"][0]["connections"] = {"i": {"rz": 1e6}}
    assert not session.replace_editing_model(candidate).ok
    assert session.model == before


def test_workflow_timer_does_not_recheck_an_unchanged_model(qt_app, monkeypatch):
    """清单每秒刷新一次；原来每次都完整检查，1331 节点框架每次 2 秒，界面一直卡住。

    模型与状态不变时必须复用上次结果，模型一变就要重算。
    """
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QWidget
    from desktop import modeling_workflow
    from desktop.commands import COMMANDS
    window = QWidget()
    window.session = frame_session()
    window.runner = SimpleNamespace(busy=False)
    window.analysis_type = "linear"
    window._modeling_reviews = {}
    window._analysis_mesh_key = None
    window._show_ribbon_page = lambda _: None
    window.actions_by_name = {cmd.name: QAction(cmd.label, window) for cmd in COMMANDS}
    window.statusBar = lambda: SimpleNamespace(currentMessage=lambda: "")
    dialog = modeling_workflow.ModelingWorkflowDialog(window)
    calls = []
    real = modeling_workflow.inspect_modeling
    monkeypatch.setattr(modeling_workflow, "inspect_modeling",
                        lambda w: calls.append(1) or real(w))
    for _ in range(5):
        dialog.refresh()
    assert calls == []
    window.session.model["nodes"][1]["x"] = 3
    dialog.refresh()
    assert calls == [1]


def test_editing_validation_shares_the_formal_rules():
    """编辑态原来另写一套检查：同一工况的同名荷载被正式校验拒绝，编辑态却放行。

    放宽的只能是「还没做完」：未指派属性、未加约束、悬空节点。
    """
    from model_io import validate_editing_payload
    session = frame_session()
    duplicate = deepcopy(session.model)
    duplicate["load_cases"][0]["nodal_loads"] = [
        {"name": "P", "node": 2, "load": [0, 1, 0, 0, 0, 0]},
        {"name": "P", "node": 3, "load": [0, 2, 0, 0, 0, 0]}]
    assert any("载荷名称重复" in e for e in validate_editing_payload(duplicate))

    unfinished = deepcopy(session.model)
    unfinished["supports"] = []
    unfinished["members"][0]["section"] = unfinished["members"][0]["material"] = ""
    unfinished["nodes"].append({"id": 9, "x": 5, "y": 5, "z": 0})
    assert validate_editing_payload(unfinished) == []


def test_assembly_check_stays_fast_on_a_thousand_member_frame():
    """装配检查原来逐杆调用 draft.length（每次线性扫描）且重合节点两两比较，
    10×10×10 框架（1331 节点、3410 杆件）一次 2.1 s；修正后约 40 ms。"""
    import time
    from desktop.modeling_workflow import assembly_issues
    session = Session()
    assert session.define_materials_and_sections(
        [{"name": "M", "E": 2e11, "nu": .3}],
        [{"name": "S", "A": .01, "Iy": 1e-5, "Iz": 2e-5, "J": 1e-5}]).ok
    assert session.generate_frame(spans=[6] * 10, bays=[6] * 10, storeys=[3.6] * 10,
                                  column_section="S", beam_section="S").ok
    start = time.perf_counter()
    assert assembly_issues(session) == []
    assert time.perf_counter() - start < 1.0
