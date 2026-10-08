"""星标改进的回归：结果被报告覆盖、后台编辑竞态与保存中断等真实问题。"""

from copy import deepcopy
from threading import Event

import numpy as np
import pytest

from agent import Session
from model_io import atomic_write_text
from report import gather, to_markdown
from task_control import TaskCancelled, checkpoint, task_scope


def built(analysis="linear"):
    session = Session()
    result = session.set_model(model={
        "units": "N-m-Pa",
        "materials": [{"name": "steel", "E": 2.06e11, "nu": 0.3,
                       "density": 7850, "yield_stress": 355e6,
                       "allow_tension": 215e6, "allow_compression": 215e6}],
        "sections": [{"name": "column", "A": 0.005, "Iy": 4e-5,
                      "Iz": 4e-5, "J": 8e-5, "cy": 0.1, "cz": 0.1}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 0, "y": 0, "z": 3}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "material": "steel", "section": "column"}],
        "supports": [{"node": 1, "fix": [1] * 6}],
        "load_cases": [{"name": "load", "nodal_loads": [
            {"node": 2, "load": [1000, 0, -50000, 0, 0, 0]}]}],
    })
    assert result.ok, result.payload
    assert session.solve_model(analysis=analysis, increments=3).ok
    return session


@pytest.mark.parametrize("analysis", ["linear", "pdelta", "material"])
def test_report_preserves_the_current_solution_and_analysis(analysis, tmp_path, monkeypatch):
    """防止报告默认重解线性静力，覆盖已完成的 P-Δ 或材料非线性结果。"""
    session = built(analysis)
    solution, db = session.solution, session.result_db
    displacement = solution["load"].U.copy()
    identity = deepcopy(session.result_identity)

    def forbidden(**_kwargs):
        pytest.fail("报告不能重新求解")

    monkeypatch.setattr(session, "solve_model", forbidden)
    document = gather(session, out_dir=tmp_path, figures=False)
    assert session.solution is solution and session.result_db is db
    np.testing.assert_array_equal(solution["load"].U, displacement)
    assert document["result_identity"] == identity
    assert session.result_db.metadata["result_identity"] == identity
    assert identity["parameters"]["analysis"] == analysis
    assert identity["solve_id"] in to_markdown(document)
    assert document["checks"][0]["最大节点位移 (mm)"] == (
        session.solve_summary.payload["cases"]["load"]["max_displacement_mm"])


@pytest.mark.parametrize("entry", ["query", "diagram", "envelope", "plot", "report", "strength"])
def test_all_result_entries_reject_a_direct_model_edit(entry):
    """防止绕过编辑工具修改字典后，后处理继续读取旧模型数值。"""
    session = built()
    session.model["nodes"][1]["z"] = 4
    calls = {
        "query": lambda: session.query_results("max_displacement"),
        "diagram": lambda: session.query_diagram("Mz", member=1),
        "envelope": lambda: session.query_envelope("Mz"),
        "plot": lambda: session.plot_results("deformed"),
        "report": session.write_report,
        "strength": session.check_strength,
    }
    result = calls[entry]()
    assert not result.ok and "失效" in result.payload["error"]
    assert session.solution is None and session.result_db is None
    assert session.result_identity is None


def test_each_solve_has_an_independent_identity():
    """防止同模型重新求解后，探针与缓存仍被误认成上一次结果。"""
    session = built()
    previous = deepcopy(session.result_identity)
    assert session.solve_model().ok
    assert session.result_identity["solve_id"] != previous["solve_id"]
    assert session.result_identity["model_fingerprint"] == previous["model_fingerprint"]


def test_cancelled_solver_restores_the_previous_complete_result(monkeypatch):
    """防止求解取消后发布半次结果，或破坏之前已经完成的分析。"""
    import session_solving
    session = built()
    previous = (session.solution, session.result_db, session.result_identity,
                session.frame, session.solve_summary)
    event = Event()
    original = session_solving.solve

    def cancel_after_matrix(frame):
        outcome = original(frame)
        event.set()
        return outcome

    monkeypatch.setattr(session_solving, "solve", cancel_after_matrix)
    with task_scope(event), pytest.raises(TaskCancelled):
        session.solve_model()
    assert (session.solution, session.result_db, session.result_identity,
            session.frame, session.solve_summary) == previous


def test_cancelled_scope_does_not_affect_later_scripts():
    """防止一个窗口的取消标志残留，导致后续普通脚本也被取消。"""
    event = Event()
    event.set()
    with task_scope(event), pytest.raises(TaskCancelled):
        checkpoint()
    checkpoint()


def test_atomic_save_preserves_the_original_when_replace_fails(tmp_path, monkeypatch):
    """防止正式保存写入中断时，把原有完整 JSON 覆盖为半个文件。"""
    import model_io
    path = tmp_path / "中文模型.json"
    path.write_text('{"original": true}', encoding="utf-8")

    def failed_replace(_source, _target):
        raise PermissionError("文件正在被占用")

    monkeypatch.setattr(model_io.os, "replace", failed_replace)
    with pytest.raises(PermissionError):
        atomic_write_text(path, '{"new": true}')
    assert path.read_text(encoding="utf-8") == '{"original": true}'
    assert list(tmp_path.iterdir()) == [path]


def test_optimizer_stops_between_samples_and_keeps_the_model():
    """防止优化取消异常被当成坏截面吞掉，继续遍历剩余样本。"""
    from section_optimizer import SectionOptimizer
    model = built().model
    original = deepcopy(model)
    optimizer = SectionOptimizer(
        model=model, section_name="column", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.6, 3)}, objectives=["weight"])
    event = Event()
    progress = []

    def stop(done, total):
        progress.append((done, total))
        event.set()

    with task_scope(event), pytest.raises(TaskCancelled):
        optimizer.grid_search(progress=stop)
    assert progress == [(1, 3)] and model == original


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_busy_runner_blocks_editors_and_recent_file_open(qt_app, tmp_path):
    """防止优化或识别开始后，属性编辑和最近打开仍能替换后台使用的模型。"""
    from desktop.main_window import MainWindow
    window = MainWindow(built())
    session = window.session
    release = Event()
    try:
        assert window.runner.submit(lambda: release.wait(5))
        for name in ("new", "open", "units", "tables", "undo", "model_node"):
            assert not window.actions_by_name[name].isEnabled()
        for widget in (window.properties, window.bc, window.timeline, window.sketch):
            assert not widget.isEnabled()
        assert window.actions_by_name["front"].isEnabled()
        assert not window._load_model_path(tmp_path / "other.json")
        assert window.session is session
    finally:
        release.set()
        assert window.runner.wait(10000)
        window.close()


def test_window_close_waits_asynchronously_for_the_job(qt_app):
    """防止关闭窗口等待超时后仍销毁正在被工作线程使用的控件和视口。"""
    from desktop.main_window import MainWindow
    window = MainWindow()
    release = Event()

    def job():
        release.wait(5)
        checkpoint()

    window.show()
    assert window.runner.submit(job)
    window.close()
    assert window.isVisible() and window._close_pending
    release.set()
    assert window.runner.wait(10000)
    qt_app.processEvents()
    assert not window.isVisible()


def test_result_filters_keep_locators_after_sorting(qt_app):
    """防止超限筛选经过数值排序后，点击行定位到错误杆件。"""
    from PySide6.QtCore import Qt
    from desktop.panels import LOCATE, ResultPanel
    panel = ResultPanel()
    panel.show_rows("验算", ["杆件", "比值"], [[10, 0.5], [2, 1.2], [3, None]],
                    [("member", 10), ("member", 2), ("member", 3)],
                    ["pass", "fail", "unclear"])
    panel.row_filter.setCurrentIndex(1)
    panel.table.sortItems(0, Qt.SortOrder.AscendingOrder)
    visible = [row for row in range(3) if not panel.table.isRowHidden(row)]
    assert len(visible) == 1
    assert panel.table.item(visible[0], 0).data(LOCATE) == ("member", 2)
    panel.row_filter.setCurrentIndex(2)
    visible = [row for row in range(3) if not panel.table.isRowHidden(row)]
    assert panel.table.item(visible[0], 0).data(LOCATE) == ("member", 3)
    panel.show_message("结果已失效", "请重新求解")
    assert panel.table.rowCount() == 0 and panel.row_filter.currentIndex() == 0


def test_empty_state_recommends_the_deterministic_path(qt_app):
    """防止首屏把实验性识别作为主按钮，与手册推荐的新手路线相反。"""
    from PySide6.QtWidgets import QPushButton
    from desktop.main_window import MainWindow
    window = MainWindow()
    buttons = {button.text(): button for button in
               window.empty_state.findChildren(QPushButton)}
    assert buttons["参数化框架"].property("role") == "primary"
    assert "实验性" in buttons["AI 识别图纸"].toolTip()
    window.close()


def test_direct_edit_clears_table_probe_and_curve_together(qt_app):
    """防止绕过编辑入口修改模型后，云图清空而旧探针与曲线仍可读。"""
    from desktop.main_window import MainWindow
    session = built()
    window = MainWindow(session)
    window._solved(session.solve_summary)
    window.probe_member_result(1, (0, 0, 1.5))
    assert window._last_probe is not None
    session.model["nodes"][1]["z"] = 4
    window.refresh()
    assert window.result is None and window.mode == "模型"
    assert window._last_probe is None and window.results.table.rowCount() == 0
    assert window.diagram.series() is None
    window.close()


def test_combined_annotations_prioritize_the_selected_object(qt_app, monkeypatch):
    """防止支座与荷载分用布局器而互相遮挡，选中对象的数值被普通标签挤掉。"""
    from desktop import viewport as module
    from desktop.viewport import Viewport
    from types import SimpleNamespace
    viewport = Viewport()
    session = built()
    captured = {}

    class Mapper:
        def GetInputAlgorithm(self):
            return SimpleNamespace(SetPriorityArrayName=lambda name: captured.update(array=name))

        def SetPlaceAllLabels(self, value):
            captured["place_all"] = value

        def SetMaximumLabelFraction(self, value):
            captured["fraction"] = value

    def labels(points, array_name, **_kwargs):
        captured["points"], captured["name"] = points, array_name
        return SimpleNamespace(GetMapper=lambda: Mapper())

    viewport.plotter = SimpleNamespace(remove_actor=lambda *_a, **_k: None,
                                       add_point_labels=labels)
    viewport.selection = ("node", 2)
    monkeypatch.setattr(module, "CAN_RENDER", True)
    viewport._show_model_annotations(session.frame, "load", True, True)
    assert not captured["place_all"]
    priorities = captured["points"]["annotation_priority"]
    assert 10 in priorities and 2 in priorities
    assert captured["array"] == "annotation_priority"
