"""The workflow bar must describe the real Session, not optimistic UI state."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from agent import Session  # noqa: E402
from desktop.workflow_bar import WorkflowBar, draft_target  # noqa: E402
from workflow import inspect_workflow  # noqa: E402


MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7},
]


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def ready_session() -> Session:
    session = Session()
    assert session.define_materials_and_sections(MATERIALS, SECTIONS).ok
    assert session.generate_frame(
        spans=[6.0], storeys=[3.6], column_section="COLUMN",
        beam_section="BEAM", material="STEEL").ok
    assert session.set_load_cases(cases=[{
        "name": "DL", "member_loads": [{"member": 3, "w": [0, 0, -1000]}],
    }]).ok
    return session


def state(bar: WorkflowBar, stage: str) -> str:
    return str(bar.buttons[stage].property("workflowState"))


def test_empty_session_offers_the_real_first_action(qt_app):
    bar = WorkflowBar()
    bar.update_state(Session())

    assert state(bar, "model") == "active"
    assert bar.next_button.text() == "开始建模"
    assert "尚未创建" in bar.guidance.text()


def test_incomplete_geometry_does_not_claim_modeling_is_done(qt_app):
    session = Session()
    assert session.add_nodes([[0.0, 0.0, 0.0]]).ok
    bar = WorkflowBar()
    bar.update_state(session)

    assert state(bar, "model") == "active"
    assert state(bar, "define") == "pending"
    assert state(bar, "validate") == "blocked"
    assert "至少创建 2 个节点" in bar.guidance.text()
    assert "!" in bar.buttons["validate"].text()


def test_draft_target_routes_to_the_matching_editor():
    nodes_only = Session()
    assert nodes_only.add_nodes([[0, 0, 0], [1, 0, 0]]).ok
    assert draft_target(inspect_workflow(nodes_only))[:2] == ("model", "建模")

    geometry = Session()
    assert geometry.add_nodes([[0, 0, 0], [1, 0, 0]]).ok
    assert geometry.add_members([[1, 2]]).ok
    assert draft_target(inspect_workflow(geometry))[:2] == ("define", "属性")


def test_ready_and_solved_states_expose_only_valid_next_steps(qt_app):
    session = ready_session()
    bar = WorkflowBar()
    bar.update_state(session)

    assert state(bar, "validate") == "done"
    assert state(bar, "solve") == "active"
    assert bar.next_button.text() == "开始求解"

    assert session.solve_model().ok
    bar.update_state(session)
    assert state(bar, "results") == "active"
    assert bar.next_button.text() == "查看结果"


def test_primary_button_emits_the_backend_derived_stage(qt_app):
    bar = WorkflowBar()
    requested = []
    bar.stage_requested.connect(requested.append)
    bar.update_state(ready_session())

    bar.next_button.click()

    assert requested == ["solve"]
