"""Agent 工作流状态机：状态由确定性模型事实推导，不让大模型自己猜。"""

from agent import Session
from workflow import WorkflowPhase, inspect_workflow


MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "COLUMN", "A": 0.012, "Iy": 8e-5,
             "Iz": 2.4e-4, "J": 1e-6},
            {"name": "BEAM", "A": 0.010, "Iy": 4e-5,
             "Iz": 3e-4, "J": 8e-7}]


def ready_session() -> Session:
    s = Session()
    assert s.define_materials_and_sections(MATERIALS, SECTIONS).ok
    assert s.generate_frame(
        spans=[6.0], storeys=[3.6], column_section="COLUMN",
        beam_section="BEAM", material="STEEL").ok
    assert s.set_load_cases(cases=[{
        "name": "DL", "member_loads": [{"member": 3, "w": [0, 0, -1000]}],
    }]).ok
    return s


def test_empty_draft_ready_and_solved_are_distinct_states():
    empty = Session()
    assert inspect_workflow(empty).phase is WorkflowPhase.EMPTY

    draft = Session()
    assert draft.add_nodes([[0.0, 0.0, 0.0]]).ok
    status = inspect_workflow(draft)
    assert status.phase is WorkflowPhase.DRAFT
    assert status.counts["nodes"] == 1
    assert status.validation_errors

    ready = ready_session()
    status = inspect_workflow(ready)
    assert status.phase is WorkflowPhase.READY
    assert not status.validation_errors
    assert "solve_model" in status.recommended_tools

    assert ready.solve_model().ok
    status = inspect_workflow(ready)
    assert status.phase is WorkflowPhase.SOLVED
    assert "query_results" in status.recommended_tools


def test_a_model_change_moves_solved_back_to_ready():
    s = ready_session()
    assert s.solve_model().ok
    assert inspect_workflow(s).phase is WorkflowPhase.SOLVED

    assert s.set_load_cases(cases=[{
        "name": "DL", "member_loads": [{"member": 3, "w": [0, 0, -2000]}],
    }]).ok

    assert inspect_workflow(s).phase is WorkflowPhase.READY


def test_workflow_payload_is_small_structured_and_versioned():
    status = inspect_workflow(ready_session())
    payload = status.to_dict()

    assert payload["schema"] == "agent-workflow/v1"
    assert payload["phase"] == "ready"
    assert payload["counts"]["members"] > 0
    assert isinstance(payload["recommended_tools"], list)
    assert len(status.system_text()) < 1400
