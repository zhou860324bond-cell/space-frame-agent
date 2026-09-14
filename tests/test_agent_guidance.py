"""下一步建议只依赖工程状态，不依赖 Qt 或大模型发挥。"""

from types import SimpleNamespace

from desktop.agent_guidance import next_suggestions
from workflow import WorkflowPhase


def status(phase, **counts):
    base = {"materials": 1, "sections": 1, "supports": 1,
            "load_cases": 1}
    base.update(counts)
    return SimpleNamespace(phase=phase, counts=base)


def labels(items):
    return [label for label, _prompt in items]


def test_empty_project_offers_three_real_starting_paths():
    items = next_suggestions(status(WorkflowPhase.EMPTY))
    assert labels(items) == ["参数化建模", "从图纸开始", "手动建模步骤"]


def test_draft_selection_takes_priority_over_generic_missing_items():
    items = next_suggestions(
        status(WorkflowPhase.DRAFT, materials=0, sections=0, supports=0),
        "member", 12)
    assert labels(items)[0] == "编辑杆件荷载"
    assert "杆件 12" in items[0][1]


def test_ready_node_suggests_node_work_before_solving():
    items = next_suggestions(status(WorkflowPhase.READY), "node", 7)
    assert labels(items) == ["复核节点边界", "设置节点荷载", "开始求解"]


def test_solved_contour_suggests_result_interpretation():
    items = next_suggestions(status(WorkflowPhase.SOLVED), mode="云图")
    assert labels(items) == ["定位云图极值", "解释颜色分布", "查看控制杆件"]
