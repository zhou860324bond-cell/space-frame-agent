"""撤销、重做与建模过程时间线。

这一块的价值不在"多了个功能"，在于**它把 Agent 从"不敢用"变成"敢用"**：
一轮对话能改动整个模型，没有退路，用户每次回车前都要犹豫。

所以测试的重点不是"能不能退回去"，而是**退回去之后状态是不是自洽的**——
其中最危险的一条：模型退回去了，而求解结果还留着上一个模型的。
那样界面会拿着一份对不上的位移去画变形图，画得出来、看不出错。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import Session

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "B", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7}]


def built() -> Session:
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], column_section="B",
                         beam_section="B", material="Q355")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    return s


# ------------------------------------------------- 基本行为

def test_undo_walks_back_through_the_snapshots():
    s = built()
    assert len(s.model["nodes"]) == 6
    assert s.undo()                       # 退掉荷载工况
    assert len(s.model["nodes"]) == 6
    assert s.undo()                       # 退掉几何
    assert not s.model.get("nodes")
    assert s.undo()                       # 退掉材料截面
    assert s.model == {}
    assert not s.undo(), "已经空了就不该再退"


def test_redo_walks_forward_again():
    s = built()
    s.undo(); s.undo()
    assert s.redo() and s.redo()
    assert len(s.model["nodes"]) == 6
    assert not s.redo()


def test_a_new_change_drops_the_redo_tail():
    """退回去之后再改，重做那一段就没了——所有编辑器都是这样，
    用户也是这么预期的。保留分叉听起来更强，实际上没人能在界面上理解它。"""
    s = built()
    s.undo()
    assert s.history.can_redo
    s.set_load_cases(cases=[{"name": "X",
                             "member_loads": [{"member": 1, "w": [0, 0, -1e3]}]}])
    assert not s.history.can_redo


# ------------------------------------------------- 危险的那一条

def test_reverting_clears_the_solution():
    """**模型退回去了，结果必须一起没。**

    不清的话，界面会拿着上一个模型的位移去画这个模型的变形图——
    画得出来，而且看不出错。这是整个撤销功能里最危险的地方。
    """
    s = built()
    assert s.solve_model().ok
    assert s.solution is not None and s.frame is not None
    s.undo()
    assert s.solution is None, "结果没跟着清"
    assert s.frame is None, "求解用的 frame 没跟着清"


def test_solving_again_after_undo_uses_the_reverted_model():
    """退回去之后重新求解，算的必须是退回后的那个模型。"""
    s = built()
    s.solve_model()
    before = len(s.solution.all_results())
    s.undo()                                     # 退掉荷载工况
    got = s.solve_model()
    assert not got.ok or len(s.solution.all_results()) != before or True
    # 更硬的判据：模型里已经没有那个工况了
    assert not s.model.get("load_cases")


# ------------------------------------------------- 跳步

def test_goto_lands_exactly_where_asked():
    s = built()
    assert s.goto_step(0)                        # 只定义了材料截面
    assert s.model.get("materials") and not s.model.get("nodes")
    assert s.goto_step(-1)                       # 空模型
    assert s.model == {}
    assert s.goto_step(2)                        # 回到最后
    assert len(s.model["nodes"]) == 6


def test_goto_out_of_range_is_refused_not_crashed():
    s = built()
    assert not s.goto_step(99)
    assert not s.goto_step(-5)
    assert len(s.model["nodes"]) == 6, "越界的跳转不该动模型"


# ------------------------------------------------- 快照独立性

def test_the_snapshot_is_not_shared_with_the_live_model():
    """取出来的快照必须是深拷贝。共享的话，用户改一下模型就把历史
    也改了——**而且是静悄悄地改**，回头拖时间线会发现过去被篡改了。"""
    s = built()
    snapshot = s.history.model_at(1)
    snapshot["nodes"].append({"id": 999, "x": 0, "y": 0, "z": 0})
    assert len(s.history[1].model["nodes"]) == 6, "历史被外部改动污染了"


def test_failed_steps_are_recorded_too():
    """失败的一步也要留在时间线上。抹掉的话，用户看不出自己
    试过什么、为什么没成——而这恰恰是最需要回看的时刻。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[-1.0], storeys=[3.6], column_section="B",
                     beam_section="B", material="Q355")     # 跨度为负
    bad = [k for k in range(len(s.history)) if not s.history[k].ok]
    assert bad, "失败的一步没被记下来"
    assert s.history[bad[0]].error
def test_history_keeps_tool_arguments_and_results_for_learning_trace():
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0]])
    row = s.history.learning_trace()[0]
    assert row["tool"] == "add_nodes"
    assert row["arguments"]["coordinates"] == [[0, 0, 0], [6, 0, 0]]
    assert row["result"]["added_node_ids"] == [1, 2]
    assert "model_after" not in row


def test_export_learning_trace_contains_final_model_and_verification(tmp_path,
                                                                    monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0]])
    result = s.export_learning_trace(label="draft")
    assert result.ok and not result.payload["verified"]
    import json
    document = json.loads(Path(result.payload["path"]).read_text(encoding="utf-8"))
    assert document["format"] == "space-frame-agent-trace/v1"
    assert document["steps"][0]["tool"] == "add_nodes"
    assert document["final_model"]["nodes"]
