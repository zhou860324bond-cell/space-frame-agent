"""真实子Agent闭环：权限、候选隔离、预算恢复、审阅提交与实际悬臂结果。"""
import json
from copy import deepcopy

import pytest

from agent import ScriptedProvider, Session
from agent_task import AgentTask, TaskBudget
from change_preview import model_digest
from conversation import check_pairing


def call(name, **arguments):
    return {"tool_calls": [{"id": name, "name": name, "arguments": arguments}]}


def plan(role="modeler", name="build"):
    return call("set_task_plan", steps=[{"id": name, "title": "实际子任务", "role": role}])


def task(tmp_path, script, children, session=None, **kwargs):
    return AgentTask(session or Session(), ScriptedProvider(script),
                     lambda role, sandbox: children[role], directory=tmp_path, **kwargs)


def cantilever_model():
    return {"units": "N-m-Pa", "materials": [{"name": "S", "E": 2.1e11, "nu": .3}],
            "sections": [{"name": "B", "A": .01, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}],
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 3, "y": 0, "z": 0}],
            "members": [{"id": 1, "i": 1, "j": 2, "material": "S", "section": "B"}],
            "supports": [{"node": 1, "fix": [1]*6}],
            "load_cases": [{"name": "P", "nodal_loads": [{"node": 2, "load": [0, 0, -1000, 0, 0, 0]}]}]}


def built_task(tmp_path, session=None):
    worker = ScriptedProvider([call("set_model", model=cantilever_model()), {"content": "候选已建立"}])
    result = task(tmp_path, [plan(), call("delegate_agent", step_id="build", instruction="按明确参数建模"),
                              {"content": "候选等待审阅"}], {"modeler": worker}, session)
    result.ask("建立指定悬臂")
    return result


def test_child_model_never_changes_main_before_review(tmp_path):
    """旧的单Conversation直接改主模型；子Agent应始终只改候选。"""
    item = built_task(tmp_path)
    assert item.status == "ready"
    assert item.session.model == {} and len(item.session.history) == 0
    assert len(item.children["build"].conversation.session.model["nodes"]) == 2
    assert "建立指定悬臂" in item.children["build"].conversation.object_context
    assert item.preview("build")["diff"]["changed"]


def test_checked_candidate_runs_real_solver_and_matches_closed_form(tmp_path):
    """检查Agent必须读取建模候选，悬臂端位移与PL³/(3EI)相符。"""
    worker = ScriptedProvider([call("set_model", model=cantilever_model()), {"content": "已建立"}])
    reviewer = ScriptedProvider([call("validate_model"), call("solve_model"),
                                 call("query_results", what="max_displacement", case="P"), {"content": "已检查"}])
    item = task(tmp_path, [call("set_task_plan", steps=[
        {"id": "build", "title": "建立悬臂", "role": "modeler"},
        {"id": "review", "title": "检查求解", "role": "reviewer"}]),
        call("delegate_agent", step_id="build", instruction="使用给定全部参数"),
        call("delegate_agent", step_id="review", instruction="检查求解并查询", source="build"),
        {"content": "已检查候选"}], {"modeler": worker, "reviewer": reviewer})
    item.ask("建立并求解悬臂")
    assert item.session.model == {} and item.session.solution is None
    child = item.children["review"]
    evidence = next(e for e in item.events if e.get("tool") == "query_results")
    assert evidence["result"]["magnitude_mm"] == pytest.approx(1000*3**3/(3*2.1e11*1e-5)*1000, rel=1e-5)
    assert evidence["model_hash"] == model_digest(child.conversation.session.model)
    assert child.source == "build" and child.status == "completed"
    assert child.conversation.session.solution is not item.children["build"].conversation.session.solution
    assert next((tmp_path/"review"/"capsules").glob("*.json"))


def test_reviewer_cannot_mutate_model_even_if_model_requests_it(tmp_path):
    """只过滤提示词不够；恶意返回的建模工具也必须在调度层拒绝。"""
    main = Session(model=cantilever_model())
    before = deepcopy(main.model)
    provider = ScriptedProvider([call("edit_node", node_id=2, x=100), {"content": "尝试改动"}])
    item = task(tmp_path, [plan("reviewer", "review"),
        call("delegate_agent", step_id="review", instruction="检查"), {"content": "结束"}],
        {"reviewer": provider}, main)
    item.ask("检查模型")
    assert item.children["review"].conversation.session.model == before
    assert main.model == before
    assert item.status == "waiting_input"
    assert any(e.get("tool") == "edit_node" and not e["ok"] for e in item.events)


def test_manager_cannot_call_core_write_tools(tmp_path):
    """主Agent不能绕过子任务隔离直接对原模型写工具。"""
    item = task(tmp_path, [call("set_model", model=cantilever_model()), {"content": "结束"}], {})
    item.ask("任务")
    assert not item.session.model and not item.children
    assert not item.manager.session.tool_log[-1][2].ok


def test_missing_input_stops_requests_and_choices_are_structured(tmp_path):
    """缺参问题返回后必须等待用户，不能继续消耗模型预算或编造参数。"""
    questions = [{"prompt": "柱脚采用哪种约束？", "choices": ["固接", "铰接", "只建几何"]}]
    item = task(tmp_path, [call("request_input", questions=questions)], {})
    data = item.ask("还没确定柱脚")
    assert data["status"] == "waiting_input" and data["questions"] == questions
    assert item.budget.model_calls == 1 and not item.children
    assert not item.session.model


def test_multiple_agents_share_budget_not_independent_limits(tmp_path):
    """子Agent不能各自重置预算，递归分派不会无限调用。"""
    worker = ScriptedProvider([call("set_model", model=cantilever_model()), {"content": "已建立"}])
    item = task(tmp_path, [plan(), call("delegate_agent", step_id="build", instruction="建模")],
                {"modeler": worker}, budget=TaskBudget(max_model_calls=3))
    item.ask("任务")
    assert item.budget.model_calls == 3 and item.status == "budget_exhausted"
    assert item.session.model == {}


def test_tool_budget_blocks_side_effects_after_limit(tmp_path):
    """预算在执行工具前扣除，不能执行超出预算的候选操作。"""
    worker = ScriptedProvider([call("set_model", model=cantilever_model())])
    item = task(tmp_path, [plan(), call("delegate_agent", step_id="build", instruction="建模"), {"content": "结束"}],
                {"modeler": worker}, budget=TaskBudget(max_tool_calls=2))
    item.ask("任务")
    assert item.status == "budget_exhausted" and item.budget.tool_calls == 2
    assert item.children["build"].conversation.session.model == {}


def test_user_apply_requires_review_and_cannot_repeat(tmp_path):
    """候选不能由模型自己提交，用户应用后重复点击也不能再记历史。"""
    item = built_task(tmp_path)
    preview = item.preview("build")
    with pytest.raises(ValueError, match="明确应用"):
        item.apply("build", preview["candidate_hash"])
    item.status = "running"
    with pytest.raises(ValueError, match="仍在执行"):
        item.apply("build", preview["candidate_hash"], confirmed=True)
    item.status = "ready"
    assert item.apply("build", preview["candidate_hash"], confirmed=True).ok
    history = len(item.session.history)
    assert len(item.session.model["nodes"]) == 2 and item.session.solution is None
    with pytest.raises(ValueError, match="重复"):
        item.apply("build", preview["candidate_hash"], confirmed=True)
    assert len(item.session.history) == history


def test_manual_model_edit_invalidates_old_task_and_candidate(tmp_path):
    """人工编辑后不可恢复旧计划，也不可覆盖人工改动。"""
    item = built_task(tmp_path)
    item.session.model["units"] = "N-mm-MPa"
    with pytest.raises(ValueError, match="失效"):
        item.ask("继续")
    preview = item.preview("build")
    with pytest.raises(ValueError, match="失效"):
        item.apply("build", preview["candidate_hash"], confirmed=True)
    assert item.session.model == {"units": "N-mm-MPa"}


def test_candidate_changed_after_preview_cannot_be_applied(tmp_path):
    """不能只检查主基线，预览后的候选内容也必须重新核对。"""
    item = built_task(tmp_path)
    preview = item.preview("build")
    item.children["build"].conversation.session.model["nodes"][1]["x"] = 4
    with pytest.raises(ValueError, match="候选在预览后"):
        item.apply("build", preview["candidate_hash"], confirmed=True)
    assert not item.session.model


def test_failure_resume_reuses_child_session_without_repeated_build(tmp_path):
    """网络在建模后断开，恢复必须保留已有候选与执行记录。"""
    class Recovering:
        count = 0
        def complete(self, messages, tools):
            self.count += 1
            if self.count == 1:
                return call("set_model", model=cantilever_model())
            if self.count == 2:
                raise TimeoutError("network timeout")
            return {"content": "使用已有候选继续"}
    worker = Recovering()
    manager = [plan(), call("delegate_agent", step_id="build", instruction="建立"), {"content": "可继续"},
               call("delegate_agent", step_id="build", instruction="保留模型继续"), {"content": "结束"}]
    item = task(tmp_path, manager, {"modeler": worker})
    item.ask("建立模型")
    child = item.children["build"]
    assert child.status == "failed"
    item.ask("继续")
    assert item.goal == "建立模型"
    assert "建立模型" in child.conversation.object_context and "继续" in child.conversation.object_context
    assert item.children["build"] is child
    assert sum(name == "set_model" for name, _args, _result in child.conversation.session.tool_log) == 1
    assert not check_pairing(child.conversation.build_messages("下一步"))
    assert item.status == "ready" and item.session.model == {}


def test_cancellation_between_tools_preserves_main_and_pairing(tmp_path):
    """点击停止后不执行后续写工具，原工程与上下文配对保持。"""
    worker = ScriptedProvider([call("set_model", model=cantilever_model()),
                               call("edit_node", node_id=2, x=999), {"content": "结束"}])
    item = task(tmp_path, [plan(), call("delegate_agent", step_id="build", instruction="建立"),
                           {"content": "结束"}], {"modeler": worker})
    def stop_after_tool(name, payload, ok):
        if payload["record"].get("tool") == "set_model":
            item.cancel()
    data = item.ask("任务", on_event=stop_after_tool)
    assert data["status"] == "cancelled" and item.session.model == {}
    assert item.children["build"].conversation.session.model["nodes"][1]["x"] == 3
    assert not check_pairing(item.manager.build_messages("继续"))


@pytest.mark.parametrize("steps", [[], [{"id": "../outside", "title": "bad", "role": "modeler"}],
                                  [{"id": "a", "title": "bad", "role": "unknown"}]])
def test_invalid_plan_cannot_escape_task_directory(tmp_path, steps):
    """模型返回的角色与步骤编号必须校验，不能变成文件路径穿越。"""
    item = task(tmp_path, [call("set_task_plan", steps=steps), {"content": "结束"}], {})
    item.ask("任务")
    assert not item.steps and not item.children


@pytest.mark.parametrize("secret", ["sk-agent-task-test-secret", 'sk-agent-"quote\\secret'])
def test_task_audit_never_writes_known_key(tmp_path, secret):
    """子模型异常和主回复可能回显密钥，审计文件必须先隐藏。"""
    item = task(tmp_path, [{"content": f"error {secret}"}], {}, secret_keys=[secret])
    item.ask(f"任务 {secret}")
    text = (tmp_path/"task.json").read_text(encoding="utf-8")
    assert secret not in text and "密钥已隐藏" in text
    assert json.loads(text)["schema"] == "engineering-agent-task/v1"
    assert secret not in json.dumps(item.snapshot(), ensure_ascii=False)
    assert secret not in json.dumps(item.manager.provider.provider.seen, ensure_ascii=False)


def test_unexecuted_plan_step_cannot_claim_task_complete(tmp_path):
    """建立计划卡不代表执行完成，漏分派的步骤仍要显示待继续。"""
    worker = ScriptedProvider([call("set_model", model=cantilever_model()), {"content": "已建立"}])
    item = task(tmp_path, [call("set_task_plan", steps=[
        {"id": "build", "title": "建立", "role": "modeler"},
        {"id": "review", "title": "检查", "role": "reviewer"}]),
        call("delegate_agent", step_id="build", instruction="建立"), {"content": "完成"}],
        {"modeler": worker})
    assert item.ask("任务")["status"] == "needs_continuation"


def test_child_generation_does_not_assume_fixed_supports(tmp_path):
    """几何子任务遗漏base时不能套用生成器的默认固接支座。"""
    worker = ScriptedProvider([call("generate_frame", spans=[6], storeys=[3]), {"content": "需要支座形式"}])
    item = task(tmp_path, [plan(), call("delegate_agent", step_id="build", instruction="只建立几何"),
                           {"content": "需补充"}], {"modeler": worker})
    item.ask("几何任务")
    assert not item.children["build"].conversation.session.model
    assert not item.children["build"].conversation.session.tool_log[-1][2].ok


def test_review_source_must_be_explicit_after_building_candidate(tmp_path):
    """有候选时不能把原模型检查结果悄悄当成候选结果。"""
    item = built_task(tmp_path)
    item.steps.append({"id": "review", "role": "reviewer", "title": "检查"})
    with pytest.raises(ValueError, match="明确检查"):
        item.delegate_agent("review", "检查")
    assert "review" not in item.children


def test_audit_failure_does_not_hide_successfully_applied_model(tmp_path, monkeypatch):
    """审计磁盘错误不能令已应用模型缺少刷新或被重复提交。"""
    item = built_task(tmp_path)
    preview = item.preview("build")
    def fail():
        raise OSError("disk full")
    monkeypatch.setattr(item, "save", fail)
    assert item.apply("build", preview["candidate_hash"], confirmed=True).ok
    assert item.applied and item.audit_error
    assert len(item.session.model["nodes"]) == 2


def test_idempotent_build_completes_without_empty_candidate(tmp_path):
    """用户重复要求现有模型时，不应伪造候选差异或不断要求补参。"""
    main = Session()
    assert main.set_model(cantilever_model()).ok
    before = deepcopy(main.model)
    item = built_task(tmp_path, main)
    assert item.status == "completed" and not item.candidates()
    assert main.model == before
