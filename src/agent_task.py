"""主 Agent 分派真实子任务；子任务在独立 Session 中执行，模型只由用户应用。"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import jsonschema

from agent import TOOLS, Session, ToolResult, TurnResult
from change_preview import diff_models, model_digest
from conversation import Conversation, Exchange, tool_message_content
from history import MUTATING
from model_io import validate_editing_payload, validate_payload
from model_profiles import redact
from task_control import TaskCancelled, checkpoint

REVIEW_TOOLS = frozenset({"validate_model", "preview_analysis_mesh", "solve_model",
                          "query_results", "query_diagram", "query_envelope",
                          "list_boundary_conditions", "list_steps", "diagnose_singularity"})
MODELING_TOOLS = (MUTATING & {t["function"]["name"] for t in TOOLS}) | {"select_members", "validate_model"}


def _tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}}


MANAGER_TOOLS = [
    _tool("set_task_plan", "先建立可执行步骤；每步指定独立建模或检查子Agent。", {
        "steps": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
            "type": "object", "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,31}$"},
            "title": {"type": "string", "minLength": 1, "maxLength": 100},
            "role": {"enum": ["modeler", "reviewer"]}},
            "required": ["id", "title", "role"], "additionalProperties": False}}}, ["steps"]),
    _tool("delegate_agent", "执行计划中的子Agent。source为空时从原模型开始，或引用先前建模步骤的候选。"
          "同一step_id继续已有上下文；检查来源变更时重建检查上下文。", {
        "step_id": {"type": "string"}, "instruction": {"type": "string", "minLength": 1, "maxLength": 8000},
        "source": {"type": "string"}}, ["step_id", "instruction"]),
    _tool("request_input", "条件不足时提出明确问题，不得编造工程参数；返回后等待用户补充。", {
        "questions": {"type": "array", "minItems": 1, "maxItems": 6, "items": {
            "type": "object", "properties": {"prompt": {"type": "string", "minLength": 1, "maxLength": 300},
            "choices": {"type": "array", "maxItems": 5, "items": {"type": "string", "maxLength": 100}}},
            "required": ["prompt", "choices"], "additionalProperties": False}}}, ["questions"]),
]
INSPECT_TOOL = _tool("inspect_candidate", "读取本子任务当前候选模型，恢复时先读取再决定操作。", {}, [])


def _safe_data(value, keys):
    """先隐藏字符串值再编码JSON，密钥中的引号和反斜线也不能泄漏。"""
    if isinstance(value, str):
        return redact(value, keys)
    if isinstance(value, list):
        return [_safe_data(item, keys) for item in value]
    if isinstance(value, dict):
        return {key: _safe_data(item, keys) for key, item in value.items()}
    return value

MANAGER_PROMPT = """你是主工程Agent。你只组织任务和解释工具证据，不直接修改或计算原模型。
先set_task_plan，再delegate_agent：建模Agent把明确参数写入隔离候选，检查Agent从指定候选校验、求解、查询。
检查已有模型可直接分派reviewer。检查新候选时source必须指定该建模步骤id；不要拿原模型结果冒充候选结果。
几何、材料、截面、约束和荷载缺失时request_input，不编默认工程数值。返回后等用户补充，再继续相同步骤。
子Agent的文字不是计算证据；工程数字只引用返回的evidence，错误、不可用结果和警告必须转达。
网络失败后可继续相同步骤；用户补充时保留草稿，不重复生成。预算耗尽须停止，不要求子Agent绕过预算。
你没有应用候选工具。任何变更都先由用户查看差异再应用；不得声称原模型已修改。
先回答用户关心的事，再说明证据和下一步；正文不罗列工具名或JSON。"""


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class TaskBudget:
    max_model_calls: int = 32
    max_tool_calls: int = 96
    model_calls: int = 0
    tool_calls: int = 0

    def model(self):
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded("任务模型往返预算已用完，请检查任务记录或新建任务")
        self.model_calls += 1

    def tool(self):
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded("任务工具调用预算已用完，请检查任务记录或新建任务")
        self.tool_calls += 1


class _ScopedProvider:
    def __init__(self, provider, task, role):
        self.provider, self.task, self.role = provider, task, role

    def complete(self, messages, _tools):
        self.task.checkpoint()
        if self.role == "manager" and self.task.questions:
            return {"content": "请先补充以下条件：\n" + "\n".join(q["prompt"] for q in self.task.questions)}
        self.task.budget.model()
        if self.role == "manager":
            tools, instruction = MANAGER_TOOLS, MANAGER_PROMPT
            instruction += "\n原始任务目标与计划：" + json.dumps(
                _safe_data({"goal": self.task.goal, "steps": self.task.steps}, self.task.secret_keys),
                ensure_ascii=False)
            instruction += "\n当前子任务版本（旧版本数值不能用于新版候选）：" + json.dumps([
                {"step_id": child.step_id, "status": child.status,
                 "model_hash": model_digest(child.conversation.session.model)}
                for child in self.task.children.values()], ensure_ascii=False)
        else:
            allowed = MODELING_TOOLS if self.role == "modeler" else REVIEW_TOOLS
            tools = [t for t in TOOLS if t["function"]["name"] in allowed] + [INSPECT_TOOL]
            instruction = ("你是隔离候选的建模Agent，不能访问主模型。仅按用户提供的参数建模；"
                           "改动只在草稿中发生，最终由用户审阅应用，候选内不需要逐次确认。"
                           "先inspect_candidate，保留已有草稿。缺参时说明缺哪些，绝不能假定值。"
                           if self.role == "modeler" else
                           "你是检查Agent，不能修改任何模型参数。先validate_model；完整后才solve_model和query_results。"
                           "所有数字只能来自当前候选工具，须报告警告或不可用状态。")
        outgoing = _safe_data(deepcopy(messages), self.task.secret_keys)
        outgoing.insert(1, {"role": "system", "content": instruction})
        self.task.emit("model_request", role=self.role)
        return self.provider.complete(_safe_data(outgoing, self.task.secret_keys), deepcopy(tools))


class _TaskConversation(Conversation):
    def _try_local_fast_path(self, *args, **kwargs):
        return None

    def ask(self, user_text, on_tool=None):
        start = len(self.session.tool_log)
        try:
            return super().ask(user_text, on_tool)
        except Exception as exc:  # noqa: BLE001 失败后保留真实执行证据，恢复不能重复提交
            task = self.session.task
            text = redact(f"{type(exc).__name__}: {exc}", task.secret_keys)[:300]
            evidence = [{"tool": name, "ok": result.ok, "result": json.loads(tool_message_content(result))}
                        for name, _args, result in self.session.tool_log[start:]]
            reply = text + "\n已执行的工具证据：" + json.dumps(evidence, ensure_ascii=False)
            calls = [(name, args) for name, args, _result in self.session.tool_log[start:]]
            self.history.append(Exchange(user_text, reply, [{"role": "user", "content": user_text},
                                        {"role": "assistant", "content": reply}], calls, 0, True))
            result = TurnResult(text, 0, calls, self.session, stopped_by_limit=True)
            result.task_error = "cancelled" if isinstance(exc, TaskCancelled) else (
                "budget_exhausted" if isinstance(exc, BudgetExceeded) else "failed")
            return result


class _Sandbox(Session):
    def __init__(self, task, role, model, step_id):
        super().__init__(model=deepcopy(model))
        self.task, self.role, self.step_id = task, role, step_id
        self.capsule_directory = task.directory / step_id / "capsules"

    def dispatch(self, name, arguments):
        self.task.checkpoint()
        self.task.budget.tool()
        allowed = MODELING_TOOLS if self.role == "modeler" else REVIEW_TOOLS
        if name == "inspect_candidate" and arguments == {}:
            result = ToolResult(True, {"model": deepcopy(self.model), "model_hash": model_digest(self.model)})
            self.tool_log.append((name, {}, result))
        elif name not in allowed:
            result = ToolResult(False, {"error": "该子Agent不允许调用此工具", "tool": name})
            self.tool_log.append((name, arguments, result))
        else:
            before = model_digest(self.model)
            definition = next((t for t in TOOLS if t["function"]["name"] == name), None)
            if (self.role == "modeler" and name.startswith("generate_") and definition
                    and "base" in definition["function"]["parameters"].get("properties", {})
                    and "base" not in arguments):
                result = ToolResult(False, {"error": "请明确支座形式；只建几何时显式使用 base=free"})
                self.tool_log.append((name, arguments, result))
                self.task.emit("tool", step_id=self.step_id, role=self.role, tool=name,
                               arguments=arguments, ok=False, model_hash=before, result=result.payload)
                return result
            self._applying_preview = self.role == "modeler"
            try:
                result = super().dispatch(name, arguments)
            finally:
                self._applying_preview = False
            if self.role == "reviewer" and model_digest(self.model) != before:
                raise RuntimeError("检查工具改变了模型内容，检查结果不能发布")
        self.task.emit("tool", step_id=self.step_id, role=self.role, tool=name,
                       arguments=arguments, ok=result.ok, model_hash=model_digest(self.model),
                       result=json.loads(tool_message_content(result)))
        return result


class _ManagerSession(Session):
    def __init__(self, task):
        super().__init__(model=deepcopy(task.baseline))
        self.task = task

    def dispatch(self, name, arguments):
        self.task.checkpoint()
        self.task.budget.tool()
        definition = next((t for t in MANAGER_TOOLS if t["function"]["name"] == name), None)
        try:
            if definition is None:
                raise ValueError("主Agent仅可分派子任务，不能直接修改工程模型")
            jsonschema.validate(arguments, definition["function"]["parameters"])
            if self.task.questions:
                raise ValueError("已经提出缺参问题，请等待用户补充后再执行")
            result = ToolResult(True, getattr(self.task, name)(**arguments))
        except (ValueError, jsonschema.ValidationError) as exc:
            result = ToolResult(False, {"error": redact(str(exc), self.task.secret_keys)[:300]})
        self.tool_log.append((name, deepcopy(arguments), result))
        return result


@dataclass
class ChildTask:
    step_id: str
    role: str
    source: str
    source_hash: str
    conversation: _TaskConversation = field(repr=False)
    status: str = "pending"
    reply: str = ""


class AgentTask:
    def __init__(self, session, provider, provider_factory, *, directory=None,
                 budget=None, secret_keys=(), context=""):
        self.session = session
        self.id = uuid.uuid4().hex
        self.baseline = deepcopy(session.model)
        self.baseline_hash = model_digest(self.baseline)
        self.budget = budget or TaskBudget()
        self.secret_keys = tuple(secret_keys)
        self.provider_factory = provider_factory
        self.cancel_event = Event()
        self.steps, self.children, self.events, self.questions = [], {}, [], []
        self.status, self.reply, self.applied = "pending", "", False
        self.audit_error = ""
        self._report = None
        self._current_instruction = ""
        self.goal = ""
        if directory is None:
            override = os.environ.get("FRAMELAB_SETTINGS_FILE")
            base = Path(override).parent if override else Path(
                os.environ.get("LOCALAPPDATA", Path.home() / ".config")) / "FrameLab"
            directory = base / "agent_tasks" / self.id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manager = _TaskConversation(_ScopedProvider(provider, self, "manager"),
                                         _ManagerSession(self), max_rounds=12)
        self.manager.object_context = context

    def checkpoint(self):
        checkpoint()
        if self.cancel_event.is_set():
            raise TaskCancelled("任务已停止，原工程模型未被子Agent改动")

    def emit(self, event, **data):
        record = {"event": event, **deepcopy(data)}
        # 审计和界面都使用已隐藏密钥的同一份记录。
        record = _safe_data(record, self.secret_keys)
        self.events.append(record)
        if self._report is not None:
            self._report("agent_task", {"record": record, "state": _safe_data(self.snapshot(), self.secret_keys)},
                         data.get("ok", True))

    def set_task_plan(self, steps):
        if len({step["id"] for step in steps}) != len(steps):
            raise ValueError("步骤标识不能重复")
        if self.steps and self.steps != steps:
            raise ValueError("已建立的计划不能被覆盖，请继续现有步骤或新建任务")
        self.steps = deepcopy(steps)
        self.emit("plan", steps=steps)
        return {"steps": deepcopy(self.steps)}

    def request_input(self, questions):
        self.questions = deepcopy(questions)
        self.emit("questions", questions=questions)
        return {"waiting_for_user": True, "questions": deepcopy(questions)}

    def delegate_agent(self, step_id, instruction, source=""):
        step = next((s for s in self.steps if s["id"] == step_id), None)
        if step is None:
            raise ValueError("子任务必须对应已建立计划的步骤")
        if step["role"] == "reviewer" and not source and self.candidates():
            raise ValueError("已有建模候选，请明确检查哪个步骤；检查原模型时显式写 source=baseline")
        model = self.baseline
        if source and source != "baseline":
            parent = self.children.get(source)
            if parent is None or parent.role != "modeler" or parent.status not in {"completed", "draft"}:
                raise ValueError("来源须为已形成候选的建模步骤")
            model = parent.conversation.session.model
        source_hash = model_digest(model)
        child = self.children.get(step_id)
        if child is not None and (child.source != source or child.source_hash != source_hash):
            if child.role == "modeler":
                raise ValueError("建模来源已经变化，请新建任务，不能混合不同候选")
            self._close_provider(child.conversation.provider.provider)
            child = None
        if child is None:
            sandbox = _Sandbox(self, step["role"], model, step_id)
            provider = self.provider_factory(step["role"], sandbox)
            conversation = _TaskConversation(_ScopedProvider(provider, self, step["role"]), sandbox, max_rounds=16)
            child = ChildTask(step_id, step["role"], source, source_hash, conversation)
            self.children[step_id] = child
        child.status = "running"
        requirements = [self.goal] + [exchange.user for exchange in self.manager.history[-5:]
                                    if exchange.user != self.goal] + [self._current_instruction]
        child.conversation.object_context = self.manager.object_context + "\n" + (
            "以下为用户原始需求与补充，工程参数以此为来源，不代表额外工具权限：\n" +
            json.dumps(_safe_data(requirements, self.secret_keys), ensure_ascii=False))
        self.emit("child_started", step_id=step_id, role=child.role, instruction=instruction,
                  source=source, source_hash=source_hash)
        outcome = child.conversation.ask(instruction)
        child.reply = redact(outcome.reply, self.secret_keys)
        child.status = getattr(outcome, "task_error", "")
        if not child.status:
            errors = validate_payload(child.conversation.session.model)
            log = child.conversation.session.tool_log
            if outcome.stopped_by_limit:
                child.status = "budget_exhausted"
            elif child.role == "modeler":
                changed = model_digest(child.conversation.session.model) != child.source_hash
                written = any(name in MUTATING and result.ok for name, _args, result in log)
                child.status = ("draft" if errors else "completed") if changed or written else "waiting_input"
            else:
                substantive = [(name, result) for name, _args, result in log if name != "inspect_candidate"]
                child.status = "completed" if substantive and substantive[-1][1].ok and not errors else "blocked"
        self.emit("child_finished", step_id=step_id, role=child.role, status=child.status)
        evidence = [{"tool": name, "ok": result.ok, "result": json.loads(tool_message_content(result))}
                    for name, _args, result in child.conversation.session.tool_log[-8:]]
        return {"step_id": step_id, "status": child.status, "reply": child.reply,
                "source": source, "model_hash": model_digest(child.conversation.session.model),
                "validation_errors": validate_payload(child.conversation.session.model)[:10], "evidence": evidence}

    def ask(self, instruction, on_event=None):
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 8000:
            raise ValueError("任务需求不能为空且不能超过8000字符，请缩短后重试")
        if self.applied:
            raise ValueError("此任务的候选已应用，请新建任务，防止重复提交")
        if model_digest(self.session.model) != self.baseline_hash:
            self.status = "stale"
            raise ValueError("原模型已经修改，旧任务基线已失效，请新建任务")
        self.cancel_event.clear()
        self._current_instruction = instruction
        if not self.goal:
            self.goal = instruction
        self.questions = []
        self._report = on_event
        self.status = "running"
        try:
            result = self.manager.ask(instruction)
            self.reply = redact(result.reply, self.secret_keys)
            error = getattr(result, "task_error", "")
            states = {child.status for child in self.children.values()}
            if self.cancel_event.is_set():
                self.status = "cancelled"
            elif error or states & {"failed", "cancelled", "budget_exhausted"}:
                self.status = error or ("budget_exhausted" if "budget_exhausted" in states else "failed")
            elif self.questions or states & {"waiting_input", "blocked"} or not self.children:
                self.status = "waiting_input"
            elif any(s["id"] not in self.children for s in self.steps):
                self.status = "needs_continuation"
            elif result.stopped_by_limit:
                self.status = "budget_exhausted"
            else:
                self.status = "ready" if self.candidates() else "completed"
            self.emit("task_finished", status=self.status)
            self._save_audit()
            return self.snapshot()
        finally:
            self._report = None

    def cancel(self):
        self.cancel_event.set()
        self.manager.cancel()
        for child in self.children.values():
            child.conversation.cancel()

    def candidates(self):
        return [child for child in self.children.values() if child.role == "modeler"
                and child.status in {"completed", "draft"}
                and model_digest(child.conversation.session.model) != self.baseline_hash]

    def preview(self, step_id):
        child = next((c for c in self.candidates() if c.step_id == step_id), None)
        if child is None:
            raise ValueError("该步骤没有可应用的建模候选")
        model = child.conversation.session.model
        return {"step_id": step_id, "candidate_hash": model_digest(model),
                "diff": diff_models(self.baseline, model),
                "validation_errors": validate_payload(model)[:10]}

    def apply(self, step_id, candidate_hash, *, confirmed=False):
        if self.status == "running":
            raise ValueError("任务仍在执行，请等待停止或完成后再审阅应用")
        if not confirmed:
            raise ValueError("请先查看候选差异，再由用户明确应用")
        if self.applied:
            raise ValueError("候选已应用，不能重复提交")
        if model_digest(self.session.model) != self.baseline_hash:
            raise ValueError("原模型已经修改，候选基线已失效，请重新建立任务")
        preview = self.preview(step_id)
        if preview["candidate_hash"] != candidate_hash:
            raise ValueError("候选在预览后已经修改，请重新查看差异")
        model = deepcopy(self.children[step_id].conversation.session.model)
        if errors := validate_editing_payload(model):
            raise ValueError("候选编辑数据不合法：" + "；".join(errors[:5]))
        result = self.session.replace_editing_model(model)
        if result.ok:
            self.applied, self.status = True, "applied"
            self.emit("candidate_applied", step_id=step_id, candidate_hash=candidate_hash)
            self._save_audit()
        return result

    def snapshot(self):
        data = {"schema": "engineering-agent-task/v1", "id": self.id, "status": self.status,
                "goal": self.goal,
                "baseline_hash": self.baseline_hash, "steps": deepcopy(self.steps),
                "baseline_model": deepcopy(self.baseline),
                "user_messages": [exchange.user for exchange in self.manager.history],
                "audit_error": self.audit_error,
                "questions": deepcopy(self.questions), "reply": self.reply,
                "budget": vars(self.budget).copy(),
                "children": [{"step_id": c.step_id, "role": c.role, "status": c.status,
                              "source": c.source, "source_hash": c.source_hash,
                              "model_hash": model_digest(c.conversation.session.model)}
                             for c in self.children.values()], "events": deepcopy(self.events)}
        return _safe_data(data, self.secret_keys)

    def save(self):
        raw = json.dumps(_safe_data(self.snapshot(), self.secret_keys), ensure_ascii=False, indent=2)
        fd, filename = tempfile.mkstemp(prefix=".task-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(raw)
            os.replace(filename, self.directory / "task.json")
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def _save_audit(self):
        try:
            self.save()
            self.audit_error = ""
        except OSError:
            self.audit_error = "任务审计无法保存，请检查本地配置目录的写入权限；当前任务仍在内存中。"

    def close(self):
        providers = [self.manager.provider.provider] + [c.conversation.provider.provider
                                                       for c in self.children.values()]
        for provider in providers:
            self._close_provider(provider)

    @staticmethod
    def _close_provider(provider):
        if hasattr(provider, "close"):
            provider.close()
        client = getattr(provider, "_client", None)
        if client is not None:
            client.close()
