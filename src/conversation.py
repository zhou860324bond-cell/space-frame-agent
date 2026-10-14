"""多轮对话。

原来的 `run_turn` 每次调用都从空白 messages 开始——它是**单轮**的。
用户说完"三跨两层的框架"，它建模算完；接着说"柱子加大一点再算一遍"，
它完全不记得刚才算过什么。那不是 Agent，是个一次性的自然语言编译器。

这一层补的就是记忆，两样东西都要留住：

1. **对话历史** —— 模型看得见前几轮说过什么；
2. **模型状态** —— `Session` 跨轮复用，所以"再算一遍"是在同一个刚架上改，
   而不是重建一个。

真正的工程难点在**裁剪**。上下文会越滚越长（一次求解的工具返回就上千 token），
必须丢掉旧消息；而 OpenAI 兼容接口有一条硬约束：

    每条 role="tool" 的消息，前面必须有请求它的那条 assistant.tool_calls；
    每条 assistant.tool_calls，后面必须跟齐它请求的**全部** tool 结果。

朴素的"保留最后 N 条"会把这对配对拆散，接口直接返回 400。所以裁剪以
**回合**为单位：一个回合 = 一条 user 消息，加上它引出的全部 assistant/tool 消息。
要么整段留，要么整段丢。`test_conversation.py` 里有一条专门验这个不变量。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from agent import (MAX_ROUNDS, SYSTEM_PROMPT, TOOLS, Provider, Session,
                   TurnResult)
from workflow import refresh_workflow_message, workflow_message

# 默认保留多少个回合。一个回合可能有十几条消息，六个回合已经够
# "改一下再算"这类连续追问了，再多只是在烧 token。
DEFAULT_KEEP_TURNS = 6
# 回合数只是第一道闸；一次内力查询可能比六个普通回合还大，因此再加字符预算。
# 预算只作用于旧历史，当前请求和完整系统约束永远保留。
DEFAULT_CONTEXT_CHARS = 24_000
TOOL_MESSAGE_MAX_CHARS = 6_000

# 计划清单工具。只在桌面对话里开启：它不改模型、不产数值，只让界面把
# "先建模→再指派截面→再求解"这样的多步任务列成清单逐项打勾。评测集与单轮
# 调用不开，避免改变那边的工具集与提示。
PLAN_TOOL = {"type": "function", "function": {
    "name": "update_plan",
    "description": ("多步任务开始前列出计划，之后每完成一步再调用一次更新状态。"
                    "只用于向用户展示进度，不修改模型。单步问答不要调用。"),
    "parameters": {"type": "object", "properties": {
        "steps": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string", "description": "这一步要做什么，简短中文"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "done"]}},
            "required": ["title", "status"]}}},
        "required": ["steps"]}}}

_PLAN_HINT = ("需要三步及以上工具调用的任务，先调用 update_plan 列出计划；"
              "每完成一步调用 update_plan 更新状态。计划只给用户看进度，不能代替工具调用。")

_CONTINUATION_HINT = """
你正在一次**多轮对话**中。此前的模型状态还在：材料、截面、节点、杆件、荷载
都保留着，不必重建。用户说"改一下""再算一遍"时，用 add_members、set_load_cases、
set_model 等工具在现有模型上改动，然后重新 solve_model，**不要从头生成一个新模型**。
如果用户的指代不明（"那根梁"是哪根），就追问，不要猜。
""".strip()


@dataclass
class Exchange:
    """一个回合：用户说了什么，模型答了什么，中间调了哪些工具。"""
    user: str
    reply: str
    messages: list[dict]                      # 该回合产生的全部消息（含 user）
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    rounds: int = 0
    stopped_by_limit: bool = False


def _tool_call_ids(message: dict) -> set[str]:
    return {c["id"] for c in message.get("tool_calls") or []}


def check_pairing(messages: list[dict]) -> list[str]:
    """检查 tool_call 配对是否完整，返回问题清单（空列表表示没问题）。

    这是发给接口前的自检。裁剪逻辑写错时，这里会先报出来，
    而不是等接口回一个 400 让人去猜哪里断了。
    """
    problems: list[str] = []
    pending: set[str] = set()
    for k, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            if pending:
                problems.append(f"第 {k} 条 assistant 之前还有未应答的调用 {sorted(pending)}")
            pending = _tool_call_ids(m)
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid not in pending:
                problems.append(f"第 {k} 条 tool 结果 {cid} 没有对应的 assistant 调用")
            else:
                pending.discard(cid)
        elif pending:
            problems.append(f"第 {k} 条 {role} 打断了未应答的调用 {sorted(pending)}")
    if pending:
        problems.append(f"末尾还有未应答的调用 {sorted(pending)}")
    return problems


def _message_chars(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def _compact_value(value: Any, list_edge: int = 10) -> Any:
    """缩短只供模型阅读的工具回包，Session 中的完整结果不受影响。"""
    if isinstance(value, dict):
        return {str(key): _compact_value(item, list_edge)
                for key, item in value.items()}
    if isinstance(value, list):
        if len(value) <= list_edge * 2:
            return [_compact_value(item, list_edge) for item in value]
        return ([_compact_value(item, list_edge) for item in value[:list_edge]]
                + [{"_omitted_items": len(value) - list_edge * 2}]
                + [_compact_value(item, list_edge) for item in value[-list_edge:]])
    if isinstance(value, str) and len(value) > 2_000:
        return value[:1_000] + f"…（省略 {len(value) - 2_000} 字符）…" + value[-1_000:]
    return value


def tool_message_content(result, max_chars: int = TOOL_MESSAGE_MAX_CHARS) -> str:
    """返回有效 JSON；大数组仅在 LLM 上下文中折叠，避免后续每轮反复上传。"""
    raw = result.to_json()
    if len(raw) <= max_chars:
        return raw
    compact = {"ok": result.ok, **_compact_value(result.payload)}
    compact["_context_note"] = (
        "回包为降低模型等待时间已折叠长数组；完整结果仍保存在 Session 和界面中。")
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))



def result_brief(result, max_chars: int = 900) -> str:
    """工具结果的短摘要，供界面卡片展开查看；完整结果仍在 Session 里。"""
    payload = result.payload if isinstance(result.payload, dict) else {"value": result.payload}

    def rounded(value):
        if isinstance(value, float):
            return float(f"{value:.6g}")
        if isinstance(value, list):
            return [rounded(v) for v in value]
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        return value

    # 只影响界面摘要的显示位数；模型上下文与 Session 中的结果保持全精度
    text = json.dumps(rounded(_compact_value(payload, list_edge=3)), ensure_ascii=False, indent=1)
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


def _plan_steps(arguments: dict) -> list[dict] | None:
    steps = arguments.get("steps") if isinstance(arguments, dict) else None
    if not isinstance(steps, list) or not steps or len(steps) > 20:
        return None
    clean = []
    for step in steps:
        if not isinstance(step, dict) or not str(step.get("title", "")).strip():
            return None
        status = step.get("status")
        clean.append({"title": str(step["title"]).strip()[:80],
                      "status": status if status in ("pending", "in_progress", "done") else "pending"})
    return clean


def _cancelled_message(session, calls: list[tuple[str, dict]]) -> str:
    """用户按了停止之后给的话。

    要说清楚**已经改了什么**——"已取消"三个字会让人以为什么都没发生，
    而实际上前几个工具调用已经改了模型。
    """
    names = []
    for name, _ in calls:
        if name not in names:
            names.append(name)
    if not names:
        return "已停止，模型未做任何改动。"
    model = session.model
    scale = (f"当前模型 {len(model.get('nodes') or [])} 节点 "
             f"{len(model.get('members') or [])} 杆件") if model.get("nodes") \
        else "当前模型为空"
    return (f"已停止。停止前已执行 {len(calls)} 次工具调用"
            f"（{'、'.join(names[:6])}），{scale}。"
            "这些改动已经生效，要退回请按 Ctrl+Z。")


def _stalled_message(session, calls: list[tuple[str, dict]], rounds: int) -> str:
    """轮数用尽时给用户的话。

    重点是**说清楚现在是什么状态**，而不是笼统地报一句失败：
    模型建了没有、算了没有、下一步该做什么。
    """
    done = []
    model = session.model
    if model.get("nodes"):
        done.append(f"模型已建好（{len(model['nodes'])} 节点 "
                    f"{len(model.get('members') or [])} 杆件）")
    if session.solution:
        done.append(f"已求解 {len(list(session.solution.all_results()))} 个工况/组合")
    names = []
    for name, _ in calls:
        if name not in names:
            names.append(name)

    head = f"这一轮调了 {len(calls)} 次工具（{rounds} 轮）还没收尾，先停下来。"
    body = ("已经完成的：" + "；".join(done) + "。") if done else "模型还没建起来。"
    tail = ("这些结果是真的，没有白跑——直接接着问下一步就行，不必重来。"
            if done else "把要求拆成一两句再说一次，通常就过去了。")
    if names:
        body += f"（用到：{'、'.join(names[:8])}）"
    return f"{head}{body}{tail}"


class Conversation:
    """一段多轮对话。Session 跨轮复用，所以模型状态是连续的。"""

    def __init__(self, provider: Provider, session: Session | None = None,
                 keep_turns: int = DEFAULT_KEEP_TURNS,
                 max_rounds: int = MAX_ROUNDS,
                 context_chars: int = DEFAULT_CONTEXT_CHARS,
                 plan_tool: bool = False):
        if keep_turns < 1:
            raise ValueError("keep_turns 至少为 1")
        if context_chars < 1:
            raise ValueError("context_chars 必须大于 0")
        self.provider = provider
        self.session = session or Session()
        self.keep_turns = keep_turns
        self.max_rounds = max_rounds
        self.context_chars = context_chars
        self.history: list[Exchange] = []
        self.object_context = ""
        self.plan_tool = plan_tool
        self.plan: list[dict] = []
        self._on_event = None
        # 界面上的"停止"置位它。**只在轮与轮之间检查**——
        # 一次工具调用中途掐断，模型可能停在改了一半的状态，
        # 那比多跑一轮糟糕得多
        self._cancelled = False

    # --- 上下文 ---

    def system_messages(self) -> list[dict]:
        """系统提示。第二轮起追加一段"你在多轮对话里"的说明。

        不把它并进 SYSTEM_PROMPT，是因为单轮调用（评测集就是单轮）
        不该看到这段——那会让它以为存在并不存在的历史状态。
        """
        out = [{"role": "system", "content": SYSTEM_PROMPT},
               workflow_message(self.session)]
        if self.plan_tool:
            out.append({"role": "system", "content": _PLAN_HINT})
        if self.history:
            out.append({"role": "system", "content": _CONTINUATION_HINT})
        if self.object_context:
            out.append({"role": "system", "content": self.object_context})
        return out

    def build_messages(self, user_text: str) -> list[dict]:
        """拼出这一轮要发给模型的完整 messages。

        裁剪以回合为单位：要么整段留，要么整段丢。**绝不从回合中间切开**——
        那会拆散 assistant.tool_calls 与它的 tool 结果。
        """
        candidates = self.history[-self.keep_turns:] if self.keep_turns else []
        base = self.system_messages()
        current = {"role": "user", "content": user_text}
        budget = max(0, self.context_chars - _message_chars(base + [current]))
        kept_reversed: list[Exchange] = []
        used = 0
        for ex in reversed(candidates):
            size = _message_chars(ex.messages)
            # 至少保留最近一轮，延续“它/那根梁/再算”的指代；更早历史严格受预算约束。
            if kept_reversed and used + size > budget:
                break
            kept_reversed.append(ex)
            used += size
        messages = base
        for ex in reversed(kept_reversed):
            messages.extend(ex.messages)
        messages.append(current)
        return messages

    def _finish_local(self, user_text: str, text: str,
                      calls: list[tuple[str, dict]], started: float,
                      on_tool=None) -> TurnResult:
        messages = [{"role": "user", "content": user_text},
                    {"role": "assistant", "content": text}]
        self.history.append(Exchange(user_text, text, messages, calls, 0))
        return TurnResult(text, 0, calls, self.session, metrics={
            "total_ms": (time.perf_counter() - started) * 1000,
            "provider_ms": 0.0,
            "tool_ms": 0.0,
            "provider_calls": 0,
            "prompt_chars": 0,
            "fast_path": True,
        })

    def _try_local_fast_path(self, user_text: str, started: float,
                             on_tool=None) -> TurnResult | None:
        """确定性处理极常见的短命令；其余语言理解仍交给模型。

        边界刻意很窄，避免把“把荷载改了再算”误判成无修改的重新求解。
        """
        normalized = re.sub(r"[\s，。！？、,.!?]", "", user_text).lower()
        if normalized in {"下一步", "下一步呢", "next", "next可以干什么",
                          "下一步可以干什么"}:
            from workflow import WorkflowPhase, inspect_workflow

            status = inspect_workflow(self.session)
            if status.phase is WorkflowPhase.EMPTY:
                text = "下一步先建立几何；请给出结构类型、跨度、层高和开间。"
            elif status.phase is WorkflowPhase.DRAFT:
                detail = "；".join(status.validation_errors[:2])
                text = f"下一步先补全模型再求解。当前问题：{detail}"
            elif status.phase is WorkflowPhase.READY:
                text = "模型已经可分析。下一步运行求解，然后检查位移、反力和杆件内力。"
            else:
                text = "求解已经完成。下一步建议先看最大位移与控制工况，再检查内力和边界反力。"
            return self._finish_local(user_text, text, [], started, on_tool)

        if normalized in {"最大位移", "查看最大位移", "查询最大位移"} \
                and self.session.solution is not None:
            args = {"what": "max_displacement"}
            result, tool_ms = self._run_tool("query_results", args, on_tool)
            if result.ok:
                p = result.payload
                text = (f"最大位移为 {p['magnitude_mm']} mm，控制工况 {p['case']}，"
                        f"位置坐标 {p['coordinates_xyz']} m；分量 {p['components_mm']} mm。")
            else:
                text = str(result.payload.get("error", "结果查询失败"))
            out = self._finish_local(user_text, text,
                                     [("query_results", args)], started, on_tool)
            out.metrics["tool_ms"] = tool_ms
            return out

        if normalized in {"检查模型", "校验模型", "验证模型"}:
            args: dict[str, Any] = {}
            result, tool_ms = self._run_tool("validate_model", args, on_tool)
            if result.ok:
                summary = result.payload.get("summary") or {}
                text = ("模型校验通过，可以求解。"
                        f"当前 {summary.get('nodes', 0)} 个节点、"
                        f"{summary.get('members', 0)} 根杆件。")
            else:
                issues = result.payload.get("errors") or [result.payload.get("error")]
                text = "模型尚不能求解：" + "；".join(str(item) for item in issues[:3])
            out = self._finish_local(user_text, text,
                                     [("validate_model", args)], started, on_tool)
            out.metrics["tool_ms"] = tool_ms
            return out

        if normalized in {"重新求解", "重新计算", "再算一遍", "求解模型"}:
            args = {}
            result, tool_ms = self._run_tool("solve_model", args, on_tool)
            if result.ok:
                cases = result.payload.get("cases") or {}
                rows = [f"{name}: {row.get('max_displacement_mm')} mm"
                        for name, row in list(cases.items())[:4]]
                text = "求解完成。" + ("；".join(rows) if rows else "结果已更新。")
            else:
                issues = result.payload.get("errors") or [result.payload.get("error")]
                text = "求解未通过：" + "；".join(str(item) for item in issues[:3])
            out = self._finish_local(user_text, text,
                                     [("solve_model", args)], started, on_tool)
            out.metrics["tool_ms"] = tool_ms
            return out

        if normalized in {"查看反力", "查询反力", "支座反力"} \
                and self.session.solution is not None:
            args = {"what": "reactions"}
            result, tool_ms = self._run_tool("query_results", args, on_tool)
            if result.ok:
                p = result.payload
                text = (f"控制工况 {p['case']} 的支座反力已读取；"
                        f"竖向合力 {p['vertical_total_kN']} kN。"
                        "各支座分量已保留在结果检查器中。")
            else:
                text = str(result.payload.get("error", "反力查询失败"))
            out = self._finish_local(user_text, text,
                                     [("query_results", args)], started, on_tool)
            out.metrics["tool_ms"] = tool_ms
            return out
        return None

    def _emit(self, kind: str, data: dict) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, data)
        except Exception:  # noqa: BLE001  界面回调出错不能中断对话，模型已做的改动不会因显示失败而撤销
            pass

    def _run_tool(self, name: str, args: dict, on_tool=None, call_id: str = ""):
        """执行一次工具，前后各发一个事件；返回结果和耗时（毫秒）。"""
        self._emit("tool_start", {"id": call_id, "name": name, "args": args})
        started = time.perf_counter()
        result = self.session.dispatch(name, args)
        elapsed = (time.perf_counter() - started) * 1000
        if on_tool is not None:
            try:
                on_tool(name, args, result.ok)
            except Exception:  # noqa: BLE001  隔离外部回调：on_tool 是界面传进来的，它出问题不该把对话流程带走
                pass
        self._emit("tool_end", {"id": call_id, "name": name, "args": args, "ok": result.ok,
                                "ms": elapsed, "summary": result_brief(result)})
        return result, elapsed

    def cancel(self) -> None:
        """请求停止。**下一轮开始前生效，不打断正在跑的这一轮。**

        中途掐断的诱惑很大，但工具调用是会改模型的：掐在
        `generate_frame` 和 `set_load_cases` 之间，得到的是一个有几何
        没荷载的半成品，而用户以为自己"取消"了。让当前这轮跑完再停，
        状态始终是完整的。
        """
        self._cancelled = True

    # --- 主循环 ---

    def ask(self, user_text: str, on_tool=None, on_event=None) -> TurnResult:
        """问一轮。工具调用循环与单轮一致，区别只在带着历史进去。

        `on_tool(name, args, ok)` 在**每个工具跑完的那一刻**回调一次。
        一轮对话十几秒，期间界面上什么都没有的话用户只能干等；逐个报出来
        不只是"看着不慌"——他能在第三行就发现模型理解错了跨度，立刻按停止，
        而不是等它把报告都写完。

        `on_event(kind, data)` 给桌面对话用的细粒度事件：``text``（流式文字片段）、
        ``round``（开始一次模型请求）、``tool_start`` / ``tool_end``（含耗时与结果
        摘要）、``plan``（计划清单）。不传则与原来完全一样。
        """
        self._on_event = on_event
        try:
            return self._ask(user_text, on_tool)
        finally:
            self._on_event = None

    def _complete(self, messages: list[dict], tools: list[dict]) -> dict:
        stream = getattr(self.provider, "complete_stream", None)
        if self._on_event is not None and stream is not None:
            return stream(messages, tools, lambda piece: self._emit("text", {"delta": piece}))
        return self.provider.complete(messages, tools)

    def _ask(self, user_text: str, on_tool=None) -> TurnResult:
        started = time.perf_counter()
        self.session.begin_user_turn(user_text)
        local = self._try_local_fast_path(user_text, started, on_tool)
        if local is not None:
            return local
        messages = self.build_messages(user_text)
        problems = check_pairing(messages)
        if problems:
            # 走到这里说明裁剪逻辑有 bug。宁可当场炸，也别把断裂的
            # 上下文发出去换一个语焉不详的 400
            raise AssertionError("上下文的工具调用配对断裂：" + "；".join(problems))

        this_turn: list[dict] = [{"role": "user", "content": user_text}]
        calls: list[tuple[str, dict]] = []
        self._cancelled = False
        provider_ms = 0.0
        tool_ms = 0.0
        prompt_chars = 0
        tools = TOOLS + [PLAN_TOOL] if self.plan_tool else TOOLS

        for round_index in range(self.max_rounds):
            if self._cancelled:
                text = _cancelled_message(self.session, calls)
                self.history.append(Exchange(
                    user_text, text,
                    [{"role": "user", "content": user_text},
                     {"role": "assistant", "content": text}],
                    calls, round_index + 1, True))
                return TurnResult(text, round_index + 1, calls, self.session,
                                  stopped_by_limit=True)
            prompt_chars += _message_chars(messages)
            provider_started = time.perf_counter()
            self._emit("round", {"index": round_index})
            reply = self._complete(messages, tools)
            provider_ms += (time.perf_counter() - provider_started) * 1000
            if not reply.get("tool_calls"):
                text = reply.get("content", "")
                this_turn.append({"role": "assistant", "content": text})
                self.history.append(Exchange(user_text, text, this_turn,
                                             calls, round_index + 1))
                return TurnResult(text, round_index + 1, calls, self.session,
                                  metrics={
                                      "total_ms": (time.perf_counter() - started) * 1000,
                                      "provider_ms": provider_ms,
                                      "tool_ms": tool_ms,
                                      "provider_calls": round_index + 1,
                                      "prompt_chars": prompt_chars,
                                      "fast_path": False,
                                  })

            assistant = {
                "role": "assistant", "content": None,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": json.dumps(c["arguments"],
                                                          ensure_ascii=False)}}
                    for c in reply["tool_calls"]],
            }
            messages.append(assistant)
            this_turn.append(assistant)

            for call in reply["tool_calls"]:
                if self.plan_tool and call["name"] == "update_plan":
                    steps = _plan_steps(call["arguments"])
                    if steps is not None:
                        self.plan = steps
                        self._emit("plan", {"steps": steps})
                    content = json.dumps({"ok": steps is not None} if steps is not None else
                                         {"ok": False, "error": "steps 须为 1–20 项，每项含 title 与 status"},
                                         ensure_ascii=False)
                    tool_msg = {"role": "tool", "tool_call_id": call["id"], "content": content}
                    messages.append(tool_msg)
                    this_turn.append(tool_msg)
                    continue
                result, elapsed = self._run_tool(call["name"], call["arguments"], on_tool, call["id"])
                tool_ms += elapsed
                calls.append((call["name"], call["arguments"]))
                tool_msg = {"role": "tool", "tool_call_id": call["id"],
                            "content": tool_message_content(result)}
                messages.append(tool_msg)
                this_turn.append(tool_msg)
            refresh_workflow_message(messages, self.session)

        # 撞上限时**不能只说"没得到结论"**：工具其实已经跑过了，模型很可能
        # 已经建好甚至算完了，摆在界面上。这时候说"没结论"会让用户以为白干了，
        # 于是重来一遍——而重来只会再撞一次上限。所以把已经做到的说出来
        text = _stalled_message(self.session, calls, self.max_rounds)
        # 轮数用尽时**不把这半截回合写进历史**：里面全是未收尾的工具调用，
        # 留着只会污染下一轮的上下文
        self.history.append(Exchange(user_text, text,
                                     [{"role": "user", "content": user_text},
                                      {"role": "assistant", "content": text}],
                                     calls, self.max_rounds, True))
        return TurnResult(text, self.max_rounds, calls, self.session,
                          stopped_by_limit=True, metrics={
                              "total_ms": (time.perf_counter() - started) * 1000,
                              "provider_ms": provider_ms,
                              "tool_ms": tool_ms,
                              "provider_calls": self.max_rounds,
                              "prompt_chars": prompt_chars,
                              "fast_path": False,
                          })

    # --- 观察 ---

    def transcript(self) -> list[dict[str, Any]]:
        """给界面看的对话记录。"""
        return [{"user": ex.user, "reply": ex.reply, "rounds": ex.rounds,
                 "tools": [name for name, _ in ex.tool_calls],
                 "stopped_by_limit": ex.stopped_by_limit}
                for ex in self.history]

    def reset_history(self) -> None:
        """只清对话，不动模型——"重新问一遍，但别推倒重建"。"""
        self.history.clear()

    def drop_last_exchange(self) -> Exchange | None:
        """去掉最后一个回合，供"重新生成 / 编辑后重发"使用；模型状态由调用方处理。"""
        return self.history.pop() if self.history else None

    def export_history(self) -> list[dict[str, Any]]:
        """可写入 JSON 的完整回合记录，重启后用 ``import_history`` 恢复上下文。"""
        return [{"user": ex.user, "reply": ex.reply, "messages": ex.messages,
                 "tool_calls": [[name, args] for name, args in ex.tool_calls],
                 "rounds": ex.rounds, "stopped_by_limit": ex.stopped_by_limit}
                for ex in self.history]

    def import_history(self, records: list[dict[str, Any]]) -> None:
        """恢复 ``export_history`` 的结果；配对断裂的记录整体丢弃，不发出坏上下文。"""
        restored = []
        for item in records or []:
            try:
                messages = list(item["messages"])
                if check_pairing(messages):
                    continue
                restored.append(Exchange(str(item["user"]), str(item["reply"]), messages,
                                         [(str(n), dict(a)) for n, a in item.get("tool_calls", [])],
                                         int(item.get("rounds", 0)),
                                         bool(item.get("stopped_by_limit", False))))
            except (KeyError, TypeError, ValueError):
                continue
        self.history = restored
