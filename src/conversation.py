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
from dataclasses import dataclass, field
from typing import Any

from agent import (MAX_ROUNDS, SYSTEM_PROMPT, TOOLS, Provider, Session,
                   TurnResult)
from workflow import refresh_workflow_message, workflow_message

# 默认保留多少个回合。一个回合可能有十几条消息，六个回合已经够
# "改一下再算"这类连续追问了，再多只是在烧 token。
DEFAULT_KEEP_TURNS = 6

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
                 max_rounds: int = MAX_ROUNDS):
        if keep_turns < 1:
            raise ValueError("keep_turns 至少为 1")
        self.provider = provider
        self.session = session or Session()
        self.keep_turns = keep_turns
        self.max_rounds = max_rounds
        self.history: list[Exchange] = []
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
        if self.history:
            out.append({"role": "system", "content": _CONTINUATION_HINT})
        return out

    def build_messages(self, user_text: str) -> list[dict]:
        """拼出这一轮要发给模型的完整 messages。

        裁剪以回合为单位：要么整段留，要么整段丢。**绝不从回合中间切开**——
        那会拆散 assistant.tool_calls 与它的 tool 结果。
        """
        kept = self.history[-self.keep_turns:] if self.keep_turns else []
        messages = self.system_messages()
        for ex in kept:
            messages.extend(ex.messages)
        messages.append({"role": "user", "content": user_text})
        return messages

    def cancel(self) -> None:
        """请求停止。**下一轮开始前生效，不打断正在跑的这一轮。**

        中途掐断的诱惑很大，但工具调用是会改模型的：掐在
        `generate_frame` 和 `set_load_cases` 之间，得到的是一个有几何
        没荷载的半成品，而用户以为自己"取消"了。让当前这轮跑完再停，
        状态始终是完整的。
        """
        self._cancelled = True

    # --- 主循环 ---

    def ask(self, user_text: str, on_tool=None) -> TurnResult:
        """问一轮。工具调用循环与单轮一致，区别只在带着历史进去。

        `on_tool(name, args, ok)` 在**每个工具跑完的那一刻**回调一次。
        一轮对话十几秒，期间界面上什么都没有的话用户只能干等；逐个报出来
        不只是"看着不慌"——他能在第三行就发现模型理解错了跨度，立刻按停止，
        而不是等它把报告都写完。
        """
        self.session.receive_user_confirmation(user_text)
        messages = self.build_messages(user_text)
        problems = check_pairing(messages)
        if problems:
            # 走到这里说明裁剪逻辑有 bug。宁可当场炸，也别把断裂的
            # 上下文发出去换一个语焉不详的 400
            raise AssertionError("上下文的工具调用配对断裂：" + "；".join(problems))

        this_turn: list[dict] = [{"role": "user", "content": user_text}]
        calls: list[tuple[str, dict]] = []
        self._cancelled = False

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
            reply = self.provider.complete(messages, TOOLS)
            if not reply.get("tool_calls"):
                text = reply.get("content", "")
                this_turn.append({"role": "assistant", "content": text})
                self.history.append(Exchange(user_text, text, this_turn,
                                             calls, round_index + 1))
                return TurnResult(text, round_index + 1, calls, self.session)

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
                result = self.session.dispatch(call["name"], call["arguments"])
                calls.append((call["name"], call["arguments"]))
                if on_tool is not None:
                    try:
                        on_tool(call["name"], call["arguments"], result.ok)
                    except Exception:               # noqa: BLE001
                        # 回调是界面的事。它出问题不该把这一轮对话带垮——
                        # 模型已经改过的东西不会因为显示不出来就撤销
                        pass
                tool_msg = {"role": "tool", "tool_call_id": call["id"],
                            "content": result.to_json()}
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
                          stopped_by_limit=True)

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
