"""Agent 层：把自然语言变成结构模型，再由确定性工具算出全部数字。

设计原则（这几条是整个项目立得住的根据，改代码时别破坏）：

1. **大模型只产结构，不产数值。** 它可以决定"三跨两层、柱底固接"，但位移、内力、
   反力必须来自工具返回。系统提示里明令禁止它自己算，报告里每个数字都能溯源到
   某次工具调用。
2. **拓扑由代码展开。** 多层多跨让模型逐个列节点坐标必然出错，所以给它
   `generate_frame(spans, storeys, bays, ...)` 这样的参数入口。不规则结构才走
   `set_model` 这条逃生舱。
3. **算前守门。** `solve_model` 内部先跑 `validate_model`，不通过就不算，把中文
   错误清单回给模型，让它自己改——这就是自修复闭环的一轮。
4. **信息不全就追问，不许编默认值。** 这是 Agent 区别于脚本的地方，也是评测集
   里专门要统计的一项指标。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from generator import (describe)
import changes as _changes
from history import BuildHistory, MUTATING
from model_compiler import CompilationError, compile_model
from model_io import (CURRENT_SCHEMA_VERSION, migrate_payload,
                      validate_payload)
from units import convert_model
from units import system as unit_system

# 求解后自动跑的两项质检：静默失败检测 + 实验胶囊存档。
# 这两项失败不影响求解结果本身——用 try/except 兜住，只记警告。

MAX_ROUNDS = 24     # 一次完整建模+求解+查询，模型若逐个发调用能到十几轮。
                    # 真正的省时手段是让它一条消息发多个调用（见 SYSTEM_PROMPT 第 9 条），
                    # 上限只是最后那道防跑飞的闸

# SELF_WEIGHT_NOTE / _UNSET / ToolResult / _records 搬去了 session_base.py，
# 因为拆出去的 session_*.py 每个都要用它们。这里再导出一次，
# `from agent import ToolResult` 这类既有写法继续成立。
from session_base import (SELF_WEIGHT_NOTE, ToolResult, _records,  # noqa: F401
                         _UNSET)

SYSTEM_PROMPT = """你是空间刚架结构分析助手。你的职责是把用户的自然语言描述转成结构模型，
调用工具完成计算，然后解释结果。

硬性规则，任何情况下都不得违反：

1. 你不做任何数值计算。位移、内力、反力、应力的每一个数字都必须来自工具返回值。
   如果你没有调用工具就说出一个数，那就是编的。禁止心算、估算、类比推测。
2. 建模按这个顺序挑工具，**不要自己逐个列节点坐标**——多层多跨你一定会数错：
   · 正交规则框架 → generate_frame
   · 坡屋面门式刚架 → generate_portal_frame
   · 其它形状（单坡、悬挑、错层、任意折线屋面）→ generate_bent，
     屋面形状全由 profile 折线决定；要做成空间结构再调 extrude_bents
   · 在已有模型上局部改动 → 算子：add_bracing 加支撑、remove_members 抽柱开洞、
     raise_nodes 抬高一片区域、retaper 换截面
   算子要的杆件编号**一律先用 select_members 按几何条件挑**，不许凭空猜编号——
   编号规则是生成器定的，你并不知道 27 号在哪。
   同一组杆件要反复引用时，用 define_set 给它起个工程上的名字（「顶层梁」「边柱」），
   之后 remove_members / retaper 直接写名字即可。回答里也用这个名字，
   比报一串编号好懂得多。
   只有以上都覆盖不了时才用 set_model 传完整 JSON。
3. 几何、属性、边界和载荷分阶段建立，顺序参照 Abaqus/CAE：可以先生成或手工绘制
   节点与杆件拓扑，暂不指派材料和截面；之后用 Property 阶段补齐属性，再定义工况、
   边界条件和载荷。只有 solve_model 前必须全部完整。几何尺寸缺失时必须追问，
   不许编造；用户只要求先建几何时，不要因材料、截面或载荷未给而阻止建模。

   **但"可以先不指派"不等于"可以自己编出来"。** 材料、截面、荷载、约束里
   任何一项用户没给，你都不许替他假定一个值——哪怕是很常见的值（Q235、
   E=2.1e11、HN400×200 之类）。缺了就问，问完再算。

   具体到你会怎么犯这个错：用户说"算一下最大位移"但没给截面和材料时，
   **不要**调 define_materials_and_sections 填一组自己挑的数再 solve_model。
   那样算出来的位移是从你编的刚度来的，**比不给答案更有害**——用户看到的
   是一个有单位、有量级、看着很合理的数字，没有任何迹象表明它是假的。
   正确的做法是：几何可以先建（那是用户给全了的），然后停下来问缺的那几项。
4. solve_model 返回错误清单时，读懂它、改模型、重试。同一个错误连续两次没改对，
   就停下来告诉用户你卡在哪里。

   **solve_model 的 payload 里有 silent_failures，你必须看。** 它是"算出来了
   但结果可疑"的检测结果，和"报错"是两回事——报错是算不出来，它是算出来了
   却不能用。带 `unusable` 字段时尤其要注意：那表示这组数**不代表任何受力
   状态**（常见的是压根没有荷载、结构是机构、荷载没传到杆件上）。

   这种情况下**绝对不许把结果当有效结果汇报**。哪怕位移和内力都有数字、
   哪怕每个数字都能溯源到工具调用——一个从空工况算出来的 0.0 溯源溯得
   再清楚，也还是个废数。按 `next` 里的建议改模型重算，或者如实告诉用户
   这次结果不可用、原因是什么。

   severity 为 warning 的项不拦你，但汇报时要提一句。
5. 默认单位制 N-m-Pa：长度米，力牛顿，弹性模量帕斯卡。用户说"20 kN/m"你要换算成
   20000 再传给工具。换算过程写在回答里。用户明确要求用毫米制时才调 set_units
   切到 N-mm-MPa，那会把整份模型按物理量等价地换算过去。
   查询结果一律以 mm、kN、kN·m 返回，与模型用哪套制无关。
6. 材料与截面调用 define_materials_and_sections 定义一次即可，generate_frame 和
   set_model 都会自动沿用。几何草稿的 analysis_ready=false 不是建模失败；继续指派属性、
   边界和载荷，最后调用 validate_model。不要因为同一报错反复重发模型。
   同一个工具连续两次返回同类错误，就停下来说明你卡在哪里，不要继续试。
7. 多个荷载工况必须用 set_load_cases 分别定义再做组合，**不许把几个工况的荷载先加起来
   当成单一工况**。线性叠加下组合结果虽然相同，但那样拿不到各工况的分项内力，也无法做包络。
8. 描述结果位置时只能引用工具返回的坐标。不要凭节点编号推测它在结构的什么位置——
   编号规则你并不知道。

9. **互不依赖的工具调用要放在同一条消息里一次发出。** 定义材料截面、生成框架、
   设定荷载工况这一串是固定顺序的，可以一次全发；查位移、查内力、查包络彼此无关，
   更该一次全发。一次只发一个会让每一步都多一个网络往返——用户等的是秒，不是毫秒。
   只有当后一步真的要读前一步的返回值时（比如要用生成的杆件编号去加荷载），
   才分两次发。

10. 删除已有节点或杆件必须先调用 preview_change，把结构化差异和预演编号告诉用户，
    然后停止本轮。只有用户在**下一条消息**明确确认后才调用 apply_preview；你无权替用户
    确认，也不得在同一轮或同一批工具调用中预演并执行。

11. 校核类工具（check_strength / check_symmetry / check_numbering）返回的
    **限制与警告要原样转达**，不许只报好消息：
    · check_strength 的 failed_members 是真的超限；inconclusive_members 是
      「欧拉公式在这根杆上不适用」，**既不是通过也不是不通过**，
      不要说成"不安全"，也不要说成"通过"。
    · μ 标着"由杆端释放推定"时，必须说明它只对无侧移结构成立，
      有侧移的框架柱要用户自己给 mu_y / mu_z。
    · 强度验算只含正应力，不含剪应力与扭转，**不是规范意义上的承载力验算**。
    · check_symmetry 的位移镜像校核通过，只说明结果自洽，不等于模型建对了。

回答用中文，简洁，先给结论再给数据。"""


# --------------------------------------------------------------------------- 工具定义
#
# 清单本体在 agent_tools.py。这里原样再导出，是为了让
# `from agent import TOOLS` 这个既有写法继续成立——conversation.py、
# 桌面端和七八个测试都是这么写的，没必要为了搬个文件去改它们。
from agent_tools import TOOLS  # noqa: F401  (re-export)


# --------------------------------------------------------------------------- 会话

# Session 的五个域拆在 session_*.py 里，这里组装起来。
#
# 拆分是纯搬运：方法体一个字节没改，dispatch 仍然靠 getattr(self, name) 找
# 工具，mixin 继承来的方法和写在本文件里的没有任何区别。这么做只为一件事：
# 四千多行一个文件时，想找一个方法得先翻过另外三千行不相干的。
#
# 撤销/预演那组**有意**留在下面没搬：preview_change 要在运行时造一个沙箱
# Session，搬出去就是循环依赖。
from session_modeling import ModelingMixin
from session_loads import LoadsMixin
from session_solving import SolvingMixin
from session_query import QueryMixin
from session_checks import ChecksMixin


# 批量修改闸门看这几类实体：它们有稳定 ID，改了就是改了「用户已经认过的结构」。
# 荷载、工况不在里面——set_load_cases 本来就是整体重写，把它也拦住，
# 每次调荷载都要确认一遍，闸门就成了噪声。
_GUARDED_COLLECTIONS = {"nodes": "id", "members": "id", "supports": "node"}
# 一轮里改到第几个已有实体就要预演。1 个是「改这根杆」，2 个起才算批量。
BATCH_EDIT_THRESHOLD = 2


@dataclass
class Session(ModelingMixin, LoadsMixin, SolvingMixin,
              QueryMixin, ChecksMixin):
    """一次会话持有的全部确定性状态。模型与结果只在这里，不在提示词里。"""
    model: dict[str, Any] = field(default_factory=dict)
    frame: Any = None
    solution: Any = None
    compilation: Any = None
    result_db: Any = None
    tool_log: list[tuple[str, dict, "ToolResult"]] = field(default_factory=list)
    # 建模过程。界面拖时间轴回看每一步靠它
    history: BuildHistory = field(default_factory=BuildHistory)
    # 图片识别证据不进入求解 Domain IR，但与模型共用撤销事务。
    multimodal_provenance: dict[str, Any] | None = None
    # 预演属于对话控制状态，不属于 Domain IR，也不进入撤销历史。
    pending_change: dict[str, Any] | None = field(default=None, repr=False)
    authorized_preview_id: str | None = field(default=None, repr=False)
    _applying_preview: bool = field(default=False, repr=False)
    # 批量修改闸门的记账：本轮用户消息开始时已存在的实体，以及本轮已经改动过的。
    # None 表示不在对话轮次里（界面直接调方法、离线脚本直接 dispatch），不设闸。
    _turn_baseline: dict[str, set] | None = field(default=None, repr=False)
    _turn_touched: set = field(default_factory=set, repr=False)

    # --- 撤销 / 重做 ---
    #
    # 这三个方法**不是工具**，不进 TOOLS，大模型调不到它们。
    # 撤销是用户对界面的操作，不是模型该自己决定的事——让模型能撤销
    # 自己刚做的事，只会让它在犯错时反复横跳。

    def undo(self) -> bool:
        """退一步。成功返回 True。"""
        model = self.history.undo()
        if model is None:
            return False
        self._restore(model)
        self.multimodal_provenance = self.history.provenance_at(self.history.cursor)
        return True

    def redo(self) -> bool:
        model = self.history.redo()
        if model is None:
            return False
        self._restore(model)
        self.multimodal_provenance = self.history.provenance_at(self.history.cursor)
        return True

    def goto_step(self, cursor: int) -> bool:
        """跳到第 cursor 步之后的状态。cursor = -1 表示回到空模型。"""
        model = self.history.goto(cursor)
        if model is None:
            return False
        self._restore(model)
        self.multimodal_provenance = self.history.provenance_at(self.history.cursor)
        return True

    def _restore(self, model: dict[str, Any]) -> None:
        """装回一个快照。

        **必须同时清掉 frame 和 solution。** 不清的话，模型退回去了而结果
        还是旧模型的——界面会拿着一份对不上的位移去画变形图，
        而且画得出来、看不出错。这是这个功能最危险的地方。
        """
        self.model = model
        self.frame = None
        self.solution = None
        self.compilation = None
        self.result_db = None
        self.pending_change = None
        self.authorized_preview_id = None

    def begin_user_turn(self, user_text: str) -> bool:
        """一条新的用户消息到达：记下此刻已有哪些实体，再看它是不是确认口令。

        批量修改闸门只保护**这条消息之前就存在**的节点、杆件与支座——本轮刚建出来的
        东西随便改，否则「先生成几何、再指派截面」这条正常建模路径也会被拦。
        """
        self._turn_baseline = {
            key: {item.get(ident) for item in self.model.get(key) or []
                  if isinstance(item, dict)}
            for key, ident in _GUARDED_COLLECTIONS.items()}
        self._turn_touched = set()
        return self.receive_user_confirmation(user_text)

    def _existing_entities_touched(self, name: str,
                                   arguments: dict[str, Any]) -> set | None:
        """在副本上试跑写工具，返回它会改动或删除的「本轮之前已有」的实体。

        与 preview_change 同一套做法：副本上跑、diff 前后模型。试跑失败返回 None，
        交给真实调用去报它自己的错误。
        """
        from copy import deepcopy
        from change_preview import diff_models

        sandbox = Session(model=deepcopy(self.model))
        try:
            attempted = getattr(sandbox, name)(**deepcopy(arguments))
        except TypeError:
            return None
        if not attempted.ok:
            return None
        delta = diff_models(self.model, sandbox.model)
        touched = set()
        for key in _GUARDED_COLLECTIONS:
            col = delta["collections"].get(key) or {}
            for ident in list(col.get("changed", [])) + list(col.get("removed", [])):
                if ident in self._turn_baseline.get(key, set()):
                    touched.add((key, ident))
        return touched

    def receive_user_confirmation(self, user_text: str) -> bool:
        """只接受一条新的、明确的用户消息；工具调用本身不能走这条路。"""
        if self.pending_change is None:
            return False
        text = str(user_text).strip().lower()
        if text in {"取消", "取消变更", "cancel"}:
            self.pending_change = None
            self.authorized_preview_id = None
            return False
        preview_id = self.pending_change["preview_id"]
        accepted = {"确认", "确认执行", "应用变更", "confirm", "apply",
                    f"确认 {preview_id.lower()}", f"confirm {preview_id.lower()}",
                    f"apply {preview_id.lower()}"}
        if text not in accepted:
            return False
        self.authorized_preview_id = preview_id
        return True

    def preview_change(self, tool: str,
                       arguments: dict[str, Any]) -> ToolResult:
        """在隔离的 Session 上执行写工具，返回紧凑 diff，不碰当前状态。"""
        from copy import deepcopy
        from change_preview import diff_models, preview_digest

        exposed = {item["function"]["name"] for item in TOOLS}
        if tool not in MUTATING or tool not in exposed:
            return ToolResult(False, {
                "error": f"{tool!r} 不是可预演的 Agent 写工具",
                "allowed_tools": sorted(MUTATING & exposed),
            })
        if not isinstance(arguments, dict):
            return ToolResult(False, {"error": "可预演工具的 arguments 必须是对象"})

        before = deepcopy(self.model)
        sandbox = Session(model=deepcopy(before))
        attempted = getattr(sandbox, tool)(**deepcopy(arguments))
        if not attempted.ok:
            detail = attempted.payload.get("error") or attempted.payload.get("errors")
            return ToolResult(False, {
                "schema": "agent-change-preview/v1",
                "tool": tool,
                "preview_error": str(detail or "预演未通过"),
            })

        delta = diff_models(before, sandbox.model)
        errors = [str(error)[:240]
                  for error in validate_payload(sandbox.model)[:10]]
        preview_id = preview_digest(tool, arguments, delta["before_hash"],
                                    delta["after_hash"])
        payload = {
            "schema": "agent-change-preview/v1",
            "preview_id": preview_id,
            "tool": tool,
            "diff": delta,
            "validation_errors": errors,
            "analysis_ready": not errors,
            "would_invalidate_results": bool(delta["changed"]),
            "preview_result": attempted.payload,
        }
        self.pending_change = {
            "preview_id": preview_id,
            "tool": tool,
            "arguments": deepcopy(arguments),
            "before_hash": delta["before_hash"],
            "diff": deepcopy(delta),
        }
        self.authorized_preview_id = None
        return ToolResult(True, payload)

    def apply_preview(self, preview_id: str) -> ToolResult:
        """应用已由用户消息授权的预演；哈希不一致时拒绝陈旧变更。"""
        from copy import deepcopy
        from change_preview import model_digest

        pending = self.pending_change
        if pending is None or pending.get("preview_id") != preview_id:
            return ToolResult(False, {
                "error": "没有这个待确认预演，可能已失效或已被新的预演替换",
                "confirmation_required": True,
            })
        if self.authorized_preview_id != preview_id:
            return ToolResult(False, {
                "error": "需要用户在预演之后发送一条新的明确确认消息",
                "confirmation_required": True,
                "preview_id": preview_id,
            })
        if model_digest(self.model) != pending["before_hash"]:
            self.pending_change = None
            self.authorized_preview_id = None
            return ToolResult(False, {
                "error": "预演后模型已变化，该预演已失效；请重新预演",
                "stale_preview": True,
            })

        tool = pending["tool"]
        self._applying_preview = True
        try:
            applied = getattr(self, tool)(**deepcopy(pending["arguments"]))
        finally:
            self._applying_preview = False
        if not applied.ok:
            return applied
        self.pending_change = None
        self.authorized_preview_id = None
        return ToolResult(True, {
            "schema": "agent-change-application/v1",
            "preview_id": preview_id,
            "applied_tool": tool,
            "result": applied.payload,
            "diff": pending["diff"],
        })

    @_records
    def set_model(self, model: dict[str, Any]) -> ToolResult:
        """提交完整模型。已定义的材料与截面会自动带上，不必重复书写。

        （早期版本这里是整体替换，模型以为材料定义过一次就够了，结果被
        "materials is required" 反复打回，陷入死循环。与 generate_frame 保持一致。）
        """
        from copy import deepcopy

        merged = deepcopy(model)
        legacy = "schema_version" not in merged
        carried = []
        for key in ("materials", "sections"):
            if key not in merged and key in self.model:
                merged[key] = self.model[key]
                carried.append(key)
        merged.setdefault("units", self.model.get("units", "N-m-Pa"))
        # 立即校验。早期版本只存不验，模型提交坏模型却拿到"成功"，
        # 错误要到下一个工具才炸，它就以为是下一个工具的参数写错了，白转好几轮。
        errors = validate_payload(merged)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "完整模型未写入，原模型保持不变。"
                                              "也可改用 add_nodes/add_members 分阶段建几何"})
        self.model = migrate_payload(merged)
        self._invalidate()
        payload: dict[str, Any] = {
            "summary": describe(self.model),
            "schema_version": CURRENT_SCHEMA_VERSION,
        }
        if legacy:
            payload["migrated_from"] = 0
        if carried:
            payload["note"] = f"沿用了已定义的 {'、'.join(carried)}，不必重复提交"
        return ToolResult(True, payload)

    def open_draft(self) -> "Any":
        """从当前模型开一份草稿。**不记账**——开草稿本身什么都没改。

        草稿是「编辑中」的模型：可以悬空、可以没支座、可以断成两片。
        改完调 `apply_draft` 才落到正式模型上，中途丢掉不留任何痕迹。
        """
        from draft import FrameDraft
        return FrameDraft.from_model(self.model)

    @_records
    def apply_draft(self, draft: "Any", note: str | None = None) -> ToolResult:
        """把草稿提交为正式模型。**草稿进入模型只有这一道门。**

        门后面走的是和其它工具完全一样的一套：校验、失效、清理、记账。
        草稿要是能绕过去自己写 `self.model`，手工画的图就会既不进撤销链、
        也不触发结果失效——画完还显示着上一版的变形图。
        """
        blocking = draft.finish_diagnostics()

        before_m = {int(m["id"]) for m in self.model.get("members") or []}
        candidate = draft.to_model()
        after_m = {int(m["id"]) for m in candidate.get("members") or []}

        # 草稿只动几何，但删掉的杆件上可能挂着荷载。**清理走同一张影响表**，
        # 不在这里另写一遍——另写的那一份就是下一个漏掉某类荷载的地方
        report = _changes.apply(candidate, _changes.Change.GEOMETRY,
                                dropped_members=before_m - after_m)

        errors = validate_payload(candidate)

        self.model = candidate
        self._invalidate()
        payload: dict[str, Any] = {
            "summary": describe(self.model),
            "note": note or "手工修改已提交，原有分析结果已失效，请重新求解。",
            "analysis_ready": not errors,
            "warnings": [n.message for n in blocking] or errors,
        }
        payload.update(report)
        return ToolResult(True, payload)

    def apply_multimodal_draft(
            self, candidate_model: dict[str, Any], provenance: dict[str, Any],
            *, expected_baseline_hash: str) -> ToolResult:
        """原子写入识别模型与侧车状态；失败不改状态也不记历史。"""
        from copy import deepcopy
        from change_preview import model_digest
        from draft import FrameDraft

        if model_digest(self.model) != expected_baseline_hash:
            return ToolResult(False, {"error": "提交基线已变化，请重新预演"})
        before = {
            "model": deepcopy(self.model), "frame": self.frame,
            "solution": self.solution, "compilation": self.compilation,
            "result_db": self.result_db,
            "provenance": deepcopy(self.multimodal_provenance),
            "history": deepcopy(self.history),
        }
        try:
            draft = FrameDraft.from_model(candidate_model)
            # 调用未装饰的实现，避免内层先记一条 apply_draft 历史。
            result = Session.apply_draft.__wrapped__(
                self, draft, "图片识别草稿已提交；分析结果已失效，请按需求解。")
            if not result.ok:
                raise ValueError(result.payload.get("error") or "草稿提交失败")
            self.multimodal_provenance = deepcopy(provenance)
            self.history.record(
                "apply_multimodal_draft",
                {"expected_baseline_hash": expected_baseline_hash}, True,
                result.payload, self.model, self.multimodal_provenance)
            return result
        except Exception as exc:  # noqa: BLE001 - 原子边界必须兜住全部异常
            self.model = before["model"]
            self.frame = before["frame"]
            self.solution = before["solution"]
            self.compilation = before["compilation"]
            self.result_db = before["result_db"]
            self.multimodal_provenance = before["provenance"]
            self.history = before["history"]
            return ToolResult(False, {"error": str(exc)})

    @_records
    def set_units(self, units: str) -> ToolResult:
        """换算整份模型到另一套单位制。"""
        before = self.model.get("units", "N-m-Pa")
        source = self.model or {"units": before}
        try:
            converted = convert_model(source, units)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # 编辑态允许只有节点、还没有杆件/属性；convert_model 对所有已存在
        # 的物理量逐项换算即可，完整合法性仍由提交/求解阶段统一把关。
        self.model = converted
        self._invalidate()
        U = self.units
        return ToolResult(True, {
            "units": units, "was": before,
            "note": f"长度用 {U.length}、密度用 {U.density}、"
                    f"自重 g 取 {U.gravity}。查询结果仍以 mm/kN/kN·m 返回，"
                    "换算前后的数完全相同。"})

    def validate_model(self) -> ToolResult:
        errors = validate_payload(self.model)
        return ToolResult(not errors, {"errors": errors} if errors else {"summary": describe(self.model)})

    def preview_analysis_mesh(self) -> ToolResult:
        """返回物理构件到分析单元的映射，不改写会话的求解状态。"""
        try:
            compiled = compile_model(self.model)
        except CompilationError as exc:
            return ToolResult(False, {
                "errors": list(exc.diagnostics),
                "hint": "先修正模型，再预览分析网格",
            })

        frame = compiled.analysis_model
        mapping = compiled.mapping
        length_unit = "mm" if self.model.get("units") == "N-mm-MPa" else "m"
        generated_node_details = []
        for node_id, (physical_member, ratio) in sorted(mapping.generated_nodes.items()):
            node = frame.nodes[node_id]
            generated_node_details.append({
                "node": node_id,
                "physical_member": physical_member,
                "s": ratio,
                "coordinates": [float(value) for value in node.xyz],
            })

        analysis_element_details = []
        for element_id in sorted(frame.members):
            member = frame.members[element_id]
            analysis_element_details.append({
                "element": element_id,
                "physical_member": mapping.physical_member_id(element_id),
                "i": member.i,
                "j": member.j,
            })

        split_node_details = []
        for physical_member, element_ids in sorted(mapping.physical_to_elements.items()):
            lengths = []
            for element_id in element_ids:
                member = frame.members[element_id]
                lengths.append(float(np.linalg.norm(
                    frame.nodes[member.j].xyz - frame.nodes[member.i].xyz)))
            total = sum(lengths)
            distance = 0.0
            for index, element_id in enumerate(element_ids[:-1]):
                distance += lengths[index]
                node_id = frame.members[element_id].j
                node = frame.nodes[node_id]
                split_node_details.append({
                    "node": node_id,
                    "physical_member": physical_member,
                    "s": distance / total,
                    "origin": ("generated_for_point_load"
                               if node_id in mapping.generated_nodes
                               else "explicit_physical_node"),
                    "coordinates": [float(value) for value in node.xyz],
                })

        physical_nodes = len(compiled.source_ir.get("nodes", ()))
        physical_members = len(compiled.source_ir.get("members", ()))
        return ToolResult(True, {
            "physical_nodes": physical_nodes,
            "physical_members": physical_members,
            "analysis_nodes": len(frame.nodes),
            "analysis_elements": len(frame.members),
            "generated_nodes": len(mapping.generated_nodes),
            "split_nodes": len(split_node_details),
            "has_automatic_splits": len(frame.members) > physical_members,
            "length_unit": length_unit,
            "member_mapping": {
                str(member_id): list(element_ids)
                for member_id, element_ids in sorted(mapping.physical_to_elements.items())
            },
            "generated_node_details": generated_node_details,
            "split_node_details": split_node_details,
            "analysis_element_details": analysis_element_details,
            "diagnostics": list(compiled.diagnostics),
        })

    def export_learning_trace(self, label: str = "",
                              include_snapshots: bool = False) -> ToolResult:
        """把自己的确定性工具过程落成训练/回放样本。

        这里只记录工具事实，不把聊天模型的自然语言结论当真值；最终模型和严格
        校验结果一起保存，后续筛选样本时能排除尚未完成或未通过的轨迹。
        """
        import hashlib
        from datetime import datetime

        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                             for ch in str(label).strip())[:40]
        digest = hashlib.sha256(json.dumps(self.model, sort_keys=True,
                                           ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{stamp}_{digest}" + (f"_{safe_label}" if safe_label else "") + ".json"
        directory = Path("learning_traces")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        errors = validate_payload(self.model)
        document = {
            "format": "space-frame-agent-trace/v1",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "label": str(label),
            "verified": not errors,
            "validation_errors": errors,
            "steps": self.history.learning_trace(bool(include_snapshots)),
            "final_model": self.model,
        }
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return ToolResult(True, {"path": str(path.resolve()),
                                 "steps": len(document["steps"]),
                                 "verified": document["verified"],
                                 "format": document["format"],
                                 "note": "verified=true 的轨迹才适合作为已完成建模样本；"
                                         "未完成轨迹保留用于错误修复学习。"})

    # 扫描指标：名字 → (取值函数, 单位, 越大越好吗)
    _SWEEP_BIGGER_IS_BETTER = {"buckling_factor", "first_frequency"}

    def write_report(self, case: str | None = None, fmt: str = "both",
                     filename: str = "报告") -> ToolResult:
        """产出分析报告。"""
        if self.solution is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        if fmt not in ("markdown", "docx", "both"):
            return ToolResult(False, {"error": "fmt 只能取 markdown / docx / both"})
        try:
            import report as _report
        except ImportError as exc:
            return ToolResult(False, {"error": f"报告模块不可用：{exc}"})

        out_dir = Path("results")
        safe = "".join(c if c.isalnum() or c in "-_（）()" else "_"
                       for c in str(filename)) or "报告"
        try:
            doc = _report.gather(self, case=case, out_dir=out_dir)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})

        written: dict[str, str] = {}
        problems: dict[str, str] = {}
        if fmt in ("markdown", "both"):
            path = out_dir / f"{safe}.md"
            _report.to_markdown(doc, path)
            written["markdown"] = str(path)
        if fmt in ("docx", "both"):
            try:
                written["docx"] = str(_report.to_docx(doc, out_dir / f"{safe}.docx"))
            except Exception as exc:      # noqa: BLE001 docx 失败不该毁掉 md
                problems["docx"] = f"{type(exc).__name__}: {str(exc)[:200]}"

        if not written:
            return ToolResult(False, {"error": "两种格式都没写成", "detail": problems})
        payload: dict[str, Any] = {
            "files": written,
            "case": doc["case"],
            "sections": ["模型概况", "求解与校验", "位移", "支座反力"]
                        + (["内力极值与控制组合"] if doc.get("envelope") else [])
                        + (["自振特性"] if doc.get("modal") else [])
                        + (["稳定"] if doc.get("buckling") else [])
                        + (["图"] if doc["figures"] else []),
            "figures": list(doc["figures"]),
            "note": "报告已写到 results/ 目录。里面的数与你查到的完全一致，"
                    "回答里可以直接引用，不必重述整张表。",
        }
        if doc["skipped"]:
            payload["skipped"] = doc["skipped"]
        if problems:
            payload["partial"] = problems
        return ToolResult(True, payload)

    # --- 内部 ---
    def preview_frame(self):
        """不求解也装配出一个 Frame，供"先看一眼再算"用。

        `self.frame` 要到 solve_model 才有值——那是刻意的，免得别处
        误用一个没算过的模型去读结果。但**看模型不该要求先求解**：
        荷载方向加反、支座漏了，正是应该在算之前就看出来的事。
        所以单开这个入口，语义明确：只装配，不算。
        """
        if self.frame is not None:
            return self.frame
        errors = validate_payload(self.model)
        if errors:
            raise ValueError("模型不合法：" + "；".join(errors[:3]))
        return compile_model(self.model).analysis_model

    def _invalidate(self) -> None:
        self.frame = None
        self.solution = None
        self.compilation = None
        self.result_db = None

    def _applied_load_magnitude(self, case: str) -> float:
        """该工况施加的荷载总量级，用来分辨"没有荷载"和"荷载被约束吃掉了"。"""
        load_case = self.frame.load_cases.get(case)
        if load_case is None:                      # 组合：按系数合成后再看
            factors = self.frame.combos.get(case, {})
            total = 0.0
            for name, f in factors.items():
                total += abs(f) * self._applied_load_magnitude(name)
            return total
        total = 0.0
        for load in load_case.nodal_loads.values():
            total += float(np.abs(np.asarray(load, dtype=float)).sum())
        for w in load_case.member_loads.values():
            total += float(np.abs(np.asarray(w, dtype=float)).sum())
        return total

    def _reference_length(self) -> float:
        """取最短杆件长度作为位移量级的参照。"""
        best = 0.0
        for m in self.frame.members.values():
            a, b = self.frame.nodes[m.i], self.frame.nodes[m.j]
            L = float(np.linalg.norm(b.xyz - a.xyz))
            best = L if best == 0.0 else min(best, L)
        return best

    def _max_displacement(self, res) -> tuple[int, float]:
        best_node, best = -1, -1.0
        for nid in self.frame.order():
            d = self.frame.node_dofs(nid)
            mag = float(np.linalg.norm(res.U[d[:3]]))
            if mag > best:
                best, best_node = mag, nid
        return best_node, best

    @property
    def units(self):
        """当前模型的单位制。位移一律报 mm、力报 kN、弯矩报 kN·m，
        与模型内部用 m 还是 mm 无关——读结果的人不用先想是哪套制。"""
        return unit_system(self.model.get("units"))

    def _controlling_case(self) -> str:
        return max(self.solution.all_results().values(),
                   key=lambda r: max(abs(f).max() for f in r.member_forces.values())).name

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        confirmation_tools = {"remove_members", "remove_nodes"}
        has_geometry = bool(self.model.get("nodes") or self.model.get("members"))
        destructive = name in confirmation_tools and has_geometry
        if name == "set_model" and has_geometry and isinstance(arguments, dict):
            replacement = arguments.get("model")
            if isinstance(replacement, dict):
                try:
                    old_nodes = {int(item["id"])
                                 for item in self.model.get("nodes") or []}
                    new_nodes = {int(item["id"])
                                 for item in replacement.get("nodes") or []}
                    old_members = {int(item["id"])
                                   for item in self.model.get("members") or []}
                    new_members = {int(item["id"])
                                   for item in replacement.get("members") or []}
                    destructive = bool((old_nodes - new_nodes)
                                       or (old_members - new_members))
                except (KeyError, TypeError, ValueError):
                    # 格式错误仍交给 set_model 自己返回原有的结构化校验信息。
                    destructive = False
        if destructive and not self._applying_preview:
            result = ToolResult(False, {
                "error": f"{name} 会删除已有模型实体，请先 preview_change 并等待用户确认",
                "confirmation_required": True,
                "required_tool": "preview_change",
            })
            self.tool_log.append((name, arguments, result))
            return result

        # 批量修改闸门。原先只有删除由代码强制预演，批量修改只靠工具描述里一句
        # 「先调用 preview_change」——评测 P02（批量换柱截面）三轮里有一轮没守，
        # 直接改了模型。所以这里按**效果**而不是按工具名判：在副本上试跑，
        # 看它会动多少个本轮之前就存在的节点 / 杆件 / 支座。
        #
        # 按本轮**累计**计数，不按单次调用：否则把一次 assign_properties 拆成
        # 三次 edit_member 就绕过去了。代价是第二次才拦，第一次那一处已经改了——
        # 拦截回包里点名已改的实体，让模型如实告诉用户。
        touched: set | None = None
        if (name in MUTATING and not self._applying_preview
                and self._turn_baseline is not None
                and any(self._turn_baseline.values())
                and isinstance(arguments, dict)):
            touched = self._existing_entities_touched(name, arguments)
            if touched and len(self._turn_touched | touched) >= BATCH_EDIT_THRESHOLD:
                already = sorted(f"{k[:-1]} {i}" for k, i in self._turn_touched)
                result = ToolResult(False, {
                    "error": (f"{name} 会批量修改已有模型实体（本轮累计 "
                              f"{len(self._turn_touched | touched)} 个），"
                              "请先 preview_change 并等待用户确认"),
                    "confirmation_required": True,
                    "required_tool": "preview_change",
                    "would_modify": sorted(f"{k[:-1]} {i}" for k, i in touched),
                    "already_modified_this_turn": already,
                    "hint": ("能用一次工具完成的批量修改（如 assign_properties）就整体预演一次；"
                             "本轮已经改掉的实体要如实告诉用户"),
                })
                self.tool_log.append((name, arguments, result))
                return result
        handler: Callable[..., ToolResult] | None = getattr(self, name, None)
        if handler is None or name.startswith("_"):
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工具"})
        try:
            result = handler(**arguments)
        except TypeError as exc:
            result = ToolResult(False, {"error": f"参数不对：{exc}"})
        if touched and result.ok:
            self._turn_touched |= touched
        self.tool_log.append((name, arguments, result))
        return result


# --------------------------------------------------------------------------- 模型提供方

class Provider(Protocol):
    """只要能吃 messages + tools 吐回复，什么后端都行。"""

    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...


@dataclass
class ScriptedProvider:
    """按脚本回放的假后端：离线测试与演示都用它，不需要密钥也不怕断网。"""
    script: Sequence[dict]
    seen: list[list[dict]] = field(default_factory=list)
    _cursor: int = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        self.seen.append(list(messages))
        if self._cursor >= len(self.script):
            raise AssertionError("脚本已用尽，说明轮数比预期多")
        item = self.script[self._cursor]
        self._cursor += 1
        return item


class DeepSeekProvider:
    """OpenAI 兼容接口的云端后端。

    密钥从外部传入，**不落盘、不进日志、不进 repr**。调用方自己从环境变量读：

        DeepSeekProvider(api_key=os.environ["DEEPSEEK_API_KEY"])

    temperature 默认 0：评测集要可复现，演示也要和报告里的数字一致。
    """

    def __init__(self, api_key: str, model: str = "deepseek-v4-flash",
                 base_url: str = "https://api.deepseek.com", temperature: float = 0.0):
        from openai import OpenAI          # 延迟导入：没装 openai 也能跑离线测试
        if not api_key or not api_key.strip():
            raise ValueError("缺少 API 密钥。请设置环境变量 DEEPSEEK_API_KEY 后重试。")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                      "cache_hit_tokens": 0, "calls": 0}

    def __repr__(self) -> str:              # 防止密钥随对象打印泄露
        return f"DeepSeekProvider(model={self._model!r})"

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, tools=tools,
            temperature=self._temperature,
        )
        u = getattr(response, "usage", None)
        if u is not None:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            self.usage["cache_hit_tokens"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0

        message = response.choices[0].message
        if message.tool_calls:
            calls = []
            for c in message.tool_calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"__invalid_json__": c.function.arguments}
                calls.append({"id": c.id, "name": c.function.name, "arguments": args})
            return {"tool_calls": calls}
        return {"content": message.content or ""}


# --------------------------------------------------------------------------- 主循环

@dataclass
class TurnResult:
    reply: str
    rounds: int
    tool_calls: list[tuple[str, dict]]
    session: Session
    stopped_by_limit: bool = False
    # 由对话层填写的分段耗时。保留默认值，让单元测试、离线脚本和第三方调用
    # 不必为了新增观测能力同步修改构造代码。
    metrics: dict[str, Any] = field(default_factory=dict)


def run_turn(user_text: str, provider: Provider, session: Session | None = None,
             max_rounds: int = MAX_ROUNDS) -> TurnResult:
    """跑完一轮对话：模型调工具、读结果、再调，直到给出最终回答。"""
    from workflow import refresh_workflow_message, workflow_message

    session = session or Session()
    session.begin_user_turn(user_text)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            workflow_message(session),
                            {"role": "user", "content": user_text}]
    calls: list[tuple[str, dict]] = []

    for round_index in range(max_rounds):
        reply = provider.complete(messages, TOOLS)
        if not reply.get("tool_calls"):
            return TurnResult(reply.get("content", ""), round_index + 1, calls, session)

        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"],
                                         "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                           for c in reply["tool_calls"]],
        })
        for call in reply["tool_calls"]:
            result = session.dispatch(call["name"], call["arguments"])
            calls.append((call["name"], call["arguments"]))
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": result.to_json()})
        refresh_workflow_message(messages, session)

    return TurnResult("已达到最大工具调用轮数，仍未得到结论。请检查模型或缩小问题范围。",
                      max_rounds, calls, session, stopped_by_limit=True)
