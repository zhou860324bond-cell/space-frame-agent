"""模型构建历史。

界面上原来只看得见**最终状态**——表格、图、数字。看不出这个模型是怎么
一步步搭起来的：先定了材料截面，再生成拓扑，再加约束，再加荷载，最后求解。
对一个"智能建模"的工具来说，过程恰恰是最该被看见的那部分：
它到底做了什么、每一步之后模型变成了什么样。

这一层就干这个：每次有工具改动了模型，记一条

* **人话摘要** —— "生成 3 跨 2 层框架：24 节点、36 杆件"，不是 JSON；
* **模型快照** —— 该步**之后**的完整模型，可以拿去重建、可以拿去画图；
* **规模变化** —— 节点/杆件/约束/工况的增减，一眼看出这步动了什么。

挂在 `Session.dispatch` 上，所以三条建模路径（参数化、自然语言、手工编辑）
自动都有历史，不必各自埋点。

快照是深拷贝。刚架模型撑死几百 KB，几十步也就几兆；`MAX_STEPS` 兜个底，
免得长会话把内存吃光。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

MAX_STEPS = 200

# 会改动模型的工具。不在这个集合里的（各种 query_*、plot_*、solve_model）
# 不记历史——否则查十次结果就多出十条一模一样的快照，时间轴全是噪声。
MUTATING = frozenset({
    "define_materials_and_sections", "generate_frame", "generate_portal_frame",
    "add_nodes", "add_members", "remove_members", "remove_nodes", "set_model",
    "set_load_cases", "add_load_case", "set_nodal_load", "set_member_load",
    "set_member_span_load", "set_prescribed_displacement",
    "add_self_weight", "set_units",
    # 一榀 / 拉伸 / 算子。**漏登记的话撤销和时间线就看不见它们**——
    # 用户抽了一根柱，按 Ctrl+Z 却什么都没发生
    "generate_bent", "extrude_bents", "add_bracing", "raise_nodes", "retaper",
    # 属性面板的就地修改。**必须登记**——不登记的话用户改完按 Ctrl+Z
    # 什么也不会发生，而他完全预期能撤销
    "edit_member", "edit_node", "assign_properties", "set_supports", "define_set",
    # 手工画图的提交。一次可能画了很多根杆，但**在时间轴上是一步**——
    # 画的过程中的每一次点击不该各占一格，那样撤销一次只退回一根杆
    "apply_draft",
    "apply_multimodal_draft",
})


def _stats(model: dict[str, Any]) -> dict[str, int]:
    return {"节点": len(model.get("nodes") or []),
            "杆件": len(model.get("members") or []),
            "约束": len(model.get("supports") or []),
            "材料": len(model.get("materials") or []),
            "截面": len(model.get("sections") or []),
            "工况": len(model.get("load_cases") or []),
            "组合": len(model.get("combos") or [])}


def _describe(tool: str, args: dict[str, Any], payload: dict[str, Any],
              model: dict[str, Any]) -> str:
    """把一次工具调用说成人话。

    人看时间轴是为了快速理解"这步干了什么"，不是为了读 JSON。
    所以宁可写死几条模板，也不要把参数原样倒出来。
    """
    if tool == "define_materials_and_sections":
        mats = "、".join(m["name"] for m in args.get("materials", []))
        secs = "、".join(s["name"] for s in args.get("sections", []))
        return f"定义材料 {mats}，截面 {secs}"
    if tool == "generate_frame":
        spans = args.get("spans") or []
        storeys = args.get("storeys") or []
        bays = args.get("bays") or []
        shape = f"{len(spans)} 跨 {len(storeys)} 层"
        if bays:
            shape += f" {len(bays)} 开间"
        base = "柱底固接" if args.get("base", "fixed") == "fixed" else "柱底铰接"
        return (f"生成规整框架：{shape}，{base}"
                + ("，梁两端铰接" if args.get("beam_release") else ""))
    if tool == "generate_portal_frame":
        spans = args.get("spans") or []
        return (f"生成门式刚架：{len(spans)} 跨，檐口 {args.get('eave_height')} m，"
                f"屋脊升高 {args.get('ridge_rise')} m")
    if tool == "add_nodes":
        return f"追加 {len(args.get('coordinates') or [])} 个节点"
    if tool == "add_members":
        return (f"追加 {len(args.get('pairs') or [])} 根杆件"
                f"（截面 {args.get('section')}）")
    if tool == "remove_members":
        # 参数名是 ids 不是 members。取错键不会报错，只会一直显示 None——
        # 这类摘要 bug 没人会专门去看，得靠测试盯住
        dropped = payload.get("removed") or args.get("ids") or []
        orphans = payload.get("removed_orphan_nodes") or []
        text = f"删除 {len(dropped)} 根杆件"
        if len(dropped) <= 6:
            text += f"（{'、'.join(str(x) for x in dropped)}）"
        if orphans:
            text += f"，并清除 {len(orphans)} 个孤立节点"
        return text
    if tool == "remove_nodes":
        dropped = payload.get("removed") or args.get("ids") or []
        connected = payload.get("removed_connected_members") or []
        text = f"删除 {len(dropped)} 个节点"
        if len(dropped) <= 6:
            text += f"（{'、'.join(str(x) for x in dropped)}）"
        if connected:
            text += f"，并清除 {len(connected)} 根相连杆件"
        return text
    if tool == "set_model":
        return f"整体提交模型：{_stats(model)['节点']} 节点、{_stats(model)['杆件']} 杆件"
    if tool == "set_load_cases":
        cases = [c.get("name") for c in args.get("cases") or []]
        combos = [c.get("name") for c in args.get("combos") or []]
        text = f"设定荷载工况 {'、'.join(cases)}"
        if combos:
            text += f"；组合 {'、'.join(combos)}"
        return text
    if tool == "add_load_case":
        return f"创建分析工况「{payload.get('case', args.get('name'))}」"
    if tool == "set_nodal_load":
        return (f"在工况「{payload.get('case')}」设置载荷「{payload.get('name')}」，节点 "
                f"{payload.get('node')} 集中力")
    if tool == "set_member_load":
        return (f"在工况「{payload.get('case')}」设置载荷「{payload.get('name')}」，杆件 "
                f"{payload.get('member')} 均布荷载")
    if tool == "set_member_span_load":
        return (f"在工况「{payload.get('case')}」设置载荷「{payload.get('name')}」，"
                f"杆件 {payload.get('member')} {payload.get('kind')}")
    if tool == "set_prescribed_displacement":
        return (f"在工况「{payload.get('case')}」设置边界条件「{payload.get('name')}」，"
                f"节点 {payload.get('node')} 给定位移")
    if tool == "add_self_weight":
        n = payload.get("members", "")
        return f"按 ρ·A·g 生成自重，覆盖 {n} 根杆件"
    if tool == "generate_bent":
        cols = len(args.get("columns") or [])
        levels = len(args.get("levels") or [])
        pts = args.get("profile") or []
        shape = ("平屋面" if len(pts) <= 2 else
                 ("双坡" if len(pts) == 3 else f"{len(pts)} 折点屋面"))
        return (f"生成一榀刚架：{cols} 柱"
                + (f"、{levels} 层楼面" if levels else "")
                + f"、{shape}")
    if tool == "extrude_bents":
        n = payload.get("bent_count", "?")
        return f"沿纵向拉伸为 {n} 榀，生成 {len(payload.get('tie_member_ids') or [])} 根连系梁"
    if tool == "add_bracing":
        kind = {"X": "交叉撑", "V": "人字撑", "single": "单斜撑"}.get(
            args.get("kind", "X"), args.get("kind"))
        return (f"加{kind} {len(payload.get('brace_member_ids') or [])} 根，"
                f"覆盖 {payload.get('panels', '?')} 个框格")
    if tool == "raise_nodes":
        return (f"抬高 {len(payload.get('moved_nodes') or [])} 个节点 "
                f"{args.get('dz', 0):+g} m")
    if tool == "retaper":
        return (f"{len(payload.get('changed_members') or [])} 根杆件换成截面 "
                f"{args.get('section')}")
    if tool == "edit_member":
        bits = "、".join(f"{k}→{v['now']}" for k, v in
                        (payload.get("changed") or {}).items())
        return f"修改杆件 {args.get('member_id')}：{bits or '无改动'}"
    if tool == "edit_node":
        bits = "、".join(f"{k}→{v['now']}" for k, v in
                        (payload.get("changed") or {}).items())
        return f"修改节点 {args.get('node_id')}：{bits or '无改动'}"
    if tool == "assign_properties":
        return (f"给 {len(payload.get('members') or [])} 根杆件指派材料 "
                f"{args.get('material')}、截面 {args.get('section')}")
    if tool == "set_supports":
        return (f"给 {len(payload.get('nodes') or [])} 个节点设置边界条件 "
                f"{args.get('name') or 'BC'}")
    if tool == "define_set":
        n = len(payload.get("members") or payload.get("nodes") or [])
        return f"定义集合「{args.get('name')}」，{n} 个对象"
    if tool == "apply_draft":
        s = payload.get("summary") or {}
        return (f"手工修改：{s.get('nodes', 0)} 个节点、{s.get('members', 0)} 根杆件"
                if not args.get("note") else str(args["note"]))
    if tool == "apply_multimodal_draft":
        return "提交图片识别草稿"
    if tool == "set_units":
        return f"单位制换算：{payload.get('was')} → {payload.get('units')}"
    return tool


@dataclass
class BuildStep:
    """一步。model 是这步**之后**的完整模型，可以直接拿去重建或画图。"""
    index: int
    tool: str
    summary: str
    model: dict[str, Any]
    stats: dict[str, int]
    delta: dict[str, int]
    ok: bool
    error: str | None = None
    # 不只保存“最后长什么样”，还保存 Agent 调了哪个工具、给了什么参数、
    # 工具回了什么。没有这三项，历史只能撤销，不能成为可复用的建模示例。
    args: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    multimodal_provenance: dict[str, Any] | None = None

    @property
    def changed(self) -> str:
        """这步动了什么，人话。没变就说没变——比留空诚实。"""
        bits = [f"{k} {v:+d}" for k, v in self.delta.items() if v]
        return "　".join(bits) if bits else "规模未变"


@dataclass
class BuildHistory:
    """一串 BuildStep，外加一个游标。

    **游标是撤销的全部机制。** 每一步都存了这步之后的完整模型快照，
    所以"撤销"不是反向计算，而是把某个快照装回去——这对我们特别合适：
    模型是纯数据，没有副作用要回滚。

    游标语义：`cursor` 是 `steps` 里的**列表下标**，指向最后一步已生效的步。
    （不要和 `BuildStep.index` 混淆——那是给用户看的步号，
    步数超过 MAX_STEPS 丢弃最早几步之后，两者就不再相等了。）

        cursor = -1   还没有任何一步生效，模型是空的
        cursor = 2    第 0、1、2 步都生效了，当前模型是 steps[2].model

    在游标退回去之后再做新的改动，重做那一段就**丢掉**——这是所有
    编辑器的通行行为，用户也是这么预期的。
    """
    steps: list[BuildStep] = field(default_factory=list)
    cursor: int = -1

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, k: int) -> BuildStep:
        return self.steps[k]

    # --- 撤销 / 重做 ---

    @property
    def can_undo(self) -> bool:
        return self.cursor >= 0

    @property
    def can_redo(self) -> bool:
        return self.cursor < len(self.steps) - 1

    def model_at(self, cursor: int) -> dict[str, Any]:
        """游标处的模型快照。**返回深拷贝**——直接把快照交出去的话，
        调用方一改模型就把历史也改了，而且是静悄悄地改。"""
        if cursor < 0:
            return {}
        return copy.deepcopy(self.steps[cursor].model)

    def provenance_at(self, cursor: int) -> dict[str, Any] | None:
        if cursor < 0:
            return None
        return copy.deepcopy(self.steps[cursor].multimodal_provenance)

    def undo(self) -> dict[str, Any] | None:
        """退一步，返回退到的那个模型。已经在最前面就返回 None。"""
        if not self.can_undo:
            return None
        self.cursor -= 1
        return self.model_at(self.cursor)

    def redo(self) -> dict[str, Any] | None:
        if not self.can_redo:
            return None
        self.cursor += 1
        return self.model_at(self.cursor)

    def goto(self, cursor: int) -> dict[str, Any] | None:
        """直接跳到某一步。时间线面板点某一行走这条路。"""
        if not -1 <= cursor <= len(self.steps) - 1:
            return None
        self.cursor = cursor
        return self.model_at(cursor)

    # --- 记录 ---

    def record(self, tool: str, args: dict[str, Any], ok: bool,
               payload: dict[str, Any], model: dict[str, Any],
               multimodal_provenance: dict[str, Any] | None = None) -> BuildStep | None:
        """记一步。不改模型的工具直接跳过。"""
        if tool not in MUTATING:
            return None
        # 声明自己什么都没改的调用不占一步。**属性面板会反复提交当前值**，
        # 记下来的话时间轴会被空步骤淹掉，而撤销一次看不出任何变化
        if ok and payload.get("no_change"):
            return None
        # 游标退回去过，就把后面那段重做丢掉。分叉出第二条历史线
        # 听起来更强，实际上没人能在界面上理解它
        if self.cursor < len(self.steps) - 1:
            del self.steps[self.cursor + 1:]
        stats = _stats(model)
        before = self.steps[-1].stats if self.steps else {k: 0 for k in stats}
        step = BuildStep(
            index=(self.steps[-1].index + 1) if self.steps else 0,
            tool=tool,
            summary=_describe(tool, args, payload, model) if ok
                    else f"{tool}（未通过）",
            model=copy.deepcopy(model),
            stats=stats,
            delta={k: stats[k] - before.get(k, 0) for k in stats},
            ok=ok,
            error=None if ok else str(payload.get("error")
                                      or payload.get("errors") or "")[:200],
            args=copy.deepcopy(args),
            payload=copy.deepcopy(payload),
            multimodal_provenance=copy.deepcopy(multimodal_provenance),
        )
        self.steps.append(step)
        if len(self.steps) > MAX_STEPS:
            # 丢最早的，但把索引留住——用户看到的步号得和实际发生的次序一致
            del self.steps[0]
        self.cursor = len(self.steps) - 1
        return step

    def clear(self) -> None:
        self.steps.clear()
        self.cursor = -1

    def timeline(self) -> list[dict[str, Any]]:
        """给界面用的紧凑清单，不带快照。"""
        return [{"步": s.index + 1, "工具": s.tool, "做了什么": s.summary,
                 "规模变化": s.changed,
                 "状态": "成功" if s.ok else "未通过"}
                for s in self.steps]

    def learning_trace(self, include_snapshots: bool = False) -> list[dict[str, Any]]:
        """导出可复用的 Agent 建模轨迹，而不是只给人看的摘要。

        默认不重复塞每一步的完整模型，避免大模型训练样本膨胀；需要精确回放或
        排查某一步时可打开 ``include_snapshots``。
        """
        rows = []
        for step in self.steps:
            row: dict[str, Any] = {
                "step": step.index + 1,
                "tool": step.tool,
                "arguments": copy.deepcopy(step.args),
                "ok": step.ok,
                "summary": step.summary,
                "error": step.error,
                "result": copy.deepcopy(step.payload),
                "stats": copy.deepcopy(step.stats),
                "delta": copy.deepcopy(step.delta),
            }
            if include_snapshots:
                row["model_after"] = copy.deepcopy(step.model)
            rows.append(row)
        return rows
