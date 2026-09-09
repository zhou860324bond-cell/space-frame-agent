"""改动的影响：改了什么 → 什么失效 → 什么要跟着清理。

**这张表存在的理由是一次真实的 bug。**

删杆件时要清掉引用它的荷载。我按记忆写了清理代码，只过滤了
`member_loads`——而模型里的荷载有四类：`nodal_loads`、`member_loads`、
`member_spans`、`settlements`。自重写的恰好是 `member_spans`，于是
"加自重 → 删一根柱"这个组合会留下指向不存在杆件的荷载，
校验报「工况 D 在不存在的杆件 1 上施加了荷载」，而用户只是删了根柱。

单独删没问题，加了自重才出问题——**这类交互 bug 靠一处处手写清理是防不住的**。

所以这里立两条规矩：

1. **键从 schema 取，不从记忆取。** `LOAD_KEYS` 是照着 `model_io` 的
   schema 列的，加了新荷载类型时这里会被测试拎出来。
2. **清理与失效只写一遍。** 删杆件、删节点、撤销、手工改模型都调这里，
   而不是各自写一遍——加第四个入口时，漏的就是第四次。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# 荷载在模型里的四种写法。**顺序与 model_io 的 schema 一致。**
# 每一项是 (键名, 它引用什么)
LOAD_KEYS: tuple[tuple[str, str], ...] = (
    ("nodal_loads", "node"),
    ("member_loads", "member"),
    ("member_spans", "member"),
    ("settlements", "node"),
)

# 荷载既可以直接写在模型顶层（单工况），也可以写在 load_cases 里（多工况）。
# **两处都要清。** 只清一处的话，用哪种写法建的模型会有不同的行为。


class Change(str, Enum):
    """一次改动动了什么。名字按用户的说法，不按代码的说法。"""

    GEOMETRY = "geometry"        # 节点、杆件增删或移动
    SECTIONS = "sections"        # 材料与截面定义
    SUPPORTS = "supports"        # 约束
    LOADS = "loads"              # 荷载工况与组合
    UNITS = "units"              # 单位制换算
    SETS = "sets"                # 命名集合
    NOTHING = "nothing"          # 只查询，不改模型


@dataclass(frozen=True)
class Impact:
    """一类改动的后果。

    `solution` 为真表示**已有的求解结果失效**——这是最容易漏、
    后果也最隐蔽的一条：模型变了而结果还在，界面会拿着对不上的位移
    去画变形图，画得出来，且看不出错。
    """

    solution: bool          # 求解结果失效
    frame: bool             # 编译好的 Frame 失效
    prune_loads: bool       # 要清掉指向已删对象的荷载
    prune_sets: bool        # 要修剪命名集合
    note: str


IMPACT: dict[Change, Impact] = {
    Change.GEOMETRY: Impact(True, True, True, True,
                            "几何变了：结果失效；引用已删对象的荷载与集合要清"),
    Change.SECTIONS: Impact(True, True, False, False,
                            "刚度变了：结果失效，但拓扑没动"),
    Change.SUPPORTS: Impact(True, True, False, False,
                            "边界变了：结果失效"),
    Change.LOADS: Impact(True, False, False, False,
                         "荷载变了：结果失效，但 Frame 还能用"),
    Change.UNITS: Impact(True, True, False, False,
                         "换算是物理等价的，但数值全变了，结果必须重算"),
    Change.SETS: Impact(False, False, False, False,
                        "集合是纯命名层，不影响任何计算结果"),
    Change.NOTHING: Impact(False, False, False, False, "只读"),
}


# --------------------------------------------------------------- 清理

def prune_loads(model: dict[str, Any], *,
                dropped_members: set[int] | None = None,
                dropped_nodes: set[int] | None = None) -> dict[str, int]:
    """清掉引用已删杆件或节点的荷载。返回每类清了几条。

    **四类荷载、两个位置，一处都不能漏。** 漏一处的表现不是报错，
    是校验在下一步才失败，而错误信息指向的是删除操作之后的某一步。
    """
    members = dropped_members or set()
    nodes = dropped_nodes or set()
    if not members and not nodes:
        return {}

    counted: dict[str, int] = {}

    def clean(holder: dict[str, Any]) -> None:
        for key, refers_to in LOAD_KEYS:
            entries = holder.get(key)
            if not entries:
                continue
            gone = members if refers_to == "member" else nodes
            kept = [e for e in entries if int(e.get(refers_to, -1)) not in gone]
            if len(kept) != len(entries):
                counted[key] = counted.get(key, 0) + len(entries) - len(kept)
            holder[key] = kept

    clean(model)                                    # 单工况写法
    for case in model.get("load_cases") or []:      # 多工况写法
        clean(case)
    return counted


def prune_sets(model: dict[str, Any]) -> tuple[dict[str, int], list[str]]:
    """把命名集合里已经不存在的编号剪掉；剪空的集合整个去掉。

    返回 (每个集合剪了几条, 被剪空而删除的集合名)。

    剪空的整个去掉，是因为**空集合被引用时会静默什么都不做**——
    比当场报「没有这个集合」难查得多。
    """
    sets = model.get("sets")
    if not sets:
        return {}, []
    alive_m = {int(m["id"]) for m in model.get("members") or []}
    alive_n = {int(n["id"]) for n in model.get("nodes") or []}

    pruned: dict[str, int] = {}
    emptied: list[str] = []
    kept_sets: dict[str, Any] = {}
    for name, entry in sets.items():
        kept = dict(entry)
        for key, alive in (("members", alive_m), ("nodes", alive_n)):
            if key in kept:
                before = len(kept[key])
                kept[key] = [k for k in kept[key] if k in alive]
                if len(kept[key]) != before:
                    pruned[name] = pruned.get(name, 0) + before - len(kept[key])
        if kept.get("members") or kept.get("nodes"):
            kept_sets[name] = kept
        else:
            emptied.append(name)
    model["sets"] = kept_sets
    return pruned, emptied


def prune_orphan_nodes(model: dict[str, Any]) -> list[int]:
    """删掉不再与任何杆件相连的节点，并连带清掉它们的约束。

    留着孤立节点，刚度矩阵就有一整行零，求解直接报奇异——
    而用户以为自己只是删了一根杆，完全对不上。
    """
    members = model.get("members") or []
    used = {int(m["i"]) for m in members} | {int(m["j"]) for m in members}
    orphans = [int(n["id"]) for n in model.get("nodes") or []
               if int(n["id"]) not in used]
    if not orphans:
        return []
    gone = set(orphans)
    model["nodes"] = [n for n in model["nodes"] if int(n["id"]) not in gone]
    model["supports"] = [s for s in model.get("supports") or []
                         if int(s["node"]) not in gone]
    prune_loads(model, dropped_nodes=gone)
    return orphans


def apply(model: dict[str, Any], change: Change, *,
          dropped_members: set[int] | None = None,
          dropped_nodes: set[int] | None = None) -> dict[str, Any]:
    """按改动类型做完整清理，返回一份"做了什么"的报告。

    调用方只需说清楚**改动属于哪一类**，剩下的由这张表决定——
    这样加新的改动入口时，不必再想一遍"我该清什么"。
    """
    impact = IMPACT[change]
    report: dict[str, Any] = {}

    if impact.prune_loads:
        orphans = prune_orphan_nodes(model)
        counted = prune_loads(model, dropped_members=dropped_members,
                              dropped_nodes=set(orphans) | (dropped_nodes or set()))
        if orphans:
            report["removed_orphan_nodes"] = orphans
        if counted:
            report["pruned_loads"] = counted

    if impact.prune_sets:
        pruned, emptied = prune_sets(model)
        if pruned:
            report["pruned_sets"] = pruned
        if emptied:
            report["emptied_sets"] = emptied

    return report
