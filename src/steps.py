"""分析步：Abaqus 式的传播（Propagated）与失活（Deactivated）。

Abaqus 里荷载和边界条件不是"属于模型"的，而是**属于某个分析步**：在
Step-1 里建的东西会自动沿用到后面每一个分析步，直到被改写或失活。这件事
看着只是省了重复声明，实际上它是"同一个结构在不同配置下的一串工况"这个
概念的载体——拆一个支座、撤一阵风、换一种约束，都是改配置而不是改模型。

本模块只管**输入侧的配置解析**：给定一串分析步，算出每一步实际生效的荷载
与边界条件。求解交给 session 层。

能力边界，写在前面免得被当成 Abaqus 的等价物：

* **每一步都从未变形、无应力的状态重解**，不把上一步的状态带进来。
  线弹性下这没有问题——撤掉一个支座之后的平衡态就是重解出来的那个，
  弹性没有路径依赖。但它表达不了施工过程：后装的杆件不会"躲开"先前的
  变形，塑性也不会把残余应力留到下一步。
* **杆件不能在步之间生灭**。没有单元的 birth/death，所以"先装钢柱再浇
  楼板"这类序列做不了。

这两条都不是实现偷懒，是模型层面的取舍；要突破得先有能携带状态的增量
求解器，那是另一件事。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

#: 初始步的名字。Abaqus 里它固定叫 Initial，放模型自带的那套支座。
INITIAL_STEP = "Initial"

#: 分析步支持的分析类型。"step" 不在其列——分析步之间的非比例加载由每个
#: 工况各自的幅值曲线表达，不需要再套一层。
#:
#: **material 暂时不在其列**，而且这是个真限制不是遗漏：一个分析步里往往
#: 有好几个工况同时生效，材料非线性不能事后叠加，得有一个"把几个工况合成
#: 一个再增量求解"的入口——P-Δ 那边是 solve_step，材料那边还没有。放进来
#: 就只能偷偷按单工况各算各的，那是错的答案而且看不出来。
STEP_ANALYSES = ("linear", "pdelta")

#: 内置幅值曲线，与 frame3d.BUILTIN_AMPLITUDES 对齐。
_BUILTIN_AMPLITUDES = ("RAMP", "STEP")


@dataclass(frozen=True)
class Step:
    """一个分析步声明的**增量**，不是它的全量状态。

    ``loads`` 与 ``supports`` 写的是这一步新建或改写的东西；没提到的沿用
    上一步。要让某样东西消失得显式写进 deactivate——**不写就是继续生效**，
    这正是 Abaqus 的语义，也是最容易被误解的一点。
    """

    name: str
    analysis: str = "linear"
    #: 本步新建或改写的荷载：工况名 -> 幅值曲线名
    loads: dict[str, str] = field(default_factory=dict)
    #: 本步失活的工况名
    deactivate_loads: tuple[str, ...] = ()
    #: 本步新建或改写的支座：节点号 -> {"fix": [...], "spring": [...]}
    supports: dict[int, dict[str, Any]] = field(default_factory=dict)
    #: 本步失活的支座所在节点号
    deactivate_supports: tuple[int, ...] = ()
    increments: int = 10

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("分析步要有名字")
        if self.name == INITIAL_STEP:
            raise ValueError(
                f"{INITIAL_STEP!r} 是初始步的保留名字，它由模型自带的支座构成")
        if self.analysis not in STEP_ANALYSES:
            raise ValueError(
                f"分析步 {self.name!r} 的 analysis 只能是 {list(STEP_ANALYSES)}，"
                f"收到 {self.analysis!r}")
        if self.increments < 1:
            raise ValueError(f"分析步 {self.name!r} 的 increments 必须为正整数")
        clash = set(self.loads) & set(self.deactivate_loads)
        if clash:
            raise ValueError(
                f"分析步 {self.name!r} 同时激活又失活了工况 {sorted(clash)}；"
                "这两件事的先后顺序没有公认答案，不替你猜")
        clash = set(self.supports) & set(self.deactivate_supports)
        if clash:
            raise ValueError(
                f"分析步 {self.name!r} 同时改写又失活了节点 {sorted(clash)} 的支座")


@dataclass(frozen=True)
class EffectiveStep:
    """某个分析步**实际生效**的全量配置，传播与失活都已经结算完。"""

    name: str
    analysis: str
    loads: dict[str, str]
    supports: dict[int, dict[str, Any]]
    increments: int
    #: 相对上一步的变化，给用户看的。传播是隐式的，不把变化摆出来，
    #: "这一步到底和上一步差在哪"就只能靠人对着两份全量自己比。
    changes: tuple[str, ...]


def _support_entries(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """从模型载荷里取出初始支座，键是节点号。"""
    out: dict[int, dict[str, Any]] = {}
    for item in payload.get("supports") or []:
        out[int(item["node"])] = {k: v for k, v in item.items() if k != "node"}
    return out


def _known_cases(payload: dict[str, Any]) -> set[str]:
    cases = {str(c.get("name")) for c in (payload.get("load_cases") or [])}
    return cases or {"default"}


def resolve(steps: list[Step], payload: dict[str, Any]) -> list[EffectiveStep]:
    """把一串分析步的增量声明结算成每一步的全量配置。

    初始状态是模型自带的支座（对应 Abaqus 的 Initial 步）与"没有荷载"。

    失活一个本来就没生效的东西会**报错而不是默默跳过**。默默跳过意味着
    把节点号写错、或者把失活写在激活之前，都不会有任何提示，而结果看着
    完全正常——这正是这个项目里反复出现的那一类错。
    """
    active_loads: dict[str, str] = {}
    active_supports = _support_entries(payload)
    cases = _known_cases(payload)
    amplitudes = set(payload.get("amplitudes") or {}) | set(_BUILTIN_AMPLITUDES)

    seen: set[str] = set()
    out: list[EffectiveStep] = []
    for step in steps:
        if step.name in seen:
            raise ValueError(f"分析步名字重复：{step.name!r}")
        seen.add(step.name)
        changes: list[str] = []

        for case in step.deactivate_loads:
            if case not in active_loads:
                raise ValueError(
                    f"分析步 {step.name!r} 要失活工况 {case!r}，但它在这一步本来"
                    f"就没生效；当前生效的是 {sorted(active_loads)}")
            del active_loads[case]
            changes.append(f"失活荷载 {case}")
        for node in step.deactivate_supports:
            if node not in active_supports:
                raise ValueError(
                    f"分析步 {step.name!r} 要失活节点 {node} 的支座，但它在这一步"
                    f"本来就没生效；当前有支座的是 {sorted(active_supports)}")
            del active_supports[node]
            changes.append(f"失活支座 节点{node}")

        for case, amp in step.loads.items():
            if case not in cases:
                raise ValueError(
                    f"分析步 {step.name!r} 引用了未定义的荷载工况 {case!r}；"
                    f"已定义的是 {sorted(cases)}")
            if amp not in amplitudes:
                raise ValueError(
                    f"分析步 {step.name!r} 给工况 {case!r} 指定了未定义的幅值曲线 "
                    f"{amp!r}；可用的是 {sorted(amplitudes)}")
            if active_loads.get(case) != amp:
                verb = "改写" if case in active_loads else "新建"
                changes.append(f"{verb}荷载 {case}={amp}")
            active_loads[case] = amp
        for node, entry in step.supports.items():
            if active_supports.get(node) != entry:
                verb = "改写" if node in active_supports else "新建"
                changes.append(f"{verb}支座 节点{node}")
            active_supports[node] = dict(entry)

        if not active_supports:
            raise ValueError(
                f"分析步 {step.name!r} 结算下来一个支座都没有了，结构成了机构。"
                "失活支座时至少要留下足以消除刚体位移的那些")
        if not active_loads:
            raise ValueError(
                f"分析步 {step.name!r} 结算下来一个荷载都没有。分析步之间是传播的，"
                "第一步要显式给出 loads")

        out.append(EffectiveStep(
            name=step.name,
            analysis=step.analysis,
            loads=dict(active_loads),
            supports={k: dict(v) for k, v in active_supports.items()},
            increments=step.increments,
            changes=tuple(changes) or ("与上一步相同（全部沿用）",)))
    return out


def payload_for(effective: EffectiveStep,
                payload: dict[str, Any]) -> dict[str, Any]:
    """把某一步生效的支座写回模型载荷，得到这一步要求解的那个模型。

    只改 supports。哪些工况生效、各按什么幅值曲线加，是求解时的事；写进
    模型会把"结构配置"和"荷载历程"两件事混在一起。
    """
    out = deepcopy(payload)
    out["supports"] = [{"node": node, **entry}
                       for node, entry in sorted(effective.supports.items())]
    return out


def from_payload(payload: dict[str, Any]) -> list[Step]:
    """从模型载荷里读出分析步声明。"""
    return [
        Step(name=str(item["name"]),
             analysis=str(item.get("analysis", "linear")),
             loads={str(k): str(v) for k, v in (item.get("loads") or {}).items()},
             deactivate_loads=tuple(
                 str(c) for c in (item.get("deactivate_loads") or ())),
             supports={int(k): dict(v)
                       for k, v in (item.get("supports") or {}).items()},
             deactivate_supports=tuple(
                 int(n) for n in (item.get("deactivate_supports") or ())),
             increments=int(item.get("increments", 10)))
        for item in payload.get("steps") or []
    ]


def to_payload(steps: list[Step]) -> list[dict[str, Any]]:
    """把分析步声明写回模型载荷的形状。"""
    out = []
    for s in steps:
        item: dict[str, Any] = {"name": s.name, "analysis": s.analysis,
                                "increments": s.increments}
        if s.loads:
            item["loads"] = dict(s.loads)
        if s.deactivate_loads:
            item["deactivate_loads"] = list(s.deactivate_loads)
        if s.supports:
            item["supports"] = {str(k): dict(v) for k, v in s.supports.items()}
        if s.deactivate_supports:
            item["deactivate_supports"] = list(s.deactivate_supports)
        out.append(item)
    return out


__all__ = ["INITIAL_STEP", "STEP_ANALYSES", "EffectiveStep", "Step",
           "from_payload", "payload_for", "resolve", "to_payload"]
