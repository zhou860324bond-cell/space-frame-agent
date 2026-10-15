"""对话输入框上方的"接下来可以问"建议，随模型状态和对话进展变化。

原来是两颗写死的按钮（门式刚架 / 检查模型），模型都算完了还在提示"建一榀门式
刚架"。主流 Agent 桌面端的做法是根据当前进度给出两三个后续问题，点一下直接发送。

这里不调用大模型去猜——那要多花一次请求、多等几秒，而且猜出来的问题可能根本
没有对应的工具。改为按工作流阶段（空模型 / 待补全 / 可求解 / 已求解）、待确认的
批量修改和用户引用的对象确定性地给出建议；每条建议都能落到已有工具或本地快速路径上。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Followup:
    label: str          # 按钮上的短词
    prompt: str         # 点击后发送的完整句子


def suggest(session, references=(), last_user: str = "", limit: int = 3) -> list[Followup]:
    from workflow import WorkflowPhase, inspect_workflow

    pending = getattr(session, "pending_change", None)
    if pending:
        return [Followup("确认修改", f"确认 {pending['preview_id']}"),
                Followup("取消修改", "取消")]

    status = inspect_workflow(session)
    names = "、".join(f"{'节点' if kind == 'node' else '杆件'}{ident}" for kind, ident in references)
    out: list[Followup] = []
    if status.phase is WorkflowPhase.EMPTY:
        out += [Followup("门式刚架", "创建一榀 24 米门式刚架，柱脚铰接，并说明还缺哪些工程参数"),
                Followup("多层框架", "建一个 3 跨 2 层的空间框架，跨度 6 m，层高 3.6 m，柱脚固接"),
                Followup("能做什么", "你能帮我完成哪些结构建模与分析工作？")]
    elif status.phase is WorkflowPhase.DRAFT:
        issues = " ".join(status.validation_errors)
        out.append(Followup("还缺什么", "检查模型，告诉我还缺哪些定义才能求解"))
        if "截面" in issues or "材料" in issues:
            out.append(Followup("补齐截面", "给还没有截面的杆件指派 Q355 H 型钢截面，并说明选型依据"))
        if "支座" in issues or "约束" in issues:
            out.append(Followup("补齐支座", "在底部节点加固接支座"))
        if "荷载" in issues or "工况" in issues:
            out.append(Followup("添加荷载", "添加恒载和活载工况，并生成基本组合"))
    elif status.phase is WorkflowPhase.READY:
        if names:
            out.append(Followup("查看所选", f"说明 {names} 的截面、支座和荷载"))
        out += [Followup("求解", "求解模型"), Followup("检查模型", "检查模型")]
    else:
        if names:
            out.append(Followup("所选内力", f"{names} 的内力和应力比是多少"))
        out += [Followup("最大位移", "最大位移"),
                Followup("强度验算", "对全部杆件做强度验算，列出最不利的几根"),
                Followup("支座反力", "查看反力"),
                Followup("生成计算书", "生成计算书，并概述主要结论")]
    seen = (last_user or "").strip()
    return [f for f in out if f.prompt != seen][:limit]
