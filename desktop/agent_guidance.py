"""根据确定性工程状态给出下一步动作，不依赖 Qt 或大模型。"""

from __future__ import annotations

from workflow import WorkflowPhase

Suggestion = tuple[str, str]


def next_suggestions(status, selection_kind: str | None = None,
                     selection_id: int | None = None,
                     mode: str = "模型") -> list[Suggestion]:
    """返回最多三个可解释建议，排序即界面优先级。"""
    counts = status.counts
    if status.phase is WorkflowPhase.EMPTY:
        return [
            ("参数化建模",
             "我要创建参数化结构。请先问我结构类型、跨度、层高和开间，不要假设数值。"),
            ("从图纸开始",
             "我要从结构图纸开始建模，请告诉我需要提供哪些视图和尺寸信息。"),
            ("手动建模步骤",
             "我想手动建立结构模型，请按当前界面告诉我下一步操作。"),
        ]

    if status.phase is WorkflowPhase.DRAFT:
        items: list[Suggestion] = []
        if selection_kind == "member":
            items.append(("编辑杆件荷载",
                          f"检查杆件 {selection_id} 的荷载，并告诉我如何修改。"))
        elif selection_kind == "node":
            items.append(("编辑节点边界",
                          f"检查节点 {selection_id} 的约束和节点荷载，并告诉我如何修改。"))
        if not counts["materials"] or not counts["sections"]:
            items.append(("补全材料截面",
                          "根据当前模型补全材料和截面；缺少的参数先问我。"))
        if not counts["supports"]:
            items.append(("设置边界条件",
                          "告诉我如何在当前界面为节点设置边界条件。"))
        if not counts["load_cases"]:
            items.append(("添加荷载工况",
                          "根据当前模型创建荷载工况；荷载参数不清楚时先问我。"))
        items.append(("检查模型缺项",
                      "检查当前模型还缺哪些分析条件，并按优先级列出。"))
        return items[:3]

    if status.phase is WorkflowPhase.READY:
        if selection_kind == "member":
            return [
                ("复核杆件属性", f"复核杆件 {selection_id} 的材料、截面和释放。"),
                ("设置杆件荷载", f"检查并设置杆件 {selection_id} 的荷载。"),
                ("开始求解", "求解当前模型，并先报告模型检查结果。"),
            ]
        if selection_kind == "node":
            return [
                ("复核节点边界", f"复核节点 {selection_id} 的约束和给定位移。"),
                ("设置节点荷载", f"检查并设置节点 {selection_id} 的荷载。"),
                ("开始求解", "求解当前模型，并先报告模型检查结果。"),
            ]
        return [
            ("开始求解", "求解当前模型，并先报告模型检查结果。"),
            ("预览分析网格", "预览当前模型的分析网格和物理杆件映射。"),
            ("复核边界条件", "复核当前模型的边界条件和荷载是否合理。"),
        ]

    if selection_kind == "member":
        return [
            ("查看杆件内力", f"查看杆件 {selection_id} 的内力分布和控制截面。"),
            ("查看截面应力", f"查看杆件 {selection_id} 控制截面的应力。"),
            ("调整杆件参数", f"分析杆件 {selection_id} 是否需要调整截面并先预演修改。"),
        ]
    if selection_kind == "node":
        return [
            ("查看节点位移", f"查看节点 {selection_id} 的位移和转角。"),
            ("查看节点反力", f"查看节点 {selection_id} 的支座反力。"),
            ("复核节点约束", f"复核节点 {selection_id} 的边界条件是否合理。"),
        ]
    if mode == "云图":
        return [
            ("定位云图极值", "定位当前云图绝对极值，并说明所在杆件和工况。"),
            ("解释颜色分布", "解释当前云图颜色分布、正负号和异常跳变。"),
            ("查看控制杆件", "查看当前云图控制杆件的内力和截面应力。"),
        ]
    return [
        ("查看最大位移", "查看当前结果的最大位移和控制工况。"),
        ("查看支座反力", "查看当前工况的支座反力和平衡情况。"),
        ("执行强度校核", "对当前结果执行强度校核并解释控制杆件。"),
    ]
