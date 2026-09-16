"""多参数截面优化（section_optimizer）的单元测试。

测七件事：
1. SectionOptimizer 能正常创建，变量归一化正确
2. 单目标网格搜索能找到最优解（更大截面 → 更小位移，但更重）
3. 多目标优化能返回非空 Pareto 前沿
4. 位移约束能正确排除大位移的不可行解
5. 随机搜索能返回指定数量的评估点
6. 不支持的截面类型/目标/约束会报错
7. OptimizationResult.summary() 能输出可读摘要
"""

from __future__ import annotations

import pytest

from section_optimizer import SectionOptimizer, EvaluationPoint


# 一个简单的单跨单层框架模型
def _make_model() -> dict:
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}],
        "sections": [
            {"name": "COL", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7},
            {"name": "BEAM", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8e-7},
        ],
        "nodes": [
            {"id": 1, "x": 0, "y": 0, "z": 0},
            {"id": 2, "x": 0, "y": 0, "z": 3.6},
            {"id": 3, "x": 6.0, "y": 0, "z": 3.6},
            {"id": 4, "x": 6.0, "y": 0, "z": 0},
        ],
        "members": [
            {"id": 1, "i": 1, "j": 2, "section": "COL", "material": "Q355"},
            {"id": 2, "i": 2, "j": 3, "section": "BEAM", "material": "Q355"},
            {"id": 3, "i": 3, "j": 4, "section": "COL", "material": "Q355"},
        ],
        "supports": [
            {"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
            {"node": 4, "fix": [1, 1, 1, 1, 1, 1]},
        ],
        "load_cases": [{"name": "DL", "member_loads": [
            {"member": 2, "w": [0, 0, -10e3]}]}],
    }


# --------------------------------------------------------- 创建与校验

def test_create_optimizer():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.6, 3)},
        objectives=["weight"],
    )
    assert opt is not None
    assert "height" in opt._variables
    assert len(opt._variables["height"]) == 3


def test_variable_normalization_linspace():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.2, 0.8, 4)},
    )
    values = opt._variables["height"]
    assert len(values) == 4
    assert values[0] == pytest.approx(0.2)
    assert values[-1] == pytest.approx(0.8)


def test_variable_normalization_explicit_list():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": [0.3, 0.5, 0.7]},
    )
    assert opt._variables["height"] == [0.3, 0.5, 0.7]


def test_unsupported_section_type_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="不支持的截面类型"):
        SectionOptimizer(
            model=model, section_name="BEAM", section_type="不存在的截面",
            variables={"height": (0.3, 0.6, 3)},
        )


def test_unsupported_objective_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="不支持的目标"):
        SectionOptimizer(
            model=model, section_name="BEAM", section_type="工字形 / H 型钢",
            variables={"height": (0.3, 0.6, 3)},
            objectives=["nonexistent"],
        )


def test_unsupported_constraint_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="不支持的约束"):
        SectionOptimizer(
            model=model, section_name="BEAM", section_type="工字形 / H 型钢",
            variables={"height": (0.3, 0.6, 3)},
            constraints={"nonexistent": 1.0},
        )


def test_unknown_param_raises():
    model = _make_model()
    with pytest.raises(ValueError, match="没有参数"):
        SectionOptimizer(
            model=model, section_name="BEAM", section_type="工字形 / H 型钢",
            variables={"nonexistent_param": (0.3, 0.6, 3)},
        )


# --------------------------------------------------------- 单目标优化

def test_single_objective_grid_search_returns_best():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.6, 3)},
        objectives=["weight"],
    )
    result = opt.grid_search()
    assert result.algorithm == "grid_search"
    assert result.total_calls == 3
    assert result.best is not None
    assert result.best.feasible
    assert "weight" in result.best.objectives
    assert result.feasible_count > 0


def test_larger_section_has_smaller_displacement():
    """更大的截面（高度更大）应该有更小的位移，但更大的重量。"""
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": [0.3, 0.6]},
        objectives=["weight", "max_displacement"],
    )
    result = opt.grid_search()
    feasible = [e for e in result.evaluations if e.feasible]
    assert len(feasible) == 2
    # 按高度排序
    feasible.sort(key=lambda e: e.params["height"])
    small, large = feasible
    assert large.objectives["weight"] > small.objectives["weight"], "大截面更重"
    assert large.objectives["max_displacement"] <= small.objectives["max_displacement"], \
        "大截面位移更小"


# --------------------------------------------------------- 多目标优化

def test_multi_objective_returns_pareto_front():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.8, 5)},
        objectives=["weight", "max_displacement"],
    )
    result = opt.grid_search()
    assert len(result.pareto) > 0, "多目标优化应返回非空 Pareto 前沿"
    # Pareto 前沿上的点都是可行的
    assert all(p.feasible for p in result.pareto)


# --------------------------------------------------------- 约束

def test_displacement_constraint_filters_infeasible():
    """极小的位移约束应该排除所有解（因为实际位移不可能那么小）。"""
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.6, 3)},
        objectives=["weight"],
        constraints={"max_displacement_limit": 1e-9},  # 1nm，不可能满足
    )
    result = opt.grid_search()
    assert result.feasible_count == 0, "极小位移约束应排除所有解"
    assert result.best is None


def test_min_area_constraint():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.6, 3)},
        objectives=["weight"],
        constraints={"min_section_area": 0.001},
    )
    result = opt.grid_search()
    # 工字形截面面积应该大于 0.001
    assert result.feasible_count > 0


# --------------------------------------------------------- 随机搜索

def test_random_search_returns_n_samples():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.8, 10), "flange_width": (0.15, 0.3, 10)},
        objectives=["weight"],
        seed=42,
    )
    result = opt.random_search(n_samples=5)
    assert result.total_calls == 5
    assert result.algorithm == "random_search"


def test_optimize_method_dispatches():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.6, 3)},
        objectives=["weight"],
    )
    r1 = opt.optimize("grid_search")
    assert r1.algorithm == "grid_search"
    r2 = opt.optimize("random_search", n_samples=3)
    assert r2.algorithm == "random_search"
    with pytest.raises(ValueError, match="不支持的算法"):
        opt.optimize("nonexistent")


# --------------------------------------------------------- 结果摘要

def test_result_summary_contains_key_info():
    model = _make_model()
    opt = SectionOptimizer(
        model=model, section_name="BEAM", section_type="工字形 / H 型钢",
        variables={"height": (0.3, 0.6, 3)},
        objectives=["weight"],
    )
    result = opt.grid_search()
    summary = result.summary()
    assert "grid_search" in summary
    assert "总评估次数" in summary
    assert "可行解数量" in summary
    assert "最优解" in summary


def test_evaluation_point_fields():
    pt = EvaluationPoint(
        params={"height": 0.5},
        objectives={"weight": 100.0},
        constraints={"max_stress": True},
        feasible=True,
        duration_ms=12.3,
    )
    assert pt.params["height"] == 0.5
    assert pt.objectives["weight"] == 100.0
    assert pt.feasible is True
    assert pt.error == ""


# ------------------------------------- 回归：位移口径必须含跨中挠度

def _simply_supported_one_element() -> dict:
    """单跨简支梁，**只离散成一个单元**。

    两端都是支座，所以全部节点平动位移恒等于零；控制量完全在跨中。
    这正是"只看节点位移"会漏掉的情形——旧实现在这个模型上算出的
    最大位移是 0，任何位移限值都满足。
    """
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}],
        "sections": [{"name": "BEAM", "A": 0.04, "Iy": 1.3333e-4,
                      "Iz": 1.3333e-4, "J": 2.2e-4}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6.0, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "BEAM", "material": "Q355"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 0, 0, 0]}],
        "load_cases": [{"name": "DL", "member_loads": [
            {"member": 1, "w": [0, 0, -10e3]}]}],
    }


def test_nodal_displacement_alone_is_zero_on_this_model():
    """先钉住前提：这个模型的节点平动位移确实全是零。

    如果哪天模型改了、节点位移不再为零，下面那条回归测试就失去意义，
    应该由这条先失败来提醒。
    """
    import numpy as np
    from agent import Session

    session = Session()
    session.set_model(model=_simply_supported_one_element())
    assert session.solve_model().ok
    frame, sol = session.frame, session.solution
    worst_nodal = max(
        float(np.linalg.norm(res.U[frame.node_dofs(nid)[:3]]))
        for res in sol.all_results().values()
        for nid in frame.order())
    assert worst_nodal == pytest.approx(0.0, abs=1e-12)


def test_displacement_limit_uses_centerline_not_nodes():
    """位移约束必须按跨中挠度判定，不能只看节点。

    5wL⁴/(384EIz)：h=0.2 时约 6.1 mm，远超 1 mm 限值，必须判为不可行。
    旧实现只取节点位移（恒为 0），会把它判成可行。
    """
    opt = SectionOptimizer(
        model=_simply_supported_one_element(),
        section_name="BEAM", section_type="矩形",
        variables={"width": [0.2], "height": [0.2]},
        objectives=["weight"],
        constraints={"max_displacement_limit": 1e-3},   # 1 mm
    )
    result = opt.grid_search()
    point = result.evaluations[0]
    assert point.error == "", point.error
    assert point.constraints["max_displacement_limit"] is False
    assert point.feasible is False


def test_centerline_displacement_objective_is_not_zero():
    """位移目标同样不能是节点口径，否则这个模型上恒等于 0。"""
    opt = SectionOptimizer(
        model=_simply_supported_one_element(),
        section_name="BEAM", section_type="矩形",
        variables={"width": [0.2], "height": [0.2]},
        objectives=["max_displacement"],
    )
    result = opt.grid_search()
    value = result.evaluations[0].objectives["max_displacement"]
    assert value == pytest.approx(6.14e-3, rel=5e-2)


def _stress_feasibility(allowable: float) -> list[bool]:
    """在同一组截面上按给定许用应力取可行性，供下面两条测试对比。

    模型里的 COL 是直接以 A/Iy/Iz/J 给出的、没有极端纤维距离的截面；
    优化的是 BEAM。这组测试同时守住"只校核被优化截面的杆件"这条范围规则——
    如果实现去算了 COL，就会抛 StressUnavailable 而不是给出可行性。
    """
    opt = SectionOptimizer(
        model=_make_model(), section_name="BEAM", section_type="矩形",
        variables={"width": [0.2], "height": [0.2, 0.5]},
        objectives=["weight"],
        constraints={"max_stress": allowable},
    )
    result = opt.grid_search()
    for point in result.evaluations:
        assert point.error == "", point.error
    return [p.constraints["max_stress"] for p in result.evaluations]


def test_max_stress_constraint_actually_discriminates():
    """守旧实现的那个 bug：约束曾经恒为 True。

    同一组截面换一个足够严的许用值，结果必须变——否则这个约束没在干活。
    """
    loose = _stress_feasibility(1e12)      # 高到谁都满足
    tight = _stress_feasibility(1.0)       # 低到谁都不满足
    assert all(loose)
    assert not any(tight)


def test_max_stress_constraint_separates_by_section_size():
    """矮截面超应力、高截面不超：σ = 6PL/(b·h²)，随 h 单调下降。"""
    verdicts = _stress_feasibility(2.0e7)
    assert verdicts[0] is False            # h=0.2
    assert verdicts[1] is True             # h=0.5
