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

from section_optimizer import SectionOptimizer, EvaluationPoint, OptimizationResult


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
