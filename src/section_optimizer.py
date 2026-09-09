"""多参数多目标截面优化。

从"单参数截面扫描"升级为"多参数多目标优化"：同时变化多个截面参数
（如工字形的高度、翼缘宽度、腹板厚度、翼缘厚度），同时优化多个目标
（如最小重量、最小位移），并满足约束（如应力不超过许用值、位移不超限）。

**支持的算法**：
1. `grid_search` — 网格搜索。在每个参数的取值范围内均匀取点，穷举所有组合。
   优点：结果确定、可复现、能找到全局最优；缺点：参数多时组合爆炸。
   适合：2-3 个参数，每个 5-10 个取值。
2. `random_search` — 随机搜索。在参数范围内随机采样指定次数。
   优点：参数多时比网格搜索高效，能探索更大空间；缺点：结果不确定。
   适合：4+ 个参数，或需要快速得到近似最优。

**目标函数**（可多选，多目标时返回 Pareto 前沿）：
- `weight` — 结构总重量（最小化）
- `max_displacement` — 最大合位移（最小化）
- `material_cost` — 材料成本（最小化，重量 × 单价）

**约束**（可多选，不满足的解被排除）：
- `max_displacement_limit` — 最大位移不超过限值
- `min_section_area` — 截面面积不小于最小值

**用法**：
```python
from section_optimizer import SectionOptimizer

opt = SectionOptimizer(
    model=my_model,                    # 已定义的模型 dict（截面名会被替换）
    section_name="BEAM",               # 要优化的截面名
    section_type="工字形 / H 型钢",     # 截面类型
    variables={
        "height": (0.3, 0.8, 5),       # (min, max, num_points) 或取值列表
        "flange_width": (0.15, 0.3, 4),
    },
    objectives=["weight", "max_displacement"],
    constraints={"max_displacement_limit": 0.02},   # m
)
result = opt.grid_search()
print(result.best)          # 最优解（单目标时）
print(result.pareto)        # Pareto 前沿（多目标时）
print(result.evaluations)   # 所有评估点
```
"""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from agent import Session
from internal_forces import max_centerline_displacement
import sections as sec


# 中心线挠度的采样密度。用 21 个点做两次梯形积分，跨中挠度约差 0.4%；
# 优化要拿这个数直接和限值比大小，采样不足会系统性地低估。
_DISPLACEMENT_STATIONS = 101


def _worst_centerline_displacement(frame, sol) -> float:
    """全工况下杆件中心线的最大合位移。

    **不能只取节点位移**：一根按单个单元离散的梁，两端节点位移可以很小，
    跨中挠度却是控制量。查询、绘图和金标准 8~10 都用中心线恢复，
    优化器必须用同一把尺子，否则会把一个偏小的截面判成满足位移限值。
    """
    worst = 0.0
    for name in sol.all_results():
        got = max_centerline_displacement(frame, sol, name,
                                          stations=_DISPLACEMENT_STATIONS)
        worst = max(worst, float(got["value"]))
    return worst


@dataclass
class EvaluationPoint:
    """一个评估点：一组参数 + 求解结果 + 目标值 + 约束满足情况。"""
    params: dict[str, float]
    objectives: dict[str, float]
    constraints: dict[str, bool]
    feasible: bool
    error: str = ""           # 求解失败时的错误信息
    duration_ms: float = 0.0


@dataclass
class OptimizationResult:
    """优化结果。"""
    algorithm: str
    best: EvaluationPoint | None              # 单目标时的最优解
    pareto: list[EvaluationPoint]             # 多目标时的 Pareto 前沿
    evaluations: list[EvaluationPoint]        # 所有评估点
    total_calls: int
    feasible_count: int
    duration_ms: float

    def summary(self) -> str:
        lines = [
            f"优化算法: {self.algorithm}",
            f"总评估次数: {self.total_calls}",
            f"可行解数量: {self.feasible_count}/{self.total_calls}",
            f"耗时: {self.duration_ms:.0f} ms",
        ]
        if self.best:
            lines.append(f"最优解参数: {self.best.params}")
            lines.append(f"最优解目标: {self.best.objectives}")
        if self.pareto:
            lines.append(f"Pareto 前沿点数: {len(self.pareto)}")
        return "\n".join(lines)


class SectionOptimizer:
    """多参数多目标截面优化器。

    对指定截面的参数进行搜索，每次评估时：
    1. 用当前参数构建新截面
    2. 替换模型中的对应截面
    3. 求解
    4. 计算目标值和约束满足情况
    """

    SUPPORTED_OBJECTIVES = ("weight", "max_displacement", "material_cost")
    SUPPORTED_CONSTRAINTS = ("max_displacement_limit", "min_section_area")

    def __init__(
        self,
        model: dict[str, Any],
        section_name: str,
        section_type: str,
        variables: dict[str, tuple[float, float, int] | list[float]],
        objectives: list[str] | None = None,
        constraints: dict[str, float] | None = None,
        material_density: float = 7850.0,
        material_unit_cost: float = 4.5,       # 元/kg，用于 material_cost
        seed: int = 42,
    ):
        if section_type not in sec.BUILDERS:
            raise ValueError(f"不支持的截面类型 {section_type!r}，"
                             f"支持: {list(sec.BUILDERS.keys())}")
        self._model = model
        self._section_name = section_name
        self._section_type = section_type
        self._builder, self._param_names, self._param_labels = sec.BUILDERS[section_type]
        self._variables = self._normalize_variables(variables)
        self._objectives = objectives or ["weight"]
        self._constraints = constraints or {}
        self._density = material_density
        self._unit_cost = material_unit_cost
        self._rng = random.Random(seed)

        # 校验目标和约束名
        for obj in self._objectives:
            if obj not in self.SUPPORTED_OBJECTIVES:
                raise ValueError(f"不支持的目标 {obj!r}，支持: {self.SUPPORTED_OBJECTIVES}")
        for con in self._constraints:
            if con == "max_stress":
                raise NotImplementedError(
                    "max_stress 约束尚未实现：截面只存 A/Iy/Iz/J，没有截面模量，"
                    "算不出弯曲应力。此前的实现恒为满足，会把超应力的截面判成可行，"
                    "所以现在明确拒绝，而不是给出一个假结论。")
            if con not in self.SUPPORTED_CONSTRAINTS:
                raise ValueError(f"不支持的约束 {con!r}，支持: {self.SUPPORTED_CONSTRAINTS}")

    def _normalize_variables(
        self, variables: dict[str, tuple[float, float, int] | list[float]]
    ) -> dict[str, list[float]]:
        """把变量定义统一成 {param_name: [value1, value2, ...]}。"""
        normalized = {}
        for name, spec in variables.items():
            if name not in self._param_names:
                raise ValueError(f"截面 {self._section_type!r} 没有参数 {name!r}，"
                                 f"可用参数: {self._param_names}")
            if isinstance(spec, (list, tuple)) and len(spec) == 3 and isinstance(spec[2], int):
                lo, hi, n = spec
                normalized[name] = list(np.linspace(lo, hi, n))
            else:
                normalized[name] = list(spec)
        return normalized

    def _build_section(self, params: dict[str, float]) -> dict[str, Any]:
        """用当前参数构建截面 dict。未指定的参数用默认值。"""
        defaults = dict(zip(self._param_names, sec.DEFAULT_DIMENSIONS[self._section_type]))
        merged = {**defaults, **params}
        args = [merged[name] for name in self._param_names]
        return self._builder(self._section_name, *args)

    def _evaluate(self, params: dict[str, float]) -> EvaluationPoint:
        """评估一组参数：构建截面 → 替换模型 → 求解 → 计算目标和约束。"""
        start = time.monotonic()
        try:
            new_section = self._build_section(params)
            # 复制模型并替换对应截面
            model_copy = _deep_copy_model(self._model)
            for i, s in enumerate(model_copy.get("sections", [])):
                if s.get("name") == self._section_name:
                    model_copy["sections"][i] = new_section
                    break
            else:
                # 没找到同名截面，添加一个
                model_copy.setdefault("sections", []).append(new_section)

            session = Session()
            session.set_model(model=model_copy)
            result = session.solve_model()
            if not result.ok:
                return EvaluationPoint(
                    params=params, objectives={}, constraints={}, feasible=False,
                    error=f"求解失败: {result.payload.get('error', '未知错误')}",
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            # 计算目标
            objectives = self._compute_objectives(session, new_section)
            # 计算约束
            constraints = self._compute_constraints(session, new_section)
            feasible = all(constraints.values())

            return EvaluationPoint(
                params=params, objectives=objectives, constraints=constraints,
                feasible=feasible, duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            return EvaluationPoint(
                params=params, objectives={}, constraints={}, feasible=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

    def _compute_objectives(self, session: Session, section: dict[str, Any]) -> dict[str, float]:
        """计算目标函数值。"""
        out = {}
        frame = session.frame
        sol = session.solution

        if "weight" in self._objectives or "material_cost" in self._objectives:
            total_volume = 0.0
            for mid, m in frame.members.items():
                sec_name = m.section
                # 正在优化的截面用新构建的面积，其他截面用原始模型中的面积
                if sec_name == self._section_name:
                    area = section.get("A", 0.0)
                else:
                    area = 0.0
                    for s in self._model.get("sections", []):
                        if s.get("name") == sec_name:
                            area = s.get("A", area)
                            break
                length = _member_length(frame, m)
                total_volume += area * length
            weight = total_volume * self._density
            if "weight" in self._objectives:
                out["weight"] = weight
            if "material_cost" in self._objectives:
                out["material_cost"] = weight * self._unit_cost

        if "max_displacement" in self._objectives:
            out["max_displacement"] = _worst_centerline_displacement(frame, sol)

        return out

    def _compute_constraints(self, session: Session, section: dict[str, Any]) -> dict[str, bool]:
        """计算约束满足情况。"""
        out = {}
        frame = session.frame
        sol = session.solution

        if "max_displacement_limit" in self._constraints:
            limit = self._constraints["max_displacement_limit"]
            out["max_displacement_limit"] = (
                _worst_centerline_displacement(frame, sol) <= limit)

        if "min_section_area" in self._constraints:
            min_area = self._constraints["min_section_area"]
            out["min_section_area"] = section.get("A", 0) >= min_area

        return out

    # --------------------------------------------------------- 优化算法

    def grid_search(self) -> OptimizationResult:
        """网格搜索：穷举所有参数组合。"""
        start = time.monotonic()
        param_names = list(self._variables.keys())
        value_lists = [self._variables[name] for name in param_names]
        combinations = list(itertools.product(*value_lists))

        evaluations = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            evaluations.append(self._evaluate(params))

        return self._make_result("grid_search", evaluations, start)

    def random_search(self, n_samples: int = 100) -> OptimizationResult:
        """随机搜索：在参数范围内随机采样 n_samples 次。"""
        start = time.monotonic()
        param_names = list(self._variables.keys())
        evaluations = []
        for _ in range(n_samples):
            params = {}
            for name in param_names:
                values = self._variables[name]
                params[name] = self._rng.uniform(min(values), max(values))
            evaluations.append(self._evaluate(params))
        return self._make_result("random_search", evaluations, start)

    def optimize(self, algorithm: str = "grid_search", **kwargs) -> OptimizationResult:
        """统一入口。algorithm: 'grid_search' 或 'random_search'。"""
        if algorithm == "grid_search":
            return self.grid_search()
        elif algorithm == "random_search":
            return self.random_search(**kwargs)
        else:
            raise ValueError(f"不支持的算法 {algorithm!r}，支持: grid_search, random_search")

    # --------------------------------------------------------- 结果整理

    def _make_result(self, algorithm: str, evaluations: list[EvaluationPoint],
                     start: float) -> OptimizationResult:
        feasible = [e for e in evaluations if e.feasible]
        best = None
        pareto = []

        if feasible:
            if len(self._objectives) == 1:
                # 单目标：取目标值最小的
                obj = self._objectives[0]
                best = min(feasible, key=lambda e: e.objectives.get(obj, float("inf")))
            else:
                # 多目标：计算 Pareto 前沿
                pareto = _pareto_front(feasible, self._objectives)
                if pareto:
                    best = pareto[0]  # 取第一个作为代表

        return OptimizationResult(
            algorithm=algorithm,
            best=best,
            pareto=pareto,
            evaluations=evaluations,
            total_calls=len(evaluations),
            feasible_count=len(feasible),
            duration_ms=(time.monotonic() - start) * 1000,
        )


# --------------------------------------------------------- 辅助函数

def _deep_copy_model(model: dict[str, Any]) -> dict[str, Any]:
    """深拷贝模型。用 json 序列化/反序列化，避免嵌套 dict 的引用问题。"""
    import json
    return json.loads(json.dumps(model))


def _member_length(frame, member) -> float:
    """计算杆件长度。member 是 frame3d.Member 对象。"""
    ni = frame.nodes[member.i]
    nj = frame.nodes[member.j]
    return float(np.sqrt(
        (nj.x - ni.x) ** 2 +
        (nj.y - ni.y) ** 2 +
        (nj.z - ni.z) ** 2
    ))


def _pareto_front(points: list[EvaluationPoint], objectives: list[str]) -> list[EvaluationPoint]:
    """计算 Pareto 前沿（最小化所有目标）。

    一个点是 Pareto 最优的，当且仅当不存在另一个点在所有目标上都不劣于它，
    且至少在一个目标上严格优于它。
    """
    front = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            # q 在所有目标上都不劣于 p
            all_better_or_equal = all(
                q.objectives.get(obj, float("inf")) <= p.objectives.get(obj, float("inf"))
                for obj in objectives
            )
            # q 至少在一个目标上严格优于 p
            any_strictly_better = any(
                q.objectives.get(obj, float("inf")) < p.objectives.get(obj, float("inf"))
                for obj in objectives
            )
            if all_better_or_equal and any_strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    # 按第一个目标排序
    if front and objectives:
        front.sort(key=lambda e: e.objectives.get(objectives[0], 0))
    return front
