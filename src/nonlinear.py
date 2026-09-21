"""刚架非线性分析。

当前模块把能力边界写清楚：

* ``solve_pdelta`` 是小应变、二阶弹性 P-Δ，逐级加载并迭代更新杆件轴力；
* ``solve_material_nonlinear`` 是双线性**轴向**材料塑性，采用返回映射与 Newton
  增量迭代。弯曲塑性需要纤维截面，不能仅凭 A/I/J 可靠推断，暂不冒充支持。

两者都不是 Abaqus 的通用 ``NLGEOM`` 替代品，但具备明确增量、残差、收敛记录，
适合作为 Agent 学习“设置分析步—提交—检查收敛”的内核基础。
"""

from __future__ import annotations

from copy import copy

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from buckling import assemble_geometric
from frame3d import (BUILTIN_AMPLITUDES, Amplitude, CaseResult, Frame, LoadCase,
                     Solution, _member_forces, _condense, _member_matrices,
                     assemble, combined_case, constrained_dofs, fixed_end,
                     span_loads_of)
from scipy.sparse import coo_matrix


def _solve_constrained(K, rhs: np.ndarray, fixed: np.ndarray,
                       prescribed: np.ndarray | None = None) -> np.ndarray:
    """带 Jacobi 缩放的约束线性子问题。"""
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), fixed)
    U = np.zeros(n) if prescribed is None else prescribed.copy()
    if not free.size:
        return U
    Kff = K[free][:, free].tocsc()
    diagonal = np.abs(Kff.diagonal())
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise np.linalg.LinAlgError("非线性迭代遇到零/负切线刚度，结构可能已失稳")
    scale = 1.0 / np.sqrt(diagonal)
    D = diags(scale)
    Ks = (D @ Kff @ D).tocsc()
    try:
        lu = splu(Ks)
    except RuntimeError as exc:
        raise np.linalg.LinAlgError("非线性迭代切线刚度奇异，结构可能已失稳") from exc
    b = rhs[free].copy()
    if fixed.size and np.any(U[fixed]):
        b -= K[free][:, fixed] @ U[fixed]
    U[free] = scale * lu.solve(scale * b)
    if not np.all(np.isfinite(U)):
        raise np.linalg.LinAlgError("非线性迭代产生非有限位移")
    return U


def _scaled_case(case: LoadCase, factor: float) -> LoadCase:
    out = LoadCase(case.name)
    out.nodal_loads = {nid: tuple(factor * v for v in load)
                       for nid, load in case.nodal_loads.items()}
    out.member_loads = {mid: tuple(factor * v for v in load)
                        for mid, load in case.member_loads.items()}
    out.member_spans = {mid: [item.scaled(factor) for item in items]
                        for mid, items in case.member_spans.items()}
    # 初应变与初曲率是荷载，照比例缩放。温度、装配误差都走这里——
    # 漏掉它们的后果是**静默的**：增量循环里荷载向量变成零，位移算出来
    # 全是 0，而最终内力又是拿 original 重算的，看不出异常。
    out.member_strains = {mid: factor * float(v)
                          for mid, v in case.member_strains.items()}
    out.member_curvatures = {mid: (factor * float(k[0]), factor * float(k[1]))
                             for mid, k in case.member_curvatures.items()}
    # 给定位移是边界条件，不随荷载比例缩放。
    out.settlements = dict(case.settlements)
    return out


def _case_frame(model: Frame, case: LoadCase) -> Frame:
    probe = copy(model)
    probe.load_cases = {case.name: case}
    probe.combos = {}
    return probe


def _linear_parts(model: Frame, case_name: str):
    """弹性刚度与约束自由度。刚度与荷载工况无关，工况名只是装配的入口。"""
    return assemble(model, [case_name])[0], constrained_dofs(model)


def _pdelta_run(model: Frame, base_K, fixed, name: str, case_at,
                increments: int, max_iter: int, tolerance: float):
    """一条 P-Δ 加载路径的增量—迭代循环。

    ``case_at(t)`` 给出伪时间 t∈(0,1] 处的荷载工况。solve_pdelta 传进来的是
    「整个工况乘以 t」，solve_step 传进来的是「各工况各按自己的幅值曲线合成」——
    两者共用同一套 Newton 循环，不会出现两份各自漂移的增量代码。
    """
    final = case_at(1.0)
    U = np.zeros(model.num_dofs)
    prescribed = np.zeros(model.num_dofs)
    for nid, values in final.settlements.items():
        mask = model.supports.get(nid, (0,) * 6)
        dofs = model.node_dofs(nid)
        for k, on in enumerate(mask):
            if on:
                prescribed[dofs[k]] = values[k]
    history: list[dict] = []
    Kt = base_K
    F = np.zeros(model.num_dofs)
    free = np.setdiff1d(np.arange(model.num_dofs), fixed)
    for step in range(1, increments + 1):
        load_factor = step / increments
        scaled = case_at(load_factor)
        probe = _case_frame(model, scaled)
        _, loads, _ = assemble(probe, [scaled.name])
        F = loads[:, 0]
        converged = False
        for iteration in range(1, max_iter + 1):
            member_forces = _member_forces(probe, U, scaled)
            axial = {mid: float(force[6]) for mid, force in member_forces.items()}
            Kt = base_K + assemble_geometric(model, axial)
            candidate = _solve_constrained(Kt, F, fixed, prescribed)
            delta = np.linalg.norm(candidate[free] - U[free])
            reference = max(np.linalg.norm(candidate[free]), 1e-14)
            ratio = float(delta / reference)
            U = candidate
            if ratio <= tolerance:
                converged = True
                # 记下这一增量的位移峰值：加载路径要能看见，否则非比例
                # 加载做了什么全凭想象。推覆曲线就是它对 load_factor 的图。
                history.append({"increment": step, "iterations": iteration,
                                "load_factor": load_factor,
                                "max_displacement": float(
                                    np.max(np.abs(U)) if U.size else 0.0),
                                "relative_change": ratio})
                break
        if not converged:
            raise RuntimeError(
                f"P-Δ {name!r} 在增量 {step}/{increments} 未收敛；"
                "结构可能接近失稳，请减小荷载或增加增量")
    forces = _member_forces(model, U, final)
    reaction = np.zeros(model.num_dofs)
    reaction[fixed] = (Kt @ U - F)[fixed]
    return CaseResult(name, U, reaction, forces), history


def solve_pdelta(model: Frame, cases: list[str] | None = None,
                 increments: int = 10, max_iter: int = 40,
                 tolerance: float = 1e-7) -> Solution:
    """二阶弹性 P-Δ 分析，迭代更新初应力几何刚度。

    每个工况独立分析；非线性问题不做结果事后线性叠加。当前版本不接收组合名，
    避免把不成立的叠加结果展示成“非线性组合”。
    """
    if increments < 1 or max_iter < 1 or tolerance <= 0:
        raise ValueError("increments/max_iter 必须为正整数，tolerance 必须大于 0")
    names = cases or list(model.load_cases)
    unknown = [name for name in names if name not in model.load_cases]
    if unknown:
        raise ValueError(f"P-Δ 当前只接受基础荷载工况，不接受组合或未知名称：{unknown}")

    fixed = constrained_dofs(model)
    base_K, _, _ = assemble(model, [names[0]])
    results: dict[str, CaseResult] = {}
    convergence: dict[str, list[dict]] = {}
    for name in names:
        original = model.load_cases[name]
        results[name], convergence[name] = _pdelta_run(
            model, base_K, fixed, name,
            lambda t, c=original: _scaled_case(c, t),
            increments, max_iter, tolerance)

    return Solution(results, {}, base_K,
                    "default" if "default" in results else names[0],
                    {"type": "pdelta", "increments": increments,
                     "tolerance": tolerance, "convergence": convergence,
                     "scope": "second_order_elastic_small_strain"})


#: 分析步结果的默认名字。
STEP_CASE = "STEP"


def solve_step(model: Frame, loads: dict[str, str], name: str = STEP_CASE,
               increments: int = 10, max_iter: int = 40,
               tolerance: float = 1e-7) -> Solution:
    """一个 P-Δ 分析步：几个工况**同时**施加，各按自己的幅值曲线随伪时间变化。

    ``loads`` 把工况名映射到幅值曲线名，例如
    ``{"DL": "STEP", "WX": "RAMP"}``——重力全程为 1，侧力 0→1 线性上升。
    曲线取自 ``model.amplitudes``，内置 RAMP（0→1 斜坡）与 STEP（恒为 1）两条。

    **弹性分析里这不改变终点。** 二阶弹性问题的收敛解是同一个方程的根，与
    加载路径无关：重力先加满和按比例同加，t=1 处的位移逐位相同。别把它当成
    "更准的算法"。

    它改变的是**路径**，而路径正是推覆分析要的东西：重力全程为 1 时轴力恒定，
    切线刚度恒定，顶点位移对侧力是一条直线——这才是弹性能力曲线。按比例同加
    得到的那条"先硬后软"的曲线不是任何真实工况的响应，只是数值加载过程的副产品。
    路径记在 ``analysis["convergence"]`` 里，每个增量带 ``max_displacement``。

    路径还决定两件事：沿途某个中间状态若接近失稳，这里会不收敛而终点法看不见；
    以及一旦换到路径相关的材料非线性，终点本身就会随路径变。

    这不是结果的事后叠加：每个增量都先把各工况按当前系数合成一个真实工况，
    再对合成工况做 Newton 迭代，几何刚度用的是合成状态下的轴力。

    **副作用**：t=1 的合成工况会以 ``name`` 写进 ``model.load_cases``，
    这样平衡校核与内力恢复按结果名回查时能找到它。
    """
    if increments < 1 or max_iter < 1 or tolerance <= 0:
        raise ValueError("increments/max_iter 必须为正整数，tolerance 必须大于 0")
    if not loads:
        raise ValueError("分析步至少要施加一个荷载工况")
    unknown = [c for c in loads if c not in model.load_cases]
    if unknown:
        raise ValueError(f"分析步引用了未定义的荷载工况：{unknown}")
    curves = {}
    for case_name, amp_name in loads.items():
        curve = model.amplitudes.get(amp_name)
        if curve is None and amp_name in BUILTIN_AMPLITUDES:
            curve = Amplitude(amp_name, BUILTIN_AMPLITUDES[amp_name])
        if curve is None:
            raise ValueError(
                f"工况 {case_name!r} 引用了未定义的幅值曲线 {amp_name!r}；"
                f"已定义的有 {sorted(model.amplitudes) or '（无）'}，"
                f"内置的有 {sorted(BUILTIN_AMPLITUDES)}")
        curves[case_name] = curve

    def case_at(time: float) -> LoadCase:
        merged = combined_case(model, {c: k.at(time) for c, k in curves.items()})
        merged.name = name
        return merged

    # 把 t=1 的合成工况登记进模型。**不是可有可无的**：平衡校核、内力恢复、
    # 内力图和强度验算都按结果名回查荷载工况，查不到就是 KeyError。分析步的
    # 荷载本来就该是一个实在的工况，Abaqus 里也是这样物化的。
    model.load_cases[name] = case_at(1.0)
    base_K, fixed = _linear_parts(model, next(iter(loads)))
    result, history = _pdelta_run(model, base_K, fixed, name, case_at,
                                  increments, max_iter, tolerance)
    return Solution({name: result}, {}, base_K, name,
                    {"type": "pdelta_step", "increments": increments,
                     "tolerance": tolerance, "convergence": {name: history},
                     "amplitudes": {c: k.name for c, k in curves.items()},
                     "scope": "second_order_elastic_small_strain"})


def _bilinear_update(strain: float, material, committed: tuple[float, float]
                     ) -> tuple[float, float, tuple[float, float]]:
    """一维双线性等向强化返回映射，返回 (stress, tangent, trial_state)。"""
    if material.yield_stress is None:
        return material.E * strain, material.E, committed
    plastic_strain, alpha = committed
    ratio = float(material.hardening_ratio)
    H = material.E * ratio / (1.0 - ratio) if ratio > 0.0 else 0.0
    trial = material.E * (strain - plastic_strain)
    residual = abs(trial) - (material.yield_stress + H * alpha)
    if residual <= 0.0:
        return trial, material.E, committed
    direction = 1.0 if trial >= 0.0 else -1.0
    gamma = residual / (material.E + H)
    stress = trial - material.E * gamma * direction
    state = (plastic_strain + gamma * direction, alpha + gamma)
    tangent = material.E * H / (material.E + H) if H > 0.0 else 0.0
    return stress, tangent, state


def _material_system(model: Frame, U: np.ndarray, case: LoadCase,
                     committed: dict[int, tuple[float, float]]):
    """组装材料非线性的切线刚度、内力与试算状态。"""
    n = model.num_dofs
    rows, cols, values = [], [], []
    internal = np.zeros(n)
    trial_states = {}
    member_forces = {}
    for mid, member in model.members.items():
        if "ux" in member.releases_i + member.releases_j:
            raise ValueError("材料非线性暂不支持杆端轴向 ux 释放")
        L, R, T, B, k_elastic, _, cond = _member_matrices(model, member)
        dofs = model.node_dofs(member.i) + model.node_dofs(member.j)
        q = T @ B @ U[dofs]
        p = np.zeros(12)
        for item in span_loads_of(case, mid):
            p += fixed_end(item, L, R)
        if cond is not None:
            kept, released = list(cond.kept), list(cond.released)
            q[released] = cond.inv_rr @ (p[released] - cond.k_rk @ q[kept])

        material = model.materials[member.material]
        section = model.sections[member.section]
        strain = float((q[6] - q[0]) / L)
        stress, tangent, trial_state = _bilinear_update(
            strain, material, committed[mid])
        trial_states[mid] = trial_state
        axial = stress * section.A

        force = k_elastic @ q - p
        force[0], force[6] = -axial, axial
        # 释放端的广义力必须严格为零。
        if cond is not None:
            force[list(cond.released)] = 0.0
        member_forces[mid] = force.copy()
        internal[dofs] += B.T @ T.T @ force

        kt = k_elastic.copy()
        axial_k = tangent * section.A / L
        kt[0, 0] = kt[6, 6] = axial_k
        kt[0, 6] = kt[6, 0] = -axial_k
        kt, _ = _condense(kt, member.released_indices(), member.id)
        kg = B.T @ T.T @ kt @ T @ B
        for a in range(12):
            for b in range(12):
                rows.append(dofs[a]); cols.append(dofs[b]); values.append(kg[a, b])
    tangent = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    return tangent, internal, trial_states, member_forces


def solve_material_nonlinear(model: Frame, cases: list[str] | None = None,
                             increments: int = 20, max_iter: int = 40,
                             tolerance: float = 1e-8) -> Solution:
    """双线性轴向材料非线性静力分析。

    采用增量荷载、Newton-Raphson 与一维返回映射。弯曲/扭转仍保持弹性；若要
    梁截面塑性铰或渐进屈服，必须再引入纤维截面，不能从 A/I/J 猜测。
    """
    if increments < 1 or max_iter < 1 or tolerance <= 0:
        raise ValueError("increments/max_iter 必须为正整数，tolerance 必须大于 0")
    names = cases or list(model.load_cases)
    unknown = [name for name in names if name not in model.load_cases]
    if unknown:
        raise ValueError(f"材料非线性当前只接受基础荷载工况：{unknown}")
    fixed = constrained_dofs(model)
    results = {}
    all_history = {}
    last_tangent, _, _ = assemble(model, [names[0]])

    for name in names:
        original = model.load_cases[name]
        U = np.zeros(model.num_dofs)
        for nid, values in original.settlements.items():
            mask = model.supports.get(nid, (0,) * 6)
            dofs = model.node_dofs(nid)
            for k, on in enumerate(mask):
                if on:
                    U[dofs[k]] = values[k]
        committed = {mid: (0.0, 0.0) for mid in model.members}
        history = []
        forces = {}
        external_nodal = np.zeros(model.num_dofs)
        internal = np.zeros(model.num_dofs)
        for step in range(1, increments + 1):
            scaled = _scaled_case(original, step / increments)
            external_nodal[:] = 0.0
            for nid, load in scaled.nodal_loads.items():
                external_nodal[model.node_dofs(nid)] += np.asarray(load)
            converged = False
            for iteration in range(1, max_iter + 1):
                last_tangent, internal, trial, forces = _material_system(
                    model, U, scaled, committed)
                residual = external_nodal - internal
                free = np.setdiff1d(np.arange(model.num_dofs), fixed)
                norm = float(np.linalg.norm(residual[free]))
                reference = max(float(np.linalg.norm(external_nodal[free])),
                                float(np.linalg.norm(internal[free])), 1.0)
                ratio = norm / reference
                if ratio <= tolerance:
                    committed = trial
                    history.append({"increment": step, "iterations": iteration,
                                    "load_factor": step / increments,
                                    "relative_residual": ratio})
                    converged = True
                    break
                correction = _solve_constrained(last_tangent, residual, fixed)
                U += correction
            if not converged:
                raise RuntimeError(
                    f"材料非线性工况 {name!r} 在增量 {step}/{increments} 未收敛；"
                    "可增加增量，或检查是否达到理想塑性极限")
        reaction = np.zeros(model.num_dofs)
        reaction[fixed] = (internal - external_nodal)[fixed]
        results[name] = CaseResult(name, U.copy(), reaction, forces)
        all_history[name] = history

    return Solution(results, {}, last_tangent,
                    "default" if "default" in results else names[0],
                    {"type": "material_nonlinear_axial_bilinear",
                     "increments": increments, "tolerance": tolerance,
                     "convergence": all_history,
                     "scope": "small_strain_axial_plasticity_bending_elastic"})
