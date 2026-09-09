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
from frame3d import (CaseResult, Frame, LoadCase, Solution, _member_forces,
                     _condense, _member_matrices, assemble, constrained_dofs,
                     fixed_end, span_loads_of)
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
    # 给定位移是边界条件，不随荷载比例缩放。
    out.settlements = dict(case.settlements)
    return out


def _case_frame(model: Frame, case: LoadCase) -> Frame:
    probe = copy(model)
    probe.load_cases = {case.name: case}
    probe.combos = {}
    return probe


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
        U = np.zeros(model.num_dofs)
        prescribed = np.zeros(model.num_dofs)
        for nid, values in original.settlements.items():
            mask = model.supports.get(nid, (0,) * 6)
            dofs = model.node_dofs(nid)
            for k, on in enumerate(mask):
                if on:
                    prescribed[dofs[k]] = values[k]
        history = []
        Kt = base_K
        scaled = original
        F = np.zeros(model.num_dofs)
        for step in range(1, increments + 1):
            load_factor = step / increments
            scaled = _scaled_case(original, load_factor)
            probe = _case_frame(model, scaled)
            _, loads, _ = assemble(probe, [name])
            F = loads[:, 0]
            converged = False
            for iteration in range(1, max_iter + 1):
                member_forces = _member_forces(probe, U, scaled)
                axial = {mid: float(force[6]) for mid, force in member_forces.items()}
                Kg = assemble_geometric(model, axial)
                Kt = base_K + Kg
                candidate = _solve_constrained(Kt, F, fixed, prescribed)
                free = np.setdiff1d(np.arange(model.num_dofs), fixed)
                delta = np.linalg.norm(candidate[free] - U[free])
                reference = max(np.linalg.norm(candidate[free]), 1e-14)
                ratio = float(delta / reference)
                U = candidate
                if ratio <= tolerance:
                    converged = True
                    history.append({"increment": step, "iterations": iteration,
                                    "load_factor": load_factor, "relative_change": ratio})
                    break
            if not converged:
                raise RuntimeError(
                    f"P-Δ 工况 {name!r} 在增量 {step}/{increments} 未收敛；"
                    "结构可能接近失稳，请减小荷载或增加增量")
        forces = _member_forces(model, U, original)
        reaction = np.zeros(model.num_dofs)
        reaction[fixed] = (Kt @ U - F)[fixed]
        results[name] = CaseResult(name, U, reaction, forces)
        convergence[name] = history

    return Solution(results, {}, base_K,
                    "default" if "default" in results else names[0],
                    {"type": "pdelta", "increments": increments,
                     "tolerance": tolerance, "convergence": convergence,
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
