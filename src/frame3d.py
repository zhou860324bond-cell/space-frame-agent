"""空间刚架梁柱求解器（3D Timoshenko / Euler-Bernoulli 梁单元）。

约定
----
* 每节点 6 个自由度，顺序 (ux, uy, uz, rx, ry, rz)。
* 单元局部坐标：局部 x 沿 i->j；局部 y 由参考向量投影得到（水平杆件时指向全局 +Z，
  即"向上"），局部 z = x × y。竖直杆件参考向量退化，改用全局 +X —— 这是空间刚架
  最经典的 bug 来源，见 `local_axes` 与对应测试。
* 杆端释放通过静力凝聚实现：被释放的局部自由度不传力，见 `_condense`。
* 多工况共用一次矩阵分解；荷载组合按线性叠加由工况结果合成。
* 单位制：N-m-Pa 或 N-mm-MPa 均可，全程不做换算，由调用方保证一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from span_loads import (POINT, TRAPEZOID, UNIFORM, SpanLoad, fixed_end,
                        moment_about_origin, span_force)
from units import MM as UNITS_MM  # noqa: F401  （对外转出，便于 from frame3d import）
from units import SI as UNITS_SI
from units import system as unit_system

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu

DOF_PER_NODE = 6
DEFAULT_CASE = "default"
LOCAL_DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")
_VERTICAL_TOL = 1e-6
_RELEASE_COND_LIMIT = 1e12
_SINGULAR_PIVOT_RATIO = 1e-12


# --------------------------------------------------------------------------- 模型

@dataclass(frozen=True)
class Section:
    """截面特性。

    ``Ay/Az`` 是局部 y/z 方向的有效剪切面积。留空时保持经典
    Euler-Bernoulli 梁；给出正值时对应弯曲平面启用 Timoshenko 剪切变形。
    用有效面积而不是再存一个剪切修正系数，可以直接容纳箱形、工字形等截面。
    """
    name: str
    A: float
    Iy: float
    Iz: float
    J: float
    Ay: float | None = None
    Az: float | None = None
    # 极端纤维距离与圆形标志，仅用于正应力计算；由 sections.py 的 builder 给出。
    # 直接以 A/Iy/Iz/J 定义的截面留空，此时正应力被拒绝而不是估算。
    cy: float | None = None
    cz: float | None = None
    circular: bool = False


@dataclass(frozen=True)
class Material:
    name: str
    E: float
    nu: float
    density: float = 0.0          # kg/m³，只有算自重时才用得上
    yield_stress: float | None = None  # 双线性轴向屈服应力；留空为线弹性
    hardening_ratio: float = 0.01     # 屈服后切线模量 / E
    # 线膨胀系数 1/℃。只有算温度应力时才用得上；留 0 表示不考虑温度。
    alpha: float = 0.0

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float
    z: float

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class Member:
    """一根杆件。

    releases_i / releases_j 是该端**不传力**的局部自由度名（取自 LOCAL_DOF_NAMES）。
    最常见的是梁端铰接，即释放绕局部 z 的弯矩：releases_j=("rz",)。
    """
    id: int
    i: int
    j: int
    section: str
    material: str
    ref_vector: tuple[float, float, float] | None = None
    releases_i: tuple[str, ...] = ()
    releases_j: tuple[str, ...] = ()
    # 节点到可变形梁端的刚域偏移，使用全局坐标。零值保持旧模型行为。
    offset_i: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_j: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def released_indices(self) -> tuple[int, ...]:
        idx: set[int] = set()
        for name in self.releases_i:
            idx.add(_dof_index(name, self.id))
        for name in self.releases_j:
            idx.add(6 + _dof_index(name, self.id))
        return tuple(sorted(idx))


def _dof_index(name: str, member_id: int) -> int:
    try:
        return LOCAL_DOF_NAMES.index(name)
    except ValueError:
        raise ValueError(
            f"杆件 {member_id} 的释放自由度 {name!r} 无效，"
            f"应取自 {LOCAL_DOF_NAMES}"
        ) from None


@dataclass
class LoadCase:
    """一个荷载工况。

    member_loads 是满跨均布线荷载 (wx, wy, wz)，全局坐标，写起来最省事；
    member_spans 放梯形与集中力这类不满跨的荷载。两者可以同时存在，
    同一根杆件上的各项**直接叠加**——线性问题里这是精确的。

    下游只经 `span_loads_of()` 一个入口取用，不会有两条计算路径。
    """
    name: str
    nodal_loads: dict[int, tuple[float, float, float, float, float, float]] = field(default_factory=dict)
    member_loads: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    member_spans: dict[int, list[SpanLoad]] = field(default_factory=dict)
    # 支座沉降：给定位移，不是荷载。只有被约束住的那个方向才有意义
    settlements: dict[int, tuple[float, float, float, float, float, float]] = field(default_factory=dict)
    # 初应变 ε₀（无量纲），按杆件。装配误差 Δl/L 与温度 α·ΔT 都归到这里——
    # 讲义把温度应力"转化为装配内力问题"，本质就是同一个初应变。
    # 存应变而不是存 Δl 或 ΔT：应变沿杆是常量，杆件被自动剖分时各段直接继承，
    # 不需要按段长重新分配。
    member_strains: dict[int, float] = field(default_factory=dict)


def span_loads_of(case: LoadCase, member_id: int) -> list[SpanLoad]:
    """一根杆件在该工况下的全部杆间荷载，均布也归一成同一种表示。"""
    items: list[SpanLoad] = []
    w = case.member_loads.get(member_id)
    if w is not None:
        items.append(SpanLoad(UNIFORM, tuple(float(v) for v in w)))
    items.extend(case.member_spans.get(member_id, ()))
    return items


def _default_cases() -> dict[str, LoadCase]:
    return {DEFAULT_CASE: LoadCase(DEFAULT_CASE)}


@dataclass
class Frame:
    nodes: dict[int, Node] = field(default_factory=dict)
    members: dict[int, Member] = field(default_factory=dict)
    sections: dict[str, Section] = field(default_factory=dict)
    materials: dict[str, Material] = field(default_factory=dict)
    supports: dict[int, tuple[int, int, int, int, int, int]] = field(default_factory=dict)
    load_cases: dict[str, LoadCase] = field(default_factory=_default_cases)
    combos: dict[str, dict[str, float]] = field(default_factory=dict)
    # 单位制只影响自重的 g 与结果的显示换算，不影响刚度方程本身
    units: str = UNITS_SI

    @property
    def unit_system(self):
        return unit_system(self.units)

    # --- 荷载工况 ---
    def case(self, name: str = DEFAULT_CASE) -> LoadCase:
        """取一个工况，不存在就新建。"""
        if name not in self.load_cases:
            self.load_cases[name] = LoadCase(name)
        return self.load_cases[name]

    @property
    def nodal_loads(self):
        """默认工况的节点荷载。单工况模型直接用这个即可。"""
        return self.case().nodal_loads

    @property
    def member_loads(self):
        """默认工况的杆间荷载。"""
        return self.case().member_loads

    @property
    def member_spans(self):
        """默认工况的梯形与集中杆间荷载。"""
        return self.case().member_spans

    @property
    def settlements(self):
        """默认工况的支座沉降。"""
        return self.case().settlements

    # --- 自由度映射 ---
    def order(self) -> list[int]:
        return sorted(self.nodes)

    def index_of(self) -> dict[int, int]:
        return {nid: k for k, nid in enumerate(self.order())}

    @property
    def num_dofs(self) -> int:
        return DOF_PER_NODE * len(self.nodes)

    def node_dofs(self, nid: int) -> list[int]:
        k = self.index_of()[nid]
        return list(range(DOF_PER_NODE * k, DOF_PER_NODE * (k + 1)))


# --------------------------------------------------------------------------- 单元

def local_axes(pi: np.ndarray, pj: np.ndarray,
               ref: tuple[float, float, float] | None = None) -> tuple[float, np.ndarray]:
    """返回 (杆长, 3x3 方向余弦阵)，阵的行依次是局部 x、y、z 的全局分量。"""
    d = pj - pi
    L = float(np.linalg.norm(d))
    if L <= 0.0:
        raise ValueError("零长度杆件")
    lx = d / L

    if ref is not None:
        r = np.asarray(ref, dtype=float)
    else:
        r = np.array([0.0, 0.0, 1.0])          # 默认局部 y 朝上
        if abs(abs(float(lx @ r)) - 1.0) < _VERTICAL_TOL:
            r = np.array([1.0, 0.0, 0.0])      # 竖直杆件：参考向量退化，改用全局 X

    ly = r - float(r @ lx) * lx
    n = float(np.linalg.norm(ly))
    if n < 1e-12:
        raise ValueError(f"参考向量与杆轴平行，无法确定截面主轴（杆长 {L}）")
    ly /= n
    lz = np.cross(lx, ly)
    return L, np.vstack([lx, ly, lz])


def local_stiffness(L: float, E: float, G: float, sec: Section) -> np.ndarray:
    """12x12 局部刚度矩阵。

    ``Ay/Az`` 缺省时退化为原有 Euler-Bernoulli 单元；给出后采用两节点
    Timoshenko 梁的闭式刚度。这样旧模型数值不变，深梁/短梁才显式承担剪切柔度。
    """
    A, Iy, Iz, J = sec.A, sec.Iy, sec.Iz, sec.J
    k = np.zeros((12, 12))

    k[0, 0] = k[6, 6] = E * A / L
    k[0, 6] = -E * A / L

    k[3, 3] = k[9, 9] = G * J / L
    k[3, 9] = -G * J / L

    # 局部 x-y 平面弯曲（绕 z），涉及 uy(1,7) 与 rz(5,11)
    phi_y = 0.0 if sec.Ay is None else 12.0 * E * Iz / (G * sec.Ay * L**2)
    a = 12.0 * E * Iz / ((1.0 + phi_y) * L**3)
    b = 6.0 * E * Iz / ((1.0 + phi_y) * L**2)
    k[1, 1] = k[7, 7] = a
    k[1, 7] = -a
    k[1, 5] = k[1, 11] = b
    k[5, 7] = k[7, 11] = -b
    k[5, 5] = k[11, 11] = (4.0 + phi_y) * E * Iz / ((1.0 + phi_y) * L)
    k[5, 11] = (2.0 - phi_y) * E * Iz / ((1.0 + phi_y) * L)

    # 局部 x-z 平面弯曲（绕 y），涉及 uz(2,8) 与 ry(4,10)
    phi_z = 0.0 if sec.Az is None else 12.0 * E * Iy / (G * sec.Az * L**2)
    c = 12.0 * E * Iy / ((1.0 + phi_z) * L**3)
    d = 6.0 * E * Iy / ((1.0 + phi_z) * L**2)
    k[2, 2] = k[8, 8] = c
    k[2, 8] = -c
    k[2, 4] = k[2, 10] = -d
    k[4, 8] = k[8, 10] = d
    k[4, 4] = k[10, 10] = (4.0 + phi_z) * E * Iy / ((1.0 + phi_z) * L)
    k[4, 10] = (2.0 - phi_z) * E * Iy / ((1.0 + phi_z) * L)

    return k + np.triu(k, 1).T


def transformation(R: np.ndarray) -> np.ndarray:
    """由 3x3 方向余弦阵拼出 12x12 变换阵。"""
    T = np.zeros((12, 12))
    for b in range(4):
        T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
    return T


def _skew(vector: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rigid_offset_transform(member: Member) -> np.ndarray:
    """节点自由度到可变形梁端自由度的 12×12 刚臂变换阵。

    梁端平移为 ``u_end = u_node + theta × offset``。转角保持相同，因而
    刚域会正确传递偏心弯矩，而不是仅把显示几何挪一下。
    """
    B = np.eye(12)
    B[0:3, 3:6] = -_skew(member.offset_i)
    B[6:9, 9:12] = -_skew(member.offset_j)
    return B


def member_endpoints(model: Frame, member: Member) -> tuple[np.ndarray, np.ndarray]:
    """返回可变形梁段的两个端点（节点坐标加刚域偏移）。"""
    return (model.nodes[member.i].xyz + np.asarray(member.offset_i, dtype=float),
            model.nodes[member.j].xyz + np.asarray(member.offset_j, dtype=float))


def fixed_end_equivalent(L: float, w_local: np.ndarray) -> np.ndarray:
    """局部坐标下均布线荷载 (wx, wy, wz) 的等效节点荷载向量（12,）。"""
    wx, wy, wz = w_local
    p = np.zeros(12)
    p[0] = p[6] = wx * L / 2.0
    p[1] = p[7] = wy * L / 2.0
    p[2] = p[8] = wz * L / 2.0
    p[5] = wy * L**2 / 12.0
    p[11] = -wy * L**2 / 12.0
    p[4] = -wz * L**2 / 12.0
    p[10] = wz * L**2 / 12.0
    return p


@dataclass(frozen=True)
class _Condensation:
    """静力凝聚的中间量，用于求解后回算被释放自由度的真实位移。"""
    kept: tuple[int, ...]
    released: tuple[int, ...]
    inv_rr: np.ndarray
    k_rk: np.ndarray


def _condense(k: np.ndarray, released: tuple[int, ...], member_id: int):
    """按 f_released = 0 做静力凝聚，返回 (凝聚后刚度, 凝聚信息)。

    释放端不传力，故 k* = k_KK - k_KR k_RR^-1 k_RK，等效荷载同理缩并。
    两端同时释放轴向或扭转会让 k_RR 奇异——那在力学上就是一根散架的杆，直接报错。
    """
    if not released:
        return k, None
    kept = tuple(i for i in range(12) if i not in released)
    k_rr = k[np.ix_(released, released)]
    if np.linalg.cond(k_rr) > _RELEASE_COND_LIMIT:
        names = [(LOCAL_DOF_NAMES[i] if i < 6 else LOCAL_DOF_NAMES[i - 6]) for i in released]
        raise ValueError(
            f"杆件 {member_id} 的释放组合 {names} 使单元成为机构"
            "（例如两端同时释放轴向或扭转），无法静力凝聚"
        )
    inv_rr = np.linalg.inv(k_rr)
    k_kr = k[np.ix_(kept, released)]
    k_rk = k[np.ix_(released, kept)]
    star = np.zeros((12, 12))
    star[np.ix_(kept, kept)] = k[np.ix_(kept, kept)] - k_kr @ inv_rr @ k_rk
    return star, _Condensation(kept, released, inv_rr, k_rk)


def _condense_load(p: np.ndarray, cond: _Condensation | None) -> np.ndarray:
    if cond is None:
        return p
    star = np.zeros(12)
    k_kr = cond.k_rk.T
    star[list(cond.kept)] = p[list(cond.kept)] - k_kr @ cond.inv_rr @ p[list(cond.released)]
    return star


def _released_projection(cond: _Condensation | None) -> np.ndarray:
    """把节点保留自由度映射为含释放端转角的完整单元自由度。"""
    P = np.eye(12)
    if cond is None:
        return P
    P[list(cond.released), :] = 0.0
    P[np.ix_(cond.released, cond.kept)] = -cond.inv_rr @ cond.k_rk
    return P


def project_secondary_matrix(matrix: np.ndarray,
                             cond: _Condensation | None) -> np.ndarray:
    """按弹性释放关系投影质量/几何刚度等二次型矩阵。"""
    P = _released_projection(cond)
    return P.T @ matrix @ P


# --------------------------------------------------------------------------- 求解

@dataclass
class CaseResult:
    """一个工况（或组合）的结果。"""
    name: str
    U: np.ndarray
    R: np.ndarray
    member_forces: dict[int, np.ndarray]


@dataclass
class Solution:
    cases: dict[str, CaseResult]
    combos: dict[str, CaseResult]
    K: csr_matrix
    primary: str
    analysis: dict = field(default_factory=lambda: {"type": "linear_static"})

    def __getitem__(self, name: str) -> CaseResult:
        if name in self.cases:
            return self.cases[name]
        return self.combos[name]

    def all_results(self) -> dict[str, CaseResult]:
        return {**self.cases, **self.combos}

    # 单工况模型的便捷访问，保持与早期版本一致
    @property
    def U(self) -> np.ndarray:
        return self.cases[self.primary].U

    @property
    def R(self) -> np.ndarray:
        return self.cases[self.primary].R

    @property
    def member_forces(self) -> dict[int, np.ndarray]:
        return self.cases[self.primary].member_forces


def _member_matrices(model: Frame, m: Member):
    """返回 ``(L, R, T, B, k, k*, condensation)``。"""
    pi, pj = member_endpoints(model, m)
    L, R = local_axes(pi, pj, m.ref_vector)
    mat, sec = model.materials[m.material], model.sections[m.section]
    k_local = local_stiffness(L, mat.E, mat.G, sec)
    k_star, cond = _condense(k_local, m.released_indices(), m.id)
    return L, R, transformation(R), rigid_offset_transform(m), k_local, k_star, cond


def equivalent_local_load(model: "Frame", member: "Member", case: LoadCase,
                          L: float, rot: np.ndarray) -> np.ndarray:
    """一根杆件在某工况下的**全部**局部等效节点荷载（12,）。

    杆间荷载和初应变都从这里出。**右端项组装、杆端力回算、端释放位移还原
    这三处必须共用同一个入口**——任何一处漏算初应变，方程和内力就会各说各话，
    而且不会报任何错：自由伸长的杆会凭空出现轴力，或者约束住的杆算不出温度应力。

    初应变的等效节点力由 ``f₀ = ∫Bᵀ E ε₀ dV = EA·ε₀·[−1 … +1]`` 得到：
    升温（ε₀>0）把两端往外推，所以自由杆自由伸长、轴力为零；两端固定时
    位移为零，回算得到 ``N = −EA·ε₀``，即受压。这与讲义 §3-9 五、六一致。
    """
    p = np.zeros(12)
    for item in span_loads_of(case, member.id):
        p += fixed_end(item, L, rot)
    strain = float(case.member_strains.get(member.id, 0.0))
    if strain:
        axial = model.materials[member.material].E * model.sections[member.section].A * strain
        p[0] -= axial
        p[6] += axial
    return p


def assemble(model: Frame, case_names: list[str] | None = None):
    """装配整体刚度阵与各工况右端项。

    返回 (K, F, case_names)，F 形状为 (num_dofs, 工况数)。
    """
    if case_names is None:
        case_names = list(model.load_cases)
    n = model.num_dofs
    rows, cols, vals = [], [], []
    F = np.zeros((n, len(case_names)))

    for m in model.members.values():
        L, R, T, B, _, k_star, cond = _member_matrices(model, m)
        ke = B.T @ T.T @ k_star @ T @ B
        dofs = model.node_dofs(m.i) + model.node_dofs(m.j)
        for a in range(12):
            for b in range(12):
                rows.append(dofs[a]); cols.append(dofs[b]); vals.append(ke[a, b])
        for c, name in enumerate(case_names):
            p = equivalent_local_load(model, m, model.load_cases[name], L, R)
            if p.any():
                F[dofs, c] += B.T @ T.T @ _condense_load(p, cond)

    for c, name in enumerate(case_names):
        for nid, load in model.load_cases[name].nodal_loads.items():
            F[model.node_dofs(nid), c] += np.asarray(load, dtype=float)

    K = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return K, F, case_names


def constrained_dofs(model: Frame) -> np.ndarray:
    fixed = []
    for nid, mask in model.supports.items():
        dofs = model.node_dofs(nid)
        fixed += [dofs[k] for k, flag in enumerate(mask) if flag]
    return np.array(sorted(set(fixed)), dtype=int)


def _member_forces(model: Frame, U: np.ndarray, case: LoadCase) -> dict[int, np.ndarray]:
    """回算杆端力。被释放的自由度用凝聚关系还原其真实位移。"""
    out: dict[int, np.ndarray] = {}
    for m in model.members.values():
        L, R, T, B, k_local, _, cond = _member_matrices(model, m)
        dofs = model.node_dofs(m.i) + model.node_dofs(m.j)
        u_local = T @ B @ U[dofs]
        p = equivalent_local_load(model, m, case, L, R)
        if cond is not None:
            kept, rel = list(cond.kept), list(cond.released)
            u_local[rel] = cond.inv_rr @ (p[rel] - cond.k_rk @ u_local[kept])
        # 内力回算必须把固端力减回来，否则跨中弯矩全错
        out[m.id] = k_local @ u_local - p
    return out


def member_local_displacements(model: Frame, member: Member,
                               U: np.ndarray,
                               case: LoadCase | None = None) -> np.ndarray:
    """恢复杆件的 12 个真实局部端位移，包含刚域与端释放回算。

    后处理和模态显示必须走这个入口；直接拿节点平移连线会漏掉弯曲形状，
    释放端还会错误地使用共享节点转角。
    """
    L, R, T, B, _, _, cond = _member_matrices(model, member)
    dofs = model.node_dofs(member.i) + model.node_dofs(member.j)
    q = T @ B @ np.asarray(U, dtype=float)[dofs]
    p = (np.zeros(12) if case is None
         else equivalent_local_load(model, member, case, L, R))
    if cond is not None:
        kept, rel = list(cond.kept), list(cond.released)
        q[rel] = cond.inv_rr @ (p[rel] - cond.k_rk @ q[kept])
    return q


def solve(model: Frame, cases: list[str] | None = None) -> Solution:
    """求解全部工况，再按线性叠加合成荷载组合。

    整体刚度阵只分解一次，各工况共用——这是多工况相对逐个求解的全部好处。
    """
    K, F, case_names = assemble(model, cases)
    n = model.num_dofs
    fixed = constrained_dofs(model)
    free = np.setdiff1d(np.arange(n), fixed)

    U = np.zeros((n, len(case_names)))
    # 支座沉降是**给定位移**，先填进 U 的受约束行，再把它对自由行的影响
    # 移到右端：K_ff u_f = F_f − K_fs u_s。当成荷载去凑是错的。
    for c, name in enumerate(case_names):
        for nid, given in model.load_cases[name].settlements.items():
            if nid not in model.nodes:
                continue
            dofs = model.node_dofs(nid)
            fix = model.supports.get(nid, (0,) * 6)
            for k in range(DOF_PER_NODE):
                if fix[k]:
                    U[dofs[k], c] = float(given[k])
    has_settlement = bool(np.any(U[fixed])) if fixed.size else False

    if free.size:
        K_ff = K[free][:, free].tocsc()
        # 平动与转动自由度本来就量纲不同，复杂刚架再叠加长短杆、大小截面后，
        # K 的对角项可以跨十几个数量级。直接拿原矩阵主元比判断奇异，会把
        # “尺度差异”误报成机构。先做对称 Jacobi 缩放：
        #   Ks = D K D,  u = D y,  Ks y = D f
        # 这不改变物理解，只把每个自由度拉到可比较的数值尺度。
        diagonal = np.abs(K_ff.diagonal())
        if diagonal.size and (not np.all(np.isfinite(diagonal))
                              or np.any(diagonal <= 0.0)):
            raise np.linalg.LinAlgError(
                "求解失败：自由自由度存在零刚度，约束不足或存在机构。"
                "调用 diagnose_singularity 定位。")
        jacobi = 1.0 / np.sqrt(diagonal)
        D = diags(jacobi)
        K_scaled = (D @ K_ff @ D).tocsc()
        try:
            lu = splu(K_scaled)
        except RuntimeError as exc:
            raise np.linalg.LinAlgError(
                "求解失败：刚度矩阵奇异，约束不足或存在机构。"
                "调用 diagnose_singularity 定位。"
            ) from exc
        # splu 遇到奇异矩阵不一定抛异常——若荷载恰好不激发那个刚体方向，
        # 它会安静地返回一个错误答案。查一遍主元，把这种情况拦下来。
        pivots = np.abs(lu.U.diagonal())
        if pivots.size and pivots.min() <= _SINGULAR_PIVOT_RATIO * pivots.max():
            raise np.linalg.LinAlgError(
                "求解失败：刚度矩阵奇异（存在近零主元），约束不足或存在机构。"
                "调用 diagnose_singularity 定位。"
            )
        K_fs = K[free][:, fixed] if (fixed.size and has_settlement) else None
        for c in range(len(case_names)):
            rhs = F[free, c]
            if K_fs is not None:
                rhs = rhs - K_fs @ U[fixed, c]
            # 先解缩放变量 y，再由 u = D y 回到模型的原始单位。
            U[free, c] = jacobi * lu.solve(jacobi * rhs)
    if not np.all(np.isfinite(U)):
        raise np.linalg.LinAlgError("求解失败：位移出现非有限值，刚度矩阵接近奇异。")

    results: dict[str, CaseResult] = {}
    for c, name in enumerate(case_names):
        Uc = U[:, c]
        Rc = np.zeros(n)
        Rc[fixed] = (K @ Uc - F[:, c])[fixed]
        results[name] = CaseResult(
            name=name, U=Uc, R=Rc,
            member_forces=_member_forces(model, Uc, model.load_cases[name]),
        )

    combos: dict[str, CaseResult] = {}
    for combo_name, factors in model.combos.items():
        missing = [c for c in factors if c not in results]
        if missing:
            raise KeyError(f"荷载组合 {combo_name!r} 引用了未定义的工况 {missing}")
        Uc = sum(f * results[c].U for c, f in factors.items())
        Rc = sum(f * results[c].R for c, f in factors.items())
        forces = {
            mid: sum(f * results[c].member_forces[mid] for c, f in factors.items())
            for mid in model.members
        }
        combos[combo_name] = CaseResult(combo_name, Uc, Rc, forces)

    primary = DEFAULT_CASE if DEFAULT_CASE in results else case_names[0]
    return Solution(cases=results, combos=combos, K=K, primary=primary)


# --------------------------------------------------------------------------- 自校验

def self_weight_loads(model: Frame, factor: float = 1.0,
                      direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
                      ) -> dict[int, SpanLoad]:
    """按截面面积与材料密度算每根杆的自重，返回满跨均布荷载。

    做成"生成一批显式荷载"而不是求解时隐式加上去：这样自重在模型表格里
    看得见、进 Abaqus 导出、也参与平衡校核，不会变成一项谁也查不到的暗账。
    代价是改了截面要重新生成一次——这一点在工具说明里写清楚。

    密度为零的材料不产生自重（默认就是零，老模型的结果一个字都不会变）。

    g 取自模型的单位制。写死成 9.80665 的话，N-mm-MPa 下自重会小 1000 倍
    而且不报错——这是这套代码里唯一一处真正依赖单位制的常数。
    """
    g = model.unit_system.gravity
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise ValueError("自重方向不能是零向量")
    d = d / norm
    out: dict[int, SpanLoad] = {}
    for mid, m in model.members.items():
        mat = model.materials.get(m.material)
        sec = model.sections.get(m.section)
        if mat is None or sec is None or not mat.density:
            continue
        w = factor * mat.density * sec.A * g
        if w == 0.0:
            continue
        out[mid] = SpanLoad(UNIFORM, tuple(float(v) for v in w * d))
    return out


def combined_case(model: Frame, factors: dict[str, float]) -> LoadCase:
    """把若干工况按系数合成一个等效工况，用于组合的平衡校核。"""
    merged = LoadCase("__combined__")
    for name, f in factors.items():
        case = model.load_cases[name]
        for nid, load in case.nodal_loads.items():
            base = np.asarray(merged.nodal_loads.get(nid, (0.0,) * 6))
            merged.nodal_loads[nid] = tuple(base + f * np.asarray(load, dtype=float))
        for mid, w in case.member_loads.items():
            base = np.asarray(merged.member_loads.get(mid, (0.0,) * 3))
            merged.member_loads[mid] = tuple(base + f * np.asarray(w, dtype=float))
        for mid, items in case.member_spans.items():
            merged.member_spans.setdefault(mid, []).extend(
                item.scaled(f) for item in items)
        for nid, given in case.settlements.items():
            base = np.asarray(merged.settlements.get(nid, (0.0,) * 6))
            merged.settlements[nid] = tuple(base + f * np.asarray(given, dtype=float))
    return merged


def check_equilibrium(model: Frame, sol: Solution, case: str | None = None,
                      rtol: float = 1e-8) -> dict:
    """整体静力平衡：支座反力合力 + 外荷载合力 = 0。"""
    name = case or sol.primary
    if name in model.combos:
        load_case = combined_case(model, model.combos[name])
    else:
        load_case = model.load_cases[name]
    result = sol[name]

    applied = np.zeros(6)
    for nid, load in load_case.nodal_loads.items():
        p = np.asarray(load, dtype=float)
        r = model.nodes[nid].xyz
        applied[:3] += p[:3]
        applied[3:] += p[3:] + np.cross(r, p[:3])
    for mid in sorted(set(load_case.member_loads) | set(load_case.member_spans)):
        m = model.members[mid]
        pi, pj = member_endpoints(model, m)
        L = float(np.linalg.norm(pj - pi))
        for item in span_loads_of(load_case, mid):
            force, moment = moment_about_origin(item, L, pi, pj)
            applied[:3] += force
            applied[3:] += moment

    reac = np.zeros(6)
    for nid in model.supports:
        d = model.node_dofs(nid)
        p = result.R[d]
        r = model.nodes[nid].xyz
        reac[:3] += p[:3]
        reac[3:] += p[3:] + np.cross(r, p[:3])

    resid = applied + reac
    scale = max(np.abs(applied).max(initial=0.0), np.abs(reac).max(initial=0.0), 1.0)
    return {"case": name, "applied": applied, "reaction": reac, "residual": resid,
            "ok": bool(np.abs(resid).max() / scale < rtol),
            "relative": float(np.abs(resid).max() / scale)}


_DOF_NAME = ["沿 X 平动", "沿 Y 平动", "沿 Z 平动", "绕 X 转动", "绕 Y 转动", "绕 Z 转动"]


def diagnose_singularity(model: Frame, tol: float = 1e-8) -> list[dict]:
    """定位约束不足的刚体位移模态，输出可读的机构描述。

    对约束后的刚度阵做特征分解，取近零特征值对应的向量，映射回节点自由度，
    找出参与最大的节点与方向。大模型拿到这份结构化诊断即可翻译成建议。
    """
    K, _, _ = assemble(model)
    fixed = constrained_dofs(model)
    free = np.setdiff1d(np.arange(model.num_dofs), fixed)
    if free.size == 0:
        return []

    Kff = np.asarray(K[free][:, free].todense())
    vals, vecs = np.linalg.eigh(Kff)
    ref = max(abs(vals).max(), 1.0)

    modes = []
    for k, lam in enumerate(vals):
        if abs(lam) / ref >= tol:
            continue
        v = np.zeros(model.num_dofs)
        v[free] = vecs[:, k]
        contrib = []
        for nid in model.order():
            d = model.node_dofs(nid)
            for c in range(6):
                a = abs(v[d[c]])
                if a > 0.15 * np.abs(v).max():
                    contrib.append({"node": nid, "dof": c,
                                    "dof_name": _DOF_NAME[c], "amplitude": float(v[d[c]])})
        contrib.sort(key=lambda e: -abs(e["amplitude"]))
        modes.append({"eigenvalue": float(lam), "participants": contrib[:8]})
    return modes


def check_model(model: Frame) -> list[str]:
    """几何、拓扑与荷载定义的合法性。返回问题清单，空列表表示通过。"""
    issues: list[str] = []
    used: set[int] = set()
    seen: dict[tuple[int, int], int] = {}

    # 当前内核没有 tie/MPC。两个坐标重合但编号不同的节点在图上像一个节点，
    # 在刚度矩阵里却完全断开，是最危险的“看起来正确”。严格模型中直接拒绝。
    node_list = list(model.nodes.values())
    for index, a in enumerate(node_list):
        for b in node_list[index + 1:]:
            if float(np.linalg.norm(a.xyz - b.xyz)) < 1e-9:
                issues.append(f"节点 {a.id} 与节点 {b.id} 坐标重合；请合并为同一节点")

    for m in model.members.values():
        for nid in (m.i, m.j):
            if nid not in model.nodes:
                issues.append(f"杆件 {m.id} 引用了不存在的节点 {nid}")
        if m.i == m.j:
            issues.append(f"杆件 {m.id} 首尾为同一节点 {m.i}")
        if m.section not in model.sections:
            issues.append(f"杆件 {m.id} 引用了未定义的截面 {m.section!r}")
        if m.material not in model.materials:
            issues.append(f"杆件 {m.id} 引用了未定义的材料 {m.material!r}")
        for name in m.releases_i + m.releases_j:
            if name not in LOCAL_DOF_NAMES:
                issues.append(
                    f"杆件 {m.id} 的释放自由度 {name!r} 无效，应取自 {LOCAL_DOF_NAMES}")
        if m.i in model.nodes and m.j in model.nodes:
            pi, pj = member_endpoints(model, m)
            L = float(np.linalg.norm(pj - pi))
            if L < 1e-9:
                issues.append(f"杆件 {m.id} 长度为零")
            key = (min(m.i, m.j), max(m.i, m.j))
            if key in seen:
                issues.append(f"杆件 {m.id} 与杆件 {seen[key]} 连接同一对节点")
            seen[key] = m.id
        # 释放组合是否让单元变成机构，这里就查出来，不用等到求解报错
        if (m.i in model.nodes and m.j in model.nodes
                and m.section in model.sections and m.material in model.materials
                and all(n in LOCAL_DOF_NAMES for n in m.releases_i + m.releases_j)):
            try:
                _member_matrices(model, m)
            except ValueError as exc:
                issues.append(str(exc))
        used.update((m.i, m.j))

    for nid in model.nodes:
        if nid not in used:
            node = model.nodes[nid]
            for member in model.members.values():
                if member.i not in model.nodes or member.j not in model.nodes:
                    continue
                pi, pj = member_endpoints(model, member)
                direction = pj - pi
                length = float(np.linalg.norm(direction))
                if length < 1e-9:
                    continue
                position = float(np.dot(node.xyz - pi, direction) / length)
                tolerance = max(1e-9, length * 1e-9)
                if tolerance < position < length - tolerance:
                    projection = pi + (position / length) * direction
                    if float(np.linalg.norm(node.xyz - projection)) <= tolerance:
                        used.add(nid)
                        break
        if nid not in used:
            issues.append(f"节点 {nid} 未被任何杆件引用（悬空节点）")
    # 注意：不检查"没有支座"——人工建模时先建几何再加支座是正常流程，
    # 求解时无支座会自然报错（刚度矩阵奇异），不需要在这里拦

    for name, case in model.load_cases.items():
        for nid in case.nodal_loads:
            if nid not in model.nodes:
                issues.append(f"工况 {name!r} 在不存在的节点 {nid} 上施加了荷载")
        for mid in case.member_loads:
            if mid not in model.members:
                issues.append(f"工况 {name!r} 在不存在的杆件 {mid} 上施加了荷载")
        for mid, items in case.member_spans.items():
            if mid not in model.members:
                issues.append(f"工况 {name!r} 在不存在的杆件 {mid} 上施加了荷载")
                continue
            m = model.members[mid]
            if m.i not in model.nodes or m.j not in model.nodes:
                continue
            L = float(np.linalg.norm(model.nodes[m.j].xyz - model.nodes[m.i].xyz))
            for item in items:
                if item.kind != POINT:
                    continue
                # 落在杆件之外的集中力算不出来，公式里 b = L - a 会变号，
                # 结果看着像个数其实没有意义——必须在这里拦住
                if not (0.0 <= item.a <= L):
                    length_unit = "mm" if model.units == "N-mm-MPa" else "m"
                    issues.append(
                        f"工况 {name!r} 杆件 {mid} 上的集中力位置 "
                        f"a={item.a:g} {length_unit} 超出杆长 0~{L:g} {length_unit}")
    for name, case in model.load_cases.items():
        for nid, given in case.settlements.items():
            if nid not in model.nodes:
                issues.append(f"工况 {name!r} 在不存在的节点 {nid} 上给了支座沉降")
                continue
            fix = model.supports.get(nid)
            if fix is None:
                issues.append(f"工况 {name!r} 给节点 {nid} 指定了沉降，但它不是支座节点。"
                              "沉降是给定位移，只能加在被约束住的方向上")
                continue
            loose = [LOCAL_DOF_NAMES[k] for k in range(DOF_PER_NODE)
                     if given[k] and not fix[k]]
            if loose:
                # 自由方向上"指定位移"没有意义，代码会静静忽略它——
                # 用户却以为自己施加了。必须说出来
                issues.append(
                    f"工况 {name!r} 节点 {nid} 在未约束方向 {loose} 上指定了沉降，"
                    "这些分量不会生效；先把该方向约束住")

    for combo, factors in model.combos.items():
        if not factors:
            issues.append(f"荷载组合 {combo!r} 为空")
        for cname in factors:
            if cname not in model.load_cases:
                issues.append(f"荷载组合 {combo!r} 引用了未定义的工况 {cname!r}")
    return issues
