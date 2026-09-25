"""高级梁柱能力的解析与不变量测试。"""

import numpy as np
import pytest

import dataclasses
import math

from frame3d import (DEFAULT_CASE, Amplitude, Frame, LoadCase, Material, Member,
                     Node, Section, check_equilibrium, combined_case, solve)
from modal import member_mode_displacement
from nonlinear import (STEP_CASE, _scaled_case, solve_material_nonlinear,
                       solve_pdelta, solve_step)
from span_loads import POINT, SpanLoad


E = 210e9
NU = 0.3
G = E / (2 * (1 + NU))
FIX = (1, 1, 1, 1, 1, 1)


def cantilever(length=1.0, section=None, member_kwargs=None):
    f = Frame()
    f.nodes[1] = Node(1, 0, 0, 0)
    f.nodes[2] = Node(2, length, 0, 0)
    f.materials["M"] = Material("M", E, NU, 7850)
    f.sections["S"] = section or Section("S", 0.02, 2e-5, 4e-5, 1e-5)
    f.members[1] = Member(1, 1, 2, "S", "M", **(member_kwargs or {}))
    f.supports[1] = FIX
    return f


@pytest.mark.gold
def test_timoshenko_tip_deflection_includes_shear_term():
    length = 0.8
    area_y = 0.012
    sec = Section("S", 0.02, 2e-5, 4e-5, 1e-5, Ay=area_y)
    f = cantilever(length, sec)
    force = -40e3
    f.nodal_loads[2] = (0, 0, force, 0, 0, 0)
    result = solve(f)
    expected = force * length**3 / (3 * E * sec.Iz) + force * length / (G * area_y)
    assert result.U[f.node_dofs(2)[2]] == pytest.approx(expected, rel=1e-11)


def test_missing_shear_area_keeps_euler_bernoulli_result():
    f = cantilever()
    force = -10e3
    f.nodal_loads[2] = (0, 0, force, 0, 0, 0)
    result = solve(f)
    assert result.U[f.node_dofs(2)[2]] == pytest.approx(
        force / (3 * E * f.sections["S"].Iz), rel=1e-11)


def test_rigid_offset_shortens_flexible_span_and_transfers_moment():
    offset = 0.2
    f = cantilever(member_kwargs={"offset_i": (offset, 0, 0)})
    force = -10e3
    f.nodal_loads[2] = (0, 0, force, 0, 0, 0)
    result = solve(f)
    flexible = 1.0 - offset
    expected = force * flexible**3 / (3 * E * f.sections["S"].Iz)
    assert result.U[f.node_dofs(2)[2]] == pytest.approx(expected, rel=1e-10)
    assert abs(result.R[f.node_dofs(1)[4]]) == pytest.approx(abs(force), rel=1e-10)


def test_mode_recovery_uses_rotations_and_keeps_exact_endpoints():
    f = cantilever()
    shapes = np.zeros((f.num_dofs, 1))
    shapes[f.node_dofs(2)[4], 0] = -1.0
    points, displacement = member_mode_displacement(f, shapes, 0, 1, stations=5)
    assert np.allclose(displacement[0], 0.0)
    assert np.allclose(displacement[-1], 0.0)
    assert abs(displacement[2, 2]) > 0.0
    assert np.allclose(points[[0, -1]], [[0, 0, 0], [1, 0, 0]])


def test_pdelta_amplifies_lateral_displacement_under_compression():
    f = cantilever(length=3.0)
    # 局部轴向压力与横向扰动力同时施加。
    f.nodal_loads[2] = (-1.0e6, 0, -2.0e3, 0, 0, 0)
    linear = solve(f).U[f.node_dofs(2)[2]]
    second = solve_pdelta(f, increments=6).U[f.node_dofs(2)[2]]
    assert abs(second) > abs(linear)
    assert second < 0.0


@pytest.mark.gold
def test_bilinear_axial_material_uses_incremental_return_mapping():
    f = cantilever(length=2.0)
    base = f.materials["M"]
    fy = 200e6
    ratio = 0.02
    f.materials["M"] = Material("M", base.E, base.nu, base.density,
                                yield_stress=fy, hardening_ratio=ratio)
    force = 2.1e6  # A=0.02 -> 105 MPa, still elastic; lower A to cross yield
    f.sections["S"] = Section("S", 0.01, 2e-5, 4e-5, 1e-5)
    force = 2.1e6  # stress = 210 MPa
    f.nodal_loads[2] = (force, 0, 0, 0, 0, 0)
    result = solve_material_nonlinear(f, increments=12)
    expected_strain = fy / E + (force / 0.01 - fy) / (ratio * E)
    assert result.U[f.node_dofs(2)[0]] == pytest.approx(
        2.0 * expected_strain, rel=2e-7)
    assert result.analysis["type"] == "material_nonlinear_axial_bilinear"


# ------------------------------------- P-Δ 的精确二阶弹性解

def _meshed_cantilever(n: int, length: float) -> Frame:
    """把悬臂分成 n 个单元。几何刚度阵是近似的，验精确解必须能加密。"""
    f = Frame()
    for index in range(n + 1):
        f.nodes[index + 1] = Node(index + 1, length * index / n, 0, 0)
    f.materials["M"] = Material("M", E, NU, 7850)
    f.sections["S"] = Section("S", 0.02, 2e-5, 4e-5, 1e-5)
    for index in range(n):
        f.members[index + 1] = Member(index + 1, index + 1, index + 2, "S", "M")
    f.supports[1] = FIX
    return f


def _exact_pdelta_tip(axial: float, lateral: float, length: float,
                      ei: float) -> float:
    """悬臂柱顶同时受轴压 P 与横向力 H 时的**精确**二阶弹性挠度。

        δ = H/(P·k) · ( tan(kL) − kL ),   k = √(P/EI)

    P→0 时 tan(kL) − kL → (kL)³/3，退化为 HL³/(3EI)，与线性解一致。
    这个解来自二阶微分方程本身，不是"线性解乘一个放大系数"的近似，
    也不来自本项目的任何一段实现。
    """
    k = np.sqrt(axial / ei)
    return lateral / (axial * k) * (np.tan(k * length) - k * length)


@pytest.mark.gold
def test_pdelta_converges_to_the_exact_second_order_solution():
    """P-Δ 必须收敛到精确二阶解，而不只是"比线性解大"。

    取 P/P_cr ≈ 0.43，二阶效应把挠度放大到 1.757 倍——量级足够大，
    实现里少算一项也躲不过去。实测相对误差：
    1 单元 2.3e-3、2 单元 1.7e-4、4 单元 1.1e-5、8 单元 6.8e-7、16 单元 4.3e-8。
    """
    length, axial, lateral = 3.0, 1.0e6, 2.0e3
    ei = E * 4e-5
    exact = _exact_pdelta_tip(axial, lateral, length, ei)

    # 二阶效应确实显著，否则这条测试等于在验线性解
    assert exact / (lateral * length ** 3 / (3 * ei)) > 1.5

    errors = []
    for n in (2, 4, 8):
        f = _meshed_cantilever(n, length)
        f.nodal_loads[n + 1] = (-axial, 0, -lateral, 0, 0, 0)
        tip = solve_pdelta(f, increments=8).U[f.node_dofs(n + 1)[2]]
        errors.append(abs(abs(tip) - abs(exact)) / abs(exact))

    assert errors[-1] < 1e-5                       # 8 单元贴合精确解
    assert all(a > b for a, b in zip(errors, errors[1:], strict=False)), errors   # 单调收敛


def test_pdelta_reduces_to_the_linear_solution_without_axial_force():
    """轴力撤掉后二阶解必须退回线性解——守的是几何刚度没有被无条件叠加。"""
    length, lateral = 3.0, 2.0e3
    ei = E * 4e-5
    f = _meshed_cantilever(4, length)
    f.nodal_loads[5] = (0.0, 0, -lateral, 0, 0, 0)
    tip = solve_pdelta(f, increments=4).U[f.node_dofs(5)[2]]
    assert abs(tip) == pytest.approx(lateral * length ** 3 / (3 * ei), rel=1e-9)


# ---------------------------------------------------------------------------
# 增量求解器的荷载覆盖闸门
#
# solve_pdelta 与 solve_material_nonlinear 按增量重建荷载工况（_scaled_case）。
# 那个函数原先只抄了四个字段，后来新增的初应变与初曲率被**静默丢弃**：荷载
# 向量变成零、位移全是 0，而最终内力又是拿未缩放的 original 重算的，所以
# 自检、平衡校核、残差全都正常。这跟编译器漏掉 partial 是同一类错。
#
# 下面的闸门按 LoadCase 的 dataclass 字段逐个参数化。**新增一个字段而不在
# _scaled_case 里处理，这里就会红。**
# ---------------------------------------------------------------------------

#: 不随荷载比例缩放的字段，各自附理由。
_NOT_A_LOAD = {
    "name": "工况名，不是荷载",
    "settlements": "给定位移是边界条件，缩放它等于改变边界",
}


def _thermal_cantilever(strain=0.0, curvature=(0.0, 0.0)):
    f = cantilever(6.0)
    f.materials["M"] = Material("M", E, NU, 7850)
    case = f.case()
    if strain:
        case.member_strains[1] = strain
    if any(curvature):
        case.member_curvatures[1] = curvature
    return f


#: 每个可缩放字段配一个"只用这个字段加载"的模型。键必须覆盖 LoadCase 的
#: 全部荷载字段——缺一个，下面第一个测试就会指名道姓地失败。
_ONLY_THIS_FIELD = {
    "nodal_loads": lambda: _loaded(lambda f: f.nodal_loads.__setitem__(
        2, (0, 0, -30e3, 0, 0, 0))),
    "member_loads": lambda: _loaded(lambda f: f.case().member_loads.__setitem__(
        1, (0.0, 0.0, -8e3))),
    "member_spans": lambda: _loaded(lambda f: f.case().member_spans.__setitem__(
        1, [SpanLoad(POINT, (0.0, 0.0, -25e3), a=3.0)])),
    "member_strains": lambda: _thermal_cantilever(strain=1.2e-5 * 40),
    "member_curvatures": lambda: _thermal_cantilever(curvature=(0.0, 1.0e-4)),
}


def _loaded(apply):
    f = cantilever(6.0)
    apply(f)
    return f


def test_every_load_case_field_is_covered_by_the_gate():
    """LoadCase 新增字段时，这张表必须跟着加——否则下面的测试形同虚设。"""
    declared = {f.name for f in dataclasses.fields(LoadCase)}
    handled = set(_ONLY_THIS_FIELD) | set(_NOT_A_LOAD)
    assert declared == handled, (
        f"LoadCase 的字段 {declared - handled} 还没有人认领："
        "要么在 _ONLY_THIS_FIELD 里给它一个模型（它是荷载，要缩放），"
        "要么在 _NOT_A_LOAD 里写明为什么不缩放。")


@pytest.mark.parametrize("field_name", sorted(_ONLY_THIS_FIELD))
def test_pdelta_does_not_drop_any_kind_of_load(field_name):
    """每一种荷载单独加载时，P-Δ 都必须真的把它算进去。

    判据是"位移非零且与线性解一致"：只验非零不够——把荷载丢掉之后位移
    恰好是零，而只验"有结果"是看不出来的。
    """
    model = _ONLY_THIS_FIELD[field_name]()
    linear = solve(model).U
    assert np.linalg.norm(linear) > 0, "线性解就是零，这个模型验不出东西"
    second = solve_pdelta(model, increments=4).U
    assert np.linalg.norm(second) > 0, (
        f"{field_name} 在 P-Δ 里被丢掉了：位移全是零")
    # 无轴力的悬臂上二阶效应为零，两者必须逐个自由度相等
    assert second == pytest.approx(linear, rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("field_name", sorted(_ONLY_THIS_FIELD))
def test_scaled_case_actually_halves_each_kind_of_load(field_name):
    """缩放一半，线性响应就该是一半——证明缩放是按比例而不是照抄或清零。"""
    model = _ONLY_THIS_FIELD[field_name]()
    full = solve(model).U
    model.load_cases[DEFAULT_CASE] = _scaled_case(model.case(), 0.5)
    half = solve(model).U
    assert half == pytest.approx(0.5 * full, rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("field_name", sorted(_ONLY_THIS_FIELD))
def test_combined_case_carries_every_kind_of_load(field_name):
    """合成工况也必须搬运每一种荷载。

    它与 _scaled_case 是两份各自维护的搬运代码，初应变在两边都漏过。
    组合目前靠结果叠加，所以漏字段暂时不改变组合的答案——但非比例加载
    的分析步是直接拿合成工况去解的，那里漏一个字段就是错的结果。
    """
    model = _ONLY_THIS_FIELD[field_name]()
    full = solve(model).U
    model.load_cases[DEFAULT_CASE] = combined_case(model, {DEFAULT_CASE: 0.5})
    half = solve(model).U
    assert np.linalg.norm(half) > 0, (
        f"{field_name} 没有进到合成工况里：位移全是零")
    assert half == pytest.approx(0.5 * full, rel=1e-9, abs=1e-14)


# ---------------------------------------------------------------------------
# 幅值曲线与分析步
# ---------------------------------------------------------------------------


def sway_column(height=6.0, axial=1.6e6, lateral=40e3):
    """一根悬臂柱，重力与侧力分成两个工况——非比例加载要两个工况才谈得上。"""
    f = Frame()
    f.nodes[1] = Node(1, 0, 0, 0)
    f.nodes[2] = Node(2, 0, 0, height)
    f.members[1] = Member(1, 1, 2, "S", "M")
    f.sections["S"] = Section("S", 0.02, 2e-4, 2e-4, 1e-5)
    f.materials["M"] = Material("M", E, NU, 7850)
    f.supports[1] = FIX
    f.load_cases.clear()
    f.load_cases["DL"] = LoadCase("DL")
    f.load_cases["DL"].nodal_loads[2] = (0, 0, -axial, 0, 0, 0)
    f.load_cases["WX"] = LoadCase("WX")
    f.load_cases["WX"].nodal_loads[2] = (lateral, 0, 0, 0, 0, 0)
    return f


def test_amplitude_interpolates_linearly_and_does_not_extrapolate():
    curve = Amplitude("试", ((0.0, 0.0), (0.4, 1.0), (1.0, 1.0)))
    assert curve.at(0.2) == pytest.approx(0.5)
    assert curve.at(0.4) == pytest.approx(1.0)
    assert curve.at(0.7) == pytest.approx(1.0)
    # 表外不外推——否则浮点边界上会冒出一个表里没有的系数
    assert curve.at(-5.0) == pytest.approx(0.0)
    assert curve.at(99.0) == pytest.approx(1.0)


@pytest.mark.parametrize("points", [((0.0, 0.0),), ((0.0, 0.0), (0.0, 1.0)),
                                    ((1.0, 1.0), (0.0, 0.0))])
def test_a_malformed_amplitude_is_refused(points):
    with pytest.raises(ValueError):
        Amplitude("坏", points)


def test_a_step_with_all_ramps_reproduces_plain_pdelta():
    """全用线性斜坡时必须与原来的 solve_pdelta 逐位相同——新通路不能改旧答案。"""
    f = sway_column()
    f.load_cases["ALL"] = combined_case(f, {"DL": 1.0, "WX": 1.0})
    reference = solve_pdelta(f, cases=["ALL"], increments=8).U
    stepped = solve_step(f, {"DL": "RAMP", "WX": "RAMP"}, increments=8).U
    assert stepped == pytest.approx(reference, rel=1e-10, abs=1e-14)


def test_the_elastic_endpoint_does_not_depend_on_the_load_path():
    """二阶弹性问题的解是同一个方程的根，路径改变不了它。

    这条是**反过来用的**：它挡住"把幅值曲线宣传成更准的算法"这种误解，
    也挡住合成工况在 t=1 处没把某个工况加满的实现错误。
    """
    f = sway_column()
    proportional = solve_step(f, {"DL": "RAMP", "WX": "RAMP"}, increments=8).U
    pushover = solve_step(f, {"DL": "STEP", "WX": "RAMP"}, increments=8).U
    assert pushover == pytest.approx(proportional, rel=1e-9, abs=1e-12)


def test_holding_gravity_constant_makes_the_pushover_curve_straight():
    """轴力恒定 ⇒ 切线刚度恒定 ⇒ 顶点位移对侧力是直线。

    这才是幅值曲线真正买到的东西。比例加载那条曲线不是任何真实工况的响应。
    """
    f = sway_column()
    steps = 8
    path = solve_step(f, {"DL": "STEP", "WX": "RAMP"},
                      increments=steps).analysis["convergence"][STEP_CASE]
    final = path[-1]["max_displacement"]
    for entry in path:
        assert entry["max_displacement"] == pytest.approx(
            final * entry["load_factor"], rel=1e-8)

    # 比例加载则明显不是直线：中点处的位移远低于终点的一半
    mixed = solve_step(f, {"DL": "RAMP", "WX": "RAMP"},
                       increments=steps).analysis["convergence"][STEP_CASE]
    half = mixed[steps // 2 - 1]
    assert half["max_displacement"] < 0.8 * 0.5 * mixed[-1]["max_displacement"]


def test_the_second_order_amplifier_matches_the_closed_form():
    """顶点位移放大到一阶解的 1/(1−P/Pcr) 倍——对着闭合解验，不是自洽。"""
    height, axial, lateral = 6.0, 1.6e6, 40e3
    f = sway_column(height, axial, lateral)
    inertia = f.sections["S"].Iz
    first_order = lateral * height**3 / (3 * E * inertia)
    critical = math.pi**2 * E * inertia / (2 * height) ** 2
    result = solve_step(f, {"DL": "STEP", "WX": "RAMP"}, increments=12)
    amplified = result.U[f.node_dofs(2)[0]]
    assert amplified / first_order == pytest.approx(
        1 / (1 - axial / critical), rel=0.02)


def test_a_custom_amplitude_is_read_from_the_model():
    """自定义曲线：前 30% 伪时间就把侧力加满，其后保持。"""
    f = sway_column()
    f.amplitudes["前段加满"] = Amplitude("前段加满", ((0.0, 0.0), (0.3, 1.0),
                                                     (1.0, 1.0)))
    path = solve_step(f, {"DL": "STEP", "WX": "前段加满"},
                      increments=10).analysis["convergence"][STEP_CASE]
    final = path[-1]["max_displacement"]
    assert path[2]["max_displacement"] == pytest.approx(final, rel=1e-8)
    assert path[0]["max_displacement"] == pytest.approx(final / 3, rel=1e-8)


def test_an_unknown_amplitude_is_refused_by_name():
    f = sway_column()
    with pytest.raises(ValueError, match="幅值曲线"):
        solve_step(f, {"DL": "没这条"})


def test_an_unknown_case_is_refused_by_name():
    f = sway_column()
    with pytest.raises(ValueError, match="未定义的荷载工况"):
        solve_step(f, {"没这个": "RAMP"})


def test_an_empty_step_is_refused():
    with pytest.raises(ValueError, match="至少要施加一个"):
        solve_step(sway_column(), {})


def test_second_order_equilibrium_needs_the_deformed_geometry():
    """二阶解在未变形几何上查平衡，残差恰好是 P·Δ——那不是误差。

    这条测的是一次**假警报**：一根轴压 1600 kN、顶点侧移 54 mm 的柱子，
    拿未变形几何查会报出 5% 的"不平衡"，而模型完全正确。用户会以为自己
    的荷载加错了，然后去改一个没有错的模型。
    """
    axial, lateral = 1.6e6, 40e3
    f = sway_column(axial=axial, lateral=lateral)
    sol = solve_step(f, {"DL": "STEP", "WX": "RAMP"}, increments=8)

    def relative(deformed):
        report = check_equilibrium(f, sol, STEP_CASE, deformed=deformed)
        scale = max(np.abs(report["applied"]).max(),
                    np.abs(report["reaction"]).max(), 1.0)
        return float(np.abs(report["residual"]).max() / scale)

    drift = sol.U[f.node_dofs(2)[0]]
    # 未变形几何下的残差就是 P·Δ 除以最大力分量，不是随便一个数
    assert relative(False) == pytest.approx(axial * drift / axial, rel=0.05)
    # 变形后位形下残差掉到几何刚度近似本身的量级
    assert relative(True) < 1e-3
    assert relative(True) < relative(False) / 100


def test_a_first_order_solution_balances_in_either_geometry():
    """一阶分析里两种查法必须等价——位移小，力臂的改变量是高阶小量。"""
    f = sway_column(axial=0.0)
    sol = solve(f, cases=["WX"])
    for deformed in (False, True):
        report = check_equilibrium(f, sol, "WX", deformed=deformed)
        scale = max(np.abs(report["applied"]).max(), 1.0)
        assert np.abs(report["residual"]).max() / scale < 1e-9
