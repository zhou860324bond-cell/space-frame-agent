"""模态参与系数、有效质量与振型分解反应谱法。

反应谱法有一条**精确**的恒等式可以拿来验实现：各阶基底剪力就是
M_eff,k·Sa(T_k)，取满振型、谱取常量时它们的和等于「能参与振动的质量」
乘以 Sa。这比"数看着合理"强得多，所以这里的主干都挂在它上面。

另一条同样精确：完备性关系 Σ_全部振型 M_eff = rᵀM_ff r。注意右边**不是**
Σ ρAL——压在支座上的质量永远不参与振动。
"""

import numpy as np
import pytest

from frame3d import (DOF_PER_NODE, Frame, LoadCase, Material, Member, Node,
                     Section, _member_forces, constrained_dofs)
from modal import assemble_mass, modal
from spectrum import (gb50011_spectrum, response_spectrum, table_spectrum)

E, NU, RHO = 2.1e11, 0.3, 7850
AREA, IY, IZ = 0.02, 2e-4, 4e-4
HEIGHT = 6.0


def column(elements=12, height=HEIGHT):
    """沿全局 Z 的悬臂柱，剖成若干段。"""
    f = Frame()
    for k in range(elements + 1):
        f.nodes[k + 1] = Node(k + 1, 0.0, 0.0, height * k / elements)
    for k in range(elements):
        f.members[k + 1] = Member(k + 1, k + 1, k + 2, "S", "M")
    f.sections["S"] = Section("S", AREA, IY, IZ, 1e-5)
    f.materials["M"] = Material("M", E, NU, RHO)
    f.supports[1] = (1,) * 6
    return f


def participable(model, axis=0):
    """rᵀ M_ff r：能参与振动的那部分质量，直接从质量矩阵算，独立于 modal()。"""
    mass = assemble_mass(model).tocsr()
    fixed = constrained_dofs(model)
    free = np.setdiff1d(np.arange(model.num_dofs), fixed)
    rigid = np.zeros(model.num_dofs)
    rigid[axis::DOF_PER_NODE] = 1.0
    reduced = rigid[free]
    return float(reduced @ (mass[np.ix_(free, free)] @ reduced))


# --- 有效质量的完备性关系 -------------------------------------------------

@pytest.mark.gold
@pytest.mark.parametrize("elements", [4, 8, 16])
def test_all_modes_account_for_exactly_the_participable_mass(elements):
    """Σ_全部振型 M_eff = rᵀM_ff r，**精确**成立。

    这条是有效质量实现对不对的终审：它是完备性关系，不是近似。
    """
    model = column(elements)
    result = modal(model, num_modes=6 * elements)
    assert result.effective_mass.sum(axis=0)[0] == pytest.approx(
        participable(model, 0), rel=1e-12)


@pytest.mark.parametrize("elements", [4, 8, 16])
def test_the_participation_ratio_reaches_one_hundred_percent(elements):
    """取满振型时累计参与质量比必须是 100%，**与网格粗细无关**。

    分母要是用 Σ ρAL，4 单元的模型把振型取满也只有 84%——用户照
    GB 50011「累计 ≥90%」去判会以为阶数不够，然后不断加阶数，
    加到天荒地老也过不去，因为差的那部分在支座上。
    """
    result = modal(column(elements), num_modes=6 * elements)
    assert result.cumulative_ratio[-1, 0] == pytest.approx(1.0, rel=1e-12)


def test_the_participable_mass_is_smaller_than_the_structural_mass():
    """支座上那部分质量参与不了，所以两个数不可能相等。

    差多少取决于网格：柱脚那个节点的从属质量占比随剖分变细而变小。
    """
    coarse = modal(column(4), num_modes=6)
    fine = modal(column(16), num_modes=6)
    for result in (coarse, fine):
        assert result.participable_mass[0] < result.total_mass
    assert (fine.participable_mass[0] / fine.total_mass
            > coarse.participable_mass[0] / coarse.total_mass)


def test_the_effective_mass_is_the_participation_factor_squared_times_modal_mass():
    """M_eff = Γ²·(φᵀMφ)。两个量分别给出去，就必须自洽。"""
    result = modal(column(8), num_modes=10)
    expected = result.participation[:, 0] ** 2 * result.modal_mass
    assert result.effective_mass[:, 0] == pytest.approx(expected, rel=1e-10)


# --- 反应谱的基底剪力恒等式 -----------------------------------------------

@pytest.mark.gold
def test_a_constant_spectrum_gives_the_total_mass_times_the_acceleration():
    """常量谱 + 取满振型 ⇒ Σ各阶基底剪力 = 可参与质量 × Sa。

    反应谱法最有名的那条恒等式，也是检查实现最硬的判据：参与系数、
    有效质量、谱查值、缩放，任何一处差一个因子这里都对不上。
    """
    model = column(12)
    acceleration = 2.0
    result = response_spectrum(
        model, table_spectrum([(0.0, acceleration), (10.0, acceleration)]),
        direction="x", num_modes=72, combination="SRSS")
    assert result.modal_base_shear.sum() == pytest.approx(
        participable(model, 0) * acceleration, rel=1e-12)
    assert result.mass_ratio == pytest.approx(1.0, rel=1e-12)


def test_the_base_shear_from_member_forces_matches_the_modal_formula():
    """从杆端力算出来的基底剪力必须等于 M_eff·Sa。

    这验的是**另一条路径**：位移 → 杆端力。它和质量侧的公式是两套独立的
    算法，对得上才说明缩放因子没错。
    """
    model = column(12)
    result = modal(model, num_modes=12)
    axis = 0
    dominant = int(np.argmax(result.effective_mass[:, axis]))
    acceleration = 2.0
    displacement = (result.participation[dominant, axis]
                    * result.shapes[:, dominant] * acceleration
                    / result.omega[dominant] ** 2)
    forces = _member_forces(model, displacement, LoadCase("x"))
    shear = float(np.linalg.norm(forces[1][1:3]))
    assert shear == pytest.approx(
        result.effective_mass[dominant, axis] * acceleration, rel=1e-8)


def test_doubling_the_spectrum_doubles_the_response():
    """谱是线性进去的，翻倍就该整体翻倍——不然是哪里混进了常数项。"""
    model = column(8)
    single = response_spectrum(model, table_spectrum([(0, 1.0), (10, 1.0)]),
                               num_modes=12)
    double = response_spectrum(model, table_spectrum([(0, 2.0), (10, 2.0)]),
                               num_modes=12)
    assert double.base_shear == pytest.approx(2 * single.base_shear, rel=1e-10)
    assert double.displacement == pytest.approx(
        2 * single.displacement, rel=1e-10)


# --- 组合方式 -------------------------------------------------------------

def test_the_combined_response_is_never_negative():
    """SRSS 与 CQC 都是平方和开方，出来的只有大小。

    **这不是小节。** 把它当普通工况直接叠加到重力上，得到的只是两个方向里
    恰好同号的那一个；地震往复，必须自己带 ±。
    """
    result = response_spectrum(column(8),
                               table_spectrum([(0, 1.0), (10, 1.0)]),
                               num_modes=12)
    assert np.all(result.displacement >= 0)
    assert all(np.all(value >= 0) for value in result.member_forces.values())


def test_cqc_and_srss_agree_when_the_frequencies_are_well_separated():
    """频率拉得开时两种组合应当很接近——相关系数趋于单位阵。

    悬臂柱的前几阶差好几倍，正是这种情形。差得远就说明相关系数写错了。
    """
    model = column(8)
    spectrum = table_spectrum([(0, 1.0), (10, 1.0)])
    srss = response_spectrum(model, spectrum, num_modes=4, combination="SRSS")
    cqc = response_spectrum(model, spectrum, num_modes=4, combination="CQC")
    assert cqc.base_shear == pytest.approx(srss.base_shear, rel=0.05)


def test_the_correlation_matrix_is_one_on_the_diagonal():
    """同一阶与自己的相关系数必须是 1，否则组合会整体偏大或偏小。"""
    from spectrum import _cqc_correlation

    omega = np.array([1.0, 3.0, 11.0])
    rho = _cqc_correlation(omega, 0.05)
    assert np.diag(rho) == pytest.approx(np.ones(3), rel=1e-12)
    assert rho == pytest.approx(rho.T, rel=1e-12)     # 必须对称


# --- GB 50011 设计谱 ------------------------------------------------------

def test_the_design_spectrum_has_the_shape_the_code_specifies():
    """αmax=0.08、Tg=0.35、ζ=0.05 时的分段值，逐段对着规范 5.1.5 算。"""
    alpha = gb50011_spectrum(0.08, 0.35)
    # ζ=0.05 时 η₂=1、γ=0.9
    assert alpha(0.0) == pytest.approx(0.45 * 0.08)          # 起点
    assert alpha(0.1) == pytest.approx(0.08)                 # 平台起点
    assert alpha(0.2) == pytest.approx(0.08)                 # 平台
    assert alpha(0.35) == pytest.approx(0.08)                # 平台终点 Tg
    assert alpha(1.75) == pytest.approx(0.2 ** 0.9 * 0.08)   # 5Tg
    assert alpha(0.7) == pytest.approx((0.35 / 0.7) ** 0.9 * 0.08)


def test_the_design_spectrum_never_goes_negative_past_six_seconds():
    """直线下降段外推下去会变负。谱值为负没有意义，必须截住。"""
    alpha = gb50011_spectrum(0.08, 0.9)
    assert alpha(20.0) >= 0.0
    assert alpha(100.0) >= 0.0


@pytest.mark.parametrize("damping", [0.02, 0.05, 0.10])
def test_a_larger_damping_lowers_the_plateau(damping):
    """阻尼越大谱值越低——调整系数写反了这条立刻红。"""
    alpha = gb50011_spectrum(0.08, 0.35, damping=damping)
    reference = gb50011_spectrum(0.08, 0.35, damping=0.05)
    if damping > 0.05:
        assert alpha(0.2) < reference(0.2)
    elif damping < 0.05:
        assert alpha(0.2) > reference(0.2)
    else:
        assert alpha(0.2) == pytest.approx(reference(0.2))


# --- 拒绝没有意义的输入 ---------------------------------------------------

@pytest.mark.parametrize(("kwargs", "match"), [
    ({"direction": "w"}, "direction"),
    ({"combination": "ABS"}, "combination"),
    ({"damping": 0.0}, "阻尼比"),
    ({"damping": 1.5}, "阻尼比"),
])
def test_meaningless_options_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        response_spectrum(column(4), table_spectrum([(0, 1.0), (10, 1.0)]),
                          num_modes=6, **kwargs)


def test_a_one_point_spectrum_is_refused():
    with pytest.raises(ValueError, match="至少要两个点"):
        table_spectrum([(0.0, 1.0)])


def test_the_spectrum_does_not_extrapolate_outside_its_table():
    """表外取端点值。外推会在浮点边界上冒出一个表里没有的谱值。"""
    spectrum = table_spectrum([(0.5, 2.0), (1.0, 1.0)])
    assert spectrum(0.0) == pytest.approx(2.0)
    assert spectrum(99.0) == pytest.approx(1.0)
    assert spectrum(0.75) == pytest.approx(1.5)
