"""初应变：装配内力与温度应力（讲义 §3-9 五、六）。

讲义把温度应力"转化为装配内力问题"——两者是同一个机制：杆件有一个初始
长度变化 Δl，转成沿杆轴的等效节点力 ``EA·ε₀`` 进入右端项，回算内力时再减掉。

这三个算例是闭式解，不是回归快照：

* 两端固定 + 升温 → 位移恒为零，``N = −EA·α·ΔT``（受压）
* 一端自由 + 升温 → ``N = 0``，自由伸长 ``α·ΔT·L``
* 两端固定 + 装配误差 Δl → ``N = EA·Δl/L``

**第二条最容易错**：如果回算内力时忘了减初应变项，自由伸长的杆会凭空出现
轴力，而且方程照样有解、不报任何错。
"""

from __future__ import annotations

import numpy as np
import pytest

from frame3d import (Frame, LoadCase, Material, Member, Node, Section, solve)


E = 2.06e11          # Pa
AREA = 3.0e-3        # m²
ALPHA = 1.2e-5       # 1/℃
LENGTH = 4.0         # m


def _bar(*, fix_far_end: bool, strain: float) -> Frame:
    """沿 x 的单根杆。近端总是固定；远端按需固定或自由。"""
    frame = Frame(
        nodes={1: Node(1, 0.0, 0.0, 0.0), 2: Node(2, LENGTH, 0.0, 0.0)},
        members={1: Member(1, 1, 2, "S", "M")},
        sections={"S": Section("S", A=AREA, Iy=1e-5, Iz=1e-5, J=2e-5)},
        materials={"M": Material("M", E=E, nu=0.3, alpha=ALPHA)},
        supports={1: (1, 1, 1, 1, 1, 1)},
        load_cases={"T": LoadCase("T", member_strains={1: strain})},
    )
    if fix_far_end:
        frame.supports[2] = (1, 1, 1, 1, 1, 1)
    else:
        # 远端只锁住横向与转动，轴向放开，这样它能自由伸长
        frame.supports[2] = (0, 1, 1, 1, 1, 1)
    return frame


def _axial(solution, case: str = "T") -> float:
    """杆件轴力，受拉为正。

    member_forces[0] 是 i 端沿局部 x 的杆端力；受拉时它指向 −x，
    所以取负号才是习惯意义上的轴力。
    """
    return -float(solution.cases[case].member_forces[1][0])


def test_a_fully_restrained_bar_heated_goes_into_compression():
    """两端固定升温：位移恒为零，轴力精确等于 −EA·α·ΔT。

    这是温度应力最经典的闭式解，也是讲义 §3-9 六的直接对应。
    """
    delta_t = 40.0
    frame = _bar(fix_far_end=True, strain=ALPHA * delta_t)
    solution = solve(frame)

    assert np.allclose(solution.cases["T"].U, 0.0, atol=1e-14)
    expected = -E * AREA * ALPHA * delta_t
    assert _axial(solution) == pytest.approx(expected, rel=1e-12)


def test_a_free_bar_heated_carries_no_axial_force_at_all():
    """一端自由升温：自由伸长 α·ΔT·L，轴力精确为零。

    **这条是整套实现最容易错的地方。** 回算内力时若忘了把初应变减掉，
    这里会凭空出现 EA·α·ΔT 的轴力——而方程照样有解，不报任何错。
    """
    delta_t = 40.0
    frame = _bar(fix_far_end=False, strain=ALPHA * delta_t)
    solution = solve(frame)

    case = solution.cases["T"]
    tip_x = case.U[frame.node_dofs(2)[0]]
    assert tip_x == pytest.approx(ALPHA * delta_t * LENGTH, rel=1e-12)
    assert _axial(solution) == pytest.approx(0.0, abs=1e-6)


def test_cooling_a_restrained_bar_puts_it_into_tension():
    """降温号相反：受拉。符号错了这条会立刻失败。"""
    frame = _bar(fix_far_end=True, strain=ALPHA * -40.0)
    assert _axial(solve(frame)) == pytest.approx(
        E * AREA * ALPHA * 40.0, rel=1e-12)


def test_a_lack_of_fit_bar_forced_into_place_carries_EA_dl_over_L():
    """装配内力（讲义 §3-9 五）：杆短了 Δl，强行装上后受拉 EA·Δl/L。

    杆做短了 Δl，相当于初应变 ε₀ = −Δl/L（它想更短）；两端固定后必须被
    拉长 Δl 才装得进去，因此受拉。
    """
    shortfall = 0.002                      # 杆比设计短 2 mm
    frame = _bar(fix_far_end=True, strain=-shortfall / LENGTH)
    solution = solve(frame)

    assert _axial(solution) == pytest.approx(
        E * AREA * shortfall / LENGTH, rel=1e-12)


def test_thermal_and_lack_of_fit_are_the_same_mechanism():
    """讲义说温度应力"转化为装配内力问题"——那么等价的两组输入必须给出
    逐位相同的结果，而不只是接近。"""
    delta_t = 40.0
    equivalent_dl = ALPHA * delta_t * LENGTH

    by_temperature = _axial(solve(_bar(fix_far_end=True,
                                       strain=ALPHA * delta_t)))
    by_assembly = _axial(solve(_bar(fix_far_end=True,
                                    strain=equivalent_dl / LENGTH)))
    assert by_temperature == pytest.approx(by_assembly, rel=1e-14)


def test_initial_strain_superposes_with_an_ordinary_load():
    """线性问题里初应变与外荷载必须可叠加。

    分别算再相加，与一次算完，结果要一致——不一致说明初应变漏进了某条
    只走一次的路径。
    """
    delta_t, tip_force = 40.0, 5.0e4

    both = Frame(
        nodes={1: Node(1, 0.0, 0.0, 0.0), 2: Node(2, LENGTH, 0.0, 0.0)},
        members={1: Member(1, 1, 2, "S", "M")},
        sections={"S": Section("S", A=AREA, Iy=1e-5, Iz=1e-5, J=2e-5)},
        materials={"M": Material("M", E=E, nu=0.3, alpha=ALPHA)},
        supports={1: (1, 1, 1, 1, 1, 1), 2: (0, 1, 1, 1, 1, 1)},
        load_cases={"T": LoadCase(
            "T", nodal_loads={2: (tip_force, 0, 0, 0, 0, 0)},
            member_strains={1: ALPHA * delta_t})},
    )
    only_load = Frame(
        nodes=both.nodes, members=both.members, sections=both.sections,
        materials=both.materials, supports=both.supports,
        load_cases={"T": LoadCase("T",
                                  nodal_loads={2: (tip_force, 0, 0, 0, 0, 0)})},
    )
    only_strain = _bar(fix_far_end=False, strain=ALPHA * delta_t)

    dof = both.node_dofs(2)[0]
    assert (solve(both).cases["T"].U[dof] == pytest.approx(
        solve(only_load).cases["T"].U[dof]
        + solve(only_strain).cases["T"].U[dof], rel=1e-12))


def test_a_zero_strain_case_is_bit_for_bit_the_old_behaviour():
    """没有初应变的模型必须与加这个特性之前完全一样。

    新特性最常见的伤害不是算错新东西，是悄悄改了旧东西。
    """
    frame = _bar(fix_far_end=True, strain=0.0)
    frame.load_cases["T"].member_strains.clear()
    solution = solve(frame)
    assert np.allclose(solution.cases["T"].U, 0.0, atol=1e-16)
    assert _axial(solution) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------- 接到用户能用的层

def test_the_json_model_carries_temperature_end_to_end():
    """从 JSON 模型到轴力全链路。材料给 alpha，工况给 delta_t。"""
    from frame3d import solve
    from model_io import from_dict, validate_payload

    model = {
        "units": "N-m-Pa",
        "materials": [{"name": "Q355", "E": E, "nu": 0.3, "alpha": ALPHA}],
        "sections": [{"name": "S", "A": AREA, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": LENGTH, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "Q355"}],
        "supports": [{"node": 1, "fix": [1] * 6}, {"node": 2, "fix": [1] * 6}],
        "load_cases": [{"name": "T",
                        "member_strains": [{"member": 1, "delta_t": 40.0}]}],
    }
    assert validate_payload(model) == []
    frame = from_dict(model)
    assert frame.materials["Q355"].alpha == ALPHA
    assert _axial(solve(frame)) == pytest.approx(-E * AREA * ALPHA * 40.0, rel=1e-12)


def test_lack_of_fit_in_json_is_divided_by_the_real_member_length():
    """Δl 是长度，必须除以杆件真实长度才变成应变。
    写死一个长度或者忘了除，短杆和长杆会得到同样的内力。"""
    from frame3d import solve
    from model_io import from_dict

    def axial_for(span: float) -> float:
        return _axial(solve(from_dict({
            "units": "N-m-Pa",
            "materials": [{"name": "M", "E": E, "nu": 0.3}],
            "sections": [{"name": "S", "A": AREA, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}],
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                      {"id": 2, "x": span, "y": 0, "z": 0}],
            "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"}],
            "supports": [{"node": 1, "fix": [1] * 6}, {"node": 2, "fix": [1] * 6}],
            "load_cases": [{"name": "A", "member_strains": [
                {"member": 1, "lack_of_fit": -0.002}]}],
        })), "A")

    # 同样做短 2 mm，杆越短内力越大，且精确等于 EA·Δl/L
    assert axial_for(4.0) == pytest.approx(E * AREA * 0.002 / 4.0, rel=1e-12)
    assert axial_for(2.0) == pytest.approx(E * AREA * 0.002 / 2.0, rel=1e-12)


def test_temperature_survives_a_unit_system_change_but_length_error_is_converted():
    """换单位制时 lack_of_fit 是长度要换算、delta_t 是温度不能换算。

    同一条目里两种量纲，漏掉任一边都不报错，只会静默算错——
    这个项目以前在截面的 cy/cz 上正是这么栽过一次。
    """
    from agent import Session
    import sections as S

    session = Session()
    assert session.define_materials_and_sections(
        [{"name": "M", "E": E, "nu": 0.3, "alpha": ALPHA}],
        [S.circular_tube("P", 0.219, 0.008)]).ok
    assert session.add_nodes([[0, 0, 0], [LENGTH, 0, 0]]).ok
    assert session.add_members([[1, 2]], section="P", material="M").ok
    assert session.set_supports([1, 2], fix=[1] * 6).ok
    assert session.set_member_strain(1, lack_of_fit=-0.002, delta_t=40.0).ok

    before = _entry(session.model)
    assert session.set_units("N-mm-MPa").ok
    after = _entry(session.model)

    assert after["lack_of_fit"] == pytest.approx(before["lack_of_fit"] * 1000.0)
    assert after["delta_t"] == pytest.approx(before["delta_t"]), "温度不该被换算"


def _entry(model) -> dict:
    for case in model.get("load_cases") or [model]:
        for item in case.get("member_strains") or []:
            return item
    raise AssertionError("模型里没有初应变条目")


def test_deleting_a_member_takes_its_initial_strain_with_it():
    """删杆件必须把它的初应变一起清掉，否则留下引用不存在杆件的孤儿条目。"""
    from agent import Session
    import sections as S

    session = Session()
    assert session.define_materials_and_sections(
        [{"name": "M", "E": E, "nu": 0.3, "alpha": ALPHA}],
        [S.circular_tube("P", 0.219, 0.008)]).ok
    assert session.add_nodes([[0, 0, 0], [LENGTH, 0, 0], [2 * LENGTH, 0, 0]]).ok
    assert session.add_members([[1, 2], [2, 3]], section="P", material="M").ok
    assert session.set_member_strain(2, delta_t=40.0).ok

    session._applying_preview = True          # 跳过删除确认闸门
    assert session.remove_members([2]).ok
    leftover = [item for case in (session.model.get("load_cases") or [session.model])
                for item in (case.get("member_strains") or [])]
    assert leftover == [], f"删完还剩下孤儿初应变：{leftover}"


def test_temperature_without_alpha_is_refused_with_a_usable_hint():
    """材料没给线膨胀系数就算温度应力，等于默默按 α=0 算出"没有温度应力"。
    宁可拒绝并告诉用户去哪儿补。"""
    from agent import Session
    import sections as S

    session = Session()
    assert session.define_materials_and_sections(
        [{"name": "M", "E": E, "nu": 0.3}],       # 没有 alpha
        [S.circular_tube("P", 0.219, 0.008)]).ok
    assert session.add_nodes([[0, 0, 0], [LENGTH, 0, 0]]).ok
    assert session.add_members([[1, 2]], section="P", material="M").ok

    result = session.set_member_strain(1, delta_t=40.0)
    assert not result.ok
    assert "alpha" in result.payload["error"]
    assert "1.2e-5" in result.payload["hint"]


def test_a_split_member_passes_the_same_strain_to_every_segment():
    """初应变沿杆是常量，杆件被自动剖分时各段原样继承，不按段长分配。"""
    from model_compiler import compile_model

    compiled = compile_model({
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": E, "nu": 0.3, "alpha": ALPHA}],
        "sections": [{"name": "S", "A": AREA, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6.0, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"}],
        "supports": [{"node": 1, "fix": [1] * 6}, {"node": 2, "fix": [1] * 6}],
        "load_cases": [{"name": "T", "member_strains": [
            {"member": 1, "delta_t": 40.0}],
            "member_spans": [{"member": 1, "kind": "point",
                              "w1": [0, 0, -1000.0], "a": 2.0}]}],
    })
    elements = compiled.mapping.element_ids(1)
    assert len(elements) > 1, "集中力应当触发剖分，否则这条测不到东西"
    strains = compiled.analysis_model.load_cases["T"].member_strains
    assert set(strains) == set(elements), "剖分出来的段没有全部继承初应变"
    # 各段拿到同一个应变值，而不是按段长分掉的一部分
    for element_id, value in strains.items():
        assert value == pytest.approx(ALPHA * 40.0), f"段 {element_id} 的应变不对" 
