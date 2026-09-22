"""半刚性连接（杆端转动弹簧）。

理想铰与刚接是同一条公式的两端：静力凝聚里把 K_rr 换成 K_rr + k_s，
k_s=0 退回释放、k_s→∞ 退回刚接。**所以这两个极限是最有力的判据**——
它们各自对着一条早就存在、独立维护的代码路径。

中间段对闭合解 δ = PL³/3EI + PL²/k 验，不是"数变大了"。
"""

import numpy as np
import pytest

from frame3d import (Frame, Material, Member, Node, Section,
                     _member_matrices, _recover_end_rotations,
                     equivalent_local_load, solve)

E, NU = 2.1e11, 0.3
IZ, AREA = 4e-4, 0.02
SPAN, UDL = 6.0, -20e3


def cantilever(length=4.0, load=-30e3, **member_kwargs):
    """悬臂：节点 1 完全固定，连接弹簧在杆件 i 端。

    节点被固定死，所以杆端相对节点的转角**只能**来自连接弹簧——
    这样闭合解里就只剩连接刚度一个未知量。
    """
    f = Frame()
    f.nodes[1] = Node(1, 0, 0, 0)
    f.nodes[2] = Node(2, length, 0, 0)
    f.members[1] = Member(1, 1, 2, "S", "M", **member_kwargs)
    f.sections["S"] = Section("S", AREA, 2e-4, IZ, 1e-5)
    f.materials["M"] = Material("M", E, NU)
    f.supports[1] = (1,) * 6
    f.nodal_loads[2] = (0, 0, load, 0, 0, 0)
    return f


def two_span(**member_kwargs):
    """两跨连续梁，杆件 1 的 j 端装连接弹簧。

    节点 2 的转动刚度由杆件 2 提供，所以 k→0 不会变成机构——这才比得了
    理想释放。这也正是真实工况：梁以半刚性节点接到中柱上。
    """
    f = Frame()
    for index, x in enumerate((0.0, SPAN, 2 * SPAN), start=1):
        f.nodes[index] = Node(index, x, 0.0, 0.0)
    f.members[1] = Member(1, 1, 2, "S", "M", **member_kwargs)
    f.members[2] = Member(2, 2, 3, "S", "M")
    f.sections["S"] = Section("S", AREA, 2e-4, IZ, 1e-5)
    f.materials["M"] = Material("M", E, NU)
    f.supports[1] = (1, 1, 1, 1, 0, 0)
    f.supports[2] = (0, 1, 1, 1, 0, 0)
    f.supports[3] = (0, 1, 1, 1, 0, 0)
    for member in (1, 2):
        f.case().member_loads[member] = (0.0, 0.0, UDL)
    return f


# --- 对闭合解 -------------------------------------------------------------

@pytest.mark.gold
@pytest.mark.parametrize("stiffness", [1e12, 1e10, 1e9, 3e8, 1e8, 3e7])
def test_the_tip_deflection_matches_the_closed_form(stiffness):
    """δ = PL³/3EI + PL²/k。

    第二项就是连接转动 M/k 转过一个力臂 L，M = PL。跨五个数量级都必须成立，
    只验一个值看不出"弹簧接上了但系数错了"。
    """
    length, load = 4.0, -30e3
    model = cantilever(length, load, springs_i={"rz": stiffness})
    tip = solve(model).U[model.node_dofs(2)[2]]
    expected = (load * length**3 / (3 * E * IZ)
                + load * length**2 / stiffness)
    assert tip == pytest.approx(expected, rel=1e-10)


def test_without_a_connection_the_answer_is_the_textbook_cantilever():
    """不装弹簧时必须还是 PL³/3EI，一个字都不能差。"""
    length, load = 4.0, -30e3
    model = cantilever(length, load)
    tip = solve(model).U[model.node_dofs(2)[2]]
    assert tip == pytest.approx(load * length**3 / (3 * E * IZ), rel=1e-12)


def test_a_rigid_two_span_beam_gives_the_classic_support_moment():
    """刚接时中支座弯矩是 wL²/8——先确认基准是对的，再谈半刚性。"""
    forces = solve(two_span()).cases["default"].member_forces[1]
    assert forces[11] == pytest.approx(UDL * SPAN**2 / 8, rel=1e-10)


# --- 两个极限：各自对着一条独立的既有路径 ---------------------------------

@pytest.mark.parametrize("stiffness", [1e3, 1.0, 1e-3])
def test_a_vanishing_connection_converges_to_the_ideal_hinge(stiffness):
    """k→0 必须收敛到杆端释放——那是另一条早就存在的代码路径。"""
    released = solve(two_span(releases_j=("rz",)))
    soft = solve(two_span(springs_j={"rz": stiffness}))
    # 收敛是一阶的：k 小十倍，差距小十倍
    assert np.max(np.abs(released.U - soft.U)) < 1e-6 * stiffness + 1e-12


@pytest.mark.parametrize("stiffness", [1e12, 1e15, 1e18])
def test_a_huge_connection_converges_to_the_rigid_joint(stiffness):
    """k→∞ 必须收敛到刚接。"""
    rigid = solve(two_span())
    stiff = solve(two_span(springs_j={"rz": stiffness}))
    scale = np.max(np.abs(rigid.U))
    assert np.max(np.abs(rigid.U - stiff.U)) < 1e-3 * scale * (1e12 / stiffness)


# --- 内力回算 -------------------------------------------------------------

@pytest.mark.parametrize("stiffness", [1e9, 1e8, 3e7])
def test_the_end_moment_equals_the_spring_stiffness_times_its_rotation(stiffness):
    """杆端弯矩必须等于 k × 弹簧转角。

    **这条抓到过一个真 bug。** 位移还原原本套用的是理想铰那个式子，
    它默认节点不转；半刚性下杆端转角是「节点转角 + 弹簧变形」，少掉前一项
    让两跨连续梁的支座弯矩差了 32%，而位移解本身是对的——
    也就是说错误只出现在内力那一栏。
    """
    model = two_span(springs_j={"rz": stiffness})
    solution = solve(model)
    member = model.members[1]
    moment = solution.cases["default"].member_forces[1][11]

    length, rot, transform, offset, _, _, cond = _member_matrices(model, member)
    dofs = model.node_dofs(member.i) + model.node_dofs(member.j)
    node_local = transform @ offset @ solution.U[dofs]
    load = equivalent_local_load(model, member, model.case(), length, rot)
    beam_local = _recover_end_rotations(node_local, load, cond)
    twist = beam_local[11] - node_local[11]
    assert moment == pytest.approx(-stiffness * twist, rel=1e-9)


@pytest.mark.parametrize("stiffness", [1e18, 1e9, 1e8, 3e7])
def test_the_node_is_in_rotational_equilibrium(stiffness):
    """节点 2 的转动无约束、无外加力矩，两根杆交给它的弯矩必须抵消。

    这条不依赖任何符号约定，所以它是最硬的判据：弹簧传出去的弯矩要是算错了，
    这里立刻就不平衡——实测错误版本差 17 kN·m。
    """
    solution = solve(two_span(springs_j={"rz": stiffness}))
    forces = solution.cases["default"].member_forces
    total = forces[1][11] + forces[2][5]
    assert abs(total) < 1e-6 * abs(forces[1][11])


def test_a_softer_connection_moves_moment_from_the_support_to_midspan():
    """连接越柔，支座弯矩越小——单调，而且刚接那端收敛到 wL²/8。"""
    moments = [abs(solve(two_span(springs_j={"rz": k}))
                   .cases["default"].member_forces[1][11])
               for k in (1e18, 1e9, 1e8, 3e7, 1e6)]
    assert moments == sorted(moments, reverse=True)
    assert moments[0] == pytest.approx(abs(UDL) * SPAN**2 / 8, rel=1e-6)
    assert moments[-1] < 0.1 * moments[0]


# --- 自相矛盾的输入要被拒绝 -----------------------------------------------

def test_releasing_and_stiffening_the_same_dof_is_refused():
    """同一个自由度既释放又给刚度是自相矛盾的，不替用户猜。"""
    member = Member(1, 1, 2, "S", "M", releases_j=("rz",),
                    springs_j={"rz": 1e8})
    with pytest.raises(ValueError, match="既释放又给了连接刚度"):
        member.spring_indices()


@pytest.mark.parametrize("stiffness", [0.0, -1.0])
def test_a_non_positive_connection_stiffness_is_refused(stiffness):
    """0 会被读成"刚度为零即铰接"，而那该用 releases 明确写。"""
    member = Member(1, 1, 2, "S", "M", springs_j={"rz": stiffness})
    with pytest.raises(ValueError, match="不是正数"):
        member.spring_indices()


# --- 通过编译器与会话层 ---------------------------------------------------

def test_the_connection_survives_automatic_subdivision():
    """杆件被自动剖分时，连接弹簧只能留在**整根构件的两端**。

    每段都来一个的话，一根梁上会凭空多出几个半刚性节点——弯矩分布整个变样，
    而且不会报任何错。
    """
    from model_compiler import compile_model

    payload = {
        "schema_version": 1, "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": SPAN, "y": 0, "z": 0}],
        "materials": [{"name": "M", "E": E, "nu": NU}],
        "sections": [{"name": "S", "A": AREA, "Iy": 2e-4, "Iz": IZ, "J": 1e-5}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M",
                     "connections": {"j": {"rz": 1e8}}}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        # 跨中集中力会触发剖分
        "load_cases": [{"name": "default", "member_spans": [
            {"member": 1, "kind": "point", "w1": [0, 0, -30e3], "a": SPAN / 2}]}],
    }
    frame = compile_model(payload).analysis_model
    elements = sorted(frame.members)
    assert len(elements) > 1, "这个模型本来就该被剖分，否则测不到东西"
    sprung = {eid: frame.members[eid].springs_j for eid in elements}
    assert sprung[elements[-1]] == {"rz": 1e8}, "末段必须带着 j 端的连接"
    assert all(not sprung[eid] for eid in elements[:-1]), (
        f"中间段不该有连接弹簧：{sprung}")


def test_the_session_tool_scales_stiffness_by_the_beam_line_stiffness():
    """按 EI/L 的倍数给刚度时，换算出来的绝对值必须对。

    绝对刚度离开截面和跨度没有意义——同一个端板装在不同梁上相对刚度差几倍，
    所以这个换算错了，"端板连接"这四个字就没有意义了。
    """
    from agent import Session

    session = Session()
    session.add_nodes([[0, 0, 0], [SPAN, 0, 0]])
    session.define_materials_and_sections(
        materials=[{"name": "M", "E": E, "nu": NU}],
        sections=[{"name": "S", "A": AREA, "Iy": 2e-4, "Iz": IZ, "J": 1e-5}])
    session.add_members([[1, 2]], "S", "M")
    session.set_supports([1], fix=[1, 1, 1, 1, 1, 1])

    result = session.set_member_connection([1], end="j",
                                           connection_type="顶底角钢")
    assert result.ok, result.payload
    assert result.payload["stiffness"][1] == pytest.approx(
        3.0 * E * IZ / SPAN)


def test_a_load_case_the_user_named_default_is_not_dropped():
    """用户自己把工况起名叫 default 时，它不能被当成空壳清掉。

    自动建出来的那个空 default 和用户这条是同一个对象。一并删掉的话，
    整条工况连同荷载一起消失，而模型照样读得进来——接下来报的是
    "求解时没有任何荷载"，指向的却不是真正的原因。
    """
    from model_io import from_dict

    payload = {
        "schema_version": 1, "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": SPAN, "y": 0, "z": 0}],
        "materials": [{"name": "M", "E": E, "nu": NU}],
        "sections": [{"name": "S", "A": AREA, "Iy": 2e-4, "Iz": IZ, "J": 1e-5}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "load_cases": [{"name": "default",
                        "member_loads": [{"member": 1, "w": [0, 0, UDL]}]}],
    }
    frame = from_dict(payload)
    assert list(frame.load_cases) == ["default"]
    assert frame.load_cases["default"].member_loads[1][2] == pytest.approx(UDL)
