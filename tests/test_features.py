"""建模对象层：截面指派、边界条件、荷载作为有名字的对象施加在区域上。

防的问题：原 Domain IR 只有逐节点、逐杆件的条目——集合在写入时就展开成编号，
之后改集合，约束和荷载不跟着变；对一串杆件加荷载会被拆成「名字-杆号」的若干条，
没有一个「荷载对象」可以编辑、抑制或删除。Abaqus 的对象体系做不出来。

核心判据：用对象建的模型与用旧工具逐条建的模型，求解结果逐位相同——对象层只改
组织方式，不改力学。
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from agent import Session
from model_io import validate_payload
from units import convert_model


def _geometry() -> Session:
    """两跨门架：柱 1、2、3，梁 4、5；柱脚节点 1、3、5。"""
    s = Session()
    assert s.add_nodes([[0, 0, 0], [0, 0, 4], [6, 0, 0], [6, 0, 4],
                        [12, 0, 0], [12, 0, 4]]).ok
    assert s.define_materials_and_sections(
        [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}],
        [{"name": "H", "A": 8e-3, "Iy": 2e-5, "Iz": 1.5e-4, "J": 5e-7},
         {"name": "B", "A": 6e-3, "Iy": 1e-5, "Iz": 9e-5, "J": 3e-7}]).ok
    assert s.add_members([[1, 2], [3, 4], [5, 6], [2, 4], [4, 6]]).ok
    assert s.define_set(name="柱", member_ids=[1, 2, 3]).ok
    assert s.define_set(name="梁", member_ids=[4, 5]).ok
    assert s.define_set(name="柱脚", node_ids=[1, 3, 5]).ok
    return s


def _case(s: Session) -> None:
    # 现有 set_load_cases 要求模型已完整（有支座）才能建工况——这是 Load 模块
    # 重做时要去掉的限制；这里先按「先约束、后工况」的顺序建。
    assert s.set_load_cases([{"name": "DL"}]).ok


def _with_objects() -> Session:
    s = _geometry()
    assert s.create_section_assignment("SA-柱", "柱", "H", "Q355").ok
    assert s.create_section_assignment("SA-梁", "梁", "B", "Q355").ok
    assert s.create_boundary_condition("BC-柱脚", "柱脚", preset="ENCASTRE").ok
    _case(s)
    assert s.create_load("楼面", "DL", "line_load", "梁", w=[0, 0, -2e4]).ok
    assert s.create_load("风", "DL", "concentrated_force", [2], force=[5e3, 0, 0]).ok
    return s


def _with_direct_tools() -> Session:
    s = _geometry()
    assert s.assign_properties([1, 2, 3], "H", "Q355").ok
    assert s.assign_properties([4, 5], "B", "Q355").ok
    assert s.set_supports([1, 3, 5], fix=[1] * 6).ok
    _case(s)
    assert s.set_member_load([4, 5], [0, 0, -2e4], case_name="DL").ok
    assert s.set_nodal_load(2, [5e3, 0, 0, 0, 0, 0], case_name="DL").ok
    return s


def _u(session: Session) -> np.ndarray:
    assert session.solve_model().ok
    return session.solution.cases["DL"].U


def test_objects_give_exactly_the_same_answer_as_direct_entries():
    np.testing.assert_allclose(_u(_with_objects()), _u(_with_direct_tools()),
                               rtol=1e-12, atol=1e-15)


def test_a_load_on_a_region_is_one_object_not_one_entry_per_member():
    s = _with_objects()
    listed = s.list_features("load").payload["features"]["loads"]
    assert [f["name"] for f in listed] == ["楼面", "风"]
    entries = [e for e in s.model["load_cases"][0]["member_loads"] if e.get("owner") == "楼面"]
    assert [e["member"] for e in entries] == [4, 5]
    assert all("name" not in e for e in entries), "派生条目不再各自起名"


def test_a_set_region_is_a_live_reference():
    """改集合，引用它的截面指派、边界条件与荷载随之更新——Abaqus 的行为。"""
    s = _with_objects()
    assert s.add_nodes([[18, 0, 0], [18, 0, 4]]).ok
    assert s.add_members([[7, 8], [6, 8]]).ok
    assert s.define_set(name="柱", member_ids=[1, 2, 3, 6]).ok
    assert s.define_set(name="梁", member_ids=[4, 5, 7]).ok
    assert s.define_set(name="柱脚", node_ids=[1, 3, 5, 7]).ok
    members = {m["id"]: m for m in s.model["members"]}
    assert members[6]["section"] == "H" and members[6]["assignment"] == "SA-柱"
    assert members[7]["section"] == "B"
    assert sorted(int(x["node"]) for x in s.model["supports"]) == [1, 3, 5, 7]
    loaded = {e["member"] for e in s.model["load_cases"][0]["member_loads"]}
    assert loaded == {4, 5, 7}
    assert not validate_payload(s.model)


def test_two_bcs_may_share_a_node_on_different_dofs_but_not_the_same_dof():
    s = _geometry()
    assert s.create_boundary_condition("BC-平动", "柱脚", fix=[1, 1, 1, 0, 0, 0]).ok
    assert s.create_boundary_condition("BC-转动", "柱脚", fix=[0, 0, 0, 1, 1, 1]).ok
    entry = next(x for x in s.model["supports"] if x["node"] == 1)
    assert entry["fix"] == [1] * 6 and entry["owners"] == ["BC-平动", "BC-转动"]
    before = deepcopy(s.model)
    got = s.create_boundary_condition("BC-重复", [1], fix=[1, 0, 0, 0, 0, 0])
    assert not got.ok and any("重复约束" in e for e in got.payload["errors"])
    assert s.model == before


def test_a_bc_object_cannot_silently_override_a_direct_support():
    s = _geometry()
    assert s.set_supports([1], fix=[1] * 6).ok
    got = s.create_boundary_condition("BC-柱脚", "柱脚", preset="PINNED")
    assert not got.ok and any("直接定义的支座" in e for e in got.payload["errors"])


def test_overlapping_section_assignments_are_rejected():
    s = _geometry()
    assert s.create_section_assignment("SA-柱", "柱", "H", "Q355").ok
    got = s.create_section_assignment("SA-1号", [1], "B", "Q355")
    assert not got.ok and any("重复指派" in e for e in got.payload["errors"])


def test_suppress_resume_rename_copy_delete():
    s = _with_objects()
    assert s.suppress_feature("load", "楼面").ok
    assert not any(e.get("owner") == "楼面" for e in s.model["load_cases"][0]["member_loads"])
    assert s.suppress_feature("load", "楼面", suppressed=False).ok
    assert any(e.get("owner") == "楼面" for e in s.model["load_cases"][0]["member_loads"])

    assert s.rename_feature("bc", "BC-柱脚", "固接柱脚").ok
    assert {x["name"] for x in s.model["supports"]} == {"固接柱脚"}
    # 边界条件副本默认抑制，否则与原对象在同一区域必然过约束
    copied = s.copy_feature("bc", "固接柱脚", "固接柱脚-副本")
    assert copied.ok
    assert s.list_features("bc").payload["features"]["bcs"][1]["suppressed"] is True

    assert s.delete_feature("load", "风").ok
    assert not s.model["load_cases"][0].get("nodal_loads")
    assert not s.delete_feature("load", "风").ok


def test_removing_a_section_assignment_leaves_members_unassigned_not_stale():
    s = _with_objects()
    assert s.delete_feature("section_assignment", "SA-梁").ok
    members = {m["id"]: m for m in s.model["members"]}
    assert members[4]["section"] == "" and "assignment" not in members[4]
    assert any("截面" in e for e in s.validation_errors()), "未指派的杆件不能直接求解"


def test_every_object_operation_is_one_undo_step():
    s = _geometry()
    before = deepcopy(s.model)
    assert s.create_boundary_condition("BC-柱脚", "柱脚", preset="ENCASTRE").ok
    assert s.undo()
    assert s.model == before


def test_objects_survive_the_unit_conversion():
    si = _with_objects()
    mm = Session(model=convert_model(deepcopy(si.model), "N-mm-MPa"))
    assert not validate_payload(mm.model), "换算后对象与派生条目仍一致"
    u_si, u_mm = _u(si), _u(mm)
    k = si.frame.index_of()[2]
    assert u_mm[6 * k] / 1e3 == pytest.approx(u_si[6 * k], rel=1e-9)


def test_deleting_members_trims_explicit_regions_and_a_hand_edit_is_caught():
    s = _with_objects()
    assert s.create_load("端部", "DL", "line_load", [5], w=[0, 0, -1e3]).ok
    assert s.remove_members([5]).ok
    region = next(f for f in s.model["features"]["loads"] if f["name"] == "端部")["region"]
    assert region == {"members": []}
    assert any("端部" in e for e in s.validation_errors())

    edited = _with_objects()
    edited.model["supports"][0]["fix"] = [0] * 6
    assert any("不一致" in e for e in validate_payload(edited.model))


def test_gravity_object_matches_add_self_weight():
    a = _with_direct_tools()
    assert a.add_self_weight(case="DL").ok
    b = _with_direct_tools()
    assert b.create_load("自重", "DL", "gravity", "梁", g=[0, 0, -9.80665]).ok
    assert b.edit_feature("load", "自重", {"region": {"members": [1, 2, 3, 4, 5]}}).ok
    np.testing.assert_allclose(_u(b), _u(a), rtol=1e-9, atol=1e-15)


def test_beam_orientation_object_matches_direct_reference_vectors():
    """梁方向对象与逐杆写参考向量求解结果逐位一致；集合区域是活引用。"""
    a = _with_objects()
    assert a.define_set(name="柱", member_ids=[1, 2, 3]).ok
    assert a.create_beam_orientation("Ori-柱", "柱", [0, 1, 0]).ok
    assert {m["id"]: m.get("ref_vector") for m in a.model["members"]}[1] == [0.0, 1.0, 0.0]
    b = _with_objects()
    for mid in (1, 2, 3):
        assert b.edit_member(mid, ref_vector=[0, 1, 0]).ok
    np.testing.assert_allclose(_u(a), _u(b), rtol=1e-12, atol=1e-15)

    assert a.delete_feature("beam_orientation", "Ori-柱").ok
    assert all("ref_vector" not in m for m in a.model["members"] if m["id"] in (1, 2, 3))


def test_a_beam_orientation_parallel_to_a_member_is_rejected_with_its_id():
    """n1 与杆轴平行时截面朝向无法确定；原来要到求解时才以难懂的形式报出。"""
    s = _geometry()
    got = s.create_beam_orientation("Ori-错", "柱", [0, 0, 1])     # 柱沿 Z
    assert not got.ok
    assert any("杆件 1" in e and "平行" in e for e in got.payload["errors"])


def test_load_cases_can_be_defined_before_any_support():
    """原来 set_load_cases 拿整个模型去校验，没有支座就建不了工况，建模顺序被定死；
    Abaqus 允许先定义荷载、后加边界条件。"""
    s = _geometry()
    got = s.set_load_cases([{"name": "DL"}, {"name": "LL"}])
    assert got.ok, got.payload
    assert s.create_load("楼面", "LL", "line_load", "梁", w=[0, 0, -1e3]).ok


def test_manager_step_matrix_shows_created_propagated_and_active():
    """管理器按分析步分列（Abaqus 的 Created / Propagated），传播关系一眼可见。"""
    from desktop.object_dialogs import step_matrix
    s = _with_objects()
    assert s.add_step("S1", loads={"DL": "RAMP"}).ok
    assert s.add_step("S2").ok
    bcs = s.list_features("bc").payload["features"]["bcs"]
    names, table = step_matrix(s.model, "bcs", bcs)
    assert names == ["Initial", "S1", "S2"]
    assert table["BC-柱脚"] == ["创建", "传播", "传播"]
    loads = s.list_features("load").payload["features"]["loads"]
    _, table = step_matrix(s.model, "loads", loads)
    assert table["楼面"] == ["", "生效（RAMP）", "生效（RAMP）"]


def test_perturbation_steps_do_not_change_the_base_state():
    """频率步、屈曲步是 Abaqus 的线性摄动步：本步声明的荷载与支座改写不传播。"""
    import steps as step_module
    s = _with_objects()
    assert s.add_step("S1", loads={"DL": "RAMP"}).ok
    assert s.add_step("B1", analysis="buckle", loads={"DL": "STEP"}, num_modes=2).ok
    assert s.add_step("F1", analysis="frequency", num_modes=3).ok
    assert s.add_step("S2").ok
    eff = {e.name: e for e in step_module.resolve(step_module.from_payload(s.model), s.model)}
    assert eff["B1"].loads == {"DL": "STEP"}
    assert eff["F1"].loads == {}
    assert eff["S2"].loads == {"DL": "RAMP"}, "屈曲步的 STEP 幅值不能传播到后面的一般步"

    bad = _with_objects()
    assert bad.add_step("S1", loads={"DL": "RAMP"}).ok
    refused = bad.add_step("B1", analysis="buckle")
    assert not refused.ok and "参考荷载" in refused.payload["error"], "屈曲步必须声明参考荷载"


def test_frequency_and_buckle_steps_match_the_standalone_analyses():
    s = _with_objects()
    assert s.add_step("S1", loads={"DL": "RAMP"}).ok
    assert s.add_step("F1", analysis="frequency", num_modes=3).ok
    assert s.add_step("B1", analysis="buckle", loads={"DL": "RAMP"}, num_modes=2).ok
    got = s.solve_steps()
    assert got.ok, got.payload
    rows = {r["step"]: r for r in got.payload["steps"]}
    assert got.payload["inspecting"] == "S1", "后处理留一般步的结果"

    ref = _with_objects()
    modes = ref.modal_analysis(num_modes=3).payload["modes"]
    assert rows["F1"]["frequencies_Hz"] == pytest.approx([m["frequency_Hz"] for m in modes],
                                                         rel=1e-9)
    assert ref.solve_model().ok
    lam = ref.buckling_analysis(case="DL", num_modes=2).payload["critical_factor"]
    assert rows["B1"]["critical_factor"] == pytest.approx(lam, rel=1e-9)
    assert not s.solve_steps(inspect="F1").ok


def test_partition_adds_a_vertex_without_changing_the_answer():
    """分割杆件（Partition Edge）只加顶点：物理杆件仍一根，求解结果不变；新顶点可连杆件。"""
    a, b = _with_objects(), _with_objects()
    got = b.partition_member(4, ratio=0.25)
    assert got.ok, got.payload
    node = got.payload["node"]
    assert len(b.model["members"]) == len(a.model["members"]), "物理杆件不增加"
    ua, ub = _u(a), _u(b)
    for nid in (1, 2, 3, 4, 5, 6):
        ka, kb = a.frame.index_of()[nid], b.frame.index_of()[nid]
        np.testing.assert_allclose(ub[6 * kb:6 * kb + 6], ua[6 * ka:6 * ka + 6],
                                   rtol=1e-9, atol=1e-14)
    assert b.add_nodes([[1.5, 0, 0]]).ok
    assert b.add_members([[node, max(n["id"] for n in b.model["nodes"])]], "H", "Q355").ok
    assert not b.partition_member(4, ratio=1.0).ok


def test_datum_points_and_measure():
    s = _geometry()
    assert s.create_datum_point("中点", midpoint_of=[1, 6]).ok
    assert s.datum_points()["中点"] == pytest.approx([6.0, 0.0, 2.0])
    assert s.create_datum_point("偏移", offset_from=2, offset=[0, 0, 1]).ok
    assert not s.create_datum_point("错", xyz=[0, 0, 0], midpoint_of=[1, 2]).ok
    got = s.measure(nodes=[1, 6]).payload
    assert got["distance"] == pytest.approx((12 ** 2 + 4 ** 2) ** 0.5)
    assert s.measure(members=[1, 4]).payload["angle_deg"] == pytest.approx(90.0)
    assert s.delete_datum_point("偏移").ok and "偏移" not in s.datum_points()


def test_rigid_arm_constraint_equals_force_plus_moment_at_the_master():
    """MPC BEAM 把加在偏心点上的力原样传到主节点：等于主节点上的力加 d×F 力矩（静力等效）。

    防的问题：加载点、设备质量点这类不连杆件的节点以前只能悬空（校验拒绝）或者
    手算等效力矩。约束对象要和手算的等效荷载逐位一致，偏心点自己还要按刚体跟着动。
    """
    a = _with_objects()
    assert a.add_nodes([[0, 0.5, 4.3]]).ok
    arm = a.create_constraint("刚臂", "beam", 2, [7])
    assert arm.ok, arm.payload
    assert arm.payload["nodes"] == [7]
    assert a.create_load("偏心", "DL", "concentrated_force", [7], force=[0, 3e3, -8e3]).ok
    b = _with_objects()
    d, force = np.array([0, 0.5, 0.3]), np.array([0, 3e3, -8e3])
    assert b.create_load("偏心", "DL", "concentrated_force", [2], force=force.tolist()).ok
    assert b.create_load("偏心矩", "DL", "moment", [2], moment=np.cross(d, force).tolist()).ok
    ua, ub = _u(a), _u(b)
    for nid in range(1, 7):
        ka, kb = a.frame.index_of()[nid], b.frame.index_of()[nid]
        np.testing.assert_allclose(ua[6 * ka:6 * ka + 6], ub[6 * kb:6 * kb + 6],
                                   rtol=1e-10, atol=1e-15)
    k2, k7 = a.frame.index_of()[2], a.frame.index_of()[7]
    u2, u7 = ua[6 * k2:6 * k2 + 6], ua[6 * k7:6 * k7 + 6]
    np.testing.assert_allclose(u7[:3], u2[:3] + np.cross(u2[3:], d), rtol=1e-12, atol=1e-15)

    # 删别的杆件时，只靠约束连着的节点不算孤立节点
    assert a.remove_members([5]).ok
    assert 7 in {int(n["id"]) for n in a.model["nodes"]}

    # 单位制换算后结果不变（刚臂系数带长度量纲，跟着节点坐标一起换）
    c = _with_objects()
    assert c.add_nodes([[0, 0.5, 4.3]]).ok
    assert c.create_constraint("刚臂", "beam", 2, [7]).ok
    assert c.create_load("偏心", "DL", "concentrated_force", [7], force=[0, 3e3, -8e3]).ok
    mm = Session(model=convert_model(deepcopy(c.model), "N-mm-MPa"))
    u_si, u_mm = _u(c), _u(mm)
    k = c.frame.index_of()[7]
    np.testing.assert_allclose(u_mm[6 * k:6 * k + 3] / 1e3, u_si[6 * k:6 * k + 3], rtol=1e-9)


def test_floor_coupling_warns_about_undriven_reference_point_dofs(tmp_path):
    """刚性楼板（运动耦合只选 ux uy rz）：悬空参考点的平面外方向没有刚度，创建时就提示。

    同时检查 Abaqus 导出带上了约束——漏写不会报错，只会让对拍的那边算的是
    约束全断开的结构。
    """
    s = _with_objects()
    assert s.add_nodes([[6, 1, 4]]).ok                 # 参考点在楼板平面内
    got = s.create_constraint("楼板", "coupling", 7, [2, 4, 6], dofs=[1, 1, 0, 0, 0, 1])
    assert got.ok, got.payload
    assert "['uz', 'rx', 'ry']" in got.payload["warning"]
    assert not s.solve_model().ok, "参考点平面外无刚度，必须拒绝而不是给出奇异解"
    assert s.create_boundary_condition("RP", [7], fix=[0, 0, 1, 1, 1, 0]).ok
    u = _u(s)
    ux = [u[6 * s.frame.index_of()[n]] for n in (2, 4, 6)]
    assert ux[0] == pytest.approx(ux[1], rel=1e-12) and ux[2] == pytest.approx(ux[1], rel=1e-12)

    from inp_writer import write_inp
    text = write_inp(s.model, path=tmp_path / "floor.inp", case="DL").read_text()
    assert "*KINEMATIC COUPLING, REF NODE=7" in text and " 2, 6, 6" in text
    assert "*MPC" not in text


def test_constraint_over_constraint_and_manager_operations():
    """从自由度已有支座时拒绝并点名；约束对象和其他对象一样能抑制、改名、删除。"""
    s = _with_objects()
    bad = s.create_constraint("错", "beam", 2, [1])
    assert not bad.ok and any("既有支座" in e for e in bad.payload["errors"])
    assert s.add_nodes([[0, 0.0, 4.0001]]).ok
    assert s.create_constraint("连", "tie", 2, [7]).ok
    assert s.rename_feature("constraint", "连", "连接").ok
    assert s.suppress_feature("constraint", "连接").ok
    assert s.suppress_feature("constraint", "连接", False).ok
    assert [f["name"] for f in s.list_features("constraint").payload["features"]["constraints"]] == ["连接"]
    assert s.delete_feature("constraint", "连接").ok
    assert "features" not in s.model or "constraints" not in s.model["features"]


def _beam_with_steps() -> Session:
    """单跨梁 L=6，两端各一个固接边界条件，满跨均布 w；S1 原样、S2 右端改铰、S3 右端失活。"""
    s = Session()
    assert s.add_nodes([[0, 0, 0], [3, 0, 0], [6, 0, 0]]).ok
    assert s.define_materials_and_sections(
        [{"name": "S", "E": 2e11, "nu": 0.3}],
        [{"name": "R", "A": 1e-2, "Iy": 2e-5, "Iz": 8e-5, "J": 1e-5}]).ok
    assert s.add_members([[1, 2], [2, 3]]).ok
    assert s.create_section_assignment("SA", [1, 2], "R", "S").ok
    assert s.create_boundary_condition("BC-左", [1], preset="ENCASTRE").ok
    assert s.create_boundary_condition("BC-右", [3], preset="ENCASTRE").ok
    _case(s)
    assert s.create_load("均布", "DL", "line_load", [1, 2], w=[0, 0, -1e4]).ok
    for name in ("S1", "S2", "S3"):
        assert s.add_step(name, loads={"DL": "RAMP"} if name == "S1" else None).ok, name
    return s


def test_bc_modified_and_deactivated_in_later_steps_match_closed_forms():
    """Abaqus 管理器在步列上 Modify / Deactivate：同一根梁三步三种支承。

    固端 R = wL/2；右端改铰（固-铰梁）铰端 R = 3wL/8；右端失活（悬臂）左端 R = wL。
    原来只能在分析步里按节点号写支座改写，对象层的边界条件改不了、也失活不了。
    """
    s = _beam_with_steps()
    got = s.modify_boundary_condition("BC-右", "S2", preset="PINNED")
    assert got.ok, got.payload
    got = s.deactivate_boundary_condition("BC-右", "S3")
    assert got.ok, got.payload
    assert [x["status"] for x in got.payload["steps"]] == ["创建", "传播", "修改", "失活"]
    w, L = 1e4, 6.0
    expected = {"S1": (3, w * L / 2), "S2": (3, 3 * w * L / 8), "S3": (1, w * L)}
    for step, (node, reaction) in expected.items():
        assert s.solve_steps(inspect=step).ok, step
        k = s.frame.index_of()[node]
        assert s.solution.R[6 * k + 2] == pytest.approx(reaction, rel=1e-9), step
    # 撤销修改、重新激活：回到三步都是固端
    assert s.modify_boundary_condition("BC-右", "S2").ok
    assert s.deactivate_boundary_condition("BC-右", None).ok
    assert "deactivate_supports" not in s.model["steps"][2] and not s.model["steps"][1].get("supports")
    assert not s.deactivate_boundary_condition("BC-右", "Initial").ok, "失活不能早于创建"


def test_a_bc_created_in_a_later_step_merges_with_the_existing_one_on_the_node():
    """后面步里新建的边界条件只加它自己的方向，不能把节点上已有的约束整个顶掉。"""
    s = _beam_with_steps()
    assert s.edit_feature("bc", "BC-右", {"fix": [0, 0, 1, 0, 0, 0]}).ok
    assert s.create_boundary_condition("BC-右加", [3], fix=[1, 1, 0, 1, 1, 1], step="S2").ok
    entry = s.model["steps"][1]["supports"]["3"]
    assert entry["fix"] == [1] * 6 and sorted(entry["owners"]) == ["BC-右", "BC-右加"]


def _four_steps() -> Session:
    s = _with_objects()
    assert s.add_step("静力", loads={"DL": "RAMP"}).ok
    assert s.add_step("二阶", analysis="pdelta", increments=4).ok
    assert s.add_step("频率", analysis="frequency", num_modes=3).ok
    assert s.add_step("屈曲", analysis="buckle", loads={"DL": "RAMP"}, num_modes=2).ok
    return s


def test_every_step_result_is_kept_and_inspect_step_switches_without_resolving():
    """Abaqus 结果树：提交一次，各步结果都在，点哪一步看哪一步。

    原来会话只留 inspect 指定的那一步，换一步看结果要把全部分析步重算一遍。
    切换得到的结果要与直接 solve_steps(inspect=…) 逐位相同；P-Δ 步的每个增量是一帧。
    """
    s = _four_steps()
    got = s.solve_steps()
    assert got.ok, got.payload
    assert list(s.step_results) == ["静力", "二阶", "频率", "屈曲"]
    switched = s.inspect_step("静力")
    assert switched.ok and switched.payload["frames"] == [1.0]
    u_switched = s.solution[switched.payload["case"]].U.copy()
    direct = _four_steps()
    assert direct.solve_steps(inspect="静力").ok
    np.testing.assert_array_equal(u_switched, direct.solution["静力"].U)

    second = s.inspect_step("二阶")
    assert second.payload["frames"] == [0.25, 0.5, 0.75, 1.0]
    result = s.solution[second.payload["case"]]
    np.testing.assert_array_equal(result.frames[-1][1], result.U)
    assert s.result_identity["step"] == "二阶"

    modal = s.inspect_step("频率")
    assert modal.ok and modal.payload["perturbation"] and len(modal.payload["summary"]["frequencies_Hz"]) == 3
    assert s.step_results["屈曲"]["view"][2].factors[0] > 0

    assert s.create_load("追加", "DL", "concentrated_force", [4], force=[0, 0, -1e3]).ok
    stale = s.inspect_step("静力")
    assert not stale.ok and "作废" in stale.payload["error"]


def _settled_beam() -> Session:
    """两端固接梁 L=6，右端是带给定位移的边界条件：随工况「沉降」下沉 δ。"""
    s = Session()
    assert s.add_nodes([[0, 0, 0], [3, 0, 0], [6, 0, 0]]).ok
    assert s.define_materials_and_sections(
        [{"name": "S", "E": 2e11, "nu": 0.3}],
        [{"name": "R", "A": 1e-2, "Iy": 2e-5, "Iz": 8e-5, "J": 1e-5}]).ok
    assert s.add_members([[1, 2], [2, 3]]).ok
    assert s.create_section_assignment("SA", [1, 2], "R", "S").ok
    assert s.create_boundary_condition("BC-左", [1], preset="ENCASTRE").ok
    assert s.set_load_cases([{"name": "沉降"}]).ok
    got = s.create_boundary_condition("BC-右", [3], preset="ENCASTRE",
                                      displacement=[0, 0, -0.01, 0, 0, 0], case="沉降")
    assert got.ok, got.payload
    return s


def test_prescribed_displacement_bc_matches_the_settlement_closed_form():
    """支座沉降作为边界条件对象（Abaqus 的 Displacement 给非零值）：固端弯矩 6EIδ/L²。

    原来沉降只能逐节点写直接条目，删了旧边界表单后界面只能临时挂在「创建载荷」下；
    语义上它是边界条件，随分析步与工况施加。
    """
    s = _settled_beam()
    entry = s.model["load_cases"][0]["settlements"][0]
    assert entry["owner"] == "BC-右" and entry["node"] == 3
    assert s.solve_model().ok
    u = s.solution.cases["沉降"].U
    k3 = s.frame.index_of()[3]
    assert u[6 * k3 + 2] == pytest.approx(-0.01, abs=1e-15)
    E, Iz, L, d = 2e11, 8e-5, 6.0, 0.01
    k1 = s.frame.index_of()[1]
    r = s.solution.cases["沉降"].R
    assert abs(r[6 * k1 + 2]) == pytest.approx(12 * E * Iz * d / L**3, rel=1e-9)
    assert abs(r[6 * k1 + 4]) == pytest.approx(6 * E * Iz * d / L**2, rel=1e-9)
    bad = s.create_boundary_condition("错", [2], fix=[0, 0, 1, 0, 0, 0],
                                      displacement=[0.01, 0, 0, 0, 0, 0], case="沉降")
    assert not bad.ok, "给定位移只能写在固定的方向上"


def test_deactivating_a_settled_bc_releases_its_displacement_in_that_step():
    """后一步失活带给定位移的边界条件：那一步该节点自由，给定位移也随之不施加。

    同时防一个分析步的真实问题：S1 里「端部」工况没生效，它的力却挂在 S1 仍有支座的
    节点 3 上——以前每步都把全部工况算一遍并过静默检测，这个没生效的工况被判成
    「内力全零」，整步 S1 失败。
    """
    s = _settled_beam()
    assert s.add_load_case("端部").ok
    P = -1e3
    assert s.create_load("P", "端部", "concentrated_force", [3], force=[0, 0, P]).ok
    assert s.add_step("S1", loads={"沉降": "RAMP"}).ok
    assert s.add_step("S2", loads={"端部": "RAMP"}).ok
    assert s.deactivate_boundary_condition("BC-右", "S2").ok
    got = s.solve_steps(inspect="S1")
    assert got.ok, got.payload
    k3 = s.frame.index_of()[3]
    assert s.solution["S1"].U[6 * k3 + 2] == pytest.approx(-0.01, abs=1e-15)
    assert s.inspect_step("S2").ok
    E, Iz, L = 2e11, 8e-5, 6.0
    assert s.solution["S2"].U[6 * k3 + 2] == pytest.approx(P * L**3 / (3 * E * Iz), rel=1e-9),         "右端失活后是悬臂，沉降不再施加"
    mm = Session(model=convert_model(deepcopy(s.model), "N-mm-MPa"))
    bc = next(f for f in mm.model["features"]["bcs"] if f["name"] == "BC-右")
    assert bc["displacement"][2] == pytest.approx(-10.0)
