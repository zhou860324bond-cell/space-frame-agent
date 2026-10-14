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
