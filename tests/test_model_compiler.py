"""物理构件到分析单元的最小编译边界。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from agent import Session
from frame3d import solve
from model_compiler import CompilationError, compile_model
from model_io import from_dict


def _cantilever() -> dict:
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": 2.0e11, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 1.0e-5,
                      "Iz": 2.0e-5, "J": 5.0e-6}],
        "nodes": [{"id": 11, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 22, "x": 3.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 101, "i": 11, "j": 22,
                     "section": "S", "material": "Steel"}],
        "supports": [{"node": 11, "fix": [1, 1, 1, 1, 1, 1]}],
        "nodal_loads": [{"node": 22, "load": [0, 0, -1000, 0, 0, 0]}],
    }


def _cantilever_with_point_load(position: float = 1.5) -> dict:
    model = _cantilever()
    model.pop("nodal_loads")
    model["members"][0]["releases"] = {"i": ["ry"], "j": ["rz"]}
    model["member_spans"] = [
        {"member": 101, "kind": "point", "w1": [0, 0, -1000],
         "a": position},
    ]
    return model


def _cantilever_with_explicit_inner_node() -> dict:
    model = _cantilever()
    model["nodes"].extend([
        {"id": 33, "x": 1.5, "y": 0.0, "z": 0.0},
        {"id": 44, "x": 1.5, "y": 1.0, "z": 0.0},
    ])
    model["members"].append({
        "id": 202, "i": 33, "j": 44,
        "section": "S", "material": "Steel",
    })
    return model


def test_one_physical_member_compiles_to_one_analysis_element():
    compiled = compile_model(_cantilever())

    assert set(compiled.analysis_model.members) == {101}
    assert compiled.mapping.element_ids(101) == (101,)
    assert compiled.mapping.physical_member_id(101) == 101
    assert compiled.source_ir["members"][0]["id"] == 101


def test_compiler_returns_structured_diagnostics_for_dangling_reference():
    model = _cantilever()
    model["members"][0]["j"] = 999

    with pytest.raises(CompilationError) as caught:
        compile_model(model)

    assert any("999" in item for item in caught.value.diagnostics)


def test_linear_solve_uses_compiled_model_and_keeps_mapping():
    session = Session()
    assert session.set_model(_cantilever()).ok

    result = session.solve_model()

    assert result.ok
    assert session.compilation is not None
    assert session.frame is session.compilation.analysis_model
    assert session.compilation.mapping.element_ids(101) == (101,)


def test_model_change_invalidates_compilation_together_with_results():
    session = Session()
    assert session.set_model(_cantilever()).ok
    assert session.solve_model().ok

    changed = session.edit_node(22, x=4.0)

    assert changed.ok
    assert session.compilation is None
    assert session.frame is None
    assert session.solution is None


def test_interior_point_load_creates_node_and_two_analysis_elements():
    compiled = compile_model(_cantilever_with_point_load())
    frame = compiled.analysis_model

    assert compiled.mapping.element_ids(101) == (101, 102)
    assert compiled.mapping.physical_member_id(102) == 101
    assert compiled.mapping.generated_nodes == {23: (101, 0.5)}
    assert frame.nodes[23].xyz == pytest.approx([1.5, 0.0, 0.0])
    assert (frame.members[101].i, frame.members[101].j) == (11, 23)
    assert (frame.members[102].i, frame.members[102].j) == (23, 22)
    assert frame.members[101].releases_i == ("ry",)
    assert frame.members[101].releases_j == ()
    assert frame.members[102].releases_i == ()
    assert frame.members[102].releases_j == ("rz",)
    assert frame.case().nodal_loads[23][:3] == pytest.approx((0, 0, -1000))
    assert not frame.case().member_spans
    assert compiled.diagnostics


def test_analysis_mesh_preview_exposes_split_without_changing_physical_model():
    session = Session()
    payload = _cantilever_with_point_load()
    assert session.set_model(payload).ok
    before = deepcopy(session.model)

    result = session.preview_analysis_mesh()

    assert result.ok
    assert result.payload["physical_nodes"] == 2
    assert result.payload["physical_members"] == 1
    assert result.payload["analysis_nodes"] == 3
    assert result.payload["analysis_elements"] == 2
    assert result.payload["generated_nodes"] == 1
    assert result.payload["has_automatic_splits"] is True
    assert result.payload["length_unit"] == "m"
    assert result.payload["member_mapping"] == {"101": [101, 102]}
    assert result.payload["generated_node_details"] == [{
        "node": 23,
        "physical_member": 101,
        "s": pytest.approx(0.5),
        "coordinates": pytest.approx([1.5, 0.0, 0.0]),
    }]
    assert result.payload["split_node_details"] == [{
        "node": 23,
        "physical_member": 101,
        "s": pytest.approx(0.5),
        "origin": "generated_for_point_load",
        "coordinates": pytest.approx([1.5, 0.0, 0.0]),
    }]
    assert result.payload["analysis_element_details"] == [
        {"element": 101, "physical_member": 101, "i": 11, "j": 23},
        {"element": 102, "physical_member": 101, "i": 23, "j": 22},
    ]
    assert session.model == before
    assert session.frame is None
    assert session.solution is None
    assert session.compilation is None
    assert session.result_db is None


def test_explicit_inner_node_splits_member_without_creating_a_node():
    compiled = compile_model(_cantilever_with_explicit_inner_node())

    assert compiled.mapping.element_ids(101) == (101, 203)
    assert compiled.mapping.element_ids(202) == (202,)
    assert not compiled.mapping.generated_nodes
    assert (compiled.analysis_model.members[101].i,
            compiled.analysis_model.members[101].j) == (11, 33)
    assert (compiled.analysis_model.members[203].i,
            compiled.analysis_model.members[203].j) == (33, 22)
    assert any("显式内节点" in item for item in compiled.diagnostics)


def test_explicit_inner_node_solution_matches_manually_split_reference():
    model = _cantilever_with_explicit_inner_node()
    compiled = compile_model(model)

    reference = deepcopy(model)
    reference["members"][0]["j"] = 33
    reference["members"].append({
        "id": 203, "i": 33, "j": 22,
        "section": "S", "material": "Steel",
    })
    reference_frame = from_dict(reference)
    automatic = solve(compiled.analysis_model)
    manual = solve(reference_frame)

    for node_id in (11, 22, 33, 44):
        automatic_dofs = compiled.analysis_model.node_dofs(node_id)
        manual_dofs = reference_frame.node_dofs(node_id)
        assert automatic.U[automatic_dofs] == pytest.approx(
            manual.U[manual_dofs], rel=1e-9, abs=1e-12)
        assert automatic.R[automatic_dofs] == pytest.approx(
            manual.R[manual_dofs], rel=1e-9, abs=1e-8)


def test_point_load_reuses_explicit_inner_node():
    model = _cantilever_with_explicit_inner_node()
    model.pop("nodal_loads")
    model["member_spans"] = [{
        "member": 101, "kind": "point", "w1": [0, 0, -1000], "a": 1.5,
    }]

    compiled = compile_model(model)

    assert not compiled.mapping.generated_nodes
    assert compiled.mapping.element_ids(101) == (101, 203)
    assert compiled.analysis_model.case().nodal_loads[33][:3] == pytest.approx(
        (0, 0, -1000))


def test_crossing_members_without_an_explicit_node_remain_unconnected():
    model = _cantilever()
    model["nodes"].extend([
        {"id": 33, "x": 1.5, "y": -1.0, "z": 0.0},
        {"id": 44, "x": 1.5, "y": 1.0, "z": 0.0},
    ])
    model["members"].append({
        "id": 202, "i": 33, "j": 44,
        "section": "S", "material": "Steel",
    })

    compiled = compile_model(model)

    assert compiled.mapping.element_ids(101) == (101,)
    assert compiled.mapping.element_ids(202) == (202,)


def test_preview_identifies_an_explicit_split_node():
    session = Session()
    assert session.set_model(_cantilever_with_explicit_inner_node()).ok

    result = session.preview_analysis_mesh()

    assert result.ok
    assert result.payload["generated_nodes"] == 0
    assert result.payload["split_nodes"] == 1
    assert result.payload["has_automatic_splits"] is True
    assert result.payload["split_node_details"] == [{
        "node": 33,
        "physical_member": 101,
        "s": pytest.approx(0.5),
        "origin": "explicit_physical_node",
        "coordinates": pytest.approx([1.5, 0.0, 0.0]),
    }]


def test_analysis_mesh_preview_preserves_existing_solution_objects():
    session = Session()
    payload = _cantilever_with_point_load()
    payload["members"][0].pop("releases")
    assert session.set_model(payload).ok
    assert session.solve_model().ok
    state = (session.frame, session.solution, session.compilation, session.result_db)

    result = session.preview_analysis_mesh()

    assert result.ok
    assert (session.frame, session.solution, session.compilation,
            session.result_db) == state


def test_analysis_mesh_preview_reports_invalid_model_without_mutation():
    session = Session(model=_cantilever())
    session.model["members"][0]["j"] = 999
    before = deepcopy(session.model)

    result = session.preview_analysis_mesh()

    assert not result.ok
    assert any("999" in error for error in result.payload["errors"])
    assert session.model == before
    assert session.compilation is None


def test_repeated_point_load_position_uses_one_node_and_adds_forces():
    model = _cantilever_with_point_load()
    model["member_spans"].append(
        {"member": 101, "kind": "point", "w1": [0, -250, 0], "a": 1.5})

    compiled = compile_model(model)
    generated = next(iter(compiled.mapping.generated_nodes))

    assert compiled.mapping.element_ids(101) == (101, 102)
    assert compiled.analysis_model.case().nodal_loads[generated][:3] == pytest.approx(
        (0, -250, -1000))


def test_endpoint_point_load_becomes_nodal_load_without_splitting():
    compiled = compile_model(_cantilever_with_point_load(position=0.0))

    assert compiled.mapping.element_ids(101) == (101,)
    assert not compiled.mapping.generated_nodes
    assert compiled.analysis_model.case().nodal_loads[11][:3] == pytest.approx(
        (0, 0, -1000))


def test_point_load_with_rigid_offset_fails_instead_of_losing_eccentric_moment():
    model = _cantilever_with_point_load(position=0.0)
    model["members"][0]["offset_i"] = [0.0, 0.1, 0.0]

    with pytest.raises(CompilationError, match="刚域偏移"):
        compile_model(model)


def test_split_solution_matches_previous_exact_span_load_solution():
    payload = _cantilever_with_point_load()
    # 这个等价性测试只验证剖分，不把端部释放的影响混进来。
    payload["members"][0].pop("releases")
    direct_frame = from_dict(payload)
    direct = solve(direct_frame)
    compiled = compile_model(payload)
    split = solve(compiled.analysis_model)

    for node_id in (11, 22):
        direct_dofs = direct_frame.node_dofs(node_id)
        split_dofs = compiled.analysis_model.node_dofs(node_id)
        assert split.U[split_dofs] == pytest.approx(direct.U[direct_dofs], rel=1e-9,
                                                    abs=1e-12)
        assert split.R[split_dofs] == pytest.approx(direct.R[direct_dofs], rel=1e-9,
                                                    abs=1e-8)
    first, last = compiled.mapping.element_ids(101)
    assert split.member_forces[first][:6] == pytest.approx(
        direct.member_forces[101][:6], rel=1e-9, abs=1e-8)
    assert split.member_forces[last][6:] == pytest.approx(
        direct.member_forces[101][6:], rel=1e-9, abs=1e-8)


def test_session_queries_and_plot_return_physical_member_after_split(
        tmp_path, monkeypatch):
    payload = _cantilever_with_point_load()
    payload["members"][0].pop("releases")
    session = Session()
    assert session.set_model(payload).ok
    assert session.solve_model().ok

    forces = session.query_results("member_forces").payload["all_members"]
    diagram = session.query_diagram("Mz", member=101, stations=31).payload
    deflection = session.query_results("max_deflection").payload
    monkeypatch.chdir(tmp_path)
    plotted = session.plot_results("deformed").payload

    assert set(forces) == {"101"}
    assert diagram["member"] == 101
    assert diagram["length_m"] == pytest.approx(3.0)
    assert diagram["x_m"][0] == 0.0
    assert diagram["x_m"][-1] == 3.0
    assert deflection["at_member"] == 101
    assert deflection["member_length_m"] == pytest.approx(3.0)
    assert plotted["at_member"] == 101
    assert plotted["at_x_m"] == pytest.approx(3.0, abs=0.02)
    assert session.result_db.metadata["analysis_elements"] == 2
    assert session.result_db.metadata["physical_members"] == 1


def test_trapezoid_load_is_restricted_to_each_generated_element():
    model = _cantilever_with_point_load()
    model["members"][0].pop("releases")
    model["member_spans"].append({
        "member": 101, "kind": "trapezoid", "w1": [0, 0, 0],
        "w2": [0, 0, -300],
    })

    frame = compile_model(model).analysis_model
    first = frame.case().member_spans[101][0]
    second = frame.case().member_spans[102][0]

    assert first.w1 == pytest.approx((0, 0, 0))
    assert first.w2 == pytest.approx((0, 0, -150))
    assert second.w1 == pytest.approx((0, 0, -150))
    assert second.w2 == pytest.approx((0, 0, -300))


def test_all_cases_share_union_of_split_positions():
    model = _cantilever()
    model.pop("nodal_loads")
    model["load_cases"] = [
        {"name": "A", "member_spans": [
            {"member": 101, "kind": "point", "w1": [0, 0, -100], "a": 1.0}]},
        {"name": "B", "member_spans": [
            {"member": 101, "kind": "point", "w1": [0, 0, -200], "a": 2.0}]},
    ]

    compiled = compile_model(model)

    assert compiled.mapping.element_ids(101) == (101, 102, 103)
    assert len(compiled.mapping.generated_nodes) == 2
    for case in compiled.analysis_model.load_cases.values():
        assert not case.member_spans


def test_envelope_reports_physical_member_and_full_length_after_split():
    model = _cantilever()
    model.pop("nodal_loads")
    model["load_cases"] = [
        {"name": "A", "member_spans": [
            {"member": 101, "kind": "point", "w1": [0, 0, -100], "a": 1.0}]},
        {"name": "B", "member_spans": [
            {"member": 101, "kind": "point", "w1": [0, 0, -200], "a": 2.0}]},
    ]
    session = Session()
    assert session.set_model(model).ok
    assert session.solve_model().ok

    global_result = session.query_envelope("Mz")
    member_result = session.query_envelope("Mz", member=101)

    assert global_result.ok
    assert global_result.payload["at_member"] == 101
    assert member_result.ok
    assert member_result.payload["member"] == 101
    assert member_result.payload["length_m"] == pytest.approx(3.0)


# --------------------------------- 编译不得制造重复传力路径

def _two_spans_plus_a_spanning_member() -> dict:
    """1-2、2-3 两跨，外加一根跨越两跨的 1-3。

    三根构件两两连接不同的节点对，物理层完全合法，`validate_payload` 放行；
    1-3 被节点 2 剖分之后才出现重复。
    """
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": 2.0e11, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 1.0e-5,
                      "Iz": 2.0e-5, "J": 5.0e-6}],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 3.0, "y": 0.0, "z": 0.0},
                  {"id": 3, "x": 6.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "Steel"},
                    {"id": 2, "i": 2, "j": 3, "section": "S", "material": "Steel"},
                    {"id": 3, "i": 1, "j": 3, "section": "S", "material": "Steel"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": 3, "fix": [1, 1, 1, 1, 1, 1]}],
        "nodal_loads": [{"node": 2, "load": [0, 0, -10e3, 0, 0, 0]}],
    }


def test_physical_model_alone_passes_validation():
    """先钉住前提：重复只在编译之后出现，编译前的 IR 是合法的。

    这条守的是"为什么校验必须在编译之后再跑一次"——如果哪天物理层校验
    自己就能拦下它，下面那条测试的理由就变了，应该由这条先失败来提醒。
    """
    from model_io import validate_payload
    assert validate_payload(_two_spans_plus_a_spanning_member()) == []


def test_compilation_rejects_duplicated_load_paths():
    """剖分产生重复传力路径时必须拒绝，并指出是哪几根物理构件。"""
    with pytest.raises(CompilationError) as caught:
        compile_model(_two_spans_plus_a_spanning_member())
    text = str(caught.value)
    assert "重复传力" in text
    assert "物理构件 3" in text          # 跨越的那根
    assert "1-2" in text or "2-3" in text


def test_legitimate_split_at_a_column_base_still_compiles():
    """对照组：柱脚落在梁跨内的剖分是想要的行为，不能被误伤。"""
    model = _two_spans_plus_a_spanning_member()
    model["members"] = [
        {"id": 1, "i": 1, "j": 3, "section": "S", "material": "Steel"},
        {"id": 2, "i": 2, "j": 4, "section": "S", "material": "Steel"},
    ]
    model["nodes"].append({"id": 4, "x": 3.0, "y": 0.0, "z": 3.0})
    model["nodal_loads"] = [{"node": 4, "load": [0, 0, -10e3, 0, 0, 0]}]
    compiled = compile_model(model)
    assert compiled.mapping.element_ids(1) == (1, 3)     # 梁被剖分成两段
    assert compiled.mapping.element_ids(2) == (2,)
