"""MM2-07 structured preview, atomic commit, and sidecar tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent import Session
from change_preview import model_digest
from draft_commit import (DraftCommitError, commit_prepared, load_sidecar,
                          materialize_candidate, prepare_commit, save_sidecar,
                          sidecar_path)
from multimodal_workflow import MultimodalControllerState
from recognition_draft import DRAFT_FORMAT, migrate_to_v2


def ready_draft() -> dict:
    v1 = {
        "format": DRAFT_FORMAT, "scale": {"status": "unknown"},
        "model": {
            "units": "N-m-Pa", "materials": [], "sections": [],
            "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                      {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}],
            "members": [{"id": 1, "i": 1, "j": 2,
                         "material": "", "section": ""}],
            "supports": [], "load_cases": [],
        }, "entities": [], "questions": [], "warnings": [],
    }
    draft = migrate_to_v2(v1, image_hash="a" * 64, source_path="drawing.png")
    draft["source"]["preprocessing"]["perspective_status"] = "accepted"
    draft["work_plane"]["status"] = "confirmed"
    draft["scale"].update(status="confirmed", length_per_pixel=0.01)
    draft["model"] = deepcopy(v1["model"])
    draft["issues"] = []
    return draft


def armed_state(draft: dict, preview: dict) -> MultimodalControllerState:
    state = MultimodalControllerState()
    state.load_image("a" * 64)
    job = state.start_recognition("job")
    assert state.complete_recognition(job, draft)
    state.set_commit_preview(preview)
    return state


def test_prepare_is_pure_and_preview_is_structured():
    session = Session()
    before = deepcopy(session.model)
    prepared, preview = prepare_commit(ready_draft(), session.model)
    assert session.model == before
    assert prepared["merge_plan"]["mode"] == "replace_empty"
    assert [item["target_id"] for item in prepared["merge_plan"]["node_actions"]] == [1, 2]
    assert {item["kind"] for item in preview["operations"]} == {"node", "member"}
    assert "candidate_model" not in preview


def test_commit_is_one_undo_step_and_provenance_undoes_with_model():
    session = Session()
    prepared, preview = prepare_commit(ready_draft(), session.model)
    state = armed_state(prepared, preview)
    result = commit_prepared(session, state, confirmed_at="2026-09-08T00:00:00Z")
    assert result.ok
    assert len(session.history) == 1
    assert session.multimodal_provenance["image_hash"] == "a" * 64
    committed = deepcopy(session.model)
    assert session.undo() and session.model == {}
    assert session.multimodal_provenance is None
    assert session.redo() and session.model == committed
    assert session.multimodal_provenance["image_hash"] == "a" * 64


def test_stale_baseline_rejected_without_session_or_history_change():
    session = Session()
    prepared, preview = prepare_commit(ready_draft(), session.model)
    state = armed_state(prepared, preview)
    session.model = {"units": "N-m-Pa", "nodes": [{"id": 9, "x": 9, "y": 0, "z": 0}],
                     "members": []}
    before = deepcopy(session.model)
    with pytest.raises(DraftCommitError, match="陈旧|状态"):
        commit_prepared(session, state, confirmed_at="2026-09-08T00:00:00Z")
    assert session.model == before
    assert len(session.history) == 0
    assert session.multimodal_provenance is None


def test_session_failure_rolls_back_model_provenance_and_history():
    session = Session(model={"units": "N-m-Pa"})
    baseline_hash = model_digest(session.model)
    bad = {"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}],
           "members": [{"id": 1, "j": 1}], "supports": []}
    result = session.apply_multimodal_draft(
        bad, {"image_hash": "a" * 64}, expected_baseline_hash=baseline_hash)
    assert not result.ok
    assert session.model == {"units": "N-m-Pa"}
    assert session.multimodal_provenance is None
    assert len(session.history) == 0


def test_sidecar_round_trip_mismatch_and_legacy_absence(tmp_path):
    model_path = tmp_path / "frame.json"
    model = {"units": "N-m-Pa", "nodes": [], "members": []}
    provenance = {"image_hash": "a" * 64}
    written = save_sidecar(model_path, model, provenance)
    assert written == sidecar_path(model_path)
    loaded, warning = load_sidecar(model_path, model)
    assert loaded == provenance and warning is None
    loaded, warning = load_sidecar(model_path, {**model, "units": "N-mm-MPa"})
    assert loaded is None and "哈希不匹配" in warning
    assert load_sidecar(tmp_path / "legacy.json", model) == (None, None)


def test_add_only_can_reuse_one_node_and_remap_support_and_named_load():
    baseline = {
        "schema_version": 1, "units": "N-m-Pa", "materials": [], "sections": [],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "", "section": ""}],
        "supports": [], "load_cases": [],
    }
    draft = ready_draft()
    draft["model"]["nodes"] = [
        {"id": 1, "x": 1.0, "y": 0.0, "z": 0.0},
        {"id": 2, "x": 2.0, "y": 0.0, "z": 0.0},
    ]
    draft["model"]["supports"] = [
        {"name": "new-end", "node": 2, "fix": [1, 1, 1, 0, 0, 0]}]
    draft["model"]["load_cases"] = [{
        "name": "LC1", "nodal_loads": [
            {"name": "P", "node": 2, "load": [0, -1, 0, 0, 0, 0]}]}]

    prepared, preview = prepare_commit(draft, baseline, node_reuse={1: 2})
    candidate = materialize_candidate(prepared, baseline)
    assert prepared["merge_plan"]["node_actions"] == [
        {"draft_id": 1, "action": "reuse", "target_id": 2},
        {"draft_id": 2, "action": "create", "target_id": 3},
    ]
    assert candidate["members"][-1].get("i") == 2
    assert candidate["members"][-1].get("j") == 3
    assert candidate["supports"][-1]["node"] == 3
    assert candidate["load_cases"][-1]["nodal_loads"][0]["node"] == 3
    assert any(item["op"] == "reuse" and item["kind"] == "node"
               for item in preview["operations"])
    assert any(item["kind"] == "load" and item["key"] == "LC1:nodal_loads:P"
               for item in preview["operations"])


def test_add_only_refuses_duplicate_member_created_by_reuse():
    baseline = {
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2}], "supports": []}
    draft = ready_draft()
    with pytest.raises(DraftCommitError, match="重复"):
        prepared, _ = prepare_commit(draft, baseline, node_reuse={1: 1, 2: 2})
        materialize_candidate(prepared, baseline)


# --- 并入已有模型时，每一种荷载的编号都要重映射 ---------------------------
#
# add_only 模式下草稿的节点/杆件编号要往后排。合并代码按一张 load_collections
# 表逐类重映射，**不在表里的键会被当成工况元数据原样拷贝**——连同里面的
# node / member 引用。member_strains 就这么漏在外面过：草稿写"杆件 1 升温
# 30 度"，并进一个已有两根杆的模型后，那条初应变仍然指着**基线的**杆件 1，
# 而不是它自己那根。模型照样合法，自检也查不出来。

def baseline_with_two_members() -> dict:
    """已有 3 节点 / 2 杆件的模型——草稿并进来时编号必须从 3、4 往后排。"""
    session = Session()
    session.add_nodes([[0, 0, 0], [6, 0, 0], [12, 0, 0]])
    session.define_materials_and_sections(
        materials=[{"name": "M", "E": 2.1e11, "nu": 0.3}],
        sections=[{"name": "S", "A": 0.02, "Iy": 2e-4, "Iz": 4e-4,
                   "J": 1e-5}])
    session.add_members([[1, 2], [2, 3]], "S", "M")
    session.set_supports([1], fix=[1, 1, 1, 1, 1, 1])
    return session.model


def draft_with_case(case: dict) -> dict:
    draft = ready_draft()
    # 挪开草稿几何，避开"与现有节点重合"的保护——那是另一条规则
    for node in draft["model"]["nodes"]:
        node["y"] = 5.0
    draft["model"]["load_cases"] = [case]
    return draft


@pytest.mark.parametrize(("collection", "item", "key"), [
    ("member_loads", {"member": 1, "w": [0.0, 0.0, -20e3]}, "member"),
    ("member_spans", {"member": 1, "kind": "point", "w1": [0, 0, -1e4],
                      "a": 0.5}, "member"),
    ("member_strains", {"member": 1, "delta_t": 30.0}, "member"),
    ("nodal_loads", {"node": 1, "load": [0, 0, -1e3, 0, 0, 0]}, "node"),
    ("settlements", {"node": 1, "d": [0.0, 0.0, -0.01, 0.0, 0.0, 0.0]},
     "node"),
])
def test_every_load_collection_gets_its_ids_remapped(collection, item, key):
    """每一种荷载集合的编号都必须跟着合并计划走。

    判据是**对着合并计划里的映射表**，不是"看起来变了"：漏映射的那一项
    会停在草稿里的原编号上，而那个编号在基线里往往也存在——荷载就悄悄
    落到了另一根杆件上。
    """
    baseline = baseline_with_two_members()
    draft = draft_with_case({"name": "C", collection: [item]})
    prepared, _ = prepare_commit(draft, baseline)
    plan = prepared["merge_plan"]
    candidate = materialize_candidate(prepared, baseline)

    if key == "member":
        expected = int(plan["member_id_map"][str(item["member"])])
    else:
        expected = next(int(action["target_id"])
                        for action in plan["node_actions"]
                        if int(action["draft_id"]) == item["node"])
    case = next(c for c in candidate["load_cases"] if c["name"] == "C")
    assert [entry[key] for entry in case[collection]] == [expected], (
        f"{collection} 的 {key} 没有跟着合并计划重映射")


def test_a_load_collection_the_merge_does_not_know_is_refused():
    """工况里出现合并流程不认识的键时要**拦住**，不能原样拷贝。

    原样拷贝正是上面那个 bug 的来路：不在 load_collections 表里的键被当成
    元数据带过去，里面的编号一个都不会改。Schema 将来再长一种荷载时，
    这条会立刻红，而不是等到某个模型把荷载加到了别的杆件上。
    """
    baseline = baseline_with_two_members()
    draft = draft_with_case({
        "name": "X",
        "member_loads": [{"member": 1, "w": [0.0, 0.0, -1e3]}],
        "将来的新集合": [{"member": 1}],
    })
    with pytest.raises(DraftCommitError, match="不认识的字段"):
        prepare_commit(draft, baseline)


def test_the_guard_lets_the_known_collections_through():
    """拦截不能误伤：五种已知集合同时出现也要过。"""
    baseline = baseline_with_two_members()
    draft = draft_with_case({
        "name": "ALL",
        "nodal_loads": [{"node": 2, "load": [0, 0, -1e3, 0, 0, 0]}],
        "member_loads": [{"member": 1, "w": [0.0, 0.0, -2e4]}],
        "member_spans": [{"member": 1, "kind": "point", "w1": [0, 0, -1e4],
                          "a": 0.5}],
        "member_strains": [{"member": 1, "delta_t": 30.0}],
    })
    prepared, _ = prepare_commit(draft, baseline)
    candidate = materialize_candidate(prepared, baseline)
    case = next(c for c in candidate["load_cases"] if c["name"] == "ALL")
    target = int(prepared["merge_plan"]["member_id_map"]["1"])
    for collection in ("member_loads", "member_spans", "member_strains"):
        assert [e["member"] for e in case[collection]] == [target], collection


# --- 草稿与模型的单位制 ---------------------------------------------------
#
# 识别管线产出的草稿固定是 N-m-Pa，而用户可以把项目切到 N-mm-MPa（钢结构
# 详图常用）。合并以前**完全不看单位**：草稿里写 1.0 意思是 1 米，直接并进
# 毫米制模型就成了 1 毫米——小 1000 倍。12 米的框架旁边挂一根 1 毫米的杆，
# 模型依然合法，能求解、出数、没有任何提示。

def mm_baseline() -> dict:
    from units import convert_model
    return convert_model(baseline_with_two_members(), "N-mm-MPa")


def test_a_metre_draft_lands_at_the_right_size_in_a_millimetre_model():
    """1 米的草稿并进毫米制模型，必须变成 1000 mm，不是 1 mm。"""
    from units import convert_model

    baseline = mm_baseline()
    draft = draft_with_case({"name": "D",
                             "member_loads": [{"member": 1,
                                               "w": [0.0, 0.0, -20e3]}]})
    assert draft["model"]["units"] == "N-m-Pa"
    span = abs(draft["model"]["nodes"][1]["x"] - draft["model"]["nodes"][0]["x"])

    prepared, _ = prepare_commit(draft, baseline)
    candidate = materialize_candidate(prepared, baseline)
    added = [n for n in candidate["nodes"]
             if n["id"] not in {item["id"] for item in baseline["nodes"]}]
    got = abs(added[1]["x"] - added[0]["x"])
    assert got == pytest.approx(span * 1e3), (
        f"草稿跨度 {span} m 并进毫米制模型后是 {got}，应当是 {span * 1e3} mm")
    # 换算不能只动坐标：线荷载在毫米制下是 N/mm
    case = next(c for c in candidate["load_cases"] if c["name"] == "D")
    expected = convert_model(
        {"units": "N-m-Pa", "materials": [], "sections": [], "nodes": [],
         "members": [], "supports": [],
         "member_loads": [{"member": 1, "w": [0.0, 0.0, -20e3]}]},
        "N-mm-MPa")["member_loads"][0]["w"][2]
    assert case["member_loads"][0]["w"][2] == pytest.approx(expected)


def test_committing_into_an_empty_model_keeps_the_projects_unit_system():
    """replace_empty 原先整个 return deepcopy(model)，连 units 一起替换。

    用户把项目切到毫米制、再提交一张图，结果单位被悄悄换回米制——
    之后所有输入都按错的单位理解。
    """
    from units import convert_model

    empty = convert_model(Session().model, "N-mm-MPa")
    draft = ready_draft()
    prepared, _ = prepare_commit(draft, empty)
    candidate = materialize_candidate(prepared, empty)
    assert candidate["units"] == "N-mm-MPa", "提交之后项目的单位制被改掉了"
    span = abs(candidate["nodes"][1]["x"] - candidate["nodes"][0]["x"])
    assert span == pytest.approx(1000.0), (
        f"草稿的 1 m 在毫米制空模型里应当是 1000 mm，实际 {span}")


def test_the_coincidence_tolerance_is_the_same_physical_distance():
    """重合容差是**物理距离**，不能随单位制变。

    写死的 1e-6 只在米制下是 1 微米；毫米制下同一个数是 1 纳米，紧了
    1000 倍——两个实际重合的节点会被当成两个，用户得到一根没连上的杆。
    """
    from multimodal_contract import MODEL_COINCIDENCE_M
    from units import of

    metres = MODEL_COINCIDENCE_M / of({"units": "N-m-Pa"}).length_to_m
    millimetres = MODEL_COINCIDENCE_M / of({"units": "N-mm-MPa"}).length_to_m
    assert metres * of({"units": "N-m-Pa"}).length_to_m == pytest.approx(
        millimetres * of({"units": "N-mm-MPa"}).length_to_m)


def test_a_coincident_node_is_still_caught_in_a_millimetre_model():
    """容差放宽之后，重合保护不能失效——那是另一条该守住的规则。"""
    baseline = mm_baseline()
    draft = ready_draft()
    # 草稿节点 1 在原点，基线节点 1 也在原点：并进去必须被拦
    prepared = None
    with pytest.raises(DraftCommitError, match="重合"):
        prepared, _ = prepare_commit(draft, baseline)
    assert prepared is None
