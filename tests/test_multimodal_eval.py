"""MM2-08 release corpus and deterministic score tests."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

from multimodal_contract import canonical_digest, file_digest, load_manifest
from multimodal_eval.evaluate import _numeric_equal, evaluate_manifest
from multimodal_eval.provider_smoke import PROVIDERS, run
from multimodal_eval.reference_topology import FORMAT as REFERENCE_FORMAT, check_reference_topology
from multimodal_eval.real_world_eval import (FORMAT, PROMPT_HASH, SCHEMA_HASH,
                                              audit_dataset, create_templates,
                                              draft_to_prediction,
                                              evaluate_real_world, freeze_manifest,
                                              input_context, recognize, replay_real_world)

ROOT = Path(__file__).resolve().parents[1] / "multimodal_eval"


def test_release_manifest_has_30_licensed_bound_cases_and_required_subsets():
    manifest = load_manifest(ROOT / "manifest.json")
    gate = [case for case in manifest["cases"] if case["gate"]]
    assert len(gate) >= 30
    assert all(case["source"] == "project-generated-original" and
               case["license"] == "CC0-1.0" for case in gate)
    labels = {label for case in gate for label in case["subset_labels"]}
    assert {"clear", "handdrawn", "missing_dimension", "intersection",
            "blurred_symbol", "photo_noise"} <= labels
    for case in gate:
        fixture = json.loads((ROOT / case["response_fixture_path"])
                             .read_text(encoding="utf-8"))
        assert fixture["redaction"] == {
            "credentials": "removed", "user_metadata": "removed"}
        assert fixture["request_fingerprint"]["prompt_hash"] == case["prompt_hash"]
        assert fixture["request_fingerprint"]["schema_hash"] == case["schema_hash"]


def test_offline_manifest_gate_scores_every_required_target_and_passes():
    report = evaluate_manifest(ROOT / "manifest.json")
    assert report["gate_cases"] >= 30
    assert report["passed"] and report["failures"] == []
    for metric in ("node_f1", "member_f1", "intersection_accuracy",
                   "support_f1", "load_f1", "load_numeric_unit_accuracy",
                   "scale_accuracy", "issue_recall"):
        assert report["metrics"][metric] == 1.0


def test_missing_credentials_are_explicit_skips_and_never_enter_offline_gate(monkeypatch):
    for env_var in PROVIDERS.values():
        monkeypatch.delenv(env_var, raising=False)
    # 也要挡掉密钥文件回退，否则这条测试的结果取决于跑它的机器上有没有
    # deepseek.key——在 CI 上过、在开发者机器上挂，是最消耗信任的那种不稳定。
    import credentials
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: None)
    report = run()
    assert report["excluded_from_offline_gate"] is True
    assert {item["provider"] for item in report["results"]} == set(PROVIDERS)
    assert all(item["status"] == "SKIPPED_NO_CREDENTIALS"
               for item in report["results"])


def test_gate_fails_when_bound_fixtures_lose_all_nodes(tmp_path):
    for directory in ("images", "ground_truth", "responses"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        response_path = tmp_path / case["response_fixture_path"]
        fixture = json.loads(response_path.read_text(encoding="utf-8"))
        fixture["prediction"]["nodes"] = []
        response_path.write_text(json.dumps(fixture), encoding="utf-8")
        case["response_hash"] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate_manifest(manifest_path)
    assert not report["passed"]
    assert {"node_f1", "member_f1"} <= set(report["failures"])


def test_load_numeric_comparison_converts_units_and_checks_zero():
    truth = {"collection": "nodal_loads", "load": [0, -1000, 0, 0, 0, 0],
             "force_unit": "N", "moment_unit": "N*m"}
    prediction = {"collection": "nodal_loads", "load": [0, -1, 0, 0, 0, 0],
                  "force_unit": "kN", "moment_unit": "kN*m"}
    assert _numeric_equal(prediction, truth)
    prediction["load"][0] = 1.0e-5
    assert not _numeric_equal(prediction, truth)


def _real_image(root: Path, image_id: str = "site_photo") -> Path:
    path = root / "images" / f"{image_id}.png"
    path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "images" / "seed_01_portal.png", path)
    return path


def reference_triangle():
    return {"width_px": 101, "height_px": 101,
        "topology_reference": {"format": REFERENCE_FORMAT, "scope": "all_member_pairs", "status": "verified"},
        "nodes": [{"id": 1, "point": [.1, .9]}, {"id": 2, "point": [.5, .1]}, {"id": 3, "point": [.9, .9]}],
        "members": [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 2, "j": 3}, {"id": 3, "i": 3, "j": 1}],
        "intersections": [
            {"id": "a", "members": [1, 2], "point": [.5, .1], "decision": "connect", "node_id": 2},
            {"id": "b", "members": [1, 3], "point": [.1, .9], "decision": "connect", "node_id": 1},
            {"id": "c", "members": [2, 3], "point": [.9, .9], "decision": "connect", "node_id": 3}]}


@pytest.mark.parametrize("shape", ["triangle", "collinear_shared", "t_connect", "x_connect", "x_cross", "disjoint"])
def test_reference_pairwise_contract_accepts_explicit_endpoint_t_and_x_decisions(shape):
    """完整参考须覆盖共享端点、共线接头及 T/X 相交；不相连交叉不能自动当作连接。"""
    from copy import deepcopy
    truth = reference_triangle()
    if shape == "collinear_shared":
        truth["nodes"][1]["point"] = [.5, .9]
        truth["members"].pop()
        truth["intersections"] = [{"id": "a", "members": [1, 2], "point": [.5, .9], "decision": "connect", "node_id": 2}]
    elif shape != "triangle":
        truth["nodes"] = [{"id": 1, "point": [.1, .5]}, {"id": 2, "point": [.9, .5]},
                          {"id": 3, "point": [.5, .1]}, {"id": 4, "point": [.5, .9]}, {"id": 5, "point": [.5, .5]}]
        truth["members"] = [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 3, "j": 4}]
        truth["intersections"] = [{"id": "x", "members": [2, 1], "point": [.5, .5],
                                   "decision": "cross" if shape == "x_cross" else "connect",
                                   "node_id": None if shape == "x_cross" else 5}]
        if shape == "t_connect":
            truth["members"][1]["i"] = 5
        if shape == "disjoint":
            truth["nodes"][2]["point"] = [.1, .7]
            truth["nodes"][3]["point"] = [.9, .7]
            truth["intersections"] = []
    original = deepcopy(truth)
    assert check_reference_topology(truth) == {"valid": True, "legacy": False, "errors": []}
    assert truth == original


@pytest.mark.parametrize(("change", "error"), [
    ("missing", "missing_intersection"), ("duplicate_pair", "duplicate_intersection_pair"),
    ("duplicate_id", "duplicate_intersection_id"), ("wrong_point", "intersection_point_mismatch"),
    ("unknown", "intersection_decision_unverified"), ("wrong_node", "connection_node_mismatch"),
    ("shared_cross", "cross_has_connected_node"), ("foreign_pair", "unexpected_intersection"),
    ("overlap", "overlapping_members"), ("duplicate_geometry", "duplicate_member_geometry"),
    ("dangling", "invalid_reference_geometry"), ("boolean_point", "invalid_reference_geometry"),
    ("pending", "contract_not_verified"), ("empty", "invalid_reference_geometry")])
def test_reference_incompleteness_is_rejected_without_generating_answers(change, error):
    """三角桁架空交点参考曾导致额外预测；错误/缺项不能被校验器自动补齐或静默放过。"""
    from copy import deepcopy
    truth = reference_triangle()
    if change == "missing":
        truth["intersections"].pop()
    elif change == "duplicate_pair":
        truth["intersections"].append({**truth["intersections"][0], "id": "extra"})
    elif change == "duplicate_id":
        truth["intersections"][1]["id"] = "a"
    elif change == "wrong_point":
        truth["intersections"][0]["point"] = [.7, .7]
    elif change == "unknown":
        truth["intersections"][0]["decision"] = "unknown"
    elif change == "wrong_node":
        truth["intersections"][0]["node_id"] = 1
    elif change == "shared_cross":
        truth["intersections"][0].update(decision="cross", node_id=None)
    elif change == "foreign_pair":
        truth["intersections"][0]["members"] = [1, 99]
    elif change == "overlap":
        truth["nodes"].append({"id": 4, "point": [.3, .5]})
        truth["members"].append({"id": 4, "i": 1, "j": 4})
    elif change == "duplicate_geometry":
        truth["members"].append({"id": 4, "i": 2, "j": 1})
    elif change == "dangling":
        truth["members"][0]["i"] = 99
    elif change == "boolean_point":
        truth["nodes"][0]["point"][0] = True
    elif change == "pending":
        truth["topology_reference"]["status"] = "pending"
    else:
        truth["nodes"] = truth["members"] = []
    original = deepcopy(truth)
    result = check_reference_topology(truth)
    assert not result["valid"] and error in {item["code"] for item in result["errors"]}
    assert truth == original


def test_reference_position_rounding_has_one_pixel_bound_without_changing_score_tolerance():
    """人工坐标舍入允许一像素误差，不能借参考校验放宽原五像素预测匹配阈值。"""
    from multimodal_contract import point_match_threshold
    truth = reference_triangle()
    truth["intersections"][0]["point"][0] += .009
    assert check_reference_topology(truth)["valid"]
    truth["intersections"][0]["point"][0] += .002
    assert not check_reference_topology(truth)["valid"]
    assert point_match_threshold(101, 101) == 5


def test_real_world_templates_never_claim_consent_or_verified_truth(tmp_path):
    _real_image(tmp_path)
    result = create_templates(tmp_path)
    assert result["images"] == 1
    metadata = json.loads((tmp_path / "metadata" / "site_photo.json")
                          .read_text(encoding="utf-8"))
    truth = json.loads((tmp_path / "ground_truth" / "site_photo.json")
                       .read_text(encoding="utf-8"))
    assert metadata["format"] == FORMAT
    assert metadata["consent_to_evaluate"] is False
    assert truth["annotation_status"] == "pending"
    assert metadata["reference_contract"] == truth["topology_reference"]["format"] == REFERENCE_FORMAT
    assert truth["topology_reference"]["status"] == "pending"
    status = audit_dataset(tmp_path)
    assert status["ready_count"] == 0
    assert {"consent_not_confirmed", "ground_truth_not_verified",
            "missing_response"} <= set(status["cases"][0]["reasons"])


def _complete_real_case(tmp_path):
    image = _real_image(tmp_path)
    create_templates(tmp_path)
    metadata_path = tmp_path / "metadata" / "site_photo.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(consent_to_evaluate=True, subset_labels=["clear"])
    metadata["work_plane"]["status"] = "confirmed"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    truth_path = tmp_path / "ground_truth" / "site_photo.json"
    fixture_truth = json.loads((ROOT / "ground_truth" / "seed_01_portal.json")
                               .read_text(encoding="utf-8"))
    fixture_truth.update(image_id="site_photo", annotation_status="verified")
    # 测试专用参考在模拟响应之前补齐门式结构的两个共享端点，不修改冻结语料。
    fixture_truth["topology_reference"] = {"format": REFERENCE_FORMAT, "scope": "all_member_pairs", "status": "verified"}
    fixture_truth["intersections"] = [
        {"id": "shared-1", "members": [1, 2], "point": fixture_truth["nodes"][1]["point"], "decision": "connect", "node_id": 2},
        {"id": "shared-2", "members": [2, 3], "point": fixture_truth["nodes"][2]["point"], "decision": "connect", "node_id": 3},
    ]
    fixture_truth.update(image_hash=file_digest(image),
                         input_context_hash=canonical_digest(input_context(metadata)))
    truth_path.write_text(json.dumps(fixture_truth), encoding="utf-8")
    response_path = tmp_path / "responses" / "site_photo.json"
    fixture = json.loads((ROOT / "responses" / "seed_01_portal.json")
                         .read_text(encoding="utf-8"))
    fixture["prediction"]["image_id"] = "site_photo"
    fixture["request_fingerprint"] = {
        "provider": "openai", "model": "test-model",
        "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH,
        "image_hash": file_digest(image),
        "input_context_hash": canonical_digest(input_context(metadata)),
        "ground_truth_hash": file_digest(truth_path), "reference_contract": REFERENCE_FORMAT,
    }
    fixture.update(outcome="PASS", runtime={"attempts": 1, "duration_ms": 125.0})
    response_path.parent.mkdir()
    response_path.write_text(json.dumps(fixture), encoding="utf-8")
    return image, metadata_path, truth_path, response_path


@pytest.mark.parametrize("legacy", [False, True])
def test_freeze_again_preserves_existing_manifest_bytes_and_legacy_version(tmp_path, legacy):
    """增加参考版本字段后再次 freeze 不能重写既有历史 manifest 或悄悄迁移旧契约。"""
    _, metadata_path, truth_path, response_path = _complete_real_case(tmp_path)
    if legacy:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.pop("reference_contract")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        truth.pop("topology_reference")
        truth["input_context_hash"] = canonical_digest(input_context(metadata))
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        response["request_fingerprint"].pop("ground_truth_hash")
        response["request_fingerprint"].pop("reference_contract")
        response_path.write_text(json.dumps(response), encoding="utf-8")
    manifest = freeze_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    if legacy:
        # 只在临时测试目录模拟早期 manifest，没有触及真实冻结语料。
        manifest["cases"][0].pop("reference_contract")
        path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")
    before = path.read_bytes()
    assert freeze_manifest(tmp_path) == manifest and path.read_bytes() == before


@pytest.mark.parametrize("change", ["missing_pair", "missing_contract", "deleted_markers", "pending", "wrong_decision"])
def test_new_calls_require_complete_verified_topology_before_client_creation(tmp_path, monkeypatch, change):
    """完整性标记被删、共享端点漏标或未确认时，付费接口客户端都不能创建。"""
    _, metadata_path, truth_path, response_path = _complete_real_case(tmp_path)
    response_path.unlink()
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if change == "missing_pair":
        truth["intersections"].pop()
    elif change in ("missing_contract", "deleted_markers"):
        truth.pop("topology_reference")
        if change == "deleted_markers":
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("reference_contract")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif change == "pending":
        truth["topology_reference"]["status"] = "pending"
    else:
        truth["intersections"][0].update(decision="cross", node_id=None)
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    before = truth_path.read_bytes()
    monkeypatch.setattr("sketch_parser.SketchParser.from_env", lambda *a, **k: pytest.fail("不完整参考不能创建客户端"))
    result = recognize(tmp_path, provider="openai")
    assert result["results"][0]["status"] == "SKIPPED_NOT_READY"
    assert "ground_truth_topology_incomplete" in result["results"][0]["reasons"]
    assert truth_path.read_bytes() == before and not response_path.exists()


@pytest.mark.parametrize("change", ["changed_truth", "unbound_response"])
def test_reference_hash_change_after_response_blocks_freeze_and_reuse(tmp_path, monkeypatch, change):
    """获得响应后改参考或删掉调用前参考哈希，不能重新冻结成独立结果或重新付费覆盖。"""
    _, _, truth_path, response_path = _complete_real_case(tmp_path)
    if change == "changed_truth":
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        truth["notes"] = "after response"
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
    else:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        response["request_fingerprint"].pop("ground_truth_hash")
        response_path.write_text(json.dumps(response), encoding="utf-8")
    assert audit_dataset(tmp_path)["ready_count"] == 0
    with pytest.raises(ValueError, match="尚未完整冻结"):
        freeze_manifest(tmp_path)
    monkeypatch.setattr("sketch_parser.SketchParser._call_llm", lambda *a, **k: pytest.fail("旧响应不能覆盖"))
    before = response_path.read_bytes()
    assert recognize(tmp_path, provider="openai", model="test-model")["results"][0]["status"] == "STALE_RESPONSE"
    assert response_path.read_bytes() == before


def test_reference_hash_is_captured_before_parse_not_after_response(tmp_path, monkeypatch):
    """调用过程中修改参考也不能把事后参考当成调用前绑定，解析失败的首响应仍保留。"""
    from sketch_parser import SketchParser, V2ParseResult
    _, _, truth_path, response_path = _complete_real_case(tmp_path)
    response_path.unlink()
    original_hash = file_digest(truth_path)
    parser = SketchParser(api_key="test-key", model="test-model")
    monkeypatch.setattr(SketchParser, "from_env", lambda *a, **k: parser)

    def failed_parse(*a, **k):
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        truth["notes"] = "changed during request"
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        return V2ParseResult(attempts=1, duration_ms=1, errors=["模拟失败"], raw_responses=["invalid"])

    monkeypatch.setattr(parser, "parse_v2_with_retry", failed_parse)
    assert recognize(tmp_path, provider="openai", model="test-model", max_repairs=0)["results"][0]["status"] == "FAIL"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["request_fingerprint"]["ground_truth_hash"] == original_hash != file_digest(truth_path)
    assert "response_ground_truth_mismatch" in audit_dataset(tmp_path)["cases"][0]["reasons"]


def test_legacy_reference_warnings_keep_frozen_online_metrics_and_bytes(monkeypatch):
    """历史数值图漏共享端点参考时只注明限制，不能补旧标注、删交点或改写旧在线失败。"""
    root = ROOT / "public_cases/numeric_01"
    monkeypatch.setattr("sketch_parser.SketchParser._call_llm", lambda *a, **k: pytest.fail("历史参考检查不能调用模型"))
    before = {p: p.read_bytes() for folder in ("ground_truth", "responses") for p in (root / folder).glob("*.json")}
    audit = audit_dataset(root)
    assert audit["ready_count"] == 2
    assert all("legacy_topology_reference_unchecked" in c["warnings"] for c in audit["cases"])
    report = evaluate_real_world(root)
    frozen = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["metrics"] == frozen["metrics"] and not report["passed"]
    assert report["reference_contract"]["checked_cases"] == 0
    assert len(report["reference_contract"]["legacy_cases"]) == 2
    assert replay_real_world(root)["reference_contract"] == report["reference_contract"]
    assert before == {p: p.read_bytes() for p in before}


def test_real_world_freeze_hash_binds_a_complete_case(tmp_path):
    """防止修改冻结响应后仍能报告旧样本的识别成绩。"""
    image, _, _, response_path = _complete_real_case(tmp_path)

    manifest = freeze_manifest(tmp_path)
    case = manifest["cases"][0]
    assert case["image_hash"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert case["source"] == "user-provided"
    assert evaluate_manifest(tmp_path / "manifest.json")["gate_cases"] == 1
    response_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希漂移"):
        evaluate_manifest(tmp_path / "manifest.json")


@pytest.mark.parametrize("change", ["enabled", "prompt"])
def test_action_review_mode_and_prompt_changes_prevent_reuse_or_new_paid_calls(tmp_path, monkeypatch, change):
    """改变可选复核模式或其提示后不能复用旧响应，也不能静默覆盖冻结证据并重新付费。"""
    from multimodal_eval.real_world_eval import ACTION_PROMPT_HASH, PIPELINE_HASH
    from sketch_parser import SketchParser
    _, _, _, response_path = _complete_real_case(tmp_path)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    parser = SketchParser(api_key="test", model="test-model")
    response["request_fingerprint"].update(
        pipeline_hash=PIPELINE_HASH, request_options=parser.request_options(),
        review_actions=change == "prompt", action_prompt_hash="old" if change == "prompt" else None)
    response_path.write_text(json.dumps(response), encoding="utf-8")
    before = response_path.read_bytes()
    monkeypatch.setattr(SketchParser, "_call_llm", lambda *args, **kwargs: pytest.fail("不应发起接口调用"))
    result = recognize(tmp_path, provider="openai", model="test-model", review_actions=True)
    assert result["results"][0]["status"] == "STALE_RESPONSE"
    assert response_path.read_bytes() == before
    assert ACTION_PROMPT_HASH


def test_frozen_two_stage_development_round_keeps_raws_calls_and_inaccurate_moment(monkeypatch):
    """旧在线力矩错误不能被新像素回放覆盖；回放提升要注明零新增调用，支座错误仍保留。"""
    root = ROOT / "public_cases/action_review_01"
    info = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / "pipeline_snapshot.json").read_text(encoding="utf-8"))
    assert canonical_digest(snapshot) == info["pipeline_hash"]
    assert "sketch_action_review.py" in snapshot
    monkeypatch.setattr("sketch_parser.SketchParser._call_llm",
                        lambda *args, **kwargs: pytest.fail("历史回放不能新增调用"))
    online, replay = evaluate_real_world(root), replay_real_world(root)
    assert online["runtime"]["total_calls"] == 6
    assert online["action_review"] == {"attempted_cases": 3, "completed_cases": 3, "failed_or_missing_cases": 0}
    assert replay["new_api_calls"] == 0
    assert online["metrics"]["node_f1"] == pytest.approx(8 / 9)
    assert replay["metrics"]["node_f1"] == replay["metrics"]["member_f1"] == 1
    assert online["metrics"]["support_f1"] == replay["metrics"]["support_f1"] == .8
    assert not online["passed"] and not replay["passed"]
    for case in online["case_reports"]:
        name = case["image_id"]
        source = ROOT / "public_cases" / ("validation_01" if name == "various_loads" else "")
        assert (root / "ground_truth" / f"{name}.json").read_bytes() == (source / "ground_truth" / f"{name}.json").read_bytes()
        response = json.loads((root / "responses" / f"{name}.json").read_text(encoding="utf-8"))
        assert len(response["raw_responses"]) == len(response["call_metadata"]) == 2
        assert [m["stage"] for m in response["call_metadata"]] == ["geometry", "action_review"]
        if name == "various_loads":
            assert len(json.loads(response["raw_responses"][0])["image_model"]["nodes"]) == 2
            assert len(response["prediction"]["nodes"]) == 4
            assert case["metrics"]["node_f1"] == .75
            assert case["extra_nodes"] and case["missing_nodes"]


def test_failed_optional_stage_is_reported_separately_from_usable_base_parse(tmp_path):
    """基础草稿进入审核不能掩盖复核阶段失败，也不能把两次调用统计成一次。"""
    _, _, _, response_path = _complete_real_case(tmp_path)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["request_fingerprint"]["review_actions"] = True
    response["action_review"] = {"status": "failed", "message": "请人工核对作用点。"}
    response["runtime"]["attempts"] = 2
    response_path.write_text(json.dumps(response), encoding="utf-8")
    freeze_manifest(tmp_path)
    report = evaluate_real_world(tmp_path)
    assert report["runtime"]["parse_successes"] == 1
    assert report["runtime"]["total_calls"] == 2
    assert report["action_review"]["completed_cases"] == 0
    assert report["action_review"]["failed_or_missing_cases"] == 1


@pytest.mark.parametrize("failure", ["malformed", "missing"])
def test_optional_replay_failure_preserves_valid_base_without_network_calls(tmp_path, monkeypatch, failure):
    """可选复核回放失败曾丢弃基础草稿，使回放与桌面失败处理不一致。"""
    _, _, _, response_path = _complete_real_case(tmp_path)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    prediction = response["prediction"]
    base = {"image_model": {
        "nodes": [{"id": n["id"], "u": n["point"][0], "v": n["point"][1]} for n in prediction["nodes"]],
        "members": prediction["members"], "supports": prediction["supports"], "load_cases": []}, "entities": []}
    response["raw_responses"] = [json.dumps(base)] + (["not-json"] if failure == "malformed" else [])
    response["request_fingerprint"]["review_actions"] = True
    response["action_review"] = {"status": "failed", "message": "冻结的可选复核失败，请人工核对。"}
    response["runtime"]["attempts"] = 2
    response_path.write_text(json.dumps(response), encoding="utf-8")
    freeze_manifest(tmp_path)
    monkeypatch.setattr("sketch_parser.SketchParser._call_llm",
                        lambda *args, **kwargs: pytest.fail("回放不应发起接口调用"))
    replay = replay_real_world(tmp_path)
    assert replay["new_api_calls"] == 0 and replay["replay_successes"] == 1
    assert replay["action_review"] == {"attempted_cases": 1, "completed_cases": 0, "failed_or_missing_cases": 1}
    assert replay["counts"]["nodes"]["tp"] == len(prediction["nodes"])


def test_second_action_round_retains_online_optional_failure_and_current_replay_fix(monkeypatch):
    """兼容空置信度的修复只能体现在新回放中，不能覆盖首次在线阶段失败和旧回放报告。"""
    root = ROOT / "public_cases/action_review_02"
    original = {p: p.read_bytes() for folder in ("responses", "ground_truth") for p in (root / folder).glob("*.json")}
    info = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / "pipeline_snapshot.json").read_text(encoding="utf-8"))
    assert canonical_digest(snapshot) == info["pipeline_hash"]
    assert "sketch_moment_refinement.py" in snapshot
    online = evaluate_real_world(root)
    assert online["runtime"]["total_calls"] == 6 and online["runtime"]["parse_successes"] == 3
    assert online["action_review"]["completed_cases"] == 2 and online["action_review"]["failed_or_missing_cases"] == 1
    assert online["metrics"]["node_f1"] == online["metrics"]["member_f1"] == online["metrics"]["support_f1"] == 1
    assert not online["passed"] and online["metrics"]["load_f1"] == 0
    old_replay = json.loads((root / "replay_report.json").read_text(encoding="utf-8"))
    assert old_replay["metrics"]["node_f1"] == .8
    monkeypatch.setattr("sketch_parser.SketchParser._call_llm",
                        lambda *args, **kwargs: pytest.fail("修复回放不得新增调用"))
    current = replay_real_world(root)
    assert current["action_review"]["completed_cases"] == 3 and current["new_api_calls"] == 0
    assert current["metrics"]["node_f1"] == current["metrics"]["member_f1"] == current["metrics"]["support_f1"] == 1
    assert not current["passed"]
    assert all(p.read_bytes() == data for p, data in original.items())


def test_postcall_replay_snapshot_is_bound_separately_from_original_online_pipeline():
    """响应暴露兼容问题后的源码与回放必须另行冻结，不能替换调用前快照或首次失败。"""
    root = ROOT / "public_cases/action_review_02"
    report = json.loads((root / "current_replay.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / report["replay_snapshot"]).read_text(encoding="utf-8"))
    assert canonical_digest(snapshot) == report["replay_implementation_hash"]
    assert file_digest(root / "report.json") == report["original_report_hash"]
    assert file_digest(root / "replay_report.json") == report["original_replay_hash"]
    info = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    assert info["pipeline_hash"] != report["pipeline_hash"]
    assert report["new_api_calls"] == 0


def test_review_edit_evidence_keeps_symbol_load_and_original_metrics():
    """界面分组不能删除问题或无数值荷载来美化成绩，新回放须绑定原记录和完整源码。"""
    original = ROOT / "public_cases/action_review_02"
    root = ROOT / "public_cases/review_edit_01"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / report["replay_snapshot"]).read_text(encoding="utf-8"))
    previous = json.loads((original / "current_replay.json").read_text(encoding="utf-8"))
    assert canonical_digest(snapshot) == report["replay_implementation_hash"]
    assert len(snapshot) == 14 and "desktop/issue_panel.py" in snapshot
    assert file_digest(original / "report.json") == report["original_online_report_hash"]
    assert file_digest(original / "current_replay.json") == report["original_postcall_replay_hash"]
    assert report["metrics"] == previous["metrics"] and report["new_api_calls"] == 0 and not report["passed"]
    assert len(report["native_ui"]) == 4
    assert all(i["raw_open_records"] > i["visible_rows"] and i["commit_blocked"] for i in report["native_ui"])


@pytest.mark.parametrize("change", ["image", "crop", "rotation", "plane", "offset"])
def test_real_world_input_changes_block_old_truth_and_network_calls(tmp_path, monkeypatch, change):
    """同名图片或输入参数变化必须重新标注，不能复用旧真值或继续付费调用。"""
    from PIL import Image
    image, metadata_path, _, response_path = _complete_real_case(tmp_path)
    response_path.unlink()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if change == "image":
        with Image.open(image) as pixels:
            pixels.putpixel((0, 0), (0, 0, 0))
            pixels.save(image)
    elif change in ("crop", "rotation"):
        metadata["preprocessing"] = {"crop": [0, 0, 100, 80]} if change == "crop" else {"rotation_quarters_cw": 2}
    elif change == "plane":
        metadata["work_plane"]["plane"] = "XY"
    else:
        metadata["work_plane"]["offset"] = 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr("sketch_parser.SketchParser.from_env",
                        lambda *a, **k: pytest.fail("失效真值不允许调用接口"))
    assert recognize(tmp_path, provider="openai")["results"][0]["status"] == "SKIPPED_NOT_READY"
    with pytest.raises(ValueError, match="尚未完整冻结"):
        freeze_manifest(tmp_path)


@pytest.mark.parametrize("field", ["consent_to_evaluate", "annotation_status"])
def test_real_world_missing_authorization_or_annotation_never_calls_provider(tmp_path, monkeypatch, field):
    """防止评测命令在未授权或未独立标注时发送原图。"""
    _, metadata_path, truth_path, response_path = _complete_real_case(tmp_path)
    response_path.unlink()
    path = metadata_path if field == "consent_to_evaluate" else truth_path
    value = json.loads(path.read_text(encoding="utf-8"))
    value[field] = False if field == "consent_to_evaluate" else "pending"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr("sketch_parser.SketchParser.from_env",
                        lambda *a, **k: pytest.fail("未确认输入不允许调用接口"))
    assert recognize(tmp_path, provider="openai")["results"][0]["status"] == "SKIPPED_NOT_READY"


def test_real_world_failure_is_frozen_scored_and_never_automatically_retried(tmp_path, monkeypatch):
    """接口失败必须保存到同一评分分母，再次执行不应重复收费。"""
    from sketch_parser import SketchParser, V2ParseResult
    _, _, _, response_path = _complete_real_case(tmp_path)
    response_path.unlink()
    parser = SketchParser(provider="openai", api_key="private-test-key", model="test-model")
    monkeypatch.setattr(SketchParser, "from_env", lambda *a, **k: parser)
    calls = []

    def fail_parse(*args, **kwargs):
        calls.append(kwargs)
        return V2ParseResult(attempts=1, duration_ms=240.0,
                             errors=["接口失败 private-test-key"], raw_responses=["private-test-key"])

    monkeypatch.setattr(parser, "parse_v2_with_retry", fail_parse)
    first = recognize(tmp_path, provider="openai", model="test-model", max_repairs=0)
    assert first["results"][0]["status"] == "FAIL"
    assert calls[0]["max_repairs"] == 0
    assert "private-test-key" not in response_path.read_text(encoding="utf-8")
    frozen_bytes = response_path.read_bytes()
    assert recognize(tmp_path, provider="openai", model="test-model")["results"][0]["status"] == "SKIPPED_EXISTS"
    assert len(calls) == 1 and response_path.read_bytes() == frozen_bytes
    freeze_manifest(tmp_path)
    report = evaluate_real_world(tmp_path)
    assert report["metrics"]["node_f1"] == 0
    assert report["runtime"] == {"attempted_cases": 1, "parse_successes": 0,
                                 "parse_failures": 1, "total_calls": 1,
                                 "median_duration_ms": 240.0, "total_duration_ms": 240.0}


def test_real_world_metadata_drift_rejects_frozen_score(tmp_path):
    """冻结后修改授权说明或工作平面不能沿用旧评分。"""
    _, metadata_path, _, _ = _complete_real_case(tmp_path)
    freeze_manifest(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["consent_to_evaluate"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="授权或输入参数哈希漂移"):
        evaluate_real_world(tmp_path)


def test_public_replay_fixes_frozen_responses_without_calling_provider(monkeypatch):
    """真实公开响应复现支座默认名冲突；修复回放不能覆盖首轮失败或新增接口调用。"""
    monkeypatch.setattr("sketch_parser.SketchParser._call_llm",
                        lambda *a, **k: pytest.fail("离线回放禁止调用接口"))
    root = ROOT / "public_cases"
    original = {path: path.read_bytes() for path in (root / "responses").glob("*.json")}
    baseline = evaluate_real_world(root)
    replay = replay_real_world(root)
    assert baseline["runtime"]["parse_failures"] == 5
    assert replay["new_api_calls"] == 0 and replay["replay_successes"] == 4
    assert replay["mode"] == "offline-replay"
    assert replay["gate_cases"] == 5
    assert replay["metrics"]["load_numeric_unit_accuracy"] is None
    assert all(path.read_bytes() == value for path, value in original.items())


@pytest.mark.parametrize("round_name", ["round_02", "round_03", "round_04", "round_05"])
def test_new_public_rounds_keep_original_pixels_and_independent_truth(round_name):
    """新轮次只能改变识别过程，不能偷偷改原图或评分真值来提高成绩。"""
    root = ROOT / "public_cases"
    round_root = root / round_name
    for folder in ("images", "ground_truth"):
        for path in (root / folder).iterdir():
            assert (round_root / folder / path.name).read_bytes() == path.read_bytes()
    report = evaluate_real_world(round_root)
    assert report["runtime"]["total_calls"] == 5 and report["gate_cases"] == 5
    assert report["metrics"]["load_numeric_unit_accuracy"] is None
    assert all(item["call_metadata"][0]["finish_reason"] == "stop" for item in report["case_outcomes"])


def test_current_pixel_replay_is_separate_from_frozen_online_score(monkeypatch):
    """纵坐标倒置修复只在离线回放体现，不能覆盖首轮在线分数或声称又做了一次识别。"""
    root = ROOT / "public_cases" / "round_03"
    from multimodal_contract import canonical_digest
    snapshot = json.loads((root / "pipeline_snapshot.json").read_text(encoding="utf-8"))
    round_info = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    assert canonical_digest(snapshot) == round_info["pipeline_hash"]
    monkeypatch.setattr("sketch_parser.SketchParser._call_llm",
                        lambda *a, **k: pytest.fail("回放不能调用接口"))
    original = {p: p.read_bytes() for p in (root / "responses").glob("*.json")}
    online = evaluate_real_world(root)
    replay = replay_real_world(root)
    assert online["metrics"]["node_f1"] == 0.5
    historical_replay = json.loads((root / "replay.json").read_text(encoding="utf-8"))
    assert historical_replay["metrics"]["node_f1"] == 0.75
    assert replay["metrics"]["node_f1"] == 1.0
    assert replay["new_api_calls"] == 0 and replay["replay_successes"] == 5
    assert replay["pipeline_hash"]
    assert all(p.read_bytes() == data for p, data in original.items())


@pytest.mark.parametrize("changed", ["request_options", "pipeline_hash"])
def test_request_or_pipeline_drift_never_reuses_or_overwrites_a_response(tmp_path, monkeypatch, changed):
    """改变视觉推理设置或像素后处理后，应提示旧响应过期，不能沿用旧预测或重复付费覆盖。"""
    from multimodal_eval.real_world_eval import PIPELINE_HASH
    from sketch_parser import SketchParser
    _, _, _, response_path = _complete_real_case(tmp_path)
    value = json.loads(response_path.read_text(encoding="utf-8"))
    value["request_fingerprint"].update(request_options=SketchParser().request_options(), pipeline_hash=PIPELINE_HASH)
    value["request_fingerprint"][changed] = {"max_tokens": 2000} if changed == "request_options" else "old-pipeline"
    response_path.write_text(json.dumps(value), encoding="utf-8")
    original = response_path.read_bytes()
    monkeypatch.setattr("sketch_parser.SketchParser.from_env",
                        lambda *a, **k: pytest.fail("不能覆盖冻结响应"))
    result = recognize(tmp_path, provider="openai", model="test-model")
    assert result["results"][0]["status"] == "STALE_RESPONSE"
    assert response_path.read_bytes() == original


def test_independent_mixed_load_failure_is_preserved_in_the_score_denominator():
    """新混合荷载图漏掉集中作用点时不能删除失败或把解析成功记为识别正确。"""
    from multimodal_contract import canonical_digest, file_digest
    root = ROOT / "public_cases" / "validation_01"
    record = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / "pipeline_snapshot.json").read_text(encoding="utf-8"))
    assert canonical_digest(snapshot) == record["pipeline_hash"]
    assert file_digest(root / "ground_truth/various_loads.json") == record["annotations_fixed_before_calls"]["various_loads"]
    report = evaluate_real_world(root)
    assert report["gate_cases"] == 1 and report["runtime"]["total_calls"] == 1
    assert report["runtime"]["parse_successes"] == 1
    assert report["metrics"]["node_f1"] == report["metrics"]["member_f1"] == 0
    assert report["metrics"]["load_numeric_unit_accuracy"] is None
    assert not report["passed"]


@pytest.mark.parametrize("round_name", ["load_points_01", "load_points_crop_01"])
def test_mixed_load_development_failures_and_precall_annotations_remain_frozen(round_name):
    """全图或裁剪后的识别仍漏点时，保留首次失败、调用前参考标注和重复问题的错误计数。"""
    from multimodal_contract import canonical_digest, file_digest
    root = ROOT / "public_cases" / round_name
    record = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / "pipeline_snapshot.json").read_text(encoding="utf-8"))
    assert canonical_digest(snapshot) == record["pipeline_hash"]
    assert file_digest(root / "ground_truth/various_loads.json") == record["annotations_fixed_before_calls"]["various_loads"]
    raw_path = root / "responses/various_loads.json"
    original = raw_path.read_bytes()
    report = evaluate_real_world(root)
    assert report["gate_cases"] == 1 and report["runtime"]["total_calls"] == 1
    assert report["metrics"]["node_f1"] == report["metrics"]["member_f1"] == 0
    assert report["metrics"]["issue_false_positives"] == 5
    assert not report["passed"] and raw_path.read_bytes() == original
    if round_name == "load_points_crop_01":
        old = json.loads((ROOT / "public_cases/validation_01/ground_truth/various_loads.json").read_text(encoding="utf-8"))
        truth = json.loads((root / "ground_truth/various_loads.json").read_text(encoding="utf-8"))
        crop = record["crop_fixed_before_call"]
        assert len(truth["nodes"]) == len(old["nodes"])
        for new_node, old_node in zip(truth["nodes"], old["nodes"], strict=True):
            assert new_node["point"][0] * truth["width_px"] + crop[0] == pytest.approx(old_node["point"][0] * old["width_px"])
            assert new_node["point"][1] * truth["height_px"] + crop[1] == pytest.approx(old_node["point"][1] * old["height_px"])


def test_duplicate_numeric_predictions_cannot_score_above_one(tmp_path):
    """相同荷载重复输出曾使正确数大于真值数，数值准确率超过 100%。"""
    _, _, truth_path, response_path = _complete_real_case(tmp_path)
    load = {"collection": "nodal_loads", "node": 1, "load": [0, -1000, 0, 0, 0, 0]}
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["loads"] = [load]
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["prediction"]["loads"] = [load, load]
    response["request_fingerprint"]["ground_truth_hash"] = file_digest(truth_path)
    response_path.write_text(json.dumps(response), encoding="utf-8")
    freeze_manifest(tmp_path)
    report = evaluate_real_world(tmp_path)
    assert report["metrics"]["load_numeric_unit_accuracy"] == 1
    assert report["counts"]["loads"]["fp"] == 1
    assert report["case_reports"][0]["metrics"] == report["metrics"]


def test_real_world_templates_bind_exif_and_transformed_image_dimensions(tmp_path):
    """横置照片和裁剪旋转输入必须用与桌面一致的衍生图尺寸标注。"""
    from PIL import Image
    image = tmp_path / "images" / "rotated.jpg"
    image.parent.mkdir()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 120), "white").save(image, exif=exif)
    create_templates(tmp_path)
    truth_path = tmp_path / "ground_truth" / "rotated.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    assert (truth["width_px"], truth["height_px"]) == (120, 80)
    truth_path.unlink()
    metadata_path = tmp_path / "metadata" / "rotated.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["preprocessing"] = {"crop": [10, 20, 100, 70], "rotation_quarters_cw": 1}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    create_templates(tmp_path)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    assert (truth["width_px"], truth["height_px"]) == (50, 90)


def test_v2_draft_converts_to_metric_prediction_without_model_coordinates():
    draft = {
        "image_model": {
            "nodes": [{"id": 1, "u": 0.1, "v": 0.8},
                      {"id": 2, "u": 0.9, "v": 0.2}],
            "members": [{"id": 1, "i": 1, "j": 2, "section": None}],
            "supports": [{"node": 1, "fix": [1, 1, 1, 0, 0, 0]}],
            "load_cases": [{"name": "DL", "nodal_loads": [
                {"name": "P1", "node": 2, "load": [0, -1, 0, 0, 0, 0],
                 "force_unit": "kN", "moment_unit": "kN*m"}]}],
        },
        "intersections": [], "issues": [], "scale": {"status": "unknown"},
    }
    prediction = draft_to_prediction(draft, image_id="photo", width=800, height=600)
    assert prediction["nodes"][0] == {"id": 1, "point": [0.1, 0.8]}
    assert prediction["members"] == [{"id": 1, "i": 1, "j": 2}]
    assert prediction["loads"][0]["case"] == "DL"
    assert prediction["loads"][0]["collection"] == "nodal_loads"


@pytest.mark.parametrize("image_id", ["moj_labels", "moj_bridge"])
def test_new_numeric_public_case_is_bound_before_single_call(image_id):
    """新数值图首测不能换图、改真值或重复付费挑结果，授权与输入须独立哈希绑定。"""
    root = ROOT / "public_cases" / "numeric_01"
    info = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    truth_path = root / "ground_truth" / f"{image_id}.json"
    image_path = root / "images" / f"{image_id}.png"
    meta = json.loads((root / "metadata" / f"{image_id}.json").read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    response = json.loads((root / "responses" / f"{image_id}.json").read_text(encoding="utf-8"))
    assert info["annotations_fixed_before_calls"][image_id] == file_digest(truth_path)
    assert info["images_fixed_before_calls"][image_id] == file_digest(image_path)
    assert truth["input_context_hash"] == canonical_digest(input_context(meta))
    assert meta["consent_to_evaluate"] and meta["license"] == "CC BY-SA 4.0"
    assert meta["author"] and meta["image_url"] and meta["license_url"]
    assert len(response["raw_responses"]) == response["runtime"]["attempts"] == 1
    assert info["max_repairs"] == 0 and not info["review_actions"]
    assert response["request_fingerprint"]["pipeline_hash"] == info["pipeline_hash"]
    assert response["request_fingerprint"]["prompt_hash"] == info["prompt_hash"]
    snapshot = json.loads((root / "pipeline_snapshot.json").read_text(encoding="utf-8"))
    assert canonical_digest(snapshot) == info["pipeline_hash"]
    assert truth["dimension_observations"] and truth["scale"]["status"] == "unknown"
    assert response["prediction"]["scale"]["status"] == "unknown"
    assert all("load" in item for item in response["prediction"]["loads"])
    assert len(response["prediction"]["loads"]) == len(truth["loads"])


def test_new_numeric_first_results_remain_frozen_failed_and_auditable():
    """节点错位导致真实荷载首测失败时不能改标注/容差，也不能把换算成功报告为完整识别成功。"""
    root = ROOT / "public_cases" / "numeric_01"
    assert audit_dataset(root)["ready_count"] == 2
    frozen = json.loads((root / "report.json").read_text(encoding="utf-8"))
    report = evaluate_real_world(root)
    assert report["metrics"] == frozen["metrics"]
    assert report["counts"] == frozen["counts"]
    assert not report["passed"] and report["metrics"]["load_numeric_unit_accuracy"] == 0
    assert frozen["runtime"]["total_calls"] == 2


@pytest.mark.parametrize(("image_id", "node_count", "member_count", "joint_count"),
                         [("wall_truss", 5, 7, 14), ("tower_truss", 12, 21, 59)])
def test_holdout_reference_is_complete_and_bound_before_first_call(
        image_id, node_count, member_count, joint_count):
    """新独立图样不能重用历史图片、漏共线接头或在首响应后修改参考来提高分数。"""
    from PIL import Image, ImageOps

    root = ROOT / "public_cases" / "holdout_01"
    info = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    truth_path = root / "ground_truth" / f"{image_id}.json"
    image_path = root / "images" / f"{image_id}.png"
    meta_path = root / "metadata" / f"{image_id}.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    response = json.loads((root / "responses" / f"{image_id}.json").read_text(encoding="utf-8"))
    assert check_reference_topology(truth)["valid"]
    assert (len(truth["nodes"]), len(truth["members"]), len(truth["intersections"])) == (
        node_count, member_count, joint_count)
    assert info["annotations_fixed_before_calls"][image_id] == file_digest(truth_path)
    assert info["metadata_fixed_before_calls"][image_id] == file_digest(meta_path)
    assert info["images_fixed_before_calls"][image_id] == file_digest(image_path)
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        decoded = canonical_digest({"size": list(image.size), "rgb": hashlib.sha256(image.tobytes()).hexdigest()})
    assert decoded == info["decoded_pixels_fixed_before_calls"][image_id]
    assert decoded not in {p["pixels"] for p in info["prior_project_images"].values()}
    assert file_digest(image_path) not in {p["bytes"] for p in info["prior_project_images"].values()}
    assert meta["consent_to_evaluate"] and meta["license"] == "CC BY-SA 4.0"
    assert meta["reference_contract"] == truth["topology_reference"]["format"] == REFERENCE_FORMAT
    assert truth["input_context_hash"] == canonical_digest(input_context(meta))
    fingerprint = response["request_fingerprint"]
    assert fingerprint["ground_truth_hash"] == info["annotations_fixed_before_calls"][image_id]
    for key in ("provider", "model", "prompt_hash", "schema_hash", "pipeline_hash", "request_options"):
        assert fingerprint[key] == info[key]
    snapshot = json.loads((root / "pipeline_snapshot.json").read_text(encoding="utf-8"))
    assert len(snapshot) == 8 and canonical_digest(snapshot) == info["pipeline_hash"]
    assert info["max_repairs"] == 0 and not info["review_actions"]
    assert len(response["raw_responses"]) == len(response["call_metadata"]) == response["runtime"]["attempts"] == 1


def test_holdout_failed_inputs_stay_in_score_and_full_reference_denominator():
    """截断及重复杆件的首测失败不能被丢弃，也不能把完整参考删除成无可评分项。"""
    root = ROOT / "public_cases" / "holdout_01"
    frozen = json.loads((root / "report.json").read_text(encoding="utf-8"))
    report = evaluate_real_world(root)
    assert report["metrics"] == frozen["metrics"] and report["counts"] == frozen["counts"]
    assert report["runtime"]["total_calls"] == report["runtime"]["parse_failures"] == 2
    assert report["runtime"]["parse_successes"] == 0 and not report["passed"]
    assert report["reference_contract"]["legacy_cases"] == []
    assert report["reference_contract"]["checked_cases"] == 2
    assert report["counts"]["nodes"]["fn"] == 17
    assert report["counts"]["members"]["fn"] == 28
    assert report["counts"]["intersections"]["fn"] == 73
    assert report["metrics"]["intersection_accuracy"] == report["metrics"]["load_numeric_unit_accuracy"] == 0
    assert report["metrics"]["scale_accuracy"] is None


def test_holdout_reference_vectors_use_explicit_original_units_and_angles():
    """墙面滚动方向、磅力换算及斜荷载竖直夹角不能被预测的错误轴向或默认单位改写。"""
    import math

    root = ROOT / "public_cases" / "holdout_01" / "ground_truth"
    wall = json.loads((root / "wall_truss.json").read_text(encoding="utf-8"))
    tower = json.loads((root / "tower_truss.json").read_text(encoding="utf-8"))
    assert wall["supports"][1]["fix"] == [True, False, True, False, False, False]
    assert wall["loads"][0]["load"] == pytest.approx([0, -500*0.45359237*9.80665, 0, 0, 0, 0])
    for item, force in zip(tower["loads"], (40000, 50000), strict=True):
        assert item["load"] == pytest.approx([
            force*math.sin(math.radians(15)), -force*math.cos(math.radians(15)), 0, 0, 0, 0])
    assert wall["scale"]["status"] == tower["scale"]["status"] == "unknown"


def test_holdout_failure_keeps_original_truncation_and_reversed_duplicate_evidence():
    """冻结原文须保留塔架截断与墙上桁架反向重复杆件，不能用补全或删除覆盖首次失败。"""
    from sketch_parser import SketchParser

    root = ROOT / "public_cases" / "holdout_01" / "responses"
    tower = json.loads((root / "tower_truss.json").read_text(encoding="utf-8"))
    wall = json.loads((root / "wall_truss.json").read_text(encoding="utf-8"))
    assert tower["outcome"] == wall["outcome"] == "FAIL"
    assert tower["call_metadata"][0]["finish_reason"] == "length"
    assert tower["call_metadata"][0]["completion_tokens"] == 4000
    with pytest.raises(ValueError):
        SketchParser._extract_json(tower["raw_responses"][0])
    raw = SketchParser._extract_json(wall["raw_responses"][0])
    members = {m["id"]: m for m in raw["image_model"]["members"]}
    assert (members[5]["i"], members[5]["j"]) == (members[6]["j"], members[6]["i"])
    assert "重复" in wall["errors"][0]
    assert wall["call_metadata"][0]["finish_reason"] == "stop"
    assert tower["prediction"]["nodes"] == wall["prediction"]["nodes"] == []


def test_holdout_existing_failure_never_triggers_a_second_provider_call(monkeypatch):
    """再次执行识别只能提示首响应存在或过期，不能自动付费重试挑选新独立成绩。"""
    root = ROOT / "public_cases" / "holdout_01"
    original = {p: p.read_bytes() for p in (root / "responses").glob("*.json")}
    monkeypatch.setattr("sketch_parser.SketchParser.from_env",
                        lambda *a, **k: pytest.fail("已有首响应不能创建视觉客户端"))
    result = recognize(root, provider="deepseek", model="deepseek-flash", max_repairs=0)
    assert len(result["results"]) == 2
    assert all(item["status"] in {"SKIPPED_EXISTS", "STALE_RESPONSE"} for item in result["results"])
    assert all(p.read_bytes() == data for p, data in original.items())



def test_joint_node_replay_is_separate_from_failed_online_results():
    """开发图像素位置改善只能记作绑定源码的零调用回放，原在线失败及参考契约缺口不能重写。"""
    root = ROOT / "public_cases" / "joint_nodes_01"
    original = ROOT / "public_cases" / "numeric_01"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / report["replay_snapshot"]).read_text(encoding="utf-8"))
    assert len(snapshot) == 15 and canonical_digest(snapshot) == report["replay_implementation_hash"]
    assert report["original_online_report_hash"] == file_digest(original / "report.json")
    assert report["original_ui_review_hash"] == file_digest(original / "ui_review.json")
    online = json.loads((original / "report.json").read_text(encoding="utf-8"))
    assert online["metrics"]["node_f1"] == .3 and not online["passed"]
    assert report["new_api_calls"] == 0 and report["mode"] == "offline-replay"
    assert not report["passed"] and report["metrics"]["load_f1"] < 1
    assert report["metrics"]["scale_accuracy"] is None
    assert len(report["native_ui"]) == 4 and all(x["commit_blocked"] for x in report["native_ui"])
    old = json.loads((ROOT / "public_cases" / "review_edit_01" / "report.json").read_text(encoding="utf-8"))
    assert report["legacy_replay_metrics"] == old["metrics"]


@pytest.mark.parametrize("image_id", ["wall_truss", "tower_truss"])
def test_compact_development_retest_preserves_reference_and_pre_call_bindings(image_id):
    """复测不能改真实节点和杆件来适配预测，新输入须在调用前绑定并明确开发身份。"""
    root = ROOT / "public_cases" / "compact_01"
    old = ROOT / "public_cases" / "holdout_01"
    info = json.loads((root / "round_info.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "metadata" / f"{image_id}.json").read_text(encoding="utf-8"))
    truth = json.loads((root / "ground_truth" / f"{image_id}.json").read_text(encoding="utf-8"))
    original = json.loads((old / "ground_truth" / f"{image_id}.json").read_text(encoding="utf-8"))
    response = json.loads((root / "responses" / f"{image_id}.json").read_text(encoding="utf-8"))
    assert "development_retest" in meta["subset_labels"] and "project_holdout" not in meta["subset_labels"]
    assert meta["work_plane"]["plane"] == "XY"
    assert truth["input_context_hash"] == canonical_digest(input_context(meta))
    for key in ("nodes", "members", "intersections", "supports", "loads", "scale", "issues"):
        assert truth[key] == original[key]
    assert (root / "images" / f"{image_id}.png").read_bytes() == (old / "images" / f"{image_id}.png").read_bytes()
    for kind, folder, ext in (("image_hash", "images", "png"), ("metadata_hash", "metadata", "json"),
                              ("ground_truth_hash", "ground_truth", "json")):
        assert info["bindings_before_calls"][image_id][kind] == file_digest(root / folder / f"{image_id}.{ext}")
    fingerprint = response["request_fingerprint"]
    assert fingerprint["ground_truth_hash"] == info["bindings_before_calls"][image_id]["ground_truth_hash"]
    for key in ("prompt_hash", "schema_hash", "pipeline_hash", "request_options"):
        assert fingerprint[key] == info[key]
    snapshot = json.loads((root / "pipeline_snapshot.json").read_text(encoding="utf-8"))
    assert len(snapshot) == 8 and canonical_digest(snapshot) == info["pipeline_hash"]
    assert info["max_repairs"] == 1 and not info["review_actions"]
    assert len(response["raw_responses"]) == response["runtime"]["attempts"] == 1
    assert response["call_metadata"][0]["finish_reason"] == "stop"
    assert info["original_report_hash"] == file_digest(old / "report.json")


def test_compact_format_success_does_not_erase_geometry_and_load_failures():
    """两图输出变完整不能冒充准确率达标，漏杆、位置和单位错误仍须进入严格评分。"""
    root = ROOT / "public_cases" / "compact_01"
    frozen = json.loads((root / "report.json").read_text(encoding="utf-8"))
    report = evaluate_real_world(root)
    assert report["metrics"] == frozen["metrics"] and report["counts"] == frozen["counts"]
    assert report["runtime"]["parse_successes"] == report["runtime"]["total_calls"] == 2
    assert not report["passed"]
    assert report["counts"]["nodes"] == {"tp": 0, "fp": 16, "fn": 17}
    assert report["counts"]["members"] == {"tp": 0, "fp": 22, "fn": 28}
    assert report["counts"]["intersections"]["fn"] == 73
    assert report["metrics"]["load_numeric_unit_accuracy"] == 0 and report["metrics"]["scale_accuracy"] is None


def test_original_plane_inconsistency_is_preserved_and_explicitly_corrected_in_new_round():
    """首测 XZ 与 XY 参考不一致不能覆盖历史输入，复测须明确记录这项混杂因素。"""
    old = ROOT / "public_cases" / "holdout_01"
    new = ROOT / "public_cases" / "compact_01"
    for image_id in ("wall_truss", "tower_truss"):
        meta = json.loads((old / "metadata" / f"{image_id}.json").read_text(encoding="utf-8"))
        assert meta["work_plane"]["plane"] == "XZ"
    info = json.loads((new / "round_info.json").read_text(encoding="utf-8"))
    assert info["input_correction"]["old_plane"] == "XZ" and info["input_correction"]["new_plane"] == "XY"
    assert any("不能分离" in text for text in info["limitations"])


def test_compact_frozen_retest_never_pays_for_another_response(monkeypatch):
    """已冻结开发图不能再次付费识别挑选结果，原成功草稿及错误分数须保留。"""
    root = ROOT / "public_cases" / "compact_01"
    original = {p: p.read_bytes() for p in (root / "responses").glob("*.json")}
    monkeypatch.setattr("sketch_parser.SketchParser.from_env",
                        lambda *a, **k: pytest.fail("冻结复测不能创建视觉客户端"))
    result = recognize(root, provider="deepseek", model="deepseek-flash", max_repairs=1)
    assert all(item["status"] in {"SKIPPED_EXISTS", "STALE_RESPONSE"} for item in result["results"])
    assert all(p.read_bytes() == raw for p, raw in original.items())
