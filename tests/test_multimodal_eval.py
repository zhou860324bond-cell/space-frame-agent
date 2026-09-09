"""MM2-08 release corpus and deterministic score tests."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

from multimodal_contract import load_manifest
from multimodal_eval.evaluate import _numeric_equal, evaluate_manifest
from multimodal_eval.provider_smoke import PROVIDERS, run
from multimodal_eval.real_world_eval import (FORMAT, PROMPT_HASH, SCHEMA_HASH,
                                              audit_dataset, create_templates,
                                              draft_to_prediction,
                                              freeze_manifest)

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
    status = audit_dataset(tmp_path)
    assert status["ready_count"] == 0
    assert {"consent_not_confirmed", "ground_truth_not_verified",
            "missing_response"} <= set(status["cases"][0]["reasons"])


def test_real_world_freeze_hash_binds_a_complete_case(tmp_path):
    image = _real_image(tmp_path)
    create_templates(tmp_path)
    metadata_path = tmp_path / "metadata" / "site_photo.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(consent_to_evaluate=True, subset_labels=["clear"])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    truth_path = tmp_path / "ground_truth" / "site_photo.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    fixture_truth = json.loads((ROOT / "ground_truth" / "seed_01_portal.json")
                               .read_text(encoding="utf-8"))
    fixture_truth.update(image_id="site_photo", annotation_status="verified")
    truth_path.write_text(json.dumps(fixture_truth), encoding="utf-8")
    response_path = tmp_path / "responses" / "site_photo.json"
    fixture = json.loads((ROOT / "responses" / "seed_01_portal.json")
                         .read_text(encoding="utf-8"))
    fixture["prediction"]["image_id"] = "site_photo"
    fixture["request_fingerprint"] = {
        "provider": "openai", "model": "test-model",
        "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH,
    }
    response_path.parent.mkdir()
    response_path.write_text(json.dumps(fixture), encoding="utf-8")

    manifest = freeze_manifest(tmp_path)
    case = manifest["cases"][0]
    assert case["image_hash"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert case["source"] == "user-provided"
    assert evaluate_manifest(tmp_path / "manifest.json")["gate_cases"] == 1
    response_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希漂移"):
        evaluate_manifest(tmp_path / "manifest.json")


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
