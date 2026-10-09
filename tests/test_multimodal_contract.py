"""MM2-00 deterministic multimodal contract tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from multimodal_contract import (canonical_digest, load_manifest, match_intersections,
                                 match_members, match_points, point_match_threshold,
                                 score_open_issues)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "multimodal_eval" / "manifest.json"


def test_canonical_digest_ignores_dictionary_order():
    assert canonical_digest({"b": 2, "a": [1]}) == canonical_digest(
        {"a": [1], "b": 2})


def test_point_matching_is_deterministic_and_one_to_one():
    predictions = [{"id": "b", "point": [0.5, 0.5]},
                   {"id": "a", "point": [0.5, 0.5]}]
    truths = [{"id": "2", "point": [0.5, 0.5]},
              {"id": "1", "point": [0.5, 0.5]}]
    first = match_points(predictions, truths, 101, 101)
    second = match_points(reversed(predictions), reversed(truths), 101, 101)
    assert first == second
    assert first.pairs == (("a", "1", 0.0), ("b", "2", 0.0))
    assert point_match_threshold(101, 101) == 5.0


def test_member_and_intersection_matching_use_mapped_topology():
    nodes = (("p1", "t1", 0.0), ("p2", "t2", 0.0),
             ("p3", "t3", 0.0), ("p4", "t4", 0.0))
    members = match_members(
        [{"id": "pm2", "i": "p3", "j": "p4"},
         {"id": "pm1", "i": "p1", "j": "p2"}],
        [{"id": "tm1", "i": "t2", "j": "t1"},
         {"id": "tm2", "i": "t3", "j": "t4"}], nodes)
    assert [(a, b) for a, b, _ in members.pairs] == [("pm1", "tm1"),
                                                     ("pm2", "tm2")]
    intersections = match_intersections(
        [{"id": "pi", "members": ["pm2", "pm1"], "point": [0.5, 0.5]}],
        [{"id": "ti", "members": ["tm1", "tm2"], "point": [0.5, 0.5]}],
        members.pairs, 128, 96)
    assert intersections.pairs == (("pi", "ti", 0.0),)


def test_issue_scoring_counts_severity_mismatch_as_fp_and_fn():
    prediction = [{"category": "scale_unknown", "entity_refs": [],
                   "severity": "warning", "status": "open"}]
    truth = [{"category": "scale_unknown", "entity_refs": [],
              "severity": "blocking", "status": "open"}]
    assert score_open_issues(prediction, truth) == {
        "tp": 0, "fp": 1, "fn": 1, "recall": 0.0,
    }


def test_repeated_missing_load_issues_are_counted_once_and_extras_are_false_positives():
    """真实混合荷载首响应在同一杆件上报告多条缺值，不能使整组评分中断或重复得分。"""
    issue = {"category": "load_incomplete", "entity_refs": ["member:1"],
             "severity": "blocking", "status": "open"}
    assert score_open_issues([issue, issue, issue], [issue]) == {
        "tp": 1, "fp": 2, "fn": 0, "recall": 1.0}


def test_repeated_issue_severity_matching_does_not_depend_on_response_order():
    """同对象有 warning 和 blocking 两条预测时，只配对一次，剩余预测仍计错误。"""
    blocking = {"category": "load_incomplete", "entity_refs": ["node:2"],
                "severity": "blocking", "status": "open"}
    warning = {**blocking, "severity": "warning"}
    expected = {"tp": 1, "fp": 1, "fn": 0, "recall": 1.0}
    assert score_open_issues([warning, blocking], [blocking]) == expected
    assert score_open_issues([blocking, warning], [blocking]) == expected


def test_repeated_ground_truth_issues_still_require_annotation_correction():
    """允许模型重复问题计分不能掩盖真值重复标注。"""
    issue = {"category": "load_incomplete", "entity_refs": [], "severity": "blocking", "status": "open"}
    with pytest.raises(ValueError, match="重复 open issue"):
        score_open_issues([], [issue, issue])


def test_seed_manifest_binds_thirty_authorized_cases():
    manifest = load_manifest(MANIFEST)
    assert len(manifest["cases"]) >= 30
    assert all(case["gate"] for case in manifest["cases"])
    assert all(case["source"] == "project-generated-original"
               for case in manifest["cases"])


def test_manifest_rejects_hash_drift(tmp_path):
    for directory in ("images", "ground_truth", "responses"):
        shutil.copytree(MANIFEST.parent / directory, tmp_path / directory)
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["cases"][0]["image_hash"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="缺失或哈希漂移"):
        load_manifest(path)
