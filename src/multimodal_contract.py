"""Deterministic constants, matching, and manifest checks for multimodal evals."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

POINT_MATCH_ABS_PX = 5.0
POINT_MATCH_DIAGONAL_RATIO = 0.005
ENDPOINT_MATCH_ABS_PX = 6.0
ENDPOINT_MATCH_DIAGONAL_RATIO = 0.005
PARALLEL_ANGLE_DEG = 2.0
SCALE_CONFLICT_RATIO = 0.02
MODEL_COINCIDENCE_M = 1.0e-6
MANIFEST_FORMAT = "space-frame-multimodal-eval/v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_match_threshold(width_px: int, height_px: int) -> float:
    if width_px <= 1 or height_px <= 1:
        raise ValueError("派生图宽高必须大于 1")
    return max(POINT_MATCH_ABS_PX,
               math.hypot(width_px - 1, height_px - 1)
               * POINT_MATCH_DIAGONAL_RATIO)


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[tuple[str, str, float], ...]
    unmatched_predictions: tuple[str, ...]
    unmatched_truths: tuple[str, ...]


def _stable_items(items: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in items:
        stable_id = str(item["id"])
        if stable_id in result:
            raise ValueError(f"重复稳定 ID: {stable_id}")
        result[stable_id] = item
    return result


def _pixel_distance(a: Sequence[float], b: Sequence[float],
                    width_px: int, height_px: int) -> float:
    if len(a) != 2 or len(b) != 2:
        raise ValueError("point 必须包含两个坐标")
    values = tuple(float(value) for value in (*a, *b))
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0
               for value in values):
        raise ValueError("归一化坐标必须是 [0,1] 内有限数")
    return math.hypot((values[0] - values[2]) * (width_px - 1),
                      (values[1] - values[3]) * (height_px - 1))


def _greedy_candidates(prediction_ids: Iterable[str], truth_ids: Iterable[str],
                       candidates: Iterable[tuple[float, str, str]]) -> MatchResult:
    prediction_ids = tuple(sorted(prediction_ids))
    truth_ids = tuple(sorted(truth_ids))
    used_predictions: set[str] = set()
    used_truths: set[str] = set()
    pairs: list[tuple[str, str, float]] = []
    for distance, prediction_id, truth_id in sorted(candidates):
        if prediction_id in used_predictions or truth_id in used_truths:
            continue
        used_predictions.add(prediction_id)
        used_truths.add(truth_id)
        pairs.append((prediction_id, truth_id, distance))
    return MatchResult(
        tuple(pairs),
        tuple(item for item in prediction_ids if item not in used_predictions),
        tuple(item for item in truth_ids if item not in used_truths),
    )


def match_points(predictions: Iterable[Mapping[str, Any]],
                 truths: Iterable[Mapping[str, Any]],
                 width_px: int, height_px: int) -> MatchResult:
    predicted = _stable_items(predictions)
    expected = _stable_items(truths)
    threshold = point_match_threshold(width_px, height_px)
    candidates = []
    for prediction_id, prediction in predicted.items():
        for truth_id, truth in expected.items():
            distance = _pixel_distance(prediction["point"], truth["point"],
                                       width_px, height_px)
            if distance <= threshold:
                candidates.append((distance, prediction_id, truth_id))
    return _greedy_candidates(predicted, expected, candidates)


def match_members(predictions: Iterable[Mapping[str, Any]],
                  truths: Iterable[Mapping[str, Any]],
                  node_pairs: Iterable[tuple[str, str, float]]) -> MatchResult:
    predicted = _stable_items(predictions)
    expected = _stable_items(truths)
    node_map = {str(prediction): str(truth)
                for prediction, truth, _ in node_pairs}
    truth_groups: dict[tuple[str, str], list[str]] = {}
    for truth_id, member in expected.items():
        key = tuple(sorted((str(member["i"]), str(member["j"]))))
        truth_groups.setdefault(key, []).append(truth_id)
    prediction_groups: dict[tuple[str, str], list[str]] = {}
    for prediction_id, member in predicted.items():
        try:
            key = tuple(sorted((node_map[str(member["i"])],
                                node_map[str(member["j"])])))
        except KeyError:
            continue
        prediction_groups.setdefault(key, []).append(prediction_id)
    candidates = []
    for key in sorted(set(prediction_groups) & set(truth_groups)):
        # 这里两边长度本来就可能不等：同一对节点上，预测画了两根平行杆而
        # 标准答案只有一根是常事。多出来的那几根配不上对，截断正是想要的。
        for prediction_id, truth_id in zip(sorted(prediction_groups[key]),
                                           sorted(truth_groups[key]), strict=False):
            candidates.append((0.0, prediction_id, truth_id))
    return _greedy_candidates(predicted, expected, candidates)


def match_intersections(predictions: Iterable[Mapping[str, Any]],
                        truths: Iterable[Mapping[str, Any]],
                        member_pairs: Iterable[tuple[str, str, float]],
                        width_px: int, height_px: int) -> MatchResult:
    predicted = _stable_items(predictions)
    expected = _stable_items(truths)
    member_map = {str(prediction): str(truth)
                  for prediction, truth, _ in member_pairs}
    threshold = point_match_threshold(width_px, height_px)
    candidates = []
    for prediction_id, prediction in predicted.items():
        try:
            mapped = tuple(sorted(member_map[str(item)]
                                  for item in prediction["members"]))
        except KeyError:
            continue
        for truth_id, truth in expected.items():
            if mapped != tuple(sorted(str(item) for item in truth["members"])):
                continue
            distance = _pixel_distance(prediction["point"], truth["point"],
                                       width_px, height_px)
            if distance <= threshold:
                candidates.append((distance, prediction_id, truth_id))
    return _greedy_candidates(predicted, expected, candidates)


def score_open_issues(predictions: Iterable[Mapping[str, Any]],
                      truths: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    def indexed(items: Iterable[Mapping[str, Any]]) -> dict[tuple[str, tuple[str, ...]], str]:
        result = {}
        for item in items:
            if item.get("status") != "open":
                continue
            key = (str(item["category"]),
                   tuple(sorted(set(str(ref) for ref in item.get("entity_refs", ())))))
            if key in result:
                raise ValueError(f"重复 open issue 键: {key}")
            result[key] = str(item["severity"])
        return result

    predicted = indexed(predictions)
    expected = indexed(truths)
    tp = sum(predicted[key] == expected[key] for key in predicted.keys() & expected.keys())
    severity_mismatches = sum(predicted[key] != expected[key]
                              for key in predicted.keys() & expected.keys())
    fp = len(predicted.keys() - expected.keys()) + severity_mismatches
    fn = len(expected.keys() - predicted.keys()) + severity_mismatches
    recall = None if tp + fn == 0 else tp / (tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall}


_MANIFEST_FIELDS = {
    "image_id", "image_path", "image_hash", "ground_truth_path",
    "ground_truth_hash", "response_fixture_path", "response_hash", "provider",
    "model", "prompt_hash", "schema_hash", "subset_labels", "gate", "source",
    "license",
}


def load_manifest(path: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("format") != MANIFEST_FORMAT or not isinstance(data.get("cases"), list):
        raise ValueError("无效的多模态评测 manifest")
    seen: set[str] = set()
    for case in data["cases"]:
        missing = _MANIFEST_FIELDS - set(case)
        if missing:
            raise ValueError(f"manifest case 缺少字段: {sorted(missing)}")
        image_id = str(case["image_id"])
        if image_id in seen:
            raise ValueError(f"重复 image_id: {image_id}")
        seen.add(image_id)
        if not isinstance(case["gate"], bool) or not isinstance(case["subset_labels"], list):
            raise ValueError(f"manifest case {image_id} 的 gate/labels 无效")
        if verify_files:
            for path_key, hash_key in (("image_path", "image_hash"),
                                       ("ground_truth_path", "ground_truth_hash"),
                                       ("response_fixture_path", "response_hash")):
                target = manifest_path.parent / str(case[path_key])
                if not target.is_file() or file_digest(target) != case[hash_key]:
                    raise ValueError(f"manifest case {image_id} 的 {path_key} 缺失或哈希漂移")
    return data
