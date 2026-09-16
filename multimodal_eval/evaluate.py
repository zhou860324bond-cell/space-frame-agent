"""MM2-08 manifest-bound offline evaluator. No provider/network access."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from multimodal_contract import (load_manifest, match_intersections, match_members,
                                 match_points, score_open_issues)

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.json"
THRESHOLDS = {
    "node_f1": 0.85, "member_f1": 0.85,
    "clear_node_f1": 0.95, "clear_member_f1": 0.95,
    "intersection_accuracy": 0.90, "support_f1": 0.80, "load_f1": 0.80,
}


def _f1(counts: dict[str, int]) -> float | None:
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return None if denominator == 0 else 2 * counts["tp"] / denominator


def _add(target: dict[str, int], *, tp: int, fp: int, fn: int) -> None:
    target["tp"] += tp
    target["fp"] += fp
    target["fn"] += fn


def _match_counts(result: Any) -> dict[str, int]:
    return {"tp": len(result.pairs), "fp": len(result.unmatched_predictions),
            "fn": len(result.unmatched_truths)}


def _multiset_counts(predicted: Iterable[Any], expected: Iterable[Any]) -> dict[str, int]:
    pred, truth = Counter(predicted), Counter(expected)
    tp = sum((pred & truth).values())
    return {"tp": tp, "fp": sum(pred.values()) - tp,
            "fn": sum(truth.values()) - tp}


def _support_key(item: dict[str, Any], node_map: dict[str, str] | None = None):
    node = str(item["node"])
    if node_map is not None:
        node = node_map.get(node, "__unmatched__:" + node)
    return node, tuple(int(value) for value in item["fix"])


def _load_key(item: dict[str, Any], node_map: dict[str, str] | None = None,
              member_map: dict[str, str] | None = None):
    collection = str(item["collection"])
    if "node" in item:
        raw = str(item["node"])
        target = ("node", raw if node_map is None else
                  node_map.get(raw, "__unmatched__:" + raw))
    else:
        raw = str(item["member"])
        target = ("member", raw if member_map is None else
                  member_map.get(raw, "__unmatched__:" + raw))
    return (str(item.get("case", "__top__")), collection, str(item.get("kind", "")),
            str(item.get("name", "")), target)


def _unit_factor(unit: str) -> float:
    normalized = unit.replace("·", "*").replace(" ", "").lower()
    factors = {
        "n": 1.0, "kn": 1.0e3,
        "m": 1.0, "mm": 1.0e-3,
        "rad": 1.0, "deg": 0.017453292519943295,
        "n*m": 1.0, "kn*m": 1.0e3, "n*mm": 1.0e-3, "kn*mm": 1.0,
        "n/m": 1.0, "kn/m": 1.0e3, "n/mm": 1.0e3, "kn/mm": 1.0e6,
    }
    if normalized not in factors:
        raise ValueError(f"不支持的评测单位: {unit}")
    return factors[normalized]


def _scaled(values: Iterable[Any], unit: str, tolerance: float) -> list[tuple[float, float]]:
    factor = _unit_factor(unit)
    return [(float(value) * factor, tolerance) for value in values]


def _numeric_payload(item: dict[str, Any]) -> list[tuple[float, float]]:
    collection = item["collection"]
    if collection == "nodal_loads":
        values = item["load"]
        force_unit = str(item.get("force_unit", item.get("unit", "N")))
        moment_unit = str(item.get("moment_unit",
                                   "kN*m" if force_unit.lower() == "kn" else "N*m"))
        return (_scaled(values[:3], force_unit, 1.0e-6)
                + _scaled(values[3:], moment_unit, 1.0e-6))
    if collection == "settlements":
        values = item["d"]
        return (_scaled(values[:3], str(item.get("length_unit", "m")), 1.0e-9)
                + _scaled(values[3:], str(item.get("angle_unit", "rad")), 1.0e-9))
    if collection == "member_loads":
        return _scaled(item["w"], str(item.get("unit", "N/m")), 1.0e-6)
    kind = str(item["kind"])
    result = _scaled([item["a"]], str(item.get("length_unit", "m")), 1.0e-9)
    load_unit = str(item.get("unit", "N" if kind == "point" else "N/m"))
    result += _scaled(item["w1"], load_unit, 1.0e-6)
    if kind == "trapezoid":
        result += _scaled(item["w2"], load_unit, 1.0e-6)
    return result


def _numeric_equal(prediction: dict[str, Any], truth: dict[str, Any]) -> bool:
    try:
        actual, expected = _numeric_payload(prediction), _numeric_payload(truth)
    except (KeyError, TypeError, ValueError):
        return False
    return len(actual) == len(expected) and all(
        abs(got - wanted) <= max(tolerance, 1.0e-6 * abs(wanted))
        for (got, _), (wanted, tolerance) in zip(actual, expected, strict=True))


def evaluate_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = load_manifest(manifest_path)
    totals = {name: {"tp": 0, "fp": 0, "fn": 0}
              for name in ("nodes", "members", "clear_nodes", "clear_members",
                           "supports", "loads", "issues")}
    intersection = {"correct": 0, "wrong": 0, "fp": 0, "fn": 0}
    scale = {"correct": 0, "total": 0}
    load_numeric = {"correct": 0, "total": 0}
    evaluated = []

    for case in manifest["cases"]:
        if not case["gate"]:
            continue
        truth = json.loads((manifest_path.parent / case["ground_truth_path"])
                           .read_text(encoding="utf-8"))
        fixture = json.loads((manifest_path.parent / case["response_fixture_path"])
                             .read_text(encoding="utf-8"))
        prediction = fixture["prediction"]
        width, height = int(truth["width_px"]), int(truth["height_px"])
        nodes = match_points(prediction.get("nodes", ()), truth.get("nodes", ()),
                             width, height)
        members = match_members(prediction.get("members", ()), truth.get("members", ()),
                                nodes.pairs)
        _add(totals["nodes"], **_match_counts(nodes))
        _add(totals["members"], **_match_counts(members))
        if "clear" in case["subset_labels"]:
            _add(totals["clear_nodes"], **_match_counts(nodes))
            _add(totals["clear_members"], **_match_counts(members))
        node_map = {str(pred): str(expected) for pred, expected, _ in nodes.pairs}
        member_map = {str(pred): str(expected) for pred, expected, _ in members.pairs}

        intersections = match_intersections(
            prediction.get("intersections", ()), truth.get("intersections", ()),
            members.pairs, width, height)
        pred_i = {str(item["id"]): item for item in prediction.get("intersections", ())}
        truth_i = {str(item["id"]): item for item in truth.get("intersections", ())}
        for pred_id, truth_id, _ in intersections.pairs:
            if pred_i[pred_id].get("decision") == truth_i[truth_id].get("decision"):
                intersection["correct"] += 1
            else:
                intersection["wrong"] += 1
        intersection["fp"] += len(intersections.unmatched_predictions)
        intersection["fn"] += len(intersections.unmatched_truths)

        support_counts = _multiset_counts(
            (_support_key(item, node_map) for item in prediction.get("supports", ())),
            (_support_key(item) for item in truth.get("supports", ())))
        _add(totals["supports"], **support_counts)
        pred_loads = list(prediction.get("loads", ()))
        truth_loads = list(truth.get("loads", ()))
        load_counts = _multiset_counts(
            (_load_key(item, node_map, member_map) for item in pred_loads),
            (_load_key(item) for item in truth_loads))
        _add(totals["loads"], **load_counts)
        truth_by_key = {_load_key(item): item for item in truth_loads}
        load_numeric["total"] += len(truth_loads)
        for item in pred_loads:
            mapped_key = _load_key(item, node_map, member_map)
            if mapped_key in truth_by_key:
                load_numeric["correct"] += _numeric_equal(
                    item, truth_by_key[mapped_key])

        issue_counts = score_open_issues(prediction.get("issues", ()),
                                         truth.get("issues", ()))
        _add(totals["issues"], tp=issue_counts["tp"], fp=issue_counts["fp"],
             fn=issue_counts["fn"])
        expected_scale = truth.get("scale") or {}
        if expected_scale.get("status") == "confirmed":
            scale["total"] += 1
            predicted_scale = prediction.get("scale") or {}
            expected_value = float(expected_scale["length_per_pixel"])
            actual = predicted_scale.get("length_per_pixel")
            if predicted_scale.get("status") == "confirmed" and isinstance(actual, (int, float)):
                scale["correct"] += abs(float(actual) - expected_value) / expected_value <= 0.02
        evaluated.append(case["image_id"])

    metrics = {
        "node_f1": _f1(totals["nodes"]),
        "member_f1": _f1(totals["members"]),
        "clear_node_f1": _f1(totals["clear_nodes"]),
        "clear_member_f1": _f1(totals["clear_members"]),
        "intersection_accuracy": None,
        "support_f1": _f1(totals["supports"]),
        "load_f1": _f1(totals["loads"]),
        "load_numeric_unit_accuracy": (None if not load_numeric["total"] else
                                       load_numeric["correct"] / load_numeric["total"]),
        "scale_accuracy": None if not scale["total"] else scale["correct"] / scale["total"],
        "issue_recall": (None if totals["issues"]["tp"] + totals["issues"]["fn"] == 0
                         else totals["issues"]["tp"] /
                         (totals["issues"]["tp"] + totals["issues"]["fn"])),
        "issue_false_positives": totals["issues"]["fp"],
    }
    intersection_denominator = sum(intersection.values())
    if intersection_denominator:
        metrics["intersection_accuracy"] = intersection["correct"] / intersection_denominator
    failures = [name for name, threshold in THRESHOLDS.items()
                if metrics[name] is None or metrics[name] < threshold]
    return {
        "format": "space-frame-multimodal-eval-report/v1",
        "manifest": manifest_path.name,
        "gate_cases": len(evaluated), "evaluated_image_ids": evaluated,
        "metrics": metrics, "counts": {**totals, "intersections": intersection,
                                         "scale": scale, "load_numeric": load_numeric},
        "thresholds": THRESHOLDS, "failures": failures, "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output")
    args = parser.parse_args()
    report = evaluate_manifest(args.manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
