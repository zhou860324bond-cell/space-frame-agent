"""Prepare, freeze, recognize, and score private real-world image cases.

The synthetic release corpus remains immutable.  This module keeps real images
in a separate, explicitly consented manifest and never reports a score for a
case until image, verified ground truth, and provider response are hash-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from multimodal_contract import canonical_digest, file_digest, load_manifest
from image_preprocess import preprocess_image, work_plane_payload
from multimodal_eval.evaluate import evaluate_manifest
from multimodal_workflow import MultimodalControllerState, V2_DRAFT_FORMAT, validate_v2_draft
from sketch_parser import (IMAGE_MIME_TYPES, V2_SKETCH_SYSTEM_PROMPT,
                           SketchParser, _fill_bookkeeping)
from sketch_action_review import ACTION_REVIEW_PROMPT

ROOT = Path(__file__).resolve().parent / "real_world_cases"
FORMAT = "space-frame-real-world-case/v1"
PROMPT_HASH = hashlib.sha256(V2_SKETCH_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
ACTION_PROMPT_HASH = hashlib.sha256(ACTION_REVIEW_PROMPT.encode("utf-8")).hexdigest()
SCHEMA_HASH = hashlib.sha256(V2_DRAFT_FORMAT.encode("utf-8")).hexdigest()
PIPELINE_HASH = canonical_digest({name: (Path(__file__).resolve().parents[1] / "src" / name)
                                 .read_text(encoding="utf-8") for name in (
    "sketch_parser.py", "multimodal_workflow.py", "sketch_axis_refinement.py",
    "image_preprocess.py", "dimension_constraints.py", "sketch_topology.py", "sketch_action_review.py",
    "sketch_moment_refinement.py")})
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _images(root: Path) -> dict[str, Path]:
    directory = root / "images"
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_MIME_TYPES:
            continue
        if path.stem in found:
            raise ValueError(f"真实样本存在重复文件名主干: {path.stem}")
        found[path.stem] = path
    return found


def input_context(metadata: dict) -> dict:
    """只绑定会改变识别输入的参数，授权说明和人工真值不发送给模型。"""
    if metadata.get("preprocessing") is not None and not isinstance(metadata["preprocessing"], dict):
        raise ValueError("preprocessing 必须为对象")
    if metadata.get("work_plane") is not None and not isinstance(metadata["work_plane"], dict):
        raise ValueError("work_plane 必须为对象")
    return {"work_plane": metadata.get("work_plane"),
            "preprocessing": metadata.get("preprocessing") or {}}


def _input_size(image_path: Path, metadata: dict) -> tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = ImageOps.exif_transpose(image).size
    options = input_context(metadata)["preprocessing"]
    crop = options.get("crop")
    if crop is not None:
        if (not isinstance(crop, list) or len(crop) != 4
                or any(not isinstance(v, int) or isinstance(v, bool) for v in crop)
                or not 0 <= crop[0] < crop[2] <= width
                or not 0 <= crop[1] < crop[3] <= height):
            raise ValueError("真实样本裁剪范围无效")
        width, height = crop[2] - crop[0], crop[3] - crop[1]
    rotation = options.get("rotation_quarters_cw", 0)
    if not isinstance(rotation, int) or isinstance(rotation, bool):
        raise ValueError("真实样本旋转参数必须为整数")
    return (height, width) if rotation % 2 else (width, height)


def create_templates(root: str | Path = ROOT) -> dict[str, Any]:
    """Create only missing annotation/metadata templates for discovered images."""
    root = Path(root)
    created: list[str] = []
    for image_id, image_path in _images(root).items():
        metadata_path = root / "metadata" / f"{image_id}.json"
        if not metadata_path.exists():
            _write(metadata_path, {
                "format": FORMAT, "image_id": image_id,
                "source": "user-provided", "license": "private-evaluation-only",
                "consent_to_evaluate": False, "subset_labels": [],
                "work_plane": work_plane_payload(), "preprocessing": {},
                "notes": "确认无敏感信息后，将 consent_to_evaluate 改为 true。",
            })
            created.append(metadata_path.relative_to(root).as_posix())
        truth_path = root / "ground_truth" / f"{image_id}.json"
        if not truth_path.exists():
            metadata = _read(metadata_path)
            width, height = _input_size(image_path, metadata)
            _write(truth_path, {
                "image_id": image_id, "width_px": width, "height_px": height,
                "image_hash": file_digest(image_path),
                "input_context_hash": canonical_digest(input_context(metadata)),
                "annotation_status": "pending", "nodes": [], "members": [],
                "intersections": [], "supports": [], "loads": [], "issues": [],
                "scale": {"status": "unknown"},
            })
            created.append(truth_path.relative_to(root).as_posix())
    return {"images": len(_images(root)), "created": created}


def audit_dataset(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    cases, ready = [], 0
    for image_id, image_path in _images(root).items():
        reasons: list[str] = []
        metadata_path = root / "metadata" / f"{image_id}.json"
        truth_path = root / "ground_truth" / f"{image_id}.json"
        response_path = root / "responses" / f"{image_id}.json"
        metadata = _read(metadata_path) if metadata_path.is_file() else None
        truth = _read(truth_path) if truth_path.is_file() else None
        response = _read(response_path) if response_path.is_file() else None
        image_hash = file_digest(image_path)
        context_hash = canonical_digest(input_context(metadata or {}))
        if metadata is None:
            reasons.append("missing_metadata")
        else:
            if metadata.get("format") != FORMAT or metadata.get("image_id") != image_id:
                reasons.append("invalid_metadata")
            if metadata.get("consent_to_evaluate") is not True:
                reasons.append("consent_not_confirmed")
            if not isinstance(metadata.get("subset_labels"), list):
                reasons.append("invalid_subset_labels")
            if not str(metadata.get("source", "")).strip() \
                    or not str(metadata.get("license", "")).strip():
                reasons.append("missing_source_or_license")
        if truth is None:
            reasons.append("missing_ground_truth")
        else:
            if truth.get("annotation_status") != "verified":
                reasons.append("ground_truth_not_verified")
            try:
                dimensions_match = [truth.get("width_px"), truth.get("height_px")] \
                    == list(_input_size(image_path, metadata or {}))
            except (OSError, ValueError, AttributeError):
                dimensions_match = False
            if truth.get("image_id") != image_id or not dimensions_match:
                reasons.append("ground_truth_image_mismatch")
            if truth.get("image_hash") != image_hash:
                reasons.append("ground_truth_image_hash_mismatch")
            if truth.get("input_context_hash") != context_hash:
                reasons.append("ground_truth_input_context_mismatch")
        if response is None:
            reasons.append("missing_response")
        else:
            prediction = response.get("prediction")
            fingerprint = response.get("request_fingerprint")
            if not isinstance(prediction, dict) or prediction.get("image_id") != image_id:
                reasons.append("invalid_response")
            if not isinstance(fingerprint, dict) or any(
                    not str(fingerprint.get(key, "")).strip()
                    for key in ("provider", "model", "prompt_hash", "schema_hash")):
                reasons.append("invalid_response_fingerprint")
            elif fingerprint.get("image_hash") != image_hash \
                    or fingerprint.get("input_context_hash") != context_hash:
                reasons.append("response_input_mismatch")
            if response.get("outcome") not in ("PASS", "FAIL"):
                reasons.append("missing_response_outcome")
            runtime = response.get("runtime") or {}
            duration = runtime.get("duration_ms")
            attempts = runtime.get("attempts")
            if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
                    or not math.isfinite(duration) or duration < 0
                    or not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1):
                reasons.append("invalid_runtime")
        if not reasons:
            ready += 1
        cases.append({"image_id": image_id, "image_hash": file_digest(image_path),
                      "ready": not reasons, "reasons": reasons})
    return {"format": "space-frame-real-world-audit/v1",
            "root": str(root), "image_count": len(cases),
            "ready_count": ready, "cases": cases}


def draft_to_prediction(draft: dict[str, Any], *, image_id: str,
                        width: int, height: int) -> dict[str, Any]:
    image_model = draft["image_model"]
    prediction = {
        "image_id": image_id, "width_px": width, "height_px": height,
        "nodes": [{"id": item["id"], "point": [item["u"], item["v"]]}
                  for item in image_model.get("nodes", [])],
        "members": [{key: item[key] for key in ("id", "i", "j")}
                    for item in image_model.get("members", [])],
        "supports": deepcopy(image_model.get("supports", [])),
        "intersections": deepcopy(draft.get("intersections", [])),
        "issues": deepcopy(draft.get("issues", [])),
        "scale": deepcopy(draft.get("scale", {"status": "unknown"})),
        "loads": [],
    }
    blocks = [("__top__", image_model)] + [
        (str(case.get("name", "default")), case)
        for case in image_model.get("load_cases", [])]
    for case_name, block in blocks:
        for collection in ("nodal_loads", "member_loads", "member_spans",
                           "settlements"):
            for load in block.get(collection, []):
                item = deepcopy(load)
                item.update(case=case_name, collection=collection)
                prediction["loads"].append(item)
    return prediction


def recognize(root: str | Path = ROOT, *, provider: str, model: str = "",
              max_repairs: int = 2, review_actions: bool = False) -> dict[str, Any]:
    """冻结首轮成功或失败；授权、输入与真值校验必须在网络调用之前。"""
    if not isinstance(max_repairs, int) or isinstance(max_repairs, bool) or not 0 <= max_repairs <= 2:
        raise ValueError("max_repairs 必须在 0 到 2 之间")
    root = Path(root)
    parser = None
    results = []
    audit = {case["image_id"]: case for case in audit_dataset(root)["cases"]}
    for image_id, image_path in _images(root).items():
        metadata = _read(root / "metadata" / f"{image_id}.json") \
            if (root / "metadata" / f"{image_id}.json").exists() else {}
        reasons = [reason for reason in audit[image_id]["reasons"]
                   if reason.startswith(("consent", "ground_truth", "missing_ground_truth",
                                         "missing_metadata", "invalid_metadata", "missing_source"))]
        plane = metadata.get("work_plane") or {}
        if plane.get("status") != "confirmed":
            reasons.append("work_plane_not_confirmed")
        if reasons:
            results.append({"image_id": image_id, "status": "SKIPPED_NOT_READY", "reasons": reasons})
            continue
        output_path = root / "responses" / f"{image_id}.json"
        if output_path.exists():
            invalid = [reason for reason in audit[image_id]["reasons"] if reason != "missing_response"]
            fingerprint = _read(output_path).get("request_fingerprint") or {}
            if any(fingerprint.get(key) != value for key, value in {
                    "provider": provider, "model": model or SketchParser._default_model(provider),
                    "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH}.items()):
                invalid.append("request_fingerprint_changed")
            options = SketchParser(provider=provider, model=model).request_options()
            if fingerprint.get("request_options") != options:
                invalid.append("request_options_changed")
            if fingerprint.get("pipeline_hash") != PIPELINE_HASH:
                invalid.append("pipeline_changed")
            if (bool(fingerprint.get("review_actions", False)) != review_actions
                    or (review_actions and fingerprint.get("action_prompt_hash") != ACTION_PROMPT_HASH)):
                invalid.append("action_review_options_changed")
            results.append({"image_id": image_id,
                            "status": "STALE_RESPONSE" if invalid else "SKIPPED_EXISTS",
                            "reasons": invalid})
            continue
        if parser is None:
            try:
                parser = SketchParser.from_env(provider, model=model)
            except ValueError:
                parser = None
            if parser is None or not parser._api_key:
                results.append({"image_id": image_id, "status": "SKIPPED_NO_CREDENTIALS"})
                parser = None
                continue
        options = input_context(metadata)["preprocessing"]
        prepared = preprocess_image(
            image_path, root / "derived", crop=tuple(options["crop"]) if options.get("crop") else None,
            rotation_quarters_cw=options.get("rotation_quarters_cw", 0))
        state = MultimodalControllerState()
        image_hash = prepared.image_hash
        state.load_image(image_hash)
        job = state.start_recognition(f"real-{provider}-{image_id}")
        review_options = {"review_actions": True} if review_actions else {}
        parsed = parser.parse_v2_with_retry(
            prepared.derived_path, state, job, max_repairs=max_repairs, work_plane=plane, **review_options)
        outcome = "PASS" if parsed.success else "FAIL"
        prediction = {"image_id": image_id, "width_px": prepared.width_px,
                      "height_px": prepared.height_px, "nodes": [], "members": [],
                      "intersections": [], "supports": [], "loads": [], "issues": [],
                      "scale": {"status": "unknown"}}
        if parsed.success:
            from dimension_constraints import apply_scale_to_draft
            from sketch_topology import detect_topology
            draft = deepcopy(parsed.draft)
            draft["source"] = prepared.source_metadata()
            draft["work_plane"] = deepcopy(plane)
            try:
                draft = detect_topology(apply_scale_to_draft(draft))
                prediction = draft_to_prediction(draft, image_id=image_id,
                                                 width=prepared.width_px, height=prepared.height_px)
            except (KeyError, TypeError, ValueError) as exc:
                outcome = "FAIL"
                parsed.errors.append(f"识别后尺度或拓扑处理失败：{exc}")
        _write(output_path, {
            "fixture": "real-provider-response-v2", "outcome": outcome,
            "prediction": prediction,
            "errors": [error.replace(parser._api_key, "[REDACTED]") for error in parsed.errors],
            "raw_responses": [raw.replace(parser._api_key, "[REDACTED]") for raw in parsed.raw_responses],
            "call_metadata": deepcopy(parsed.call_metadata),
            "action_review": deepcopy((parsed.draft or {}).get("action_review")),
            "position_refinements": [deepcopy(entity["position_refinement"])
                                     for entity in (parsed.draft or {}).get("entities", [])
                                     if "position_refinement" in entity],
            "redaction": {"credentials": "not_recorded",
                          "user_metadata": "not_recorded"},
            "request_fingerprint": {
                "provider": provider, "model": parser._model,
                "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH,
                "image_hash": image_hash, "derived_image_hash": prepared.derived_image_hash,
                "input_context_hash": canonical_digest(input_context(metadata)),
                "request_options": parser.request_options(),
                "pipeline_hash": PIPELINE_HASH,
                "review_actions": review_actions,
                "action_prompt_hash": ACTION_PROMPT_HASH if review_actions else None,
            },
            "preprocessing": prepared.source_metadata()["preprocessing"],
            "runtime": {"attempts": parsed.attempts,
                        "duration_ms": parsed.duration_ms},
        })
        results.append({"image_id": image_id, "status": outcome,
                        "attempts": parsed.attempts, "duration_ms": parsed.duration_ms})
    return {"provider": provider, "model": parser._model if parser else model, "status": "COMPLETE",
            "results": results}


def evaluate_real_world(root: str | Path = ROOT) -> dict:
    """评分必须包括失败输入，耗时只统计已实际调用的冻结响应。"""
    root = Path(root)
    path = root / "manifest.json"
    manifest = load_manifest(path)
    report = evaluate_manifest(path)
    attempts, durations, outcomes = [], [], []
    for case in manifest["cases"]:
        if not case["gate"]:
            continue
        if file_digest(root / case["metadata_path"]) != case["metadata_hash"]:
            raise ValueError(f"样本 {case['image_id']} 的授权或输入参数哈希漂移")
        response = _read(root / case["response_fixture_path"])
        outcomes.append({"image_id": case["image_id"], "outcome": response["outcome"],
                         **response["runtime"], "errors": response.get("errors", []),
                         "call_metadata": response.get("call_metadata", []),
                         "action_review": response.get("action_review"),
                         "review_actions": response["request_fingerprint"].get("review_actions", False)})
        attempts.append(response["runtime"]["attempts"])
        durations.append(response["runtime"]["duration_ms"])
    report.update(sample_kind="external-real-world-pilot", case_outcomes=outcomes,
                  runtime={"attempted_cases": len(outcomes),
                           "parse_successes": sum(item["outcome"] == "PASS" for item in outcomes),
                           "parse_failures": sum(item["outcome"] == "FAIL" for item in outcomes),
                           "total_calls": sum(attempts), "median_duration_ms": statistics.median(durations)
                           if durations else None, "total_duration_ms": sum(durations)},
                  action_review={"attempted_cases": sum(item["review_actions"] for item in outcomes),
                                 "completed_cases": sum((item["action_review"] or {}).get("status") == "completed"
                                                        for item in outcomes),
                                 "failed_or_missing_cases": sum(item["review_actions"] and
                                     (item["action_review"] or {}).get("status") != "completed" for item in outcomes)},
                  limitations=["小样本单次结果，不代表真实手绘、手机照片或所有提供商的准确率。",
                               "无参考对象的指标为 null，不补成 100%；失败输入不从分母移除。"])
    return report


def replay_real_world(root: str | Path = ROOT) -> dict:
    """只回放冻结的首次原文；不调用接口，也不改写首次成绩或耗时。"""
    from dimension_constraints import apply_scale_to_draft
    from sketch_topology import detect_topology
    from sketch_axis_refinement import refine_horizontal_axis, refine_joint_nodes

    root = Path(root)
    evaluate_real_world(root)  # 先验证所有原始证据，禁止回放已被修改的样本。
    manifest = load_manifest(root / "manifest.json")
    predictions, outcomes = {}, []
    for case in manifest["cases"]:
        if not case["gate"]:
            continue
        image_id = case["image_id"]
        response = _read(root / case["response_fixture_path"])
        metadata = _read(root / case["metadata_path"])
        review_requested = bool(response["request_fingerprint"].get("review_actions"))
        review_status = None
        try:
            raw = response.get("raw_responses") or []
            if not raw or not raw[0].strip():
                raise ValueError("首次接口未返回可回放的 JSON 内容")
            draft = _fill_bookkeeping(SketchParser._extract_json(raw[0]), case["image_hash"], case["image_path"])
            errors = validate_v2_draft(draft)
            if errors:
                raise ValueError("；".join(errors))
            options = input_context(metadata)["preprocessing"]
            prepared = preprocess_image(root / case["image_path"], root / "derived",
                                        crop=tuple(options["crop"]) if options.get("crop") else None,
                                        rotation_quarters_cw=options.get("rotation_quarters_cw", 0))
            draft["source"] = prepared.source_metadata()
            draft["work_plane"] = deepcopy(metadata["work_plane"])
            draft = refine_horizontal_axis(draft, prepared.derived_path)
            draft = refine_joint_nodes(draft, prepared.derived_path)
            if review_requested:
                from sketch_action_review import apply_action_review
                try:
                    if len(raw) < 2:
                        raise ValueError("作用点复核缺少冻结原文，不能伪造完整回放。")
                    draft = apply_action_review(draft, SketchParser._extract_json(raw[-1]), image_path=prepared.derived_path)
                    review_status = "completed"
                except (KeyError, TypeError, ValueError) as exc:
                    # 可选复核失败与桌面保持一致：基础草稿留在分母，单独记录复核失败。
                    review_status = "failed"
                    frozen_review = response.get("action_review") or {}
                    message = frozen_review.get("message") or f"作用点复核不可完整回放：{exc}，请人工核对作用节点和支座。"
                    draft["action_review"] = {"status": "failed", "message": message}
                    draft.setdefault("issues", []).append({"id": "action-review-failed", "category": "load_incomplete",
                        "severity": "blocking", "status": "open", "entity_refs": [], "message": message,
                        "resolution": None, "resolved_by": None})
            draft = detect_topology(apply_scale_to_draft(draft))
            predictions[image_id] = draft_to_prediction(draft, image_id=image_id,
                                                        width=prepared.width_px, height=prepared.height_px)
            outcomes.append({"image_id": image_id, "outcome": "PASS",
                             **({"action_review_status": review_status} if review_requested else {})})
        except (KeyError, TypeError, ValueError) as exc:
            original = response["prediction"]
            predictions[image_id] = {"image_id": image_id, "width_px": original["width_px"],
                                     "height_px": original["height_px"], "nodes": [], "members": [],
                                     "supports": [], "loads": [], "intersections": [], "issues": [],
                                     "scale": {"status": "unknown"}}
            outcomes.append({"image_id": image_id, "outcome": "FAIL", "error": str(exc),
                             **({"action_review_status": review_status or "missing"} if review_requested else {})})
    report = evaluate_manifest(root / "manifest.json", prediction_overrides=predictions)
    report.update(case_outcomes=outcomes, new_api_calls=0, replay_successes=sum(
        item["outcome"] == "PASS" for item in outcomes),
        pipeline_hash=PIPELINE_HASH,
        action_review={"attempted_cases": sum("action_review_status" in item for item in outcomes),
                       "completed_cases": sum(item.get("action_review_status") == "completed" for item in outcomes),
                       "failed_or_missing_cases": sum(item.get("action_review_status") in {"failed", "missing"}
                                                      for item in outcomes)},
        limitations=["修复后离线解析同一批首次响应，不等于修复后重新在线识别的成功率。",
                     "输入与真值不变；失败保留在分母，无证据指标仍为 null。"])
    return report


def freeze_manifest(root: str | Path = ROOT) -> dict[str, Any]:
    """Hash-bind all ready cases; reject partial datasets instead of hiding gaps."""
    root = Path(root)
    audit = audit_dataset(root)
    if not audit["cases"]:
        raise ValueError("没有真实图片；请先放入 real_world_cases/images")
    pending = [case for case in audit["cases"] if not case["ready"]]
    if pending:
        summary = ", ".join(f"{item['image_id']}:{'/'.join(item['reasons'])}"
                            for item in pending)
        raise ValueError(f"真实样本尚未完整冻结: {summary}")
    cases = []
    for item in audit["cases"]:
        image_id = item["image_id"]
        image_path = next(path for stem, path in _images(root).items()
                          if stem == image_id)
        metadata_path = root / "metadata" / f"{image_id}.json"
        truth_path = root / "ground_truth" / f"{image_id}.json"
        response_path = root / "responses" / f"{image_id}.json"
        metadata, response = _read(metadata_path), _read(response_path)
        fingerprint = response["request_fingerprint"]
        cases.append({
            "image_id": image_id,
            "image_path": image_path.relative_to(root).as_posix(),
            "image_hash": file_digest(image_path),
            "metadata_path": metadata_path.relative_to(root).as_posix(),
            "metadata_hash": file_digest(metadata_path),
            "ground_truth_path": truth_path.relative_to(root).as_posix(),
            "ground_truth_hash": file_digest(truth_path),
            "response_fixture_path": response_path.relative_to(root).as_posix(),
            "response_hash": file_digest(response_path),
            "provider": fingerprint["provider"], "model": fingerprint["model"],
            "prompt_hash": fingerprint["prompt_hash"],
            "schema_hash": fingerprint["schema_hash"],
            "request_options": fingerprint.get("request_options"),
            "pipeline_hash": fingerprint.get("pipeline_hash"),
            "review_actions": fingerprint.get("review_actions", False),
            "action_prompt_hash": fingerprint.get("action_prompt_hash"),
            "subset_labels": metadata["subset_labels"], "gate": True,
            "source": metadata["source"], "license": metadata["license"],
        })
    manifest = {"format": "space-frame-multimodal-eval/v1", "cases": cases}
    _write(root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status", "recognize", "freeze", "evaluate", "replay"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--provider", choices=tuple(PROVIDER_KEYS), default="openai")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--review-actions", action="store_true", help="额外调用一次视觉接口复核作用点")
    parser.add_argument("--output", help="保存本次审计、识别或评分结果")
    args = parser.parse_args()
    root = Path(args.root)
    try:
        if args.command == "init":
            result = create_templates(root)
        elif args.command == "status":
            result = audit_dataset(root)
        elif args.command == "recognize":
            result = recognize(root, provider=args.provider, model=args.model, max_repairs=args.max_repairs,
                               review_actions=args.review_actions)
        elif args.command == "freeze":
            result = freeze_manifest(root)
        elif args.command == "replay":
            result = replay_real_world(root)
        else:
            result = evaluate_real_world(root)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        _write(Path(args.output), result)
    if args.command == "evaluate" and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
