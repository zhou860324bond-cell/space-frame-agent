"""Prepare, freeze, recognize, and score private real-world image cases.

The synthetic release corpus remains immutable.  This module keeps real images
in a separate, explicitly consented manifest and never reports a score for a
case until image, verified ground truth, and provider response are hash-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from multimodal_contract import file_digest
from multimodal_eval.evaluate import evaluate_manifest
from multimodal_workflow import MultimodalControllerState, V2_DRAFT_FORMAT
from sketch_parser import (IMAGE_MIME_TYPES, V2_SKETCH_SYSTEM_PROMPT,
                           SketchParser)

ROOT = Path(__file__).resolve().parent / "real_world_cases"
FORMAT = "space-frame-real-world-case/v1"
PROMPT_HASH = hashlib.sha256(V2_SKETCH_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
SCHEMA_HASH = hashlib.sha256(V2_DRAFT_FORMAT.encode("utf-8")).hexdigest()
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


def create_templates(root: str | Path = ROOT) -> dict[str, Any]:
    """Create only missing annotation/metadata templates for discovered images."""
    root = Path(root)
    created: list[str] = []
    for image_id, image_path in _images(root).items():
        with Image.open(image_path) as image:
            width, height = image.size
        metadata_path = root / "metadata" / f"{image_id}.json"
        if not metadata_path.exists():
            _write(metadata_path, {
                "format": FORMAT, "image_id": image_id,
                "source": "user-provided", "license": "private-evaluation-only",
                "consent_to_evaluate": False, "subset_labels": [],
                "notes": "确认无敏感信息后，将 consent_to_evaluate 改为 true。",
            })
            created.append(metadata_path.relative_to(root).as_posix())
        truth_path = root / "ground_truth" / f"{image_id}.json"
        if not truth_path.exists():
            _write(truth_path, {
                "image_id": image_id, "width_px": width, "height_px": height,
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
                with Image.open(image_path) as image:
                    dimensions_match = [truth.get("width_px"), truth.get("height_px")] \
                        == list(image.size)
            except OSError:
                dimensions_match = False
            if truth.get("image_id") != image_id or not dimensions_match:
                reasons.append("ground_truth_image_mismatch")
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
              max_repairs: int = 2) -> dict[str, Any]:
    """Recognize every image without an existing frozen response."""
    root = Path(root)
    env_var = PROVIDER_KEYS[provider]
    if not os.environ.get(env_var):
        return {"provider": provider, "status": "SKIPPED_NO_CREDENTIALS",
                "environment_variable": env_var, "results": []}
    parser = SketchParser.from_env(provider, model=model)
    results = []
    for image_id, image_path in _images(root).items():
        output_path = root / "responses" / f"{image_id}.json"
        if output_path.exists():
            results.append({"image_id": image_id, "status": "SKIPPED_EXISTS"})
            continue
        state = MultimodalControllerState()
        image_hash = file_digest(image_path)
        state.load_image(image_hash)
        job = state.start_recognition(f"real-{provider}-{image_id}")
        parsed = parser.parse_v2_with_retry(
            image_path, state, job, max_repairs=max_repairs)
        if not parsed.success:
            results.append({"image_id": image_id, "status": "FAIL",
                            "attempts": parsed.attempts, "errors": parsed.errors})
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        _write(output_path, {
            "fixture": "real-provider-response-v1",
            "prediction": draft_to_prediction(
                parsed.draft, image_id=image_id, width=width, height=height),
            "redaction": {"credentials": "not_recorded",
                          "user_metadata": "not_recorded"},
            "request_fingerprint": {
                "provider": provider, "model": parser._model,
                "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH,
            },
            "runtime": {"attempts": parsed.attempts,
                        "duration_ms": parsed.duration_ms},
        })
        results.append({"image_id": image_id, "status": "PASS",
                        "attempts": parsed.attempts})
    return {"provider": provider, "model": parser._model, "status": "COMPLETE",
            "results": results}


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
            "ground_truth_path": truth_path.relative_to(root).as_posix(),
            "ground_truth_hash": file_digest(truth_path),
            "response_fixture_path": response_path.relative_to(root).as_posix(),
            "response_hash": file_digest(response_path),
            "provider": fingerprint["provider"], "model": fingerprint["model"],
            "prompt_hash": fingerprint["prompt_hash"],
            "schema_hash": fingerprint["schema_hash"],
            "subset_labels": metadata["subset_labels"], "gate": True,
            "source": metadata["source"], "license": metadata["license"],
        })
    manifest = {"format": "space-frame-multimodal-eval/v1", "cases": cases}
    _write(root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status", "recognize", "freeze", "evaluate"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--provider", choices=tuple(PROVIDER_KEYS), default="openai")
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    root = Path(args.root)
    try:
        if args.command == "init":
            result = create_templates(root)
        elif args.command == "status":
            result = audit_dataset(root)
        elif args.command == "recognize":
            result = recognize(root, provider=args.provider, model=args.model)
        elif args.command == "freeze":
            result = freeze_manifest(root)
        else:
            result = evaluate_manifest(root / "manifest.json")
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "evaluate" and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
