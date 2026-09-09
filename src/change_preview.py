"""紧凑、可复核的 Domain IR 变更摘要。"""

from __future__ import annotations

import hashlib
import json
from typing import Any


_IDENTITY_FIELDS = {
    "nodes": "id",
    "members": "id",
    "materials": "name",
    "sections": "name",
    "supports": "node",
    "load_cases": "name",
    "combos": "name",
    "sets": "name",
}


def model_digest(model: dict[str, Any]) -> str:
    canonical = json.dumps(model, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _indexed(items: list[Any], identity: str) -> dict[Any, Any] | None:
    if not all(isinstance(item, dict) and identity in item for item in items):
        return None
    indexed = {item[identity]: item for item in items}
    return indexed if len(indexed) == len(items) else None


def _display(value: Any) -> Any:
    """标量原样展示；复杂顶层值只报类型和规模，避免把模型复制进 diff。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {"type": "object", "items": len(value)}
    if isinstance(value, list):
        return {"type": "array", "items": len(value)}
    return {"type": type(value).__name__}


def diff_models(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """比较两份模型，只返回实体 ID 和计数，不泄漏一整份候选模型。"""
    changed_keys = sorted(key for key in set(before) | set(after)
                          if before.get(key) != after.get(key))
    collections: dict[str, dict[str, Any]] = {}
    fields: dict[str, dict[str, Any]] = {}

    for key in changed_keys:
        old, new = before.get(key), after.get(key)
        identity = _IDENTITY_FIELDS.get(key)
        if identity and isinstance(old or [], list) and isinstance(new or [], list):
            old_map = _indexed(old or [], identity)
            new_map = _indexed(new or [], identity)
            if old_map is not None and new_map is not None:
                old_ids, new_ids = set(old_map), set(new_map)
                collections[key] = {
                    "before": len(old_map),
                    "after": len(new_map),
                    "added": sorted(new_ids - old_ids, key=str),
                    "removed": sorted(old_ids - new_ids, key=str),
                    "changed": sorted(
                        (item for item in old_ids & new_ids
                         if old_map[item] != new_map[item]), key=str),
                }
                continue
        fields[key] = {"before": _display(old), "after": _display(new)}

    return {
        "changed": bool(changed_keys),
        "before_hash": model_digest(before),
        "after_hash": model_digest(after),
        "changed_top_level": changed_keys,
        "collections": collections,
        "fields": fields,
    }


def preview_digest(tool: str, arguments: dict[str, Any],
                   before_hash: str, after_hash: str) -> str:
    payload = {"tool": tool, "arguments": arguments,
               "before_hash": before_hash, "after_hash": after_hash}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), default=str)
    return "chg_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
