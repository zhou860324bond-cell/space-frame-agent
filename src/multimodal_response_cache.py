"""Explicit offline cache for reproducible multimodal provider responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from multimodal_contract import canonical_digest


_KEY_FIELDS = ("image_hash", "provider", "model", "prompt_hash", "schema_hash")


@dataclass(frozen=True)
class CachedVisionResponse:
    key: dict[str, str]
    raw_response: str


class OfflineVisionResponseCache:
    """Store raw text plus non-secret provenance; never stores image bytes or keys."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    @staticmethod
    def _key(metadata: Mapping[str, Any]) -> dict[str, str]:
        key = {field: str(metadata.get(field, "")) for field in _KEY_FIELDS}
        if any(not value for value in key.values()):
            raise ValueError(f"缓存键必须包含: {', '.join(_KEY_FIELDS)}")
        return key

    def put(self, metadata: Mapping[str, Any], raw_response: str) -> Path:
        key = self._key(metadata)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{canonical_digest(key)}.json"
        payload = {"format": "space-frame-vision-response/v1",
                   "key": key, "raw_response": str(raw_response)}
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                   indent=2), encoding="utf-8")
        return path

    def get(self, metadata: Mapping[str, Any]) -> CachedVisionResponse | None:
        key = self._key(metadata)
        path = self.directory / f"{canonical_digest(key)}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "space-frame-vision-response/v1" \
                or payload.get("key") != key:
            raise ValueError("离线响应缓存已损坏或键不匹配")
        return CachedVisionResponse(key, str(payload.get("raw_response", "")))
