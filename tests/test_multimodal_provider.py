"""MM2-03 provider, repair, cancellation, and offline-cache contracts."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from multimodal_contract import canonical_digest
from multimodal_response_cache import OfflineVisionResponseCache
from multimodal_workflow import (MultimodalControllerState, MultimodalPhase,
                                 migrate_v1_payload)
from sketch_parser import SketchParser, V2_SKETCH_SYSTEM_PROMPT


IMAGE_HASH = "a" * 64


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "derived.png"
    Image.new("RGB", (4, 3), "white").save(path)
    return str(path)


def v2_draft():
    return migrate_v1_payload({
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 1, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "material": "", "section": ""}],
        "materials": [], "sections": [], "supports": [], "load_cases": [],
    }, image_hash=IMAGE_HASH, source_path="derived.png")


def running_state():
    state = MultimodalControllerState()
    state.load_image(IMAGE_HASH)
    return state, state.start_recognition("job")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "deepseek"])
def test_all_providers_use_the_same_v2_contract_without_touching_session(
        provider, image_path, monkeypatch):
    parser = SketchParser(provider=provider, api_key="test")
    state, job = running_state()
    formal_session = {"sentinel": [1, 2, 3]}
    captured = {}

    def call(image, correction="", *, system_prompt=None):
        captured.update(correction=correction, system_prompt=system_prompt)
        return json.dumps(v2_draft())

    monkeypatch.setattr(parser, "_call_llm", call)
    result = parser.parse_v2_with_retry(image_path, state, job)
    assert result.success and result.attempts == 1
    assert captured["system_prompt"] == V2_SKETCH_SYSTEM_PROMPT
    assert IMAGE_HASH in captured["correction"]
    assert state.phase(formal_session) == MultimodalPhase.REVIEW_REQUIRED
    assert formal_session == {"sentinel": [1, 2, 3]}


def test_v2_format_error_is_repaired_at_most_twice(image_path, monkeypatch):
    parser = SketchParser(api_key="test")
    state, job = running_state()
    responses = iter(["not-json", json.dumps(v2_draft())])
    corrections = []

    def call(image, correction="", *, system_prompt=None):
        corrections.append(correction)
        return next(responses)

    monkeypatch.setattr(parser, "_call_llm", call)
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=2)
    assert result.success and result.attempts == 2
    assert "上次输出错误" in corrections[1]


def test_two_failed_repairs_enter_failed_state(image_path, monkeypatch):
    parser = SketchParser(api_key="test")
    state, job = running_state()
    monkeypatch.setattr(
        parser, "_call_llm",
        lambda image, correction="", *, system_prompt=None: "not-json")
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=2)
    assert not result.success and result.attempts == 3
    assert len(result.errors) == 3
    assert state.phase({}) == MultimodalPhase.RECOGNITION_FAILED


def test_api_failure_does_not_enter_repair_loop(image_path, monkeypatch):
    parser = SketchParser(api_key="test")
    state, job = running_state()

    def fail(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(parser, "_call_llm", fail)
    result = parser.parse_v2_with_retry(image_path, state, job)
    assert not result.success and result.attempts == 1
    assert "API 调用失败" in result.errors[0]
    assert state.phase({}) == MultimodalPhase.RECOGNITION_FAILED


def test_cancel_discards_provider_response(image_path, monkeypatch):
    parser = SketchParser(api_key="test")
    state, job = running_state()
    checks = iter([False, True])
    monkeypatch.setattr(
        parser, "_call_llm",
        lambda image, correction="", *, system_prompt=None: json.dumps(v2_draft()))
    result = parser.parse_v2_with_retry(
        image_path, state, job, is_cancelled=lambda: next(checks))
    assert result.cancelled and not result.success
    assert state.draft is None
    assert state.phase({}) == MultimodalPhase.IMAGE_LOADED


def test_offline_response_cache_is_keyed_by_full_provenance(tmp_path):
    cache = OfflineVisionResponseCache(tmp_path / "cache")
    key = {"image_hash": IMAGE_HASH, "provider": "openai", "model": "vision",
           "prompt_hash": canonical_digest("prompt"),
           "schema_hash": canonical_digest("schema")}
    path = cache.put(key, '{"format":"fixture"}')
    assert path.is_file()
    restored = cache.get(key)
    assert restored is not None
    assert restored.raw_response == '{"format":"fixture"}'
    assert cache.get({**key, "model": "other"}) is None
