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


def test_flat_targets_and_missing_observations_remain_reviewable():
    """防止提示词中的平铺杆件引用无法定位，或模型漏报实体绕过未知置信度审核。"""
    from sketch_parser import _fill_bookkeeping
    draft = v2_draft()
    draft["entities"] = [{"id": "E1", "kind": "member", "member": 1,
                          "confidence": 0.4, "source": "user", "verified": True}]
    got = _fill_bookkeeping(draft, IMAGE_HASH, "drawing.png")
    member = got["entities"][0]
    assert member["target"] == {"member": 1}
    assert member["source"] == "vision" and not member["verified"]
    assert member["image_geometry"]["line"]
    missing = [e for e in got["entities"] if e["kind"] == "node"]
    assert len(missing) == 2
    assert all(e["confidence"] is None and e["source"] == "derived" for e in missing)


@pytest.mark.parametrize("confidence", [True, float("nan"), float("inf"), -0.1, 1.2])
def test_invalid_confidence_is_repaired_instead_of_treated_as_high(confidence):
    """防止布尔值、非有限数或越界置信度被画成高确定性实体。"""
    from sketch_parser import _fill_bookkeeping
    draft = v2_draft()
    draft["entities"][0]["confidence"] = confidence
    with pytest.raises(ValueError, match="置信度"):
        _fill_bookkeeping(draft, IMAGE_HASH, "drawing.png")


def test_vision_cannot_resolve_its_own_review_issues():
    """防止模型声称用户已审核，消除本应由人工处理的阻断问题。"""
    from sketch_parser import _fill_bookkeeping
    draft = v2_draft()
    draft["issues"] = [{"category": "low_confidence", "status": "resolved",
                        "resolved_by": "user", "resolution": "confirmed"}]
    got = _fill_bookkeeping(draft, IMAGE_HASH, "drawing.png")
    assert got["issues"][0]["status"] == "open"
    assert got["issues"][0]["resolved_by"] is None


@pytest.mark.parametrize("geometry", [{"point": [float("nan"), 0.5]},
                                      {"line": [[0.5, 0.5], [200, 10]]}, "bad"])
def test_invalid_overlay_coordinates_trigger_repair(geometry):
    """防止像素坐标或非法位置进入覆盖层，审核时对象不可见或 Qt 绘图报错。"""
    from sketch_parser import _fill_bookkeeping
    draft = v2_draft()
    draft["entities"][0]["image_geometry"] = geometry
    with pytest.raises(ValueError, match="识别"):
        _fill_bookkeeping(draft, IMAGE_HASH, "drawing.png")


def test_bad_confidence_receives_one_format_repair(image_path, monkeypatch):
    """防止新增置信度校验被当成网络失败，无法沿已有有限重试流程修复。"""
    parser = SketchParser(provider="openai", api_key="test")
    state, job = running_state()
    calls = []

    def call(image, correction="", **kwargs):
        calls.append(correction)
        draft = v2_draft()
        if len(calls) == 1:
            draft["entities"][0]["confidence"] = 1.2
        return json.dumps(draft)

    monkeypatch.setattr(parser, "_call_llm", call)
    result = parser.parse_v2_with_retry(image_path, state, job)
    assert result.success and result.attempts == 2
    assert "置信度" in calls[1]


def test_confirmed_work_plane_survives_format_repair(image_path, monkeypatch):
    """防止识别及格式修复请求丢失 XY/YZ 平面，沿用提示词默认 XZ 荷载方向。"""
    from image_preprocess import work_plane_payload
    parser = SketchParser(provider="openai", api_key="test")
    state, job = running_state()
    calls = []

    def call(image, correction="", **kwargs):
        calls.append(correction)
        return "invalid" if len(calls) == 1 else json.dumps(v2_draft())

    monkeypatch.setattr(parser, "_call_llm", call)
    result = parser.parse_v2_with_retry(
        image_path, state, job, work_plane=work_plane_payload("YZ", offset=2.5, confirmed=True))
    assert result.success and len(calls) == 2
    assert all('"plane": "YZ"' in c and '"second_axis": "Z"' in c for c in calls)


def test_support_and_load_review_targets_are_distinct_from_attached_node():
    """防止确认节点位置时顺便确认支座类型及荷载大小，或无名荷载互相覆盖。"""
    from sketch_parser import _fill_bookkeeping
    draft = v2_draft()
    draft["entities"] = []
    draft["image_model"]["supports"] = [{"node": 1, "kind": "pinned"}]
    draft["image_model"]["load_cases"] = [{"name": "D", "nodal_loads": [
        {"node": 1, "load": [0, 0, -100, 0, 0, 0]},
        {"node": 1, "load": [10, 0, 0, 0, 0, 0]}]}]
    got = _fill_bookkeeping(draft, IMAGE_HASH, "drawing.png")
    support = next(e for e in got["entities"] if e["kind"] == "support")
    loads = [e for e in got["entities"] if e["kind"] == "load"]
    assert "support" in support["target"] and "node" not in support["target"]
    assert len(loads) == 2 and loads[0]["target"] != loads[1]["target"]
    assert all(e["image_geometry"]["point"] == support["image_geometry"]["point"] for e in loads)


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
