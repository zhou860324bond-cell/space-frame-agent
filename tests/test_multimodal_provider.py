"""MM2-03 provider, repair, cancellation, and offline-cache contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from multimodal_contract import canonical_digest
from multimodal_response_cache import OfflineVisionResponseCache
from multimodal_workflow import (MultimodalControllerState, MultimodalPhase,
                                 migrate_v1_payload)
from sketch_parser import SketchParser, V2_SKETCH_SYSTEM_PROMPT


IMAGE_HASH = "a" * 64


def _chat_response(raw, finish="stop", *, reasoning=""):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
        message=SimpleNamespace(content=raw, reasoning_content=reasoning))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=80, total_tokens=200,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=0)))


def _chat_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.parametrize(("provider", "model", "disabled"), [
    ("deepseek", "deepseek-flash", True),
    ("deepseek", "deepseek-v4-flash-vision-exp", True),
    ("deepseek", "custom-vision", False), ("openai", "gpt-4o", False),
])
def test_vision_request_controls_thinking_only_for_supported_deepseek_models(provider, model, disabled):
    """视觉提取曾默认消耗推理输出预算；兼容设置不能误传给其他提供商或自定义模型。"""
    parser = SketchParser(provider=provider, model=model, api_key="test")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _chat_response("{}")

    parser._client = _chat_client(create)
    assert parser._call_llm("data:image/png;base64,AA==") == "{}"
    assert captured["max_tokens"] == 4000
    assert (captured.get("extra_body") == {"thinking": {"type": "disabled"}}) is disabled


@pytest.mark.parametrize(("raw", "finish", "error"), [
    ("", "stop", "未返回识别内容"),
    ("", "length", "长度限制"),
    ("valid", "length", "长度限制"),
])
def test_empty_and_truncated_responses_preserve_diagnostics_without_paid_repairs(image_path, raw, finish, error):
    """公开图空响应被当成普通 JSON 错误重试，截断的有效片段也可能被误收为完整草稿。"""
    parser = SketchParser(provider="deepseek", api_key="test-key")
    calls = []
    actual = json.dumps(v2_draft()) if raw == "valid" else raw

    def create(**kwargs):
        calls.append(kwargs)
        return _chat_response(actual, finish, reasoning="不可记录的推理内容")

    parser._client = _chat_client(create)
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=2)
    assert not result.success and result.attempts == 1 and len(calls) == 1
    assert error in result.errors[0] and "请" in result.errors[0]
    assert result.raw_responses == [actual]
    assert result.call_metadata == [{"attempt": 1, "finish_reason": finish,
        "prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200,
        "reasoning_tokens": 0, "content_characters": len(actual), "has_reasoning_content": True}]
    assert "不可记录的推理内容" not in json.dumps(result.call_metadata, ensure_ascii=False)
    assert state.phase({}) == MultimodalPhase.RECOGNITION_FAILED


def test_missing_choice_is_reported_without_index_error_or_repair(image_path):
    """接口缺少 choices 时应保留明确诊断并停止，不报越界或继续付费修复。"""
    parser = SketchParser(api_key="test")
    parser._client = _chat_client(lambda **kwargs: SimpleNamespace(choices=[], usage=None))
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job)
    assert result.attempts == 1 and "未返回识别内容" in result.errors[0]
    assert result.call_metadata[0]["finish_reason"] == "missing_choice"


def test_api_error_never_reuses_previous_attempt_diagnostics(image_path):
    """格式修复后的网络失败不能沿用上一请求的结束原因或 token 用量。"""
    parser = SketchParser(api_key="test")
    responses = iter([_chat_response("not-json"), ConnectionError("offline")])

    def create(**kwargs):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    parser._client = _chat_client(create)
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job)
    assert result.attempts == 2 and "offline" in result.errors[-1]
    assert len(result.call_metadata) == 1 and result.call_metadata[0]["attempt"] == 1


def test_anthropic_text_blocks_preserve_stop_reason_without_thinking_content(image_path):
    """文本前出现非文本推理块时不能 AttributeError，也不能把推理内容写进识别原文。"""
    parser = SketchParser(provider="anthropic", api_key="test")
    raw = json.dumps(v2_draft())
    response = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="不可记录"),
        SimpleNamespace(type="text", text=raw[:20]), SimpleNamespace(type="text", text=raw[20:])],
        stop_reason="end_turn", usage=SimpleNamespace(input_tokens=50, output_tokens=60))
    parser._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job)
    assert result.success and result.raw_responses == [raw]
    assert result.call_metadata[0]["finish_reason"] == "end_turn"
    assert result.call_metadata[0]["completion_tokens"] == 60


def test_anthropic_token_limit_stops_instead_of_accepting_a_partial_draft(image_path):
    """Anthropic 的 max_tokens 停止原因与 OpenAI 的 length 都表示输出未完整完成。"""
    parser = SketchParser(provider="anthropic", api_key="test")
    response = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(v2_draft()))], stop_reason="max_tokens")
    parser._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job)
    assert not result.success and result.attempts == 1 and "长度限制" in result.errors[0]


def test_unbound_symbol_observations_keep_geometry_and_one_blocking_record():
    """外伸图只给符号观察和问题、没有求解条目，不能丢弃几何或补出数值荷载。"""
    from sketch_parser import _fill_bookkeeping
    draft = v2_draft()
    draft["entities"].extend([{"id": name, "kind": "load", "confidence": 0.6,
        "target": {"load": {"case": "D", "collection": "nodal_loads", "name": name}},
        "image_geometry": {"point": [0.5, 0.2]}} for name in ("RA", "F")])
    got = _fill_bookkeeping(draft, IMAGE_HASH, "drawing.png")
    assert len(got["image_model"]["nodes"]) == 2 and got["image_model"]["load_cases"] == []
    assert not any(e["kind"] == "load" for e in got["entities"])
    pending = [i for i in got["issues"] if i.get("observations")]
    assert len(pending) == 1 and len(pending[0]["observations"]) == 2
    assert pending[0]["severity"] == "blocking" and pending[0]["status"] == "open"
    assert pending[0]["unbound_symbols_only"] is True
    assert {e["target"]["load"]["name"] for e in pending[0]["observations"]} == {"RA", "F"}


def test_unbound_load_quarantine_does_not_weaken_other_entity_references():
    """符号荷载隔离不能顺便接受不存在的杆件或支座引用。"""
    from sketch_parser import _fill_bookkeeping
    draft = v2_draft()
    draft["entities"].append({"id": "bad", "kind": "member", "target": {"member": 99}, "confidence": None})
    with pytest.raises(ValueError, match="审核对象"):
        _fill_bookkeeping(draft, IMAGE_HASH, "drawing.png")


def test_existing_global_load_blocker_cannot_become_an_ignorable_symbol_group():
    """已有全局缺值问题合并符号观察时，不能因忽略符号解除原数值问题。"""
    from sketch_parser import _fill_bookkeeping
    draft = v2_draft()
    draft["issues"].append({"id": "missing", "category": "load_incomplete",
                            "entity_refs": [], "message": "荷载数值缺失"})
    draft["entities"].append({"id": "P", "kind": "load", "confidence": None,
                             "target": {"load": {"name": "P"}}})
    got = _fill_bookkeeping(draft, IMAGE_HASH, "drawing.png")
    problem = next(i for i in got["issues"] if i["id"] == "missing")
    assert problem["unbound_symbols_only"] is False and problem["status"] == "open"
    assert problem["message"] == "荷载数值缺失" and problem["observations"]


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
