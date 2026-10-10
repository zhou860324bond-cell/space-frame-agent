"""MM2-03 provider, repair, cancellation, and offline-cache contracts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
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
def test_empty_and_truncated_responses_preserve_diagnostics_without_enabled_repairs(image_path, raw, finish, error):
    """关闭修复时公开图空响应或截断片段必须拒收，不能隐藏新增付费调用。"""
    parser = SketchParser(provider="deepseek", api_key="test-key")
    calls = []
    actual = json.dumps(v2_draft()) if raw == "valid" else raw

    def create(**kwargs):
        calls.append(kwargs)
        return _chat_response(actual, finish, reasoning="不可记录的推理内容")

    parser._client = _chat_client(create)
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=0)
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
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=0)
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


@pytest.mark.parametrize("enabled,expected_calls", [(False, 1), (True, 2)])
def test_optional_action_review_counts_each_call_and_retains_stage_metadata(image_path, monkeypatch, enabled, expected_calls):
    """默认不得增加收费调用；开启后两个阶段的原文、耗时计数和接口元数据必须保留。"""
    parser = SketchParser(api_key="test")
    state, job = running_state()
    calls = []

    def call(*args, **kwargs):
        calls.append(kwargs["system_prompt"])
        parser._last_call_metadata = {"total_tokens": 100, "finish_reason": "stop"}
        return json.dumps(v2_draft() if len(calls) == 1 else {"actions": [], "warnings": []})

    monkeypatch.setattr(parser, "_call_llm", call)
    result = parser.parse_v2_with_retry(image_path, state, job, review_actions=enabled)
    assert result.success and result.attempts == expected_calls
    assert len(result.raw_responses) == len(calls) == len(result.call_metadata) == expected_calls
    if enabled:
        assert [m["stage"] for m in result.call_metadata] == ["geometry", "action_review"]
        assert result.draft["action_review"]["status"] == "completed"
    else:
        assert "action_review" not in result.draft


@pytest.mark.parametrize("failure", ["invalid", "numeric", "offline", "truncated"])
def test_action_review_failure_retains_base_draft_without_paid_repair(image_path, monkeypatch, failure):
    """可选复核的格式/网络失败不能丢失基础草稿或继续收费修复，更不能补造荷载。"""
    parser = SketchParser(api_key="test")
    state, job = running_state()
    calls = []

    def call(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return json.dumps(v2_draft())
        if failure == "offline":
            raise ConnectionError("offline")
        if failure == "truncated":
            parser._last_call_metadata = {"finish_reason": "length"}
        return "invalid" if failure == "invalid" else json.dumps({"actions": [], "warnings": [], "force": 100})

    monkeypatch.setattr(parser, "_call_llm", call)
    result = parser.parse_v2_with_retry(image_path, state, job, review_actions=True)
    assert result.success and result.attempts == 2 and len(calls) == 2
    assert result.draft["action_review"]["status"] == "failed"
    assert result.errors and result.draft["image_model"]["load_cases"] == []
    assert "。。" not in result.errors[0]
    assert any(i["id"] == "action-review-failed" and i["severity"] == "blocking" for i in result.draft["issues"])


@pytest.mark.parametrize("cancel_stage", ["before_review", "after_review", "new_image"])
def test_cancel_or_stale_image_during_action_review_cannot_commit_any_draft(image_path, monkeypatch, cancel_stage):
    """取消应阻止额外复核；复核期间的新图片或取消不能被旧响应覆盖。"""
    parser = SketchParser(api_key="test")
    state, job = running_state()
    calls, checks = [], 0

    def cancelled():
        nonlocal checks
        checks += 1
        return cancel_stage == "before_review" and checks >= 3 or cancel_stage == "after_review" and len(calls) >= 2

    def call(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2 and cancel_stage == "new_image":
            state.load_image("b" * 64)
        return json.dumps(v2_draft() if len(calls) == 1 else {"actions": [], "warnings": []})

    monkeypatch.setattr(parser, "_call_llm", call)
    result = parser.parse_v2_with_retry(image_path, state, job, review_actions=True, is_cancelled=cancelled)
    assert result.cancelled and not result.success and state.draft is None
    assert len(calls) == (1 if cancel_stage == "before_review" else 2)


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


def _holdout_response(name):
    path = Path(__file__).resolve().parents[1] / "multimodal_eval/public_cases/holdout_01/responses"
    return json.loads((path / f"{name}.json").read_text(encoding="utf-8"))


def test_compact_observations_reuse_geometry_without_changing_frozen_wall_evidence():
    """墙图重复输出所有节点及杆件位置；省略副本应逐项还原且不抹掉未知单位和重复边。"""
    from sketch_parser import _fill_bookkeeping
    raw = _holdout_response("wall_truss")["raw_responses"][0]
    original = json.loads(raw)
    compact = deepcopy(original)
    for entity in compact["entities"]:
        if entity["kind"] in {"node", "member"}:
            entity.pop("image_geometry")
    assert _fill_bookkeeping(compact, IMAGE_HASH, "drawing.png") == _fill_bookkeeping(
        original, IMAGE_HASH, "drawing.png")
    assert len(compact["image_model"]["members"]) == 7
    assert any(issue["category"] == "load_incomplete" for issue in compact["issues"])


@pytest.mark.parametrize("confidence", [None, 0.0, 0.75])
def test_compact_geometry_keeps_observed_confidence_and_manual_review(confidence):
    """缩减输出不能把 null 或低置信度变成高分，也不能替用户确认节点与杆件。"""
    from sketch_parser import _fill_bookkeeping
    payload = v2_draft()
    payload["entities"] = [{"id": "N1", "kind": "node", "target": {"node": 1},
                            "confidence": confidence},
                           {"id": "M1", "kind": "member", "target": {"member": 1},
                            "confidence": confidence}]
    got = _fill_bookkeeping(payload, IMAGE_HASH, "drawing.png")
    for entity in got["entities"][:2]:
        assert entity["confidence"] == entity["recognition_confidence"] == confidence
        assert entity["source"] == "vision" and entity["verified"] is False
        assert entity["image_geometry"]
    assert got["entities"][2]["confidence"] is None
    assert got["entities"][2]["source"] == "derived"
    state, job = running_state()
    assert state.complete_recognition(job, got)
    assert state.phase({}) == MultimodalPhase.REVIEW_REQUIRED


@pytest.mark.parametrize("finish", ["length", "max_tokens"])
def test_frozen_tower_truncation_can_receive_only_enabled_compact_repair(image_path, finish):
    """塔架真实首响应超限曾直接停止；允许修复时重读原图，不拼接原文或接受部分对象。"""
    from image_preprocess import work_plane_payload
    original = _holdout_response("tower_truss")["raw_responses"][0]
    repaired = json.dumps(v2_draft(), separators=(",", ":"))  # 模拟协议恢复，不是图样准确率。
    replies = iter([_chat_response(original, finish), _chat_response(repaired)])
    calls = []
    parser = SketchParser(provider="deepseek", api_key="test")

    def create(**kwargs):
        calls.append(kwargs)
        return next(replies)

    parser._client = _chat_client(create)
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=1,
                                       work_plane=work_plane_payload("XY", confirmed=True))
    assert result.success and result.attempts == len(calls) == 2
    assert result.raw_responses == [original, repaired]
    assert [m["finish_reason"] for m in result.call_metadata] == [finish, "stop"]
    assert "长度限制" in result.errors[0] and "API 调用失败" not in result.errors[0]
    correction = calls[1]["messages"][1]["content"][1]["text"]
    assert "完整紧凑" in correction and "不补全截断片段" in correction
    assert '"plane": "XY"' in correction and IMAGE_HASH in correction
    assert calls[0]["messages"][1]["content"][2] == calls[1]["messages"][1]["content"][2]
    assert all(call["max_tokens"] == 4000 for call in calls)
    assert state.phase({}) == MultimodalPhase.REVIEW_REQUIRED


@pytest.mark.parametrize("max_repairs", [0, 1, 2])
@pytest.mark.parametrize("finish", ["length", "max_tokens"])
def test_repeated_truncation_never_exceeds_budget_or_accepts_closed_json(image_path, max_repairs, finish):
    """即使超限响应看似完整 JSON，也不能收下；持续超限最多消耗明确设置的修复次数。"""
    parser = SketchParser(api_key="test")
    calls = []
    raw = json.dumps(v2_draft())

    def create(**kwargs):
        calls.append(kwargs)
        return _chat_response(raw, finish)

    parser._client = _chat_client(create)
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=max_repairs)
    assert not result.success and result.draft is None and state.draft is None
    assert result.attempts == len(calls) == len(result.errors) == max_repairs + 1
    assert result.raw_responses == [raw] * len(calls)
    assert len(result.call_metadata) == len(calls)
    assert state.phase({}) == MultimodalPhase.RECOGNITION_FAILED


@pytest.mark.parametrize("stale", [False, True])
def test_cancellation_before_truncation_repair_prevents_another_call(image_path, stale):
    """超限修复间取消或换图时旧任务必须停止，不能继续调用或交付旧草稿。"""
    parser = SketchParser(api_key="test")
    state, job = running_state()
    checks = 0
    calls = []

    def cancelled():
        nonlocal checks
        checks += 1
        if checks == 3 and stale:
            state.load_image("b" * 64)
            return False
        return checks == 3

    def create(**kwargs):
        calls.append(kwargs)
        return _chat_response('{"image_model":', "length")

    parser._client = _chat_client(create)
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=2, is_cancelled=cancelled)
    assert result.cancelled and not result.success and len(calls) == 1
    assert state.draft is None
    assert state.phase({}) == MultimodalPhase.IMAGE_LOADED


def test_network_failure_during_truncation_repair_stops_with_separate_metadata(image_path):
    """超限后修复请求断网不应再重试，不能把上一请求的 token 用量算到失败请求。"""
    parser = SketchParser(api_key="test")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise ConnectionError("offline")
        return _chat_response('{"image_model":', "length")

    parser._client = _chat_client(create)
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=2)
    assert not result.success and len(calls) == result.attempts == 2
    assert "长度限制" in result.errors[0] and "API 调用失败" in result.errors[1]
    assert len(result.call_metadata) == 1 and result.call_metadata[0]["attempt"] == 1


@pytest.mark.parametrize("repair", [False, True])
def test_frozen_wall_duplicate_gives_exact_repair_feedback_without_deleting_members(image_path, repair):
    """墙图杆件 5/6 反向重复必须指出端点和原编号；无修复时不能自动删边收下错误结构。"""
    raw = _holdout_response("wall_truss")["raw_responses"][0]
    calls = []
    parser = SketchParser(api_key="test")

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _chat_response(raw)
        return _chat_response(json.dumps(v2_draft()))  # 只验证模拟修复通道，不评分墙图。

    parser._client = _chat_client(create)
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=int(repair))
    assert result.success is repair and len(calls) == 1 + int(repair)
    assert result.raw_responses[0] == raw
    assert "杆件 6 与杆件 5 重复" in result.errors[0]
    assert "节点 3 和 5" in result.errors[0]
    if repair:
        text = calls[1]["messages"][1]["content"][1]["text"]
        assert "不得仅删除报错项" in text and "节点 3 和 5" in text
    else:
        assert state.draft is None and len(json.loads(raw)["image_model"]["members"]) == 7


def test_self_connection_diagnostic_is_distinct_from_reversed_duplicates():
    """自连接与反向重复需要不同的端点诊断，防止修复反馈只让模型随意删除合法边。"""
    from multimodal_workflow import validate_v2_draft
    payload = v2_draft()
    payload["image_model"]["members"] = [{"id": 8, "i": 1, "j": 1}]
    before = deepcopy(payload)
    errors = validate_v2_draft(payload)
    assert any("杆件 8 自连接" in e and "两端均为节点 1" in e for e in errors)
    assert not any("重复" in e for e in errors)
    assert payload == before


def test_anthropic_truncation_repair_uses_native_image_transport(image_path):
    """有限超限修复也要兼容 Anthropic 原生图片块，保留两次原文与各自停止原因。"""
    raw = _holdout_response("tower_truss")["raw_responses"][0]
    repaired = json.dumps(v2_draft())
    replies = iter([SimpleNamespace(content=[SimpleNamespace(text=raw)], stop_reason="max_tokens"),
                    SimpleNamespace(content=[SimpleNamespace(text=repaired)], stop_reason="end_turn")])
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return next(replies)

    parser = SketchParser(provider="anthropic", api_key="test")
    parser._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=1)
    assert result.success and result.attempts == len(calls) == 2
    assert result.raw_responses == [raw, repaired]
    assert [m["finish_reason"] for m in result.call_metadata] == ["max_tokens", "end_turn"]
    assert all(call["messages"][0]["content"][-1]["type"] == "image" for call in calls)


def test_empty_content_still_stops_with_repairs_enabled(image_path):
    """缩减和超限修复不能恢复空响应的重复付费问题，空内容仍立即停止。"""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _chat_response("")

    parser = SketchParser(api_key="test")
    parser._client = _chat_client(create)
    state, job = running_state()
    result = parser.parse_v2_with_retry(image_path, state, job, max_repairs=2)
    assert not result.success and result.attempts == len(calls) == 1
    assert "未返回识别内容" in result.errors[0]


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
