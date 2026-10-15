"""应用连接的真实边界：密钥加密、协议适配与对象版本，均不请求真实账户。"""
import json
import os
from dataclasses import replace

import pytest

from model_profiles import ModelProfile, ProfileStore, redact
from object_context import context_message, model_revision


def profile(**kwargs):
    return replace(ModelProfile("a", "连接 A", "model-a", "https://api.example.com/v1",
                                api_key="sk-test-only-123456789"), **kwargs)


def test_session_only_key_never_reaches_disk(tmp_path):
    """未选择安全存储时，只写配置；重启后明确需要补充密钥。"""
    store = ProfileStore(tmp_path / "models.json")
    p = profile()
    store.save([p], p.id)
    assert p.api_key not in store.path.read_text(encoding="utf-8")
    loaded, selected = store.load()
    assert selected == p.id and loaded[0].api_key == ""
    store.save(loaded + [profile(id="b")], "b")
    assert len(store.load()[0]) == 2
    assert p.api_key not in repr(p)


@pytest.mark.skipif(os.name != "nt", reason="DPAPI 是 Windows 当前用户安全存储")
def test_windows_secure_key_roundtrip(tmp_path):
    """真实 DPAPI 回读假密钥，防止保存后丢失或明文落盘。"""
    p = profile(persist_key=True)
    store = ProfileStore(tmp_path / "models.json")
    store.save([p], p.id)
    assert p.api_key not in store.path.read_text(encoding="utf-8")
    assert store.load()[0][0].api_key == p.api_key


def test_secure_store_failure_leaves_existing_file(monkeypatch, tmp_path):
    """加密失败不能损坏原连接，也不能退回保存明文。"""
    store = ProfileStore(tmp_path / "models.json")
    store.save([profile()], "a")
    before = store.path.read_bytes()
    def fail(*args):
        raise OSError("encryption failed")
    monkeypatch.setattr("model_profiles._dpapi", fail)
    with pytest.raises(OSError):
        store.save([profile(persist_key=True)], "a")
    assert store.path.read_bytes() == before


def test_corrupt_secure_key_reports_reconfigure(tmp_path):
    """损坏密文给出重配说明，不能静默改用离线。"""
    store = ProfileStore(tmp_path / "models.json")
    store.save([profile()], "a")
    doc = json.loads(store.path.read_text(encoding="utf-8"))
    doc["profiles"][0]["sealed_key"] = "invalid!"
    store.path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="重新配置"):
        store.load()


@pytest.mark.parametrize("overrides", [
    {"base_url": "http://api.example.com"}, {"base_url": "https://user:secret@api.example.com"},
    {"base_url": "https://api.example.com?key=abc"}, {"protocol": "invalid"},
    {"model": ""}, {"api_key": "two keys"}, {"temperature": 2}, {"max_tokens": 1},
])
def test_invalid_connection_is_rejected_before_network(overrides):
    """错误地址、鉴权和参数在请求前阻止，避免发往错误地址。"""
    with pytest.raises(ValueError):
        profile(**overrides).validate()


def test_identity_includes_endpoint_key_and_parameters():
    """同名模型在另一提供商或另一密钥下必须使用新连接。"""
    p = profile()
    for other in [replace(p, base_url="https://other.example.com"),
                  replace(p, api_key="another-secret"), replace(p, temperature=0.5)]:
        assert p.identity() != other.identity()
    assert p.api_key not in str(p.identity())


def test_redaction_removes_keys_and_authorization():
    """SDK 错误可能回显密钥，状态与日志必须先隐藏。"""
    key = profile().api_key
    output = redact(f"401 {key}, Authorization: Bearer unknown-secret api_key=other-secret", [key])
    assert key not in output and "unknown-secret" not in output and "other-secret" not in output


def test_object_reference_rejects_edited_or_replaced_model():
    """同编号坐标变化后，旧对象引用不能继续发送。"""
    model = {"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}], "units": "kN-m"}
    revision = model_revision(model)
    data = context_message(model, [("node", 1)], revision)
    assert json.loads(data.split("\n", 1)[1])["selected_objects"][0]["object"]["x"] == 0
    model["nodes"][0]["x"] = 2
    with pytest.raises(ValueError, match="过期"):
        context_message(model, [("node", 1)], revision)


def test_object_context_does_not_rewrite_user_confirmation():
    """引用放入系统数据，保持“确认”等授权原文。"""
    from agent import ScriptedProvider, Session
    from conversation import Conversation
    conversation = Conversation(ScriptedProvider([]), Session())
    conversation.object_context = "用户所选对象数据"
    messages = conversation.build_messages("确认")
    assert messages[-1] == {"role": "user", "content": "确认"}
    assert any(m["content"] == conversation.object_context for m in messages[:-1])


def test_native_anthropic_keeps_tool_pairs(monkeypatch):
    """从兼容历史切到原生 Messages 时保留工具编号与结果配对。"""
    from configured_provider import AnthropicProvider
    provider = AnthropicProvider(profile(protocol="anthropic"))
    requests = []
    def request(path, payload):
        requests.append(payload)
        return {"content": [{"type": "tool_use", "id": "next", "name": "query", "input": {}}]}
    monkeypatch.setattr(provider, "_request", request)
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "query"},
                {"role": "assistant", "tool_calls": [{"id": "old", "function":
                 {"name": "query", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "old", "content": "{}"},
                {"role": "user", "content": "continue"}]
    tools = [{"type": "function", "function": {"name": "query", "parameters": {"type": "object"}}}]
    assert provider.complete(messages, tools)["tool_calls"][0]["id"] == "next"
    payload = requests[0]
    assert payload["system"] == "rules" and "temperature" not in payload
    assert payload["messages"][1]["content"][0]["id"] == "old"
    assert payload["messages"][2]["content"][0]["tool_use_id"] == "old"
    assert len(payload["messages"][2]["content"]) == 2
    assert provider.profile.api_key not in json.dumps(payload)


def test_connection_probe_never_sends_engineering_data(monkeypatch):
    """模型目录可达不能冒充工具检测通过，探针不能触碰工程 Session。"""
    from configured_provider import test_tools as probe
    captured = []
    class Fake:
        def complete(self, messages, tools):
            captured.append((messages, tools))
            return {"content": "connected"}
    monkeypatch.setattr("configured_provider.create_provider", lambda p: Fake())
    with pytest.raises(ValueError, match="未通过工具调用检测"):
        probe(profile())
    assert captured[0][1][0]["function"]["name"] == "connection_probe"
    assert profile().api_key not in json.dumps(captured)


def test_router_does_not_share_same_name_connections(monkeypatch):
    """实际复现旧路由按名称缓存，导致备用同名连接重复调用主地址。"""
    from model_router import ModelConfig, ModelRouter
    configs = [ModelConfig("same", "https://one.example.com", api_key="one"),
               ModelConfig("same", "https://two.example.com", api_key="two")]
    class Fake:
        def __init__(self, config):
            self.config = config
        def complete(self, messages, tools):
            if self.config is configs[0]:
                raise RuntimeError("failed one")
            return {"content": "two"}
    monkeypatch.setattr("model_router._OpenAICompatibleProvider", Fake)
    router = ModelRouter(configs)
    assert router.complete([], []) == {"content": "two"}
    assert len(router._providers) == 2


def test_protocol_fallback_uses_own_keys_and_redacts_failure():
    """跨协议备用不能共用密钥，失败日志也不能回显主密钥。"""
    from configured_provider import FallbackProvider
    calls = []
    class Fake:
        def __init__(self, key, fail):
            self.key, self.fail = key, fail
        def complete(self, messages, tools):
            calls.append(self.key)
            if self.fail:
                raise RuntimeError(f"401 invalid {self.key}")
            return {"content": "backup"}
    router = FallbackProvider([("main", lambda: Fake("secret-main", True)),
                               ("backup", lambda: Fake("secret-backup", False))],
                              ["secret-main", "secret-backup"])
    assert router.complete([], []) == {"content": "backup"}
    assert calls == ["secret-main", "secret-backup"]
    assert router.call_history[-1].was_fallback
    assert "secret-main" not in router.call_history[0].error


def test_native_anthropic_rejects_truncated_tool_calls(monkeypatch):
    """输出被截断时不得执行可能缺参数的工具。"""
    from configured_provider import AnthropicProvider
    provider = AnthropicProvider(profile(protocol="anthropic"))
    monkeypatch.setattr(provider, "_request", lambda *args: {"stop_reason": "max_tokens",
        "content": [{"type": "tool_use", "id": "partial", "name": "query", "input": {}}]})
    with pytest.raises(RuntimeError, match="输出上限"):
        provider.complete([{"role": "user", "content": "query"}], [])


def test_router_final_failure_does_not_expose_key(monkeypatch):
    """不只日志，最终传播到 CLI 或界面的异常也必须隐藏密钥。"""
    from model_router import ModelConfig, ModelRouter
    key = "sk-final-error-secret"
    config = ModelConfig("model", "https://api.example.com", api_key=key)
    class Fail:
        def complete(self, messages, tools):
            raise RuntimeError(f"401 invalid {key}")
    router = ModelRouter([config])
    router._providers[config.identity()] = Fail()
    with pytest.raises(RuntimeError) as failure:
        router.complete([], [])
    assert key not in str(failure.value) and "401" in str(failure.value)
