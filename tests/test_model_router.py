"""多模型路由（model_router）的单元测试。

ModelRouter 的核心价值是"主模型失败时自动降级"。这里用 mock provider 模拟
各种失败场景，验证降级逻辑、调用历史、错误传播都正确。

测六件事：
1. 单模型配置能正常创建并调用
2. 主模型成功时不触发降级，直接返回结果
3. 主模型失败、备用模型成功时自动降级
4. 全部模型失败时抛出最后一个错误
5. call_history 正确记录每次调用的模型、成功状态、是否降级
6. from_env 能从环境变量加载多模型配置，没有配置时退化为单模型
"""

from __future__ import annotations

import json

import pytest

from model_router import ModelConfig, ModelRouter, CallRecord


# --------------------------------------------------------- mock provider

class _MockProvider:
    """模拟 LLM provider，可以配置成功或失败。"""
    def __init__(self, name: str, should_fail: bool = False, error_msg: str = "mock error"):
        self.name = name
        self.should_fail = should_fail
        self.error_msg = error_msg
        self.called = False

    def complete(self, messages, tools):
        self.called = True
        if self.should_fail:
            raise RuntimeError(self.error_msg)
        return {"content": f"response from {self.name}"}


def _make_router_with_mocks(configs: list[tuple[str, bool]]) -> tuple[ModelRouter, list[_MockProvider]]:
    """创建一个用 mock provider 的 ModelRouter。

    configs 是 [(模型名, 是否失败), ...]，按优先级排序。
    返回 (router, providers)，providers 可以用来断言调用情况。
    """
    model_configs = [ModelConfig(name=n, base_url="http://mock", api_key="mock", priority=i)
                     for i, (n, _) in enumerate(configs)]
    router = ModelRouter(model_configs)
    providers = [_MockProvider(n, fail) for n, fail in configs]
    # 替换掉真实的 provider 懒加载，直接用 mock
    router._providers = {p.name: p for p in providers}  # type: ignore
    return router, providers


# --------------------------------------------------------- 基础功能

def test_single_model_success():
    router, providers = _make_router_with_mocks([("model-a", False)])
    result = router.complete([], [])
    assert result == {"content": "response from model-a"}
    assert providers[0].called
    assert router.total_calls == 1
    assert router.fallback_count == 0
    assert router.call_history[0].success is True
    assert router.call_history[0].was_fallback is False


def test_primary_success_no_fallback():
    router, providers = _make_router_with_mocks([
        ("primary", False), ("backup", False)])
    result = router.complete([], [])
    assert result["content"] == "response from primary"
    assert providers[0].called
    assert not providers[1].called, "主模型成功时不应调用备用模型"
    assert router.fallback_count == 0


# --------------------------------------------------------- 降级逻辑

def test_primary_fails_backup_succeeds():
    router, providers = _make_router_with_mocks([
        ("primary", True), ("backup", False)])
    result = router.complete([], [])
    assert result["content"] == "response from backup"
    assert providers[0].called, "主模型应该被调用（然后失败）"
    assert providers[1].called, "备用模型应该被调用（降级）"
    assert router.fallback_count == 1
    # 调用历史应该有两条：主模型失败 + 备用模型成功
    assert len(router.call_history) == 2
    assert router.call_history[0].success is False
    assert "mock error" in router.call_history[0].error
    assert router.call_history[1].success is True
    assert router.call_history[1].was_fallback is True


def test_multiple_fallbacks():
    """三个模型，前两个失败，第三个成功。"""
    router, providers = _make_router_with_mocks([
        ("a", True), ("b", True), ("c", False)])
    result = router.complete([], [])
    assert result["content"] == "response from c"
    assert all(p.called for p in providers)
    assert router.fallback_count == 1
    assert len(router.call_history) == 3


def test_all_models_fail_raises_last_error():
    router, _ = _make_router_with_mocks([
        ("a", True), ("b", True)])
    # 把第二个的错误信息改一下，确认抛的是最后一个
    router._providers["b"].error_msg = "second error"  # type: ignore
    with pytest.raises(RuntimeError, match="second error"):
        router.complete([], [])
    assert len(router.call_history) == 2
    assert all(not r.success for r in router.call_history)


# --------------------------------------------------------- 调用历史与摘要

def test_call_history_records_model_name():
    router, _ = _make_router_with_mocks([("my-model", False)])
    router.complete([], [])
    assert router.call_history[0].model == "my-model"


def test_last_call_summary_success():
    router, _ = _make_router_with_mocks([("model-x", False)])
    router.complete([], [])
    summary = router.last_call_summary()
    assert "model-x" in summary
    assert "响应" in summary


def test_last_call_summary_with_fallback():
    router, _ = _make_router_with_mocks([
        ("primary", True), ("backup", False)])
    router.complete([], [])
    summary = router.last_call_summary()
    assert "backup" in summary
    assert "降级" in summary


def test_last_call_summary_all_failed():
    router, _ = _make_router_with_mocks([("a", True), ("b", True)])
    with pytest.raises(RuntimeError):
        router.complete([], [])
    summary = router.last_call_summary()
    assert "失败" in summary


def test_models_property_lists_all():
    router, _ = _make_router_with_mocks([
        ("a", False), ("b", False), ("c", False)])
    assert router.models == ["a", "b", "c"]


# --------------------------------------------------------- 配置加载

def test_from_env_no_config_falls_back_to_deepseek():
    import os
    old = os.environ.pop("MODEL_ROUTER_CONFIG", None)
    try:
        router = ModelRouter.from_env()
        assert router.models == ["deepseek-v4-flash"]
    finally:
        if old is not None:
            os.environ["MODEL_ROUTER_CONFIG"] = old


def test_from_env_with_json_config():
    import os
    config = {
        "models": [
            {"name": "model-1", "base_url": "http://a", "api_key_env": "KEY1", "priority": 2},
            {"name": "model-2", "base_url": "http://b", "api_key": "direct-key", "priority": 1},
        ]
    }
    old = os.environ.get("MODEL_ROUTER_CONFIG")
    os.environ["MODEL_ROUTER_CONFIG"] = json.dumps(config)
    try:
        router = ModelRouter.from_env()
        # priority 1 的 model-2 应该排前面
        assert router.models == ["model-2", "model-1"]
    finally:
        if old is not None:
            os.environ["MODEL_ROUTER_CONFIG"] = old
        else:
            os.environ.pop("MODEL_ROUTER_CONFIG", None)


def test_model_config_resolved_key_from_env():
    import os
    os.environ["TEST_API_KEY"] = "secret-123"
    config = ModelConfig(name="test", base_url="http://x", api_key_env="TEST_API_KEY")
    assert config.resolved_key() == "secret-123"
    del os.environ["TEST_API_KEY"]


def test_model_config_resolved_key_direct():
    config = ModelConfig(name="test", base_url="http://x", api_key="direct-key")
    assert config.resolved_key() == "direct-key"


def test_model_config_resolved_key_none():
    config = ModelConfig(name="test", base_url="http://x")
    assert config.resolved_key() is None


def test_empty_configs_raises():
    with pytest.raises(ValueError, match="至少需要一个模型配置"):
        ModelRouter([])
