"""多模型路由 + API 失败自动降级。

**为什么需要这个**：单一 LLM 提供商不可用时（429 限流、500 宕机、密钥过期），
整个 Agent 就瘫了。多模型路由按优先级依次尝试，主模型失败自动降级到备用模型，
保证"只要有一个模型能用，Agent 就能跑"。

**设计原则**：
1. 对调用方透明——ModelRouter 实现了和 DeepSeekProvider 一样的 `complete(messages, tools)` 接口，
   可以直接传给 `run_turn()`，不需要改主循环。
2. 降级是自动的——主模型抛任何异常（网络错误、限流、鉴权失败）都自动试下一个，
   全部失败才抛最后一个错误。
3. 可观测——每次调用记录用了哪个模型、耗时、是否降级、错误原因，
   调用方可以从 `call_history` 里读出来展示给用户。
4. 配置从环境变量加载——支持 `MODEL_ROUTER_CONFIG` 环境变量（JSON 字符串），
   没有配置时退化为单模型（DeepSeek），保持向后兼容。

**配置格式**（环境变量 MODEL_ROUTER_CONFIG 的 JSON）：
```json
{
  "models": [
    {"name": "deepseek-v4-flash", "base_url": "https://api.deepseek.com",
     "api_key_env": "DEEPSEEK_API_KEY", "priority": 1},
    {"name": "gpt-4o-mini", "base_url": "https://api.openai.com/v1",
     "api_key_env": "OPENAI_API_KEY", "priority": 2},
    {"name": "claude-3-5-sonnet", "base_url": "https://api.anthropic.com/v1",
     "api_key_env": "ANTHROPIC_API_KEY", "priority": 3}
  ]
}
```
priority 数字越小优先级越高。api_key 也可以直接写在配置里（`api_key` 字段），
但推荐用 `api_key_env` 引用环境变量，避免密钥写进配置文件。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class Provider(Protocol):
    """LLM 提供商的协议——只要有 complete(messages, tools) 方法就算。"""
    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...


@dataclass
class ModelConfig:
    """一个模型的配置。"""
    name: str
    base_url: str
    api_key: str = ""
    api_key_env: str = ""       # 优先从环境变量读，读不到再用 api_key 字段
    priority: int = 1
    temperature: float = 0.0

    def resolved_key(self) -> str | None:
        """解析出实际的 API 密钥。环境变量优先，其次配置字段。"""
        if self.api_key_env:
            from_env = os.environ.get(self.api_key_env, "").strip()
            if from_env:
                return from_env
        return self.api_key.strip() or None


@dataclass
class CallRecord:
    """一次调用的记录，用于可观测性。"""
    model: str
    success: bool
    duration_ms: float
    error: str = ""
    was_fallback: bool = False    # 是否是降级后才成功的


class _OpenAICompatibleProvider:
    """OpenAI 兼容接口的通用 provider——DeepSeek、OpenAI、任何 OpenAI 兼容 API 都能用。

    和 DeepSeekProvider 一样的接口，但模型名和 base_url 可配置。
    """

    def __init__(self, config: ModelConfig):
        from openai import OpenAI          # 延迟导入
        api_key = config.resolved_key()
        if not api_key:
            raise ValueError(f"模型 {config.name} 缺少 API 密钥"
                             f"（环境变量 {config.api_key_env or '未设置'} 为空）")
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)
        self._model = config.name
        self._temperature = config.temperature
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(model={self._model!r})"

    def set_temperature(self, value: float) -> None:
        self._temperature = min(1.0, max(0.0, float(value)))

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, tools=tools,
            temperature=self._temperature,
        )
        u = getattr(response, "usage", None)
        if u is not None:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0

        message = response.choices[0].message
        if message.tool_calls:
            calls = []
            for c in message.tool_calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"__invalid_json__": c.function.arguments}
                calls.append({"id": c.id, "name": c.function.name, "arguments": args})
            return {"tool_calls": calls}
        return {"content": message.content or ""}


class ModelRouter:
    """多模型路由 + 自动降级。

    实现了 Provider 协议，可以直接传给 run_turn()。
    按 priority 从小到大依次尝试，第一个成功的返回结果；全部失败抛最后一个错误。

    用法：
        router = ModelRouter.from_env()          # 从环境变量加载配置
        result = run_turn("建个三跨两层框架", router)

        # 查看调用历史（用了哪个模型、是否降级）
        for record in router.call_history:
            print(record.model, record.success, record.was_fallback)
    """

    def __init__(self, configs: list[ModelConfig]):
        if not configs:
            raise ValueError("ModelRouter 至少需要一个模型配置")
        # 按 priority 排序，数字越小优先级越高
        self._configs = sorted(configs, key=lambda c: c.priority)
        self._providers: dict[str, _OpenAICompatibleProvider] = {}
        self.call_history: list[CallRecord] = []
        self._fallback_count = 0      # 累计降级次数

    @classmethod
    def from_env(cls) -> "ModelRouter":
        """从环境变量加载配置。

        优先读 MODEL_ROUTER_CONFIG（JSON 字符串）；没有则退化为单模型 DeepSeek，
        保持向后兼容。
        """
        config_json = os.environ.get("MODEL_ROUTER_CONFIG", "").strip()
        if config_json:
            try:
                data = json.loads(config_json)
                configs = [ModelConfig(**m) for m in data.get("models", [])]
                if configs:
                    return cls(configs)
            except (json.JSONDecodeError, TypeError) as e:
                # 配置解析失败，退化为单模型，不崩溃
                print(f"[ModelRouter] MODEL_ROUTER_CONFIG 解析失败，退化为单模型：{e}")

        # 退化：单模型 DeepSeek（和原来的行为一致）
        return cls([ModelConfig(
            name="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            priority=1,
        )])

    def _get_provider(self, config: ModelConfig) -> _OpenAICompatibleProvider:
        """懒加载 provider——第一次用到才创建，避免初始化时就因为缺密钥报错。"""
        if config.name not in self._providers:
            self._providers[config.name] = _OpenAICompatibleProvider(config)
        return self._providers[config.name]

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        """按优先级依次调用，第一个成功的返回结果。

        全部失败时抛出最后一个模型的异常。每次调用都记录到 call_history。
        """
        last_error: Exception | None = None
        for i, config in enumerate(self._configs):
            try:
                provider = self._get_provider(config)
                start = time.monotonic()
                result = provider.complete(messages, tools)
                duration = (time.monotonic() - start) * 1000
                self.call_history.append(CallRecord(
                    model=config.name, success=True, duration_ms=duration,
                    was_fallback=(i > 0),
                ))
                if i > 0:
                    self._fallback_count += 1
                return result
            except Exception as e:
                duration = 0.0
                self.call_history.append(CallRecord(
                    model=config.name, success=False, duration_ms=duration,
                    error=f"{type(e).__name__}: {str(e)[:200]}",
                    was_fallback=(i > 0),
                ))
                last_error = e
                continue

        # 全部失败，抛出最后一个错误
        if last_error:
            raise last_error
        raise RuntimeError("ModelRouter: 所有模型都失败了，但没有记录到错误（不应发生）")

    @property
    def models(self) -> list[str]:
        """配置的模型名列表（按优先级排序）。"""
        return [c.name for c in self._configs]

    @property
    def total_calls(self) -> int:
        """累计调用次数（包括失败的）。"""
        return len(self.call_history)

    @property
    def fallback_count(self) -> int:
        """累计降级次数（降级后成功的调用数）。"""
        return self._fallback_count

    def set_temperature(self, value: float) -> None:
        """同步更新尚未创建和已经创建的全部后端。"""
        temperature = min(1.0, max(0.0, float(value)))
        for config in self._configs:
            config.temperature = temperature
        for provider in self._providers.values():
            provider.set_temperature(temperature)

    def last_call_summary(self) -> str:
        """最近一次调用的摘要，用于界面展示。"""
        if not self.call_history:
            return "尚未调用"
        records = self.call_history[-len(self._configs):]  # 最近一轮的所有尝试
        successful = [r for r in records if r.success]
        if successful:
            r = successful[0]
            fallback = f"（从 {records[0].model} 降级）" if r.was_fallback else ""
            return f"模型 {r.model} 响应，耗时 {r.duration_ms:.0f}ms{fallback}"
        return f"全部模型失败：{'; '.join(r.error for r in records if r.error)}"
