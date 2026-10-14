"""将工程对话协议适配到真实提供商；连接检测不上传模型或对话。"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import time

from model_profiles import ModelProfile, redact
from model_router import CallRecord, ModelConfig, _OpenAICompatibleProvider


class FallbackProvider:
    """仅在模型请求失败时使用用户选定的独立连接，工具执行失败不重复派发。"""
    def __init__(self, connections, keys):
        self.connections = connections
        self.keys = keys
        self.providers = {}
        self.call_history = []

    def complete(self, messages, tools, on_text=None):
        for index, (label, factory) in enumerate(self.connections):
            start = time.monotonic()
            try:
                if index not in self.providers:
                    self.providers[index] = factory()
                provider = self.providers[index]
                if on_text is not None and hasattr(provider, "complete_stream"):
                    result = provider.complete_stream(messages, tools, on_text)
                else:
                    result = provider.complete(messages, tools)
                self.call_history.append(CallRecord(label, True, (time.monotonic() - start) * 1000,
                                                    was_fallback=index > 0))
                return result
            except Exception as exc:  # noqa: BLE001 只记录隐藏密钥后的错误
                error = f"{type(exc).__name__}: {redact(str(exc), self.keys)}"
                self.call_history.append(CallRecord(label, False, (time.monotonic() - start) * 1000,
                                                    error=error[:200], was_fallback=index > 0))
        raise RuntimeError(self.call_history[-1].error)

    def complete_stream(self, messages, tools, on_text):
        return self.complete(messages, tools, on_text)

    def close(self):
        for provider in self.providers.values():
            client = getattr(provider, "_client", None)
            if client is not None:
                client.close()


class AnthropicProvider:
    def __init__(self, profile: ModelProfile):
        profile.validate()
        self.profile = profile

    def _request(self, path: str, payload=None, on_lines=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.profile.base_url.rstrip("/") + path, data=data,
            headers={"x-api-key": self.profile.api_key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"})
        # 禁止自动跳转到其他主机，防止鉴权字段随重定向发送。
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=90) as response:
                return on_lines(response) if on_lines is not None else json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"API 请求被拒绝（HTTP {exc.code}），请检查密钥、模型与地址") from None

    def complete_stream(self, messages, tools, on_text):
        from llm_stream import collect_anthropic_events
        payload = dict(self._payload(messages, tools), stream=True)
        return self._request("/messages", payload,
                             on_lines=lambda response: collect_anthropic_events(response, on_text))

    def complete(self, messages, tools):
        result = self._request("/messages", self._payload(messages, tools))
        if result.get("stop_reason") == "max_tokens":
            raise RuntimeError("模型达到输出上限，请增加输出上限或缩短请求后重试")
        calls = [{"id": b["id"], "name": b["name"], "arguments": b["input"]}
                 for b in result["content"] if b["type"] == "tool_use"]
        if calls:
            return {"tool_calls": calls}
        return {"content": "\n".join(b["text"] for b in result["content"] if b["type"] == "text")}

    def _payload(self, messages, tools):
        system, converted = [], []
        for message in messages:
            role = message["role"]
            if role == "system":
                system.append(message["content"])
                continue
            content = []
            if role == "tool":
                role = "user"
                content.append({"type": "tool_result", "tool_use_id": message["tool_call_id"],
                                "content": message["content"]})
            else:
                if message.get("content"):
                    content.append({"type": "text", "text": message["content"]})
                for call in message.get("tool_calls", []):
                    content.append({"type": "tool_use", "id": call["id"],
                                    "name": call["function"]["name"],
                                    "input": json.loads(call["function"]["arguments"])})
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(content)
            else:
                converted.append({"role": role, "content": content})
        payload = {"model": self.profile.model, "max_tokens": self.profile.max_tokens,
                   "system": "\n\n".join(system), "messages": converted,
                   "tools": [{"name": t["function"]["name"],
                              "description": t["function"].get("description", ""),
                              "input_schema": t["function"]["parameters"]} for t in tools]}
        if self.profile.temperature is not None:
            payload["temperature"] = self.profile.temperature
        return payload


def create_provider(profile: ModelProfile):
    profile.validate()
    if profile.protocol == "anthropic":
        return AnthropicProvider(profile)
    return _OpenAICompatibleProvider(ModelConfig(
        name=profile.model, base_url=profile.base_url, api_key=profile.api_key,
        temperature=profile.temperature, max_tokens=profile.max_tokens))


def list_models(profile: ModelProfile) -> list[str]:
    """读取模型目录；目录可达不代表所选模型支持工程工具。"""
    provider = create_provider(profile)
    if isinstance(provider, AnthropicProvider):
        return [item["id"] for item in provider._request("/models?limit=1000")["data"]]
    try:
        return sorted(m.id for m in provider._client.models.list())
    finally:
        provider._client.close()


def test_tools(profile: ModelProfile):
    """只发送固定探针，不调用工程工具，不改变 Session。"""
    provider = create_provider(profile)
    tool = {"type": "function", "function": {"name": "connection_probe",
            "description": "Call this function to confirm tool support.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}
    try:
        result = provider.complete([{"role": "user", "content": "Call connection_probe with no arguments."}], [tool])
        if not any(c["name"] == "connection_probe" and c["arguments"] == {}
                   for c in result.get("tool_calls", [])):
            raise ValueError("目录可达，但所选模型未通过工具调用检测，请换模型或检查接口协议")
        return True
    finally:
        if hasattr(provider, "_client"):
            provider._client.close()
