"""流式读取模型回复，边生成边把文字交给界面。

非流式调用要等整段回答生成完才返回：一轮十几秒里对话框一动不动，用户分不清
是在思考还是卡死了。流式只改变**文字到达界面的时机**，返回值与
``Provider.complete`` 完全相同——工具调用仍要等参数完整才执行，
模型产出的数值也不会因此绕过求解器。

两个协议：
- OpenAI Chat Completions（DeepSeek、OpenAI 及兼容接口）：``stream=True``，
  工具调用参数按 ``index`` 分片到达，需要拼接后再解析；
- Anthropic Messages：SSE 事件流，文本与 ``input_json_delta`` 分块到达。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable

TextSink = Callable[[str], None]


def _parse_arguments(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        # 与非流式实现一致：保留原文交给工具层报错，不在这里猜
        return {"__invalid_json__": raw}


def collect_openai_chunks(chunks: Iterable[Any], on_text: TextSink | None,
                          usage: dict | None = None) -> dict:
    """把 Chat Completions 的流式分片还原成 ``complete`` 的返回格式。"""
    text: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        u = getattr(chunk, "usage", None)
        if u is not None and usage is not None:
            usage["calls"] = usage.get("calls", 0) + 1
            usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + (getattr(u, "prompt_tokens", 0) or 0)
            usage["completion_tokens"] = (usage.get("completion_tokens", 0)
                                          + (getattr(u, "completion_tokens", 0) or 0))
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            text.append(piece)
            if on_text is not None:
                on_text(piece)
        for call in getattr(delta, "tool_calls", None) or []:
            slot = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
            if call.id:
                slot["id"] = call.id
            function = getattr(call, "function", None)
            if function is not None:
                if function.name:
                    slot["name"] += function.name
                if function.arguments:
                    slot["arguments"] += function.arguments
    if calls:
        return {"tool_calls": [
            {"id": slot["id"] or f"call_{index}", "name": slot["name"],
             "arguments": _parse_arguments(slot["arguments"])}
            for index, slot in sorted(calls.items())],
            "content": "".join(text)}
    return {"content": "".join(text)}


def openai_stream(client, request: dict, on_text: TextSink | None,
                  usage: dict | None = None) -> dict:
    """发起流式 Chat Completions 请求。``request`` 与非流式调用的参数相同。"""
    stream = client.chat.completions.create(stream=True, **request)
    try:
        return collect_openai_chunks(stream, on_text, usage)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


def collect_anthropic_events(lines: Iterable[bytes | str],
                             on_text: TextSink | None) -> dict:
    """解析 Anthropic Messages 的 SSE 行，返回 ``complete`` 的格式。"""
    text: list[str] = []
    blocks: dict[int, dict[str, Any]] = {}
    stop_reason = None
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        kind = event.get("type")
        if kind == "content_block_start":
            block = event.get("content_block") or {}
            blocks[event["index"]] = {"type": block.get("type"), "id": block.get("id", ""),
                                      "name": block.get("name", ""), "json": ""}
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                piece = delta.get("text", "")
                text.append(piece)
                if on_text is not None and piece:
                    on_text(piece)
            elif delta.get("type") == "input_json_delta":
                blocks.setdefault(event["index"], {"type": "tool_use", "id": "", "name": "", "json": ""})
                blocks[event["index"]]["json"] += delta.get("partial_json", "")
        elif kind == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason", stop_reason)
        elif kind == "error":
            message = (event.get("error") or {}).get("message", "未知错误")
            raise RuntimeError(f"模型流式回复中断：{message}")
    if stop_reason == "max_tokens":
        raise RuntimeError("模型达到输出上限，请增加输出上限或缩短请求后重试")
    calls = [{"id": b["id"], "name": b["name"], "arguments": _parse_arguments(b["json"])}
             for _, b in sorted(blocks.items()) if b.get("type") == "tool_use"]
    if calls:
        return {"tool_calls": calls, "content": "".join(text)}
    return {"content": "".join(text)}
