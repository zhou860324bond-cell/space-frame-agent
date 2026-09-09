"""手绘草图 → 结构模型（多模态输入）。

**为什么需要这个**：结构工程师的工作流通常是"先在纸上画草图，再输入软件"。
多模态输入让用户直接上传手绘草图，LLM 识别节点位置、杆件连接、支座和荷载，
先转换成待人工核对的 RecognitionDraft，再进入项目的模型编辑链路。

**设计原则**：
1. LLM 只产结构，不产数值——和项目核心原则一致。LLM 负责识别"哪有节点、
   哪有杆、哪是支座、荷载在哪"，位移/内力/反力仍由确定性求解器计算。
2. 可校验、可修正——LLM 先产出允许缺属性的 RecognitionDraft；坏拓扑和坏引用
   会回给 LLM 修正，尺度与工程语义交给用户确认。
3. 多提供商支持——OpenAI GPT-4V、Anthropic Claude 3、DeepSeek VL 等，
   通过统一的接口调用。
4. 可离线降级——没有多模态 LLM API 时，提示用户手动输入或用参数化建模，
   不崩溃。

**支持的草图元素**：
- 节点：圆点、交点
- 杆件：直线、折线
- 支座：固定端（斜线填充）、铰支座（三角形）、滚动支座（三角形+圆）
- 荷载：箭头（集中力）、均布荷载（虚线+箭头）
- 尺寸标注：数字+尺寸线（用于确定节点坐标比例）

**用法**：
```python
from sketch_parser import SketchParser

parser = SketchParser(
    provider="openai",           # openai / anthropic / deepseek
    api_key="sk-...",
    model="gpt-4o",
)
draft = parser.parse_draft("sketch.jpg")  # 返回待人工确认的识别草稿
# 或带自修复：
result = parser.parse_with_retry("sketch.jpg", max_retries=3)
draft = result.draft
```
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from recognition_draft import DRAFT_FORMAT, RecognitionDraft
from multimodal_workflow import (V2_DRAFT_FORMAT, MultimodalControllerState,
                                 validate_v2_draft)


IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp",
}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


# 指导 LLM 识别结构草图的系统提示词
SKETCH_SYSTEM_PROMPT = f"""你是结构工程草图识别助手。输出识别草稿，不输出最终求解模型。

硬性规则：
1. 只识别图片中有证据的几何、支座、荷载和尺寸，不计算结构响应。
2. 不得补默认材料、默认截面、默认支座、默认荷载或默认尺寸。
3. 图片没有可靠尺寸时，scale.status 必须为 "unknown"；节点坐标只保持相对比例，
   并在 questions 中要求用户标定。绝不能把相对坐标假装成米。
4. 支座或荷载符号不清楚时不要猜，把疑问写入 questions。
5. 只有图片明确写出荷载数值和单位时才写入 load_cases；否则只记录实体和疑问。
6. 平面结构 y=0。members 的 section 和 material 固定为空字符串，稍后由用户指派。

严格输出一个 JSON 对象，格式如下，不要附加解释或 Markdown：
{{
  "format": "{DRAFT_FORMAT}",
  "scale": {{"status": "confirmed 或 unknown", "evidence": "尺寸文字或未知原因"}},
  "model": {{
    "units": "N-m-Pa",
    "materials": [],
    "sections": [],
    "nodes": [{{"id": 1, "x": 0, "y": 0, "z": 0}}],
    "members": [{{"id": 1, "i": 1, "j": 2, "section": "", "material": ""}}],
    "supports": [],
    "load_cases": []
  }},
  "entities": [
    {{"kind": "node", "id": 1, "confidence": 0.95,
      "image_geometry": {{"point": [0.1, 0.8]}}}},
    {{"kind": "member", "id": 1, "confidence": 0.9,
      "image_geometry": {{"line": [[0.1, 0.8], [0.1, 0.2]]}}}},
    {{"kind": "support", "id": "S1", "confidence": 0.9,
      "target": {{"node": 1}}, "image_geometry": {{"bbox": [0.05, 0.78, 0.15, 0.9]}}}},
    {{"kind": "load", "id": "Wind:P1", "confidence": 0.8,
      "target": {{"case": "Wind", "node": 2, "name": "P1"}},
      "image_geometry": {{"bbox": [0.4, 0.1, 0.5, 0.25]}}}}
  ],
  "questions": ["需要用户确认的问题"],
  "warnings": ["低置信度或遮挡说明"]
}}

image_geometry 使用 0 到 1 的归一化图片坐标，原点在图片左上角。节点使用
{{"point":[x,y]}}，杆件使用 {{"line":[[x1,y1],[x2,y2]]}}，支座和荷载使用
{{"bbox":[x1,y1,x2,y2]}}。
固定、铰支和滚动支座只有在符号清晰时才写入 model.supports；识别实体仍需写入
entities，kind 使用 support 或 load。support 的 target.node 必须指向对应节点；明确数值的
节点荷载使用 target.case/node/name 对应 load_cases 中的条目。节点和杆件 id 必须唯一，
杆件只能引用已识别节点。
"""

V2_SKETCH_SYSTEM_PROMPT = f"""你是结构工程草图识别助手。只输出一个 JSON 对象，
format 必须为 {V2_DRAFT_FORMAT}。输出必须符合多模态草稿 v2：包含 image_model、
model、merge_plan、source、work_plane、scale、entities、dimensions、intersections、
issues、revision、confirmation。视觉阶段只填写有图片证据的 image_model 与实体；
model 和 merge_plan 必须为 null，revision 为 0，confirmation 为 null。
不得补材料、截面、支座、荷载、尺寸或工程响应。未知尺度保持 unknown；工作平面只可
proposed。source.image_hash 必须使用调用方给出的图片哈希。只输出 JSON，不要 Markdown。
"""


@dataclass
class ParseResult:
    """草图解析结果。"""
    model: dict[str, Any] | None = None
    draft: RecognitionDraft | None = None
    success: bool = False
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    raw_response: str = ""
    duration_ms: float = 0.0


@dataclass
class V2ParseResult:
    """Result of one state-machine-bound v2 recognition job."""
    draft: dict[str, Any] | None = None
    success: bool = False
    cancelled: bool = False
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class SketchParser:
    """手绘草图 → 结构模型的多模态解析器。

    支持 OpenAI GPT-4V、Anthropic Claude 3、DeepSeek VL 等多模态 LLM。
    识别结果先进入 RecognitionDraft；坏拓扑自动重试，未知尺度交给用户标定。
    """

    SUPPORTED_PROVIDERS = ("openai", "anthropic", "deepseek")

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        temperature: float = 0.0,
        system_prompt: str = SKETCH_SYSTEM_PROMPT,
    ):
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"不支持的提供商 {provider!r}，支持: {self.SUPPORTED_PROVIDERS}")
        self._provider = provider
        self._api_key = api_key
        self._model = model or self._default_model(provider)
        self._base_url = base_url or self._default_base_url(provider)
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._client = None

    @staticmethod
    def _default_model(provider: str) -> str:
        return {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "deepseek": "deepseek-vl",
        }.get(provider, "gpt-4o")

    @staticmethod
    def _default_base_url(provider: str) -> str:
        return {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
            "deepseek": "https://api.deepseek.com",
        }.get(provider, "https://api.openai.com/v1")

    def _get_client(self):
        """懒加载 LLM 客户端。"""
        if self._client is None:
            if not self._api_key:
                raise ValueError("缺少 API 密钥。请设置 api_key 参数或环境变量。")
            if self._provider == "openai" or self._provider == "deepseek":
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError(
                        "缺少 openai 依赖。请执行 pip install -r requirements-multimodal.txt"
                    ) from exc
                self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            elif self._provider == "anthropic":
                try:
                    import anthropic
                except ImportError as exc:
                    raise RuntimeError(
                        "缺少 anthropic 依赖。请执行 pip install -r requirements-multimodal.txt"
                    ) from exc
                self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _encode_image(image_path: str | Path) -> str:
        """把图片编码成 base64。"""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在: {path}")
        suffix = path.suffix.lower()
        mime = IMAGE_MIME_TYPES.get(suffix)
        if mime is None:
            supported = "、".join(sorted(IMAGE_MIME_TYPES))
            raise ValueError(f"不支持的图片格式 {suffix or '（无扩展名）'}，支持：{supported}")
        raw = path.read_bytes()
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"图片超过 {MAX_IMAGE_BYTES // 1024 // 1024} MB，请压缩后重试")
        data = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{data}"

    @staticmethod
    def _anthropic_image(image_data_url: str) -> dict[str, Any]:
        """把 data URL 转为 Anthropic Messages API 所需的图片块。"""
        try:
            header, data = image_data_url.split(",", 1)
            prefix, encoding = header.split(";", 1)
        except ValueError as exc:
            raise ValueError("图片数据格式无效") from exc
        if not prefix.startswith("data:") or encoding != "base64":
            raise ValueError("图片数据必须是 base64 data URL")
        return {"type": "image", "source": {
            "type": "base64", "media_type": prefix[5:], "data": data}}

    def _call_llm(self, image_data_url: str, correction: str = "", *,
                  system_prompt: str | None = None) -> str:
        """调用多模态 LLM，返回原始文本响应。"""
        client = self._get_client()
        user_content = [
            {"type": "text", "text": "请识别这张结构草图，输出 JSON 格式的识别草稿。"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        if correction:
            user_content.insert(1, {"type": "text", "text":
                f"上一次的输出有以下错误，请修正后重新输出：\n{correction}\n"
                f"只输出修正后的 JSON，不要输出其他文字。"})

        if self._provider in ("openai", "deepseek"):
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt or self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=self._temperature,
                max_tokens=4000,
            )
            return response.choices[0].message.content or ""
        elif self._provider == "anthropic":
            content: list[dict[str, Any]] = [
                {"type": "text", "text": "请识别这张结构草图，输出 JSON 格式的识别草稿。"},
            ]
            if correction:
                content.append({"type": "text", "text":
                    f"上一次的输出有以下错误，请修正后重新输出：\n{correction}\n"
                    "只输出修正后的 JSON，不要输出其他文字。"})
            content.append(self._anthropic_image(image_data_url))
            response = client.messages.create(
                model=self._model,
                max_tokens=4000,
                system=system_prompt or self._system_prompt,
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text if response.content else ""
        return ""

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """从 LLM 响应中提取 JSON。LLM 可能输出 markdown 代码块或前后有文字。"""
        text = text.strip()
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 代码块
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return json.loads(text[start:end].strip())
        # 尝试提取第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("无法从响应中提取 JSON")

    @staticmethod
    def _validate_model(model: dict[str, Any]) -> list[str]:
        """兼容入口：校验旧完整模型或新版识别草稿的几何拓扑。"""
        try:
            RecognitionDraft.from_payload(model)
            return []
        except (TypeError, ValueError) as exc:
            return [str(exc)]

    @staticmethod
    def _recognition_draft(payload: dict[str, Any],
                           image_path: str | Path) -> RecognitionDraft:
        return RecognitionDraft.from_payload(payload, source_image=image_path)

    def parse(self, image_path: str | Path) -> dict[str, Any]:
        """兼容旧调用；仅在图片已提供可靠尺度时返回模型字典。"""
        draft = self.parse_draft(image_path)
        if not draft.ready_to_load:
            raise ValueError("尺度尚未确认，请改用 parse_draft() 标定后再加载")
        return draft.model

    def parse_draft(self, image_path: str | Path) -> RecognitionDraft:
        """解析草图为待人工确认的识别草稿，不直接写入 Session。

        Raises:
            ValueError: LLM 响应无法解析或草稿拓扑无效。
        """
        image_data = self._encode_image(image_path)
        raw = self._call_llm(image_data)
        payload = self._extract_json(raw)
        return self._recognition_draft(payload, image_path)

    def parse_with_retry(
        self, image_path: str | Path, max_retries: int = 3
    ) -> ParseResult:
        """解析草图，带自修复闭环。校验失败时把错误回给 LLM 修正，最多重试 max_retries 次。"""
        start = time.monotonic()
        result = ParseResult()
        image_data = self._encode_image(image_path)
        correction = ""

        for attempt in range(1, max_retries + 1):
            result.attempts = attempt
            try:
                raw = self._call_llm(image_data, correction)
                result.raw_response = raw
                payload = self._extract_json(raw)
                draft = self._recognition_draft(payload, image_path)
                result.draft = draft
                result.model = draft.model
                result.success = True
                result.duration_ms = (time.monotonic() - start) * 1000
                return result
            except (json.JSONDecodeError, ValueError) as e:
                result.errors.append(f"第{attempt}次解析失败: {e}")
                correction = f"输出格式错误：{e}。请严格输出 JSON，不要包含其他文字。"
            except Exception as e:
                result.errors.append(f"第{attempt}次调用失败: {type(e).__name__}: {e}")
                break  # API 调用失败（网络/密钥），重试没用

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    def parse_v2_with_retry(
        self, image_path: str | Path, state: MultimodalControllerState,
        job_id: str, *, max_repairs: int = 2,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> V2ParseResult:
        """Recognize a v2 draft and atomically deliver it to the controller.

        A malformed response may be repaired at most ``max_repairs`` times. API
        failures stop immediately. Cancellation cannot abort a provider socket,
        but its response is discarded and can never enter the controller.
        """
        if max_repairs < 0 or max_repairs > 2:
            raise ValueError("max_repairs 必须在 0 到 2 之间")
        if state.job_status != "running" or state.job_id != job_id:
            raise RuntimeError("job_id 不是当前识别任务")
        start = time.monotonic()
        result = V2ParseResult()
        image_data = self._encode_image(image_path)
        correction = f"当前派生图片的 SHA-256 是 {state.image_hash}。"

        def cancelled() -> bool:
            return bool((is_cancelled and is_cancelled())
                        or state.job_status != "running" or state.job_id != job_id)

        for attempt in range(1, max_repairs + 2):
            result.attempts = attempt
            if cancelled():
                state.cancel_recognition(job_id)
                result.cancelled = True
                break
            try:
                raw = self._call_llm(
                    image_data, correction, system_prompt=V2_SKETCH_SYSTEM_PROMPT)
                result.raw_responses.append(raw)
                if cancelled():
                    state.cancel_recognition(job_id)
                    result.cancelled = True
                    break
                payload = self._extract_json(raw)
                errors = validate_v2_draft(payload)
                source = payload.get("source") if isinstance(payload, dict) else None
                if (source or {}).get("image_hash") != state.image_hash:
                    errors.append("source.image_hash 与当前派生图不一致")
                if errors:
                    raise ValueError("；".join(errors))
                if not state.complete_recognition(job_id, payload):
                    result.cancelled = True
                    break
                result.draft = payload
                result.success = True
                break
            except (json.JSONDecodeError, ValueError) as exc:
                message = f"第{attempt}次 v2 校验失败: {exc}"
                result.errors.append(message)
                correction = (f"当前派生图片的 SHA-256 是 {state.image_hash}。"
                              f"上次输出错误：{exc}。严格修复为 v2 JSON。")
            except Exception as exc:  # noqa: BLE001 - provider/network boundary
                result.errors.append(
                    f"第{attempt}次 API 调用失败: {type(exc).__name__}: {exc}")
                break

        if not result.success and not result.cancelled \
                and state.job_status == "running" and state.job_id == job_id:
            state.fail_recognition(job_id, result.errors[-1] if result.errors
                                   else "识别失败")
        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    @classmethod
    def from_env(cls, provider: str = "openai", model: str = "",
                 base_url: str = "") -> "SketchParser":
        """从环境变量创建解析器。

        OpenAI: OPENAI_API_KEY
        Anthropic: ANTHROPIC_API_KEY
        DeepSeek: DEEPSEEK_API_KEY
        """
        env_var = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }.get(provider, "OPENAI_API_KEY")
        import os
        api_key = os.environ.get(env_var, "")
        return cls(provider=provider, api_key=api_key, model=model,
                   base_url=base_url)
