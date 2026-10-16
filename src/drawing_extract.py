"""结构图纸图片 → 文字证据（多模态入口）。

基准测试的结论很一致：视觉模型抄写图上文字可靠（OCR 类题目约 0.95），数构件、
量距离不可靠。所以这里只让模型**抄字**，分两遍：

1. **定位**：整图送一遍，只要平面图、立面图、构件表、说明各自所在的区域。
2. **读取**：代码按原分辨率把这些区域裁出来再送一遍，逐块抄写轴号、间距、
   标高、构件编号、规格和说明。图纸上的字很小，整图缩小后送入最容易读错。

读完先查格式（``validate_evidence``），再试组装：格式错、尺寸链不闭合、引用了
不存在的轴号，就把错误连同同一批裁剪图回喂，让模型重读（最多 ``max_repairs``
次）。图上确实没有的信息保持缺失，交给核对表由用户补，不让模型编。

客户端、图片编码、超时沿用 ``sketch_parser.SketchParser``，提供商设置与草图
识别一致。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from drawing_assemble import assemble
from drawing_evidence import EVIDENCE_FORMAT, validate_evidence
from multimodal_contract import file_digest

#: 区域边缘外扩比例，防止框得太紧把轴号圈或尺寸数字切掉一半。
REGION_PADDING = 0.02
#: 送入模型的裁剪图最长边像素；超过就等比缩小（各家接口对单图尺寸有上限）。
MAX_CROP_SIDE = 2048
#: 区域占整图面积超过此比例时直接送整图，裁剪没有意义。
WHOLE_IMAGE_RATIO = 0.8

REGION_KINDS = ("plan", "elevation", "section_table", "notes", "other")

LAYOUT_PROMPT = """你是结构施工图版面分析助手。只输出一个 JSON 对象，不要 Markdown。

找出每张图里这几类区域的位置：
- plan：结构平面布置图（有轴网、梁柱编号）
- elevation：立面图或剖面图（有标高）
- section_table：构件表、截面表（编号与规格对照）
- notes：设计说明、图名、单位说明、柱脚做法说明
- other：其他

格式：
{"regions": [{"image": 1, "kind": "plan", "bbox": [0.05, 0.10, 0.70, 0.95]}]}

image 是图的序号（从 1 开始）；bbox 是 0~1 归一化坐标 [左, 上, 右, 下]，原点在左上角。
宁可框大一点，也不要切掉轴号圈、尺寸数字和表格边缘。不要读取内容。
"""

TRANSCRIBE_PROMPT = f"""你是结构施工图抄写助手。你的任务是**原样抄写图上的文字**，
不是计算，也不是估算。只输出一个 JSON 对象，不要 Markdown、不要解释。

下面会给出若干区域图，每张前面写着「区域 k」。所有 source.region 填区域序号，
source.bbox 填该文字在**这张区域图**里的 0~1 归一化坐标 [左, 上, 右, 下]，
source.confidence 填你对这条读数的把握（0~1）。

输出格式：
{{
  "format": "{EVIDENCE_FORMAT}",
  "units": {{"length": {{"value": "mm", "text": "本图尺寸以毫米为单位", "source": S}},
             "elevation": {{"value": "m", "text": "标高以米为单位", "source": S}}}},
  "grids": {{
    "x": {{"axes": [{{"label": "1", "pos": 0.12, "source": S}}],
           "spacings": [{{"from": "1", "to": "2", "text": "6000", "source": S}}],
           "total": {{"text": "12000", "source": S}}}},
    "y": {{"axes": [{{"label": "A", "pos": 0.85, "source": S}}], "spacings": [], "total": null}}
  }},
  "levels": [{{"name": "2F", "text": "3.600", "source": S}}],
  "members": [
    {{"label": "KL1", "kind": "beam", "along": "A", "from": "1", "to": "3",
      "levels": ["2F"], "source": S}},
    {{"label": "KZ1", "kind": "column", "at": ["1", "A"],
      "from_level": "1F", "to_level": "3F", "source": S}}
  ],
  "sections": [{{"label": "KL1", "text": "HN400×200×8×13", "source": S}}],
  "material": {{"text": "钢材均采用 Q355B", "source": S}},
  "support": {{"kind": "fixed", "text": "柱脚刚接", "source": S}},
  "notes": [{{"text": "看不清或拿不准的内容", "source": S}}]
}}
其中 S 形如 {{"region": 1, "bbox": [0.1, 0.2, 0.15, 0.25], "confidence": 0.9}}。

硬性规则：
1. **text 一律照抄图上原文**，包括千分位、±号、英尺英寸符号；不要换算单位、不要补零、
   不要四舍五入。看起来无法换算或前后对不上时也照抄，不要改写成别的形式。
2. **绝不估算**：图上没写的间距、标高、规格、单位、柱脚做法，就不写这条（或写 null），
   并在 notes 里说明缺什么。不要按比例尺或像素长度推算任何尺寸。
3. x 组是数字轴号（竖向轴线，从左到右排列），y 组是字母轴号（横向轴线）。
   axes 按图上顺序列出（x 从左到右，y 从下到上）；pos 是轴线在区域图里的位置
   （x 组写横坐标、y 组写纵坐标），只用于核对。
4. spacings 写相邻两轴之间标注的间距；从第一根轴线量到最后一根轴线的总尺寸写进 total。
   量到建筑外边缘、柱外皮的总尺寸不是轴线总尺寸，不要写进 total，写进 notes。
5. levels 的 name 用图上的层名（如 2F、二层）；没有层名就用标高原文作 name。
   梁的 levels、柱的 from_level/to_level 引用这些 name。平面图标明适用于哪些楼层时，
   梁的 levels 列出所有这些楼层。
6. 梁按编号所在位置写：along 是梁所在的轴线，from/to 是起止的另一方向轴号。
   柱写所在轴线交点 at 和起止标高。只有一组轴线的立面图，梁不写 along，柱 at 只写一个轴号。
7. units.value 只能是 mm、cm、m，且只在图上有单位说明时填写。
8. support.kind 只能是 fixed（刚接/固接）、pinned（铰接）；没有说明就写 null。
9. 看不清、被遮挡、拿不准的内容写进 notes，并把对应读数的 confidence 写低。
"""


@dataclass
class Region:
    index: int
    image_id: str
    kind: str
    path: Path
    origin: tuple[int, int]         # 裁剪区域在原图中的左上角（像素）
    size: tuple[int, int]           # 裁剪区域在原图中的宽高（像素）


@dataclass
class ExtractResult:
    evidence: dict[str, Any] | None = None
    regions: list[Region] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    calls: int = 0
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.evidence is not None


def image_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    """每张图纸的编号、文件名、哈希与像素尺寸——证据里出处的锚点。"""
    from PIL import Image

    records = []
    for index, path in enumerate(paths, 1):
        path = Path(path)
        with Image.open(path) as image:
            width, height = image.size
        records.append({"id": f"P{index}", "name": path.name, "path": str(path),
                        "sha256": file_digest(path), "width": width, "height": height})
    return records


class DrawingExtractor:
    """调用多模态模型抄写图纸文字证据。``parser`` 是配置好的 SketchParser。"""

    def __init__(self, parser: Any, work_dir: str | Path):
        self.parser = parser
        self.work_dir = Path(work_dir)

    # ---- 模型调用 ----
    def _call(self, system: str, blocks: list[tuple[str, Any]]) -> str:
        """blocks 为 ("text", 字符串) 或 ("image", 图片路径) 的序列。"""
        parser = self.parser
        client = parser._get_client()
        if parser._provider in ("openai", "deepseek"):
            content = [{"type": "text", "text": value} if kind == "text" else
                       {"type": "image_url",
                        "image_url": {"url": parser._encode_image(value)}}
                       for kind, value in blocks]
            response = client.chat.completions.create(
                model=parser._model, temperature=parser._temperature, max_tokens=8000,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": content}])
            return response.choices[0].message.content or ""
        content = [{"type": "text", "text": value} if kind == "text" else
                   parser._anthropic_image(parser._encode_image(value))
                   for kind, value in blocks]
        response = client.messages.create(
            model=parser._model, max_tokens=8000, system=system,
            messages=[{"role": "user", "content": content}])
        return response.content[0].text if response.content else ""

    # ---- 第一遍：定位 ----
    def _layout(self, images: list[dict[str, Any]], result: ExtractResult) -> list[dict]:
        blocks: list[tuple[str, Any]] = []
        for index, image in enumerate(images, 1):
            blocks += [("text", f"图 {index}"), ("image", image["path"])]
        blocks.append(("text", "请标出各张图里的区域。"))
        result.calls += 1
        try:
            payload = self.parser._extract_json(self._call(LAYOUT_PROMPT, blocks))
        except (ValueError, json.JSONDecodeError) as exc:
            result.errors.append(f"版面定位未得到有效结果，改为整图读取：{exc}")
            return []
        regions = []
        for item in payload.get("regions") or []:
            try:
                image = int(item["image"])
                bbox = [float(v) for v in item["bbox"]]
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= image <= len(images) and len(bbox) == 4 \
                    and 0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1:
                regions.append({"image": image, "kind": item.get("kind", "other"),
                                "bbox": bbox})
        return regions

    def _crop(self, images: list[dict[str, Any]], layout: list[dict]) -> list[Region]:
        """按版面框裁剪原分辨率区域；没有框的图整张送入。"""
        from PIL import Image

        self.work_dir.mkdir(parents=True, exist_ok=True)
        regions: list[Region] = []
        for number, image in enumerate(images, 1):
            boxes = [item for item in layout if item["image"] == number
                     and item["kind"] != "other"]
            if not boxes or any((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
                                >= WHOLE_IMAGE_RATIO for b in boxes):
                boxes = [{"kind": "whole", "bbox": [0.0, 0.0, 1.0, 1.0]}]
            width, height = image["width"], image["height"]
            with Image.open(image["path"]) as source:
                source = source.convert("RGB")
                for box in boxes:
                    u1, v1, u2, v2 = box["bbox"]
                    left = max(0, int((u1 - REGION_PADDING) * width))
                    top = max(0, int((v1 - REGION_PADDING) * height))
                    right = min(width, int(round((u2 + REGION_PADDING) * width)))
                    bottom = min(height, int(round((v2 + REGION_PADDING) * height)))
                    crop = source.crop((left, top, right, bottom))
                    longest = max(crop.size)
                    if longest > MAX_CROP_SIDE:
                        ratio = MAX_CROP_SIDE / longest
                        crop = crop.resize((max(1, int(crop.width * ratio)),
                                            max(1, int(crop.height * ratio))))
                    index = len(regions) + 1
                    path = self.work_dir / f"drawing_region_{index:02d}.png"
                    crop.save(path)
                    regions.append(Region(index, image["id"], str(box["kind"]), path,
                                          (left, top), (right - left, bottom - top)))
        return regions

    # ---- 第二遍：读取 ----
    def _transcribe_blocks(self, regions: list[Region]) -> list[tuple[str, Any]]:
        names = {"plan": "平面图", "elevation": "立面图", "section_table": "截面表",
                 "notes": "说明", "whole": "整张图纸"}
        blocks: list[tuple[str, Any]] = []
        for region in regions:
            blocks += [("text", f"区域 {region.index}（{names.get(region.kind, '其他')}）"),
                       ("image", str(region.path))]
        return blocks

    @staticmethod
    def _map_sources(value: Any, regions: dict[int, Region],
                     images: dict[str, dict[str, Any]]) -> Any:
        """把「区域内坐标」换回「原图坐标」，出处统一指向原图。"""
        if isinstance(value, list):
            return [DrawingExtractor._map_sources(v, regions, images) for v in value]
        if not isinstance(value, dict):
            return value
        out = {key: DrawingExtractor._map_sources(v, regions, images)
               for key, v in value.items()}
        source = out.get("source")
        if isinstance(source, dict) and "region" in source:
            number = source.pop("region")
            region = regions.get(number) if isinstance(number, int) else None
            if region is not None:
                image = images[region.image_id]
                source["image"] = region.image_id
                bbox = source.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4 and all(
                        isinstance(v, (int, float)) for v in bbox):
                    w, h = region.size
                    x0, y0 = region.origin
                    source["bbox"] = [
                        min(1.0, max(0.0, (x0 + float(bbox[0]) * w) / image["width"])),
                        min(1.0, max(0.0, (y0 + float(bbox[1]) * h) / image["height"])),
                        min(1.0, max(0.0, (x0 + float(bbox[2]) * w) / image["width"])),
                        min(1.0, max(0.0, (y0 + float(bbox[3]) * h) / image["height"]))]
                else:
                    source.pop("bbox", None)
        return out

    @staticmethod
    def _map_axis_positions(evidence: dict[str, Any], images: dict[str, dict[str, Any]],
                            raw: dict[str, Any], regions: dict[int, Region]) -> None:
        """轴线 pos 也是区域内坐标，按该轴出处所在区域换回原图坐标。"""
        for family in ("x", "y"):
            raw_axes = ((raw.get("grids") or {}).get(family) or {}).get("axes") or []
            axes = ((evidence.get("grids") or {}).get(family) or {}).get("axes") or []
            for raw_axis, axis in zip(raw_axes, axes, strict=False):
                pos = axis.get("pos") if isinstance(axis, dict) else None
                src = raw_axis.get("source") if isinstance(raw_axis, dict) else None
                region = regions.get(src.get("region")) if isinstance(src, dict) else None
                if not isinstance(pos, (int, float)) or region is None:
                    if isinstance(axis, dict):
                        axis.pop("pos", None)
                    continue
                image = images[region.image_id]
                if family == "x":
                    axis["pos"] = (region.origin[0] + float(pos) * region.size[0]) \
                        / image["width"]
                else:
                    axis["pos"] = (region.origin[1] + float(pos) * region.size[1]) \
                        / image["height"]

    def _finalize(self, raw: dict[str, Any], regions: list[Region],
                  images: list[dict[str, Any]]) -> dict[str, Any]:
        by_index = {region.index: region for region in regions}
        by_id = {image["id"]: image for image in images}
        evidence = self._map_sources(raw, by_index, by_id)
        self._map_axis_positions(evidence, by_id, raw, by_index)
        evidence["format"] = EVIDENCE_FORMAT
        evidence["images"] = [{key: image[key] for key in
                               ("id", "name", "sha256", "width", "height")}
                              for image in images]
        return evidence

    def extract(self, image_paths: list[str | Path], *, max_repairs: int = 1,
                progress: Callable[[str], None] | None = None) -> ExtractResult:
        """抄写一组图纸。失败时 ``evidence`` 为 None，原因在 ``errors``。"""
        start = time.monotonic()
        result = ExtractResult()
        notify = progress or (lambda message: None)
        try:
            images = image_records(image_paths)
        except OSError as exc:
            result.errors.append(f"图纸读取失败：{exc}")
            return result
        if not images:
            result.errors.append("没有选择图纸。")
            return result
        try:
            notify("正在定位平面图、立面图、截面表与说明")
            layout = self._layout(images, result)
            result.regions = self._crop(images, layout)
            blocks = self._transcribe_blocks(result.regions)
            correction = ""
            for attempt in range(max_repairs + 1):
                notify("正在逐块抄写轴号、间距、标高与规格" if attempt == 0
                       else "正在按校验结果重读可疑区域")
                request = list(blocks)
                if correction:
                    request.append(("text", correction))
                request.append(("text", "请抄写以上区域，输出证据 JSON。"))
                result.calls += 1
                try:
                    raw = self.parser._extract_json(self._call(TRANSCRIBE_PROMPT, request))
                except (ValueError, json.JSONDecodeError) as exc:
                    result.errors.append(f"第 {attempt + 1} 次读取没有得到 JSON：{exc}")
                    correction = "上次输出不是合法 JSON。只输出一个 JSON 对象。"
                    continue
                problems = validate_evidence(raw)
                evidence = None
                if not problems:
                    evidence = self._finalize(raw, result.regions, images)
                    problems = [row.reread for row in assemble(evidence).rows if row.reread]
                if evidence is not None:
                    result.evidence = evidence
                if not problems:
                    break
                result.errors.append(f"第 {attempt + 1} 次读取需要复核：" + "；".join(problems[:6]))
                correction = ("上次输出中以下读数彼此对不上。请对照区域图逐字重读相关文字后输出完整 JSON。"
                              "如果原图本来就是这样写的，保持原文不变，不要为了让数字一致或能被换算"
                              "而改写；图上确实没有的保持缺失，不要编造：\n" + "\n".join(problems[:12]))
        except Exception as exc:  # noqa: BLE001  提供商与网络边界：原样报给用户
            result.errors.append(f"调用识别模型失败：{type(exc).__name__}: {exc}")
        result.duration_ms = (time.monotonic() - start) * 1000
        return result
