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
from copy import deepcopy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from recognition_draft import DRAFT_FORMAT, RecognitionDraft
from multimodal_workflow import (V2_DRAFT_FORMAT, MultimodalControllerState,
                                 prepare_vision_entities, validate_v2_draft)


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

V2_SKETCH_SYSTEM_PROMPT = """你是结构工程草图识别助手。只输出一个 JSON 对象，
不要 Markdown、不要解释。输出紧凑 JSON，不加缩进、空行或重复的簿记字段。

**只需要输出下面这四个字段**，其余簿记字段由调用方填写，你写了也会被覆盖：

{
  "image_model": {
    "nodes":  [{"id": 1, "u": 0.12, "v": 0.83}],
    "members":[{"id": 1, "i": 1, "j": 2}],
    "supports":[{"node": 1, "kind": "fixed"}],
    "load_cases":[{"name": "D",
      "nodal_loads":[{"node": 7, "name": "P1", "value": 30, "unit": "kN",
                      "direction": [1, 0, 0]}],
      "member_loads":[{"member": 7, "name": "q1", "value": 18, "unit": "kN/m",
                       "direction": [0, 0, -1]}],
      "member_spans":[{"member": 8, "name": "q2", "kind": "partial",
                       "value": 12, "unit": "kN/m", "direction": [0, 0, -1],
                       "range": {"start": 1000, "end": 3000, "unit": "mm"}}]}]
  },
  "entities":  [{"id": "E1", "kind": "member", "target": {"member": 1}, "confidence": 0.9}],
  "dimensions":[{"id": "D1", "text": "6000", "unit": "mm",
                 "image_geometry": {"line": [[0.12,0.95],[0.5,0.95]]}}],
  "issues":    [{"id": "I1", "category": "low_confidence",
                 "severity": "blocking", "entity_refs": ["node:3"],
                 "message": "左下角支座符号被尺寸线压住，看不清是固接还是铰接",
                 "status": "open"}]
}

关键约定，写错会被直接拒绝：

1. **image_model 是"图里那个结构"，不是图片的元信息。** 不要写图片哈希、
   尺寸、分辨率、格式——那些调用方已经有了。这里要的是节点、杆件、支座。
2. **节点坐标用 u/v**，取值 0~1 的归一化图片坐标，原点在左上角。不要用像素。
3. **杆件按最小单元拆分**：一根柱子跨两层就是两根杆件，中间那个节点必须建出来；
   一层里两跨的梁是两根杆件，不是一根。杆件只能引用已经列出的节点 id。
   每个无向端点对只列一次，i/j 对调仍是同一杆件，不得自连接。
   怀疑实际双杆时只列一条几何边并报告待核实问题，不以重复边表达双杆。
4. 节点 id 与杆件 id 都是从 1 开始的正整数，各自不重复。
   **节点必须按固定顺序编号：先按 v 从大到小分层（图片下方的先编），
   同一层内按 u 从小到大（左边的先编）。** 视觉模型给的编号本来是任意的，
   定死顺序之后人才能把"识别出的 3 号"和图上那个节点对起来。
5. **支座符号一定要认。** 它们在柱脚，形状分三类：矩形加地面阴影线是固接，
   三角形是铰接，三角形下面带滚轴（小圆）是滚动。认出来写进
   image_model.supports，kind 取 fixed / pinned / roller；
   看不清是哪一类就写进 issues，但不要漏掉这个节点有支座这件事。
6. **荷载照抄图上的数字和单位，不要自己换算。** value 写图上标的数值、
   unit 写图上标的单位（kN、kN/m、kN·m）、direction 写箭头指向的单位向量
   （向下是 [0,0,-1]，向右是 [1,0,0]）。换算成求解器单位是代码的事。
   以上方向示例适用于 XZ 平面；若调用方提供其他工作平面，向右沿 first_axis
   正向、向下沿 second_axis 负向，必须按所提供的 axis_mapping 写全局方向。
   图上没写数值的荷载，只写进 issues，不要凭箭头长短猜大小。
7. 只写图里有证据的东西。不得补材料、截面、默认支座、默认荷载或默认尺寸。
   看不清、拿不准的一律写进 issues，不要猜。
8. 图上没有可靠尺寸时，坐标只保持相对比例，并在 issues 里要求用户标定；
   绝不能把相对坐标当成米。
9. 每个节点、杆件、支座、荷载都要提供一条 entities 观察记录，confidence 为
   0~1 的识别确定性；无法判断时写 null，不能用高分掩盖不确定性。
   target 分别写 {"node":1}、{"member":1}、
   {"support":{"node":1,"name":"S1"}}、
   {"load":{"case":"D","collection":"nodal_loads","name":"P1"}}。
   支座和荷载须有对应名称。节点、杆件的 entities 不再重复 image_geometry，
   调用方从 image_model 的 u/v 和 i/j 原样复用位置；仍逐项保留 target 和 confidence。
   支座、荷载符号和尺寸的独立可见证据位置仍写 image_geometry，不能用节点代替符号。
   人工审核由用户完成，不得输出 verified 或 source=user，也不得将问题标为已解决。
10. **先区分结构主体与辅助图，再提取拓扑。** 尺寸线、坐标轴、文字引线、
    剪力图、弯矩图、变形曲线及其他结果图均不是杆件。
    同图有初始位形和变形轮廓时，按初始位形提取梁轴；无法区分时报告问题，
    不得将变形后的曲线端点替代原结构节点。
11. **节点位置取杆件中心轴。** 厚线或矩形梁用两边界的中线，不取外轮廓；
    支座节点是梁轴上的附着位置，不是三角形顶点、墙体边缘或文字位置。
    u/v 应尽量保留四位小数，位置仍必须有可见证据，不能用多位小数掩盖不确定。
12. 仅在杆件端点、真实连接、支座或集中荷载作用点建节点。字母标注的中点、
    尺寸分界或均布荷载图中单独的文字，不构成新增节点和杆件分段的证据。
13. **先判断箭头表示的物理量。** 支座旁明确标出的反力（例如 RA、RB）、
    剪力图/弯矩图的箭头不是外加荷载；不要写进 load_cases。
    不能区分反力和外荷载时只报告待核实问题，不选择一个用于求解。
    外荷载只有符号而没有数值/单位时，以 load_incomplete 问题保留作用节点或杆件，
    不得写 value=null 的求解荷载，更不能补零或推算数值。
14. **集中力和集中矩即使只有符号，也必须保留作用点。** 先逐个检查所有外加
    直箭头的接触点及圆弧箭头的中心所在梁截面，将作用点投影到初始梁中轴，
    在该处建立节点并拆分相邻杆件，再用 load_incomplete 引用 node:id。
    不以箭头尾端、圆弧外缘或字母位置建节点；不要因数值未知只保留梁两端。
    位置本身无法辨认时报告待核实问题，不推算位置。
15. **分布荷载只施加在实际覆盖范围。** 看清箭头列的起止边界，不把局部
    荷载扩大成整跨；数值可用且边界处确有荷载起止证据时可拆分杆件，只引用覆盖段。
    member_loads 仅表示所引用杆件的整段均布荷载。范围在杆件内部且图上
    明确标注距 i 端的长度时，使用 member_spans、kind=partial，range.start/end
    照抄长度及 range.unit（m/mm/cm），代码换算为 a/b；不是像素、比例或米的猜测。
    范围或数值/单位未知时保留 load_incomplete，引用覆盖杆件并记录可见起止
    线段 image_geometry；符号分布范围不强制新建节点。不得补全跨范围、默认强度或零向量。
16. 集中力 direction 是全局力方向；集中矩 direction 是右手定则的全局转轴，
    unit 使用 N·m/kN·m，代码写入 [Fx,Fy,Fz,Mx,My,Mz] 的后三项。
    例如 XZ 平面绕 +Y 的集中矩：{"node":3,"name":"M1","value":5,
    "unit":"kN·m","direction":[0,1,0]}。顺逆时针与转轴关系须结合
    工作平面；看不清旋向时仅报告问题。不能把力矩单位用于力或分布强度。
"""


# 单次视觉调用的超时（秒）。视觉模型读一张大图并吐出完整草稿，几十秒是正常的；
# 但没有上限就等于没有反馈——SDK 默认 600 秒 × 3 次重试，界面会像死掉一样。
# 设成有限值之后，慢就是慢、断就是断，两者在错误信息里分得开。
REQUEST_TIMEOUT_SECONDS = 90.0


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


def _fill_bookkeeping(payload: Any, image_hash: str, source_path: str) -> Any:
    """把**代码自己拥有的簿记字段**补齐，再交给校验。

    模型该做的是看图，不是记账。实测 deepseek-v4-flash-vision-exp 能正确认出
    柱、梁、荷载和尺寸标注，却在 format / source / work_plane / scale 这几个
    字段上翻车——它把 work_plane 写成 "proposed"、scale 写成 "unknown"
    （本该是对象），source 写成 title/type/author（本该是 original_path /
    image_hash / preprocessing），format 干脆没给。这些全是常量或调用方已知的
    东西，让模型去凑，只会把"能不能看懂图"的问题伪装成"格式对不对"的问题。

    **image_hash 尤其不该由模型回填**：让它把调用方给的哈希抄回来，既没有
    增加任何保证，还留了抄错或自己编一个的余地。由代码写入是更强的绑定，
    不是更弱的——绑定的目的是"这份草稿属于这张图"，而代码才是权威。

    只补簿记，不碰观察：entities / dimensions / intersections / issues /
    image_model 一概保持模型的原样，识别得对不对仍由后续校验和人工确认判断。
    """
    if not isinstance(payload, dict):
        return payload
    payload["format"] = V2_DRAFT_FORMAT
    payload["model"] = None
    payload["merge_plan"] = None
    payload["revision"] = 0
    payload["confirmation"] = None

    source = payload.get("source")
    observed = dict(source) if isinstance(source, dict) else {}
    observed.pop("image_hash", None)
    payload["source"] = {
        "original_path": source_path,
        "image_hash": image_hash,
        "preprocessing": {"perspective_status": "unconfirmed"},
        # 模型对图纸的描述（标题、类型）留着，它是有用的线索，只是不能当簿记用
        **{k: v for k, v in observed.items()
           if k not in ("original_path", "preprocessing")},
    }

    def _as_object(value: Any, skeleton: dict) -> dict:
        """模型常把这类对象压成一个状态字符串，按状态还原成骨架。"""
        if isinstance(value, dict):
            return {**skeleton, **value}
        got = dict(skeleton)
        if isinstance(value, str) and value:
            got["status"] = value
        return got

    payload["work_plane"] = _as_object(payload.get("work_plane"), {
        "status": "proposed", "plane": "XZ", "offset": 0.0,
        "axis_mapping": {"first_axis": "X", "second_axis": "Z",
                         "offset_axis": "Y", "image_right_sign": 1,
                         "image_up_sign": 1}})
    payload["scale"] = _as_object(payload.get("scale"), {
        "status": "unknown", "length_per_pixel": None, "unit": "m/px",
        "anchor_node": None, "anchor_coordinates_xyz": [0.0, 0.0, 0.0],
        "evidence_ids": []})
    for key in ("entities", "dimensions", "intersections", "issues"):
        payload.setdefault(key, [])

    # 规整 image_model 时顺手收集"这一项是猜的/没换算出来"，和模型自己报的
    # issues 合在一起编号——猜测必须走确认环节，不能只留在代码里。
    guessed: list[dict] = []
    image_model = payload.get("image_model")
    if isinstance(image_model, dict):
        if isinstance(image_model.get("supports"), list):
            supports = []
            for item in image_model["supports"]:
                if not isinstance(item, dict):
                    continue
                support, problem = _as_support(item)
                supports.append(support)
                if problem:
                    guessed.append(problem)
            image_model["supports"] = supports
        for case in image_model.get("load_cases") or []:
            if isinstance(case, dict):
                guessed.extend(_convert_case_loads(case))

    payload["issues"] = [
        _as_issue(item, index)
        for index, item in enumerate(list(payload["issues"]) + guessed, 1)]
    prepare_vision_entities(payload)
    return payload


# 图上写的单位 → 模型单位（N-m-Pa）的倍数。
_UNIT_FACTORS = {"n": 1.0, "kn": 1e3, "n/m": 1.0, "kn/m": 1e3,
                 "n·m": 1.0, "n.m": 1.0, "nm": 1.0,
                 "kn·m": 1e3, "kn.m": 1e3, "knm": 1e3}
_FORCE_UNITS = {"n", "kn"}
_DENSITY_UNITS = {"n/m", "kn/m"}
_MOMENT_UNITS = set(_UNIT_FACTORS) - _FORCE_UNITS - _DENSITY_UNITS
# 报错时给人看的写法。_UNIT_FACTORS 的键是归一化后的（小写、去空格、
# 认几种点号写法），直接抖出去会让人以为要写 "knm" 才认。
_UNIT_DISPLAY = "N、kN、N/m、kN/m、N·m、kN·m"


def _convert_case_loads(case: dict) -> list[dict]:
    """把 value + unit + direction 换算成求解器要的分量向量。

    **让模型照抄图上的数字和单位，换算交给代码。** 图上写 "18 kN/m"，
    模型报 value=18 / unit="kN/m" / direction=[0,0,-1]，代码算出
    w=[0,0,-18000]。反过来让模型直接吐 -18000，等于让它做单位换算——
    那是算术，不是观察，正好踩中"大模型只产结构，不产数值"这条线。

    已经给了 load / w 向量的条目不动，模型偶尔会两种都给。

    **换算完必须把 value/unit/direction 拆掉。** 契约里 image_model 的荷载
    用的是现有载荷合同（`w` / `load`），这三个字段只是识别期的脚手架；留着
    它们会一路 deepcopy 进 SI 模型，最后被 schema 以"多余属性"拒收——
    错误信息指着大模型的词汇，而不是用户能动手的东西。

    认不出单位时不闷掉：按契约记一条 `load_incomplete` 阻断问题，把图上的
    原话（值、单位、方向）带上让人来判。闷掉的后果是整条荷载凭空消失，
    而模型照样算得出一个像模像样的结果。
    """
    def vector(item: dict, size: int, units: set[str]) -> list[float] | None:
        unit = str(item.get("unit") or "").strip().lower().replace(" ", "")
        scale = _UNIT_FACTORS.get(unit) if unit in units else None
        direction = item.get("direction")
        value = item.get("value")
        if scale is None or not isinstance(direction, (list, tuple)):
            return None
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value))):
            return None
        if len(direction) not in (3, size) or any(
                not isinstance(component, (int, float)) or isinstance(component, bool)
                or not math.isfinite(float(component)) for component in direction):
            return None
        offset = 3 if size == 6 and unit in _MOMENT_UNITS else 0
        if size == 6 and len(direction) == 6 and any(
                direction[index] != 0 for index in range(6) if not offset <= index < offset + 3):
            return None
        out = [0.0] * size
        for index in range(3):
            source = index + offset if len(direction) == 6 else index
            component = direction[source]
            if isinstance(component, (int, float)) and not isinstance(component, bool):
                out[index + offset] = float(value) * scale * float(component)
        return out

    problems: list[dict] = []

    def load_target(item: dict, collection: str, index: int = 1) -> dict:
        if case.get("name"):
            return {"load_target": {"case": case["name"], "collection": collection,
                                    "name": item.get("name") or f"{collection}-{index}"}}
        return {}

    def convert(items: Any, key: str, size: int, ref: str, label: str,
                units: set[str]) -> None:
        collection = {"load": "nodal_loads", "w": "member_loads", "w1": "member_spans"}[key]
        for index, item in enumerate(items or [], 1):
            if not isinstance(item, dict):
                continue
            if not isinstance(item.get(key), (list, tuple)):
                got = vector(item, size, units)
                if got is None:
                    if all(item.get(k) is None for k in ("value", "unit", "direction")):
                        components = "Fx/Fy/Fz/Mx/My/Mz" if key == "load" else "全局 X/Y/Z 分量"
                        message = (f"{label} {item.get(ref)} 的荷载 {item.get('name') or f'{collection}-{index}'} "
                                   f"尚未提供有效分量。请在对应荷载编辑器中明确填写 {components}；"
                                   "确认该分量为零时输入 0，不能用默认零值代替缺失值。")
                    else:
                        message = (f"{label} {item.get(ref)} 上这条荷载没能换算："
                                   f"值 {item.get('value')!r}、单位 {item.get('unit')!r}、"
                                   f"方向 {item.get('direction')!r}。认得的单位是 {_UNIT_DISPLAY}。"
                                   "请确认图上写的是什么，或直接给出分量向量。")
                    problems.append({
                        "category": "load_incomplete", "severity": "blocking",
                        **load_target(item, collection, index),
                        "observations": [deepcopy(item)],
                        "entity_refs": [f"{ref}:{item.get(ref)}"],
                        "message": message,
                    })
                else:
                    item[key] = got
            for scaffold in ("value", "unit", "direction"):
                item.pop(scaffold, None)

    convert(case.get("nodal_loads"), "load", 6, "node", "节点", _FORCE_UNITS | _MOMENT_UNITS)
    convert(case.get("member_loads"), "w", 3, "member", "杆件", _DENSITY_UNITS)
    for index, item in enumerate(case.get("member_spans") or [], 1):
        if not isinstance(item, dict):
            continue
        if not item.get("name"):
            item["name"] = f"member_spans-{index}"
        kind = item.get("kind")
        units = _FORCE_UNITS if kind == "point" else _MOMENT_UNITS if kind == "moment" else _DENSITY_UNITS
        convert([item], "w1", 3, "member", "杆件", units)
        observed_range = item.pop("range", None)
        range_valid = observed_range is None
        if isinstance(observed_range, dict) and kind == "partial":
            factor = {"m": 1.0, "mm": 0.001, "cm": 0.01}.get(observed_range.get("unit"))
            a, b = observed_range.get("start"), observed_range.get("end")
            if (factor is not None and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                          and math.isfinite(v) for v in (a, b)) and 0 <= a < b):
                item.update(a=a * factor, b=b * factor)
                range_valid = True
        if not range_valid or (kind == "partial" and (not all(isinstance(item.get(k), (int, float))
                and not isinstance(item[k], bool) and math.isfinite(item[k]) for k in ("a", "b"))
                or not 0 <= item["a"] < item["b"])):
            problems.append({"category": "load_incomplete", "severity": "blocking",
                **load_target(item, "member_spans", index),
                "entity_refs": [f"member:{item.get('member')}"],
                "message": f"局部分布荷载缺少有效作用范围：{observed_range!r}。请确认距杆件 i 端的起止长度和单位，不能按整跨施加。"})
    return problems


# 支座类型 → 六自由度约束掩码。让模型判"这是铰接"，让代码写掩码：
# 认符号是视觉活，把它翻译成 [1,1,1,0,0,0] 是编码活，混在一起两头都做不好。
_SUPPORT_MASKS = {
    "fixed":  (1, 1, 1, 1, 1, 1),
    "pinned": (1, 1, 1, 0, 0, 0),
    "roller": (0, 1, 1, 0, 0, 0),      # 竖向支承，沿 x 可滑动；面外由 y 约束
}


def _as_support(item: dict) -> tuple[dict, dict | None]:
    """把识别出的支座补成项目的规范形状，并说明哪一个是猜的。

    模型按提示词给的是 {"node": 1, "kind": "pinned"}——对视觉模型来说，
    判断"三角形=铰接"远比直接吐出 [1,1,1,0,0,0] 可靠。但项目里所有消费方
    （模型树、求解器、校验）读的都是 fix 掩码，缺了它 model_tree 会
    KeyError: 'fix' 并把整棵树的重建带崩。缺什么补什么，在这里补掉。

    翻译完要把 kind 拆掉：契约里 supports 是 {name,node,fix[6]}，schema 会
    以"多余属性"拒收 kind，和荷载那三个脚手架字段是同一回事。

    认不出类型时仍按铰接兜底，但**痕迹不能只是一个默认名**：原来靠
    `setdefault("name", "待确认支座")` 留印，模型自己给了 name 就什么都不剩，
    弹性支座会一声不响地变成铰接。改成记一条问题，由确认环节兜住。
    """
    got = dict(item)
    if isinstance(got.get("fix"), (list, tuple)) and len(got["fix"]) == 6:
        got["fix"] = [int(bool(v)) for v in got["fix"]]
        got.pop("kind", None)
        return got, None
    raw = got.pop("kind", None)
    kind = str(raw or "").strip().lower()
    got["fix"] = list(_SUPPORT_MASKS.get(kind, _SUPPORT_MASKS["pinned"]))
    if kind in _SUPPORT_MASKS:
        return got, None
    got.setdefault("name", "待确认支座")
    # 归到 low_confidence 是因为界面只给这一类"确认"按钮——
    # 猜出来的铰接正需要一次人工点头，而不是一条只能干瞪眼的阻断。
    return got, {
        "category": "low_confidence", "severity": "blocking",
        "entity_refs": [f"node:{got.get('node')}"],
        "message": (f"节点 {got.get('node')} 的支座类型 {raw!r} 认不出来，"
                    f"先按铰接（平动全约束、转动全释放）处理。认得的是 "
                    f"{'、'.join(sorted(_SUPPORT_MASKS))}——请确认或改掉。"),
    }


def _as_issue(item: Any, index: int) -> dict:
    """把 issues 里的元素规整成对象。

    v2 的 issue 是带 category/severity/entity_refs/status 的对象，界面会去读
    这些字段。模型很容易只吐一句话——那样 `item.get(...)` 直接
    AttributeError: 'str' object has no attribute 'get'，整个面板崩掉。
    **模型输出的形状永远不可信，消费前必须规整**，这比在提示词里多写一句更可靠。
    """
    skeleton = {"id": f"I{index}", "category": "low_confidence",
                "severity": "blocking", "entity_refs": [], "message": "",
                "status": "open", "resolution": None, "resolved_by": None}
    if isinstance(item, dict):
        got = {**skeleton, **item}
        got["entity_refs"] = list(got.get("entity_refs") or [])
        got["message"] = str(got.get("message") or "")
        # 识别模型没有人工审核权限，不能替用户消除阻断问题。
        got.update(status="open", resolution=None, resolved_by=None)
        return got
    return {**skeleton, "message": str(item)}


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
    call_metadata: list[dict[str, Any]] = field(default_factory=list)


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
        self._last_call_metadata: dict[str, Any] = {}

    def request_options(self) -> dict[str, Any]:
        """记录实际请求参数；视觉提取直接输出结构，避免默认推理占满输出预算。"""
        options = {"max_tokens": 4000}
        if self._provider != "anthropic":
            options["temperature"] = self._temperature
        if self._provider == "deepseek" and self._model in (
                "deepseek-flash", "deepseek-v4-flash-vision-exp"):
            options["extra_body"] = {"thinking": {"type": "disabled"}}
        return options

    def _record_call_metadata(self, response: Any, raw: str, *,
                              finish_reason: Any, reasoning_present: bool = False) -> None:
        """只记录诊断标量，不保存密钥、请求全文或模型推理内容。"""
        metadata = {"content_characters": len(raw), "has_reasoning_content": reasoning_present}
        if isinstance(finish_reason, str):
            metadata["finish_reason"] = finish_reason
        usage = getattr(response, "usage", None)
        names = (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")) \
            if self._provider == "anthropic" else (("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"), ("total_tokens", "total_tokens"))
        for source, target in names:
            value = getattr(usage, source, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                metadata[target] = value
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool) and reasoning_tokens >= 0:
            metadata["reasoning_tokens"] = reasoning_tokens
        self._last_call_metadata = metadata

    @staticmethod
    def _default_model(provider: str) -> str:
        # DeepSeek 平台上**没有** deepseek-vl（那是开源权重的名字，不是 API 上的
        # 模型）。2026-10-09 核对官方 Vision 文档及 /models，当前入口为 deepseek-flash；
        # 旧视觉实验名称仍是别名，但不再作为默认值。
        # 走标准 OpenAI 兼容的 chat/completions，content 用块数组 + base64 data URL。
        # 用错名字的表现是 model not found，而不是识别不准，很容易被误读成
        # "多模态没跑通"。
        return {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "deepseek": "deepseek-flash",
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
                # **必须显式给超时和重试次数。** openai SDK 默认超时 600 秒、
                # 自动重试 2 次，最坏情况一次识别会静默阻塞半小时；界面上只有
                # 一句"正在识别"，用户分不出"还在跑"和"已经死了"。
                # 本项目自己有修复重试循环，SDK 层不需要再重试一遍。
                self._client = OpenAI(api_key=self._api_key,
                                      base_url=self._base_url,
                                      timeout=REQUEST_TIMEOUT_SECONDS,
                                      max_retries=0)
            elif self._provider == "anthropic":
                try:
                    import anthropic
                except ImportError as exc:
                    raise RuntimeError(
                        "缺少 anthropic 依赖。请执行 pip install -r requirements-multimodal.txt"
                    ) from exc
                self._client = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
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
                f"识别上下文及格式修正要求：\n{correction}\n"
                f"只输出修正后的 JSON，不要输出其他文字。"})

        if self._provider in ("openai", "deepseek"):
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt or self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                **self.request_options(),
            )
            choice = response.choices[0] if response.choices else None
            raw = (choice.message.content or "") if choice else ""
            self._record_call_metadata(response, raw,
                finish_reason=getattr(choice, "finish_reason", "missing_choice"),
                reasoning_present=bool(getattr(getattr(choice, "message", None), "reasoning_content", None)))
            return raw
        elif self._provider == "anthropic":
            content: list[dict[str, Any]] = [
                {"type": "text", "text": "请识别这张结构草图，输出 JSON 格式的识别草稿。"},
            ]
            if correction:
                content.append({"type": "text", "text":
                    f"识别上下文及格式修正要求：\n{correction}\n"
                    "只输出修正后的 JSON，不要输出其他文字。"})
            content.append(self._anthropic_image(image_data_url))
            response = client.messages.create(
                model=self._model,
                **self.request_options(),
                system=system_prompt or self._system_prompt,
                messages=[{"role": "user", "content": content}],
            )
            blocks = response.content or []
            raw = "".join(block.text for block in blocks if isinstance(getattr(block, "text", None), str))
            self._record_call_metadata(response, raw, finish_reason=getattr(response, "stop_reason", None),
                reasoning_present=any(getattr(block, "type", None) == "thinking" for block in blocks))
            return raw
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
            except Exception as e:  # noqa: BLE001  外部大模型调用失败要记进 result.errors 再决定重试，不能中断整轮解析
                result.errors.append(f"第{attempt}次调用失败: {type(e).__name__}: {e}")
                break  # API 调用失败（网络/密钥），重试没用

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    def parse_v2_with_retry(
        self, image_path: str | Path, state: MultimodalControllerState,
        job_id: str, *, max_repairs: int = 2,
        is_cancelled: Callable[[], bool] | None = None,
        work_plane: dict | None = None,
        review_actions: bool = False,
    ) -> V2ParseResult:
        """Recognize a v2 draft and atomically deliver it to the controller.

        A malformed or truncated response may be repaired at most ``max_repairs`` times. API
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
        context = f"当前派生图片的 SHA-256 是 {state.image_hash}。"
        if work_plane is not None:
            from image_preprocess import work_plane_payload
            plane = work_plane_payload(work_plane["plane"], offset=work_plane["offset"], confirmed=True)
            context += "用户已确认的工作平面：" + json.dumps(plane, ensure_ascii=False) + "。"
        correction = context

        def cancelled() -> bool:
            return bool((is_cancelled and is_cancelled())
                        or state.job_status != "running" or state.job_id != job_id)

        for attempt in range(1, max_repairs + 2):
            result.attempts = attempt
            if cancelled():
                state.cancel_recognition(job_id)
                result.cancelled = True
                break
            self._last_call_metadata = {}
            try:
                raw = self._call_llm(
                    image_data, correction, system_prompt=V2_SKETCH_SYSTEM_PROMPT)
                result.raw_responses.append(raw)
                if cancelled():
                    state.cancel_recognition(job_id)
                    result.cancelled = True
                    break
                if self._last_call_metadata.get("finish_reason") in ("length", "max_tokens"):
                    raise ValueError("视觉接口输出达到长度限制，草稿不完整。请重新输出完整紧凑 JSON；节点和杆件观察不重复位置，不得省略结构对象、置信度或待核实问题。仍超限时请裁剪到结构主体后重新识别。")
                if not raw.strip():
                    raise RuntimeError("视觉接口未返回识别内容。请检查模型是否支持图片，并裁剪到结构主体后重新识别。")
                payload = self._extract_json(raw)
                payload = _fill_bookkeeping(payload, state.image_hash, str(image_path))
                errors = validate_v2_draft(payload)
                if errors:
                    raise ValueError("；".join(errors))
                from sketch_axis_refinement import refine_horizontal_axis, refine_joint_nodes
                payload = refine_horizontal_axis(payload, image_path)
                payload = refine_joint_nodes(payload, image_path)
                if review_actions:
                    from PIL import Image, ImageOps
                    from sketch_action_review import ACTION_REVIEW_PROMPT, apply_action_review
                    if self._last_call_metadata:
                        result.call_metadata.append({"attempt": attempt, "stage": "geometry", **self._last_call_metadata})
                    self._last_call_metadata = {}
                    if cancelled():
                        state.cancel_recognition(job_id)
                        result.cancelled = True
                        break
                    try:
                        with Image.open(image_path) as source_image:
                            width, height = ImageOps.exif_transpose(source_image).size
                        payload["source"].update(width_px=width, height_px=height)
                        action_context = "待审核几何（只引用其中的杆件编号，不默认完整）：" + json.dumps(
                            {"nodes": payload["image_model"]["nodes"], "members": payload["image_model"]["members"],
                             "support_nodes": [s["node"] for s in payload["image_model"].get("supports", [])]},
                            ensure_ascii=False)
                        result.attempts += 1
                        focused_raw = self._call_llm(image_data, action_context, system_prompt=ACTION_REVIEW_PROMPT)
                        result.raw_responses.append(focused_raw)
                        if cancelled():
                            state.cancel_recognition(job_id)
                            result.cancelled = True
                            break
                        if (not focused_raw.strip() or self._last_call_metadata.get("finish_reason") in ("length", "max_tokens")):
                            raise ValueError("作用点复核内容为空或截断，请人工补充作用节点。")
                        payload = apply_action_review(payload, self._extract_json(focused_raw), image_path=image_path)
                    except Exception as exc:  # noqa: BLE001 - 单次可选视觉复核失败仍保留基础草稿
                        detail = str(exc).rstrip("。")
                        message = f"作用点复核未完成：{type(exc).__name__}: {detail}。请人工核对并补充作用节点。"
                        result.errors.append(message)
                        payload["action_review"] = {"status": "failed", "message": message}
                        payload.setdefault("issues", []).append({"id": "action-review-failed",
                            "category": "load_incomplete", "severity": "blocking", "status": "open",
                            "entity_refs": [], "message": message, "resolution": None, "resolved_by": None})
                    finally:
                        if self._last_call_metadata:
                            result.call_metadata.append({"attempt": result.attempts, "stage": "action_review", **self._last_call_metadata})
                        self._last_call_metadata = {}
                    if cancelled():
                        state.cancel_recognition(job_id)
                        result.cancelled = True
                        break
                if not state.complete_recognition(job_id, payload):
                    result.cancelled = True
                    break
                result.draft = payload
                result.success = True
                break
            except (json.JSONDecodeError, ValueError) as exc:
                message = f"第{attempt}次 v2 校验失败: {exc}"
                result.errors.append(message)
                correction = context + f"上次输出错误：{exc}。请重新看同一张图，输出完整紧凑 v2 JSON。重复端点对须核对原图后修正，不得仅删除报错项或遗漏真实杆件；不补全截断片段。"
            except Exception as exc:  # noqa: BLE001 - provider/network boundary
                result.errors.append(
                    f"第{attempt}次 API 调用失败: {type(exc).__name__}: {exc}")
                break
            finally:
                if self._last_call_metadata:
                    result.call_metadata.append({"attempt": attempt, **self._last_call_metadata})

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
        if not api_key and provider == "deepseek":
            # 项目自带 credentials.load_api_key()，会读仓库根目录的
            # deepseek.key。此前只看环境变量，于是本地明明有密钥文件，
            # 冒烟测试仍然报 SKIPPED_NO_CREDENTIALS。
            from credentials import load_api_key
            api_key = load_api_key() or ""
        return cls(provider=provider, api_key=api_key, model=model,
                   base_url=base_url)
