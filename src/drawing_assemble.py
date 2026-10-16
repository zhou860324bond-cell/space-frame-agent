"""图纸证据 → 模型 JSON 的确定性组装，以及交给用户的核对表。

输入是 ``drawing_evidence`` 定义的文字证据，输出两样东西：

* **核对表**：单位、轴网、标高、杆件、截面、材料、支座逐行列出「值 / 来源 /
  状态」。状态分四档：``ok`` 已读取、``confirm`` 需确认（置信度低、依赖制图
  约定、或像素比例报警）、``missing`` 缺失、``conflict`` 冲突。后三档都要用户
  处理，处理结果写回证据（来源标为用户），再整体重新组装。另有 ``info`` 提示：
  识别模型写下的拿不准之处，只供对照原图，不阻塞生成——它涉及的每个数都已在
  自己那一行按置信度、缺失或冲突处理过了。
* **模型 JSON**：只有在没有缺失和冲突时才生成，并通过 ``model_io.validate_payload``。

组装规则全部是确定性的：间距（含重复写法展开）累加成轴线坐标，标高或层高链
给出 z，梁沿轴线逐跨拆分，
柱在轴线交点逐层拆分，规格交给 ``sections`` 构造器算特性，柱底按图纸说明设
支座。**不从像素量任何长度**；图上轴线位置只用来检查抄写的间距是否可疑。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping

import sections as section_builders
from drawing_evidence import (EVIDENCE_FORMAT, LAYER_ROLES, LENGTH_UNITS, LOW_CONFIDENCE,
                              Dimension, expand_chain, length_factor, parse_designation,
                              parse_material, read_dimension)
from generator import _COLUMN_REF
from model_io import CURRENT_SCHEMA_VERSION, validate_payload

OK, CONFIRM, MISSING, CONFLICT, INFO = "ok", "confirm", "missing", "conflict", "info"

#: 像素比例报警阈值：某段间距的「像素/标注」比例偏离同图中位数超过此比例时
#: 要求确认。依据：扫描与拍照变形通常在 10% 以内，而抄错一位数字（6000→8000、
#: 3600→6300）至少偏 25%；取 20% 介于两者之间。它只触发确认，从不改数值。
PIXEL_SCALE_ALARM = 0.20

SUPPORT_FIX = {"fixed": [1, 1, 1, 1, 1, 1], "pinned": [1, 1, 1, 0, 0, 0]}
SUPPORT_CHOICES = {"固接": "fixed", "铰接": "pinned"}
SUPPORT_NAMES = {value: key for key, value in SUPPORT_CHOICES.items()}
LAYER_CHOICES = {name: key for key, name in LAYER_ROLES.items()}
DELETE_CHOICE = "删除此条"
#: 图上没有编号的梁柱按位置抄写，标签取这两个名字；同类共用一个截面行。
UNNUMBERED = {"beam": "未编号梁", "column": "未编号柱"}
UNIT_CHOICES = ("mm", "m")

_SECTION_KIND_NAMES = {"i": "工字形", "tube": "圆管", "circle": "实心圆",
                       "rectangle": "矩形"}
_COINCIDENT_DIGITS = 9


@dataclass
class CheckRow:
    """核对表的一行。``id`` 是证据里的路径，处理结果按它写回。"""

    id: str
    category: str
    item: str
    value: str
    source: str
    status: str
    message: str = ""
    image: str | None = None
    bbox: list[float] | None = None
    edit: str = "none"            # none / text / choice
    options: tuple[str, ...] = ()
    current: str = ""
    #: 给识别模型重读用的中性描述，只说哪些读数彼此对不上；为空表示不回喂。
    #: 与 message 分开：message 是让**用户**改值的，原样回喂会诱导模型改写原文
    #: （真实验收中 `25'-0"` 被改抄成 `25`）。读不懂的写法从不回喂。
    reread: str = ""

    @property
    def blocking(self) -> bool:
        return self.status in (MISSING, CONFLICT)

    @property
    def confirmable(self) -> bool:
        return self.status == CONFIRM


@dataclass
class Assembly:
    rows: list[CheckRow]
    model: dict[str, Any] | None
    provenance: dict[str, Any] = field(default_factory=dict)
    #: 各轴线坐标（米），供界面把算出的轴网画回原图核对；单位未定时为空。
    axes: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def blocking(self) -> list[CheckRow]:
        return [row for row in self.rows if row.blocking]

    @property
    def pending(self) -> list[CheckRow]:
        return [row for row in self.rows if row.status not in (OK, INFO)]

    @property
    def ready(self) -> bool:
        return self.model is not None and not self.pending

    def summary(self) -> str:
        counts = {status: sum(row.status == status for row in self.rows)
                  for status in (CONFIRM, MISSING, CONFLICT)}
        head = ""
        if self.model is not None:
            head = (f"{len(self.model['nodes'])} 个节点、{len(self.model['members'])} 根杆件、"
                    f"{len(self.model['sections'])} 种截面、{len(self.model['supports'])} 个支座。")
        hints = sum(row.status == INFO for row in self.rows)
        if hints:
            head += f"识别提示 {hints} 条仅供对照原图，不影响生成。"
        if not any(counts.values()):
            return head + "核对完毕，可以生成模型 JSON。"
        parts = [f"{label} {counts[status]} 项" for status, label in
                 ((CONFLICT, "冲突"), (MISSING, "缺失"), (CONFIRM, "需确认"))
                 if counts[status]]
        return head + "仍有" + "、".join(parts) + "须处理。"


def _confidence(source: Any) -> float | None:
    if not isinstance(source, Mapping):
        return None
    value = source.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _settled(item: Mapping[str, Any]) -> bool:
    source = item.get("source")
    return bool(item.get("confirmed")) or (isinstance(source, Mapping)
                                           and bool(source.get("user")))


def _base_status(item: Mapping[str, Any]) -> tuple[str, str]:
    """证据本身的置信度决定的状态。"""
    if _settled(item):
        return OK, ""
    confidence = _confidence(item.get("source"))
    if confidence is not None and confidence < LOW_CONFIDENCE:
        return CONFIRM, f"识别置信度 {confidence:.0%}，请对照原图确认。"
    if str(item.get("check") or "").strip():
        return CONFIRM, str(item["check"]).strip()
    return OK, ""


def _to_m(value: float, factor: float) -> float:
    """图纸单位 → 米。毫米、厘米按整除换算：13 × 0.001 在浮点里不等于 0.013，
    而 13 / 1000 等于，这样识别出的截面与手工按米输入的截面逐位一致。"""
    divisor = 1.0 / factor
    if divisor > 1.0 and abs(divisor - round(divisor)) < 1e-9:
        return value / round(divisor)
    return value * factor


def _from_m(value: float, factor: float) -> float:
    divisor = 1.0 / factor
    if divisor > 1.0 and abs(divisor - round(divisor)) < 1e-9:
        return value * round(divisor)
    return value / factor


def _inches_m(inches: float) -> float:
    """英寸 → 米，按定义 1 in = 254/10000 m，做一次正确舍入的除法。"""
    return inches * 254 / 10000


def _fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value != 0 else "0"


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


class _Length:
    """累加的长度，按写法分开存：不带单位的部分（图纸单位）与英寸部分。

    分开累加让同一种写法的链保持精确（6000+6000 仍是整数，最后一次换算成米），
    只在需要米时合并。"""

    __slots__ = ("plain", "inches")

    def __init__(self, plain: float = 0.0, inches: float = 0.0):
        self.plain, self.inches = plain, inches

    def plus(self, dim: Dimension, count: int = 1) -> _Length:
        step = dim.value * count
        return _Length(self.plain + (0.0 if dim.inches else step),
                       self.inches + (step if dim.inches else 0.0))

    def minus(self, other: _Length) -> _Length:
        return _Length(self.plain - other.plain, self.inches - other.inches)

    def metres(self, factor: float | None) -> float:
        plain = _to_m(self.plain, factor) if self.plain and factor else self.plain
        return plain + (_inches_m(self.inches) if self.inches else 0.0)


def _dim_m(dim: Dimension, value: float, factor: float | None) -> float:
    """读法里的一个数 → 米；不带单位的数在单位未定时暂按 1 计（只作相对核对）。"""
    if dim.inches:
        return _inches_m(value)
    return _to_m(value, factor) if factor else value


def _describe_dim(dim: Dimension) -> str:
    each = dim.show()
    if dim.inches:
        each += f"（{_fmt(_inches_m(dim.value))} m）"
    return f"{dim.count} 段 × {each}" if dim.count > 1 else each


def _length_items(evidence: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """所有按「尺寸单位」换算的条目：轴网间距、总尺寸、外边距与层高。"""
    items: list[Any] = []
    for family in ("x", "y"):
        grid = (evidence.get("grids") or {}).get(family) or {}
        if not isinstance(grid, Mapping):
            continue
        items += list(grid.get("spacings") or []) + list(grid.get("edges") or [])
        if isinstance(grid.get("total"), Mapping):
            items.append(grid["total"])
    items += list(evidence.get("level_heights") or [])
    return [item for item in items if isinstance(item, Mapping)]


def _unit_scan(items: list[Mapping[str, Any]]) -> tuple[bool, str]:
    """(有没有不带单位的数, 一个英尺英寸原文示例)。读不懂的条目不计，等用户改正后再看。"""
    plain, example = False, ""
    for item in items:
        text = item.get("text")
        if item.get("value") is None and (
                text is None or (isinstance(text, str) and not text.strip())):
            continue
        try:
            dim = read_dimension(item)
        except ValueError:
            continue
        if dim.inches:
            example = example or str(text).strip()
        else:
            plain = True
    return plain, example


class _Builder:
    def __init__(self, evidence: Mapping[str, Any]):
        self.ev = evidence
        self.rows: list[CheckRow] = []
        self._samples: list[tuple[CheckRow, Mapping[str, Any], tuple[str, float]]] = []
        self.images = {str(item.get("id")): (index, item) for index, item in
                       enumerate(evidence.get("images") or [], 1)
                       if isinstance(item, Mapping)}
        self.files = {str(item.get("id")): item for item in evidence.get("files") or []
                      if isinstance(item, Mapping)}
        self.f_len: float | None = None
        self.f_elev: float | None = None
        #: 尺寸单位未定、坐标只是相对值的轴网方向；这些方向不往原图上画轴线。
        self.provisional: set[str] = set()

    # ---- 行与来源 ----
    def source_text(self, item: Mapping[str, Any] | None) -> str:
        if not item:
            return "图上未读到"
        source = item.get("source")
        if isinstance(source, Mapping) and source.get("user"):
            return "用户填写"
        raw = item.get("text")
        parts = []
        if isinstance(source, Mapping) and str(source.get("image")) in self.images:
            index, image = self.images[str(source.get("image"))]
            parts.append(f"图 {index}「{image.get('name', '')}」")
        if isinstance(source, Mapping) and str(source.get("file")) in self.files:
            parts.append(f"DXF「{self.files[str(source.get('file'))].get('name', '')}」")
            if source.get("header"):
                parts.append(f"文件头 {source['header']}")
            if source.get("handles"):
                parts.append("实体 " + "、".join(f"#{h}" for h in source["handles"][:4])
                             + ("…" if len(source["handles"]) > 4 else ""))
            if source.get("layer"):
                parts.append(f"图层「{source['layer']}」")
        if isinstance(raw, str) and raw.strip():
            parts.append(f"原文「{raw.strip()}」")
        return " ".join(parts) or "图上未读到原文"

    def add(self, row_id: str, category: str, item_name: str, value: str,
            item: Mapping[str, Any] | None, status: str, message: str = "", *,
            edit: str = "none", options: tuple[str, ...] = (),
            current: str = "", reread: str = "") -> CheckRow:
        source = item.get("source") if item else None
        image = bbox = None
        if isinstance(source, Mapping) and not source.get("user"):
            image = str(source.get("image")) if source.get("image") is not None else None
            bbox = source.get("bbox") if isinstance(source.get("bbox"), list) else None
        row = CheckRow(row_id, category, item_name, value, self.source_text(item), status,
                       message, image, bbox, edit, options, current, reread)
        self.rows.append(row)
        return row

    # ---- 图层用途（矢量入口） ----
    def layers(self) -> None:
        for index, item in enumerate(self.ev.get("layers") or []):
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            summary = str(item.get("summary") or "")
            if role not in LAYER_ROLES:
                status, message = CONFIRM, ("尚未判断用途，暂不读取。如果这个图层上有轴线、"
                                            "梁柱、标注或说明，请选择用途；否则确认即按不使用处理。")
                value = "未判断"
            elif _settled(item):
                status, message, value = OK, "", LAYER_ROLES[role]
            else:
                status, value = CONFIRM, LAYER_ROLES[role]
                message = str(item.get("check") or "") or "图层用途须经确认。"
            self.add(f"layers.{index}", "图层", f"图层「{item.get('name')}」",
                     f"{value}（{summary}）" if summary else value, item, status, message,
                     edit="choice", options=tuple(LAYER_CHOICES),
                     current=LAYER_ROLES.get(role, ""))

    # ---- 单位 ----
    def unit(self, key: str, label: str, needed: bool, inch_example: str = "") -> float | None:
        """图上的单位说明。所有尺寸都自带英尺英寸单位时 ``needed`` 为假，不要求说明；
        说明与英尺英寸写法并存时报冲突，由用户对照原图选择。"""
        item = (self.ev.get("units") or {}).get(key) or {}
        value = item.get("value")
        factor = length_factor(value) if value in LENGTH_UNITS else None
        row_id = f"units.{key}"
        if factor is None:
            if needed:
                self.add(row_id, "单位", label, "—", item or None, MISSING,
                         f"图上没有读到{label}说明；请按图纸选择，不按惯例默认。",
                         edit="choice", options=UNIT_CHOICES)
            elif inch_example:
                self.add(row_id, "单位", label, "英尺英寸（写法自带）", None, OK,
                         f"尺寸都写明了英尺英寸（如「{inch_example}」），按 1 in = 25.4 mm "
                         "换算，不需要单位说明。", edit="choice", options=UNIT_CHOICES)
            return None
        status, message = _base_status(item)
        if status == OK and not _settled(item) and not str(item.get("text") or "").strip():
            status, message = CONFIRM, "图上未见单位说明原文，请确认。"
        if inch_example and not _settled(item):
            status = CONFLICT
            message = (f"单位说明为 {value}，但有尺寸写成英尺英寸（如「{inch_example}」）。"
                       "英尺英寸写法按自身单位换算，不受此项影响；请对照原图选择不带单位的"
                       "数字所用的单位。")
        self.add(row_id, "单位", label, str(value), item, status, message,
                 edit="choice", options=UNIT_CHOICES, current=str(value))
        return factor

    # ---- 一维尺寸链：轴网与层高共用 ----
    def chain(self, labels: list[str], items: list[Mapping[str, Any]], *, prefix: str,
              category: str, noun: str, pair_name, missing_message: str,
              name_gaps: bool = False):
        """展开一组间距（含重复写法）并逐段累加。

        返回 (展开结果, 各标签累加长度——有缺段或坏段时为 None, 各段)。读不懂、
        段数不符、缺段都进核对表；坏段只报一次，不再另报缺失。"""
        chain = expand_chain(labels, items, name_gaps=name_gaps, noun=noun)
        offsets: dict[str, _Length] | None = {chain.labels[0]: _Length()}
        rows: dict[int, CheckRow] = {}
        segments: list[tuple[CheckRow, Mapping[str, Any], str, str, Dimension]] = []
        for a, b in zip(chain.labels, chain.labels[1:], strict=False):
            link = chain.links.get((a, b))
            if link is None:
                if (a, b) not in chain.blocked:
                    self.add(f"{prefix}+{a}:{b}", category, pair_name(a, b), "—", None,
                             MISSING, missing_message, edit="text")
                offsets = None
                continue
            item, dim = items[link.index], link.dimension
            row = rows.get(link.index)
            if row is None:
                name = pair_name(str(item.get("from")).strip(), str(item.get("to")).strip())
                current = str(item.get("text") or "")
                if dim.value <= 0:
                    row = self.add(f"{prefix}.{link.index}", category, name, dim.show(), item,
                                   CONFLICT, f"「{current}」不是正的间距；请照图改正。",
                                   edit="text", current=current)
                else:
                    status, message = _base_status(item)
                    if dim.count > 1 and not message:
                        message = f"重复写法，已逐段展开为 {dim.count} 段。"
                        if link.index in chain.generated.values():
                            message += "中间各层图上没有层名，已按位置自动命名。"
                    row = self.add(f"{prefix}.{link.index}", category, name,
                                   _describe_dim(dim), item, status, message, edit="text",
                                   current=current)
                rows[link.index] = row
            if dim.value <= 0:
                offsets = None
                continue
            segments.append((row, item, a, b, dim))
            if offsets is not None:
                offsets[b] = offsets[a].plus(dim)
        for index, (message, reread) in chain.problems.items():
            item = items[index]
            a, b = str(item.get("from") or "").strip(), str(item.get("to") or "").strip()
            self.add(f"{prefix}.{index}", category, pair_name(a, b),
                     str(item.get("text") or "—"), item, CONFLICT, message, edit="text",
                     current=str(item.get("text") or ""), reread=reread)
        return chain, offsets, segments

    def check_span(self, row_id: str, category: str, item_name: str,
                   item: Mapping[str, Any], expected: _Length, factor: float | None, *,
                   edges: _Length | None = None, provisional: bool = False) -> None:
        """一条跨多段的尺寸（总尺寸、跨轴尺寸、总高）与分段之和核对。"""
        current = str(item.get("text") or "")
        try:
            dim = read_dimension(item)
        except ValueError as exc:
            self.add(row_id, category, item_name, "—", item, CONFLICT, str(exc),
                     edit="text", current=current)
            return
        if provisional:
            self.add(row_id, category, item_name, _describe_dim(dim), item, CONFIRM,
                     "尺寸单位未定，选定单位后再与分段之和核对。", edit="text", current=current)
            return
        value = _dim_m(dim, dim.total, factor)
        expected_m = expected.metres(factor)
        outer = edges is not None and _same(value, expected_m + edges.metres(factor))
        if _same(value, expected_m) or outer:
            status, message = _base_status(item)
            note = ("这是量到外边缘的总尺寸：分段间距之和加两端外边距与它一致；"
                    "不作为轴线总尺寸，坐标仍由分段间距累加。" if outer
                    else "与分段间距之和一致。")
            self.add(row_id, category, item_name, _describe_dim(dim), item, status,
                     message or note, edit="text", current=current)
            return

        def show(metres: float) -> str:
            if dim.inches:
                return dim.show(metres * 10000 / 254)
            return dim.show(_from_m(metres, factor) if factor else metres)
        extra = "（计入两端外边距也不符）" if edges is not None else ""
        self.add(row_id, category, item_name, _describe_dim(dim), item, CONFLICT,
                 f"分段间距之和为 {show(expected_m)}，与标注 {show(value)} 不符{extra}；"
                 "请对照原图改正读错的那一段或这条尺寸。", edit="text", current=current,
                 reread=f"{item_name}原文「{item.get('text')}」与各段间距原文之和对不上")

    # ---- 轴网 ----
    def grid(self, family: str, used: bool) -> tuple[list[str], dict[str, float]]:
        """返回轴号顺序与各轴坐标（米，首轴为 0；尺寸单位未定时为相对值）。"""
        grid = (self.ev.get("grids") or {}).get(family) or {}
        name = "x 向（数字轴）" if family == "x" else "y 向（字母轴）"
        labels = [str(axis["label"]).strip() for axis in grid.get("axes") or []
                  if isinstance(axis, Mapping) and str(axis.get("label") or "").strip()]
        if not labels:
            if used:
                self.add(f"grids.{family}.axes", "轴网", f"{name}轴号", "—", None,
                         CONFLICT, f"杆件引用了{name}轴线，但图上没有读到这一组轴号；"
                         "请重新识别或补充含轴号的图纸。")
            return [], {}
        spacings = [item if isinstance(item, Mapping) else {}
                    for item in grid.get("spacings") or []]
        chain, offsets, segments = self.chain(
            labels, spacings, prefix=f"grids.{family}.spacings", category="轴网", noun="轴号",
            pair_name=lambda a, b: f"轴 {a}→{b} 间距",
            missing_message="图上没有读到这段间距；请照图填写（单位同尺寸单位），不从图上量取。")
        factor = self.f_len
        kinds = {dim.inches for *_, dim in segments}
        edges: _Length | None = None
        for index, item in enumerate(grid.get("edges") or []):
            if not isinstance(item, Mapping):
                continue
            row_id, item_name = f"grids.{family}.edges.{index}", f"{name}外边距"
            current = str(item.get("text") or "")
            try:
                dim = read_dimension(item)
            except ValueError as exc:
                self.add(row_id, "轴网", item_name, "—", item, CONFLICT, str(exc),
                         edit="text", current=current)
                continue
            kinds.add(dim.inches)
            status, message = _base_status(item)
            self.add(row_id, "轴网", item_name, _describe_dim(dim), item, status,
                     message or "首末轴线到外边缘的尺寸，只用来核对量到外边缘的总尺寸。",
                     edit="text", current=current)
            edges = (edges or _Length()).plus(dim, dim.count)
        if factor is None and False in kinds:
            self.provisional.add(family)
        if offsets is None:
            return labels, {}
        provisional = factor is None and kinds == {True, False}
        self._pixel_alarm(family, grid, [(row, item, a, b, _dim_m(dim, dim.value, factor))
                                         for row, item, a, b, dim in segments])
        position = {label: index for index, label in enumerate(labels)}
        for index, a, b, _dim in chain.spans:
            low, high = sorted((a, b), key=position.get)
            self.check_span(f"grids.{family}.spacings.{index}", "轴网", f"轴 {a}→{b} 尺寸",
                            spacings[index], offsets[high].minus(offsets[low]), factor,
                            provisional=provisional)
        total = grid.get("total")
        if isinstance(total, Mapping) and str(total.get("text") or "").strip():
            self.check_span(f"grids.{family}.total", "轴网", f"{name}总尺寸", total,
                            offsets[labels[-1]], factor, edges=edges,
                            provisional=provisional)
        return labels, {label: offsets[label].metres(factor) for label in labels}

    def _pixel_alarm(self, family: str, grid: Mapping[str, Any],
                     rows: list[tuple[CheckRow, Mapping[str, Any], str, str, float]]) -> None:
        """同一张图上各段「像素长/标注值」应大致相同；离群的段多半抄错了。"""
        axes = {str(axis.get("label")).strip(): axis for axis in grid.get("axes") or []
                if isinstance(axis, Mapping)}
        for row, item, a, b, value in rows:
            ratio = self._pixel_ratio(family, axes.get(a), axes.get(b), value)
            if ratio is not None:
                self._samples.append((row, item, ratio))

    def _pixel_ratio(self, family, axis_a, axis_b, value) -> tuple[str, float] | None:
        if not (isinstance(axis_a, Mapping) and isinstance(axis_b, Mapping)):
            return None
        src_a, src_b = axis_a.get("source") or {}, axis_b.get("source") or {}
        image = src_a.get("image") if isinstance(src_a, Mapping) else None
        if image is None or not isinstance(src_b, Mapping) or src_b.get("image") != image:
            return None
        pos_a, pos_b = axis_a.get("pos"), axis_b.get("pos")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in (pos_a, pos_b)):
            return None
        meta = self.images.get(str(image))
        if meta is None:
            return None
        size = meta[1].get("width" if family == "x" else "height")
        if not isinstance(size, (int, float)) or size <= 0:
            return None
        pixels = abs(float(pos_b) - float(pos_a)) * float(size)
        if pixels <= 0:
            return None
        return str(image), pixels / value

    def finish_pixel_alarm(self) -> None:
        by_image: dict[str, list[tuple[CheckRow, Mapping[str, Any], float]]] = {}
        for row, item, (image, ratio) in self._samples:
            by_image.setdefault(image, []).append((row, item, ratio))
        for samples in by_image.values():
            if len(samples) < 3:
                continue
            middle = median(ratio for _, _, ratio in samples)
            for row, item, ratio in samples:
                if row.status != OK or _settled(item):
                    continue
                deviation = ratio / middle - 1.0
                if abs(deviation) > PIXEL_SCALE_ALARM:
                    row.status = CONFIRM
                    row.message = (f"图上这段的长度比例与同图其他间距相差 {deviation:+.0%}，"
                                   "可能抄错了数字；请对照原图确认或改正。")

    # ---- 标高 ----
    def _elevation(self, item: Mapping[str, Any]) -> tuple[float | None, str]:
        """一条标高原文 → (米, 读不懂的说明)。英尺英寸自带单位，其余按标高单位。"""
        try:
            dim = read_dimension(item)
        except ValueError:
            return None, f"读不懂标高「{item.get('text')}」；请改为数字，如 3.600。"
        if dim.count != 1:
            return None, f"「{item.get('text')}」是重复写法，不是标高；请照图填写这一层的标高。"
        return _dim_m(dim, dim.value, self.f_elev), ""

    def levels(self, needed: bool) -> dict[str, float]:
        """各层标高（米）。图上有标高就用标高；立面图只给层高链时按层高累加。"""
        items = [(index, item) for index, item in enumerate(self.ev.get("levels") or [])
                 if isinstance(item, Mapping)]
        if not items:
            if needed:
                self.add("levels+", "标高", "各层标高", "—", None, MISSING,
                         "图上没有读到标高；请照图填写各层标高，用逗号分隔，"
                         "如「±0.000, 3.600, 7.200」（单位按标高单位）。", edit="text")
            return {}
        heights = [item if isinstance(item, Mapping) else {}
                   for item in self.ev.get("level_heights") or []]
        if heights:
            return self._level_chain(items, heights)
        result: dict[str, float] = {}
        seen_values: list[tuple[float, str]] = []
        for index, item in items:
            name = str(item.get("name") or "").strip()
            row_id, label = f"levels.{index}", f"标高 {name}"
            current = str(item.get("text") or "")
            if not current.strip():
                self.add(row_id, "标高", label, "—", item, MISSING,
                         "图上没有读到这一层的标高；请照图填写，如 3.600。", edit="text")
                continue
            value, problem = self._elevation(item)
            if value is None:
                self.add(row_id, "标高", label, "—", item, CONFLICT, problem,
                         edit="text", current=current)
                continue
            other = name if name in result else next(
                (key for seen, key in seen_values if _same(seen, value)), None)
            if other is not None:
                self.add(row_id, "标高", label, _fmt(value), item, CONFLICT,
                         f"与标高 {other} 重复；请改正。", edit="text", current=current)
                continue
            status, message = _base_status(item)
            self.add(row_id, "标高", label, _fmt(value), item, status, message,
                     edit="text", current=current)
            result[name] = value
            seen_values.append((value, name))
        return result

    def _level_chain(self, items, heights) -> dict[str, float]:
        """层高链：levels 从下到上排列，level_heights 给相邻层（或重复写法跨若干层）
        的层高。有标高原文的层作基准并互相核对；一条标高都没有时最低层按 ±0.000 计，
        这是约定而非证据，所以要用户确认。"""
        by_name: dict[str, tuple[int, Mapping[str, Any]]] = {}
        for index, item in items:
            name = str(item.get("name") or "").strip()
            if name in by_name:
                self.add(f"levels.{index}", "标高", f"标高 {name}", "—", item, CONFLICT,
                         f"层名 {name} 重复；请改正或删除重复的一层。", edit="text",
                         current=str(item.get("text") or ""))
                continue
            by_name[name] = (index, item)
        chain, offsets, segments = self.chain(
            list(by_name), heights, prefix="level_heights", category="标高", noun="层名",
            pair_name=lambda a, b: f"层高 {a}→{b}", name_gaps=True,
            missing_message="图上没有读到这一层的层高；请照图填写（单位同尺寸单位），"
                            "或在各层标高里填写这一层的标高。")
        for index, a, b, _dim in chain.spans:
            if offsets is not None:
                low, high = sorted((a, b), key=chain.labels.index)
                self.check_span(f"level_heights.{index}", "标高", f"层高 {a}→{b} 合计",
                                heights[index], offsets[high].minus(offsets[low]), self.f_len)
        if offsets is None:
            return {}
        rel = {label: offsets[label].metres(self.f_len) for label in chain.labels}
        anchors: list[tuple[str, int, Mapping[str, Any], float]] = []
        for label in chain.labels:
            if label not in by_name:
                continue
            index, item = by_name[label]
            if not str(item.get("text") or "").strip():
                continue
            value, problem = self._elevation(item)
            if value is None:
                self.add(f"levels.{index}", "标高", f"标高 {label}", "—", item, CONFLICT,
                         problem, edit="text", current=str(item.get("text")))
                return {}
            anchors.append((label, index, item, value))
        base_label = chain.labels[0]
        if anchors:
            label, _index, _item, value = anchors[0]
            base = value - rel[label]
        else:
            index, item = by_name[base_label]
            base = 0.0
            status, message = (OK, "") if _settled(item) else (
                CONFIRM, "图上只给层高、没有标高；最低层按 ±0.000 计，其余各层由层高累加。"
                         "请确认，或照图填写最低层的标高。")
            self.add(f"levels.{index}", "标高", f"标高 {base_label}", "0（按 ±0.000 计）",
                     item, status, message, edit="text", current="")
        provisional = (
            (self.f_len is None and any(not dim.inches for *_, dim in segments))
            or (self.f_elev is None and any(not read_dimension(item).inches
                                            for _l, _i, item, _v in anchors)))
        anchored = {label for label, *_ in anchors}
        for label, index, item, value in anchors:
            expected = base + rel[label]
            current = str(item.get("text"))
            if label == anchors[0][0] or _same(value, expected):
                status, message = _base_status(item)
                self.add(f"levels.{index}", "标高", f"标高 {label}", _fmt(value), item, status,
                         message, edit="text", current=current)
            elif provisional:
                self.add(f"levels.{index}", "标高", f"标高 {label}", _fmt(value), item, CONFIRM,
                         "单位未定，选定单位后再与层高累加结果核对。", edit="text",
                         current=current)
            else:
                self.add(f"levels.{index}", "标高", f"标高 {label}", _fmt(value), item,
                         CONFLICT, f"由层高累加得 {_fmt(expected)} m，与标高原文 {_fmt(value)} m "
                         "不符；请对照原图改正这一层的标高或相关层高。", edit="text",
                         current=current,
                         reread=f"标高 {label} 原文「{current}」与层高原文累加结果对不上")
        for label in chain.labels:
            if label in anchored or label not in by_name or (label == base_label
                                                             and not anchors):
                continue
            index, item = by_name[label]
            self.add(f"levels.{index}", "标高", f"标高 {label}",
                     f"{_fmt(base + rel[label])} m（由层高累加）", item, OK,
                     "图上没有这一层的标高，由层高累加得到。", edit="text", current="")
        return {label: base + rel[label] for label in chain.labels}


def _level_key(levels: Mapping[str, float], evidence_levels: list, ref: Any) -> str | None:
    name = str(ref or "").strip()
    if name in levels:
        return name
    for item in evidence_levels:
        if isinstance(item, Mapping) and str(item.get("text") or "").strip() == name \
                and str(item.get("name") or "").strip() in levels:
            return str(item.get("name")).strip()
    return None


def assemble(evidence: Mapping[str, Any]) -> Assembly:
    """证据 → 核对表 +（无缺失、无冲突时）模型 JSON。"""
    b = _Builder(evidence)
    active = [(index, item) for index, item in enumerate(evidence.get("members") or [])
              if isinstance(item, Mapping) and not item.get("deleted")]

    b.layers()
    # 单位按需要求：所有尺寸都写成英尺英寸（自带单位）且没有截面规格时，不要求
    # 尺寸单位说明；有一个不带单位的数就要。
    length_items = _length_items(evidence)
    plain_length, inch_length = _unit_scan(length_items)
    has_sections = any(isinstance(item, Mapping) for item in evidence.get("sections") or [])
    len_needed = plain_length or has_sections or not length_items
    b.f_len = b.unit("length", "尺寸单位", len_needed, inch_length)
    uses_y = any(item.get("kind") == "beam" and item.get("along")
                 or item.get("kind") == "column" and len(item.get("at") or []) == 2
                 for _, item in active)
    x_labels, x_coords = b.grid("x", used=bool(active))
    y_labels, y_coords = b.grid("y", used=uses_y)
    b.finish_pixel_alarm()
    if set(x_labels) & set(y_labels):
        b.add("grids", "轴网", "轴号", "、".join(sorted(set(x_labels) & set(y_labels))),
              None, CONFLICT, "这些轴号同时出现在两个方向，无法确定杆件位置；请重新识别。")
    levels_ev = [item for item in evidence.get("levels") or [] if isinstance(item, Mapping)]
    plain_elev, inch_elev = _unit_scan(levels_ev)
    elev_needed = plain_elev or (not levels_ev and bool(active))
    if levels_ev or active:
        b.f_elev = b.unit("elevation", "标高单位", elev_needed, inch_elev)
    levels = b.levels(needed=bool(active))
    axes = {family: coords for family, coords in (("x", x_coords), ("y", y_coords))
            if coords and family not in b.provisional}

    # ---- 杆件：逐跨、逐层拆分 ----
    x_set, y_set = set(x_labels), set(y_labels)
    segments: dict[frozenset, dict[str, Any]] = {}
    member_paths: dict[frozenset, str] = {}
    used_labels: list[str] = []
    level_order = sorted(levels, key=lambda key: levels[key])
    for index, item in active:
        kind = item.get("kind")
        label = str(item.get("label") or "").strip() or UNNUMBERED.get(kind, f"杆件{index + 1}")
        row_id = f"members.{index}"
        try:
            pieces = _member_pieces(item, kind, x_labels, y_labels, x_set, y_set,
                                    levels, levels_ev, level_order)
        except ValueError as exc:
            b.add(row_id, "杆件", label, _describe_member(item), item, CONFLICT, str(exc),
                  edit="choice", options=(DELETE_CHOICE,), reread=f"杆件 {label}：{exc}")
            continue
        overlap = None
        for key, piece in pieces:
            other = segments.get(key)
            if other is not None and other["label"] != label:
                overlap = other["label"]
                break
        if overlap:
            b.add(row_id, "杆件", label, _describe_member(item), item, CONFLICT,
                  f"与 {overlap} 位置重叠；请删除读错的那一条。",
                  edit="choice", options=(DELETE_CHOICE,),
                  reread=f"杆件 {label} 与 {overlap} 落在同一位置")
            continue
        for key, piece in pieces:
            segments.setdefault(key, {"label": label, "kind": kind, "ends": piece})
            member_paths.setdefault(key, row_id)
        if label not in used_labels:
            used_labels.append(label)
        status, message = _base_status(item)
        b.add(row_id, "杆件", label, _describe_member(item) + f"（{len(pieces)} 段）",
              item, status, message, edit="choice", options=(DELETE_CHOICE,))
    if not active:
        b.add("members", "杆件", "杆件", "—", None, CONFLICT,
              "图上没有读到任何杆件；请上传含梁柱布置的平面图或立面图。")

    # ---- 截面 ----
    sec_items = {}
    for index, item in enumerate(evidence.get("sections") or []):
        if isinstance(item, Mapping) and str(item.get("label") or "").strip():
            sec_items.setdefault(str(item["label"]).strip(), (index, item))
    built_sections: dict[str, dict[str, Any]] = {}
    label_section: dict[str, str] = {}
    for label in used_labels:
        if label not in sec_items:
            message = ("截面表里没有读到这个编号的规格；请照图填写，如 HN400×200×8×13。"
                       if label not in UNNUMBERED.values() else
                       f"图上这些杆件没有编号，查不到截面表；请照图填写规格，如 HN400×200×8×13。"
                       f"所有{label}共用这一规格，规格不同的请导入后在模型中修改。")
            b.add(f"sections+{label}", "截面", f"{label} 规格", "—", None, MISSING, message,
                  edit="text")
    for label, (index, item) in sec_items.items():
        row_id = f"sections.{index}"
        text = str(item.get("text") or "")
        try:
            shape = parse_designation(text)
        except ValueError as exc:
            b.add(row_id, "截面", f"{label} 规格", "—", item, CONFLICT, str(exc),
                  edit="text", current=text)
            continue
        value = _describe_shape(shape)
        section = None
        if b.f_len is not None:
            try:
                section = _build_section(shape, b.f_len)
            except ValueError as exc:
                b.add(row_id, "截面", f"{label} 规格", value, item, CONFLICT,
                      f"尺寸不成立：{exc}", edit="text", current=text)
                continue
            value += f"；A={section['A'] * 1e4:.2f} cm²"
        status, message = _base_status(item)
        if status == OK and shape["convention"] and not _settled(item):
            status, message = CONFIRM, f"{shape['convention']}，请确认。"
        notes = [message] if message else []
        if shape["type"] == "i":
            notes.append("规格不含根部圆角 r，按直角轮廓计算；需要时导入后在「创建截面」中补 r。")
        if label not in used_labels:
            notes.append("图上没有读到这个编号的杆件。")
        b.add(row_id, "截面", f"{label} 规格", value, item, status, " ".join(notes),
              edit="text", current=text)
        if section is not None:
            built_sections.setdefault(section["name"], section)
            label_section[label] = section["name"]

    # ---- 材料与支座 ----
    material_item = evidence.get("material") if isinstance(evidence.get("material"),
                                                           Mapping) else None
    material = None
    try:
        material = parse_material((material_item or {}).get("text"))
    except ValueError as exc:
        status = CONFLICT if material_item and str(material_item.get("text") or "").strip() \
            else MISSING
        b.add("material", "材料", "材料", "—", material_item, status, str(exc), edit="text",
              current=str((material_item or {}).get("text") or ""))
    if material is not None:
        status, message = _base_status(material_item)
        if status == OK and not _settled(material_item):
            status, message = CONFIRM, "弹性模量取自规范表，请确认牌号与参数。"
        b.add("material", "材料", "材料",
              f"{material['name']}：E={material['E'] / 1e9:g} GPa，ν={material['nu']:g}"
              f"（{material['basis']}）", material_item, status, message, edit="text",
              current=str(material_item.get("text") or ""))

    support_item = evidence.get("support") if isinstance(evidence.get("support"),
                                                         Mapping) else None
    support_kind = (support_item or {}).get("kind")
    if support_kind not in SUPPORT_FIX:
        b.add("support", "支座", "柱底支座", "—", support_item, MISSING,
              "图上没有读到柱脚做法说明；请按图纸选择，不设默认。",
              edit="choice", options=tuple(SUPPORT_CHOICES))
        support_kind = None
    else:
        status, message = _base_status(support_item)
        b.add("support", "支座", "柱底支座", f"所有最低标高柱底 {SUPPORT_NAMES[support_kind]}",
              support_item, status, message, edit="choice",
              options=tuple(SUPPORT_CHOICES), current=SUPPORT_NAMES[support_kind])

    # 识别提示不逐条确认：它涉及的每个数都已在自己那一行按置信度、缺失或冲突处理，
    # 这里只列出来供对照原图（真实验收中提示多时逐条点击过于繁琐）。
    for index, note in enumerate(evidence.get("notes") or []):
        if isinstance(note, Mapping) and str(note.get("text") or "").strip():
            b.add(f"notes.{index}", "图纸", "识别提示", str(note["text"]).strip(), note, INFO,
                  "识别时看不清或拿不准的内容，仅供对照原图；相关读数已在各自的行里核对，"
                  "不影响生成。")

    blocked = any(row.blocking for row in b.rows) or material is None \
        or support_kind is None or (len_needed and b.f_len is None) \
        or (elev_needed and b.f_elev is None)
    if blocked:
        return Assembly(b.rows, None, axes=axes)

    model, provenance = _build_model(
        evidence, segments, member_paths, x_coords, y_coords, levels,
        built_sections, label_section, material, support_kind)
    if not model["supports"]:
        row = next(row for row in b.rows if row.id == "support")
        row.status = CONFLICT
        row.message = "最低标高处没有柱底节点，无法设置支座；请检查柱的起止标高。"
        return Assembly(b.rows, None, axes=axes)
    errors = [str(e) for e in validate_payload(model) if not str(e).startswith("[提示]")]
    if errors:
        b.add("model", "模型", "模型校验", "未通过", None, CONFLICT, "；".join(errors[:5]))
        return Assembly(b.rows, None, axes=axes)
    return Assembly(b.rows, model, provenance, axes)


def _member_pieces(item, kind, x_labels, y_labels, x_set, y_set, levels, levels_ev,
                   level_order) -> list[tuple[frozenset, tuple[tuple, tuple]]]:
    """一条杆件证据 → 若干段，每段两端为 (x轴, y轴, 标高名)。"""
    def family(label):
        label = str(label).strip()
        if label in x_set:
            return "x", label
        if label in y_set:
            return "y", label
        raise ValueError(f"引用了图上没有的轴号「{label}」。")

    pieces = []
    if kind == "beam":
        a, b = family(item.get("from")), family(item.get("to"))
        if a[0] != b[0]:
            raise ValueError("梁的起止轴号不在同一方向。")
        run = a[0]
        order = x_labels if run == "x" else y_labels
        along = str(item.get("along") or "").strip()
        cross = None
        if along:
            cross = family(along)
            if cross[0] == run:
                raise ValueError("梁所在轴线与起止轴号同向。")
        elif (y_labels if run == "x" else x_labels):
            raise ValueError("梁缺少所在轴线（along）。")
        lo, hi = sorted((order.index(a[1]), order.index(b[1])))
        if lo == hi:
            raise ValueError("梁的起止轴号相同。")
        names = []
        for ref in item.get("levels") or []:
            name = _level_key(levels, levels_ev, ref)
            if name is None:
                raise ValueError(f"引用了没有读到的标高「{ref}」。")
            names.append(name)
        for name in names:
            for k in range(lo, hi):
                ends = []
                for label in (order[k], order[k + 1]):
                    xl, yl = (label, None) if run == "x" else (None, label)
                    if cross is not None:
                        xl, yl = (xl, cross[1]) if run == "x" else (cross[1], yl)
                    ends.append((xl, yl, name))
                pieces.append((frozenset(ends), tuple(ends)))
        return pieces
    at = [family(label) for label in item.get("at") or []]
    xl = next((label for fam, label in at if fam == "x"), None)
    yl = next((label for fam, label in at if fam == "y"), None)
    if len({fam for fam, _ in at}) != len(at):
        raise ValueError("柱的位置给了两个同向轴号。")
    if (x_labels and xl is None) or (y_labels and yl is None):
        raise ValueError("柱的位置缺少一个方向的轴号。")
    bottom = _level_key(levels, levels_ev, item.get("from_level"))
    top = _level_key(levels, levels_ev, item.get("to_level"))
    if bottom is None or top is None:
        raise ValueError("柱引用了没有读到的标高。")
    lo, hi = sorted((level_order.index(bottom), level_order.index(top)))
    if lo == hi:
        raise ValueError("柱的起止标高相同。")
    for k in range(lo, hi):
        ends = ((xl, yl, level_order[k]), (xl, yl, level_order[k + 1]))
        pieces.append((frozenset(ends), ends))
    return pieces


def _describe_member(item: Mapping[str, Any]) -> str:
    if item.get("kind") == "beam":
        along = str(item.get("along") or "").strip()
        where = f"{along} 轴 " if along else ""
        levels = "、".join(str(v) for v in item.get("levels") or [])
        return f"梁 {where}{item.get('from')}→{item.get('to')}，标高 {levels}"
    at = "/".join(str(v) for v in item.get("at") or [])
    return f"柱 {at}，{item.get('from_level')}→{item.get('to_level')}"


def _describe_shape(shape: Mapping[str, Any]) -> str:
    dims = "、".join(f"{key}={_fmt(value)}" for key, value in shape["dimensions"].items())
    return f"{shape['name']} → {_SECTION_KIND_NAMES[shape['type']]} {dims}"


def _build_section(shape: Mapping[str, Any], factor: float) -> dict[str, Any]:
    d = {key: _to_m(value, factor) for key, value in shape["dimensions"].items()}
    name = shape["name"]
    if shape["type"] == "i":
        return section_builders.i_section(name, d["h"], d["b"], d["tw"], d["tf"])
    if shape["type"] == "tube":
        return section_builders.circular_tube(name, d["d"], d["t"])
    if shape["type"] == "circle":
        return section_builders.solid_circle(name, d["d"])
    return section_builders.rectangle(name, d["b"], d["h"])


def _build_model(evidence, segments, member_paths, x_coords, y_coords, levels,
                 built_sections, label_section, material, support_kind):
    """坐标（x_coords、y_coords、levels）均已是米。"""
    def point(end):
        xl, yl, level = end
        return (round(x_coords[xl] if xl is not None else 0.0, _COINCIDENT_DIGITS),
                round(y_coords[yl] if yl is not None else 0.0, _COINCIDENT_DIGITS),
                round(levels[level], _COINCIDENT_DIGITS))

    coords = sorted({point(end) for seg in segments.values() for end in seg["ends"]},
                    key=lambda p: (p[2], p[1], p[0]))
    node_id = {p: index for index, p in enumerate(coords, 1)}
    nodes = [{"id": node_id[p], "x": p[0], "y": p[1], "z": p[2]} for p in coords]
    node_sources = {}
    for seg in segments.values():
        for end in seg["ends"]:
            xl, yl, level = end
            node_sources.setdefault(node_id[point(end)], {
                "x": f"轴 {xl}" if xl else "无 x 向轴线（取 0）",
                "y": f"轴 {yl}" if yl else "无 y 向轴线（取 0）",
                "z": f"标高 {level}"})

    ordered = sorted(segments.items(), key=lambda kv: (
        0 if kv[1]["kind"] == "column" else 1,
        sorted(node_id[point(end)] for end in kv[1]["ends"])))
    members, sets, member_sources = [], {}, {}
    for key, seg in ordered:
        i, j = (node_id[point(end)] for end in seg["ends"])
        member = {"id": len(members) + 1, "i": i, "j": j,
                  "section": label_section[seg["label"]], "material": material["name"]}
        if seg["kind"] == "column":
            member["ref_vector"] = list(_COLUMN_REF)
        members.append(member)
        sets.setdefault(seg["label"], {"members": [], "note": "图纸杆件编号"})
        sets[seg["label"]]["members"].append(member["id"])
        member_sources[member["id"]] = member_paths[key]

    lowest = coords[0][2]
    base_nodes = sorted({min((node_id[point(end)] for end in seg["ends"]),
                             key=lambda n: nodes[n - 1]["z"])
                         for seg in segments.values() if seg["kind"] == "column"})
    supports = [{"node": n, "fix": list(SUPPORT_FIX[support_kind])}
                for n in base_nodes if nodes[n - 1]["z"] == lowest]
    used = {m["section"] for m in members}
    model = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "units": "N-m-Pa",
        "materials": [{"name": material["name"], "E": material["E"], "nu": material["nu"]}],
        "sections": [deepcopy(built_sections[name]) for name in built_sections if name in used],
        "nodes": nodes,
        "members": members,
        "supports": supports,
        "sets": sets,
    }
    provenance = {
        "format": EVIDENCE_FORMAT,
        "nodes": node_sources,
        "members": member_sources,
        "sections": {label: name for label, name in label_section.items()},
        "material": material["basis"],
        "support": SUPPORT_NAMES[support_kind],
    }
    return model, provenance


# ---- 用户处理写回证据 ----------------------------------------------------------

def _walk(container: Any, parts: list[str]) -> Any:
    for part in parts:
        container = container[int(part)] if isinstance(container, list) else container[part]
    return container


def resolve(evidence: Mapping[str, Any], row_id: str, value: str | None = None) -> dict:
    """把用户对某一行的处理写回证据，返回新证据（原证据不改）。

    ``value`` 为 None 表示「确认原值」；否则是用户填写或选择的值，来源记为用户，
    原来的出处保存在 ``source_before`` 里以便追溯。
    """
    ev = deepcopy(dict(evidence))
    user = {"user": True}
    text = None if value is None else str(value).strip()
    if text == "":
        raise ValueError("请填写内容后再应用。")

    def settle(item: dict, **fields) -> None:
        if fields:
            if item.get("source") is not None and not item.get("source", {}).get("user"):
                item.setdefault("source_before", item["source"])
            if "value" in item:     # 用户改写后以文字为准，实测值留作追溯
                item.setdefault("value_before", item.pop("value"))
            item.pop("check", None)
            item.update(fields)
            item["source"] = dict(user)
        item["confirmed"] = True

    if row_id.startswith("units."):
        key = row_id.split(".", 1)[1]
        units = ev.setdefault("units", {})
        if text is None:
            if not units.get(key):
                raise ValueError("这一项需要选择单位。")
            settle(units[key])
        else:
            if text not in LENGTH_UNITS:
                raise ValueError(f"单位只能是 {'、'.join(LENGTH_UNITS)}。")
            units[key] = dict(units.get(key) or {})
            settle(units[key], value=text)
        return ev
    if "+" in row_id:
        if text is None:
            raise ValueError("这一项图上没有读到，需要填写。")
        head, tail = row_id.split("+", 1)
        if head.startswith("grids.") and head.endswith(".spacings"):
            family = head.split(".")[1]
            a, b = tail.split(":", 1)
            grid = ev.setdefault("grids", {}).setdefault(family, {})
            grid.setdefault("spacings", []).append(
                {"from": a, "to": b, "text": text, "source": dict(user), "confirmed": True})
        elif head == "level_heights":
            a, b = tail.split(":", 1)
            ev.setdefault("level_heights", []).append(
                {"from": a, "to": b, "text": text, "source": dict(user), "confirmed": True})
        elif head == "sections":
            ev.setdefault("sections", []).append(
                {"label": tail, "text": text, "source": dict(user), "confirmed": True})
        elif head == "levels":
            entries = [part.strip() for part in text.replace("，", ",").split(",")
                       if part.strip()]
            if len(entries) < 2:
                raise ValueError("至少需要两个标高，用逗号分隔。")
            ev["levels"] = [{"name": entry, "text": entry, "source": dict(user),
                             "confirmed": True} for entry in entries]
        else:
            raise ValueError(f"无法处理的核对项 {row_id}。")
        return ev
    if row_id == "material":
        item = dict(ev.get("material") or {})
        if text is None:
            if not item:
                raise ValueError("这一项需要填写材料。")
            settle(item)
        else:
            settle(item, text=text)
        ev["material"] = item
        return ev
    if row_id == "support":
        item = dict(ev.get("support") or {})
        if text is None:
            if item.get("kind") not in SUPPORT_FIX:
                raise ValueError("这一项需要选择支座类型。")
            settle(item)
        else:
            if text not in SUPPORT_CHOICES:
                raise ValueError(f"支座只能选 {'、'.join(SUPPORT_CHOICES)}。")
            settle(item, kind=SUPPORT_CHOICES[text], text=text)
        ev["support"] = item
        return ev
    parts = row_id.split(".")
    try:
        item = _walk(ev, parts)
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError(f"核对项 {row_id} 已不存在，请重新识别。") from exc
    if not isinstance(item, dict):
        raise ValueError(f"核对项 {row_id} 无法修改。")
    if parts[0] == "layers":
        if text is None:
            item.setdefault("role", None)
            if item["role"] is None:
                item["role"] = "ignore"
            settle(item)
        elif text in LAYER_CHOICES:
            settle(item, role=LAYER_CHOICES[text])
        else:
            raise ValueError(f"图层用途只能选 {'、'.join(LAYER_CHOICES)}。")
        return ev
    if parts[0] == "members":
        if text == DELETE_CHOICE:
            item["deleted"] = True
            item["confirmed"] = True
        elif text is None:
            settle(item)
        else:
            raise ValueError("杆件只能确认或删除；需要补画的杆件请导入后在模型中添加。")
        return ev
    if text is None:
        settle(item)
    else:
        settle(item, text=text)
    return ev
