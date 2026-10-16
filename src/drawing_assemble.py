"""图纸证据 → 模型 JSON 的确定性组装，以及交给用户的核对表。

输入是 ``drawing_evidence`` 定义的文字证据，输出两样东西：

* **核对表**：单位、轴网、标高、杆件、截面、材料、支座逐行列出「值 / 来源 /
  状态」。状态分四档：``ok`` 已读取、``confirm`` 需确认（置信度低、依赖制图
  约定、或像素比例报警）、``missing`` 缺失、``conflict`` 冲突。后三档都要用户
  处理，处理结果写回证据（来源标为用户），再整体重新组装。
* **模型 JSON**：只有在没有缺失和冲突时才生成，并通过 ``model_io.validate_payload``。

组装规则全部是确定性的：间距累加成轴线坐标，标高给出 z，梁沿轴线逐跨拆分，
柱在轴线交点逐层拆分，规格交给 ``sections`` 构造器算特性，柱底按图纸说明设
支座。**不从像素量任何长度**；图上轴线位置只用来检查抄写的间距是否可疑。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping

import sections as section_builders
from drawing_evidence import (EVIDENCE_FORMAT, LENGTH_UNITS, LOW_CONFIDENCE,
                              length_factor, parse_designation, parse_material,
                              parse_number)
from generator import _COLUMN_REF
from model_io import CURRENT_SCHEMA_VERSION, validate_payload

OK, CONFIRM, MISSING, CONFLICT = "ok", "confirm", "missing", "conflict"

#: 像素比例报警阈值：某段间距的「像素/标注」比例偏离同图中位数超过此比例时
#: 要求确认。依据：扫描与拍照变形通常在 10% 以内，而抄错一位数字（6000→8000、
#: 3600→6300）至少偏 25%；取 20% 介于两者之间。它只触发确认，从不改数值。
PIXEL_SCALE_ALARM = 0.20

SUPPORT_FIX = {"fixed": [1, 1, 1, 1, 1, 1], "pinned": [1, 1, 1, 0, 0, 0]}
SUPPORT_CHOICES = {"固接": "fixed", "铰接": "pinned"}
SUPPORT_NAMES = {value: key for key, value in SUPPORT_CHOICES.items()}
DELETE_CHOICE = "删除此条"
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
        return [row for row in self.rows if row.status != OK]

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
    return OK, ""


def _to_m(value: float, factor: float) -> float:
    """图纸单位 → 米。毫米、厘米按整除换算：13 × 0.001 在浮点里不等于 0.013，
    而 13 / 1000 等于，这样识别出的截面与手工按米输入的截面逐位一致。"""
    divisor = 1.0 / factor
    if divisor > 1.0 and abs(divisor - round(divisor)) < 1e-9:
        return value / round(divisor)
    return value * factor


def _fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value != 0 else "0"


class _Builder:
    def __init__(self, evidence: Mapping[str, Any]):
        self.ev = evidence
        self.rows: list[CheckRow] = []
        self._samples: list[tuple[CheckRow, Mapping[str, Any], tuple[str, float]]] = []
        self.images = {str(item.get("id")): (index, item) for index, item in
                       enumerate(evidence.get("images") or [], 1)
                       if isinstance(item, Mapping)}

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
        if isinstance(raw, str) and raw.strip():
            parts.append(f"原文「{raw.strip()}」")
        return " ".join(parts) or "图上未读到原文"

    def add(self, row_id: str, category: str, item_name: str, value: str,
            item: Mapping[str, Any] | None, status: str, message: str = "", *,
            edit: str = "none", options: tuple[str, ...] = (),
            current: str = "") -> CheckRow:
        source = item.get("source") if item else None
        image = bbox = None
        if isinstance(source, Mapping) and not source.get("user"):
            image = str(source.get("image")) if source.get("image") is not None else None
            bbox = source.get("bbox") if isinstance(source.get("bbox"), list) else None
        row = CheckRow(row_id, category, item_name, value, self.source_text(item), status,
                       message, image, bbox, edit, options, current)
        self.rows.append(row)
        return row

    # ---- 单位 ----
    def unit(self, key: str, label: str) -> float | None:
        item = (self.ev.get("units") or {}).get(key) or {}
        value = item.get("value")
        factor = length_factor(value) if value in LENGTH_UNITS else None
        row_id = f"units.{key}"
        if factor is None:
            self.add(row_id, "单位", label, "—", item or None, MISSING,
                     f"图上没有读到{label}说明；请按图纸选择，不按惯例默认。",
                     edit="choice", options=UNIT_CHOICES)
            return None
        status, message = _base_status(item)
        if status == OK and not _settled(item) and not str(item.get("text") or "").strip():
            status, message = CONFIRM, "图上未见单位说明原文，请确认。"
        self.add(row_id, "单位", label, str(value), item, status, message,
                 edit="choice", options=UNIT_CHOICES, current=str(value))
        return factor

    # ---- 轴网 ----
    def grid(self, family: str, used: bool) -> tuple[list[str], dict[str, float]]:
        """返回轴号顺序与各轴坐标（图纸长度单位，首轴为 0）。"""
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
        spacings = [item for item in grid.get("spacings") or [] if isinstance(item, Mapping)]
        coords: dict[str, float] = {labels[0]: 0.0}
        pairs: dict[frozenset[str], tuple[int, Mapping[str, Any]]] = {}
        for index, item in enumerate(spacings):
            pairs.setdefault(frozenset((str(item.get("from")).strip(),
                                        str(item.get("to")).strip())), (index, item))
        consecutive: set[int] = set()
        complete = True
        alarm_rows: list[tuple[CheckRow, Mapping[str, Any], str, str, float]] = []
        for a, b in zip(labels, labels[1:], strict=False):
            found = pairs.get(frozenset((a, b)))
            item_name = f"轴 {a}→{b} 间距"
            if found is None:
                self.add(f"grids.{family}.spacings+{a}:{b}", "轴网", item_name, "—", None,
                         MISSING, "图上没有读到这段间距；请照图填写（单位同尺寸单位），"
                         "不从图上量取。", edit="text")
                complete = False
                continue
            index, item = found
            consecutive.add(index)
            value = parse_number(item.get("text"))
            row_id = f"grids.{family}.spacings.{index}"
            if value is None or value <= 0:
                self.add(row_id, "轴网", item_name, "—", item, CONFLICT,
                         f"读不懂间距「{item.get('text')}」；请改为纯数字。",
                         edit="text", current=str(item.get("text") or ""))
                complete = False
                continue
            status, message = _base_status(item)
            row = self.add(row_id, "轴网", item_name, _fmt(value), item, status, message,
                           edit="text", current=str(item.get("text")))
            alarm_rows.append((row, item, a, b, value))
            if complete:
                coords[b] = coords[a] + value
        if not complete:
            coords = {}
        else:
            self._pixel_alarm(family, grid, alarm_rows)
            order = {label: position for position, label in enumerate(labels)}
            for index, item in enumerate(spacings):
                if index in consecutive:
                    continue
                a, b = str(item.get("from")).strip(), str(item.get("to")).strip()
                row_id = f"grids.{family}.spacings.{index}"
                value = parse_number(item.get("text"))
                if a not in order or b not in order:
                    self.add(row_id, "轴网", f"轴 {a}→{b} 尺寸", str(item.get("text")), item,
                             CONFLICT, "引用了不存在的轴号；请改正或删除这条尺寸。",
                             edit="text", current=str(item.get("text") or ""))
                    continue
                self._check_chain(row_id, f"轴 {a}→{b} 尺寸", item, value,
                                  abs(coords[b] - coords[a]))
            total = grid.get("total")
            if isinstance(total, Mapping) and str(total.get("text") or "").strip():
                self._check_chain(f"grids.{family}.total", f"{name}总尺寸", total,
                                  parse_number(total.get("text")),
                                  coords[labels[-1]] - coords[labels[0]])
        return labels, coords

    def _check_chain(self, row_id: str, item_name: str, item: Mapping[str, Any],
                     value: float | None, expected: float) -> None:
        if value is None or value <= 0:
            self.add(row_id, "轴网", item_name, "—", item, CONFLICT,
                     f"读不懂尺寸「{item.get('text')}」；请改为纯数字。",
                     edit="text", current=str(item.get("text") or ""))
        elif abs(value - expected) > 1e-6 * max(1.0, abs(expected)):
            self.add(row_id, "轴网", item_name, _fmt(value), item, CONFLICT,
                     f"分段间距之和为 {_fmt(expected)}，与标注 {_fmt(value)} 不符；"
                     "请对照原图改正读错的那一段或这条尺寸。",
                     edit="text", current=str(item.get("text")))
        else:
            status, message = _base_status(item)
            self.add(row_id, "轴网", item_name, _fmt(value), item, status,
                     message or "与分段间距之和一致。", edit="text",
                     current=str(item.get("text")))

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
    def levels(self, needed: bool) -> dict[str, float]:
        items = [item for item in self.ev.get("levels") or [] if isinstance(item, Mapping)]
        if not items:
            if needed:
                self.add("levels+", "标高", "各层标高", "—", None, MISSING,
                         "图上没有读到标高；请照图填写各层标高，用逗号分隔，"
                         "如「±0.000, 3.600, 7.200」（单位按标高单位）。", edit="text")
            return {}
        result: dict[str, float] = {}
        seen_values: dict[float, str] = {}
        for index, item in enumerate(items):
            name = str(item.get("name") or "").strip()
            value = parse_number(item.get("text"))
            row_id = f"levels.{index}"
            label = f"标高 {name}"
            if value is None:
                self.add(row_id, "标高", label, "—", item, CONFLICT,
                         f"读不懂标高「{item.get('text')}」；请改为数字，如 3.600。",
                         edit="text", current=str(item.get("text") or ""))
                continue
            if name in result or value in seen_values:
                other = name if name in result else seen_values[value]
                self.add(row_id, "标高", label, _fmt(value), item, CONFLICT,
                         f"与标高 {other} 重复；请改正。", edit="text",
                         current=str(item.get("text")))
                continue
            status, message = _base_status(item)
            self.add(row_id, "标高", label, _fmt(value), item, status, message,
                     edit="text", current=str(item.get("text")))
            result[name] = value
            seen_values[value] = name
        return result


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

    f_len = b.unit("length", "尺寸单位")
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
    f_elev = b.unit("elevation", "标高单位") if (levels_ev or active) else None
    levels = b.levels(needed=bool(active))
    axes = {} if f_len is None else {
        family: {label: _to_m(value, f_len) for label, value in coords.items()}
        for family, coords in (("x", x_coords), ("y", y_coords)) if coords}

    # ---- 杆件：逐跨、逐层拆分 ----
    x_set, y_set = set(x_labels), set(y_labels)
    segments: dict[frozenset, dict[str, Any]] = {}
    member_paths: dict[frozenset, str] = {}
    used_labels: list[str] = []
    level_order = sorted(levels, key=lambda key: levels[key])
    for index, item in active:
        label = str(item.get("label") or "").strip() or f"杆件{index + 1}"
        row_id = f"members.{index}"
        kind = item.get("kind")
        try:
            pieces = _member_pieces(item, kind, x_labels, y_labels, x_set, y_set,
                                    levels, levels_ev, level_order)
        except ValueError as exc:
            b.add(row_id, "杆件", label, _describe_member(item), item, CONFLICT, str(exc),
                  edit="choice", options=(DELETE_CHOICE,))
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
                  edit="choice", options=(DELETE_CHOICE,))
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
            b.add(f"sections+{label}", "截面", f"{label} 规格", "—", None, MISSING,
                  "截面表里没有读到这个编号的规格；请照图填写，如 HN400×200×8×13。",
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
        if f_len is not None:
            try:
                section = _build_section(shape, f_len)
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

    for index, note in enumerate(evidence.get("notes") or []):
        if isinstance(note, Mapping) and str(note.get("text") or "").strip():
            status = OK if _settled(note) else CONFIRM
            b.add(f"notes.{index}", "图纸", "识别提示", str(note["text"]).strip(), note, status,
                  "" if status == OK else "识别时看不清或拿不准的内容；请对照原图确认已处理。")

    blocked = any(row.blocking for row in b.rows) or None in (f_len, f_elev) \
        or material is None or support_kind is None
    if blocked:
        return Assembly(b.rows, None, axes=axes)

    model, provenance = _build_model(
        evidence, segments, member_paths, x_coords, y_coords, levels, f_len, f_elev,
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


def _build_model(evidence, segments, member_paths, x_coords, y_coords, levels, f_len,
                 f_elev, built_sections, label_section, material, support_kind):
    def point(end):
        xl, yl, level = end
        return (round(_to_m(x_coords[xl], f_len) if xl is not None else 0.0,
                      _COINCIDENT_DIGITS),
                round(_to_m(y_coords[yl], f_len) if yl is not None else 0.0,
                      _COINCIDENT_DIGITS),
                round(_to_m(levels[level], f_elev), _COINCIDENT_DIGITS))

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
