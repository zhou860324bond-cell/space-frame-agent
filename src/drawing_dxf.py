"""结构图纸 DXF → 文字证据（矢量入口）。

与图片入口产出同一种证据（``drawing_evidence`` 文件头所述格式），交给同一个
``drawing_assemble.assemble``。区别只在数从哪里来：图片靠大模型抄字，DXF 的
文字、尺寸实测值和单位都直接从实体读出，出处是实体句柄而不是图上一块区域。

分工：

* **图层用途只由用户或大模型判断**（``LAYER_ROLES``）。代码不按图层名猜，
  判断结果写进证据 ``layers``，在核对表里逐个确认；改了用途就按新用途重读。
* 轴线：轴线图层上的 LINE 按方向分组（竖向为 x 族、横向为 y 族），轴号取轴号圈
  （CIRCLE）里的 TEXT/MTEXT，圈与线按几何关系配对。
* 间距：尺寸图层上的线性 DIMENSION，两条尺寸界线原点落在哪两根轴线上就是哪段
  间距；数值取实体实测值（乘 DIMLFAC），显示文字作为原文写入 ``text``。
* 标高：标高图层上能读成数的文字；它旁边的标高符号（最近的非文字实体）的最低点
  只用来判断杆件端点在哪一层，标高数值取文字。
* 杆件：梁、柱图层上的线段，端点落在哪根轴线、哪个标高上就是哪段杆件（本版读
  立面单线图）；编号与规格取杆件旁的文字和截面表。
* 材料：说明图层上能按 ``parse_material`` 读出牌号的文字。柱脚做法不按关键字猜，
  留给核对表由用户选择。

**几何坐标只用来判断「哪个实体挨着哪根轴线、哪个标高」，从不换算成长度。**
读不到、对不上的内容写进 ``notes`` 或保持缺失，由核对表交给用户。
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from drawing_evidence import (EVIDENCE_FORMAT, LAYER_ROLES, parse_designation,
                              parse_material, parse_number)
from multimodal_contract import file_digest

#: $INSUNITS 代码 → 长度单位（DXF 参考手册 HEADER 段）；其他代码不认，交给用户选。
INSUNITS = {4: "mm", 5: "cm", 6: "m"}
_UNIT_NAMES = {"mm": "毫米", "cm": "厘米", "m": "米"}

#: 端点「落在」轴线/标高上的容差，取同族相邻轴线（或标高）最小间隔的比例。
#: CAD 绘图用对象捕捉，端点与轴线重合到浮点精度；1% 远小于相邻轴线的一半间隔，
#: 不会落到隔壁轴线上。容差随图纸单位一起缩放，与单位制无关。
SNAP_RATIO = 0.01
#: 方向判定：线段横向（或竖向）分量小于长度的此比例即视为竖向（或横向），约 0.06°。
ORTHO_TOL = 1.0e-3
#: 杆件编号文字到杆件的距离上限，以字高计。制图习惯把编号写在杆件旁 0.5~1.5 倍字高处，
#: 取 3 倍留余量；更远的文字不认作编号，写进识别提示交给用户。
LABEL_DISTANCE = 3.0
#: 同一行表格文字的基线高差上限，以字高计。
ROW_TOLERANCE = 0.5

DXF_SUFFIXES = (".dxf",)

Point = tuple[float, float]


class DxfError(ValueError):
    """DXF 读不了或超出本版范围；消息写明原因和下一步。"""


@dataclass
class _Text:
    handle: str
    layer: str
    text: str
    point: Point
    height: float


@dataclass
class _Segment:
    handle: str
    layer: str
    start: Point
    end: Point

    @property
    def vertical(self) -> bool:
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        return abs(dx) <= ORTHO_TOL * math.hypot(dx, dy) and abs(dy) > 0

    @property
    def horizontal(self) -> bool:
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        return abs(dy) <= ORTHO_TOL * math.hypot(dx, dy) and abs(dx) > 0


@dataclass
class _Circle:
    handle: str
    layer: str
    center: Point
    radius: float


@dataclass
class _Content:
    """按图层归拢的实体（INSERT 已展开，句柄记为块参照的句柄）。"""

    texts: list[_Text] = field(default_factory=list)
    segments: list[_Segment] = field(default_factory=list)
    circles: list[_Circle] = field(default_factory=list)
    dimensions: list[Any] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)
    insunits: int = 0


# ---- 读取 --------------------------------------------------------------------

def _ezdxf():
    try:
        import ezdxf
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的说明
        raise DxfError("读取 DXF 需要 ezdxf：请运行 pip install ezdxf"
                       "（或安装可选依赖 space-frame-agent[dxf]）。") from exc
    return ezdxf


def _open(path: Path):
    ezdxf = _ezdxf()
    if not path.is_file():
        raise DxfError(f"找不到「{path}」；请检查路径后重新添加。")
    try:
        return ezdxf.readfile(str(path))
    except (OSError, ezdxf.DXFError) as exc:
        reason = str(exc).rstrip("。.")
        raise DxfError(f"「{path.name}」无法作为 DXF 读取（{reason}）。请在 CAD 中另存为 DXF"
                       "（AutoCAD 2000 或更新格式；DWG 也要先另存）后重试。") from exc


def _xy(point) -> Point:
    return float(point[0]), float(point[1])


def _plain(entity) -> str:
    try:
        return str(entity.plain_text()).strip()
    except (AttributeError, TypeError):
        return str(entity.dxf.get("text", "")).strip()


def _text_point(entity) -> Point:
    """TEXT 非左对齐时以对齐点为准（轴号圈里的字通常居中对齐）。"""
    if entity.dxftype() == "MTEXT":
        return _xy(entity.dxf.insert)
    align = entity.dxf.get("halign", 0) or entity.dxf.get("valign", 0)
    if align and entity.dxf.hasattr("align_point"):
        return _xy(entity.dxf.align_point)
    return _xy(entity.dxf.insert)


def _collect(doc) -> _Content:
    content = _Content(insunits=int(doc.header.get("$INSUNITS", 0) or 0))

    def note(layer: str, kind: str, text: str = "") -> None:
        counts = content.counts.setdefault(layer, {})
        counts[kind] = counts.get(kind, 0) + 1
        samples = content.samples.setdefault(layer, [])
        if text and len(samples) < 6 and text not in samples:
            samples.append(text)

    def visit(entity, handle: str | None, layer: str | None, depth: int) -> None:
        kind = entity.dxftype()
        own = entity.dxf.get("layer", "0")
        lay = layer if layer is not None and own == "0" else own
        ref = handle or entity.dxf.handle
        if kind == "INSERT" and depth < 4:
            note(lay, kind)
            for attrib in entity.attribs:
                visit(attrib, attrib.dxf.handle, lay, depth + 1)
            for child in entity.virtual_entities():
                visit(child, ref, lay, depth + 1)
            return
        if kind in ("TEXT", "MTEXT", "ATTRIB"):
            text = _plain(entity)
            if text:
                height = float(entity.dxf.get("char_height" if kind == "MTEXT" else "height",
                                              0) or 0)
                content.texts.append(_Text(ref, lay, text, _text_point(entity), height))
                note(lay, "TEXT" if kind == "ATTRIB" else kind, text)
        elif kind == "LINE":
            content.segments.append(_Segment(ref, lay, _xy(entity.dxf.start),
                                             _xy(entity.dxf.end)))
            note(lay, kind)
        elif kind == "LWPOLYLINE":
            points = [_xy(p) for p in entity.get_points("xy")]
            if entity.closed and len(points) > 2:
                points.append(points[0])
            for a, b in zip(points, points[1:], strict=False):
                if a != b:
                    content.segments.append(_Segment(ref, lay, a, b))
            note(lay, kind)
        elif kind == "CIRCLE":
            content.circles.append(_Circle(ref, lay, _xy(entity.dxf.center),
                                           float(entity.dxf.radius)))
            note(lay, kind)
        elif kind == "DIMENSION":
            content.dimensions.append(entity)
            note(lay, kind)
        else:
            note(lay, kind)

    for entity in doc.modelspace():
        visit(entity, None, None, 0)
    return content


def file_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {"id": "D1", "name": path.name, "path": str(path), "sha256": file_digest(path),
            "kind": "dxf"}


def scan_layers(path: str | Path) -> list[dict[str, Any]]:
    """各图层的实体计数与文字样本——交给用户或大模型判断用途的材料。"""
    content = _collect(_open(Path(path)))
    return _layer_summaries(content)


def _layer_summaries(content: _Content) -> list[dict[str, Any]]:
    layers = []
    for name in sorted(content.counts):
        counts = content.counts[name]
        summary = "、".join(f"{kind} {count}" for kind, count in
                           sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        layers.append({"name": name, "counts": dict(counts), "summary": summary,
                       "samples": list(content.samples.get(name, []))})
    return layers


# ---- 几何小工具 ----------------------------------------------------------------

def _distance_to_segment(point: Point, seg: _Segment) -> float:
    (x0, y0), (x1, y1), (px, py) = seg.start, seg.end, point
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length2))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _tolerance(values: Iterable[float]) -> float:
    ordered = sorted(set(values))
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False) if b - a > 0]
    if not gaps:
        return 1e-9 * max(1.0, max((abs(v) for v in ordered), default=1.0))
    return SNAP_RATIO * min(gaps)


def _match(value: float, targets: Mapping[str, Sequence[float]], tol: float) -> str | None:
    best, best_d = None, None
    for label, coords in targets.items():
        for coord in coords:
            d = abs(value - coord)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = label, d
    return best


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _src(handles: Iterable[str], layer: str | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {"file": "D1", "handles": list(dict.fromkeys(handles))}
    if layer:
        source["layer"] = layer
    return source


# ---- 证据 --------------------------------------------------------------------

class _Reader:
    def __init__(self, content: _Content, roles: Mapping[str, str | None]):
        self.c = content
        self.roles = roles
        self.notes: list[dict[str, Any]] = []

    def on(self, *roles: str) -> set[str]:
        return {name for name, role in self.roles.items() if role in roles}

    def note(self, text: str, handles: Iterable[str], layer: str | None = None) -> None:
        self.notes.append({"text": text, "source": _src(handles, layer)})

    # ---- 轴线 ----
    def axes(self) -> dict[str, dict[str, Any]]:
        layers = self.on("axis")
        lines = [s for s in self.c.segments if s.layer in layers]
        texts = [t for t in self.c.texts if t.layer in layers]
        found: dict[str, dict[str, list]] = {"x": defaultdict(list), "y": defaultdict(list)}
        unlabeled: dict[str, list[tuple[float, _Segment]]] = {"x": [], "y": []}
        bubbles = []
        for circle in (c for c in self.c.circles if c.layer in layers):
            inside = [t for t in texts if math.dist(t.point, circle.center) <= circle.radius]
            labels = {t.text for t in inside}
            if len(labels) == 1:
                bubbles.append((circle, inside[0]))
        paired: set[int] = set()
        for circle, text in bubbles:
            best, best_d = None, None
            for index, line in enumerate(lines):
                if not (line.vertical or line.horizontal):
                    continue
                across = abs(circle.center[0] - line.start[0]) if line.vertical \
                    else abs(circle.center[1] - line.start[1])
                end_d = min(math.dist(circle.center, line.start),
                            math.dist(circle.center, line.end))
                if across <= 0.5 * circle.radius and end_d <= 2.0 * circle.radius \
                        and (best_d is None or end_d < best_d):
                    best, best_d = index, end_d
            if best is None:
                self.note(f"轴号圈「{text.text}」旁没有轴线", [circle.handle, text.handle],
                          circle.layer)
                continue
            line = lines[best]
            paired.add(best)
            family = "x" if line.vertical else "y"
            coord = line.start[0] if family == "x" else line.start[1]
            found[family][text.text].append((coord, [line.handle, circle.handle, text.handle],
                                             line.layer))
        for index, line in enumerate(lines):
            if index in paired:
                continue
            if line.vertical:
                unlabeled["x"].append((line.start[0], line))
            elif line.horizontal:
                unlabeled["y"].append((line.start[1], line))
            else:
                self.note("斜向轴线本版不读取", [line.handle], line.layer)
        grids: dict[str, dict[str, Any]] = {}
        for family in ("x", "y"):
            entries = found[family]
            tol = _tolerance(coord for items in entries.values() for coord, _, _ in items)
            axes = []
            for label, items in entries.items():
                coords = sorted({coord for coord, _, _ in items})
                if coords[-1] - coords[0] > tol:
                    raise DxfError(f"轴号「{label}」在图中出现在 {len(coords)} 处，本版 DXF 入口"
                                   "一次只读一个视图；请在 CAD 中把各视图另存为单独的 DXF 再分别识别。")
                handles = [h for _, hs, _ in items for h in hs]
                axes.append({"label": label, "coord": coords[0], "handles": handles,
                             "layer": items[0][2]})
            axes.sort(key=lambda axis: axis["coord"])
            for coord, line in unlabeled[family]:
                axis = next((a for a in axes if abs(a["coord"] - coord) <= tol), None)
                if axis is not None:
                    axis["handles"].append(line.handle)
                else:
                    self.note("轴线没有读到轴号圈，未使用", [line.handle], line.layer)
            if axes:
                grids[family] = {"axes": axes, "tol": tol}
        return grids

    # ---- 尺寸 ----
    def spacings(self, grids: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        layers = self.on("dimension")
        result: dict[str, list[dict[str, Any]]] = {"x": [], "y": []}
        skipped = []
        for dim in self.c.dimensions:
            layer = dim.dxf.get("layer", "0")
            if layer not in layers:
                continue
            if dim.dimtype not in (0, 1):
                skipped.append(dim.dxf.handle)
                continue
            p2, p3 = _xy(dim.dxf.defpoint2), _xy(dim.dxf.defpoint3)
            if dim.dimtype == 0:
                angle = math.radians(float(dim.dxf.get("angle", 0.0) or 0.0))
                direction = (math.cos(angle), math.sin(angle))
            else:
                length = math.dist(p2, p3) or 1.0
                direction = ((p3[0] - p2[0]) / length, (p3[1] - p2[1]) / length)
            if abs(direction[1]) <= ORTHO_TOL:
                family, a, b = "x", p2[0], p3[0]
            elif abs(direction[0]) <= ORTHO_TOL:
                family, a, b = "y", p2[1], p3[1]
            else:
                skipped.append(dim.dxf.handle)
                continue
            grid = grids.get(family)
            targets = {axis["label"]: [axis["coord"]] for axis in (grid or {}).get("axes", [])}
            la = _match(a, targets, grid["tol"]) if grid else None
            lb = _match(b, targets, grid["tol"]) if grid else None
            if la is None or lb is None or la == lb:
                skipped.append(dim.dxf.handle)
                continue
            measured = float(dim.get_measurement()) * float(
                dim.override().get("dimlfac", 1.0) or 1.0)
            raw = str(dim.dxf.get("text", "") or "")
            shown = _fmt(measured)
            text = shown if raw in ("", "<>") else raw.replace("<>", shown)
            result[family].append({"from": la, "to": lb, "text": text, "value": measured,
                                   "source": _src([dim.dxf.handle], layer)})
        if skipped:
            self.note(f"{len(skipped)} 个尺寸标注没有落在两根已读轴线上（或不是线性尺寸），"
                      "未使用", skipped)
        return result

    # ---- 标高 ----
    def levels(self) -> list[dict[str, Any]]:
        layers = self.on("level")
        marks: dict[str, list[_Segment]] = defaultdict(list)
        for seg in self.c.segments:
            if seg.layer in layers:
                marks[seg.handle].append(seg)
        levels: dict[float, dict[str, Any]] = {}
        for text in (t for t in self.c.texts if t.layer in layers):
            value = parse_number(text.text)
            if value is None:
                continue
            mark_y, handles = None, [text.handle]
            if marks:
                handle = min(marks, key=lambda h: min(_distance_to_segment(text.point, s)
                                                      for s in marks[h]))
                mark_y = min(min(s.start[1], s.end[1]) for s in marks[handle])
                handles.append(handle)
            level = levels.setdefault(value, {"name": text.text, "text": text.text,
                                              "value": value, "ys": [], "handles": [],
                                              "layer": text.layer})
            if mark_y is not None:
                level["ys"].append(mark_y)
            level["handles"] += handles
        return sorted(levels.values(), key=lambda level: level["value"])

    # ---- 杆件 ----
    def members(self, grids, levels) -> tuple[list[dict[str, Any]], list[_Segment]]:
        roles = {name: role for name, role in self.roles.items() if role in ("beam", "column")}
        segments = [s for s in self.c.segments if s.layer in roles]
        x_axes = {a["label"]: [a["coord"]] for a in grids.get("x", {}).get("axes", [])}
        y_axes = {a["label"]: [a["coord"]] for a in grids.get("y", {}).get("axes", [])}
        x_tol = grids.get("x", {}).get("tol", 0.0)
        y_tol = grids.get("y", {}).get("tol", 0.0)
        level_ys = {level["name"]: level["ys"] for level in levels if level["ys"]}
        z_tol = _tolerance(y for ys in level_ys.values() for y in ys)
        members = []
        for seg in segments:
            kind = roles[seg.layer]
            ends = [(_match(p[0], x_axes, x_tol), _match(p[1], level_ys, z_tol),
                     _match(p[1], y_axes, y_tol)) for p in (seg.start, seg.end)]
            (xa, la, ya), (xb, lb, yb) = ends
            item: dict[str, Any] | None = None
            if kind == "beam" and la and la == lb and xa and xb and xa != xb:
                item = {"kind": "beam", "from": xa, "to": xb, "levels": [la]}
            elif kind == "column" and xa and xa == xb and la and lb and la != lb:
                order = [level["name"] for level in levels]
                lo, hi = sorted((la, lb), key=order.index)
                item = {"kind": "column", "at": [xa], "from_level": lo, "to_level": hi}
            if item is None:
                if ya and yb and not (la or lb):
                    reason = "位于平面轴网上；本版 DXF 入口只读立面单线图的杆件"
                else:
                    reason = "两端不在轴线与标高的交点上，未读取"
                self.note(f"{LAYER_ROLES[kind]}线段{reason}", [seg.handle], seg.layer)
                continue
            item["source"] = _src([seg.handle], seg.layer)
            item["entity"] = seg.handle
            members.append(item)
        return members, segments

    def labels(self, members, segments) -> list[dict[str, Any]]:
        """杆件旁的编号与规格文字 → 杆件 label 与就地规格。"""
        layers = self.on("label", "beam", "column")
        by_entity: dict[str, list[_Text]] = defaultdict(list)
        located = [s for s in segments if any(m["entity"] == s.handle for m in members)]
        for text in (t for t in self.c.texts if t.layer in layers):
            if not located:
                break
            seg = min(located, key=lambda s: _distance_to_segment(text.point, s))
            limit = LABEL_DISTANCE * (text.height or 0.0)
            if _distance_to_segment(text.point, seg) <= limit:
                by_entity[seg.handle].append(text)
            else:
                self.note(f"文字「{text.text}」附近没有杆件，未使用", [text.handle], text.layer)
        inline = []
        for member in members:
            texts = by_entity.get(member["entity"], [])
            codes = {t.text for t in texts if not _is_designation(t.text)}
            specs = [t for t in texts if _is_designation(t.text)]
            handles = [t.handle for t in texts]
            member["source"]["handles"] = list(dict.fromkeys(
                member["source"]["handles"] + handles))
            if len(codes) > 1:
                self.note(f"杆件旁有多个编号 {'、'.join(sorted(codes))}，未认定",
                          [member["entity"], *handles])
                continue
            if codes:
                member["label"] = codes.pop()
            elif specs:
                member["label"] = specs[0].text
            if specs and member.get("label"):
                inline.append({"label": member["label"], "text": specs[0].text,
                               "source": _src([specs[0].handle], specs[0].layer)})
        return inline

    def table(self) -> list[dict[str, Any]]:
        """截面表：同一行里的编号与规格配对；一格里写「KL1 HN400×…」也认。"""
        texts = sorted((t for t in self.c.texts if t.layer in self.on("table")),
                       key=lambda t: (-t.point[1], t.point[0]))
        rows: list[list[_Text]] = []
        for text in texts:
            if rows and abs(rows[-1][0].point[1] - text.point[1]) <= ROW_TOLERANCE * max(
                    text.height, rows[-1][0].height, 1e-12):
                rows[-1].append(text)
            else:
                rows.append([text])
        entries = []
        for row in rows:
            row.sort(key=lambda t: t.point[0])
            specs = [t for t in row if _is_designation(t.text)]
            codes = [t for t in row if not _is_designation(t.text)]
            if len(specs) == 1 and codes:
                entries.append({"label": codes[0].text, "text": specs[0].text,
                                "source": _src([codes[0].handle, specs[0].handle],
                                               specs[0].layer)})
                continue
            for text in row:
                head, _, tail = re.sub(r"[:：]", " ", text.text).strip().partition(" ")
                if tail.strip() and _is_designation(tail.strip()):
                    entries.append({"label": head, "text": tail.strip(),
                                    "source": _src([text.handle], text.layer)})
        return entries

    def material(self) -> dict[str, Any] | None:
        found: dict[str, _Text] = {}
        for text in (t for t in self.c.texts if t.layer in self.on("notes")):
            try:
                found.setdefault(parse_material(text.text)["name"], text)
            except ValueError:
                continue
        if len(found) > 1:
            self.note(f"说明中出现多个材料 {'、'.join(found)}，本版只支持一种材料，请在核对表中填写",
                      [t.handle for t in found.values()])
            return None
        if found:
            text = next(iter(found.values()))
            return {"text": text.text, "source": _src([text.handle], text.layer)}
        return None


def _is_designation(text: str) -> bool:
    try:
        parse_designation(text)
    except ValueError:
        return False
    return True


def _merge_sections(entries: list[dict[str, Any]], reader: _Reader) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_label[entry["label"]].append(entry)
    sections = []
    for label, items in by_label.items():
        names = {parse_designation(item["text"])["name"] for item in items}
        if len(names) > 1:
            reader.note(f"{label} 有不同规格 {'、'.join(item['text'] for item in items)}，"
                        "未认定；请在核对表中填写", [h for item in items
                                          for h in item["source"]["handles"]])
            continue
        handles = [h for item in items for h in item["source"]["handles"]]
        sections.append({"label": label, "text": items[0]["text"],
                         "source": {**items[0]["source"], "handles":
                                    list(dict.fromkeys(handles))}})
    return sections


def _layer_items(summaries: list[dict[str, Any]],
                 layers: Mapping[str, str | None] | Sequence[Mapping[str, Any]] | None
                 ) -> list[dict[str, Any]]:
    """把传入的图层用途整理成证据 ``layers``。

    传映射 {图层: 用途} 表示用户已指定（来源记为用户）；传证据里的 ``layers``
    条目则原样沿用（含大模型判断与确认状态）。文件里有、传入里没有的图层用途为空。
    """
    given: dict[str, dict[str, Any]] = {}
    if isinstance(layers, Mapping):
        for name, role in layers.items():
            if role is not None and role not in LAYER_ROLES:
                raise DxfError(f"图层「{name}」的用途「{role}」无效；只能是 "
                               f"{'、'.join(LAYER_ROLES)}。")
            given[str(name)] = {"role": role, "source": {"user": True}, "confirmed": True}
    elif layers:
        for item in layers:
            given[str(item.get("name"))] = {key: value for key, value in item.items()
                                            if key not in ("name", "summary")}
    items = []
    for summary in summaries:
        item = {"name": summary["name"], "role": None, "summary": summary["summary"],
                "source": {"file": "D1", "layer": summary["name"]}}
        item.update(given.get(summary["name"], {}))
        items.append(item)
    return items


def read_dxf(path: str | Path,
             layers: Mapping[str, str | None] | Sequence[Mapping[str, Any]] | None = None
             ) -> dict[str, Any]:
    """读一个 DXF，按图层用途产出文字证据（格式同图片入口）。"""
    path = Path(path)
    content = _collect(_open(path))
    items = _layer_items(_layer_summaries(content), layers)
    roles = {item["name"]: item.get("role") for item in items}
    reader = _Reader(content, roles)
    grids = reader.axes()
    spacings = reader.spacings(grids)
    levels = reader.levels()
    members, segments = reader.members(grids, levels)
    sections = _merge_sections(reader.table() + reader.labels(members, segments), reader)
    record = file_record(path)
    evidence: dict[str, Any] = {
        "format": EVIDENCE_FORMAT,
        "images": [],
        "files": [{key: record[key] for key in ("id", "name", "sha256", "kind")}],
        "layers": items,
        "units": {},
        "grids": {family: {
            "axes": [{"label": axis["label"], "source": _src(axis["handles"], axis["layer"])}
                     for axis in grid["axes"]],
            "spacings": spacings[family]} for family, grid in grids.items()},
        "levels": [{"name": level["name"], "text": level["text"],
                    "source": _src(level["handles"], level["layer"])} for level in levels],
        "members": [{key: value for key, value in member.items() if key != "entity"}
                    for member in members],
        "sections": sections,
        "material": reader.material(),
        "support": None,
        "notes": reader.notes,
    }
    unit = INSUNITS.get(content.insunits)
    if unit is not None:
        evidence["units"]["length"] = {
            "value": unit, "text": f"$INSUNITS = {content.insunits}（{_UNIT_NAMES[unit]}）",
            "source": {"file": "D1", "header": "$INSUNITS"},
            "check": "取自文件头 $INSUNITS；不少图纸没有正确设置这一项，请对照尺寸数值确认。"}
    return evidence


def reread(evidence: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """图层用途改变后按新用途重读；单位、材料、支座上用户已处理的结果保留。"""
    fresh = read_dxf(path, evidence.get("layers") or [])
    for key in ("material", "support"):
        item = evidence.get(key)
        if isinstance(item, Mapping) and (item.get("confirmed") or
                                          (item.get("source") or {}).get("user")):
            fresh[key] = dict(item)
    for key, item in (evidence.get("units") or {}).items():
        if isinstance(item, Mapping) and item.get("confirmed"):
            fresh["units"][key] = dict(item)
    return fresh


# ---- 大模型判断图层用途 ----------------------------------------------------------

LAYER_PROMPT = f"""你是结构施工图 CAD 图层整理助手。只输出一个 JSON 对象，不要 Markdown。

下面列出一个 DXF 文件的图层：名称、实体类型计数、部分文字样本。请判断每个图层的用途，
只能从这些用途里选：
{chr(10).join(f"- {key}：{name}" for key, name in LAYER_ROLES.items())}

说明：axis 含轴线、轴号圈与轴号文字；level 含标高符号与标高数字；label 是写在杆件旁的
编号（如 KL1）或规格；table 是编号与规格对照的截面表；notes 是设计说明（材料牌号等）。
拿不准的图层写 null，不要猜。

格式：{{"layers": {{"图层名": "axis", "另一个图层": null}}}}
"""


def judge_layers(call: Callable[[str, str], str], summaries: list[dict[str, Any]]
                 ) -> dict[str, str | None]:
    """请大模型按图层名与内容判断用途。``call(system, user)`` 返回模型文字。

    只接受 ``LAYER_ROLES`` 里的用途；其他回答当作未判断。数值不经过模型。
    """
    listing = [{"name": s["name"], "counts": s["counts"], "samples": s["samples"]}
               for s in summaries]
    reply = call(LAYER_PROMPT, json.dumps(listing, ensure_ascii=False))
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        raise ValueError("模型没有返回 JSON。")
    payload = json.loads(match.group(0))
    answer = payload.get("layers") if isinstance(payload, Mapping) else None
    if not isinstance(answer, Mapping):
        raise ValueError("模型返回的 JSON 缺少 layers。")
    names = {s["name"] for s in summaries}
    return {str(name): (role if role in LAYER_ROLES else None)
            for name, role in answer.items() if str(name) in names}


def model_layer_items(path: str | Path, roles: Mapping[str, str | None], judge: str,
                      keep: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """大模型判断结果 → 证据 ``layers`` 条目（待用户确认）；``keep`` 里用户已定的图层不改。"""
    kept = {str(item.get("name")): dict(item) for item in keep}
    items = []
    for summary in scan_layers(path):
        if summary["name"] in kept:
            items.append(kept[summary["name"]])
            continue
        role = roles.get(summary["name"])
        item = {"name": summary["name"], "role": role,
                "source": {"file": "D1", "layer": summary["name"]}}
        if role is not None:
            item["check"] = f"由大模型（{judge}）按图层名与内容判断，请确认。"
        items.append(item)
    return items
