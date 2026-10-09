"""从清晰水平梁的填充或闭合轮廓提取梁轴候选；不读取真值或求解数值。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import find_objects, label


def _support_anchor(pixels: np.ndarray, node: dict, band_bottom: int,
                    beam_left: float, beam_right: float) -> float | None:
    """只使用紧邻梁下方的唯一小支座轮廓，文字和长图线不作为附着点。"""
    height, width = pixels.shape
    centre = node["u"] * (width - 1)
    low, high = max(0, int(centre - width * 0.04)), min(width, int(centre + width * 0.04) + 1)
    bottom = min(height, band_bottom + max(12, int(min(height * 0.16, width * 0.06))))
    # 用梁身范围检查黑色边线，避免固定裁去三行时同时丢失小支座的顶端。
    start = band_bottom + 1
    for y in range(start, min(bottom, start + max(4, int(height * 0.015)))):
        if (pixels[y, int(beam_left):int(beam_right) + 1] < 100).mean() >= 0.85:
            start = y + 1
    # 支座边缘的抗锯齿/JPEG 灰度也保留，防止只剩一侧轮廓时偏移附着点。
    labels, _ = label(pixels[start:bottom, low:high] < 200,
                      structure=np.ones((3, 3), dtype=int))
    candidates = []
    for index, bounds in enumerate(find_objects(labels), 1):
        if bounds is None:
            continue
        y, x = bounds
        if (y.start <= max(4, height * 0.015) and y.stop - y.start >= 3
                and 3 <= x.stop - x.start <= width * 0.08
                and x.start > 0 and x.stop < high - low):
            component = labels[y, x] == index
            spans = []
            centres = []
            for row in component:
                points = np.flatnonzero(row)
                if len(points):
                    spans.append(points[-1] - points[0] + 1)
                    centres.append((points[-1] + points[0]) / 2)
            # 三角支座从顶端向下展开；排除细杆、文字引线和向下的箭头。
            if (spans[-1] >= max(3, spans[0] * 1.5)
                    and max(centres) - min(centres) <= max(1, (x.stop - x.start) * 0.2)):
                candidates.append(low + x.start + centres[0])
    return float(candidates[0]) if len(candidates) == 1 else None


def _outline_band(pixels: np.ndarray, x0: int, x1: int, span: float,
                  rough_v: float) -> tuple[float, float, int, int] | None:
    """闭合矩形须有两条长水平边与两端竖边；曲线和未闭合尺寸线不提出候选。"""
    height, width = pixels.shape
    rows = []
    for y in range(height):
        edges = np.diff(np.r_[False, pixels[y, x0:x1] < 230, False].astype(np.int8))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if len(starts) and max(stops - starts) >= span * 0.9:
            rows.append(y)
    groups = []
    for y in rows:
        if not groups or y != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(y)
    groups = [g for g in groups if len(g) <= max(4, height * 0.02)]
    candidates = []
    for first, second in zip(groups, groups[1:], strict=False):
        top, bottom = first[-1], second[0]
        if not 3 <= bottom - top <= height * 0.2:
            continue
        if abs((top + bottom) / 2 / (height - 1) - rough_v) > 0.15:
            continue
        body = pixels[top:bottom + 1, x0:x1] < 230
        vertical = body.mean(axis=0) >= 0.85
        boundaries = np.diff(np.r_[False, vertical, False].astype(np.int8))
        starts, stops = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
        if len(starts) != 2 or any(stops - starts > width * 0.08):
            continue
        left, right = x0 + stops[0] - 1, x0 + starts[1]
        if right - left < span * 0.9 or body[:, stops[0]:starts[1]].mean() > 0.3:
            continue
        candidates.append((float(left), float(right), top, bottom))
    return candidates[0] if len(candidates) == 1 else None


def _black_band(pixels: np.ndarray, x0: int, x1: int, span: float):
    """黑色实心梁须有持续的厚带；细尺寸线和多个候选带不用于修正。"""
    rows = []
    for y, row in enumerate(pixels[:, x0:x1]):
        edges = np.diff(np.r_[False, row < 50, False].astype(np.int8))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if len(starts):
            index = int(np.argmax(stops - starts))
            if stops[index] - starts[index] >= span * 0.85:
                rows.append((y, x0 + starts[index], x0 + stops[index] - 1))
    groups = []
    for row in rows:
        if not groups or row[0] != groups[-1][-1][0] + 1:
            groups.append([])
        groups[-1].append(row)
    groups = [g for g in groups if max(3, pixels.shape[0] * 0.012) <= len(g) <= pixels.shape[0] * 0.06]
    if len(groups) != 1:
        return None
    top, bottom = groups[0][0][0], groups[0][-1][0]
    persistent = np.flatnonzero((pixels[top:bottom + 1, x0:x1] < 50).mean(axis=0) >= 0.85)
    if (not len(persistent) or len(persistent) < (persistent[-1] - persistent[0] + 1) * 0.95
            or persistent[0] == 0 or persistent[-1] == x1 - x0 - 1):
        return None
    return float(x0 + persistent[0]), float(x0 + persistent[-1]), top, bottom


def refine_horizontal_axis(draft: dict, image_path: str | Path) -> dict:
    """仅修正水平链的唯一填充带或闭合矩形，歧义、斜杆和多梁保持原样。"""
    model = draft["image_model"]
    nodes = model.get("nodes") or []
    members = model.get("members") or []
    if len(nodes) < 2 or len(members) != len(nodes) - 1:
        return draft
    if (draft.get("scale") or {}).get("status") == "confirmed" or any(
            entity.get("verified") for entity in draft.get("entities") or []):
        return draft
    order = sorted(nodes, key=lambda node: node["u"])
    expected = {frozenset((a["id"], b["id"])) for a, b in zip(order, order[1:], strict=False)}
    if {frozenset((m["i"], m["j"])) for m in members} != expected:
        return draft
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        pixels = np.asarray(background.convert("L"))
    height, width = pixels.shape
    if width < 20 or height < 20 or max(n["v"] for n in nodes) - min(n["v"] for n in nodes) > 0.015:
        return draft
    span = (order[-1]["u"] - order[0]["u"]) * (width - 1)
    if span < max(20, width * 0.25):
        return draft
    x0 = max(0, int(order[0]["u"] * (width - 1) - width * 0.08))
    x1 = min(width, int(order[-1]["u"] * (width - 1) + width * 0.08) + 1)
    # 全图必须只有一个匹配带，才允许校正模型把图片纵坐标原点颠倒的情况。
    y0, y1 = 0, height
    rows = []
    for y in range(y0, y1):
        boundaries = np.diff(np.r_[False, pixels[y, x0:x1] < 230, False].astype(np.int8))
        starts, stops = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
        if not len(starts):
            continue
        index = int(np.argmax(stops - starts))
        left, right = int(starts[index]) + x0, int(stops[index]) + x0 - 1
        # 灰色实体填充提供连续证据；排除黑色轮廓、支座及箭头对端点的延伸。
        grey = np.flatnonzero((pixels[y, left:right + 1] > 50) & (pixels[y, left:right + 1] < 230))
        if len(grey) < (right - left + 1) * 0.85:
            continue
        left, right = left + int(grey[0]), left + int(grey[-1])
        if right - left + 1 >= span * 0.85 and left > x0 and right < x1 - 1:
            rows.append((y, left, right))
    bands = []
    for row in rows:
        if not bands or row[0] != bands[-1][-1][0] + 1:
            bands.append([])
        bands[-1].append(row)
    bands = [band for band in bands if 3 <= len(band) <= height * 0.12]
    if len(bands) > 1:
        return draft
    method = "filled-horizontal-pixel-band/v2"
    if bands:
        band = bands[0]
        top, bottom = band[0][0], band[-1][0]
        body = pixels[top:bottom + 1, x0:x1]
        persistent = np.flatnonzero(((body > 50) & (body < 230)).mean(axis=0) >= 0.75)
        if not len(persistent) or len(persistent) < (persistent[-1] - persistent[0] + 1) * 0.85:
            return draft
        left, right = float(x0 + persistent[0]), float(x0 + persistent[-1])
    else:
        outline = _black_band(pixels, x0, x1, span)
        method = "solid-black-horizontal-band/v1"
        if outline is None:
            outline = _outline_band(pixels, x0, x1, span, sum(n["v"] for n in nodes) / len(nodes))
            method = "closed-horizontal-outline/v1"
        if outline is None:
            return draft
        left, right, top, bottom = outline
    if right - left < span * 0.85:
        return draft
    if abs(left - order[0]["u"] * (width - 1)) > width * 0.08 \
            or abs(right - order[-1]["u"] * (width - 1)) > width * 0.08:
        return draft
    result = deepcopy(draft)
    target_nodes = {node["id"]: node for node in result["image_model"]["nodes"]}
    axis_v = (top + bottom) / 2 / (height - 1)
    for node in target_nodes.values():
        node["v"] = axis_v
    target_nodes[order[0]["id"]]["u"] = left / (width - 1)
    target_nodes[order[-1]["id"]]["u"] = right / (width - 1)
    # 内部节点若有唯一穿入梁身的竖直笔画，以该笔画提出横坐标候选。
    for node in order[1:-1]:
        anchor_x = node["u"] * (width - 1)
        low, high = max(int(left), int(anchor_x - width * 0.04)), min(int(right) + 1, int(anchor_x + width * 0.04) + 1)
        interior = pixels[top + 1:bottom, low:high]
        if not interior.size:
            continue
        strong = (interior < 50).sum(axis=0) >= max(3, interior.shape[0] * 0.6)
        edges = np.diff(np.r_[False, strong, False].astype(np.int8))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if len(starts) == 1 and stops[0] - starts[0] <= max(3, width * 0.02):
            target_nodes[node["id"]]["u"] = float(low + (starts[0] + stops[0] - 1) / 2) / (width - 1)
    for support in model.get("supports") or []:
        if all(support.get("fix") or []):
            continue
        node = next((n for n in nodes if n["id"] == support.get("node")), None)
        if node is not None:
            anchor = _support_anchor(pixels, target_nodes[node["id"]], bottom, left, right)
            if anchor is not None:
                target_nodes[node["id"]]["u"] = anchor / (width - 1)
    evidence = {"method": method,
                "band_px": [left, top, right, bottom],
                "original_nodes": [{"id": n["id"], "u": n["u"], "v": n["v"]} for n in nodes]}
    for entity in result.get("entities") or []:
        kind, target = entity.get("kind"), entity.get("target") or {}
        if kind not in ("node", "member", "support"):
            continue
        entity["original_observation"] = {"image_geometry": deepcopy(entity.get("image_geometry")),
                                           "confidence": entity.get("confidence")}
        entity.update(source="derived", confidence=None, recognition_confidence=None, verified=False,
                      position_refinement=deepcopy(evidence))
        if kind == "node" and target.get("node") in target_nodes:
            node = target_nodes[target["node"]]
            entity["image_geometry"] = {"point": [node["u"], node["v"]]}
        elif kind == "support":
            node = target_nodes[target["support"]["node"]]
            entity["image_geometry"] = {"point": [node["u"], node["v"]]}
        elif kind == "member":
            member = next(m for m in members if m["id"] == target["member"])
            a, b = target_nodes[member["i"]], target_nodes[member["j"]]
            entity["image_geometry"] = {"line": [[a["u"], a["v"]], [b["u"], b["v"]]]}
    issues = result.setdefault("issues", [])
    ident = "pixel-axis-review"
    while any(issue.get("id") == ident for issue in issues):
        ident += "-new"
    issues.append({"id": ident, "category": "low_confidence", "severity": "blocking", "status": "open",
                   "entity_refs": [f"node:{node['id']}" for node in nodes],
                   "message": "已根据连续像素带提出水平梁轴候选位置，原始识别坐标已保留。请核对梁轴端点、支座附着点及中间节点，确认后再提交。",
                   "resolution": None, "resolved_by": None})
    return result
