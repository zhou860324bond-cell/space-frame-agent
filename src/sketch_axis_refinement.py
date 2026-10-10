"""从清晰梁带和灰色杆线节点提出位置候选；不读取真值或求解数值。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import find_objects, label


def refine_joint_nodes(draft: dict, image_path: str | Path) -> dict:
    """多杆图须全部节点具有唯一圆点及连线证据；只移动未审核图片位置。"""
    model = draft["image_model"]
    nodes, members = model.get("nodes") or [], model.get("members") or []
    if not 3 <= len(nodes) <= 64 or not len(nodes) <= len(members) <= 128:
        return draft
    if ((draft.get("scale") or {}).get("status") == "confirmed"
            or any(e.get("verified") or e.get("source") == "user" for e in draft.get("entities") or [])
            or any(d.get("status") == "confirmed" or d.get("source") == "user"
                   for d in draft.get("dimensions") or [])
            or any(i.get("source") == "user" for i in draft.get("intersections") or [])
            or any(n.get("source") == "user" for n in nodes + members)):
        return draft
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        rgb = np.asarray(background.convert("RGB"), dtype=np.int16)
    height, width = rgb.shape[:2]
    if min(width, height) < 20:
        return draft
    maximum = rgb.max(axis=2)
    spread = maximum - rgb.min(axis=2)
    # 彩色荷载/尺寸线不能充当圆点；灰色圆点周围还须有灰色杆身证据。
    dark = (maximum < 90) & (spread < 25)
    grey = (maximum >= 100) & (maximum < 245) & (spread < 28)
    components, _ = label(dark, structure=np.ones((3, 3), dtype=int))
    candidates = []
    for ident, bounds in enumerate(find_objects(components), 1):
        if bounds is None:
            continue
        y, x = bounds
        h, w = y.stop - y.start, x.stop - x.start
        component = components[bounds] == ident
        if (not 3 <= min(h, w) <= max(h, w) <= max(10, min(width, height) * .06)
                or not .65 <= w / h <= 1.5 or component.mean() < .5):
            continue
        yy, xx = np.nonzero(component)
        centre = np.array([x.start + xx.mean(), y.start + yy.mean()])
        radius = max(w, h)
        low, high = np.floor(centre - radius).astype(int), np.ceil(centre + radius).astype(int) + 1
        if min(low) < 0 or high[0] > width or high[1] > height:
            continue
        ys, xs = np.mgrid[low[1]:high[1], low[0]:high[0]]
        distance = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2
        ring = (distance >= radius ** 2 * .4) & (distance <= radius ** 2)
        fraction = float(grey[low[1]:high[1], low[0]:high[0]][ring].mean())
        if fraction >= .65:
            candidates.append((centre, fraction))
    if len(candidates) < len(nodes) or len(candidates) > 256:
        return draft
    snapped, matches = {}, {}
    tolerance = min(width * .05, height * .12)
    for node in nodes:
        original = np.array([node["u"] * (width - 1), node["v"] * (height - 1)])
        nearby = [index for index, (centre, _) in enumerate(candidates)
                  if np.linalg.norm(centre - original) <= tolerance]
        if len(nearby) != 1 or nearby[0] in matches.values():
            return draft
        matches[node["id"]] = nearby[0]
        snapped[node["id"]] = candidates[nearby[0]][0]
    evidence_lines = []
    used = set()
    for member in members:
        if member["i"] not in snapped or member["j"] not in snapped:
            return draft
        a, b = snapped[member["i"]], snapped[member["j"]]
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 12:
            return draft
        normal = np.array([-delta[1], delta[0]]) / length
        # 灰色杆身可能被蓝色尺寸线覆盖；局部横向窄带须仍有连续证据。
        points = np.rint(a + np.linspace(.2, .8, 24)[:, None, None] * delta
                        + np.arange(-2, 3)[None, :, None] * normal).astype(int)
        if (points[:, :, 0].min() < 0 or points[:, :, 0].max() >= width
                or points[:, :, 1].min() < 0 or points[:, :, 1].max() >= height):
            return draft
        coverage = float((grey[points[:, :, 1], points[:, :, 0]].sum(axis=1) >= 2).mean())
        if coverage < .85:
            return draft
        used.update((member["i"], member["j"]))
        evidence_lines.append({"member": member["id"], "grey_coverage": coverage})
    if used != set(snapped):
        return draft
    if all(np.linalg.norm(snapped[n["id"]] - [n["u"] * (width - 1), n["v"] * (height - 1)]) < .5
           for n in nodes):
        return draft
    result = deepcopy(draft)
    target_nodes = {n["id"]: n for n in result["image_model"]["nodes"]}
    for ident, centre in snapped.items():
        target_nodes[ident].update(u=float(centre[0]) / (width - 1), v=float(centre[1]) / (height - 1))
    evidence = {"method": "local-grey-joint-dots/v1", "original_nodes": deepcopy(nodes),
                "centres_px": [{"node": ident, "point": centre.tolist(),
                                "grey_ring_fraction": candidates[matches[ident]][1]}
                               for ident, centre in snapped.items()], "member_lines": evidence_lines}
    for entity in result.get("entities") or []:
        target, kind = entity.get("target") or {}, entity.get("kind")
        node_id = target.get("node") if kind == "node" else (target.get("support") or {}).get("node")
        if kind == "load":
            load_target = target.get("load") or target
            if load_target.get("collection", "nodal_loads") != "nodal_loads":
                continue
            loads = [load for case in model.get("load_cases") or []
                     if case.get("name") == load_target.get("case")
                     for load in case.get("nodal_loads") or []
                     if load.get("name") == load_target.get("name")]
            if len(loads) != 1:
                continue
            node_id = loads[0].get("node")
        geometry = None
        if kind in ("node", "support", "load") and node_id in target_nodes:
            node = target_nodes[node_id]
            geometry = {"point": [node["u"], node["v"]]}
        elif kind == "member":
            member = next((m for m in members if m["id"] == target.get("member")), None)
            if member:
                a, b = target_nodes[member["i"]], target_nodes[member["j"]]
                geometry = {"line": [[a["u"], a["v"]], [b["u"], b["v"]]]}
        if geometry is None:
            continue
        entity["original_observation"] = {"image_geometry": deepcopy(entity.get("image_geometry")),
                                           "confidence": entity.get("confidence")}
        entity.update(image_geometry=geometry, source="derived", verified=False, confidence=None,
                      recognition_confidence=None, position_refinement=deepcopy(evidence))
    issues = result.setdefault("issues", [])
    ident = "pixel-joint-review"
    while any(item.get("id") == ident for item in issues):
        ident += "-new"
    issues.append({"id": ident, "category": "low_confidence", "severity": "blocking", "status": "open",
                   "entity_refs": [f"node:{n['id']}" for n in nodes],
                   "message": "已根据局部节点圆点及相连灰色杆线提出位置候选，原坐标已保留。请核对全部节点、杆件连接和荷载作用点后确认。",
                   "resolution": None, "resolved_by": None})
    return result


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
