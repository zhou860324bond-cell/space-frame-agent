"""用局部彩色圆弧提出力矩中心候选；符号含义由视觉观察和人工审核决定。"""

import math

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import find_objects, label


def _fit_circle(points, max_radius):
    """确定性抽样只拟合覆盖半圆以上的圆弧，箭头头部不主导圆心。"""
    if len(points) < 24:
        return None
    if len(points) > 2048:
        points = points[np.linspace(0, len(points) - 1, 2048, dtype=int)]
    # 平移后拟合，避免大图绝对坐标使最小二乘病态。
    origin = points.mean(axis=0)
    xy = points - origin
    rng = np.random.default_rng(0)
    best = None
    for _ in range(192):
        a, b, c = xy[rng.choice(len(xy), 3, replace=False)]
        matrix = 2 * np.array([b - a, c - a])
        if abs(np.linalg.det(matrix)) < 1:
            continue
        centre = np.linalg.solve(matrix, [b @ b - a @ a, c @ c - a @ a])
        radius = float(np.linalg.norm(a - centre))
        if not 6 <= radius <= max_radius:
            continue
        inliers = np.abs(np.linalg.norm(xy - centre, axis=1) - radius) <= max(1.5, radius * .06)
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0] < len(xy) * .65:
        return None
    inliers = best[1]
    for _ in range(2):
        chosen = xy[inliers]
        coefficients, _, rank, _ = np.linalg.lstsq(
            np.c_[2 * chosen, np.ones(len(chosen))], (chosen * chosen).sum(axis=1), rcond=None)
        if rank != 3:
            return None
        centre = coefficients[:2]
        squared = coefficients[2] + centre @ centre
        if squared <= 0:
            return None
        radius = math.sqrt(squared)
        if not 6 <= radius <= max_radius:
            return None
        inliers = np.abs(np.linalg.norm(xy - centre, axis=1) - radius) <= max(1.5, radius * .06)
    angles = (np.arctan2(xy[inliers, 1] - centre[1], xy[inliers, 0] - centre[0]) + 2 * np.pi) % (2 * np.pi)
    occupied = np.unique((angles / (2 * np.pi) * 24).astype(int))
    if inliers.sum() < len(xy) * .65 or len(occupied) < 14:
        return None
    return centre + origin, radius, float(inliers.mean()), len(occupied)


def moment_centres(actions, image_path):
    """只在已观察为力矩的局部区域取唯一圆弧，灰度图或歧义时保留原观察。"""
    moments = [a for a in actions if a["kind"] == "moment"]
    if not moments or image_path is None:
        return {}
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        pixels = np.asarray(background.convert("RGB"), dtype=np.int16)
    height, width = pixels.shape[:2]
    # 颜色只分离笔画，不据颜色判断其是否外力、反力或力矩。
    ink = (pixels.max(axis=2) - pixels.min(axis=2) > 80) & (pixels.min(axis=2) < 160)
    results = {}
    for action in moments:
        px, py = action["point"][0] * (width - 1), action["point"][1] * (height - 1)
        x0, x1 = max(0, int(px - width * .12)), min(width, int(px + width * .12) + 1)
        y0, y1 = max(0, int(py - height * .2)), min(height, int(py + height * .2) + 1)
        crop = ink[y0:y1, x0:x1]
        components, _ = label(crop, structure=np.ones((3, 3), dtype=int))
        candidates = []
        for ident, bounds in enumerate(find_objects(components), 1):
            if bounds is None:
                continue
            y, x = bounds
            if x.start == 0 or y.start == 0 or x.stop == crop.shape[1] or y.stop == crop.shape[0]:
                continue
            if min(y.stop - y.start, x.stop - x.start) < 12:
                continue
            if not .65 <= (x.stop - x.start) / (y.stop - y.start) <= 1.5:
                continue
            yy, xx = np.nonzero(components[y, x] == ident)
            points = np.c_[xx + x0 + x.start, yy + y0 + y.start].astype(float)
            fitted = _fit_circle(points, min(width * .08, height * .18))
            if fitted is None:
                continue
            centre, radius, ratio, bins = fitted
            if np.linalg.norm(centre - [px, py]) > radius * 1.5:
                continue
            candidates.append({"method": "local-colour-circle/v1",
                "centre_px": centre.tolist(), "radius_px": radius,
                "inlier_fraction": ratio, "angular_bins": bins,
                "point": [float(centre[0]) / (width - 1), float(centre[1]) / (height - 1)]})
        if len(candidates) == 1:
            results[action["id"]] = candidates[0]
    return results
