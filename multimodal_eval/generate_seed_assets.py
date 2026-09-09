"""Generate the original, deterministic MM2-00 seed corpus (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 128, 96
CASES = (
    ("seed_01_portal", ((18, 78), (18, 18), (110, 18), (110, 78)), ((0, 1), (1, 2), (2, 3)), ["clear", "portal"]),
    ("seed_02_gable", ((16, 80), (16, 42), (64, 14), (112, 42), (112, 80)), ((0, 1), (1, 2), (2, 3), (3, 4)), ["clear", "gable"]),
    ("seed_03_t_joint", ((16, 48), (112, 48), (64, 48), (64, 82)), ((0, 1), (2, 3)), ["intersection", "t_joint"]),
    ("seed_04_x_cross", ((20, 18), (108, 78), (20, 78), (108, 18)), ((0, 1), (2, 3)), ["intersection", "x_cross"]),
    ("seed_05_missing_scale", ((18, 76), (18, 24), (106, 24)), ((0, 1), (1, 2)), ["missing_dimension"]),
    ("seed_06_skewed", ((14, 80), (24, 20), (106, 27), (114, 82)), ((0, 1), (1, 2), (2, 3)), ["difficult", "skewed"]),
    ("seed_07_supports", ((20, 74), (20, 24), (106, 24), (106, 74)), ((0, 1), (1, 2), (2, 3)), ["clear", "supports"]),
    ("seed_08_load", ((18, 76), (18, 22), (108, 22), (108, 76)), ((0, 1), (1, 2), (2, 3)), ["clear", "load"]),
    ("seed_09_clear_portal", ((12, 82), (12, 16), (116, 16), (116, 82)), ((0, 1), (1, 2), (2, 3)), ["clear", "portal"]),
    ("seed_10_clear_gable", ((12, 82), (12, 42), (64, 10), (116, 42), (116, 82)), ((0, 1), (1, 2), (2, 3), (3, 4)), ["clear", "gable"]),
    ("seed_11_hand_portal", ((17, 81), (20, 23), (107, 19), (111, 79)), ((0, 1), (1, 2), (2, 3)), ["difficult", "handdrawn"]),
    ("seed_12_hand_gable", ((14, 80), (19, 44), (61, 13), (109, 39), (114, 82)), ((0, 1), (1, 2), (2, 3), (3, 4)), ["difficult", "handdrawn"]),
    ("seed_13_missing_beam", ((14, 50), (114, 50)), ((0, 1),), ["missing_dimension"]),
    ("seed_14_missing_portal", ((20, 80), (20, 20), (108, 20), (108, 80)), ((0, 1), (1, 2), (2, 3)), ["missing_dimension"]),
    ("seed_15_x_cross", ((14, 14), (114, 82), (14, 82), (114, 14)), ((0, 1), (2, 3)), ["intersection", "x_cross"]),
    ("seed_16_t_joint", ((12, 46), (116, 46), (62, 46), (62, 84)), ((0, 1), (2, 3)), ["intersection", "t_joint"]),
    ("seed_17_blur_support", ((18, 78), (18, 20), (110, 20), (110, 78)), ((0, 1), (1, 2), (2, 3)), ["difficult", "blurred_symbol", "supports"]),
    ("seed_18_blur_load", ((18, 78), (18, 20), (110, 20), (110, 78)), ((0, 1), (1, 2), (2, 3)), ["difficult", "blurred_symbol", "load"]),
    ("seed_19_photo_portal", ((18, 78), (18, 18), (110, 18), (110, 78)), ((0, 1), (1, 2), (2, 3)), ["difficult", "photo_noise"]),
    ("seed_20_photo_gable", ((16, 80), (16, 42), (64, 14), (112, 42), (112, 80)), ((0, 1), (1, 2), (2, 3), (3, 4)), ["difficult", "photo_noise"]),
    ("seed_21_skew_left", ((10, 84), (26, 18), (108, 30), (118, 82)), ((0, 1), (1, 2), (2, 3)), ["difficult", "skewed"]),
    ("seed_22_skew_right", ((18, 80), (10, 28), (104, 16), (118, 76)), ((0, 1), (1, 2), (2, 3)), ["difficult", "skewed"]),
    ("seed_23_support_fixed", ((16, 80), (16, 24), (112, 24), (112, 80)), ((0, 1), (1, 2), (2, 3)), ["clear", "supports"]),
    ("seed_24_support_mixed", ((18, 80), (18, 22), (108, 22), (108, 80)), ((0, 1), (1, 2), (2, 3)), ["supports"]),
    ("seed_25_nodal_load", ((16, 80), (16, 20), (112, 20), (112, 80)), ((0, 1), (1, 2), (2, 3)), ["clear", "load"]),
    ("seed_26_member_load", ((18, 78), (18, 18), (110, 18), (110, 78)), ((0, 1), (1, 2), (2, 3)), ["load"]),
    ("seed_27_clear_two_bay", ((8, 82), (8, 18), (64, 18), (120, 18), (120, 82)), ((0, 1), (1, 2), (2, 3), (3, 4)), ["clear", "multi_bay"]),
    ("seed_28_clear_brace", ((14, 82), (14, 18), (114, 18), (114, 82)), ((0, 1), (1, 2), (2, 3), (0, 2)), ["clear", "braced"]),
    ("seed_29_noisy_missing", ((16, 78), (16, 24), (108, 24)), ((0, 1), (1, 2)), ["difficult", "photo_noise", "missing_dimension"]),
    ("seed_30_hand_x", ((18, 16), (110, 80), (14, 78), (114, 20)), ((0, 1), (2, 3)), ["difficult", "handdrawn", "intersection", "x_cross"]),
)


def _png(nodes, members, labels=()) -> bytes:
    pixels = [[255] * (WIDTH * 3) for _ in range(HEIGHT)]

    def dot(x: int, y: int, radius: int = 2) -> None:
        for yy in range(max(0, y - radius), min(HEIGHT, y + radius + 1)):
            for xx in range(max(0, x - radius), min(WIDTH, x + radius + 1)):
                offset = xx * 3
                pixels[yy][offset:offset + 3] = [25, 35, 55]

    def line(a, b) -> None:
        x0, y0 = a
        x1, y1 = b
        steps = max(abs(x1 - x0), abs(y1 - y0))
        for index in range(steps + 1):
            t = index / steps if steps else 0.0
            dot(round(x0 + (x1 - x0) * t), round(y0 + (y1 - y0) * t), 1)

    for i, j in members:
        line(nodes[i], nodes[j])
    for node in nodes:
        dot(*node)
    if "blurred_symbol" in labels:
        for yy in range(68, 87):
            for xx in range(8, 29):
                if (xx + yy) % 3:
                    pixels[yy][xx * 3:xx * 3 + 3] = [155, 160, 170]
    if "photo_noise" in labels:
        # 固定伪噪点，模拟拍照纹理；不使用随机数，保证资产哈希可复现。
        for index in range(220):
            xx = (index * 47 + 11) % WIDTH
            yy = (index * 29 + 7) % HEIGHT
            shade = 205 + (index % 35)
            pixels[yy][xx * 3:xx * 3 + 3] = [shade, shade, shade]
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_intersection(a, b, c, d):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    x = ((x1 * y2 - y1 * x2) * (x3 - x4)
         - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4)
         - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return x, y


def main() -> None:
    for name in ("images", "ground_truth", "responses"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256(b"mm2-seed-prompt-v1").hexdigest()
    schema_hash = hashlib.sha256(b"space-frame-recognition-draft/v2").hexdigest()
    manifest = {"format": "space-frame-multimodal-eval/v1", "cases": []}
    for image_id, nodes, members, labels in CASES:
        image_path = ROOT / "images" / f"{image_id}.png"
        truth_path = ROOT / "ground_truth" / f"{image_id}.json"
        response_path = ROOT / "responses" / f"{image_id}.json"
        image_path.write_bytes(_png(nodes, members, labels))
        truth = {
            "image_id": image_id,
            "width_px": WIDTH,
            "height_px": HEIGHT,
            "nodes": [{"id": index + 1,
                       "point": [x / (WIDTH - 1), y / (HEIGHT - 1)]}
                      for index, (x, y) in enumerate(nodes)],
            "members": [{"id": index + 1, "i": i + 1, "j": j + 1}
                        for index, (i, j) in enumerate(members)],
            "intersections": [],
            "supports": [],
            "loads": [],
            "scale": ({"status": "unknown", "length_per_pixel": None}
                      if "missing_dimension" in labels else
                      {"status": "confirmed", "length_per_pixel": 0.05,
                       "unit": "m/px"}),
            "issues": [],
        }
        if "t_joint" in labels:
            x, y = nodes[2]
            truth["intersections"].append({"id": 1, "members": [1, 2],
                                            "point": [x / 127, y / 95],
                                            "decision": "connect", "node_id": 3})
        if "x_cross" in labels:
            x, y = _line_intersection(nodes[0], nodes[1], nodes[2], nodes[3])
            truth["intersections"].append({"id": 1, "members": [1, 2],
                                            "point": [x / 127, y / 95],
                                            "decision": "cross", "node_id": None})
        if "supports" in labels:
            last = len(nodes)
            truth["supports"] = [
                {"name": "left", "node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                {"name": "right", "node": last,
                 "fix": ([1, 1, 1, 1, 1, 1] if image_id != "seed_24_support_mixed"
                         else [1, 1, 1, 0, 0, 0])},
            ]
        if "load" in labels:
            if image_id == "seed_26_member_load":
                truth["loads"] = [{"name": "roof-udl", "case": "LC1",
                                    "collection": "member_loads", "member": 2,
                                    "w": [0, -2500, 0], "unit": "N/m"}]
            else:
                truth["loads"] = [{"name": "joint-P", "case": "LC1",
                                    "collection": "nodal_loads", "node": 2,
                                    "load": [0, -1000, 0, 0, 0, 0], "unit": "N"}]
        if "missing_dimension" in labels:
            truth["issues"].append({"category": "scale_unknown", "entity_refs": [],
                                    "severity": "blocking", "status": "open"})
        _write_json(truth_path, truth)
        _write_json(response_path, {
            "fixture": "deterministic-corpus-v2",
            "redaction": {"credentials": "removed", "user_metadata": "removed"},
            "request_fingerprint": {"provider": "fixture",
                                    "model": "deterministic-corpus-v2",
                                    "prompt_hash": prompt_hash,
                                    "schema_hash": schema_hash},
            "prediction": truth,
        })
        manifest["cases"].append({
            "image_id": image_id,
            "image_path": image_path.relative_to(ROOT).as_posix(),
            "image_hash": _digest(image_path),
            "ground_truth_path": truth_path.relative_to(ROOT).as_posix(),
            "ground_truth_hash": _digest(truth_path),
            "response_fixture_path": response_path.relative_to(ROOT).as_posix(),
            "response_hash": _digest(response_path),
            "provider": "fixture",
            "model": "deterministic-corpus-v2",
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "subset_labels": labels,
            "gate": True,
            "source": "project-generated-original",
            "license": "CC0-1.0",
        })
    _write_json(ROOT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
