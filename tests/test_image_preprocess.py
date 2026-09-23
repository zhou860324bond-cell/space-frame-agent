"""MM2-02 work-plane mapping and image preprocessing tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from image_preprocess import (axis_mapping, image_to_model, preprocess_image,
                              work_plane_payload)


@pytest.mark.parametrize(
    ("plane", "expected"),
    [("XY", (1.0, 2.0, 7.0)),
     ("XZ", (1.0, 7.0, 2.0)),
     ("YZ", (7.0, 1.0, 2.0))],
)
def test_work_plane_mapping_uses_image_right_and_up(plane, expected):
    xyz = image_to_model(
        1.0, 0.0, width=11, height=21, anchor_uv=(0.0, 1.0),
        length_per_pixel=0.1, plane=plane, offset=7.0,
    )
    assert xyz == pytest.approx(expected)
    assert axis_mapping(plane)["image_right_sign"] == 1
    assert work_plane_payload(plane, offset=7, confirmed=True)["status"] == "confirmed"


def test_crop_and_rotation_are_reversible_and_do_not_overwrite_source(tmp_path):
    source = tmp_path / "drawing.png"
    Image.new("RGB", (8, 6), "white").save(source)
    before = source.read_bytes()
    result = preprocess_image(source, tmp_path / "derived",
                              crop=(1, 1, 7, 5), rotation_quarters_cw=1)
    assert (result.width_px, result.height_px) == (4, 6)
    assert Path(result.derived_path) != source
    assert source.read_bytes() == before
    point = (3.0, 2.0)
    assert result.derived_to_oriented(*result.oriented_to_derived(*point)) \
        == pytest.approx(point)
    metadata = result.source_metadata()
    assert metadata["preprocessing"]["crop"] == [1, 1, 7, 5]
    assert metadata["preprocessing"]["perspective_status"] == "unconfirmed"


def test_rejects_out_of_bounds_crop_and_oversized_image(tmp_path, monkeypatch):
    source = tmp_path / "drawing.webp"
    Image.new("RGB", (4, 3), "white").save(source)
    with pytest.raises(ValueError, match="裁剪范围"):
        preprocess_image(source, tmp_path / "out", crop=(0, 0, 5, 3))

    monkeypatch.setattr("image_preprocess.MAX_IMAGE_PIXELS", 5)
    with pytest.raises(ValueError, match="20 MP"):
        preprocess_image(source, tmp_path / "out")


# 四角在裁剪框外、四个在框内——保证两种 crop 参数下都有足够标记可核对。
MARKS = {(0, 0): (255, 0, 0), (19, 0): (0, 255, 0), (0, 14): (0, 0, 255),
         (19, 14): (255, 255, 0), (7, 9): (255, 0, 255),
         (3, 2): (0, 255, 255), (17, 12): (255, 255, 255),
         (12, 5): (255, 128, 0)}


def _marked_source(path):
    image = Image.new("RGB", (20, 15), (10, 10, 10))
    for point, color in MARKS.items():
        image.putpixel(point, color)
    image.save(path, format="PNG")


@pytest.mark.parametrize("quarters", [0, 1, 2, 3])
@pytest.mark.parametrize("crop", [None, (3, 2, 18, 13)])
def test_derived_mapping_agrees_with_what_pillow_actually_did(tmp_path, crop,
                                                              quarters):
    """独立对照：预测的位置必须和真图里像素真正落到的位置一致。

    往返可逆只证明这对函数互为逆；它们可以**一起**错，
    和 PIL 的裁剪/旋转约定差一个像素或差一次翻转，而往返照样自洽。
    所以这里比的是真实导出图的像素颜色，不是另一个公式。
    """
    source = tmp_path / "drawing.png"
    _marked_source(source)
    result = preprocess_image(source, tmp_path / "derived", crop=crop,
                              rotation_quarters_cw=quarters)
    derived = Image.open(result.derived_path).convert("RGB")
    left, top, right, bottom = result.crop
    checked = 0
    for (ox, oy), color in MARKS.items():
        if not (left <= ox < right and top <= oy < bottom):
            continue                      # 被裁掉的标记不参与
        checked += 1
        px, py = result.oriented_to_derived(ox, oy)
        assert derived.getpixel((round(px), round(py))) == color, (
            f"原图 {(ox, oy)} 预测落在 {(px, py)}，那里却不是这个标记")
    assert checked >= 2


@pytest.mark.parametrize("quarters", [0, 1, 2, 3])
def test_round_trip_is_exact_and_in_bounds_for_every_pixel(tmp_path, quarters):
    """裁剪框里每一个像素都要往返精确，并且正向必须落在导出图幅内。

    越界那一半是重点：往返自洽的映射照样可以把点送到图外。
    """
    source = tmp_path / "drawing.png"
    _marked_source(source)
    result = preprocess_image(source, tmp_path / "derived", crop=(3, 2, 18, 13),
                              rotation_quarters_cw=quarters)
    left, top, right, bottom = result.crop
    for oy in range(top, bottom):
        for ox in range(left, right):
            px, py = result.oriented_to_derived(ox, oy)
            assert 0 <= px <= result.width_px - 1
            assert 0 <= py <= result.height_px - 1
            assert result.derived_to_oriented(px, py) == pytest.approx((ox, oy))
