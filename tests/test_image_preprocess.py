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
