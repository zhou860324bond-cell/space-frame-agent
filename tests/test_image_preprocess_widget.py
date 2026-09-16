"""MM2-02 desktop host connection smoke tests."""

import os

import pytest
from PIL import Image

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.image_preprocess_widget import ImagePreprocessWidget


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def test_widget_emits_derived_image_and_confirmed_plane(qt_app, tmp_path):
    source = tmp_path / "drawing.png"
    Image.new("RGB", (8, 6), "white").save(source)
    widget = ImagePreprocessWidget(tmp_path / "derived")
    images, planes = [], []
    widget.image_changed.connect(images.append)
    widget.work_plane_confirmed.connect(planes.append)
    widget.set_source(str(source))
    widget.rotate_clockwise()
    for box, value in zip(widget.crop_boxes, (1, 1, 7, 5), strict=True):
        box.setValue(value)
    widget.apply_crop.click()
    widget.plane.setCurrentText("YZ")
    widget.offset.setValue(2.5)
    widget.confirm_plane.click()
    assert len(images) == 3
    assert images[-1]["width_px"] == 4
    assert images[-1]["height_px"] == 6
    assert planes[-1]["plane"] == "YZ"
    assert planes[-1]["offset"] == 2.5
