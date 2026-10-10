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


def test_failed_crop_preserves_last_valid_image_and_controls(qt_app, tmp_path):
    """防止无效裁剪在 Qt 回调中抛异常并留下新参数与旧图混用的状态。"""
    source = tmp_path / "drawing.png"
    Image.new("RGB", (8, 6), "white").save(source)
    widget = ImagePreprocessWidget(tmp_path / "derived")
    images, errors = [], []
    widget.image_changed.connect(images.append)
    widget.processing_failed.connect(errors.append)
    original = widget.set_source(str(source))
    widget.crop_boxes[0].setValue(7)
    widget.crop_boxes[2].setValue(2)
    widget.apply_crop.click()
    assert images == [original]
    assert widget.crop is None and widget.metadata == original
    assert [box.value() for box in widget.crop_boxes] == [0, 0, 8, 6]
    assert "裁剪" in errors[0]


def test_failed_replacement_preserves_valid_source(qt_app, tmp_path):
    """防止第二张损坏图片抢先覆盖 source_path，后续旋转作用于错误文件。"""
    source = tmp_path / "drawing.png"
    Image.new("RGB", (8, 6), "white").save(source)
    widget = ImagePreprocessWidget(tmp_path / "derived")
    original = widget.set_source(str(source))
    invalid = tmp_path / "invalid.png"
    invalid.write_text("invalid image")
    with pytest.raises(OSError):
        widget.set_source(str(invalid))
    assert widget.source_path == str(source) and widget.metadata == original
    widget.rotate_clockwise()
    assert widget.metadata["width_px"] == 6
