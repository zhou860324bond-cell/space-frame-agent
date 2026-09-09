"""Small hostable control for deterministic image preprocessing."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from image_preprocess import preprocess_image, work_plane_payload


class ImagePreprocessWidget(QWidget):
    image_changed = Signal(dict)
    work_plane_confirmed = Signal(dict)

    def __init__(self, output_dir: str | Path, parent=None):
        super().__init__(parent)
        self.output_dir = Path(output_dir)
        self.source_path: str | None = None
        self.rotation = 0
        self.crop: tuple[int, int, int, int] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.info = QLabel("未预处理")
        self.plane = QComboBox()
        self.plane.addItems(["XZ", "XY", "YZ"])
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-1e9, 1e9)
        self.offset.setDecimals(6)
        self.rotate = QPushButton("顺时针旋转 90°")
        self.confirm_plane = QPushButton("确认工作平面")
        self.rotate.clicked.connect(self.rotate_clockwise)
        self.confirm_plane.clicked.connect(self._confirm_plane)
        for widget in (self.info, QLabel("平面"), self.plane, QLabel("偏移(m)"),
                       self.offset, self.rotate, self.confirm_plane):
            row.addWidget(widget)
        layout.addLayout(row)

        crop_row = QHBoxLayout()
        self.crop_boxes = [QSpinBox() for _ in range(4)]
        for label, box in zip(("左", "上", "右", "下"), self.crop_boxes):
            box.setRange(0, 0)
            crop_row.addWidget(QLabel(label))
            crop_row.addWidget(box)
        self.apply_crop = QPushButton("应用裁剪")
        self.apply_crop.clicked.connect(self._apply_crop)
        crop_row.addWidget(self.apply_crop)
        layout.addLayout(crop_row)

    def set_source(self, path: str) -> dict:
        self.source_path = path
        self.rotation = 0
        self.crop = None
        metadata = self._process()
        ow, oh = metadata["preprocessing"]["oriented_size_px"]
        for box, maximum in zip(self.crop_boxes, (ow - 1, oh - 1, ow, oh)):
            box.setRange(0, maximum)
        for box, value in zip(self.crop_boxes, (0, 0, ow, oh)):
            box.setValue(value)
        return metadata

    def rotate_clockwise(self) -> None:
        if self.source_path:
            self.rotation = (self.rotation + 1) % 4
            self._process()

    def _process(self) -> dict:
        result = preprocess_image(self.source_path or "", self.output_dir,
                                  crop=self.crop,
                                  rotation_quarters_cw=self.rotation)
        metadata = result.source_metadata()
        self.info.setText(f"{result.width_px}×{result.height_px}px · {result.original_format}")
        self.image_changed.emit(metadata)
        return metadata

    def _apply_crop(self) -> None:
        if self.source_path:
            self.crop = tuple(box.value() for box in self.crop_boxes)
            self._process()

    def _confirm_plane(self) -> None:
        self.work_plane_confirmed.emit(work_plane_payload(
            self.plane.currentText(), offset=self.offset.value(), confirmed=True))
