"""草图建模交互中容易形成坏拓扑的边界。"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt                       # noqa: E402
from PySide6.QtGui import QKeyEvent                         # noqa: E402
from PySide6.QtWidgets import QApplication                  # noqa: E402

from desktop.sketch_dialog import SketchCanvas, SketchDialog  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def test_reversed_duplicate_line_is_not_added(qt_app):
    canvas = SketchCanvas()
    canvas.nodes = [(0.0, 0.0), (1.0, 0.0)]
    assert canvas._add_line(0, 1)
    assert not canvas._add_line(1, 0)
    assert canvas.lines == [(0, 1)]


def test_rectangle_like_operation_undoes_as_one_step(qt_app):
    canvas = SketchCanvas()
    canvas._remember()
    ids = [canvas._get_or_add_node(point) for point in
           ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))]
    for n1, n2 in zip(ids, ids[1:] + ids[:1], strict=False):
        canvas._add_line(n1, n2)
    canvas.undo()
    assert canvas.nodes == [] and canvas.lines == []


def test_escape_really_ends_the_current_drawing_command(qt_app):
    canvas = SketchCanvas()
    canvas.pending_line_node = 0
    canvas.pending_rect_corner = (1.0, 2.0)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                      Qt.KeyboardModifier.NoModifier)
    canvas.keyPressEvent(event)
    assert canvas.pending_line_node is None
    assert canvas.pending_rect_corner is None


def test_sketch_model_uses_the_properties_chosen_by_the_user(qt_app):
    dialog = SketchDialog(materials=["Q355", "Concrete"],
                          sections=["COLUMN", "BEAM"])
    dialog.cmb_material.setCurrentText("Concrete")
    dialog.cmb_section.setCurrentText("BEAM")
    data = dialog.get_model_data()
    assert data["material"] == "Concrete"
    assert data["section"] == "BEAM"
