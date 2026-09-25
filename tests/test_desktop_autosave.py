"""模型自动保存与启动恢复。

用 AI 对话建的模型从来不经过「保存」。真实发生过：一个 27 节点 42 杆的管
结构关掉窗口就没了，是从求解时顺手存的实验胶囊里翻回来的。
"""

import json
import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
pytest.importorskip("pyvistaqt", reason="未安装 pyvistaqt，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import opengl_available                       # noqa: E402

pytestmark = pytest.mark.skipif(
    not opengl_available(),
    reason="无可用 OpenGL，桌面端 VTK 视口无法初始化，跳过")

from PySide6.QtWidgets import QApplication, QMessageBox     # noqa: E402

from agent import Session                                   # noqa: E402
from desktop.main_window import MainWindow                  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def fresh_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMELAB_AUTOSAVE_DIR", str(tmp_path))
    return tmp_path


def _model() -> Session:
    s = Session()
    s.add_nodes([[0, 0, 0], [3, 0, 0]])
    s.add_members([[1, 2]])
    return s


def test_a_model_that_was_never_saved_is_written_to_the_autosave(qt_app, fresh_dir):
    w = MainWindow(_model())
    w.autosave.flush()
    saved = json.loads((fresh_dir / "unsaved_model.json").read_text(encoding="utf-8"))
    assert saved["model"] == w.session.model and saved["saved_at"]


def test_changes_are_batched_until_the_model_stops_changing(qt_app, fresh_dir):
    """连续操作只写最后一次：refresh 只是重新计时，不当场写盘。"""
    w = MainWindow(_model())
    w.autosave.discard()
    w.refresh()
    w.refresh()
    assert not (fresh_dir / "unsaved_model.json").exists()
    assert w.autosave._timer.isActive()


def test_saving_or_opening_a_file_clears_the_autosave(qt_app, fresh_dir):
    """已经落盘的内容不再备份，下次启动也不会多问一句。"""
    w = MainWindow(_model())
    w.autosave.flush()
    assert (fresh_dir / "unsaved_model.json").exists()
    w.autosave.mark_saved()
    assert not (fresh_dir / "unsaved_model.json").exists()
    w.autosave.flush()                      # 内容没变：不重写
    assert not (fresh_dir / "unsaved_model.json").exists()
    w.session.add_nodes([[6, 0, 0]])        # 保存之后又改了：该备份了
    w.autosave.flush()
    assert (fresh_dir / "unsaved_model.json").exists()


def test_an_empty_model_leaves_no_autosave(qt_app, fresh_dir):
    w = MainWindow(_model())
    w.autosave.flush()
    w.new_model()
    w.autosave.flush()
    assert not (fresh_dir / "unsaved_model.json").exists()


def test_closing_the_window_writes_the_last_version_immediately(qt_app, fresh_dir):
    w = MainWindow(_model())
    w.autosave.discard()
    w.session.add_nodes([[9, 0, 0]])
    w.close()
    saved = json.loads((fresh_dir / "unsaved_model.json").read_text(encoding="utf-8"))
    assert len(saved["model"]["nodes"]) == 3


def test_the_next_start_offers_to_restore_it(qt_app, fresh_dir, monkeypatch):
    first = MainWindow(_model())
    first.autosave.flush()
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    second = MainWindow()
    assert second.offer_autosave_restore()
    assert second.session.model["nodes"] == first.session.model["nodes"]


def test_declining_the_restore_discards_the_autosave(qt_app, fresh_dir, monkeypatch):
    MainWindow(_model()).autosave.flush()
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    w = MainWindow()
    assert not w.offer_autosave_restore()
    assert not w.session.model.get("nodes")
    assert not (fresh_dir / "unsaved_model.json").exists()


def test_a_window_with_a_model_already_open_does_not_ask(qt_app, fresh_dir, monkeypatch):
    MainWindow(_model()).autosave.flush()
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(1))
    assert not MainWindow(_model()).offer_autosave_restore()
    assert not asked
