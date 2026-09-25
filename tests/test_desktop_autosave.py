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


# --------------------------------------------------------------------------- 界面设置

@pytest.fixture
def fresh_settings(tmp_path, monkeypatch):
    path = tmp_path / "desktop.ini"
    monkeypatch.setenv("FRAMELAB_SETTINGS_FILE", str(path))
    return path


def test_display_choices_survive_a_restart(qt_app, fresh_settings):
    """色系、分级、分量、抽屉宽度：关掉再开要还是上次的样子。"""
    first = MainWindow()
    first.restore_preferences()
    first.results.levels.setCurrentIndex(first.results.levels.findData(12))
    first.results.palette.setCurrentIndex(first.results.palette.findData("rainbow"))
    first.component = "Vz"
    first.left_drawer.extent = 333
    first.close()

    second = MainWindow()
    assert second.results.display_options()["levels"] == 0, "没恢复之前是默认值"
    second.restore_preferences()
    options = second.results.display_options()
    assert options["levels"] == 12 and options["palette"] == "rainbow"
    assert second.result_display_options["levels"] == 12, "恢复后要同步到视口用的那份"
    assert second.component == "Vz"
    assert second.left_drawer.extent == 333


def test_an_ordinary_window_never_writes_the_settings(qt_app, fresh_settings):
    """只有程序入口恢复过设置的窗口才写回；测试里建的窗口关掉不能改设置。"""
    w = MainWindow()
    w.component = "T"
    w.close()
    assert not fresh_settings.exists() or "results" not in fresh_settings.read_text()


def test_unrecognised_saved_values_are_ignored(qt_app, fresh_settings):
    """旧版本存的、已经不存在的选项值不能让下拉框进入不存在的状态。"""
    from desktop.main_window import _settings

    s = _settings()
    s.setValue("results/display", '{"palette": "no-such-palette", "levels": 999}')
    s.setValue("results/component", "sigma_bogus")
    s.sync()
    w = MainWindow()
    before = w.results.display_options()
    w.restore_preferences()
    assert w.results.display_options()["palette"] == before["palette"]
    assert w.results.display_options()["levels"] == before["levels"]
    assert w.component == "M"
