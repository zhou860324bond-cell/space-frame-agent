"""界面动效：流光状态行、用时格式与「减少动效」开关。

后台求解或对话回合几十秒没有变化时，「还在跑」和「已经卡死」看起来一样：
对话面板只有一行静止的「思考中… 125 s」，求解只在状态栏写一句「求解中…」。
现在状态行仿 Codex CLI：标题带流光、后面写用时和怎么停；动画可以整体关掉。
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop import motion  # noqa: E402


@pytest.fixture(autouse=True)
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app
    motion.set_reduced(False)       # 全局开关，别漏给后面的测试


def test_elapsed_time_reads_in_minutes_and_hours():
    """原来一律写「125 s」，过了一分钟就得自己除以 60。"""
    assert motion.format_elapsed(0) == "0 秒"
    assert motion.format_elapsed(59.9) == "59 秒"
    assert motion.format_elapsed(65) == "1 分 05 秒"
    assert motion.format_elapsed(2 * 3600 + 3 * 60 + 9) == "2 小时 03 分 09 秒"
    assert motion.format_elapsed(-3) == "0 秒", "时钟回拨也不能写出负数"


def test_shimmer_band_is_a_smooth_bump_that_sweeps_left_to_right():
    """亮带中心亮度为 1，向两侧按余弦降到 0，出了半宽一律为 0；随时间从左扫到右。"""
    count = 12
    span = count + 2 * motion.SWEEP_PADDING
    # 让中心正好落在第 3 个字符上
    at = (3 + motion.SWEEP_PADDING) / span * motion.SWEEP_SECONDS + 1e-9
    levels = motion.shimmer_levels(count, at)
    assert levels[3] == pytest.approx(1.0)
    assert levels[2] == pytest.approx(levels[4]), "两侧对称"
    assert levels[2] < 1.0 and levels[1] < levels[2]
    assert levels[3 + 5] == pytest.approx(0.0, abs=1e-12) and levels[3 + 6] == 0.0
    # 一开始亮带在文字左边的空白里：文字全暗
    assert max(motion.shimmer_levels(count, 0.0)) == 0.0

    def peak(t):
        values = motion.shimmer_levels(count, t)
        return values.index(max(values))
    assert peak(0.9) < peak(1.1), "亮带向右走"
    assert motion.shimmer_levels(count, 0.5) == motion.shimmer_levels(
        count, 0.5 + motion.SWEEP_SECONDS), "按周期重复"
    assert motion.shimmer_levels(0, 1.0) == []


def test_blend_stops_short_of_the_highlight_colour():
    from PySide6.QtGui import QColor
    base, high = QColor(100, 100, 100), QColor(200, 0, 0)
    assert motion.blend(base, high, 0.0) == base
    assert motion.blend(base, high, 1.0) == QColor(190, 10, 10), "最亮处留一成底色"


def test_shimmer_label_only_animates_when_visible_and_not_reduced():
    """看不见、没有标题或减少动效时计时器必须停着，不能白占主线程。"""
    label = motion.ShimmerLabel()
    label.resize(300, 20)
    label.set_text("求解中", "12 秒")
    assert not label.animating(), "还没显示"
    label.show()
    QApplication.processEvents()
    assert label.animating()
    assert label.text() == "求解中　12 秒" and label.toolTip() == label.text()
    motion.set_reduced(True)
    assert not label.animating(), "减少动效后立即停下"
    label.grab()                    # 静止状态也要画得出来
    motion.set_reduced(False)
    assert label.animating()
    label.clear()
    assert not label.animating() and label.text() == ""
    label.set_text("求解中")
    label.hide()
    assert not label.animating()


def test_shimmer_restarts_when_the_step_changes():
    """标题换了（进入下一步）流光从头扫，看得出状态变了；只刷新用时不重来。"""
    label = motion.ShimmerLabel()
    label.set_text("正在思考", "1 秒")
    label._started -= 1.0
    started = label._started
    label.set_text("正在思考", "2 秒")
    assert label._started == started
    label.set_text("正在执行：求解", "3 秒")
    assert label._started > started


def test_running_icon_stops_spinning_when_motion_is_reduced():
    from desktop.chat_widgets import StatusIcon, TypingDots
    icon = StatusIcon("running")
    assert icon._timer.isActive()
    motion.set_reduced(True)
    assert not icon._timer.isActive() and icon.state() == "running", "仍显示运行中，只是不转"
    icon.grab()
    dots = TypingDots()
    dots.show()
    QApplication.processEvents()
    assert not dots._timer.isActive()
    motion.set_reduced(False)
    assert icon._timer.isActive() and dots._timer.isActive()


def test_chat_status_line_shows_shimmer_title_elapsed_and_how_to_stop():
    """对话回合的状态行：标题带流光，后面写「1 分 05 秒 · Esc 停止」。"""
    from desktop.chat_widgets import EntryWidget
    entry = {"type": "assistant", "segments": [], "plan": [], "status": "streaming",
             "activity": "正在执行：求解"}
    widget = EntryWidget(0, entry)
    assert isinstance(widget.status_label, motion.ShimmerLabel)
    assert widget.status_label.header() == "正在执行：求解"
    assert widget.status_label.detail() == "Esc 停止"
    entry["activity"] = "思考中…"
    entry["elapsed"] = 65
    widget.refresh()
    assert widget.status_label.header() == "思考中", "省略号交给流光表达"
    assert widget.status_label.detail() == "1 分 05 秒 · Esc 停止"
    entry["status"] = "done"
    widget.refresh()
    assert widget.status_label.text() == ""


# --------------------------------------------------------------------------- 主窗口

def _window():
    pytest.importorskip("pyvistaqt", reason="未安装 pyvistaqt，跳过主窗口测试")
    from conftest import opengl_available
    if not opengl_available():
        pytest.skip("无可用 OpenGL，桌面端 VTK 视口无法初始化")
    from desktop.main_window import MainWindow
    return MainWindow()


def test_long_task_status_line_shows_the_task_and_elapsed_time():
    """求解期间状态栏原来只写一句「求解中…」，几十秒不变；现在有任务状态行与用时。"""
    w = _window()
    try:
        w._on_busy(True)
        w.statusBar().showMessage("求解中…")
        assert not w.lbl_task.isHidden()
        assert w.lbl_task.header() == "求解中"
        w._task_started = time.monotonic() - 65
        w._sync_task_status()
        assert w.lbl_task.detail() == "1 分 05 秒"
        assert w._task_clock.isActive()
        w.statusBar().showMessage("正在停止任务；将于安全检查点结束，矩阵分解完成前请等待。")
        assert w.lbl_task.header().startswith("正在停止任务")
        w._on_busy(False)
        assert w.lbl_task.isHidden() and not w._task_clock.isActive()
        w.statusBar().showMessage("求解完成", 100)
        assert w.lbl_task.isHidden(), "空闲时的消息不进任务状态行"
        # 对话回合也会置忙，但不写状态栏：不出现任务状态行
        w._on_busy(True)
        assert w.lbl_task.isHidden()
        w._on_busy(False)
    finally:
        w.close()


def test_reduced_motion_is_a_view_command_that_survives_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMELAB_SETTINGS_FILE", str(tmp_path / "desktop.ini"))
    first = _window()
    try:
        first.restore_preferences()
        assert not motion.reduced()
        action = first.actions_by_name["reduce_motion"]
        action.setChecked(True)
        first.toggle_reduced_motion()
        assert motion.reduced()
    finally:
        first.close()
    motion.set_reduced(False)
    second = _window()
    try:
        second.restore_preferences()
        assert motion.reduced() and second.actions_by_name["reduce_motion"].isChecked()
    finally:
        second.close()
