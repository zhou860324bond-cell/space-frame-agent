"""编码保险与桌面端自检的验证。

这两样都是"平时用不上、一旦用上就必须靠得住"的东西，所以更要测：

* `console.use_utf8()` —— Windows 控制台是 GBK，代码里的 ✓ ⚠ ↔ 打印出去会抛
  UnicodeEncodeError。**一次已经算完的分析被一个装饰性字符打死在打印环节**，
  这是最没道理的一类崩溃。这里验的是"切不成 UTF-8 时也不许崩"。

* `desktop.doctor` —— 用户只会说"打不开啊"。自检脚本要能在依赖缺失、
  绑定选错、没有 OpenGL 的机器上**照样跑完并说出人话**，
  而不是自己也崩在半路。所以每一项都得能容忍被检查的东西根本不存在。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from console import use_utf8


def test_main_batch_launcher_propagates_failures_and_stops_examples():
    """一键入口不能在 pytest 失败后继续跑示例，最后反而返回成功。"""
    text = (Path(__file__).resolve().parent.parent / "run.bat").read_text(
        encoding="utf-8")
    assert 'set "RUN_CODE=%ERRORLEVEL%"' in text
    assert "exit /b %RUN_CODE%" in text
    assert "ERROR: tests failed. Examples were not run." in text
    assert text.index("ERROR: tests failed") < text.index("examples\\run_json.py")


# ------------------------------------------------- 编码保险

def test_utf8_is_selected_when_possible():
    assert use_utf8().lower().replace("-", "") == "utf8"


def test_dangerous_characters_survive_a_gbk_stream(monkeypatch):
    """真拿一个 GBK 流来打印。没有这层保险，下面这行就是一次崩溃。"""
    raw = io.BytesIO()
    gbk = io.TextIOWrapper(raw, encoding="gbk")          # 就是 Windows 控制台
    monkeypatch.setattr(sys, "stdout", gbk)
    use_utf8()
    print("✓ 完成　⚠ 注意　↔ 双向　wL²/8")               # 一个字符都不许打死它
    sys.stdout.flush()
    assert raw.getvalue(), "该有输出"


def test_a_stream_without_reconfigure_is_tolerated(monkeypatch):
    """stdout 被重定向成别的对象是常事（日志、捕获），不能因此报错。"""
    class Plain:
        encoding = "ascii"
        def write(self, _s): return 0
        def flush(self): pass

    monkeypatch.setattr(sys, "stdout", Plain())
    monkeypatch.setattr(sys, "stderr", Plain())
    assert use_utf8() == "ascii"                          # 原样返回，不抛


# ------------------------------------------------- 桌面端自检


def test_pixel_only_application_font_is_normalised_to_points():
    from desktop.app import _normalise_application_font

    class Font:
        def __init__(self):
            self.value = None

        def pointSizeF(self):
            return -1.0

        def pixelSize(self):
            return 16

        def setPointSizeF(self, value):
            self.value = value

    class Screen:
        def logicalDotsPerInch(self):
            return 96.0

    class App:
        def __init__(self):
            self._font = Font()
            self.applied = None

        def font(self):
            return self._font

        def primaryScreen(self):
            return Screen()

        def setFont(self, font):
            self.applied = font

    app = App()
    assert _normalise_application_font(app) == pytest.approx(12.0)
    assert app._font.value == pytest.approx(12.0)
    assert app.applied is app._font

# 注意：doctor 本身不 import Qt，所以**没装 PySide6 的机器上它照样能跑**——
# 这正是它存在的意义，因此这一组测试不做 importorskip。

from desktop import doctor                                # noqa: E402


def test_the_binding_is_pinned_before_qt_is_touched():
    """QT_API 必须在 import Qt 之前就定下来。

    pyvistaqt 走 qtpy 选绑定，默认可能选中 PyQt5；而窗口代码直接 import
    PySide6。两套 Qt 在一个进程里必然出事，而且报错看不出根源。
    """
    import os
    assert os.environ.get("QT_API") == "pyside6"


def test_every_check_returns_a_verdict_and_a_sentence():
    """每一项都得说人话，而且**不许自己崩**——机器上什么都没装是常态。"""
    for label, check in doctor.CHECKS:
        ok, note = check()                                # 崩了这条就红
        assert isinstance(ok, bool), label
        assert isinstance(note, str) and note.strip(), label


def test_the_report_runs_end_to_end_and_says_something(capsys):
    code = doctor.report()
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert sys.executable in out, "得说清楚用的是哪个解释器"
    if code == 0:
        for label, _ in doctor.CHECKS:
            assert label in out, f"全通过时每一项都该列出来：{label}"
    assert "run_gui.bat" in out or "正常启动" in out, "得给出下一步"


def test_a_failing_check_does_not_stop_the_report(monkeypatch, capsys):
    """某一项的检查函数自己抛异常，报告也得继续往下走。"""
    def explodes():
        raise RuntimeError("检查函数自己炸了")

    monkeypatch.setattr(doctor, "CHECKS",
                        (("会炸的", explodes), ("Python", doctor._python)))
    assert doctor.report() == 1
    out = capsys.readouterr().out
    assert "检查本身出错" in out
    assert "Python" in out, "前一项炸了不该埋掉后一项"


def test_a_fatal_check_stops_early(monkeypatch, capsys):
    """Python/Qt/PyVista 挂了，后面必然跟着挂。报完根因就停，别刷一屏红字。"""
    monkeypatch.setattr(doctor, "CHECKS",
                        (("Python", lambda: (False, "太老")),
                         ("不该跑到", lambda: (True, "x"))))
    doctor.report()
    out = capsys.readouterr().out
    assert "不该跑到" not in out


def test_a_wrong_binding_is_called_out(monkeypatch):
    """选错绑定是"打不开"里最难自己看出来的一种，提示必须指名道姓。"""
    qtpy = pytest.importorskip("qtpy", reason="未安装 qtpy")
    monkeypatch.setattr(qtpy, "API_NAME", "PyQt5", raising=False)
    ok, note = doctor._binding()
    assert not ok
    assert "PyQt5" in note and "QT_API" in note


def test_the_kernel_check_actually_solves():
    """自检里的"求解器"一项必须真算一遍，不能只 import 一下就说没事。"""
    ok, note = doctor._kernel()
    assert ok, note
    assert "框架" in note
