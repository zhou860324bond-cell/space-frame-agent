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


def test_main_batch_launcher_never_reports_success_on_a_failed_regression():
    """一键入口不能在回归失败之后返回成功。

    **这条原先还钉着「回归必须跑在示例之前」，那一半被有意去掉了。**
    原来的顺序是先跑一千六百多项回归（几分钟），失败就直接
    "Examples were not run" 退出——于是任何一台有环境毛病的机器（没有 GPU、
    非中文 locale、缺系统库），用户等了几分钟，最后一眼都没见过这东西能干嘛。

    真正要守的是**退出码**：回归红了就不许返回成功。先给人看产品不影响这一点，
    回归照跑、结果照样决定退出码。所以改成按意图断言，不按顺序断言。
    """
    text = (Path(__file__).resolve().parent.parent / "run.bat").read_text(
        encoding="utf-8")
    assert 'set "RUN_CODE=%ERRORLEVEL%"' in text
    assert "exit /b %RUN_CODE%" in text
    tail = text[text.index("running the full regression"):]
    assert "pytest" in tail
    assert "exit /b 1" in tail, "回归失败必须非零退出"


def test_main_batch_launcher_shows_the_examples_before_the_regression():
    """示例要跑在回归前面——这是为「第一次打开」做的取舍。

    代价是零：回归紧接着就跑，退出码仍由它决定（见上一条）。收益是机器有
    毛病时，用户至少见过一次这东西能干嘛，而不是对着一屏红色测试输出猜。
    """
    text = (Path(__file__).resolve().parent.parent / "run.bat").read_text(
        encoding="utf-8")
    assert text.index("running the examples") < text.index(
        "running the full regression")


def test_main_batch_launcher_forwards_user_arguments():
    """脚本用 _inner 重新调用自己（为了把输出收进日志），必须透传用户参数。

    不透传的话 :inner 里看到的 %1 永远是 _inner，用户敲的 --quick 在这一步
    就没了——实测过，它静悄悄地跑完了全量回归。
    """
    text = (Path(__file__).resolve().parent.parent / "run.bat").read_text(
        encoding="utf-8")
    assert "_inner %*" in text, "重新调用自己时要带上 %*"
    assert '"%~2"=="--quick"' in text, "第一个参数是 _inner，用户参数从第二个起"

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


def test_the_contour_pipeline_is_checked_too():
    """OpenGL 正常不等于云图正常。

    云图这条链路（求解 → 内力 → 应力 → 分级切分 → 生成管）上任何一步坏了，
    在有屏幕的机器上表现都只是"图看着怪"，没人查得出来是哪一步。
    所以自检里单列一项，**而且连应力一起检**——应力多依赖截面的极端纤维
    距离，是另一处会断的地方。
    """
    from desktop import doctor

    names = [label for label, _ in doctor.CHECKS]
    assert "云图链路" in names
    check = dict(doctor.CHECKS)["云图链路"]
    ok, note = check()
    assert ok, note
    assert "sigma" in note, "应力那一路也要检到"


def test_the_doctor_runs_on_its_own_without_the_launcher():
    """`python -m desktop.doctor` 单独跑也要能用。

    走 app.py 时 sys.path 是它铺好的；直接跑这个模块则没有，求解器和云图
    两项会报"找不到 agent"——而真正的问题只是路径。**自检工具不该自己
    成为要排查的对象。**
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    # encoding 要写死成 utf-8：doctor 那边把 stdout 定死成了 UTF-8（它的输出
    # 全是中文，跟着系统代码页走会在英文 Windows 上直接 UnicodeEncodeError），
    # 这边就不能再用本机默认编码去解——中文 Windows 上默认是 gbk，解 UTF-8
    # 会 UnicodeDecodeError，text=True 吞掉异常后 stdout 直接变成 None。
    # 两头都定死，这条测试才不看机器脸色。
    got = subprocess.run([sys.executable, "-m", "desktop.doctor"],
                         cwd=root, capture_output=True, text=True,
                         encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONPATH": "",
                              "QT_QPA_PLATFORM": "offscreen"})
    assert "No module named" not in got.stdout, got.stdout
    assert "求解器" in got.stdout
