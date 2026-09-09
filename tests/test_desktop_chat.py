"""Agent 面板与后台线程的验证。

这两块是桌面端最容易出问题的地方，原因不一样：

* **后台线程**——出的是竞态。表现为偶发的崩溃或状态错乱，而且**现场没法复现**。
  所以这里不去"多跑几次看看会不会崩"（那种测试通过了也说明不了什么），
  而是把不变量写死：一次只跑一件、跑着的时候改模型的动作必须置灰、
  工作线程里的异常必须能传回主线程。

* **对话面板**——出的是状态不一致。Agent 在对话里求解了，而工具栏那边
  还以为没求解，于是树里没有结果分支、状态栏写着"未求解"、视口却画着变形。
  三处各说各的，比什么都没更新更让人困惑。

面板不算不画，所以这里验的是**转发和状态同步**，不碰像素。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
pytest.importorskip("pyvistaqt", reason="未安装 pyvistaqt，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyvista as pv                                        # noqa: E402

pv.OFF_SCREEN = True

from conftest import opengl_available                       # noqa: E402
pytestmark = pytest.mark.skipif(                             # noqa: E402
    not opengl_available(),
    reason="无可用 OpenGL，桌面端 VTK 视口无法初始化，跳过")

from PySide6.QtWidgets import QApplication                  # noqa: E402

from agent import Session                                   # noqa: E402
from desktop.demo_script import DEMO_PROMPT, demo_provider  # noqa: E402
from desktop.main_window import MainWindow                  # noqa: E402
from desktop.worker import Runner                           # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


# ------------------------------------------------- 后台线程

def test_a_job_runs_off_the_calling_thread(qt_app):
    import threading
    here = threading.get_ident()
    seen: list[int] = []
    runner = Runner()
    assert runner.submit(lambda: seen.append(threading.get_ident()))
    assert runner.wait()
    assert seen and seen[0] != here, "活儿必须离开界面线程，否则窗口会卡死"


def test_the_result_comes_back(qt_app):
    got: list = []
    runner = Runner()
    runner.submit(lambda: 6 * 7, on_done=got.append)
    assert runner.wait()
    assert got == [42]


def test_an_exception_in_the_worker_reaches_the_caller(qt_app):
    """工作线程里的异常不会自己冒到主线程。抓不住的话界面就永远转圈，
    那比报错更难查。"""
    failures: list[tuple[str, str]] = []
    runner = Runner()
    runner.submit(lambda: 1 / 0, on_failed=lambda k, m: failures.append((k, m)))
    assert runner.wait()
    assert failures and failures[0][0] == "ZeroDivisionError"


def test_only_one_job_at_a_time(qt_app):
    """连点两次求解，用户要的是"算一次"，不是"算两次"，更不是两个线程
    同时改同一个 Session——那会把模型改烂。"""
    import time
    runner = Runner()
    assert runner.submit(lambda: time.sleep(0.3))
    assert not runner.submit(lambda: None), "跑着的时候必须拒绝第二件"
    assert runner.wait()
    assert runner.submit(lambda: None), "跑完了就该能再提交"
    assert runner.wait()


# ------------------------------------------------- 面板与主窗口

def test_the_window_has_an_agent_panel(qt_app):
    """项目的立论是自然语言驱动。桌面端没有对话面板，
    最正式的那个界面就在否定项目本身。"""
    w = MainWindow()
    assert w.chat is not None
    assert w.chat_dock.windowTitle() == "AI 助手"


def test_the_panel_and_the_toolbar_share_one_session(qt_app):
    """两边各持一个 Session 的话，对话建的模型工具栏看不见，反之亦然。"""
    w = MainWindow()
    assert w.chat.session is w.session


def test_assistant_suggestion_prefills_without_sending(qt_app):
    w = MainWindow()
    before = len(w.session.history)
    w.chat._use_suggestion("检查当前模型")
    assert w.chat.entry.toPlainText() == "检查当前模型"
    assert len(w.session.history) == before


def test_offline_mode_is_announced_not_faked(qt_app, monkeypatch):
    """没有密钥时要说清楚现在是离线演示，而不是假装接着大模型。"""
    import credentials
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: None)
    w = MainWindow()
    w.chat._refresh_status()
    assert "离线" in w.chat.lbl_status.text()
    assert w.chat._offline


def test_the_key_is_never_shown(qt_app, monkeypatch):
    """密钥泄露最常见的路径是自己抄给别人——贴日志、发截图。
    界面上任何地方都不许出现它的原文。"""
    import credentials
    secret = "sk-thismustnotappearanywhere0123456789"
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: secret)
    w = MainWindow()
    w.chat._refresh_status()
    text = w.chat.lbl_status.text()
    assert secret not in text
    assert credentials.fingerprint(secret) in text, "但要能看出换没换钥匙"


def test_a_conversation_builds_and_solves_the_model(qt_app, monkeypatch):
    """一整轮走通：说一句话 → 调工具 → 模型建起来 → 求解 → 界面跟上。

    强制走离线后端：这条测试要验的是**链路**，不是大模型的发挥，
    也不该因为断网或欠费而变红。
    """
    import credentials
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: None)
    w = MainWindow()
    w.chat.entry.setText(DEMO_PROMPT)
    w.chat.send()
    assert w.runner.wait(60000), "对话没在时限内跑完"

    assert w.session.model.get("nodes"), "对话该把模型建起来"
    assert w.session.solution, "对话该把模型算出来"
    # 界面三处状态必须和 Session 一致
    assert w.result is not None and w.result.ok
    assert "未求解" not in w.lbl_solve.text()
    labels = [w.tree.topLevelItem(k).text(0)
              for k in range(w.tree.topLevelItemCount())]
    assert any(t == "结果" for t in labels), "树里该出现结果分支"


def test_the_tool_calls_are_shown(qt_app, monkeypatch):
    """把调用链摊开，是"这些数不是模型编的"最直接的证据。藏起来反而显得心虚。"""
    import credentials
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: None)
    w = MainWindow()
    w.chat.entry.setText("建个 24 米门式刚架并求解")
    w.chat.send()
    assert w.runner.wait(60000)
    shown = w.chat.view.toPlainText()
    for tool in ("generate_portal_frame", "set_load_cases", "solve_model"):
        assert tool in shown, f"{tool} 没显示出来"


def test_the_input_is_locked_while_thinking(qt_app):
    """跑着的时候还能再敲一句，就会有两个线程同时改一个 Session。"""
    w = MainWindow()
    w.chat._set_busy(True)
    assert not w.chat.entry.isEnabled()
    assert not w.act_new.isEnabled(), "求解途中点「新建」会把后台正在读的模型换掉"
    w.chat._set_busy(False)
    assert w.chat.entry.isEnabled()


def test_an_unrecognized_request_says_so(qt_app):
    """离线演示认不出来就直说。假装听懂了然后随便调点什么，比不懂更糟。"""
    from conversation import Conversation
    s = Session()
    reply = Conversation(demo_provider(s), session=s).ask("帮我订张机票")
    assert not reply.tool_calls, "认不出来就不该乱调工具"
    assert "离线演示" in reply.reply


def test_a_query_on_an_empty_model_builds_it_first(qt_app):
    """在空模型上查挠度只会得到一句没头没脑的报错。
    用户真正想要的显然是"建起来再看"——真的 Agent 也该这么做。"""
    from conversation import Conversation
    s = Session()
    reply = Conversation(demo_provider(s), session=s).ask("挠跨比满足吗")
    names = [n for n, _ in reply.tool_calls]
    assert "generate_portal_frame" in names
    assert names.index("generate_portal_frame") < names.index("query_results")


def test_a_failed_turn_explains_why(qt_app):
    """只显示"失败了"，用户既不知道该重试还是该改配置。"""
    w = MainWindow()
    w.chat._failed("TimeoutError", "connection timed out")
    shown = w.chat.view.toPlainText()
    assert "TimeoutError" in shown and "网络" in shown
