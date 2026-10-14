"""实际任务窗口与Runner交互，不读取用户配置或请求真实账户。"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from agent import Session  # noqa: E402
from desktop.chat_panel import ChatPanel  # noqa: E402
from desktop.worker import Runner  # noqa: E402


@pytest.fixture
def chat(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("FRAMELAB_SETTINGS_FILE", str(tmp_path/"settings.ini"))
    monkeypatch.setattr("credentials.load_api_key", lambda: None)
    panel = ChatPanel(Session(), Runner())
    yield panel
    assert panel.runner.wait(30000)
    panel.shutdown()
    panel.close()
    app.processEvents()


def test_task_button_opens_and_reuses_real_task_window(chat):
    """任务入口不能只打开原型，同一任务关闭重开应保留上下文。"""
    chat.open_task_dialog()
    dialog = chat._task_dialog
    assert dialog.isVisible() and dialog.task is None
    dialog.hide()
    chat.open_task_dialog()
    assert chat._task_dialog is dialog and dialog.isVisible()


def test_offline_task_reports_missing_model_without_faking_agents(chat):
    """离线空模型不能冒充已完成建模，必须等待用户或在线连接。"""
    chat.open_task_dialog()
    dialog = chat._task_dialog
    dialog.prompt.setPlainText("帮我建立框架")
    dialog.start()
    assert chat.runner.wait(30000)
    assert dialog.task.status == "waiting_input"
    assert not dialog.task.children and "在线主模型" in dialog.questions.text()
    assert not chat.session.model


def test_offline_review_runs_real_isolated_solver(chat):
    """检查任务真的求解独立候选，不能只在界面打印完成。"""
    from test_agent_tasks import cantilever_model
    assert chat.session.set_model(cantilever_model()).ok
    before = chat.session.model.copy()
    chat.open_task_dialog()
    dialog = chat._task_dialog
    dialog.prompt.setPlainText("求解并查询最大位移")
    dialog.start()
    assert not dialog.start_button.isEnabled()
    assert chat.runner.wait(30000)
    assert dialog.task.status == "completed"
    assert dialog.steps.topLevelItemCount() == 1
    assert "query_results" in dialog.evidence.toPlainText()
    assert chat.session.model == before and chat.session.solution is None
    assert dialog.start_button.isEnabled()


def test_task_window_shutdown_hides_and_closes_network_resources(chat):
    """主窗口关闭时不能遗留协作任务窗口。"""
    chat.open_task_dialog()
    dialog = chat._task_dialog
    dialog.prompt.setPlainText("检查")
    dialog.start()
    assert chat.runner.wait(30000)
    chat.shutdown()
    assert not dialog.isVisible()


def test_task_role_and_budget_locked_after_first_start(chat):
    """恢复必须保留原模型角色与累计预算，不能静默重置子任务。"""
    chat.open_task_dialog()
    dialog = chat._task_dialog
    dialog.prompt.setPlainText("检查")
    dialog.start()
    assert chat.runner.wait(30000)
    assert not dialog.limit.isEnabled()
    assert all(not combo.isEnabled() for combo in dialog.role_models.values())
    dialog.new_task()
    assert dialog.task is None and dialog.limit.isEnabled()


def test_changing_project_does_not_reuse_old_task_dialog(chat):
    """打开新工程后，旧任务不能继续覆盖新工程的同编号对象。"""
    chat.open_task_dialog()
    previous = chat._task_dialog
    chat.reset_session(Session())
    assert not previous.isVisible()
    chat.open_task_dialog()
    assert chat._task_dialog is not previous


def test_old_candidate_results_not_shown_as_current_values(chat):
    """候选版本变化后，旧查询数字只能留在带版本的审计中，不能冒充当前结果。"""
    from test_agent_tasks import cantilever_model
    assert chat.session.set_model(cantilever_model()).ok
    chat.open_task_dialog()
    dialog = chat._task_dialog
    dialog.prompt.setPlainText("求解最大位移")
    dialog.start()
    assert chat.runner.wait(30000)
    assert "实际工具结果" in dialog.reply.toPlainText()
    dialog.task.children["review"].conversation.session.model["nodes"][1]["x"] = 4
    dialog._done(dialog.task.snapshot())
    assert "实际工具结果" not in dialog.reply.toPlainText()
