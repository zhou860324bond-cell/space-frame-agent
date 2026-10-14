"""无需显卡的工作台回归，覆盖真实配置操作及历史和工程对象的生命周期。"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from agent import ScriptedProvider, Session  # noqa: E402
from conversation import Conversation  # noqa: E402
from desktop.chat_panel import ChatPanel  # noqa: E402
from desktop.worker import Runner  # noqa: E402
from model_profiles import ModelProfile  # noqa: E402


@pytest.fixture
def panel(monkeypatch, tmp_path):
    """用户密钥与配置均不读取，测试只使用假密钥和独立临时目录。"""
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("FRAMELAB_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setattr("credentials.load_api_key", lambda: None)
    widget = ChatPanel(Session(), Runner())
    yield widget
    assert widget.runner.wait(10000)
    widget.shutdown()
    widget.close()
    app.processEvents()


def test_model_switch_preserves_history_and_engineering_session(panel, monkeypatch):
    """旧配置开关会删除整个 Conversation；切换后必须继续原对话及工程。"""
    old = ScriptedProvider([{"content": "原对话"}])
    conversation = Conversation(old, panel.session)
    conversation.ask("请说明")
    panel.conversation = conversation
    profiles = [ModelProfile("a", "A", "same", "https://a.example.com", api_key="secret-a"),
                ModelProfile("b", "B", "same", "https://b.example.com", api_key="secret-b")]
    panel._profiles_changed(profiles, "a")
    replacement = ScriptedProvider([{"content": "后续对话"}])
    monkeypatch.setattr("configured_provider.create_provider", lambda p: replacement)
    panel.model_selector.setCurrentIndex(panel.model_selector.findData("b"))
    assert panel._ensure_conversation() is conversation
    assert conversation.session is panel.session and len(conversation.history) == 1
    assert conversation.provider is replacement
    assert conversation.build_messages("继续")[-2]["content"] == "原对话"


def test_failed_provider_initialization_keeps_draft_and_controls(panel, monkeypatch):
    """初始化失败曾在锁住输入后抛出；失败后草稿与模型必须保持。"""
    secret = "sk-should-never-show"
    p = ModelProfile("a", "A", "model", "https://a.example.com", api_key=secret)
    panel._profiles_changed([p], "a")
    def fail(profile):
        raise ValueError(f"bad credential {secret}")
    monkeypatch.setattr("configured_provider.create_provider", fail)
    panel.entry.setText("我的草稿")
    panel.send()
    assert panel.entry.toPlainText() == "我的草稿"
    assert panel.entry.isEnabled() and panel.btn_send.isEnabled()
    assert secret not in panel.view.toPlainText()
    assert panel.conversation is None


def test_chat_switch_keeps_individual_history_and_shared_model(panel):
    """多个会话不能混合历史，也不能偷偷替换用户正在编辑的工程。"""
    first = Conversation(ScriptedProvider([{"content": "第一段答复"}]), panel.session)
    first.ask("第一段")
    panel.conversation = first
    panel._append_agent("第一段答复")
    panel.entry.setText("第一段草稿")
    panel.new_chat()
    assert panel.conversation is None
    second = panel._ensure_conversation()
    assert second.session is first.session
    panel.entry.setText("第二段草稿")
    panel.chat_selector.setCurrentIndex(0)
    assert panel.conversation is first and len(first.history) == 1
    assert panel.entry.toPlainText() == "第一段草稿"
    assert "第一段答复" in panel.view.toPlainText()
    panel.chat_selector.setCurrentIndex(1)
    assert panel.conversation is second and panel.entry.toPlainText() == "第二段草稿"


def test_viewport_selection_requires_explicit_attach_and_expires(panel):
    """视口选择不能自动上传，坐标修改后也不能按旧编号继续引用。"""
    panel.session.model["nodes"] = [{"id": 1, "x": 0, "y": 0, "z": 0}]
    panel.set_selection("node", 1)
    assert not panel.references
    panel.attach_selection()
    assert panel.references == [("node", 1)]
    panel.session.model["nodes"][0]["x"] = 4
    panel.entry.setText("查询选中节点")
    panel.send()
    assert "过期" in panel.view.toPlainText()
    assert panel.entry.toPlainText() == "查询选中节点"
    assert not panel.runner.busy


def test_global_task_locks_model_chat_and_object_selection(panel):
    """求解或连接检测期间，也不能切换模型、对话及对象。"""
    panel.set_task_busy(True)
    assert not panel.model_selector.isEnabled()
    assert not panel.btn_settings.isEnabled() and not panel.chat_selector.isEnabled()
    assert not panel.btn_objects.isEnabled() and not panel.btn_send.isEnabled()
    panel.set_task_busy(False)
    assert panel.model_selector.isEnabled() and panel.btn_send.isEnabled()


def test_replacing_project_clears_all_old_chats_and_references(panel):
    """打开另一个工程后，旧工程历史和同编号引用不能传给新工程。"""
    panel.session.model["nodes"] = [{"id": 1, "x": 0, "y": 0, "z": 0}]
    panel.set_selection("node", 1)
    panel.attach_selection()
    panel._ensure_conversation()
    panel.new_chat()
    new_session = Session()
    panel.reset_session(new_session)
    assert panel.session is new_session and panel.conversation is None
    assert panel.chat_selector.count() == 1 and not panel.references
    assert panel._selection is None


def test_model_directory_can_be_read_before_selecting_model(panel, monkeypatch):
    """OpenAI 模板模型初始为空，目录按钮不能因此要求先填模型。"""
    from desktop.model_settings import ModelSettings
    dialog = ModelSettings(panel.runner, [], "", panel)
    dialog.preset.setCurrentIndex(1)
    dialog.key.setText("fake-secret")
    captured = []
    monkeypatch.setattr("configured_provider.list_models", lambda profile:
                        captured.append(profile) or ["available-model"])
    dialog._test(False)
    assert panel.runner.wait(10000)
    assert captured and captured[0].model == "directory-probe"
    assert dialog.model.currentText() == "available-model"
    assert "连接成功" in dialog.status.text()


def test_changing_provider_template_clears_previous_key(panel):
    """切换不同提供商不能把原密钥发送给另一家公司。"""
    from desktop.model_settings import ModelSettings
    dialog = ModelSettings(panel.runner, [], "", panel)
    dialog.key.setText("old-provider-secret")
    dialog.preset.setCurrentIndex(2)
    assert not dialog.key.text()
    assert dialog.protocol.currentData() == "anthropic"


def test_connection_save_preserves_current_chat(panel, monkeypatch):
    """应用内保存后立即可切换，原会话记录必须仍在。"""
    from desktop.model_settings import ModelSettings
    panel.conversation = Conversation(ScriptedProvider([]), panel.session)
    original = panel.conversation
    dialog = ModelSettings(panel.runner, [], "", panel)
    dialog.profiles_changed.connect(panel._profiles_changed)
    dialog.key.setText("test-credential")
    dialog.persist.setChecked(False)
    dialog._save()
    assert panel.profiles and panel.selected_profile == panel.profiles[0].id
    assert panel.conversation is original
    monkeypatch.setattr("configured_provider.create_provider", lambda p: ScriptedProvider([]))
    assert panel._ensure_conversation() is original
    assert "test-credential" not in panel.lbl_status.text()


def test_native_object_picker_filter_and_multiselect(panel):
    """选择窗口支持节点和杆件多选，搜索只隐藏条目、不丢已选对象。"""
    from PySide6.QtCore import Qt
    from desktop.object_picker import ObjectPicker
    model = {"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}],
             "members": [{"id": 2, "i": 1, "j": 3}]}
    dialog = ObjectPicker(model, [("node", 1)], panel)
    dialog.items.item(1).setCheckState(Qt.CheckState.Checked)
    dialog.search.setText("杆件")
    assert dialog.items.item(0).isHidden()
    assert dialog.references() == [("node", 1), ("member", 2)]


def test_missing_session_key_never_falls_back_to_offline(panel):
    """重启后的临时密钥不可用时，应提示配置，不能冒充在线成功。"""
    p = ModelProfile("a", "A", "model", "https://a.example.com", api_key="")
    panel._profiles_changed([p], "a")
    panel.entry.setText("工程请求")
    panel.send()
    assert not panel._offline and "API 密钥" in panel.view.toPlainText()
    assert panel.entry.toPlainText() == "工程请求"


def test_panel_keeps_conversation_as_main_area(panel):
    """对话区上方曾压了六层控件：备用连接与会话列表各占一行，引用是三个大按钮。
    现在备用连接和会话收进菜单，停止和清除只在有用时出现。"""
    panel.resize(420, 700)
    panel.show()
    QApplication.processEvents()
    assert not panel.backup_selector.isVisible() and not panel.chat_selector.isVisible()
    assert not panel.btn_stop.isVisible() and not panel.btn_clear.isVisible()
    assert panel.view.geometry().top() < panel.height() * 0.2
    panel._set_busy(True)
    assert panel.btn_stop.isVisible() and panel.btn_stop.isEnabled()
    panel._set_busy(False)
    panel.session.model["nodes"] = [{"id": 1, "x": 0, "y": 0, "z": 0}]
    panel.set_selection("node", 1)
    panel.attach_selection()
    assert panel.btn_clear.isVisible() and "节点 1" in panel.lbl_context.text()
    panel.hide()


def test_combo_boxes_draw_a_visible_arrow():
    """样式表给 drop-down 写了规则后 Qt 不再画箭头，下拉框和输入框长得一样。"""
    from pathlib import Path
    from desktop import theme
    assert "QComboBox::down-arrow" in theme.STYLESHEET
    assert theme._CHEVRON and Path(theme._CHEVRON).is_file()
    assert theme._CHEVRON in theme.STYLESHEET


def test_connection_dialog_has_single_primary_and_safe_delete(panel):
    """「删除连接」曾因排在第一个 autoDefault 按钮而被画成深蓝主按钮。"""
    from desktop.model_settings import ModelSettings
    dialog = ModelSettings(panel.runner, [], "", panel)
    assert dialog.save.isDefault() and dialog.save.property("role") == "primary"
    assert not dialog.remove.isDefault() and not dialog.remove.autoDefault()
    assert dialog.remove.property("role") == "danger"
    assert not any(b.isDefault() for b in (dialog.directory, dialog.probe, dialog.import_key))
