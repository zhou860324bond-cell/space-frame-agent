"""中英文切换的闸。

这套东西最容易坏的地方不是翻错，而是**加了新按钮忘了加翻译**：
界面切到英文后半中半英，看着像没做完。所以这里盯的是覆盖率，
不是逐条校对译文。
"""

from __future__ import annotations

import pytest

from desktop import commands, i18n
from tests.test_desktop_window import built, qt_app            # noqa: F401


@pytest.fixture(autouse=True)
def _restore_language():
    yield
    i18n.set_language("zh")


def test_every_command_label_has_a_translation():
    missing = [c.label for c in commands.COMMANDS if c.label not in i18n.ZH_EN]
    assert not missing, f"这些按钮没有英文：{missing}——加了按钮要顺手加一条翻译"


def test_no_translation_is_blank():
    """宁可露出中文，也不能让按钮变成没有字。"""
    blank = [k for k, v in i18n.ZH_EN.items() if not v.strip()]
    assert not blank, f"这些词翻成了空：{blank}"


def test_translations_are_not_identical_to_the_source():
    """翻译表里留着一模一样的中文，说明那条根本没翻。"""
    same = [k for k, v in i18n.ZH_EN.items() if k == v]
    assert not same, f"这些词没真的翻：{same}"


def test_switching_back_and_forth_is_lossless():
    assert i18n.language() == "zh"
    assert i18n.tr("求解") == "求解"
    assert i18n.set_language("en") is True
    assert i18n.set_language("en") is False, "没变就不该说变了"
    assert i18n.tr("求解") == "Solve"
    assert i18n.set_language("zh") is True
    assert i18n.tr("求解") == "求解"


def test_unknown_text_survives_untranslated():
    """表里没有的词原样返回——半中半英也好过一片空白。"""
    i18n.set_language("en")
    assert i18n.tr("某个还没翻的词") == "某个还没翻的词"


def test_an_unknown_language_is_refused():
    assert i18n.set_language("fr") is False
    assert i18n.language() == "zh"


def test_the_window_really_switches_and_switches_back(qt_app):
    """整窗切换：动作文字要真的变成英文，切回来要一字不差地还原。"""
    from desktop.main_window import MainWindow

    window = MainWindow(built())
    before = {name: act.text() for name, act in window.actions_by_name.items()}

    window.actions_by_name["lang"].setChecked(True)
    window.toggle_language()
    assert window.actions_by_name["solve"].text() == "Solve"
    assert window.actions_by_name["report"].text() == "Report"

    window.actions_by_name["lang"].setChecked(False)
    window.toggle_language()
    after = {name: act.text() for name, act in window.actions_by_name.items()}
    assert after == before, "切回中文后有文字没还原"


def test_load_value_labels_can_be_switched_off(qt_app):
    """载荷数值标注可以关掉，而箭头要留着。"""
    from desktop.main_window import MainWindow

    window = MainWindow(built())
    assert window.viewport.load_labels is True
    assert window.actions_by_name["load_labels"].isChecked() is True

    window.actions_by_name["load_labels"].setChecked(False)
    window.toggle_load_labels()
    assert window.viewport.load_labels is False
    assert window.viewport.set_load_labels(False) is False, "没变就不该说变了"
