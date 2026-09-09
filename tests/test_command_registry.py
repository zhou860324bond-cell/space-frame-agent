"""命令注册表的静态治理检查。

**不导入 Qt**：全部用 AST 分析源码，所以在没有图形环境的机器上也会运行。
桌面端的行为测试需要 Qt，一旦环境缺 Qt 就整体跳过，而"按钮点下去没反应"
恰恰是这类检查最该拦住的问题——所以把它放在不依赖 Qt 的地方。

守三件事：
1. 每个命令的 handler 在 MainWindow 上真的有同名方法（没有空按钮）
2. 没有 prefill 式 handler（用户明确反对把参数预填进对话框冒充功能）
3. 命令名与快捷键不重复（重复的快捷键会静默互相覆盖）
"""

from __future__ import annotations

import ast
import pathlib
from collections import Counter

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_COMMANDS = _ROOT / "desktop" / "commands.py"
_MAIN_WINDOW = _ROOT / "desktop" / "main_window.py"


def _commands() -> list[tuple[str, str, str, str]]:
    """从 commands.py 解析出 (name, label, handler, shortcut)。"""
    tree = ast.parse(_COMMANDS.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "Command"):
            continue
        positional = [a.value if isinstance(a, ast.Constant) else None
                      for a in node.args]
        keywords = {k.arg: (k.value.value if isinstance(k.value, ast.Constant)
                            else None) for k in node.keywords}
        name, label = positional[0], positional[1]
        handler = positional[4] if len(positional) > 4 else keywords.get("handler")
        shortcut = (positional[5] if len(positional) > 5
                    else keywords.get("shortcut", ""))
        found.append((name, label, handler, shortcut or ""))
    return found


def _main_window_methods() -> set[str]:
    tree = ast.parse(_MAIN_WINDOW.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "MainWindow")
    return {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_the_registry_is_not_empty():
    """解析失败会让下面几条静默通过——先确认真的读到了命令。"""
    assert len(_commands()) > 40


@pytest.mark.parametrize("name, label, handler, _shortcut", _commands())
def test_every_command_has_a_real_handler(name, label, handler, _shortcut):
    """功能区上的每个按钮都要能落到 MainWindow 的一个方法上。

    分派用的是 ``getattr(self, cmd.handler)``，晚绑定，写错名字要到用户
    点下去那一刻才发现。这条把它提前到静态检查。
    """
    assert handler, f"命令 {name}（{label}）没有 handler"
    assert handler in _main_window_methods(), (
        f"命令 {name}（{label}）的 handler {handler!r} 在 MainWindow 上不存在")


def test_no_prefill_handlers():
    """不允许把参数预填进对话框冒充功能——用户明确反对过这种做法。"""
    offenders = [(n, h) for n, _l, h, _s in _commands() if h == "prefill"]
    assert offenders == [], offenders


def test_command_names_are_unique():
    duplicates = [n for n, c in Counter(n for n, *_ in _commands()).items() if c > 1]
    assert duplicates == [], duplicates


def test_shortcuts_are_unique():
    """重复的快捷键不会报错，只会静默地互相覆盖。"""
    used = [s for *_rest, s in _commands() if s]
    duplicates = [s for s, c in Counter(used).items() if c > 1]
    assert duplicates == [], duplicates
