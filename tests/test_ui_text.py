"""界面文字的质量闸。

这条标准是被用户当面指出来才立的：**我把代码注释的语气漏到界面上了**。
最典型的是"算前守门拦下了这个模型"——"算前守门"是代码里给那套校验起的
内部叫法，写在注释里挺好，印在用户对话框上就是黑话。

标准是三段式：**术语准确 + 说清原因 + 给出下一步**。

要同时提防两个方向。口语化固然不专业，但"错误代码 0x8007：操作失败"
这种听着正式、其实什么都没说的写法更糟——所以下面既查口语词，
也查提示是不是短到没有信息量。

**只查字符串字面量，不查注释和文档字符串。**
用 AST 取字面量，注释根本不在语法树里，天然被排除；
文档字符串单独识别出来跳过。代码注释保持说人话，那是给我们自己看的。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
UI_DIRS = ["desktop"]

# 开发者用词。印在界面上会让人觉得这软件是半成品
COLLOQUIAL = ["算不了", "没跑成", "跑一遍", "跑起来", "你的数", "读不了",
              "弄一下", "搞定", "崩了", "炸了", "白干"]

# 内部黑话。在代码里是好名字，在界面上没人懂
JARGON = ["算前守门", "守门", "兜底", "打死", "拍脑袋"]


def _literals() -> list[tuple[str, int, str]]:
    """取出所有**非文档字符串**的字符串字面量。"""
    out = []
    for folder in UI_DIRS:
        for path in sorted((ROOT / folder).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = getattr(node, "body", None)
                    if (body and isinstance(body[0], ast.Expr)
                            and isinstance(body[0].value, ast.Constant)
                            and isinstance(body[0].value.value, str)):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and id(node) not in docstrings
                        and len(node.value.strip()) >= 4):
                    out.append((path.name, node.lineno, node.value))
    return out


LITERALS = _literals()


def test_there_are_strings_to_check():
    """这条测试保护上面那个提取器：改坏了会让整组测试变成空跑而不报错。"""
    assert len(LITERALS) > 80, f"只取到 {len(LITERALS)} 条，提取器八成坏了"


@pytest.mark.parametrize("word", COLLOQUIAL)
def test_no_colloquial_words_in_the_interface(word):
    hits = [f"{f}:{n} {s[:60]!r}" for f, n, s in LITERALS if word in s]
    assert not hits, f"界面文字里出现口语词「{word}」：\n  " + "\n  ".join(hits)


@pytest.mark.parametrize("word", JARGON)
def test_no_internal_jargon_in_the_interface(word):
    """在代码里是好名字，在界面上没人懂。"""
    hits = [f"{f}:{n} {s[:60]!r}" for f, n, s in LITERALS if word in s]
    assert not hits, f"界面文字里出现内部黑话「{word}」：\n  " + "\n  ".join(hits)


def test_terminology_is_consistent():
    """杆件 / 构件 / 单元 只用一个。混着用在报告里是硬伤，界面同理。"""
    from desktop import commands

    shown = " ".join(c.label + c.tip for c in commands.COMMANDS)
    assert "构件" not in shown, "界面统一用「杆件」，不要混用「构件」"
    assert "单元" not in shown, "「单元」是有限元术语，面向用户说「杆件」"


def test_every_button_explains_itself_with_substance():
    """**专业不等于生硬。** "操作失败"这种听着正式、什么都没说的提示
    比口语更糟——所以说明要有实质内容，不能只是把标签重复一遍。"""
    from desktop import commands

    for c in commands.COMMANDS:
        assert len(c.tip) >= 6, f"{c.name} 的说明太短：{c.tip!r}"
        assert c.tip != c.label, f"{c.name} 的说明只是重复了标签"


def test_command_shortcuts_are_unique():
    """快捷键冲突会让用户按到的命令取决于 Qt 注册顺序。"""
    from desktop import commands

    shortcuts = [c.shortcut for c in commands.COMMANDS if c.shortcut]
    assert len(shortcuts) == len(set(shortcuts))


def test_result_columns_carry_units():
    """报告里漏单位是硬伤，界面同理。"""
    from desktop import result_rows

    for kind, payload in (("modal", {"frequencies": [2.0]}),
                          ("buckling", {"factors": [2.0]}),
                          ("deflection", {"max_deflection": {"value_mm": 3.0}})):
        _, cols, _, _ = result_rows.to_rows(kind, payload)
        joined = " ".join(cols)
        assert any(u in joined for u in ("Hz", "mm", "m)", "λ", "s)")), \
            f"{kind} 的表头没有单位：{cols}"


def test_global_qt_stylesheet_uses_point_sized_fonts():
    """px 字号会令 QFont.pointSize() 返回 -1，第三方控件复制时会报警。"""
    import re
    from desktop import theme

    assert not re.search(r"font-size\s*:\s*[\d.]+px", theme.STYLESHEET)
