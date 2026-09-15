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


def test_check_results_do_not_leak_field_names_to_the_interface():
    """内核的说明是写给大模型看的，里面带着字段名。**原样印出来就是把内部
    接口漏给了用户**——和当初把"算前守门"印在对话框上是同一类错误。

    这一条是截图抓出来的：强度验算的警告原文写着"请在杆件上显式给
    mu_y / mu_z"，直接显示在结果面板的标题里。
    """
    import re

    from desktop import result_rows

    payloads = {
        "strength": {
            "members": [{"member": 1, "section": "C", "stress_ratio": 0.5,
                         "governs": "受压", "verdict": "通过"}],
            "cases": ["D"], "count": 1, "failed_members": [],
            "inconclusive_members": [1],
            "warnings": ["μ 由杆端释放推定。请在杆件上显式给 mu_y / mu_z，"
                         "或用特征值屈曲分析核对。"],
            "limitation": "只算正应力，特征值屈曲分析回答整体失稳。"},
        "symmetry": {"symmetric": True, "planes": [],
                     "advice": "未检测到关于坐标平面的对称性。",
                     "response_check": {"note": "还没有结果"}},
        "numbering": {"dofs": 18, "node_number_span": 3,
                      "half_bandwidth": {"current": 12, "after_rcm": 8},
                      "storage_entries": {"full": 324, "banded": 144,
                                          "skyline_current": 126,
                                          "skyline_after_rcm": 108,
                                          "sparse_nonzeros": 180},
                      "lecture": "满阵→等带宽→一维变带宽。", "note": "只在内部发生。"},
    }
    leftover = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)+")
    for kind, payload in payloads.items():
        title, cols, _, _ = result_rows.to_rows(kind, payload)
        text = title + " " + " ".join(cols)
        found = leftover.search(text)
        assert not found, f"{kind} 的界面文字里漏出了字段名：{found.group(0)}"
        assert "**" not in text, f"{kind} 的界面文字里留着 Markdown 记号"
    strength_summary = result_rows.engineering_summary(
        "strength", payloads["strength"])
    assert strength_summary["solve_status"] == "已完成 · 存在待复核项"
    assert strength_summary["severity"] == "unclear"


def test_a_failed_member_is_never_tinted_as_passing():
    """校核表要靠颜色扫。**红色只留给真的不合格**——"判不了"染成红的
    会让人以为结构有问题，那正是这一档要避免的误读。"""
    from desktop import result_rows

    payload = {"members": [{"member": 1}, {"member": 2}, {"member": 3}],
               "failed_members": [2], "inconclusive_members": [3]}
    marks = result_rows.row_marks("strength", payload)
    assert marks == [result_rows.PASS, result_rows.FAIL, result_rows.UNCLEAR]
    assert result_rows.row_marks("没见过的", payload) == []


def test_every_dialog_says_ok_in_the_same_language():
    """同一个程序里，材料对话框写着「确定」、截面对话框写着 "OK"——
    这种不一致一眼就能看见。

    Qt 的标准按钮取的是**系统语言**的文案，机器语言不是中文时就露出英文。
    所以按钮条一律走 `dialog_styles.button_box()`；这条测试盯的是有没有人
    绕过它自己 new 一个。
    """
    import re

    offenders = []
    for path in sorted((ROOT / "desktop").glob("*.py")):
        if path.name == "dialog_styles.py":
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"QDialogButtonBox\s*\(", text):
            line = text[:match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}")
    assert not offenders, (
        "这些地方自己创建了按钮条，请改用 dialog_styles.button_box()："
        + "、".join(offenders))
