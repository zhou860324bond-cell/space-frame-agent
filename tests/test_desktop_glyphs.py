"""界面字符串里不许出现字体画不出来的符号。

**中文界面字体没有 ✓。** Microsoft YaHei 不含 U+2713，也不含
✔ ✗ ✕ ✅ ❌ ⚠ ⛔ ⚙ ↔。写进界面的话用户看到一个空心方框，不崩不报错，
只是每一处状态标记都变成豆腐块——实测流程条上"1 建模 ✓"显示成"1 建模 □"，
对话面板、问题面板、截面优化、草图面板一共九种符号全中。

`QFontMetrics.inFont()` **判不了**：它按主字体查、不考虑回退，把"结""点"
"，"这些明明正常显示的常用汉字也报成缺失。所以这里的判法是**把字符画出来**，
和一个私用区码位（任何字体都没有，必然画成豆腐块）比像素。
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt                         # noqa: E402
from PySide6.QtGui import QImage, QPainter                   # noqa: E402
from PySide6.QtWidgets import QApplication                   # noqa: E402

from desktop import glyphs                                    # noqa: E402
from desktop.app import (_load_cjk_font,                      # noqa: E402
                         _normalise_application_font)

DESKTOP = pathlib.Path(__file__).resolve().parent.parent / "desktop"

#: 这个私用区码位任何字体都没有，画出来就是豆腐块的样子——拿它当参照。
MISSING = ""

#: ASCII、CJK、常用中文标点不用测：它们本来就在字体里，全测一遍只会拖慢。
def _worth_checking(ch: str) -> bool:
    code = ord(ch)
    if code < 0x2000:
        return False
    if 0x3000 <= code <= 0x303F:        # CJK 标点
        return False
    if 0x4E00 <= code <= 0x9FFF:        # 汉字
        return False
    if 0xFF00 <= code <= 0xFFEF:        # 全角字符
        return False
    return True


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    _normalise_application_font(instance)
    _load_cjk_font(instance)
    return instance


def draw(app, text: str) -> bytes:
    image = QImage(24, 24, QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    painter = QPainter(image)
    painter.setFont(app.font())
    painter.drawText(QRect(0, 0, 24, 24), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    return bytes(image.constBits())


def is_tofu(app, ch: str) -> bool:
    return draw(app, ch) == draw(app, MISSING)


def string_literals():
    """desktop/*.py 里所有的字符串字面量，连文件名和行号一起带出来。

    只看字面量，**不看注释**：注释里写 ✓ 是在说明这件事本身，那没问题。
    """
    for path in sorted(DESKTOP.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # 模块/类/函数的 docstring 也算——它会显示在帮助里
                yield path.name, node.lineno, node.value


def test_the_tofu_reference_really_is_tofu(app):
    """**闸门的自检。** 私用区码位画出来要和空格不一样，否则下面判不了。"""
    assert draw(app, MISSING) != draw(app, " "), (
        "私用区字符画成了空白，这套判法在当前环境下失效")


def test_the_safe_glyphs_really_render(app):
    """glyphs.py 里每一个都得画得出来——那是全界面取符号的唯一入口。"""
    broken = [(ch, hex(ord(ch))) for ch in glyphs.ALL if is_tofu(app, ch)]
    assert not broken, f"这些「安全符号」其实画不出来：{broken}"


def test_no_desktop_string_uses_a_glyph_the_font_cannot_draw(app):
    """扫 desktop/*.py 的全部字符串字面量。

    **新写一个界面字符串时用了字体没有的符号，这里就会红**，并指出文件、
    行号和码位——不必等用户截图来问"这个方框是什么"。
    """
    cache: dict[str, bool] = {}
    offenders = []
    for name, line, text in string_literals():
        if name == "glyphs.py":
            continue                    # 那个文件的 docstring 在举反例
        for ch in set(text):
            if not _worth_checking(ch):
                continue
            if ch not in cache:
                cache[ch] = is_tofu(app, ch)
            if cache[ch]:
                offenders.append(f"{name}:{line} U+{ord(ch):04X} {ch!r}")
    assert not offenders, (
        "这些界面字符串里的符号，当前字体画不出来，会显示成空心方框：\n  "
        + "\n  ".join(sorted(set(offenders)))
        + "\n从 desktop/glyphs.py 里取符号，别直接写。")
