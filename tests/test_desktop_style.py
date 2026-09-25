"""自绘控件与对话框尺寸。

两件事在这里验，都是"不会报错、只是看着不对"的那一类：

**一、样式表画不出形状。** 以前 ::indicator 只有颜色规则，于是勾选框的选中
态是一个纯色方块（没有对勾），单选钮走同一条规则被画成同样的方块——
用户分不出"多选"和"单选"，而这是两种不同的交互承诺。

**二、setMinimumSize 会盖掉布局算出来的最小高度。** 内容放不下时窗口不会
撑开，而是把控件按比例压扁：边界条件对话框实测需要 694px、代码里钉死
520px，六行自由度被压成 8px 高（文字本身要 17px），勾选框和标签直接重叠，
一次压扁 26 个控件。

所以这里不验"能不能画出来"，验的是**画出来的像素真的不一样**，以及
**每个对话框打开时都装得下自己的内容**。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect                      # noqa: E402
from PySide6.QtGui import QImage, QPainter                    # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel,          # noqa: E402
                               QStyle, QStyleOption)

from agent import Session                                      # noqa: E402
from desktop import qt_style                                   # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def style(app):
    # **不传 app.style()。** QProxyStyle 那个构造函数会接管基样式的
    # 所有权，之后应用换样式时基样式被删，代理手里就是野指针。
    return qt_style.FrameStyle()


def render(style, element, *, on=False, enabled=True, size=24):
    """把一个图元画到白底上，返回图像——**像素才是判据**。"""
    image = QImage(size, size, QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    option = QStyleOption()
    option.rect = QRect(0, 0, size, size)
    state = QStyle.StateFlag.State_None
    if enabled:
        state |= QStyle.StateFlag.State_Enabled
    if on:
        state |= QStyle.StateFlag.State_On
    option.state = state
    painter = QPainter(image)
    style.drawPrimitive(element, option, painter, None)
    painter.end()
    return image


def ink(image) -> set[tuple[int, int]]:
    """非白像素的位置。"""
    return {(x, y) for y in range(image.height())
            for x in range(image.width())
            if image.pixel(x, y) & 0xFFFFFF != 0xFFFFFF}


def palette(image, region=None) -> set[int]:
    """某个区域里出现过的颜色。

    **判"画没画出形状"要看颜色，不是看位置。** 未选中的框也铺满了填充色，
    所以"非白像素的位置集合"在选中前后是一样的——第一版测试就栽在这里，
    它报"两者相同"，而对勾其实画出来了。
    """
    rect = region or QRect(0, 0, image.width(), image.height())
    return {image.pixel(x, y) & 0xFFFFFF
            for y in range(rect.top(), rect.bottom() + 1)
            for x in range(rect.left(), rect.right() + 1)}


# --- 勾选框 ---------------------------------------------------------------

def test_a_checked_box_actually_draws_a_tick(style):
    """选中态必须比未选中态多出**一笔**，而不只是换个填充色。

    以前这里就是纯色方块：颜色变了，形状没变——远看是个色块，
    近看还是个色块。
    """
    element = QStyle.PrimitiveElement.PE_IndicatorCheckBox
    inner = QRect(7, 7, 10, 10)
    off = palette(render(style, element, on=False), inner)
    on = palette(render(style, element, on=True), inner)

    # 未选中：方框内部只有一种填充色。选中：填充色之外还得有勾的颜色。
    assert len(off) == 1, f"未选中的方框内部不该有别的东西：{off}"
    assert len(on) > 1, "选中之后方框内部还是纯色——只换了填充，没画出勾"


def test_a_disabled_box_is_drawn_differently(style):
    element = QStyle.PrimitiveElement.PE_IndicatorCheckBox
    enabled = render(style, element, on=True, enabled=True)
    disabled = render(style, element, on=True, enabled=False)
    assert ink(enabled) != ink(disabled) or _colors(enabled) != _colors(disabled)


def _colors(image):
    return {image.pixel(x, y) for x, y in ink(image)}


# --- 单选钮 ---------------------------------------------------------------

def test_a_radio_button_is_round_not_square(style):
    """**这是重点。** 方的单选钮和勾选框长得一样，那是在骗用户。

    判据是四个角：圆形指示器的四角应当是白的，方形的不是。
    """
    radio = render(style, QStyle.PrimitiveElement.PE_IndicatorRadioButton,
                   on=False)
    box = render(style, QStyle.PrimitiveElement.PE_IndicatorCheckBox, on=False)
    marks = ink(radio)
    corners = [(5, 5), (18, 5), (5, 18), (18, 18)]
    assert not any(c in marks for c in corners), (
        "单选钮的四角有墨，画成方的了")
    assert any(c in ink(box) for c in corners), (
        "勾选框的四角是空的，它该是方的")


def test_a_selected_radio_has_a_dot_in_the_middle(style):
    element = QStyle.PrimitiveElement.PE_IndicatorRadioButton
    middle = QRect(10, 10, 4, 4)
    off = palette(render(style, element, on=False), middle)
    on = palette(render(style, element, on=True), middle)
    # 点把这一小块整个盖住，所以**两边都是单一颜色**——判"有几种颜色"没用，
    # 要判颜色本身变没变。第一版写成 len(on) > 1，实测 1 == 1 直接红。
    assert len(off) == 1 and len(on) == 1, "圆心这一小块应当是纯色"
    assert off != on, "选中之后圆心颜色没变——没画出那个点"


# --- 数字框箭头 -----------------------------------------------------------

def test_the_spin_arrows_point_opposite_ways(style):
    """上下箭头要真的一上一下。两个画成一样的话，按哪个都像在猜。"""
    up = render(style, QStyle.PrimitiveElement.PE_IndicatorSpinUp)
    down = render(style, QStyle.PrimitiveElement.PE_IndicatorSpinDown)
    assert ink(up), "向上的箭头什么都没画"
    assert ink(up) != ink(down)
    # **判据是顶点在哪一头，不是重心。** "^" 和 "v" 的重心几乎一样
    # （第一版就是这么写的，实测 12.0 对 11.0，差一个像素，等于没判）。
    # 顶点那一行只有一两个像素，两条臂那一行有两处——数宽度就分得开。
    assert _row_width(up, top=True) < _row_width(up, top=False), "向上的不是尖朝上"
    assert _row_width(down, top=True) > _row_width(down, top=False), "向下的不是尖朝下"


def _row_width(image, top: bool) -> int:
    """墨迹最上/最下那一行的水平跨度。尖的那头窄，开口那头宽。"""
    marks = ink(image)
    row = min(y for _, y in marks) if top else max(y for _, y in marks)
    xs = [x for x, y in marks if y == row]
    return max(xs) - min(xs) + 1


def test_the_indicator_is_big_enough_to_read(style):
    """13px 下中文界面里的对勾只有几个像素，看不清是勾还是点。"""
    assert style.pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth) >= 15
    assert style.pixelMetric(
        QStyle.PixelMetric.PM_ExclusiveIndicatorWidth) >= 15


# --- 对话框装得下自己的内容 -----------------------------------------------

def dialog_session():
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0], [12, 0, 0]])
    s.define_materials_and_sections(
        materials=[{"name": "M", "E": 2.1e11, "nu": 0.3}],
        sections=[{"name": "S", "A": 0.02, "Iy": 2e-4, "Iz": 4e-4, "J": 1e-5}])
    s.add_members([[1, 2], [2, 3]], "S", "M")
    s.set_supports([1], fix=[1, 1, 1, 1, 1, 1])
    s.set_supports([2], fix=[0, 0, 1, 0, 0, 0])
    s.add_load_case("DL")
    s.set_member_load(1, [0, 0, -10e3], case_name="DL")
    s.add_step("S1", loads={"DL": "RAMP"})
    return s


def all_dialogs():
    from desktop.bc_dialog import BCDialog
    from desktop.load_code_dialog import (AreaLoadDialog, BCManagerDialog,
                                          CombinationDialog, LivePatternDialog)
    from desktop.step_dialog import AmplitudeDialog, StepManagerDialog

    return [
        ("BCDialog", lambda: BCDialog(3, current_fix=[1, 1, 1, 0, 1, 0])),
        ("StepManagerDialog", lambda: StepManagerDialog(dialog_session())),
        ("AmplitudeDialog", lambda: AmplitudeDialog(dialog_session())),
        ("CombinationDialog", lambda: CombinationDialog(dialog_session())),
        ("LivePatternDialog", lambda: LivePatternDialog(dialog_session())),
        ("AreaLoadDialog", lambda: AreaLoadDialog(dialog_session())),
        ("BCManagerDialog", lambda: BCManagerDialog(dialog_session())),
    ]


@pytest.mark.parametrize(("name", "make"), all_dialogs(),
                         ids=[n for n, _ in all_dialogs()])
def test_a_dialog_opens_at_least_as_tall_as_its_content(app, name, make):
    """打开时的高度不能小于布局算出来的最小高度。

    小了的话 Qt 不会撑开窗口，而是把控件按比例压扁——边界条件对话框实测
    一次压扁 26 个控件，勾选框只剩 8px 高，和标签重叠。
    """
    dialog = make()
    dialog.show()
    app.processEvents()
    needed = dialog.minimumSizeHint().height()
    assert dialog.height() >= needed - 1, (
        f"{name} 打开时只有 {dialog.height()}px，内容需要 {needed}px——"
        "控件会被压扁。style_dialog 的第二个参数是**初始**高度，不是硬下限，"
        "别用 setMinimumSize 去钉死它。")
    dialog.close()


@pytest.mark.parametrize(("name", "make"), all_dialogs(),
                         ids=[n for n, _ in all_dialogs()])
def test_no_wrapped_label_is_clipped(app, name, make):
    """自动换行的标签不能被切掉——被切掉的往往正是那句能力边界。"""
    dialog = make()
    dialog.show()
    app.processEvents()
    clipped = [str(w.text())[:30] for w in dialog.findChildren(QLabel)
               if w.isVisible() and w.wordWrap() and w.width() > 0
               and w.height() < w.heightForWidth(w.width()) - 1]
    dialog.close()
    assert not clipped, f"{name} 的这些说明文字被切掉了：{clipped}"


def test_installing_the_style_takes_effect(app):
    """装上之后应用的样式就是它。

    **这条不去卸载。** QProxyStyle 的所有权规则很容易踩：拿 baseStyle()
    再装回去会让 Qt 删掉一个仍被引用的对象，随后随便画点什么就访问违例——
    实测全量测试跑到 3% 崩掉。测试里只验装得上，不折腾卸载。
    """
    installed = qt_style.install(app)
    assert app.style() is installed
    assert isinstance(installed, qt_style.FrameStyle)
