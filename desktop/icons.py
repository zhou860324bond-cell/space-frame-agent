"""图标。**全部用 QPainter 画出来，不带任何图片资源。**

三个理由：

* 矢量的，任何缩放和高 DPI 下都不糊，不用为每个尺寸备一套 png；
* 颜色跟着主题走，换配色不用重新导图；
* 仓库里不多几十个二进制文件——课程作业交出去，别人 clone 下来就是全的。

画的时候有一条纪律：**图标要画这个软件真正在做的事**，不要拿通用的
"齿轮＝设置、放大镜＝查询"来凑数。求解画的是刚架加荷载箭头，屈曲画的是
压弯的柱，模态画的是振型曲线，包络画的是两条曲线夹出来的带。一个学结构的人
扫一眼就该认得出来，这比好看重要。

坐标统一在 32×32 的画布上写，`icon()` 负责缩放到实际尺寸——所以下面
每个函数里的数字都可以按"三十二分之几"来读。
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QIcon, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)

from . import theme

BOX = 32.0                      # 设计画布边长，所有坐标按它来写

_LINE = QColor(theme.INK)
_DIM = QColor(theme.INK_MUTED)
_HOT = QColor(theme.ACCENT)
_WARM = QColor("#e0a458")


def _pen(color: QColor, width: float = 2.0, cap=Qt.PenCapStyle.RoundCap) -> QPen:
    p = QPen(color, width)
    p.setCapStyle(cap)
    p.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return p


def _arrow(g: QPainter, x: float, y0: float, y1: float,
           color: QColor = None, head: float = 3.0) -> None:
    """一支竖直箭头，从 y0 指到 y1。荷载符号到处要用。"""
    color = color or _WARM
    g.setPen(_pen(color, 1.8))
    g.drawLine(QPointF(x, y0), QPointF(x, y1))
    g.setBrush(QBrush(color))
    g.setPen(Qt.PenStyle.NoPen)
    sign = 1 if y1 > y0 else -1
    g.drawPolygon(QPolygonF([QPointF(x, y1),
                             QPointF(x - head, y1 - sign * head * 1.4),
                             QPointF(x + head, y1 - sign * head * 1.4)]))


def _pin(g: QPainter, x: float, y: float, size: float = 4.0) -> None:
    """铰支座三角。"""
    g.setPen(Qt.PenStyle.NoPen)
    g.setBrush(QBrush(_DIM))
    g.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x - size, y + size * 1.5),
                             QPointF(x + size, y + size * 1.5)]))


def _curve(g: QPainter, pts: list[tuple[float, float]], color: QColor,
           width: float = 2.0) -> QPainterPath:
    """过一串点的平滑曲线。振型、弯矩图都用它。"""
    path = QPainterPath(QPointF(*pts[0]))
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        path.cubicTo(QPointF((x0 + x1) / 2, y0), QPointF((x0 + x1) / 2, y1),
                     QPointF(x1, y1))
    g.setPen(_pen(color, width))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawPath(path)
    return path


# ------------------------------------------------- 具体图标
# 每个函数在 32×32 的画布上画一个图标

def _frame(g: QPainter) -> None:
    """多层多跨框架：两层两跨的骨架。"""
    g.setPen(_pen(_LINE, 2.0))
    for x in (6, 16, 26):
        g.drawLine(QPointF(x, 7), QPointF(x, 26))
    for y in (7, 16.5, 26):
        g.drawLine(QPointF(6, y), QPointF(26, y))
    _pin(g, 6, 26); _pin(g, 16, 26); _pin(g, 26, 26)


def _portal(g: QPainter) -> None:
    """门式刚架：坡屋面 + 两根柱。"""
    g.setPen(_pen(_LINE, 2.0))
    g.drawLine(QPointF(5, 11), QPointF(16, 5))     # 左斜梁
    g.drawLine(QPointF(16, 5), QPointF(27, 11))    # 右斜梁
    g.drawLine(QPointF(5, 11), QPointF(5, 24))
    g.drawLine(QPointF(27, 11), QPointF(27, 24))
    _pin(g, 5, 24); _pin(g, 27, 24)


def _node(g: QPainter) -> None:
    g.setPen(_pen(_DIM, 1.4))
    g.drawLine(QPointF(6, 22), QPointF(26, 10))
    for x, y in ((6, 22), (26, 10)):
        g.setBrush(QBrush(_DIM)); g.setPen(Qt.PenStyle.NoPen)
        g.drawEllipse(QPointF(x, y), 2.2, 2.2)
    g.setBrush(QBrush(_HOT)); g.setPen(Qt.PenStyle.NoPen)
    g.drawEllipse(QPointF(16, 16), 4.0, 4.0)


def _member(g: QPainter) -> None:
    g.setPen(_pen(_HOT, 3.0))
    g.drawLine(QPointF(7, 24), QPointF(25, 8))
    g.setBrush(QBrush(_LINE)); g.setPen(Qt.PenStyle.NoPen)
    g.drawEllipse(QPointF(7, 24), 2.6, 2.6)
    g.drawEllipse(QPointF(25, 8), 2.6, 2.6)


def _remove(g: QPainter) -> None:
    g.setPen(_pen(_DIM, 2.0))
    g.drawLine(QPointF(7, 24), QPointF(25, 8))
    g.setPen(_pen(_WARM, 2.6))
    g.drawLine(QPointF(11, 10), QPointF(23, 22))
    g.drawLine(QPointF(23, 10), QPointF(11, 22))


def _section(g: QPainter) -> None:
    """工字钢截面——材料与截面这一组的标志。"""
    g.setPen(Qt.PenStyle.NoPen)
    g.setBrush(QBrush(_LINE))
    g.drawRect(QRectF(7, 7, 18, 3.5))       # 上翼缘
    g.drawRect(QRectF(14.2, 10.5, 3.6, 11)) # 腹板
    g.drawRect(QRectF(7, 21.5, 18, 3.5))    # 下翼缘


def _support(g: QPainter) -> None:
    g.setPen(_pen(_LINE, 2.0))
    g.drawLine(QPointF(16, 6), QPointF(16, 20))
    _pin(g, 16, 20, 6)
    g.setPen(_pen(_DIM, 1.6))
    g.drawLine(QPointF(6, 29), QPointF(26, 29))
    for x in range(8, 26, 4):               # 地面剖面线
        g.drawLine(QPointF(x, 29), QPointF(x - 2.5, 32))


def _load(g: QPainter) -> None:
    """均布荷载压在梁上。"""
    g.setPen(_pen(_LINE, 2.2))
    g.drawLine(QPointF(5, 22), QPointF(27, 22))
    for x in (8, 13.5, 19, 24.5):
        _arrow(g, x, 8, 20)


def _combo(g: QPainter) -> None:
    """两个工况按分项系数合成一个：两条汇成一条，粗细表示"合起来更大"。

    小尺寸下细节会糊，所以只留"二合一"这一个意思，不画分项系数。
    """
    g.setPen(_pen(_DIM, 2.2))
    g.drawLine(QPointF(5, 8), QPointF(15, 16))
    g.setPen(_pen(_HOT, 2.2))
    g.drawLine(QPointF(5, 24), QPointF(15, 16))
    g.setPen(_pen(_LINE, 3.2))
    g.drawLine(QPointF(15, 16), QPointF(24, 16))
    g.setBrush(QBrush(_LINE)); g.setPen(Qt.PenStyle.NoPen)
    g.drawPolygon(QPolygonF([QPointF(28, 16), QPointF(23, 12.5),
                             QPointF(23, 19.5)]))


def _gravity(g: QPainter) -> None:
    g.setPen(_pen(_LINE, 2.0))
    g.drawRect(QRectF(10, 5, 12, 10))
    _arrow(g, 16, 17, 27)


def _solve(g: QPainter) -> None:
    """刚架 + 荷载：按下去就开始算。"""
    g.setPen(_pen(_LINE, 2.0))
    g.drawLine(QPointF(7, 13), QPointF(25, 13))
    g.drawLine(QPointF(7, 13), QPointF(7, 25))
    g.drawLine(QPointF(25, 13), QPointF(25, 25))
    _pin(g, 7, 25); _pin(g, 25, 25)
    for x in (11, 16, 21):
        _arrow(g, x, 4, 11)


def _geometry(g: QPainter) -> None:
    """模型视图：只有几何和支座，**不带荷载箭头**——那是 solve 的图标。
    两个图案必须一眼分得开，否则用户不知道自己按的是"看模型"还是"开始算"。"""
    g.setPen(_pen(_LINE, 2.0))
    g.drawLine(QPointF(7, 10), QPointF(25, 10))
    g.drawLine(QPointF(7, 10), QPointF(7, 24))
    g.drawLine(QPointF(25, 10), QPointF(25, 24))
    g.drawLine(QPointF(7, 17), QPointF(25, 17))
    _pin(g, 7, 24); _pin(g, 25, 24)
    g.setBrush(QBrush(_HOT)); g.setPen(Qt.PenStyle.NoPen)
    for x, y in ((7, 10), (25, 10), (7, 17), (25, 17)):
        g.drawEllipse(QPointF(x, y), 1.9, 1.9)


def _modal(g: QPainter) -> None:
    """一阶振型。"""
    g.setPen(_pen(_DIM, 1.4))
    g.drawLine(QPointF(16, 5), QPointF(16, 27))
    _curve(g, [(16, 27), (24, 20), (9, 12), (20, 5)], _HOT, 2.4)


def _buckling(g: QPainter) -> None:
    """压弯失稳的柱：上下各一支压力，柱身鼓出去。

    箭头要和柱身分开一点。挤在一起时看不出"压力把柱压弯了"，
    只看得见一团线。
    """
    g.setPen(_pen(_DIM, 1.2, Qt.PenCapStyle.FlatCap))
    g.drawLine(QPointF(13, 9), QPointF(13, 25))     # 原始位置（细线）
    _curve(g, [(13, 9), (21, 17), (13, 25)], _HOT, 2.6)
    _arrow(g, 13, 2, 7)
    _arrow(g, 13, 32, 27)


def _sweep(g: QPainter) -> None:
    """参数扫描：一串逐渐变化的结果柱。"""
    g.setPen(Qt.PenStyle.NoPen)
    for i, h in enumerate((6, 10, 14, 19, 23)):
        g.setBrush(QBrush(_HOT if i == 3 else _DIM))
        g.drawRect(QRectF(5.5 + i * 4.6, 27 - h, 3.4, h))


def _compare(g: QPainter) -> None:
    """两个求解器的结果并排比。"""
    g.setPen(Qt.PenStyle.NoPen)
    g.setBrush(QBrush(_LINE))
    g.drawRect(QRectF(5, 10, 9, 16))
    g.setBrush(QBrush(_HOT))
    g.drawRect(QRectF(18, 7, 9, 19))
    g.setPen(_pen(_DIM, 1.4))
    g.drawLine(QPointF(14.5, 18), QPointF(17.5, 18))


def _deformed(g: QPainter) -> None:
    """未变形（虚线）与变形后（实线）叠在一起。"""
    g.setPen(QPen(_DIM, 1.4, Qt.PenStyle.DashLine))
    g.drawLine(QPointF(5, 12), QPointF(27, 12))
    _curve(g, [(5, 12), (16, 23), (27, 12)], _HOT, 2.4)
    _pin(g, 5, 12); _pin(g, 27, 12)


def _contour(g: QPainter) -> None:
    """发散色带——和视口里的云图同一套约定。"""
    cols = ("#3b76bd", "#7fa9d6", "#d9d9d9", "#e39b7b", "#c0392b")
    g.setPen(Qt.PenStyle.NoPen)
    for i, c in enumerate(cols):
        g.setBrush(QBrush(QColor(c)))
        g.drawRect(QRectF(5, 8 + i * 3.4, 22, 3.4))
    g.setPen(_pen(_LINE, 1.4))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawRect(QRectF(5, 8, 22, 17))


def _diagram(g: QPainter) -> None:
    """弯矩图：杆件 + 挂在下面的抛物线。

    杆件线要**后画**——先画会被填充盖掉，图标就只剩一个抛物线，
    看不出"图是挂在杆上的"。
    """
    path = QPainterPath(QPointF(5, 11))
    path.quadTo(QPointF(16, 37), QPointF(27, 11))
    g.setPen(_pen(_HOT, 1.6))
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 70)))
    g.drawPath(path)
    g.setPen(_pen(_LINE, 2.4))
    g.drawLine(QPointF(4, 11), QPointF(28, 11))
    g.setBrush(QBrush(_LINE)); g.setPen(Qt.PenStyle.NoPen)
    g.drawEllipse(QPointF(5, 11), 1.8, 1.8)
    g.drawEllipse(QPointF(27, 11), 1.8, 1.8)


def _envelope(g: QPainter) -> None:
    """包络：两条极值曲线夹出来的带。"""
    top = [(5, 14), (12, 8), (22, 11), (27, 7)]
    bot = [(5, 22), (12, 26), (22, 21), (27, 25)]
    band = QPainterPath(QPointF(*top[0]))
    for i in range(1, len(top)):
        band.lineTo(QPointF(*top[i]))
    for p in reversed(bot):
        band.lineTo(QPointF(*p))
    band.closeSubpath()
    g.setPen(Qt.PenStyle.NoPen)
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 60)))
    g.drawPath(band)
    _curve(g, top, _HOT, 1.8)
    _curve(g, bot, _DIM, 1.8)


def _undo(g: QPainter) -> None:
    """回转箭头。**画顺时针还是逆时针要和"退回"的方向感一致**——
    左弯是退，右弯是进，反了会让人按错。"""
    path = QPainterPath(QPointF(24, 23))
    path.cubicTo(QPointF(24, 10), QPointF(12, 8), QPointF(9, 13))
    g.setPen(_pen(_LINE, 2.4))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawPath(path)
    g.setBrush(QBrush(_HOT)); g.setPen(Qt.PenStyle.NoPen)
    g.drawPolygon(QPolygonF([QPointF(6, 10), QPointF(14, 11), QPointF(8, 18)]))


def _redo(g: QPainter) -> None:
    path = QPainterPath(QPointF(8, 23))
    path.cubicTo(QPointF(8, 10), QPointF(20, 8), QPointF(23, 13))
    g.setPen(_pen(_LINE, 2.4))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawPath(path)
    g.setBrush(QBrush(_HOT)); g.setPen(Qt.PenStyle.NoPen)
    g.drawPolygon(QPolygonF([QPointF(26, 10), QPointF(18, 11), QPointF(24, 18)]))


def _timeline(g: QPainter) -> None:
    """建模过程：一条时间轴，当前那一步是实心的。"""
    g.setPen(_pen(_DIM, 1.8))
    g.drawLine(QPointF(5, 16), QPointF(27, 16))
    for i, x in enumerate((8, 14, 20, 26)):
        cur = i == 2
        g.setBrush(QBrush(_HOT if cur else QColor(0, 0, 0, 0)))
        g.setPen(_pen(_HOT if cur else _DIM, 1.8))
        g.drawEllipse(QPointF(x, 16), 3.0 if cur else 2.4, 3.0 if cur else 2.4)


def _pick_node(g: QPainter) -> None:
    """选择节点：三个点，中间那个被选中。"""
    g.setPen(_pen(_DIM, 1.4))
    g.drawLine(QPointF(6, 22), QPointF(26, 10))
    g.setBrush(QBrush(_DIM)); g.setPen(Qt.PenStyle.NoPen)
    g.drawEllipse(QPointF(6, 22), 2.2, 2.2)
    g.drawEllipse(QPointF(26, 10), 2.2, 2.2)
    g.setBrush(QBrush(_HOT)); g.setPen(_pen(_LINE, 1.6))
    g.drawEllipse(QPointF(16, 16), 4.2, 4.2)


def _pick_member(g: QPainter) -> None:
    """选择杆件：三根杆，中间那根被选中。"""
    g.setPen(_pen(_DIM, 2.0))
    g.drawLine(QPointF(6, 26), QPointF(6, 8))
    g.drawLine(QPointF(26, 26), QPointF(26, 8))
    g.setPen(_pen(_HOT, 3.4))
    g.drawLine(QPointF(6, 9), QPointF(26, 9))


def _labels(g: QPainter) -> None:
    """编号标注开关。

    第一版在里面写 "1 2 3"，字在 32 的画布里就被裁了，20 像素下更是没有。
    换成"标签牌 + 引线"这个通用画法——不依赖能不能读出字。
    """
    g.setPen(_pen(_DIM, 1.8))
    g.drawLine(QPointF(4, 24), QPointF(28, 24))
    g.setBrush(QBrush(_DIM)); g.setPen(Qt.PenStyle.NoPen)
    g.drawEllipse(QPointF(4, 24), 2.2, 2.2)
    g.drawEllipse(QPointF(28, 24), 2.2, 2.2)
    g.setPen(_pen(_HOT, 1.6))
    g.drawLine(QPointF(16, 24), QPointF(16, 17))          # 引线
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 130)))
    g.drawRoundedRect(QRectF(7, 6, 18, 11), 2, 2)         # 标签牌
    g.setPen(_pen(QColor(theme.PANEL), 1.8))
    for y in (10, 13.5):
        g.drawLine(QPointF(10, y), QPointF(22, y))


def _table(g: QPainter) -> None:
    """表格编辑。"""
    g.setPen(_pen(_LINE, 1.6))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawRect(QRectF(5, 7, 22, 18))
    g.setPen(_pen(_DIM, 1.2))
    for y in (13, 19):
        g.drawLine(QPointF(5, y), QPointF(27, y))
    g.drawLine(QPointF(13, 7), QPointF(13, 25))
    g.drawLine(QPointF(20, 7), QPointF(20, 25))
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 110)))
    g.setPen(Qt.PenStyle.NoPen)
    g.drawRect(QRectF(13.5, 13.5, 6.2, 5.2))


def _report(g: QPainter) -> None:
    g.setPen(_pen(_LINE, 1.8))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawRoundedRect(QRectF(8, 5, 16, 22), 2, 2)
    g.setPen(_pen(_DIM, 1.6))
    for y in (11, 15, 19):
        g.drawLine(QPointF(11, y), QPointF(21, y))
    g.setPen(_pen(_HOT, 1.6))
    g.drawLine(QPointF(11, 23), QPointF(17, 23))


def _new(g: QPainter) -> None:
    g.setPen(_pen(_LINE, 1.8))
    g.drawRoundedRect(QRectF(8, 5, 16, 22), 2, 2)
    g.setPen(_pen(_HOT, 2.4))
    g.drawLine(QPointF(16, 12), QPointF(16, 21))
    g.drawLine(QPointF(11.5, 16.5), QPointF(20.5, 16.5))


def _open(g: QPainter) -> None:
    g.setPen(_pen(_LINE, 1.8))
    g.drawPolyline(QPolygonF([QPointF(5, 25), QPointF(5, 9), QPointF(13, 9),
                              QPointF(15.5, 12.5), QPointF(24, 12.5)]))
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 90)))
    g.setPen(_pen(_HOT, 1.6))
    g.drawPolygon(QPolygonF([QPointF(5, 25), QPointF(9, 16), QPointF(28, 16),
                             QPointF(24, 25)]))


def _save(g: QPainter) -> None:
    g.setPen(_pen(_LINE, 1.8))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawRoundedRect(QRectF(6, 6, 20, 20), 2, 2)
    g.setBrush(QBrush(_DIM)); g.setPen(Qt.PenStyle.NoPen)
    g.drawRect(QRectF(11, 6, 10, 7))
    g.setBrush(QBrush(_HOT))
    g.drawRect(QRectF(10, 17, 12, 9))


def _units(g: QPainter) -> None:
    g.setPen(_pen(_LINE, 1.8))
    g.drawLine(QPointF(5, 20), QPointF(27, 20))
    for x in (5, 12.3, 19.6, 27):
        g.drawLine(QPointF(x, 16), QPointF(x, 20))
    g.setPen(_pen(_HOT, 2.0))
    g.drawLine(QPointF(5, 24), QPointF(16, 24))
    g.drawLine(QPointF(5, 22), QPointF(5, 26))
    g.drawLine(QPointF(16, 22), QPointF(16, 26))


def _fit(g: QPainter) -> None:
    g.setPen(_pen(_DIM, 1.8))
    for x0, y0, x1, y1, x2, y2 in ((5, 11, 5, 5, 11, 5), (27, 11, 27, 5, 21, 5),
                                   (5, 21, 5, 27, 11, 27), (27, 21, 27, 27, 21, 27)):
        g.drawPolyline(QPolygonF([QPointF(x0, y0), QPointF(x1, y1), QPointF(x2, y2)]))
    g.setPen(_pen(_HOT, 2.0))
    g.drawRect(QRectF(11, 11, 10, 10))


def _axes(face: str):
    """标准视角：画一个立方体，**把你正对着的那个面涂亮**。

    第一版是在立方体里写 "ISO"/"XZ"，结果缩到工具栏用的 20 像素就糊成一团——
    这违反了这个模块开头立的规矩。改成涂面之后，在任何尺寸下
    "亮的那一面朝着你"这个意思都读得出来，这也是 CAE 里的通行画法。

    face: front 正面 / side 右侧面 / top 顶面 / iso 等轴测（三面都露一点）
    """
    # 立方体三个可见面的多边形（前、右、顶）
    FRONT = [(7, 12), (20, 12), (20, 25), (7, 25)]
    RIGHT = [(20, 12), (26, 7), (26, 20), (20, 25)]
    TOP = [(7, 12), (13, 7), (26, 7), (20, 12)]
    lit = {"front": [FRONT], "side": [RIGHT], "top": [TOP],
           "iso": [FRONT, RIGHT, TOP]}.get(face, [])

    def draw(g: QPainter) -> None:
        for poly in (TOP, RIGHT, FRONT):
            on = poly in lit
            g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(),
                                     150 if on else 0)))
            g.setPen(_pen(_HOT if on else _DIM, 1.6))
            g.drawPolygon(QPolygonF([QPointF(*p) for p in poly]))
    return draw


def _camera(g: QPainter) -> None:
    g.setPen(_pen(_LINE, 1.8))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawRoundedRect(QRectF(4, 10, 24, 16), 2, 2)
    g.drawLine(QPointF(11, 10), QPointF(13, 6))
    g.drawLine(QPointF(21, 10), QPointF(19, 6))
    g.drawLine(QPointF(13, 6), QPointF(19, 6))
    g.setPen(_pen(_HOT, 2.0))
    g.drawEllipse(QPointF(16, 18), 5, 5)


def _bg(g: QPainter) -> None:
    """背景主题：一个画框，左上亮色、右下深色，表示可切换底色。"""
    g.setPen(_pen(_DIM, 1.6))
    g.setBrush(QBrush(QColor(220, 224, 230)))
    g.drawRect(QRectF(5, 6, 22, 20))
    g.setPen(Qt.PenStyle.NoPen)
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 170)))
    g.drawPolygon(QPolygonF([QPointF(5, 26), QPointF(27, 14),
                             QPointF(27, 26)]))
    g.setPen(_pen(_LINE, 1.6))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawRect(QRectF(5, 6, 22, 20))


def _grid_floor(g: QPainter) -> None:
    """网格地面：透视网格，地平线上有消失点。"""
    g.setPen(_pen(_DIM, 1.5))
    # 外框（梯形，近大远小）
    g.drawPolygon(QPolygonF([QPointF(5, 12), QPointF(27, 12),
                             QPointF(24, 26), QPointF(8, 26)]))
    # 横向网格线
    for t in (0.33, 0.66):
        y_top, y_bot = 12.0, 26.0
        half_top, half_bot = 11.0, 8.0
        y = y_top + (y_bot - y_top) * t
        half = half_top + (half_bot - half_top) * t
        g.drawLine(QPointF(16 - half, y), QPointF(16 + half, y))
    # 纵向透视线，汇到中点
    for fx in (10, 13, 19, 22):
        g.drawLine(QPointF(fx, 12), QPointF(16 + (fx - 16) * 0.72, 26))
    g.setPen(_pen(_HOT, 1.8))
    g.drawLine(QPointF(5, 12), QPointF(27, 12))


def _chat(g: QPainter) -> None:
    g.setPen(_pen(_HOT, 1.8))
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 55)))
    path = QPainterPath()
    path.addRoundedRect(QRectF(4, 6, 24, 16), 3, 3)
    path.moveTo(QPointF(10, 22)); path.lineTo(QPointF(10, 28))
    path.lineTo(QPointF(16, 22)); path.closeSubpath()
    g.drawPath(path)
    g.setPen(_pen(_LINE, 1.6))
    for y in (11.5, 16.5):
        g.drawLine(QPointF(9, y), QPointF(23, y))


def _strength(g: QPainter) -> None:
    """强度验算：截面上的直线应力分布 —— N/A ± M·c/I 的那张图。

    画的是一个矩形截面加一侧受拉、一侧受压的三角形应力块，中间一条中和轴。
    一个学过材料力学的人扫一眼就认得出这是"验应力"，
    比拿一个对勾或盾牌来凑数强。
    """
    g.setPen(_pen(_LINE, 1.8))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawRect(QRectF(5, 7, 7, 18))                  # 截面
    g.setPen(QPen(_DIM, 1.0, Qt.PenStyle.DashLine))
    g.drawLine(QPointF(4, 16), QPointF(29, 16))      # 中和轴
    # 上压下拉：两个方向相反的三角形应力块
    top = QPolygonF([QPointF(12, 7), QPointF(27, 7), QPointF(12, 16)])
    bot = QPolygonF([QPointF(12, 16), QPointF(24, 25), QPointF(12, 25)])
    g.setPen(Qt.PenStyle.NoPen)
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 90)))
    g.drawPolygon(top)
    g.setBrush(QBrush(QColor(_DIM.red(), _DIM.green(), _DIM.blue(), 90)))
    g.drawPolygon(bot)
    g.setPen(_pen(_HOT, 1.6))
    g.drawLine(QPointF(27, 7), QPointF(12, 16))
    g.setPen(_pen(_DIM, 1.6))
    g.drawLine(QPointF(12, 16), QPointF(24, 25))


def _symmetry(g: QPainter) -> None:
    """对称性：中间一条对称轴，两侧互为镜像的半跨。"""
    g.setPen(QPen(_HOT, 1.4, Qt.PenStyle.DashDotLine))
    g.drawLine(QPointF(16, 3), QPointF(16, 29))
    g.setPen(_pen(_LINE, 2.0))
    for sign in (-1, 1):
        x0, x1 = 16 + sign * 3, 16 + sign * 11
        g.drawLine(QPointF(x0, 9), QPointF(x1, 9))      # 半跨梁
        g.drawLine(QPointF(x1, 9), QPointF(x1, 24))     # 柱
    _pin(g, 5, 24); _pin(g, 27, 24)


def _bandwidth(g: QPainter) -> None:
    """一维变带宽存储：方阵里靠着对角线的一条不等宽的带。

    带宽不等——那正是"变带宽"相对"等带宽"的全部区别，
    画成等宽的就把这个图标最该说的事画丢了。
    """
    g.setPen(Qt.PenStyle.NoPen)
    g.setBrush(QBrush(QColor(_HOT.red(), _HOT.green(), _HOT.blue(), 80)))
    heights = (2, 3, 3, 5, 4, 6, 6)                  # 各列列高，故意不等
    for i, h in enumerate(heights):
        x = 6.0 + i * 3.0
        g.drawRect(QRectF(x, 6.0 + i * 3.0 - h * 3.0 + 3.0, 2.4, h * 3.0))
    g.setPen(_pen(_LINE, 1.6))
    g.setBrush(Qt.BrushStyle.NoBrush)
    g.drawRect(QRectF(5, 5, 22, 22))
    g.setPen(QPen(_DIM, 1.0, Qt.PenStyle.DotLine))
    g.drawLine(QPointF(5, 5), QPointF(27, 27))       # 对角线


DRAWERS: dict[str, Callable[[QPainter], None]] = {
    "strength": _strength, "symmetry": _symmetry, "bandwidth": _bandwidth,
    "new": _new, "open": _open, "save": _save, "units": _units,
    "frame": _frame, "portal": _portal, "node": _node, "member": _member,
    "remove": _remove, "section": _section, "support": _support,
    "load": _load, "combo": _combo, "gravity": _gravity,
    "geometry": _geometry, "solve": _solve, "modal": _modal, "buckling": _buckling,
    "sweep": _sweep, "compare": _compare,
    "deformed": _deformed, "contour": _contour, "diagram": _diagram,
    "envelope": _envelope, "report": _report,
    "undo": _undo, "redo": _redo, "timeline": _timeline,
    "pick_node": _pick_node, "pick_member": _pick_member,
    "labels": _labels, "table": _table,
    "fit": _fit, "camera": _camera, "chat": _chat,
    "bg": _bg, "grid_floor": _grid_floor,
    "iso": _axes("iso"), "front": _axes("front"), "side": _axes("side"),
    "top": _axes("top"),
}


def icon(name: str, size: int = 32) -> QIcon:
    """按名字取图标。名字不认识就返回空图标——**不要因为少一个图标就崩**，
    工具栏少个图案是小事，开不了窗是大事。"""
    draw = DRAWERS.get(name)
    result = QIcon()
    if draw is None:
        return result
    for px in (size, size * 2):            # 备一份 2 倍图，高 DPI 下不糊
        pixmap = QPixmap(px, px)
        pixmap.fill(Qt.GlobalColor.transparent)
        g = QPainter(pixmap)
        g.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        g.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        g.scale(px / BOX, px / BOX)
        try:
            draw(g)
        finally:
            g.end()
        result.addPixmap(pixmap)
    return result


def names() -> list[str]:
    return sorted(DRAWERS)
