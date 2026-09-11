"""桌面端的配色与样式表。

视口配色直接复用 `viz_theme` —— 桌面端和 Streamlit 端、和报告图共用一套色号，
不让三个界面各说各话。这里只补 Qt 特有的东西：窗口 chrome 的样式表。

风格定位：Abaqus/CAE、ANSYS 一类专业工程软件的浅色控制台包围深色视口——
**清晰、克制、高信息密度**。面板靠细边框和表面层级分隔，钢蓝色只用于
当前状态与主操作，让三维模型始终是画面中心。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import viz_theme as V                                    # noqa: E402

# 浅色 CAD chrome：三层表面明确区分工作区、面板和浮层。
#
# 整体比"办公软件白"压深一档，是为了**和视口衔接**。深色视口 (#1b2027)
# 紧挨着纯白面板时，两者之间是一条硬边——眼睛会先看到那条边界，再看模型。
# 把 chrome 压到中浅灰以后，视口读起来像嵌进去的画布，不像贴上去的另一张图。
# Abaqus/ANSYS 的中灰 chrome 是同一个道理。
#
# 压深会挤压文字对比度，所以次级墨色跟着一起加深：
# INK_MUTED 对 PANEL 从 4.5:1 提到 5.1:1，INK_DIM 从 2.6:1 提到 3.6:1
# （INK_DIM 只用于"未开始"这类刻意弱化的文字，不承载正文）。
PANEL = "#e6ebf0"
PANEL_ALT = "#f5f8fa"
PANEL_RAISED = "#d9e1e9"
PANEL_HOVER = "#cbd6e0"
BORDER = "#b9c4d0"
BORDER_LIGHT = "#9dabbb"
INK = "#17212b"
INK_MUTED = "#55626f"
INK_DIM = "#6f7c88"
ACCENT = "#1769aa"
ACCENT_HOVER = "#0f79c5"
ACCENT_DIM = "#0f568e"
SELECTION = "#d8eaf8"
SELECTION_HOVER = "#c9e1f4"
WARN = "#a76512"
SUCCESS = "#237a4b"
ERROR = "#b53a3a"

VIEWPORT_BG = V.VIEWPORT_BG
# 低饱和钢灰蓝用于未求解模型；高饱和蓝只留给变形和结果强调。
MEMBER = "#91a5b9"
HIGHLIGHT = V.VIEWPORT_HIGHLIGHT
REFERENCE = V.VIEWPORT_REFERENCE
# 支座不能用正文墨色：那是全视口最亮的颜色，配上比杆件粗好几倍的符号，
# 会让十几个支座方块成为画面主体，真正要看的结果反而最细。降到次级灰。
# 深色视口专用的墨色。原来这里的文字、节点、坐标轴都用浅色主题的 INK
# (#17212b) 和 INK_MUTED，画在 #1b2027 的视口上几乎隐形——节点标记就是这么
# 消失的。视口和面板是两套底色，前景色必须各用各的。
VIEWPORT_INK = V.VIEWPORT_INK
VIEWPORT_INK_MUTED = V.VIEWPORT_INK_MUTED
SUPPORT = V.VIEWPORT_INK_MUTED
# 荷载分两族上色：力（橙）与 力矩/给定位移（青）。
# 四类各给一色跑不过 validate_palette.js 的 all-pairs 档——文档写明超过三槽
# 就该换编码方式。所以集中力与分布荷载同色，靠**形状**（单支箭头 vs 一排箭头）
# 和**数值标注**区分；这两样比第四种颜色可靠得多。
# 取值来自已验证的暗色分类色序：蓝(杆件) / 橙 / 青，三者 all-pairs 全部通过。
LOAD = "#d95926"
LOAD_MOMENT = "#199e70"
HINGE = "#ffb454"
DRAFT_SUPPORT = "#2f9e68"
# 工程软件通用局部轴色：x 红、y 绿、z 蓝。只用于方向编码，不参与结果色标。
LOCAL_AXES = {"x": "#e05a5a", "y": "#55b86a", "z": "#4f87d8"}

# 锚点色。**不要直接当 cmap 传给 add_mesh**：PyVista 会把颜色列表理解成
# ListedColormap，只有 6 级，云图渲染出来是台阶不是渐变（已实测）。
# 需要色标时调 sequential_cmap() / diverging_cmap()。
SEQUENTIAL_ANCHORS = V.VIEWPORT_SEQUENTIAL
DIVERGING_ANCHORS = V.VIEWPORT_DIVERGING
RAINBOW_ANCHORS = V.VIEWPORT_RAINBOW

# 云图默认分几级。12 是 CAE 后处理的通行值：再少读不出梯度，
# 再多人眼分辨不出相邻两级的色差，等于回到连续渐变。
# 取自 viz_theme，报告里的静态图读的是同一个数。
CONTOUR_LEVELS = V.CONTOUR_LEVELS
CONTOUR_LEVELS_RANGE = (4, 24)


def sequential_cmap(steps: int = 256):
    """由锚点色插值出的连续顺序色标。matplotlib 在这里才导入——
    theme 被整个桌面端引用，不该为了一个色标把它变重。"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "viewport_sequential", SEQUENTIAL_ANCHORS, N=steps)


def diverging_cmap(steps: int = 256):
    """发散色标：受压蓝 — 中性灰 — 受拉红，**取自 viz_theme**。

    这里原来直接用 matplotlib 的 ``"coolwarm"``，而报告里的静态图用的是
    `viz_theme` 的锚点色——**同一个结构在屏幕上和报告里配色不一样**，
    而 report.py 的说明里还写着"两者共用 viz_theme 配色，不会各说各话"。

    顺带，coolwarm 的中点接近白色：在深色视口上，零内力的杆件反而是全图
    最亮的，眼睛先看到的是最不受力的地方。视口这套锚点的中点是中性灰，
    没有这个问题。
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "viewport_diverging", DIVERGING_ANCHORS, N=steps)


def rainbow_cmap(steps: int = 256):
    """Abaqus 式彩虹色标：蓝(低) → 青 → 绿 → 黄 → 橙 → 红(高)。

    两端比蓝红双色标拉得开，中间还多出绿黄两段，同样 12 级下相邻两级的
    色差明显更大——这正是"分不清哪一级是哪一级"的解法。

    代价要说清楚：彩虹谱的明度不单调，**色盲用户和黑白打印都读不出顺序**。
    所以它是可选项不是唯一项，蓝—灰—红那套发散色标仍然保留。
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "viewport_rainbow", RAINBOW_ANCHORS, N=steps)


#: 云图可选色系。键是内部名，值是给界面看的中文名。
CONTOUR_PALETTES = {
    "rainbow": "彩虹（Abaqus 式）",
    "diverging": "蓝—灰—红（发散）",
    "sequential": "单蓝（顺序）",
}
DEFAULT_PALETTE = "rainbow"


def palette_cmap(palette: str, component: str = ""):
    """按色系名取连续色标。

    ``diverging`` 这一项对合量（V/M，定义上非负）没有意义——发散色标的
    中点代表零，而合量的零在量程端点上，用了会把"最小"画成中性灰。
    所以合量在选了发散时自动退回顺序色标。
    """
    if palette == "rainbow":
        return rainbow_cmap()
    if palette == "sequential":
        return sequential_cmap()
    return sequential_cmap() if component in {"V", "M"} else diverging_cmap()


def banded(cmap, levels: int = CONTOUR_LEVELS):
    """把连续色标切成 ``levels`` 个等宽色块。

    云图分级不是装饰：连续渐变上读不出"这一段到底是多少"，只能读出
    "这边比那边红"。切成有限级之后，每一级对应色标上一个可读的区间，
    看图的人能直接把杆件上的一段对到一个数值范围——这是 CAE 后处理
    默认分级的原因。
    """
    import numpy as np
    from matplotlib.colors import ListedColormap

    n = int(max(CONTOUR_LEVELS_RANGE[0],
                min(CONTOUR_LEVELS_RANGE[1], int(levels))))
    return ListedColormap(cmap(np.linspace(0.0, 1.0, n)), name="banded")

# 小圆角只用于区分可交互表面，不做消费产品式大胶囊。
RADIUS_SM = "4px"
RADIUS_MD = "6px"
RADIUS_LG = "8px"


STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {PANEL};
    color: {INK};
    font-size: 9pt;
    font-family: "DengXian", "Microsoft YaHei UI", "Source Han Sans SC", sans-serif;
}}
QMenuBar {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER};
    padding: 2px 8px;
}}
QMenuBar::item {{
    padding: 5px 12px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background: {SELECTION};
    color: {ACCENT_DIM};
}}
QMenu {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER_LIGHT};
    padding: 2px;
}}
QMenu::item {{
    padding: 5px 24px 5px 18px;
}}
QMenu::item:selected {{
    background: {SELECTION};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 3px 6px;
}}

QToolBar {{
    background: {PANEL};
    border: none;
    border-bottom: 1px solid {BORDER};
    spacing: 1px;
    padding: 0;
}}
QToolButton {{
    color: {INK};
    padding: 3px 8px;
    border: 1px solid transparent;
    border-radius: {RADIUS_SM};
}}
QToolButton:hover {{
    background: {PANEL_HOVER};
    border-color: {BORDER};
}}
QToolButton:checked {{
    background: {SELECTION};
    border-color: {ACCENT_DIM};
}}
QToolButton:disabled {{
    color: {INK_DIM};
}}
QToolButton#mainMenuButton {{
    min-width: 72px;
    margin: 2px 6px 2px 4px;
    padding: 4px 9px;
    background: {PANEL_RAISED};
    border-color: {BORDER};
    font-weight: 600;
}}
QToolButton#mainMenuButton:hover {{
    background: {SELECTION};
    border-color: {ACCENT_DIM};
    color: {ACCENT_DIM};
}}

QDockWidget {{
    color: {INK_MUTED};
    font-weight: 600;
}}
QDockWidget::title {{
    background: {PANEL_RAISED};
    padding: 7px 10px;
    border-bottom: 1px solid {BORDER};
    font-size: 8.5pt;
    color: {INK};
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 2px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background: {PANEL_HOVER};
}}

QTreeWidget, QTableView, QListWidget, QPlainTextEdit, QTextEdit {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    alternate-background-color: {PANEL};
    selection-background-color: {SELECTION};
    selection-color: {INK};
    outline: none;
}}
QTreeWidget::item, QTableView::item {{
    padding: 4px 5px;
}}
QTreeWidget::item:hover, QTableView::item:hover {{
    background: {PANEL_HOVER};
}}
QTreeWidget::item:selected, QTableView::item:selected {{
    background: {SELECTION};
}}
QHeaderView::section {{
    background: {PANEL_ALT};
    color: {INK_MUTED};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 4px 6px;
    font-size: 8pt;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    padding: 4px 7px;
    color: {INK};
    selection-background-color: {SELECTION};
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {BORDER_LIGHT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_DIM};
}}
QComboBox QAbstractItemView {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER_LIGHT};
    selection-background-color: {SELECTION};
    padding: 1px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QPushButton {{
    background: {PANEL_RAISED};
    border: 1px solid {BORDER_LIGHT};
    border-radius: {RADIUS_SM};
    padding: 6px 14px;
    color: {INK};
}}
QPushButton:hover {{
    background: {PANEL_HOVER};
    border-color: {ACCENT_DIM};
}}
QPushButton:pressed {{
    background: {SELECTION};
    border-color: {ACCENT_DIM};
}}
QPushButton:default {{
    background: {ACCENT_DIM};
    border-color: {ACCENT};
    color: #ffffff;
}}
QPushButton[role="primary"], QToolButton[role="primary"] {{
    background: {ACCENT_DIM};
    border-color: {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[role="primary"]:hover, QToolButton[role="primary"]:hover {{
    background: {ACCENT};
}}
QPushButton:default:hover {{
    background: {ACCENT};
}}
QPushButton:disabled {{
    color: {INK_DIM};
    border-color: {BORDER};
    background: {PANEL};
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    top: -1px;
}}
QTabBar::tab {{
    background: {PANEL};
    color: {INK_MUTED};
    padding: 5px 12px;
    border: 1px solid transparent;
    border-bottom: none;
    margin-right: 1px;
}}
QTabBar::tab:hover {{
    color: {INK};
    background: {PANEL_HOVER};
}}
QTabBar::tab:selected {{
    background: {PANEL_ALT};
    color: {INK};
    border-color: {BORDER};
}}

QStatusBar {{
    background: {PANEL};
    border-top: 1px solid {BORDER};
    color: {INK_MUTED};
    font-size: 8pt;
    padding: 2px 6px;
}}
QStatusBar::item {{
    border: none;
}}
QStatusBar QLabel {{
    padding: 2px 8px;
}}

/* 视口是深色、chrome 是浅色，两者直接相接会出现一条硬边，眼睛先看到边界
   再看模型。给它一圈 1px 内嵌边框，深色区域读起来就像"嵌进去的画布"。 */
QWidget#viewportHost {{
    background: {VIEWPORT_BG};
    border: 1px solid {BORDER_LIGHT};
}}
QWidget#workspaceBar {{
    background: {PANEL_ALT};
    border-bottom: 1px solid {BORDER};
}}
QWidget#workflowBar {{
    background: {PANEL_RAISED};
    border-bottom: 1px solid {BORDER};
}}
QLabel[workflow="caption"] {{
    color: {INK_MUTED};
    font-size: 8pt;
    font-weight: 600;
    padding-right: 6px;
}}
QLabel[workflow="guidance"] {{
    color: {INK_MUTED};
    padding-left: 10px;
}}
/* 流程条不是五个并排的按钮，是一条**有方向的路**：
   已完成→绿、当前→实心蓝（全条唯一的实心块，一眼定位）、被阻断→红、
   未开始→灰。中间由 workflow_bar 插入 › 连接符，读起来才是流程不是工具栏。 */
QPushButton[workflowStage] {{
    min-width: 66px;
    padding: 4px 10px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS_SM};
    color: {INK_DIM};
    text-align: center;
}}
QPushButton[workflowStage]:hover {{
    background: {PANEL_HOVER};
}}
QPushButton[workflowState="done"] {{
    color: {SUCCESS};
    background: #dfeee6;
    border-color: #a9cfba;
}}
QPushButton[workflowState="done"]:hover {{
    background: #d0e6da;
}}
/* 当前步是全条唯一的实心块。反白文字 + 深蓝底，"现在该干这一步"
   不需要用户去找。 */
QPushButton[workflowState="active"] {{
    color: #ffffff;
    background: {ACCENT};
    border-color: {ACCENT_DIM};
    font-weight: 600;
}}
QPushButton[workflowState="active"]:hover {{
    background: {ACCENT_HOVER};
}}
QPushButton[workflowState="blocked"] {{
    color: {ERROR};
    background: #f7e3e3;
    border-color: #d9a2a2;
    font-weight: 600;
}}
QPushButton[workflowState="blocked"]:hover {{
    background: #f0d5d5;
}}
/* 未开始：灰、无底色，但 hover 仍给反馈——它是可以点的（允许跳读流程），
   只是不推荐现在点。 */
QPushButton[workflowState="pending"] {{
    color: {INK_DIM};
}}
QLabel[workflow="arrow"] {{
    color: {BORDER_LIGHT};
    font-size: 11pt;
    padding: 0px 1px;
}}
QFrame#emptyState {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER_LIGHT};
    border-radius: {RADIUS_LG};
}}
QLabel[emptyState="title"] {{
    background: transparent;
    border: none;
    color: {INK};
    font-size: 15pt;
    font-weight: 600;
}}
QLabel[emptyState="subtitle"] {{
    background: transparent;
    border: none;
    color: {INK_MUTED};
    font-size: 9pt;
}}
QFrame[panel="config"] {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_MD};
}}
QLabel[panel="hint"] {{
    background: transparent;
    color: {INK_MUTED};
}}
QPushButton[role="suggestion"] {{
    background: {PANEL_ALT};
    border-color: {BORDER};
    color: {INK_MUTED};
    padding: 4px 8px;
}}
QPushButton[role="suggestion"]:hover {{
    color: {ACCENT_DIM};
    border-color: {ACCENT};
    background: {SELECTION};
}}
QWidget[ribbonGroup="true"] {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_MD};
}}
QLabel[toolbar="section"] {{
    color: {INK_DIM};
    font-size: 7.5pt;
    font-weight: 600;
    padding: 0 3px;
}}
QLabel[status="active"] {{
    color: {ACCENT_HOVER};
}}

QSplitter::handle {{
    background: {BORDER};
}}
QSplitter::handle:hover {{
    background: {ACCENT_DIM};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

QScrollBar:vertical {{
    background: {PANEL};
    width: 11px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_LIGHT};
    border-radius: 0;
    min-height: 28px;
    margin: 1px;
}}
QScrollBar::handle:vertical:hover {{
    background: {INK_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {PANEL};
    height: 11px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_LIGHT};
    border-radius: 0;
    min-width: 28px;
    margin: 1px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {INK_DIM};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    margin-top: 10px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {INK_MUTED};
    font-size: 8pt;
    background: {PANEL};
}}
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM};
    text-align: center;
    background: {PANEL};
    height: 15px;
    color: {INK};
}}
QProgressBar::chunk {{
    background: {ACCENT_DIM};
}}

QToolTip {{
    background: {PANEL_ALT};
    color: {INK};
    border: 1px solid {BORDER_LIGHT};
    padding: 3px 7px;
    font-size: 8pt;
}}

QCheckBox, QRadioButton {{
    color: {INK};
    spacing: 6px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {BORDER_LIGHT};
    border-radius: {RADIUS_SM};
    background: {PANEL};
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {ACCENT_DIM};
    border-color: {ACCENT};
}}
"""


# 功能区的样式。单独拼在主样式表后面，改配色时这一段可以整块换掉。
RIBBON_QSS = f"""
QTabWidget[ribbon="bar"]::pane {{
    border: 0px;
    border-bottom: 1px solid {BORDER};
    background: {PANEL};
}}
QTabWidget[ribbon="bar"] > QTabBar::tab {{
    background: transparent;
    color: {INK_MUTED};
    padding: 7px 18px;
    margin-right: 1px;
    border: 0px;
    border-bottom: 2px solid transparent;
}}
QTabWidget[ribbon="bar"] > QTabBar::tab:hover {{
    color: {INK};
    background: {PANEL_HOVER};
}}
QTabWidget[ribbon="bar"] > QTabBar::tab:selected {{
    color: {ACCENT_DIM};
    border-bottom: 2px solid {ACCENT};
    background: {PANEL_ALT};
}}

QToolButton[ribbon="large"], QToolButton[ribbon="small"] {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS_SM};
    color: {INK};
    padding: 4px 7px;
}}
QToolButton[ribbon="large"] {{
    padding: 6px 7px 4px 7px;
}}
QToolButton[ribbon="large"]:hover, QToolButton[ribbon="small"]:hover {{
    background: {PANEL_HOVER};
    border: 1px solid {BORDER};
}}
QToolButton[ribbon="large"]:pressed, QToolButton[ribbon="small"]:pressed,
QToolButton[ribbon="large"]:checked, QToolButton[ribbon="small"]:checked {{
    background: {SELECTION};
    border: 1px solid {ACCENT_DIM};
}}
QToolButton[ribbon="large"]:disabled, QToolButton[ribbon="small"]:disabled {{
    color: {INK_DIM};
}}

QLabel[ribbon="caption"] {{
    color: {INK_DIM};
    font-size: 7pt;
    font-weight: 600;
    padding-top: 3px;
}}
QFrame[ribbon="sep"] {{
    color: {BORDER};
    margin: 4px 4px 16px 4px;
}}
QFrame[toolbar="sep"] {{
    color: {BORDER};
    margin: 2px 7px;
}}
QToolButton[ribbon="large"][role="primary"],
QToolButton[ribbon="small"][role="primary"] {{
    background: {ACCENT_DIM};
    border: 1px solid {ACCENT};
    color: #ffffff;
}}
QToolButton[ribbon="large"][role="primary"]:hover,
QToolButton[ribbon="small"][role="primary"]:hover {{
    background: {ACCENT};
}}
"""

STYLESHEET = STYLESHEET + RIBBON_QSS
