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
PANEL = "#f3f5f7"
PANEL_ALT = "#ffffff"
PANEL_RAISED = "#e8edf2"
PANEL_HOVER = "#dfe6ed"
BORDER = "#cbd3dc"
BORDER_LIGHT = "#aeb9c6"
INK = "#17212b"
INK_MUTED = "#5e6b78"
INK_DIM = "#84909c"
ACCENT = "#1769aa"
ACCENT_HOVER = "#0f79c5"
ACCENT_DIM = "#0f568e"
SELECTION = "#d8eaf8"
SELECTION_HOVER = "#c9e1f4"
WARN = "#a76512"
SUCCESS = "#237a4b"
ERROR = "#b53a3a"

VIEWPORT_BG = V.VIEWPORT_BG
MEMBER = V.VIEWPORT_ACCENT
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
LOAD = V.VIEWPORT_HIGH
HINGE = "#ffb454"
DRAFT_SUPPORT = "#2f9e68"

DIVERGING = "coolwarm"
# 锚点色。**不要直接当 cmap 传给 add_mesh**：PyVista 会把颜色列表理解成
# ListedColormap，只有 6 级，云图渲染出来是台阶不是渐变（已实测）。
# 需要色标时调 sequential_cmap()。
SEQUENTIAL_ANCHORS = V.VIEWPORT_SEQUENTIAL


def sequential_cmap(steps: int = 256):
    """由锚点色插值出的连续顺序色标。matplotlib 在这里才导入——
    theme 被整个桌面端引用，不该为了一个色标把它变重。"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "viewport_sequential", SEQUENTIAL_ANCHORS, N=steps)

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
QPushButton[workflowStage] {{
    min-width: 68px;
    padding: 4px 8px;
    background: transparent;
    border: 1px solid transparent;
    color: {INK_MUTED};
}}
QPushButton[workflowState="done"] {{
    color: {SUCCESS};
}}
QPushButton[workflowState="active"] {{
    color: {ACCENT_DIM};
    background: {SELECTION};
    border-color: {ACCENT};
    font-weight: 600;
}}
QPushButton[workflowState="blocked"] {{
    color: {ERROR};
    background: #fae9e9;
    border-color: #dfaaaa;
}}
QPushButton[workflowState="pending"] {{
    color: {INK_DIM};
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
