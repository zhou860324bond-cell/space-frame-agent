"""统一的可视化配色。

取自 dataviz 参考调色板（已跑过校验脚本），报告图与界面共用一套，
不让静态图和界面各说各话。

* 变形图只有一个"系列"（变形后），原始位置是参照物，用哑光灰细虚线；
* 轴力图是**发散**编码：受压 ← 中性灰 → 受拉，两个对立色相加灰中点，
  不用彩虹，中点不上色相。
"""

from __future__ import annotations

# 面与墨
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# 标记
ACCENT = "#2a78d6"          # 变形后结构（分类槽 1）
HIGHLIGHT = "#eb6834"       # 峰值标注（分类槽 2）
REFERENCE = "#898781"       # 原始位置

# 发散：受压 ← 中性 → 受拉
DIVERGING_LOW = "#2a78d6"   # 受压（负）
DIVERGING_MID = "#e4e2dc"   # 比参考中点略深：在 #fcfcfb 画布上，原中点会褪成背景
DIVERGING_HIGH = "#e34948"  # 受拉（正）
DIVERGING_SCALE = [(0.0, DIVERGING_LOW), (0.5, DIVERGING_MID), (1.0, DIVERGING_HIGH)]

# 状态（固定，不参与主题化；始终配图标或文字，不靠颜色单独表意）
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"


# --------------------------------------------------------------- 三维视口
#
# Abaqus/CAE 与 PyVista（FEM-Python 的界面）都用**深色视口**。这不只是好看：
# 深底上浅色线条的对比度更高，细杆件和内力图看得更清楚，而且视口与周围的
# 面板一眼就分得开——浅底视口容易和页面糊成一片。
#
# 报告图仍然用浅底（`SURFACE`）：印在纸上或贴进 Word 里，深底既费墨又难读。
# 所以视口和报告图是两套底色、同一套前景色。

VIEWPORT_BG = "#1b2027"          # 视口底色，冷灰偏蓝，比纯黑柔和
VIEWPORT_GRID = "#2f3742"        # 网格线：看得见但不抢眼
VIEWPORT_AXIS = "#48525f"        # 坐标轴
VIEWPORT_INK = "#e8eaed"         # 视口上的主文字
VIEWPORT_INK_MUTED = "#9aa3ad"   # 刻度、次要文字
VIEWPORT_REFERENCE = "#7b8794"   # 未变形轮廓

# 深底上要提亮一档，否则原来那套为浅底调的颜色会发闷
VIEWPORT_ACCENT = "#5aa9ff"      # 变形后结构
VIEWPORT_HIGHLIGHT = "#ff9152"   # 峰值标注
VIEWPORT_LOW = "#5aa9ff"         # 受压
VIEWPORT_MID = "#8d949e"         # 中性
VIEWPORT_HIGH = "#ff6b6a"        # 受拉
VIEWPORT_DIVERGING = [(0.0, VIEWPORT_LOW), (0.5, VIEWPORT_MID), (1.0, VIEWPORT_HIGH)]
