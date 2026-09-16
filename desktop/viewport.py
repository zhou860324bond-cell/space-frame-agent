"""三维视口部件。

**薄。** 所有几何和标量都在 `scene.py` 里算好，这里只负责把网格塞进渲染器、
管好演员的增删、以及相机。逻辑越少越好——因为这一层是最难自动测的。

一个视口负责四种显示模式，切换时整个场景重建（刚架规模小，重建比
增量维护简单可靠得多）：

* 模型     —— 几何 + 支座 + 荷载符号，求解前就能看
* 分析网格 —— 物理构件编译后的单元与切分节点，只读预览
* 变形     —— 变形后的管 + 未变形轮廓
* 云图     —— 按某个内力分量着色
* 模态/屈曲 —— 振型或失稳模态
"""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from . import scene, theme

# 无头环境（QT_QPA_PLATFORM=offscreen）里没有可用的 OpenGL：VTK 建不出
# shader，一渲染就在 C++ 层 abort——**Python 抓不住这种崩溃**。
#
# 这不是在掩盖失败。渲染本来就不是这一层的逻辑：几何和标量在 `scene.py`
# 里算好并逐条验过，视口只负责把网格塞进渲染器。所以无头时**照常维护状态**
# （选中了谁、要不要标注），只跳过真正画的那一步——
# 逻辑照测，像素交给有 GL 的机器。
CAN_RENDER = os.environ.get("QT_QPA_PLATFORM", "") != "offscreen"


class _NullCamera:
    def zoom(self, _factor) -> None:
        pass


class _NullPlotter:
    """离屏测试占位器。

    仅仅“不调用 render”还不够：QtInteractor 自带定时器，会在
    QApplication.processEvents() 时自行 render，仍可能在 VTK C++ 层崩溃。
    无头环境因此根本不创建 VTK 窗口，状态与业务逻辑照常测试。
    """

    def __init__(self, parent=None):
        self.interactor = QWidget(parent)
        self.camera = _NullCamera()
        self.camera_position = None

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None

    def screenshot(self, _path=None, return_img=False):
        if return_img:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return None


# 视口图例的文字。**必须是 ASCII**：VTK 用自己的字体引擎，默认字体没有中文
# 字形——实测中文图例渲染成三个小点，色标那边也栽过同一个坑。要显示中文得给
# VTK 单独加载字体文件，那是另一件事；在此之前，英文术语比一排方块可读。
_LEGEND_TEXT = {
    "杆件": "Member",
    "杆端铰": "Hinge",
    "固接": "Support: fixed",
    "铰接": "Support: pinned",
    "部分约束": "Support: partial",
    "节点荷载": "Nodal force",
    "节点力矩": "Nodal moment",
    "杆间荷载": "Span load",
    "给定位移": "Prescribed disp.",
    "给定转角": "Prescribed rot.",
}


class Viewport(QWidget):
    """PyVista 渲染器的宿主。

    除了画，还负责两件和"在哪"有关的事：**拾取**和**编号标注**。
    这两样加起来才让"节点 17 缺少约束"这类信息在三维视图里变得可定位。
    """

    # 拾取到了什么：("node" | "member", 编号)
    picked = Signal(str, int)
    probed = Signal(int, object)     # 杆件编号 + 世界坐标，定位具体截面
    # 人工建模：创建了节点 / 创建了杆件
    node_created = Signal(float, float, float)   # x, y, z
    member_created = Signal(int, int)             # node_i, node_j
    # 建节点时点击位置落在已有节点上，吸附过去、不重复建，发它提示编号
    node_snapped = Signal(int)
    # 在视口里按了 Esc，请求退出当前建模/拾取模式
    escape_pressed = Signal()
    # 右键菜单由主窗口组装，这里只报告屏幕坐标。
    context_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 视口深、chrome 浅，直接相接会露出一条硬边。给它一圈 1px 内嵌边框，
        # 深色区域读起来是"嵌进去的画布"而不是"贴上去的另一张图"。
        # 纯 QWidget 子类默认不吃样式表的背景/边框，必须开 WA_StyledBackground；
        # 同时留 1px 内边距，否则边框会被 VTK 的原生子窗口整个盖住。
        self.setObjectName("viewportHost")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        if CAN_RENDER:
            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self)
        else:
            self.plotter = _NullPlotter(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.interactor.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.plotter.interactor.customContextMenuRequested.connect(
            lambda pos: self.context_requested.emit(
                self.plotter.interactor.mapToGlobal(pos)))

        if CAN_RENDER:
            self.plotter.set_background(theme.VIEWPORT_BG)
            self.plotter.add_axes(color=theme.VIEWPORT_INK_MUTED)
            self.plotter.enable_anti_aliasing("fxaa")
        self._first_render = True

        # 云图显示选项。三样都做成状态而不是每次调用的参数：用户在功能区
        # 上改一次，后面每张云图都跟着走，不用每次重新选。
        self.contour_palette = theme.DEFAULT_PALETTE
        self.contour_shading = True
        self._contour_last: dict = {}
        # 荷载数值标注。模型一大，几十个数字糊成一片，比不标还难看——
        # 所以给一个开关，而不是把"标不标"写死。
        self.load_labels = True

        # 左下角坐标系指示器
        from .axis_indicator import AxisIndicator
        self.axis_indicator = AxisIndicator(self)
        self.axis_indicator.move(10, self.height() - self.axis_indicator.height() - 10)
        self.axis_indicator.show()
        self.axis_indicator.raise_()

        # 拾取状态。**过滤器是必须的**：三维视图里点一下，
        # 你不说清楚要选什么类型，用户就得靠反复试来猜自己选中了谁
        self.pick_mode: str | None = None      # None / "node" / "member"
        self.selection: tuple[str, int] | None = None
        self.problem_refs: list[tuple[str, int]] = []
        self.show_labels = False
        self._frame = None                     # 最近一次画的 frame，拾取要用

        # 人工建模状态
        self.model_mode: str | None = None     # None / "node" / "member"
        self._pending_node: int | None = None   # 杆件创建时的第一个节点

        # 精确建模：工作平面 + 网格捕捉 + 已有节点吸附。
        # 直接在三维视图里点，深度由相机决定、节点会"飘"；把点先投影到选定
        # 工作平面、再按间距取整、靠近已有节点就吸附，坐标才可控、可复现。
        self.work_plane: str = "XY"   # "XY"(z=常数) / "XZ"(y=常数) / "YZ"(x=常数)
        self.work_offset: float = 0.0
        self.snap_size: float = 0.0   # 工作平面内网格捕捉间距，0=关闭
        self._node_snap_ratio = 0.06  # 已有节点吸附半径=模型尺寸×该比例

        # 视口外观状态
        self.bg_theme: str = "deep"            # 背景主题；"__custom__" 表示自定义
        self.custom_bg = None                  # 自定义背景：str=纯色，(mode,base,second)=渐变
        self.show_grid_floor: bool = True      # 网格地面
        self.show_axes_widget: bool = True     # 坐标轴指示器

        # 左上角模式徽章：进入建模/拾取模式时提示当前模式、工作平面、捕捉
        self.mode_badge = QLabel("", self)
        self.mode_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_badge.setVisible(False)
        self.mode_badge.setStyleSheet(
            f"QLabel{{background:{theme.ACCENT_DIM}; color:#ffffff; "
            f"border:1px solid {theme.ACCENT}; border-radius:3px; "
            f"padding:4px 10px; font-size:9pt; font-weight:600;}}")

    # --- 视口外观 ---

    # 背景主题。值有两种形态：
    #   "#rrggbb"                              —— 纯色
    #   (mode, base, second)                   —— 渐变
    # 其中 mode 取 "vertical" / "horizontal" / "radial" / "corner"，
    # base/second 的取色位置对齐 pyvista 的调用约定：
    #   vertical:   base=底端色,   second=顶端色
    #   horizontal: base=左端色,   second=右端色
    #   radial:     base=中心色,   second=边缘色
    #   corner:     base=中心色,   second=对角色
    BG_THEMES = {
        # —— 纯色 · 深色 ——
        "dark":     "#1b2027",                          # 经典深色
        "ink":      "#0f1115",                          # 墨黑
        "charcoal": "#2b2f36",                          # 炭灰
        "navy":     "#16202e",                          # 藏青
        # —— 渐变 · 深色（Abaqus 式柔和过渡）——
        "deep":        ("vertical",   "#14181e", "#2a3644"),  # 深蓝·垂直
        "midnight":    ("vertical",   "#0a0e14", "#1f2a3a"),  # 午夜·垂直
        "teal":        ("vertical",   "#141d24", "#2d4a5a"),  # 青蓝·垂直
        "plum":        ("vertical",   "#15111a", "#332a3f"),  # 暮紫·垂直
        "horizon":     ("horizontal", "#12161c", "#29405c"),  # 钢蓝·水平
        "spotlight":   ("radial",     "#33425a", "#0d1117"),  # 聚光·径向
        "corner_glow": ("corner",     "#2b3a4e", "#0b0e13"),  # 对角晕染
        # —— 渐变 · 浅色 ——
        "abaqus":   ("vertical",   "#e3eaf2", "#aec4dc"),     # Abaqus 经典蓝灰
        # —— 纯色 · 浅色 ——
        "light":    "#e8ebef",                          # 浅灰
        "white":    "#ffffff",                          # 纯白（出图用）
        "paper":    "#f5f3ee",                          # 米白
        "sky":      "#dfe7ef",                          # 浅蓝
    }

    # 渐变方向 -> pyvista set_background 的关键字
    _GRAD_MODE_KW = {
        "vertical": "top",
        "horizontal": "right",
        "radial": "side",
        "corner": "corner",
    }
    GRAD_MODES = (("vertical", "垂直（上→下）"),
                  ("horizontal", "水平（左→右）"),
                  ("radial", "径向（中心→四周）"),
                  ("corner", "对角（中心→角）"))

    # 明确的浅色预设（自定义背景另按亮度判断）
    _LIGHT_THEMES = ("light", "white", "paper", "sky", "abaqus")

    def set_background_theme(self, name: str) -> None:
        """切换预设视口背景主题。"""
        if name not in self.BG_THEMES:
            return
        self.bg_theme = name
        self.custom_bg = None
        self._apply_current_background()

    def set_custom_solid(self, hexc: str) -> None:
        """自定义纯色背景。"""
        self.custom_bg = hexc
        self.bg_theme = "__custom__"
        self._apply_current_background()

    def set_custom_gradient(self, c1: str, c2: str,
                            mode: str = "vertical") -> None:
        """自定义渐变背景。

        c1/c2 按用户直觉给色：vertical 时 c1=顶、c2=底；horizontal 时
        c1=左、c2=右；radial/corner 时 c1=中心、c2=边缘/对角。
        """
        if mode not in self._GRAD_MODE_KW:
            mode = "vertical"
        if mode == "vertical":
            value = (mode, c2, c1)      # 内部 base=底, second=顶
        else:
            value = (mode, c1, c2)
        self.custom_bg = value
        self.bg_theme = "__custom__"
        self._apply_current_background()

    # 向后兼容旧调用名
    def set_custom_background(self, top: str, bottom: str | None = None) -> None:
        if bottom:
            self.set_custom_gradient(top, bottom, "vertical")
        else:
            self.set_custom_solid(top)

    def apply_background_value(self, value) -> None:
        """直接应用一个背景值（纯色 str 或 (mode, base, second) 渐变），
        并标记为自定义——弹窗里"保留配色、只换渐变方向"时复用。"""
        self.custom_bg = value
        self.bg_theme = "__custom__"
        self._apply_current_background()

    def _apply_current_background(self) -> None:
        """按当前状态把背景、坐标轴颜色、网格地面一次性刷到渲染器。"""
        if not CAN_RENDER:
            return
        if self.bg_theme == "__custom__" and self.custom_bg is not None:
            value = self.custom_bg
        else:
            value = self.BG_THEMES.get(self.bg_theme, theme.VIEWPORT_BG)
        self._apply_bg_value(value)
        # 背景变了，坐标轴和网格地面颜色要跟着变，浅色底上画白轴看不见
        self._refresh_axes_color()
        self._refresh_grid_floor()
        self.plotter.render()

    def _apply_bg_value(self, value) -> None:
        """把 纯色字符串 或 (mode, base, second) 渐变应用到渲染器。"""
        if isinstance(value, tuple) and len(value) == 3:
            mode, base, second = value
            kw = self._GRAD_MODE_KW.get(mode, "top")
            # set_background 的纯色分支会自动关闭上一次的渐变
            self.plotter.set_background(base, **{kw: second})
        else:
            self.plotter.set_background(value)

    def _is_light_bg(self) -> bool:
        if self.bg_theme == "__custom__":
            return self._value_is_light(self.custom_bg)
        return self._value_is_light(self.BG_THEMES.get(self.bg_theme))

    @classmethod
    def _value_is_light(cls, value) -> bool:
        """判断一个背景值整体是否偏浅：渐变取两端平均亮度。"""
        if value is None:
            return False
        if isinstance(value, tuple):
            colors = [value[1], value[2]]
            lums = [cls._hex_lum(c) for c in colors]
            return sum(lums) / len(lums) > 160
        return cls._hex_is_light(value)

    @staticmethod
    def _hex_lum(hexc: str) -> float:
        try:
            c = QColor(hexc)
            return 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        except Exception:
            return 0.0

    @staticmethod
    def _hex_is_light(hexc: str) -> bool:
        """按相对亮度判断一个十六进制颜色是不是浅色（自定义背景时用）。"""
        return Viewport._hex_lum(hexc) > 160

    def _refresh_axes_color(self) -> None:
        """坐标轴颜色随背景明暗调整。"""
        if not CAN_RENDER:
            return
        try:
            axes_color = "#333333" if self._is_light_bg() else theme.INK_MUTED
            self.plotter.add_axes(color=axes_color)
        except Exception:
            pass

    def toggle_grid_floor(self, on: bool) -> None:
        """显示/隐藏 z=0 网格地面。"""
        self.show_grid_floor = bool(on)
        if not CAN_RENDER:
            return
        self.plotter.remove_actor("_grid_floor", reset_camera=False)
        self._refresh_grid_floor()
        self.plotter.render()

    def _refresh_grid_floor(self) -> None:
        """按当前状态和背景重画网格地面。"""
        if not CAN_RENDER or not self.show_grid_floor:
            return
        try:
            import pyvista as pv
            # 地面范围跟着模型走，没有模型就给一个默认 20×20
            half = 10.0
            if self._frame is not None:
                coords = np.asarray(self._frame.get("coordinates") or [])
                if coords.size:
                    span = max(
                        float(np.ptp(coords[:, 0])),
                        float(np.ptp(coords[:, 1])), 1.0)
                    half = max(span * 0.65, 2.0)
                    cx = float(np.mean(coords[:, 0]))
                    cy = float(np.mean(coords[:, 1]))
                else:
                    cx = cy = 0.0
            else:
                cx = cy = 0.0
            plane = pv.Plane(center=(cx, cy, 0),
                             direction=(0, 0, 1),
                             i_size=half * 2, j_size=half * 2,
                             i_resolution=int(max(half, 1)),
                             j_resolution=int(max(half, 1)))
            line_color = "#bbbbbb" if self._is_light_bg() else "#3a424e"
            fill_color = "#dfe3e8" if self._is_light_bg() else "#161a20"
            self.plotter.add_mesh(
                plane, color=fill_color, opacity=0.35,
                show_edges=True, edge_color=line_color,
                name="_grid_floor", reset_camera=False)
        except Exception:
            pass

    # --- 基础 ---

    def resizeEvent(self, event):
        """窗口大小变化时，更新坐标系指示器位置。"""
        super().resizeEvent(event)
        if hasattr(self, 'axis_indicator'):
            self.axis_indicator.move(
                10,
                self.height() - self.axis_indicator.height() - 10
            )
            self.axis_indicator.raise_()

    def clear(self) -> None:
        if not CAN_RENDER:
            return
        self.plotter.clear()
        # 按当前主题/自定义颜色恢复背景
        self._apply_current_background()

    # --- 标注与选择 ---

    def _decorate(self, frame) -> None:
        """编号标注 + 选中高亮。**每种显示模式画完都要调一次**，
        否则切到云图选中就没了，用户会以为选择被清掉了。"""
        self._frame = frame
        if frame is None or not CAN_RENDER:
            return
        # 模型范围变了，网格地面跟着重算大小
        if self.show_grid_floor:
            self.plotter.remove_actor("_grid_floor", reset_camera=False)
            self._refresh_grid_floor()
        if self.show_labels:
            pts, txt = scene.node_labels(frame)
            self.plotter.add_point_labels(
                pts, txt, name="_node_labels", font_size=10,
                text_color=theme.VIEWPORT_INK, shape=None, always_visible=True,
                show_points=False)
            pts, txt = scene.member_labels(frame)
            self.plotter.add_point_labels(
                pts, txt, name="_member_labels", font_size=10,
                text_color=theme.ACCENT, shape=None, always_visible=True,
                show_points=False)
        if self.selection:
            kind, ident = self.selection
            if kind == "member":
                mesh = scene.highlight_members(frame, [ident])
                if mesh.n_cells:
                    self.plotter.add_mesh(mesh, color=theme.HIGHLIGHT,
                                          name="_selection", opacity=0.85)
                for axis, glyph in scene.member_local_axis_glyphs(
                        frame, ident).items():
                    self.plotter.add_mesh(
                        glyph, color=theme.LOCAL_AXES[axis],
                        name=f"_local_axis_{axis}", smooth_shading=True,
                        render=True)
                points, texts = scene.member_local_axis_labels(frame, ident)
                if points:
                    self.plotter.add_point_labels(
                        points, texts, name="_local_axis_labels", font_size=9,
                        text_color=theme.VIEWPORT_INK, shape=None,
                        always_visible=True, show_points=False)
            else:
                mesh = scene.highlight_nodes(frame, [ident])
                if mesh.n_points:
                    self.plotter.add_mesh(mesh, color=theme.HIGHLIGHT,
                                          name="_selection", point_size=16,
                                          render_points_as_spheres=True)
        member_ids = [ident for kind, ident in self.problem_refs
                      if kind == "member"]
        node_ids = [ident for kind, ident in self.problem_refs
                    if kind == "node"]
        if member_ids:
            mesh = scene.highlight_members(frame, member_ids, radius_scale=2.0)
            if mesh.n_cells:
                self.plotter.add_mesh(mesh, color=theme.ERROR,
                                      name="_problem_members", opacity=0.9)
        if node_ids:
            mesh = scene.highlight_nodes(frame, node_ids)
            if mesh.n_points:
                self.plotter.add_mesh(mesh, color=theme.ERROR,
                                      name="_problem_nodes", point_size=20,
                                      render_points_as_spheres=True)

    def set_selection(self, kind: str | None, ident: int | None) -> None:
        self.selection = (kind, ident) if kind and ident is not None else None
        if self._frame is not None and CAN_RENDER:
            for name in ("_selection", "_local_axis_x", "_local_axis_y",
                         "_local_axis_z", "_local_axis_labels"):
                self.plotter.remove_actor(name, reset_camera=False)
            self._decorate(self._frame)
            self.plotter.render()

    def set_problem_refs(self, refs: list[tuple[str, int]]) -> None:
        """同时标出一个诊断问题涉及的节点和杆件。"""
        self.problem_refs = [(kind, int(ident)) for kind, ident in refs
                             if kind in {"node", "member"}]
        if self._frame is not None and CAN_RENDER:
            for name in ("_problem_members", "_problem_nodes"):
                self.plotter.remove_actor(name, reset_camera=False)
            self._decorate(self._frame)
            self.plotter.render()

    def set_result_marker(self, point, label: str = "") -> None:
        """在结果极值或探针截面处放置可追溯标记。"""
        if not CAN_RENDER or self._frame is None:
            return
        # pyvista 在这个文件里一律用时再导（它和 VTK 加起来启动开销不小，
        # 而且无头环境根本装不上）。漏了这一行的后果不会在测试里露头：
        # 测试跑在 QT_QPA_PLATFORM=offscreen 下，CAN_RENDER 是 False，
        # 上面那个 return 就走掉了——只有真机上点"结果探针"或"定位极值"
        # 才会撞出 NameError。是 ruff 的 F821 把它翻出来的。
        import pyvista as pv

        for name in ("_result_marker", "_result_marker_label"):
            self.plotter.remove_actor(name, reset_camera=False)
        marker = pv.Sphere(radius=0.010 * scene.model_size(self._frame),
                           center=np.asarray(point, dtype=float),
                           theta_resolution=24, phi_resolution=24)
        self.plotter.add_mesh(marker, color=theme.HIGHLIGHT,
                              name="_result_marker", smooth_shading=True)
        if label:
            self.plotter.add_point_labels(
                [point], [label], name="_result_marker_label", font_size=10,
                text_color=theme.VIEWPORT_INK, shape=None,
                always_visible=True, show_points=False)
        self.plotter.render()

    def set_labels(self, on: bool) -> None:
        self.show_labels = bool(on)
        if self._frame is not None and CAN_RENDER:
            for name in ("_node_labels", "_member_labels"):
                self.plotter.remove_actor(name, reset_camera=False)
            self._decorate(self._frame)
            self.plotter.render()

    # --- 精确建模：工作平面 / 网格捕捉 / 已有节点吸附 ---

    # (代号, 说明, 被固定的坐标轴)
    WORK_PLANES = (("XY", "XY 水平面（z=常数）", "z"),
                   ("XZ", "XZ 竖直面（y=常数）", "y"),
                   ("YZ", "YZ 竖直面（x=常数）", "x"))

    def set_work_plane(self, plane: str, offset: float | None = None) -> None:
        if plane in ("XY", "XZ", "YZ"):
            self.work_plane = plane
        if offset is not None:
            try:
                self.work_offset = float(offset)
            except (TypeError, ValueError):
                pass
        self._update_mode_badge()

    def set_work_offset(self, offset: float) -> None:
        try:
            self.work_offset = float(offset)
        except (TypeError, ValueError):
            return
        self._update_mode_badge()

    def set_snap_size(self, size: float) -> None:
        try:
            size = float(size)
        except (TypeError, ValueError):
            size = 0.0
        self.snap_size = max(0.0, size)
        self._update_mode_badge()

    def _plane_axis(self) -> int:
        # 工作平面固定的是哪根轴：XY->z(2), XZ->y(1), YZ->x(0)
        return {"XY": 2, "XZ": 1, "YZ": 0}[self.work_plane]

    def _project_to_work_plane(self, pt) -> np.ndarray:
        out = np.asarray(pt, dtype=float).copy()
        out[self._plane_axis()] = self.work_offset
        return out

    def _snap_point(self, pt) -> np.ndarray:
        """先压到工作平面，再在平面内按捕捉间距取整。"""
        out = np.asarray(pt, dtype=float).copy()
        fixed = self._plane_axis()
        out[fixed] = self.work_offset
        if self.snap_size > 0:
            for i in range(3):
                if i != fixed:
                    out[i] = round(out[i] / self.snap_size) * self.snap_size
        return out

    def _snap_to_existing_node(self, pt):
        """吸附半径内已有节点则返回其编号，否则 None（避免重复节点）。"""
        if self._frame is None:
            return None
        try:
            size = scene.model_size(self._frame) or 1.0
        except Exception:                      # noqa: BLE001
            size = 1.0
        limit = self._node_snap_ratio * size
        best, best_d = None, float("inf")
        for nid in self._frame.order():
            d = float(np.linalg.norm(pt - self._frame.nodes[nid].xyz))
            if d < best_d:
                best, best_d = nid, d
        return best if best_d <= limit else None

    def _update_mode_badge(self) -> None:
        """左上角模式徽章：当前模式 + 工作平面 + 捕捉，给明确的操作反馈。"""
        if self.model_mode:
            name = {"node": "建节点", "member": "建杆件"}.get(self.model_mode, "")
        elif self.pick_mode:
            name = {"node": "选择节点", "member": "选择杆件"}.get(self.pick_mode, "")
        else:
            self.mode_badge.setVisible(False)
            return
        ax_char = {"XY": "z", "XZ": "y", "YZ": "x"}[self.work_plane]
        bits = [name,
                f"工作平面 {self.work_plane}（{ax_char}={self.work_offset:g}）"]
        if self.snap_size > 0:
            bits.append(f"网格捕捉 {self.snap_size:g}")
        bits.append("Esc 退出")
        self.mode_badge.setText("　｜　".join(bits))
        self.mode_badge.adjustSize()
        self.mode_badge.move(12, 12)
        self.mode_badge.setVisible(True)
        self.mode_badge.raise_()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def set_pick_mode(self, mode: str | None) -> None:
        """设定拾取过滤器。传 None 关闭拾取，回到纯看图。"""
        self.pick_mode = mode
        try:
            self.plotter.disable_picking()
        except Exception:                      # noqa: BLE001
            pass                               # 本来就没开过，不是错误
        if mode is None:
            self._update_mode_badge()
            return
        try:
            self.plotter.enable_point_picking(
                callback=self._on_pick, show_message=False,
                show_point=False, use_picker=True, left_clicking=True)
        except Exception:                      # noqa: BLE001
            # 无头环境没有交互器。**不能因此崩** —— 拾取只是锦上添花，
            # 画图和求解不该被它拖累
            pass
        self._update_mode_badge()

    def set_model_mode(self, mode: str | None) -> None:
        """设定人工建模模式。None 关闭，"node" 点击创建节点，"member" 点击两节点创建杆件。"""
        self.model_mode = mode
        self._pending_node = None
        self.pick_mode = None  # 建模模式和拾取模式互斥
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        if mode is None:
            self._update_mode_badge()
            return
        try:
            self.plotter.enable_point_picking(
                callback=self._on_pick, show_message=False,
                show_point=False, use_picker=True, left_clicking=True,
                # 空模型没有可拾取网格。建节点时仍要允许在当前相机焦平面
                # 上取世界坐标，否则第一个节点永远点不出来。
                pickable_window=(mode == "node"))
        except Exception:
            pass
        self._update_mode_badge()

    def _on_pick(self, point, *_args) -> None:
        """拾取回调。建模模式下创建节点/杆件，普通模式下选中对象。"""
        if point is None:
            return
        pt = np.asarray(point, dtype=float)

        # 人工建模：节点模式 —— 投影到工作平面、网格捕捉、吸附已有节点后创建
        if self.model_mode == "node":
            snapped = self._snap_point(pt)
            existing = self._snap_to_existing_node(snapped)
            if existing is not None:
                # 点在了已有节点附近：不重复建，直接吸附并提示
                self.set_selection("node", existing)
                self.node_snapped.emit(existing)
                return
            self.node_created.emit(float(snapped[0]),
                                   float(snapped[1]),
                                   float(snapped[2]))
            return

        # 首个节点尚未建立时，没有可供选择的对象；但节点创建模式仍应可用。
        if self._frame is None:
            return

        # 人工建模：杆件模式 —— 点击第一个节点，再点击第二个节点创建杆件。
        # 用比普通选择略宽的容差，更容易点中要连接的节点。
        if self.model_mode == "member":
            found = scene.nearest(self._frame, pt, "node", tolerance=0.14)
            if found is None:
                return
            if self._pending_node is None:
                self._pending_node = found
                self.set_selection("node", found)
            else:
                if found != self._pending_node:
                    self.member_created.emit(self._pending_node, found)
                self._pending_node = None
                self.set_selection(None, None)
            return

        # 普通拾取模式
        if self.pick_mode is None:
            return
        found = scene.nearest(self._frame, pt, self.pick_mode)
        if found is None:
            return
        self.set_selection(self.pick_mode, found)
        self.picked.emit(self.pick_mode, found)
        if self.pick_mode == "member":
            self.probed.emit(found, pt.copy())

    def _fit(self) -> None:
        """首次显示时摆好相机；之后保持用户转过的视角。

        每次重画都重置相机的话，改一个参数就把视角弹回去，用起来很恼火。
        """
        if self._first_render:
            self._apply_view(scene.preferred_view(self._frame))
            self._first_render = False

    def _apply_view(self, name: str) -> None:
        """平面视图用正交投影，等轴测用透视投影。"""
        if name == "isometric":
            self.plotter.disable_parallel_projection()
        else:
            self.plotter.enable_parallel_projection()
        getattr(self.plotter, f"view_{name}", self.plotter.view_isometric)()
        self.plotter.camera.zoom(1.3)

    def reset_camera(self) -> None:
        if not CAN_RENDER:
            return
        self._apply_view(scene.preferred_view(self._frame))
        self.plotter.render()

    def set_view(self, name: str) -> None:
        """标准视角。CAE 里这几个按钮是必备的。"""
        if not CAN_RENDER:
            return
        self._apply_view(name)
        self.plotter.render()

    # --- 显示模式 ---

    def _add_legend(self, entries: list) -> None:
        """在视口角上列出画了哪些东西。

        支座和荷载靠形状区分，但形状是要学的；没有图例，用户只能猜那个小锥子
        是铰接还是滚动。**符号系统必须自带说明**，否则它只是好看，不是可读。
        条目为空时不画——一个空框比没有框更碍事。
        """
        if not entries:
            return
        try:
            self.plotter.add_legend(
                labels=[[label, color] for label, color in entries],
                bcolor=theme.VIEWPORT_BG, border=False,
                # 尺寸压到刚好够读：0.20/0.045 那一档实测把左上角的梁盖住了，
                # 图例挡住它要解释的东西，等于没有。
                size=(0.115, min(0.24, 0.024 * len(entries))),
                loc="upper left", face="none", font_family="arial")
        except Exception:      # 图例画不出来不该让整个视口刷新失败
            pass

    def _member_ink(self) -> tuple[str, str]:
        """杆件本体色与棱边色，按底色深浅各给一套。

        深底上那套钢灰蓝画到浅底上会发灰发糊；反过来浅底那套画在深底上
        又太亮，抢走结果的注意力。同一个颜色配两种底色，总有一边是错的。
        """
        return (("#6e8098", "#3c4a5a") if self._is_light_bg()
                else (theme.MEMBER, "#4a5764"))

    def _add_members(self, frame):
        """画杆件本体。**形状始终是圆管**——粗细和形态不随截面变，
        改的只有随底色深浅走的本体色。返回网格，空模型返回 None。"""
        body, _edge = self._member_ink()
        mesh = scene.member_tubes(frame)
        if mesh.n_points:
            self.plotter.add_mesh(mesh, color=body, smooth_shading=True,
                                  pbr=True, metallic=0.28, roughness=0.58,
                                  ambient=0.18, diffuse=0.82)
        return mesh

    def set_contour_palette(self, palette: str) -> bool:
        """切换云图色系。返回是否真的变了，省得上层白刷一次视口。"""
        if palette not in theme.CONTOUR_PALETTES or palette == self.contour_palette:
            return False
        self.contour_palette = palette
        return True

    def set_contour_shading(self, on: bool) -> bool:
        """云图打不打光。关掉是平涂，颜色和色标严格一一对应。"""
        on = bool(on)
        if on == self.contour_shading:
            return False
        self.contour_shading = on
        return True

    def set_load_labels(self, on: bool) -> bool:
        """荷载旁边标不标数值。"""
        on = bool(on)
        if on == self.load_labels:
            return False
        self.load_labels = on
        return True

    def show_model(self, frame, case: str | None = None,
                   supports: bool = True, loads: bool = True) -> dict:
        """求解前的模型视图。"""
        if not CAN_RENDER:
            self._frame = frame
            self._first_render = False
            return {"render_skipped": "QT_QPA_PLATFORM=offscreen"}
        self.clear()
        info: dict = {}
        tubes = self._add_members(frame)
        links = scene.rigid_link_polylines(frame)
        if links.n_points:
            self.plotter.add_mesh(links, color=theme.REFERENCE, line_width=5)
        self.plotter.add_mesh(scene.node_points(frame), color=theme.VIEWPORT_INK,
                              point_size=4, render_points_as_spheres=True)
        hinges = scene.hinge_glyphs(frame)
        if hinges.n_points:
            self.plotter.add_mesh(hinges, color=theme.HINGE)
        if supports:
            glyphs = scene.support_glyphs(frame)
            for mesh in glyphs.values():
                self.plotter.add_mesh(
                    mesh, color=theme.SUPPORT,
                    smooth_shading=True, ambient=0.28, diffuse=0.72,
                    specular=0.10, specular_power=12)
            support_points, support_texts = scene.support_labels(frame)
            if support_points:
                self.plotter.add_point_labels(
                    support_points, support_texts, font_size=9,
                    text_color=theme.VIEWPORT_INK_MUTED,
                    shape=None, always_visible=True, show_points=False)
            info["supports"] = {k: 1 for k in glyphs}
        legend: list[tuple[str, str]] = []
        if tubes is not None and tubes.n_points:
            legend.append((_LEGEND_TEXT["杆件"], self._member_ink()[0]))
        if hinges.n_points:
            legend.append((_LEGEND_TEXT["杆端铰"], theme.HINGE))
        if supports:
            legend += [(_LEGEND_TEXT.get(kind, kind), theme.SUPPORT)
                       for kind in info.get("supports", {})]
        if loads and case:
            arrows = scene.load_arrows(frame, case)
            for label, mesh in arrows.items():
                color = (theme.LOAD_MOMENT
                         if label in {"节点力矩", "给定位移", "给定转角"}
                         else theme.LOAD)
                self.plotter.add_mesh(
                    mesh, color=color, smooth_shading=True,
                    ambient=0.35, diffuse=0.65)
                legend.append((_LEGEND_TEXT.get(label, label), color))
            # **数值直接标在旁边。** 四类荷载用四种颜色跑不过 all-pairs 校验，
            # 而且颜色只回答"哪一类"、回答不了"多大"——后者才是工程师要看的。
            points, texts = scene.load_labels(frame, case)
            if points and self.load_labels:
                self.plotter.add_point_labels(
                    # 文字用文字色，不用系列色：标注是说明，不是又一个分类。
                    # 力矩标成橙色而箭头是绿色，本身就自相矛盾。
                    points, texts, font_size=11, text_color=theme.VIEWPORT_INK,
                    shape=None, always_visible=True, show_points=False)
            info["loads"] = list(arrows)
        self._add_legend(legend)
        self._decorate(frame)
        self._fit()
        self.plotter.render()
        return info

    def show_analysis_mesh(self, frame, preview: dict) -> None:
        """显示求解器实际消费的分析单元，同时保留物理构件轮廓。"""
        if not CAN_RENDER:
            self._frame = frame
            self._first_render = False
            return
        self.clear()
        self.plotter.add_mesh(scene.member_polylines(frame),
                              color=theme.REFERENCE, line_width=2)
        analysis = scene.analysis_mesh_polylines(frame, preview)
        if analysis.n_cells:
            radius = max(scene.model_size(frame) * scene.TUBE_RATIO * 0.65, 1e-9)
            self.plotter.add_mesh(analysis.tube(
                                      radius=radius, n_sides=32, capping=True),
                                  color=theme.ACCENT, smooth_shading=True)
        self.plotter.add_mesh(scene.node_points(frame), color=theme.VIEWPORT_INK,
                              point_size=5, render_points_as_spheres=True)
        splits = scene.analysis_split_points(preview)
        if splits.n_points:
            self.plotter.add_mesh(splits, color=theme.HINGE, point_size=15,
                                  render_points_as_spheres=True)
            labels = [f"切分节点 {int(node_id)}"
                      for node_id in splits.point_data["node"]]
            self.plotter.add_point_labels(
                splits.points, labels, font_size=10, text_color=theme.HINGE,
                shape=None, show_points=False, always_visible=True)
        summary = (f"物理杆件 {preview.get('physical_members', 0)}  →  "
                   f"分析杆段 {preview.get('analysis_elements', 0)}\n"
                   f"切分节点 {preview.get('split_nodes', 0)}")
        self.plotter.add_text(summary, position="upper_left",
                              color=theme.VIEWPORT_INK, font_size=10)
        self._decorate(frame)
        self._fit()
        self.plotter.render()

    def show_deformed(self, frame, solution, case: str, scale: float,
                      overlay: bool = True, supports: bool = True) -> None:
        if not CAN_RENDER:
            self._frame = frame
            self._first_render = False
            return
        self.clear()
        if overlay:
            # 未变形轮廓画成细线而不是管：它是参照物，不该和主体抢注意力
            self.plotter.add_mesh(scene.member_polylines(frame),
                                  color=theme.REFERENCE, line_width=1.5)
        tubes = scene.member_tubes(frame, solution, case, scale=scale)
        self.plotter.add_mesh(
            tubes, color=theme.MEMBER, smooth_shading=True,
            pbr=True, metallic=0.22, roughness=0.62,
            ambient=0.20, diffuse=0.80)
        links = scene.rigid_link_polylines(frame, solution, case, scale)
        if links.n_points:
            self.plotter.add_mesh(links, color=theme.REFERENCE, line_width=5)
        self.plotter.add_mesh(
            scene.displaced_node_points(frame, solution[case].U, scale),
            color=theme.MEMBER, point_size=5, render_points_as_spheres=True)
        if supports:
            for mesh in scene.support_glyphs(frame).values():
                self.plotter.add_mesh(
                    mesh, color=theme.SUPPORT, smooth_shading=True,
                    ambient=0.28, diffuse=0.72,
                    specular=0.10, specular_power=12)
        self._decorate(frame)
        self._fit()
        self.plotter.render()

    def show_contour(self, frame, solution, case: str, component: str,
                     scale: float = 0.0, title: str | None = None,
                     percentile: float | None = scene.CONTOUR_PERCENTILE,
                     sign_filter: str = "all", overlay_deformed: bool = False,
                     show_extrema: bool = True,
                     levels: int | None = None) -> tuple:
        """内力 / 应力云图。合量从零起，有符号量关于零对称。

        **分级着色（banded contour）**，级数默认 12，与色标上的分格一一对应。
        连续渐变上读不出"这一段到底是多少"，只读得出"这边比那边红"。
        """
        if not CAN_RENDER:
            self._frame = frame
            self._first_render = False
            return (0.0, 0.0)
        self.clear()
        # 云图和二维内力图都用工程显示单位，避免 MM 模型把 N·mm 标成 N·m。
        from units import of as unit_system
        system = unit_system(frame)
        if component == scene.STRESS:
            scalar_scale, unit = system.stress_scale, system.stress_unit
        elif component in {"T", "My", "Mz", "M"}:
            scalar_scale, unit = system.moment_scale, system.moment_unit
        else:
            scalar_scale, unit = system.force_scale, system.force_unit
        if component in {"M", "V"}:
            sign_filter = "all"       # 合量定义为非负，正负筛选不适用
        line = scene.contour_line(frame, solution, case, component,
                                  scale=scale, value_scale=scalar_scale,
                                  sign_filter=sign_filter)
        clim = scene.contour_clim(line, component, percentile=percentile)
        clipped = scene.clim_is_clipped(line, component, clim)
        n = scene.contour_levels(levels)
        base = theme.palette_cmap(self.contour_palette, component)
        cmap = theme.banded(base, n)
        tubes = scene.banded_tubes(
            line, component, clim, n,
            radius=scene.CONTOUR_TUBE_RATIO * scene.model_size(frame))
        # 打光与读数是一对矛盾：打了光，同一个数值在向光面和背光面是两个
        # 颜色，而看图的人正是拿杆件上的颜色去对色标读数的；完全不打光，
        # 圆管就是一条扁色带，看不出这是根三维杆件。
        #
        # 折中是**环境光托底 + 适度漫反射 + 一点高光**：弧面读得出来，
        # 每一级的颜色仍然认得出是哪一级。Abaqus 的云图也是打光的。
        # 要严格照色标读数可以关掉（set_contour_shading），那时是纯平涂。
        #
        # 环境光别调得太高。原来取 0.66/0.34 想把色偏压到最小，结果是
        # 明暗差被压没了，管子看上去还是一条扁色带——光照参数救不了太细的管，
        # 真正起作用的是 scene.CONTOUR_TUBE_RATIO 那一次加粗。
        shade = dict(lighting=True, ambient=0.42, diffuse=0.58,
                     specular=0.22, specular_power=30, smooth_shading=True)
        self.plotter.add_mesh(
            tubes, scalars=component + scene.BAND_SUFFIX,
            cmap=cmap, clim=clim, n_colors=n, show_scalar_bar=False,
            **(shade if self.contour_shading else {"lighting": False}))
        # 色标竖着放在右侧：横放时 VTK 把标题和刻度挤在同一条带上（实测重叠），
        # 而且十几级的刻度横向根本排不开。
        #
        # **必须紧挨着云图那一层加。** add_scalar_bar 绑的是最后一个加进来的
        # 网格的映射器；等把"量程外"那层纯色网格加完再加色标，色标画的就是
        # 那层的默认查找表——一条 0…1 的彩虹，和图上任何东西都对不上。
        self.plotter.add_scalar_bar(
            title="", n_labels=n + 1, n_colors=n, vertical=True, fmt="%.3g",
            color=theme.VIEWPORT_INK_MUTED, label_font_size=11,
            width=0.040, height=0.58, position_x=0.905, position_y=0.14)
        if clipped:
            # 超出量程的段单独画。分级之后饱和的那一级和正常的一级长得一样，
            # 峰值所在的位置会消失在一片同色里。
            over = scene.out_of_range_tubes(
                line, component, clim,
                radius=scene.CONTOUR_TUBE_RATIO * scene.model_size(frame) * 1.02)
            if over.n_cells:
                self.plotter.add_mesh(over, color=theme.HIGHLIGHT,
                                      lighting=False, show_scalar_bar=False,
                                      name="_contour_out_of_range")
        self._contour_last = dict(palette=self.contour_palette,
                                  shading=self.contour_shading, bands=n)
        # 标题自己画。交给 VTK 画会和最上面那个刻度撞在一起。
        # 披露文字用 ASCII：VTK 的色标与文字用自己的字体引擎，默认字体没有
        # 中文字形（实测中文渲染成方块）。写成方块等于没披露。
        label = (scene.STRESS_LABEL_ASCII if component == scene.STRESS
                 else component)
        bar_title = f"{label}  [{unit}]\n{n} bands"
        if clipped:
            bar_title += (f"\nclip p{scene.CONTOUR_PERCENTILE:.0f}"
                          "\noff scale: orange")
        self.plotter.add_text(
            bar_title, position=(0.795, 0.735), viewport=True,
            color=theme.VIEWPORT_INK, font_size=10, name="_contour_bar_title")
        self.plotter.add_text(
            scene.contour_caption(component, unit, clipped, sign_filter, n),
            position="upper_left", color=theme.VIEWPORT_INK,
            font_size=9, name="_contour_definition")
        if overlay_deformed:
            deformation_scale = scale or scene.auto_deformation_scale(
                frame, solution, case)
            deformed = scene.member_tubes(
                frame, solution, case, scale=deformation_scale,
                radius=scene.CONTOUR_TUBE_RATIO * scene.model_size(frame) * 0.45)
            self.plotter.add_mesh(
                deformed, color=theme.HIGHLIGHT, opacity=0.48,
                name="_contour_deformed_overlay", smooth_shading=True)
        if show_extrema:
            from .result_inspector import global_extreme
            extreme = global_extreme(frame, solution, component, case)
            if extreme["member"] is not None:
                peak_text = (f"{label}={extreme['value']:+.3g} {extreme['unit']}"
                             f" | M{extreme['member']} x={extreme['x']:.3g}")
                self.plotter.add_point_labels(
                    [extreme["point"]], [peak_text], name="_contour_extreme",
                    font_size=9, text_color=theme.VIEWPORT_INK, shape=None,
                    always_visible=True, show_points=True,
                    point_color=theme.HIGHLIGHT, point_size=9)
        for mesh in scene.support_glyphs(frame).values():
            self.plotter.add_mesh(
                mesh, color=theme.SUPPORT, smooth_shading=True,
                ambient=0.28, diffuse=0.72,
                specular=0.10, specular_power=12)
        self._decorate(frame)
        self._fit()
        self.plotter.render()
        return clim

    def show_mode(self, frame, shapes: np.ndarray, mode: int,
                  label: str = "") -> None:
        """振型 / 失稳模态。振型无量纲，按模型尺寸定一个好看的放大倍数。"""
        if not CAN_RENDER:
            self._frame = frame
            self._first_render = False
            return
        self.clear()
        phi = shapes[:, mode]
        # 转角是弧度、平移是长度，不能把两者混在一起取 max；应按梁形函数
        # 恢复后的中心线平移定显示比例，否则转角占优的振型会几乎看不见。
        scale = scene.auto_mode_scale(frame, shapes, mode)
        self.plotter.add_mesh(scene.member_polylines(frame),
                              color=theme.REFERENCE, line_width=1.5)
        self.plotter.add_mesh(
            scene.mode_shape_tubes(frame, shapes, mode, scale),
            color=theme.HIGHLIGHT, smooth_shading=True,
            pbr=True, metallic=0.16, roughness=0.66,
            ambient=0.20, diffuse=0.80)
        links = scene.rigid_link_polylines(
            frame, scale=scale, mode_vector=phi)
        if links.n_points:
            self.plotter.add_mesh(links, color=theme.REFERENCE, line_width=5)
        self.plotter.add_mesh(
            scene.displaced_node_points(frame, phi, scale),
            color=theme.HIGHLIGHT, point_size=5, render_points_as_spheres=True)
        if label:
            self.plotter.add_text(label, position="upper_left",
                                  color=theme.VIEWPORT_INK, font_size=10)
        self._decorate(frame)
        self._fit()
        self.plotter.render()

    def shutdown(self) -> None:
        """关掉渲染窗口。**关窗口时必须调**——VTK 的 render window 不显式
        finalize，进程退出时会崩在析构里。调用方只是点了个关闭，
        不该看到"程序已停止工作"。"""
        try:
            self.plotter.close()
        except Exception:                       # noqa: BLE001
            pass

    def screenshot(self, path: str) -> str:
        self.plotter.screenshot(path)
        return path
