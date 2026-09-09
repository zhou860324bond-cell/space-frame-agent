"""2D 草图对话框 —— Abaqus 式的草图建模。

在 2D 平面上画点、画线，构成一榀框架的草图，然后沿进深方向拉伸成 3D 空间刚架。

操作流程：
1. 选择草图平面（XZ 立面 / YZ 侧立 / XY 平面）
2. 点「画点」在画布上点击创建节点
3. 点「画线」依次点击两个节点创建杆件
4. 点「删除」选中对象后删除
5. 设置拉伸参数（进深方向、距离、榀数）
6. 点「生成模型」生成 3D 模型
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from . import theme


class SketchCanvas(QWidget):
    """2D 草图画布。

    坐标系统：画布中心为原点，x 向右，y 向上（数学坐标）。
    缩放比例：scale 像素/单位。
    """

    changed = Signal()  # 节点或杆件变化时发射

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.nodes: list[tuple[float, float]] = []
        self.lines: list[tuple[int, int]] = []
        self.mode = "chain"  # point / line / chain / delete / rectangle
        self.pending_line_node: int | None = None
        self.pending_rect_corner: tuple[float, float] | None = None
        self.scale = 30.0
        self.mouse_x = 0
        self.mouse_y = 0
        self._history: list[tuple[list[tuple[float, float]],
                                  list[tuple[int, int]]]] = []

    def set_mode(self, mode: str):
        self.mode = mode
        self.pending_line_node = None
        self.pending_rect_corner = None
        self.update()

    def clear(self):
        if self.nodes or self.lines:
            self._remember()
        self.nodes = []
        self.lines = []
        self.pending_line_node = None
        self.pending_rect_corner = None
        self.update()
        self.changed.emit()

    def undo(self):
        if not self._history:
            return
        self.nodes, self.lines = self._history.pop()
        self.pending_line_node = None
        self.pending_rect_corner = None
        self.update()
        self.changed.emit()

    def _remember(self) -> None:
        """按一次用户操作保存快照；矩形撤销时应一次撤掉四条边。"""
        self._history.append((list(self.nodes), list(self.lines)))

    def _node_at(self, point: tuple[float, float], tol: float = 1e-9) -> int | None:
        for index, existing in enumerate(self.nodes):
            if abs(existing[0] - point[0]) <= tol and abs(existing[1] - point[1]) <= tol:
                return index
        return None

    def _get_or_add_node(self, point: tuple[float, float]) -> int:
        existing = self._node_at(point)
        if existing is not None:
            return existing
        self.nodes.append(point)
        return len(self.nodes) - 1

    def _add_line(self, n1: int, n2: int) -> bool:
        """添加非零、非重复线段；方向相反仍视为同一根杆件。"""
        if n1 == n2 or self.nodes[n1] == self.nodes[n2]:
            return False
        key = (min(n1, n2), max(n1, n2))
        if any((min(a, b), max(a, b)) == key for a, b in self.lines):
            return False
        self.lines.append((n1, n2))
        return True

    def _to_world(self, x: int, y: int) -> tuple[float, float]:
        cx = self.width() / 2
        cy = self.height() / 2
        wx = (x - cx) / self.scale
        wy = (cy - y) / self.scale
        return round(wx, 3), round(wy, 3)

    def _to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        cx = self.width() / 2
        cy = self.height() / 2
        sx = cx + wx * self.scale
        sy = cy - wy * self.scale
        return sx, sy

    def _nearest_node(self, x: int, y: int, threshold: float = 12) -> int | None:
        best = None
        best_dist = threshold
        for i, (wx, wy) in enumerate(self.nodes):
            sx, sy = self._to_screen(wx, wy)
            dist = ((sx - x) ** 2 + (sy - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = i
        return best

    def _nearest_line(self, x: int, y: int, threshold: float = 8) -> int | None:
        best = None
        best_dist = threshold
        for i, (n1, n2) in enumerate(self.lines):
            if n1 >= len(self.nodes) or n2 >= len(self.nodes):
                continue
            x1, y1 = self._to_screen(*self.nodes[n1])
            x2, y2 = self._to_screen(*self.nodes[n2])
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                dist = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
            else:
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
                px, py = x1 + t * dx, y1 + t * dy
                dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = i
        return best

    def mousePressEvent(self, event):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = event.position()
        x, y = int(pos.x()), int(pos.y())

        # 右键结束当前绘制命令。
        if event.button() == Qt.MouseButton.RightButton:
            self.pending_line_node = None
            self.pending_rect_corner = None
            self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.mode == "point":
            wx, wy = self._to_world(x, y)
            if self._node_at((wx, wy)) is not None:
                return
            self._remember()
            self.nodes.append((wx, wy))
            self.update()
            self.changed.emit()

        elif self.mode == "line":
            node = self._nearest_node(x, y)
            if node is None:
                return
            if self.pending_line_node is None:
                self.pending_line_node = node
            else:
                if node != self.pending_line_node:
                    self._remember()
                    self._add_line(self.pending_line_node, node)
                self.pending_line_node = None
            self.update()
            self.changed.emit()

        elif self.mode == "chain":
            # Abaqus 式连续画线：点击自动创建端点并连线
            node = self._nearest_node(x, y)
            if node is None:
                # 点击空白处，创建新节点
                wx, wy = self._to_world(x, y)
                self._remember()
                self.nodes.append((wx, wy))
                new_idx = len(self.nodes) - 1
                if self.pending_line_node is not None:
                    self._add_line(self.pending_line_node, new_idx)
                self.pending_line_node = new_idx
            else:
                # 点击已有节点
                if self.pending_line_node is not None and node != self.pending_line_node:
                    key = (min(self.pending_line_node, node),
                           max(self.pending_line_node, node))
                    if not any((min(a, b), max(a, b)) == key for a, b in self.lines):
                        self._remember()
                        self._add_line(self.pending_line_node, node)
                self.pending_line_node = node
            self.update()
            self.changed.emit()

        elif self.mode == "rectangle":
            # Abaqus 式画矩形：点击两个对角，自动创建4个节点和4条边
            wx, wy = self._to_world(x, y)
            if self.pending_rect_corner is None:
                self.pending_rect_corner = (wx, wy)
            else:
                x1, y1 = self.pending_rect_corner
                x2, y2 = wx, wy
                if abs(x2 - x1) <= 1e-9 or abs(y2 - y1) <= 1e-9:
                    return
                self._remember()
                corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                ids = [self._get_or_add_node(point) for point in corners]
                for n1, n2 in zip(ids, ids[1:] + ids[:1]):
                    self._add_line(n1, n2)
                self.pending_rect_corner = None
            self.update()
            self.changed.emit()

        elif self.mode == "delete":
            node = self._nearest_node(x, y)
            if node is not None:
                self._remember()
                self.lines = [(n1, n2) for n1, n2 in self.lines
                              if n1 != node and n2 != node]
                idx_map = {}
                new_nodes = []
                for i, n in enumerate(self.nodes):
                    if i != node:
                        idx_map[i] = len(new_nodes)
                        new_nodes.append(n)
                self.nodes = new_nodes
                self.lines = [(idx_map[n1], idx_map[n2]) for n1, n2 in self.lines]
                self.update()
                self.changed.emit()
                return
            line = self._nearest_line(x, y)
            if line is not None:
                self._remember()
                self.lines.pop(line)
                self.update()
                self.changed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.pending_line_node = None
            self.pending_rect_corner = None
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position()
        self.mouse_x = int(pos.x())
        self.mouse_y = int(pos.y())
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale = min(self.scale * 1.15, 200)
        else:
            self.scale = max(self.scale / 1.15, 5)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)

            # 背景
            p.fillRect(self.rect(), QColor(theme.VIEWPORT_BG))

            # 网格
            self._draw_grid(p)

            # 坐标轴
            self._draw_axes(p)

            # 画线
            for n1, n2 in self.lines:
                if n1 >= len(self.nodes) or n2 >= len(self.nodes):
                    continue
                x1, y1 = self._to_screen(*self.nodes[n1])
                x2, y2 = self._to_screen(*self.nodes[n2])
                p.setPen(QPen(QColor(theme.MEMBER), 2))
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # 待画的线（从 pending 节点到鼠标）—— line 和 chain 模式都显示
            if self.mode in ("line", "chain") and self.pending_line_node is not None:
                if self.pending_line_node < len(self.nodes):
                    x1, y1 = self._to_screen(*self.nodes[self.pending_line_node])
                    p.setPen(QPen(QColor(theme.ACCENT), 1, Qt.PenStyle.DashLine))
                    p.drawLine(QPointF(x1, y1), QPointF(self.mouse_x, self.mouse_y))
                    # 显示当前线段长度
                    wx1, wy1 = self.nodes[self.pending_line_node]
                    wx2, wy2 = self._to_world(self.mouse_x, self.mouse_y)
                    length = ((wx2-wx1)**2 + (wy2-wy1)**2) ** 0.5
                    p.setPen(QColor(theme.ACCENT))
                    p.setFont(QFont("Arial", 10))
                    p.drawText(QPointF((x1+self.mouse_x)/2 + 5, (y1+self.mouse_y)/2 - 5),
                               f"{length:.2f}m")

            # 矩形预览
            if self.mode == "rectangle" and self.pending_rect_corner is not None:
                x1, y1 = self._to_screen(*self.pending_rect_corner)
                x2, y2 = self.mouse_x, self.mouse_y
                p.setPen(QPen(QColor(theme.ACCENT), 1, Qt.PenStyle.DashLine))
                p.setBrush(QBrush(QColor(theme.ACCENT, 40)))
                p.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)))
                # 显示宽高
                wx1, wy1 = self.pending_rect_corner
                wx2, wy2 = self._to_world(self.mouse_x, self.mouse_y)
                p.setPen(QColor(theme.ACCENT))
                p.setFont(QFont("Arial", 10))
                p.drawText(QPointF(min(x1,x2), min(y1,y2) - 5),
                           f"{abs(wx2-wx1):.2f} × {abs(wy2-wy1):.2f}m")

            # 画点
            for i, (wx, wy) in enumerate(self.nodes):
                sx, sy = self._to_screen(wx, wy)
                if i == self.pending_line_node:
                    p.setBrush(QBrush(QColor(theme.HIGHLIGHT)))
                    p.setPen(QPen(QColor(theme.INK), 1))
                    p.drawEllipse(QPointF(sx, sy), 7, 7)
                else:
                    p.setBrush(QBrush(QColor(theme.INK)))
                    p.setPen(QPen(QColor(theme.INK), 1))
                    p.drawEllipse(QPointF(sx, sy), 5, 5)
                p.setPen(QColor(theme.INK_MUTED))
                p.setFont(QFont("Arial", 9))
                p.drawText(QPointF(sx + 9, sy - 5), str(i + 1))

            # 鼠标坐标
            wx, wy = self._to_world(self.mouse_x, self.mouse_y)
            p.setPen(QColor(theme.INK_MUTED))
            p.setFont(QFont("Arial", 10))
            p.drawText(10, self.height() - 10,
                       f"坐标: ({wx:.2f}, {wy:.2f})   缩放: {self.scale:.0f}x")
        except Exception:
            # 绘制出错时也要结束 painter，避免 QBackingStore 错误
            import traceback
            traceback.print_exc()
        finally:
            p.end()

    def _draw_grid(self, p: QPainter):
        p.setPen(QPen(QColor(theme.BORDER), 1, Qt.PenStyle.DotLine))
        cx = self.width() / 2
        cy = self.height() / 2
        x = cx % self.scale
        while x < self.width():
            p.drawLine(int(x), 0, int(x), self.height())
            x += self.scale
        y = cy % self.scale
        while y < self.height():
            p.drawLine(0, int(y), self.width(), int(y))
            y += self.scale

    def _draw_axes(self, p: QPainter):
        cx = self.width() / 2
        cy = self.height() / 2
        p.setPen(QPen(QColor(theme.INK_MUTED), 1))
        p.drawLine(0, int(cy), self.width(), int(cy))
        p.drawLine(int(cx), 0, int(cx), self.height())
        p.setPen(QColor(theme.INK))
        p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        p.drawText(self.width() - 20, int(cy) - 5, "X")
        p.drawText(int(cx) + 5, 15, "Z")


class SketchDialog(QDialog):
    """2D 草图对话框 —— 画一榀框架，然后拉伸成 3D。"""

    def __init__(self, parent=None, materials: list[str] | None = None,
                 sections: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("草图建模 — 画一榀框架，拉伸成 3D")
        self.resize(850, 620)

        # 主布局：垂直（上：工具栏+画布，下：按钮）
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 上部：水平布局（左工具栏 + 右画布）
        top = QHBoxLayout()
        top.setSpacing(8)

        # 左侧面板
        left = QVBoxLayout()
        left.setSpacing(8)

        # 草图工具组
        tool_group = QGroupBox("草图工具")
        tool_layout = QVBoxLayout(tool_group)
        self.btn_chain = QPushButton("╱ 连续画线")
        self.btn_rect = QPushButton("▭ 画矩形")
        self.btn_point = QPushButton("● 画点")
        self.btn_delete = QPushButton("✕ 删除")
        self.btn_undo = QPushButton("↶ 撤销")
        self.btn_clear = QPushButton("清空")
        for b in (self.btn_chain, self.btn_rect, self.btn_point, self.btn_delete,
                  self.btn_undo, self.btn_clear):
            b.setMinimumHeight(32)
            tool_layout.addWidget(b)
        self.btn_chain.setCheckable(True)
        self.btn_rect.setCheckable(True)
        self.btn_point.setCheckable(True)
        self.btn_delete.setCheckable(True)
        self.btn_chain.setChecked(True)
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_chain)
        self.tool_group.addButton(self.btn_rect)
        self.tool_group.addButton(self.btn_point)
        self.tool_group.addButton(self.btn_delete)
        self.tool_group.setExclusive(True)
        left.addWidget(tool_group)

        # 操作提示
        self.lbl_hint = QLabel(
            "连续画线：点击创建端点，自动连线；\n"
            "画矩形：点击两个对角，自动创建4边；\n"
            "右键或 Esc 结束；滚轮缩放。")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet(f"color:{theme.INK_MUTED}; font-size:8pt;")
        left.addWidget(self.lbl_hint)

        # 草图平面
        plane_group = QGroupBox("草图平面")
        plane_layout = QFormLayout(plane_group)
        self.cmb_plane = QComboBox()
        self.cmb_plane.addItems(["XZ 立面（最常用）", "YZ 侧立", "XY 平面"])
        plane_layout.addRow("平面", self.cmb_plane)
        left.addWidget(plane_group)

        # 几何可以先画，属性稍后统一指派；已有属性时仍可在这里快捷选择。
        property_group = QGroupBox("杆件属性")
        property_layout = QFormLayout(property_group)
        self.cmb_material = QComboBox()
        self.cmb_material.addItem("未指派（稍后设置）", "")
        self.cmb_material.addItems(materials or [])
        self.cmb_section = QComboBox()
        self.cmb_section.addItem("未指派（稍后设置）", "")
        self.cmb_section.addItems(sections or [])
        property_layout.addRow("材料", self.cmb_material)
        property_layout.addRow("截面", self.cmb_section)
        left.addWidget(property_group)

        # 拉伸参数
        extrude_group = QGroupBox("拉伸（沿进深方向）")
        extrude_layout = QFormLayout(extrude_group)
        self.spn_bays = QSpinBox()
        self.spn_bays.setRange(1, 20)
        self.spn_bays.setValue(1)
        self.spn_bay_len = QDoubleSpinBox()
        self.spn_bay_len.setRange(0.1, 100)
        self.spn_bay_len.setValue(6.0)
        self.spn_bay_len.setSuffix(" m")
        extrude_layout.addRow("榀数", self.spn_bays)
        extrude_layout.addRow("榀距", self.spn_bay_len)
        left.addWidget(extrude_group)

        # 柱底约束
        base_group = QGroupBox("柱底约束")
        base_layout = QFormLayout(base_group)
        self.cmb_base = QComboBox()
        self.cmb_base.addItems(["固定端", "铰接"])
        base_layout.addRow("约束", self.cmb_base)
        left.addWidget(base_group)

        left.addStretch()

        # 统计
        self.lbl_stats = QLabel("节点: 0   杆件: 0")
        self.lbl_stats.setStyleSheet(f"color:{theme.INK_MUTED}; font-size:8pt;")
        left.addWidget(self.lbl_stats)

        top.addLayout(left, 1)

        # 右侧画布
        self.canvas = SketchCanvas()
        self.canvas.changed.connect(self._update_stats)
        top.addWidget(self.canvas, 4)

        root.addLayout(top, 1)

        # 底部按钮
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("生成模型")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # 连接信号
        self.btn_chain.clicked.connect(lambda: self.canvas.set_mode("chain"))
        self.btn_rect.clicked.connect(lambda: self.canvas.set_mode("rectangle"))
        self.btn_point.clicked.connect(lambda: self.canvas.set_mode("point"))
        self.btn_delete.clicked.connect(lambda: self.canvas.set_mode("delete"))
        self.btn_undo.clicked.connect(self._undo)
        self.btn_clear.clicked.connect(self._clear)

    def _update_stats(self):
        self.lbl_stats.setText(
            f"节点: {len(self.canvas.nodes)}   杆件: {len(self.canvas.lines)}")

    def _undo(self):
        self.canvas.undo()
        self._update_stats()

    def _clear(self):
        self.canvas.clear()
        self._update_stats()

    # 重写 accept，在关闭前更新统计
    def accept(self):
        self._update_stats()
        super().accept()

    def get_model_data(self) -> dict:
        """获取草图数据和拉伸参数，用于生成 3D 模型。"""
        return {
            "nodes": list(self.canvas.nodes),
            "lines": list(self.canvas.lines),
            "plane": self.cmb_plane.currentIndex(),
            "bays": self.spn_bays.value(),
            "bay_len": self.spn_bay_len.value(),
            "base": "fixed" if self.cmb_base.currentIndex() == 0 else "pinned",
            "material": str(self.cmb_material.currentData() or
                            self.cmb_material.currentText()).strip()
                        if self.cmb_material.currentIndex() > 0 else "",
            "section": str(self.cmb_section.currentData() or
                           self.cmb_section.currentText()).strip()
                       if self.cmb_section.currentIndex() > 0 else "",
        }
