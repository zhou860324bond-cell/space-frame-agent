"""节点实体收敛任务配置。明确代价和证据等级后，用户才启动后台作业。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QGridLayout, QLabel, QVBoxLayout)

from convergence import solid_mesh_plan
from .dialog_styles import (DialogHeader, DialogSection, FormRow, HintLabel,
                            button_box, style_dialog)


class SolidConvergenceDialog(QDialog):
    def __init__(self, node_id: int, case: str, reference_mm: float,
                 has_abaqus: bool = False, parent=None):
        super().__init__(parent)
        self.reference_mm = float(reference_mm)
        self.setWindowTitle("节点实体任务")
        style_dialog(self, 620, 430)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(10)
        layout.addWidget(DialogHeader(
            "节点实体任务",
            "先选择证据等级，再启动后台作业；取消不会生成网格或占用求解器。"))

        setup = DialogSection("分析对象", self)
        setup.addLayout(FormRow("节点", QLabel(str(node_id), self)))
        setup.addLayout(FormRow("工况", QLabel(str(case), self)))
        setup.addLayout(FormRow(
            "参考尺寸", QLabel(f"{self.reference_mm:g} mm（取最薄壁厚或实心圆 D/10）", self)))

        self.backend_box = QComboBox(self)
        self.backend_box.addItem("自研 C3D10 + Gmsh", "native")
        self.backend_box.addItem(
            "Abaqus 6.14" if has_abaqus else "Abaqus 6.14（当前未检测到）",
            "abaqus")
        self.backend_box.addItem(
            "双后端对标" if has_abaqus else "双后端对标（需要 Abaqus）",
            "compare")
        if not has_abaqus:
            for index in (1, 2):
                item = self.backend_box.model().item(index)
                if item is not None:
                    item.setEnabled(False)
        setup.addLayout(FormRow("求解后端", self.backend_box))

        self.mode_box = QComboBox(self)
        self.mode_box.addItem("三档收敛（推荐）", "convergence")
        self.mode_box.addItem("单档预览（更快，不发布 Kt）", "quick")
        setup.addLayout(FormRow("证据等级", self.mode_box))
        layout.addWidget(setup)

        mesh = DialogSection("将要运行的网格", self)
        self.mesh_grid = QGridLayout()
        self.mesh_grid.setHorizontalSpacing(18)
        for column, title in enumerate(
                ("后端", "档位", "全局尺寸", "局部控制尺寸", "用途")):
            label = QLabel(title, self)
            label.setStyleSheet("font-weight: 600;")
            self.mesh_grid.addWidget(label, 0, column)
        mesh.addLayout(self.mesh_grid)
        self.note = HintLabel("", self)
        mesh.addWidget(self.note)
        layout.addWidget(mesh)

        self.buttons = button_box(self, ok="开始后台计算", cancel="取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.backend_box.currentIndexChanged.connect(self._refresh)
        self.mode_box.currentIndexChanged.connect(self._refresh)
        self._refresh()

    def backend(self) -> str:
        return str(self.backend_box.currentData())

    def mode(self) -> str:
        return str(self.mode_box.currentData())

    def mesh_sizes(self) -> list[float]:
        if self.backend() == "compare":
            raise ValueError("双后端任务包含两套网格方案，请读取 mesh_plans")
        return solid_mesh_plan(
            self.reference_mm, backend=self.backend(), mode=self.mode())

    def mesh_plans(self) -> dict[str, list[float]]:
        if self.backend() == "compare":
            return {
                "native": solid_mesh_plan(
                    self.reference_mm, backend="native", mode="convergence"),
                "abaqus": solid_mesh_plan(
                    self.reference_mm, backend="abaqus", mode="convergence"),
            }
        return {self.backend(): self.mesh_sizes()}

    def _refresh(self, *_args) -> None:
        if self.backend() in {"abaqus", "compare"} and self.mode() == "quick":
            self.mode_box.setCurrentIndex(
                max(0, self.mode_box.findData("convergence")))
        quick_index = self.mode_box.findData("quick")
        if quick_index >= 0:
            item = self.mode_box.model().item(quick_index)
            if item is not None:
                item.setEnabled(self.backend() == "native")

        while self.mesh_grid.count() > 5:
            item = self.mesh_grid.takeAt(5)
            if item.widget() is not None:
                item.widget().deleteLater()
        rows = []
        for backend, sizes in self.mesh_plans().items():
            for level, size in enumerate(sizes, start=1):
                rows.append((backend, level, size, len(sizes)))
        for row, (backend, level, size, level_count) in enumerate(rows, start=1):
            if backend == "native":
                local = max(self.reference_mm, 0.4 * size)
            else:
                local = size
            purpose = ("建立趋势" if level == 1 else
                       "检查变化" if level < level_count else
                       "控制结论")
            backend_label = "native" if backend == "native" else "Abaqus"
            values = (backend_label, f"第 {level} 档", f"{size:g} mm",
                      f"{local:g} mm", purpose)
            for column, value in enumerate(values):
                label = QLabel(value, self)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.mesh_grid.addWidget(label, row, column)
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if self.backend() == "compare":
            ok_button.setText("开始双后端对标")
            self.note.setText(
                "依次运行 native 与 Abaqus，共 6 个网格作业。已完成的后端会复用；"
                "取消将在当前后端安全结束后阻止下一阶段启动。")
        elif self.mode() == "quick":
            ok_button.setText("开始后台计算")
            self.note.setText(
                "单档只用于检查网格、载荷传递和云图分布；没有趋势证据，不发布正式 Kt。")
        else:
            ok_button.setText("开始后台计算")
            self.note.setText(
                "三档从粗到细运行，最粗/最细局部尺寸达到 2 倍；"
                "只有热点外推进入稳定区才发布 Kt。作业可能持续数分钟。")
