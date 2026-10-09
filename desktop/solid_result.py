"""读取确定性实体结果文件及控制主视口的实体显示。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QLabel, QPushButton, QVBoxLayout, QWidget)


FIELDS = (("Mises_MPa", "Mises 应力", "MPa", "cell"),
          ("AbsPrincipal_MPa", "最大绝对主应力", "MPa", "cell"),
          ("DisplacementMagnitude_mm", "位移幅值", "mm", "point"))


def result_paths(summary: dict, directory: Path) -> list[Path]:
    """优先使用数值存档；结果目录搬移后仍按当前目录解析各档文件。"""
    paths = []
    for index, row in enumerate(summary.get("meshes") or []):
        native = directory / f"native_joint_{index}.npz"
        vtu = row.get("vtu")
        paths.append(native if native.is_file() or not vtu
                     else directory / Path(vtu).name)
    if not paths:
        raise ValueError("未找到实体网格档位，请重新完成节点实体分析。")
    return paths


def read_result(path: str | Path):
    """读取真实 C3D10 网格、单元应力和节点位移，不读取 pickle。"""
    import pyvista as pv

    path = Path(path).resolve()
    summary_path = path if path.suffix.lower() == ".json" else path.parent / "summary.json"
    summary = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "solid-joint-analysis/native-v1":
            raise ValueError("该文件不是自研 C3D10 实体结果，请选择对应的 VTU 或 NPZ 文件。")
    paths = result_paths(summary, path.parent) if summary else [path]
    if path.suffix.lower() == ".json":
        path = paths[-1]
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            nodes = np.asarray(data["nodes"], dtype=float)
            elements = np.asarray(data["elements"])
            if (nodes.ndim != 2 or nodes.shape[1] != 3 or not len(nodes)
                    or not np.isfinite(nodes).all()):
                raise ValueError("实体节点坐标无效，请重新生成结果。")
            if (elements.ndim != 2 or elements.shape[1] != 10 or not len(elements)
                    or not np.issubdtype(elements.dtype, np.integer)
                    or elements.min() < 0 or elements.max() >= len(nodes)):
                raise ValueError("C3D10 单元连接无效，请重新生成结果。")
            grid = pv.UnstructuredGrid({pv.CellType.QUADRATIC_TETRA: elements}, nodes)
            grid.cell_data["Mises_MPa"] = data["element_mises"]
            grid.cell_data["AbsPrincipal_MPa"] = data["element_abs_principal"]
            grid.point_data["Displacement_mm"] = data["displacement"]
    elif suffix == ".vtu":
        grid = pv.read(path)
    else:
        raise ValueError("请选择实体分析的 summary.json、VTU 或 NPZ 结果文件。")
    if (not isinstance(grid, pv.UnstructuredGrid) or not grid.n_cells or not grid.n_points
            or not np.isfinite(grid.points).all()
            or not np.all(grid.celltypes == pv.CellType.QUADRATIC_TETRA)):
        raise ValueError("结果没有有效的 C3D10 实体网格，请重新生成结果。")
    for key in ("Mises_MPa", "AbsPrincipal_MPa"):
        values = np.asarray(grid.cell_data[key])
        if (values.shape != (grid.n_cells,) or not np.isfinite(values).all()
                or np.any(values < 0)):
            raise ValueError("实体应力结果无效，请重新生成结果。")
    displacement = np.asarray(grid.point_data["Displacement_mm"])
    if displacement.shape != (grid.n_points, 3) or not np.isfinite(displacement).all():
        raise ValueError("实体位移结果无效，请重新生成结果。")
    grid.point_data["DisplacementMagnitude_mm"] = np.linalg.norm(displacement, axis=1)
    # 直接打开 VTU 时选中同档数值存档；尚无摘要的结果仅提供当前文件。
    index = next((i for i, p in enumerate(paths) if p.stem == path.stem), len(paths) - 1)
    caption = (f"节点 {summary.get('node_id')} · 工况 {summary.get('case')}"
               if summary else path.stem)
    return grid, summary, paths, index, caption


class SolidResultPanel(QWidget):
    """只读实体结果控制；各档文件仍由原确定性分析提供。"""

    changed = Signal()
    file_requested = Signal(str)
    open_requested = Signal()
    model_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filling = False
        box = QVBoxLayout(self)
        self.caption = QLabel("尚未加载实体结果")
        self.caption.setWordWrap(True)
        box.addWidget(self.caption)
        form = QFormLayout()
        self.levels = QComboBox()
        self.levels.currentIndexChanged.connect(self._request_level)
        form.addRow("网格档位", self.levels)
        self.field = QComboBox()
        for key, label, unit, _association in FIELDS:
            self.field.addItem(f"{label} ({unit})", key)
        self.field.currentIndexChanged.connect(self._changed)
        form.addRow("结果量", self.field)
        self.range = QComboBox()
        self.range.addItem("P99 色标", 99)
        self.range.addItem("完整范围", None)
        self.range.setToolTip("P99 仅截断着色范围；真实峰值始终按全部结果显示。")
        self.range.currentIndexChanged.connect(self._changed)
        form.addRow("色标范围", self.range)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0, 1000)
        self.scale.setDecimals(2)
        self.scale.setToolTip("0 显示原始几何；1 显示真实变形，其他值放大节点位移。")
        self.scale.valueChanged.connect(self._changed)
        form.addRow("变形倍数", self.scale)
        self.edges = QCheckBox("显示单元边线")
        self.edges.toggled.connect(self._changed)
        form.addRow(self.edges)
        box.addLayout(form)
        self.source = QLabel()
        self.source.setWordWrap(True)
        self.source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.source)
        note = QLabel("只读实体结果。旋转：拖动；缩放：滚轮。\n坐标和位移单位：mm；应力单位：MPa。")
        note.setWordWrap(True)
        box.addWidget(note)
        for text, signal in (("打开实体结果", self.open_requested),
                             ("返回整体模型", self.model_requested)):
            button = QPushButton(text)
            button.clicked.connect(signal.emit)
            box.addWidget(button)
        box.addStretch(1)

    def bind(self, paths, index, caption, *, saved=False):
        self._filling = True
        try:
            self.caption.setText(caption + ("\n已保存结果，独立于当前整体模型" if saved else ""))
            self.levels.clear()
            for i, path in enumerate(paths):
                self.levels.addItem(f"第 {i + 1} 档", str(path))
            self.levels.setCurrentIndex(index)
            self.source.setText(f"结果文件：{paths[index].name}")
            self.source.setToolTip(str(paths[index]))
        finally:
            self._filling = False

    def options(self):
        return dict(field=self.field.currentData(), percentile=self.range.currentData(),
                    scale=self.scale.value(), show_edges=self.edges.isChecked())

    def _changed(self, *_args):
        if not self._filling:
            self.changed.emit()

    def _request_level(self, _index):
        if not self._filling and self.levels.currentData():
            self.file_requested.emit(self.levels.currentData())
