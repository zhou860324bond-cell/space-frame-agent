"""Abaqus 式梁截面对话框。

对应 Abaqus Property 模块 → Create Section → Beam：
- 截面类型：I / Box / Pipe / Circular / Rectangular / Channel / Angle / T
- 输入尺寸，自动计算 A, Iy, Iz, J
"""

from __future__ import annotations

import math

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
                               QMessageBox, QStackedWidget, QVBoxLayout, QWidget)

from . import theme

# 截面类型定义：(显示名, 参数字典)
SECTION_TYPES = {
    "I 型钢": {"h": "高度 h", "b": "翼缘宽 b", "tw": "腹板厚 tw", "tf": "翼缘厚 tf"},
    "矩形管": {"b": "宽 b", "h": "高 h", "t": "壁厚 t"},
    "圆管": {"d": "外径 d", "t": "壁厚 t"},
    "矩形": {"b": "宽 b", "h": "高 h"},
    "圆钢": {"d": "直径 d"},
    "槽钢": {"h": "高度 h", "b": "腿宽 b", "tw": "腹板厚 tw", "tf": "腿厚 tf"},
    "等边角钢": {"b": "边宽 b", "t": "边厚 t"},
    "T 型钢": {"h": "高度 h", "b": "翼缘宽 b", "tw": "腹板厚 tw", "tf": "翼缘厚 tf"},
}

# 可直接作为起点修改的合理尺寸（米）；切到毫米制时只换显示数值。
DEFAULT_DIMENSIONS = {
    "I 型钢": {"h": 0.30, "b": 0.15, "tw": 0.008, "tf": 0.012},
    "矩形管": {"b": 0.20, "h": 0.30, "t": 0.008},
    "圆管": {"d": 0.168, "t": 0.008},
    "矩形": {"b": 0.20, "h": 0.30},
    "圆钢": {"d": 0.05},
    "槽钢": {"h": 0.25, "b": 0.09, "tw": 0.008, "tf": 0.012},
    "等边角钢": {"b": 0.075, "t": 0.008},
    "T 型钢": {"h": 0.20, "b": 0.15, "tw": 0.008, "tf": 0.012},
}


def calc_section(section_type: str, params: dict) -> dict:
    """根据截面类型和尺寸计算 A, Iy, Iz, J。

    y 轴沿截面高度方向，z 轴沿截面宽度方向。
    """
    if section_type == "I 型钢":
        h, b, tw, tf = params["h"], params["b"], params["tw"], params["tf"]
        A = b * tf * 2 + (h - 2 * tf) * tw
        Iy = (b * h ** 3 - (b - tw) * (h - 2 * tf) ** 3) / 12
        Iz = (2 * tf * b ** 3 + (h - 2 * tf) * tw ** 3) / 12
        J = (2 * b * tf ** 3 + (h - 2 * tf) * tw ** 3) / 3
    elif section_type == "矩形管":
        b, h, t = params["b"], params["h"], params["t"]
        A = b * h - (b - 2 * t) * (h - 2 * t)
        Iy = (b * h ** 3 - (b - 2 * t) * (h - 2 * t) ** 3) / 12
        Iz = (h * b ** 3 - (h - 2 * t) * (b - 2 * t) ** 3) / 12
        # 薄壁矩形管扭转常数近似
        J = 2 * t * (b - t) * (h - t) ** 2 / (b + h - 2 * t)
    elif section_type == "圆管":
        d, t = params["d"], params["t"]
        r = d / 2
        ri = r - t
        A = math.pi * (r ** 2 - ri ** 2)
        Iy = Iz = math.pi * (r ** 4 - ri ** 4) / 4
        J = math.pi * (r ** 4 - ri ** 4) / 2
    elif section_type == "矩形":
        b, h = params["b"], params["h"]
        A = b * h
        Iy = b * h ** 3 / 12
        Iz = h * b ** 3 / 12
        long_side, short_side = max(b, h), min(b, h)
        J = long_side * short_side ** 3 * (
            1 / 3 - 0.21 * short_side / long_side
            * (1 - short_side ** 4 / (12 * long_side ** 4)))
    elif section_type == "圆钢":
        d = params["d"]
        r = d / 2
        A = math.pi * r ** 2
        Iy = Iz = math.pi * r ** 4 / 4
        J = math.pi * r ** 4 / 2
    elif section_type == "槽钢":
        h, b, tw, tf = params["h"], params["b"], params["tw"], params["tf"]
        A = 2 * b * tf + (h - 2 * tf) * tw
        Iy = (b * h ** 3 - (b - tw) * (h - 2 * tf) ** 3) / 12
        # 槽钢 Iz 需要考虑形心偏移，这里近似
        Iz = (2 * tf * b ** 3 + (h - 2 * tf) * tw ** 3) / 12
        J = (2 * b * tf ** 3 + (h - 2 * tf) * tw ** 3) / 3
    elif section_type == "等边角钢":
        b, t = params["b"], params["t"]
        A = 2 * b * t - t ** 2
        Iy = Iz = (b ** 3 * t + (b - t) ** 3 * t) / 12 - A * (b / (2 * math.sqrt(2))) ** 2
        # 近似
        Iy = Iz = (2 * t * b ** 3 - (b - t) ** 3 * t) / 12
        J = (2 * b - t) * t ** 3 / 3
    elif section_type == "T 型钢":
        h, b, tw, tf = params["h"], params["b"], params["tw"], params["tf"]
        A = b * tf + (h - tf) * tw
        Iy = (b * h ** 3 - (b - tw) * (h - tf) ** 3) / 12
        Iz = (tf * b ** 3 + (h - tf) * tw ** 3) / 12
        J = (b * tf ** 3 + (h - tf) * tw ** 3) / 3
    else:
        A, Iy, Iz, J = 0.01, 1e-5, 1e-5, 1e-7

    # 不在模型层舍入。小截面的 m⁴ 数值本来就可能低于 1e-8，四舍五入
    # 会直接变成零，到求解阶段才报“截面惯性矩必须大于 0”。
    return {"A": float(A), "Iy": float(Iy), "Iz": float(Iz), "J": float(J)}


class SectionDialog(QDialog):
    """梁截面对话框。"""

    def __init__(self, section: dict | None = None, parent=None,
                 units: str = "N-m-Pa"):
        super().__init__(parent)
        self.units = units
        self.length_unit = "mm" if units == "N-mm-MPa" else "m"
        size_scale = 1000.0 if units == "N-mm-MPa" else 1.0
        self.setWindowTitle("编辑截面" if section else "创建截面")
        self.resize(400, 420)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("例如：I-200 / RHS-100x50x4 / CHS-114x4")
        form.addRow("名称", self.txt_name)

        self.cmb_type = QComboBox()
        self.cmb_type.addItems(list(SECTION_TYPES.keys()))
        self.cmb_type.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("截面类型", self.cmb_type)

        layout.addLayout(form)

        # 参数输入区（堆叠，根据类型切换）
        self.stack = QStackedWidget()
        self.param_widgets: dict[str, QWidget] = {}
        self.param_spins: dict[str, dict[str, QDoubleSpinBox]] = {}
        for stype, params in SECTION_TYPES.items():
            w = QWidget()
            f = QFormLayout(w)
            spins = {}
            for key, label in params.items():
                sp = QDoubleSpinBox()
                sp.setRange(0.001 * size_scale, 10 * size_scale)
                sp.setValue(DEFAULT_DIMENSIONS[stype][key] * size_scale)
                sp.setDecimals(3)
                sp.setSuffix(f" {self.length_unit}")
                sp.valueChanged.connect(self._update_preview)
                f.addRow(label, sp)
                spins[key] = sp
            self.param_widgets[stype] = w
            self.param_spins[stype] = spins
            self.stack.addWidget(w)
        layout.addWidget(self.stack)

        self.chk_shear = QCheckBox("考虑 Timoshenko 剪切变形（Ay=Az≈5A/6）")
        self.chk_shear.setChecked(True)
        self.chk_shear.setToolTip(
            "短梁/深梁建议启用；旧截面若没有 Ay/Az，内核自动保持 Euler-Bernoulli")
        self.chk_shear.toggled.connect(self._update_preview)
        layout.addWidget(self.chk_shear)

        # 预览区
        self.lbl_preview = QLabel()
        self.lbl_preview.setStyleSheet(
            f"background:{theme.PANEL_ALT}; padding:8px; border:1px solid {theme.BORDER};"
            f"color:{theme.INK}; font-family:monospace; font-size:8pt;")
        self.lbl_preview.setMinimumHeight(80)
        layout.addWidget(self.lbl_preview)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        if section:
            self.txt_name.setText(section.get("name", ""))
            # 无法反推截面类型，默认 I 型钢
        self._on_type_changed(0)
        self._update_preview()

    def _on_type_changed(self, _index):
        stype = self.cmb_type.currentText()
        self.stack.setCurrentWidget(self.param_widgets[stype])
        self._update_preview()

    def _update_preview(self):
        stype = self.cmb_type.currentText()
        spins = self.param_spins[stype]
        params = {k: sp.value() for k, sp in spins.items()}
        props = calc_section(stype, params)
        shear = (f"\n  Ay = Az ≈ {5 * props['A'] / 6:.6e} {self.length_unit}²"
                 if self.chk_shear.isChecked() else "\n  剪切变形：关闭")
        self.lbl_preview.setText(
            f"截面属性（自动计算）：\n"
            f"  A  = {props['A']:.6e} {self.length_unit}²\n"
            f"  Iy = {props['Iy']:.6e} {self.length_unit}⁴\n"
            f"  Iz = {props['Iz']:.6e} {self.length_unit}⁴\n"
            f"  J  = {props['J']:.6e} {self.length_unit}⁴" + shear
        )

    def accept(self):
        """在写入模型前拦住几何上不可能的薄壁截面。"""
        stype = self.cmb_type.currentText()
        p = {key: spin.value() for key, spin in self.param_spins[stype].items()}
        error = None
        if stype == "I 型钢" and (2 * p["tf"] >= p["h"] or p["tw"] >= p["b"]):
            error = "I 型钢需满足 2×tf < h 且 tw < b"
        elif stype == "矩形管" and (2 * p["t"] >= p["b"] or 2 * p["t"] >= p["h"]):
            error = "矩形管需满足 2×t < b 且 2×t < h"
        elif stype == "圆管" and 2 * p["t"] >= p["d"]:
            error = "圆管需满足 2×t < d"
        elif stype == "槽钢" and (2 * p["tf"] >= p["h"] or p["tw"] >= p["b"]):
            error = "槽钢需满足 2×tf < h 且 tw < b"
        elif stype == "等边角钢" and p["t"] >= p["b"]:
            error = "等边角钢需满足 t < b"
        elif stype == "T 型钢" and (p["tf"] >= p["h"] or p["tw"] >= p["b"]):
            error = "T 型钢需满足 tf < h 且 tw < b"
        props = calc_section(stype, p)
        if error is None and (not all(math.isfinite(v) and v > 0 for v in props.values())):
            error = "当前尺寸算出的截面属性无效，请检查各项尺寸"
        if error:
            QMessageBox.warning(self, "截面尺寸不合理", error)
            return
        super().accept()

    def get_section(self) -> dict:
        stype = self.cmb_type.currentText()
        spins = self.param_spins[stype]
        params = {k: sp.value() for k, sp in spins.items()}
        props = calc_section(stype, params)
        result = {
            "name": self.txt_name.text().strip() or stype,
            "type": stype,
            "params": params,
            **props,
        }
        if self.chk_shear.isChecked():
            result["Ay"] = 5.0 * props["A"] / 6.0
            result["Az"] = 5.0 * props["A"] / 6.0
        return result
