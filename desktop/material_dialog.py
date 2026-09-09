"""Abaqus 式材料定义对话框。

对应 Abaqus Property 模块 → Create Material：
- Mechanical → Elasticity → Elastic：E, nu
- General → Density：密度

美化版：统一标题栏、分组区域、表单布局。
"""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QVBoxLayout, QLineEdit)

from . import theme
from .dialog_styles import DialogHeader, DialogSection, FormRow, HintLabel, style_dialog


class MaterialDialog(QDialog):
    """材料定义对话框。"""

    def __init__(self, material: dict | None = None, parent=None,
                 units: str = "N-m-Pa"):
        super().__init__(parent)
        self.units = units
        mm = units == "N-mm-MPa"
        self.setWindowTitle("编辑材料" if material else "创建材料")
        style_dialog(self, 460, 480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(0)

        # 标题栏
        header = DialogHeader(
            "创建材料" if not material else "编辑材料",
            "定义各向同性线弹性材料的弹性模量、泊松比和密度"
        )
        layout.addWidget(header)

        # 内容区域
        content = QVBoxLayout()
        content.setContentsMargins(16, 12, 16, 0)
        content.setSpacing(12)

        # 基本信息分组
        basic_section = DialogSection("基本信息")
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("例如：Steel / Q355 / Concrete")
        self.txt_name.setMinimumHeight(28)
        basic_section.addLayout(FormRow("名称", self.txt_name, 90))
        content.addWidget(basic_section)

        # 弹性属性分组
        elastic_section = DialogSection("弹性属性")
        self.spn_E = QDoubleSpinBox()
        self.spn_E.setRange(1e-6, 1e15)
        self.spn_E.setValue(2.06e5 if mm else 2.06e11)
        self.spn_E.setDecimals(2)
        self.spn_E.setSuffix(" MPa" if mm else " Pa")
        self.spn_E.setMinimumHeight(28)
        elastic_section.addLayout(FormRow("弹性模量 E", self.spn_E, 90))

        self.spn_nu = QDoubleSpinBox()
        self.spn_nu.setRange(0, 0.5)
        self.spn_nu.setValue(0.3)
        self.spn_nu.setDecimals(4)
        self.spn_nu.setMinimumHeight(28)
        elastic_section.addLayout(FormRow("泊松比 ν", self.spn_nu, 90))
        elastic_section.addWidget(HintLabel("各向同性线弹性材料必须满足 ν ≤ 0.5"))
        content.addWidget(elastic_section)

        # 密度分组
        density_section = DialogSection("密度")
        self.spn_density = QDoubleSpinBox()
        self.spn_density.setRange(0, 1e6)
        self.spn_density.setDecimals(12 if mm else 1)
        self.spn_density.setSingleStep(1e-9 if mm else 10.0)
        self.spn_density.setValue(7.85e-9 if mm else 7850.0)
        self.spn_density.setSuffix(" t/mm³" if mm else " kg/m³")
        self.spn_density.setMinimumHeight(28)
        density_section.addLayout(FormRow("密度 ρ", self.spn_density, 90))
        content.addWidget(density_section)

        # 材料非线性分组
        plastic_section = DialogSection("材料非线性（可选）")
        self.chk_plastic = QCheckBox("启用双线性轴向塑性（弯曲仍为弹性）")
        self.chk_plastic.setStyleSheet(f"color: {theme.INK}; padding: 2px 0;")
        plastic_section.addWidget(self.chk_plastic)

        self.spn_yield = QDoubleSpinBox()
        self.spn_yield.setRange(1e-9, 1e15)
        self.spn_yield.setDecimals(2)
        self.spn_yield.setValue(355.0 if mm else 355e6)
        self.spn_yield.setSuffix(" MPa" if mm else " Pa")
        self.spn_yield.setMinimumHeight(28)
        self.spn_yield.setEnabled(False)
        plastic_section.addLayout(FormRow("屈服应力", self.spn_yield, 90))

        self.spn_hardening = QDoubleSpinBox()
        self.spn_hardening.setRange(0.0, 0.999)
        self.spn_hardening.setDecimals(4)
        self.spn_hardening.setSingleStep(0.01)
        self.spn_hardening.setValue(0.01)
        self.spn_hardening.setMinimumHeight(28)
        self.spn_hardening.setEnabled(False)
        plastic_section.addLayout(FormRow("屈服后切线 / E", self.spn_hardening, 90))

        self.chk_plastic.toggled.connect(self.spn_yield.setEnabled)
        self.chk_plastic.toggled.connect(self.spn_hardening.setEnabled)
        content.addWidget(plastic_section)

        layout.addLayout(content, 1)

        # 按钮
        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(16, 8, 16, 0)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btn_layout.addWidget(btns)
        layout.addLayout(btn_layout)

        if material:
            self.txt_name.setText(material.get("name", ""))
            self.spn_E.setValue(float(material.get("E", 2.06e5 if mm else 2.06e11)))
            self.spn_nu.setValue(float(material.get("nu", 0.3)))
            self.spn_density.setValue(float(material.get(
                "density", 7.85e-9 if mm else 7850.0)))
            enabled = material.get("yield_stress") is not None
            self.chk_plastic.setChecked(enabled)
            if enabled:
                self.spn_yield.setValue(float(material["yield_stress"]))
            self.spn_hardening.setValue(float(material.get("hardening_ratio", 0.01)))

    def get_material(self) -> dict:
        out = {
            "name": self.txt_name.text().strip() or "Material",
            "E": self.spn_E.value(),
            "nu": self.spn_nu.value(),
            "density": self.spn_density.value(),
        }
        if self.chk_plastic.isChecked():
            out["yield_stress"] = self.spn_yield.value()
            out["hardening_ratio"] = self.spn_hardening.value()
        return out
