"""Abaqus 式材料定义对话框。

对应 Abaqus Property 模块 → Create Material：
- Mechanical → Elasticity → Elastic：E, nu
- General → Density：密度

美化版：统一标题栏、分组区域、表单布局。
"""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFrame,
                               QLineEdit, QScrollArea, QVBoxLayout, QWidget)

from . import dialog_styles, theme
from .dialog_styles import DialogHeader, DialogSection, FormRow, HintLabel, style_dialog


class MaterialDialog(QDialog):
    """材料定义对话框。"""

    def __init__(self, material: dict | None = None, parent=None,
                 units: str = "N-m-Pa"):
        super().__init__(parent)
        self.units = units
        mm = units == "N-mm-MPa"
        self.setWindowTitle("编辑材料" if material else "创建材料")
        style_dialog(self, 470, 640)

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

        # 强度验算与温度应力用的参数（讲义 §3-9 三、六）。
        # 这两项原来只有 Agent 能填，在界面上建的材料**做不了强度验算、
        # 也算不了温度应力**——链路全通，一到校核就被拒绝。从界面走一遍
        # 才发现的。
        check_section = DialogSection("强度验算与温度（可选）")
        self.chk_allowable = QCheckBox("给出许用应力（强度验算需要）")
        self.chk_allowable.setStyleSheet(f"color: {theme.INK}; padding: 2px 0;")
        check_section.addWidget(self.chk_allowable)

        self.spn_allow_t = QDoubleSpinBox()
        self.spn_allow_t.setRange(1e-9, 1e15)
        self.spn_allow_t.setDecimals(2)
        self.spn_allow_t.setValue(215.0 if mm else 215e6)
        self.spn_allow_t.setSuffix(" MPa" if mm else " Pa")
        self.spn_allow_t.setMinimumHeight(28)
        self.spn_allow_t.setEnabled(False)
        check_section.addLayout(FormRow("许用拉应力", self.spn_allow_t, 90))

        self.chk_diff_compression = QCheckBox("抗压与抗拉不同（铸铁、砌体、木材）")
        self.chk_diff_compression.setStyleSheet(
            f"color: {theme.INK}; padding: 2px 0;")
        self.chk_diff_compression.setEnabled(False)
        check_section.addWidget(self.chk_diff_compression)

        self.spn_allow_c = QDoubleSpinBox()
        self.spn_allow_c.setRange(1e-9, 1e15)
        self.spn_allow_c.setDecimals(2)
        self.spn_allow_c.setValue(215.0 if mm else 215e6)
        self.spn_allow_c.setSuffix(" MPa" if mm else " Pa")
        self.spn_allow_c.setMinimumHeight(28)
        self.spn_allow_c.setEnabled(False)
        check_section.addLayout(FormRow("许用压应力", self.spn_allow_c, 90))
        check_section.addWidget(HintLabel(
            "不勾选「抗压与抗拉不同」时按钢材惯例取与抗拉相同。"
            "抗拉抗压相差数倍的材料必须分别给，否则受拉那侧会漏判。"))

        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.0, 1.0)
        self.spn_alpha.setDecimals(9)
        self.spn_alpha.setSingleStep(1e-6)
        self.spn_alpha.setValue(0.0)
        self.spn_alpha.setSuffix(" 1/℃")
        self.spn_alpha.setMinimumHeight(28)
        check_section.addLayout(FormRow("线膨胀系数 α", self.spn_alpha, 90))
        check_section.addWidget(HintLabel(
            "算温度应力才需要，钢约 1.2e-5。留 0 表示不考虑温度，"
            "此时施加温度变化会被明确拒绝，而不是算出「没有温度应力」。"))

        self.chk_allowable.toggled.connect(self.spn_allow_t.setEnabled)
        self.chk_allowable.toggled.connect(self.chk_diff_compression.setEnabled)
        self.chk_allowable.toggled.connect(
            lambda on: self.spn_allow_c.setEnabled(
                on and self.chk_diff_compression.isChecked()))
        self.chk_diff_compression.toggled.connect(
            lambda on: self.spn_allow_c.setEnabled(
                on and self.chk_allowable.isChecked()))
        content.addWidget(check_section)

        # 表单越加越长（塑性、许用应力、线膨胀系数都是后来补的），窗口再高
        # 也总有装不下的一天——装不下时 Qt 会把各组挤扁、文字叠在一起，
        # **看着像界面坏了**。放进滚动区，长度就不再是问题。
        holder = QWidget()
        holder.setLayout(content)
        scroll = QScrollArea(self)
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        holder.setStyleSheet("background: transparent;")
        layout.addWidget(scroll, 1)

        # 按钮
        btn_layout = QVBoxLayout()
        btn_layout.setContentsMargins(16, 8, 16, 0)
        btns = dialog_styles.button_box(self)
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
            allow_t = material.get("allow_tension")
            allow_c = material.get("allow_compression")
            self.chk_allowable.setChecked(allow_t is not None or allow_c is not None)
            if allow_t is not None:
                self.spn_allow_t.setValue(float(allow_t))
            if allow_c is not None:
                self.chk_diff_compression.setChecked(True)
                self.spn_allow_c.setValue(float(allow_c))
            self.spn_alpha.setValue(float(material.get("alpha", 0.0) or 0.0))

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
        if self.chk_allowable.isChecked():
            out["allow_tension"] = self.spn_allow_t.value()
            # 只在真的不同时才写压许用：写一个与拉相同的值看不出是"用户确认过"
            # 还是"默认带出来的"，而强度验算的输出要报出这个区别。
            if self.chk_diff_compression.isChecked():
                out["allow_compression"] = self.spn_allow_c.value()
        if self.spn_alpha.value() > 0:
            out["alpha"] = self.spn_alpha.value()
        return out
