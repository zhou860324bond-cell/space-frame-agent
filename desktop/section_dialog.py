"""Abaqus 式梁截面对话框。

对应 Abaqus Property 模块 → Create Section → Beam：
- 截面类型：I / Box / Pipe / Circular / Rectangular / Channel / Angle / T
- 输入尺寸，自动计算 A, Iy, Iz, J
"""

from __future__ import annotations

import math

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, 
                               QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
                               QMessageBox, QStackedWidget, QVBoxLayout, QWidget)

import section_geometry

from . import dialog_styles, theme

# 截面类型定义：(显示名, 参数字典)
SECTION_TYPES = {
    "I 型钢": {"h": "高度 h", "b": "翼缘宽 b", "tw": "腹板厚 tw", "tf": "翼缘厚 tf",
              "r": "根部圆角 r"},
    "矩形管": {"b": "宽 b", "h": "高 h", "t": "壁厚 t"},
    "圆管": {"d": "外径 d", "t": "壁厚 t"},
    "矩形": {"b": "宽 b", "h": "高 h"},
    "圆钢": {"d": "直径 d"},
    "槽钢": {"h": "高度 h", "b": "腿宽 b", "tw": "腹板厚 tw", "tf": "腿厚 tf",
             "r": "根部圆角 r"},
    "等边角钢": {"b": "边宽 b", "t": "边厚 t", "r": "根部圆角 r"},
    "T 型钢": {"h": "高度 h", "b": "翼缘宽 b", "tw": "腹板厚 tw", "tf": "翼缘厚 tf",
              "r": "根部圆角 r"},
}

# 热轧型钢腹板与翼缘（或两肢）交接处的根部圆角。0 表示直角轮廓；按型钢表
# 填面积、惯性矩时必须填表中的 r，否则 A 比表值小 2%~4%。
ROOT_RADIUS_KEY = "r"

# 可直接作为起点修改的合理尺寸（米）；切到毫米制时只换显示数值。
DEFAULT_DIMENSIONS = {
    "I 型钢": {"h": 0.30, "b": 0.15, "tw": 0.008, "tf": 0.012, "r": 0.0},
    "矩形管": {"b": 0.20, "h": 0.30, "t": 0.008},
    "圆管": {"d": 0.168, "t": 0.008},
    "矩形": {"b": 0.20, "h": 0.30},
    "圆钢": {"d": 0.05},
    "槽钢": {"h": 0.25, "b": 0.09, "tw": 0.008, "tf": 0.012, "r": 0.0},
    "等边角钢": {"b": 0.075, "t": 0.008, "r": 0.0},
    "T 型钢": {"h": 0.20, "b": 0.15, "tw": 0.008, "tf": 0.012, "r": 0.0},
}


# 双轴对称的截面：形心在几何中心，极端纤维距离就是半高、半宽，**是精确值**。
# 槽钢、等边角钢、T 型钢不在此列：它们形心偏置，槽钢的剪心不在形心，角钢的
# 主轴斜 45°，这些都超出了梁单元按形心主轴计算应力的前提。给它们配一个 c，
# 算出来的应力照样是错的，只是错得看不出来。所以**这三种干脆不给**，
# 强度验算会明确拒绝——那比给一个来路不明的应力安全。
_SYMMETRIC_FIBRES = {
    "I 型钢": lambda p: (p["h"] / 2.0, p["b"] / 2.0, False),
    "矩形管": lambda p: (p["h"] / 2.0, p["b"] / 2.0, False),
    "矩形": lambda p: (p["h"] / 2.0, p["b"] / 2.0, False),
    "圆管": lambda p: (p["d"] / 2.0, p["d"] / 2.0, True),
    "圆钢": lambda p: (p["d"] / 2.0, p["d"] / 2.0, True),
}

OFFSET_CENTROID_SHAPES = ("槽钢", "等边角钢", "T 型钢")


def calc_section(section_type: str, params: dict) -> dict:
    """根据截面类型和尺寸计算 A, Iy, Iz, J，以及极端纤维距离 cy/cz。

    轴的约定与内核（``src/sections.py``）一致：局部 y 沿截面高度 h，z 沿宽度 b；
    ``Iz = ∫y²dA`` 绕 z 轴，是高度方向受弯的**强轴**，``Iy = ∫z²dA`` 是弱轴。
    以前这里把 b·h³/12 写进了 Iy，水平工字钢梁的竖向挠度因此大了 Iz/Iy
    倍（默认 I 型钢约 13 倍），而且不报错。惯性矩一律对形心取。

    ``cy``/``cz`` 是强度验算算弯曲应力必需的（σ = N/A ± M·c/I）。少了它们，
    在对话框里建的截面**做不了强度验算**——这是从界面走一遍才发现的：
    工具链全通，一到校核就被拒绝。
    """
    params = dict(params)
    radius = float(params.pop(ROOT_RADIUS_KEY, 0.0) or 0.0)
    if section_type == "I 型钢":
        h, b, tw, tf = params["h"], params["b"], params["tw"], params["tf"]
        A = b * tf * 2 + (h - 2 * tf) * tw
        Iz = (b * h ** 3 - (b - tw) * (h - 2 * tf) ** 3) / 12
        Iy = (2 * tf * b ** 3 + (h - 2 * tf) * tw ** 3) / 12
        J = (2 * b * tf ** 3 + (h - 2 * tf) * tw ** 3) / 3
    elif section_type == "矩形管":
        b, h, t = params["b"], params["h"], params["t"]
        A = b * h - (b - 2 * t) * (h - 2 * t)
        Iz = (b * h ** 3 - (b - 2 * t) * (h - 2 * t) ** 3) / 12
        Iy = (h * b ** 3 - (h - 2 * t) * (b - 2 * t) ** 3) / 12
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
        Iz = b * h ** 3 / 12
        Iy = h * b ** 3 / 12
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
        hw = h - 2 * tf
        A = 2 * b * tf + hw * tw
        # 关于 y 轴（腹板所在平面）对称，强轴不受形心偏移影响
        Iz = (b * h ** 3 - (b - tw) * hw ** 3) / 12
        # 弱轴要对形心取：形心从腹板外侧偏向翼缘一侧 z̄。原来直接对腹板外侧
        # 求和，默认尺寸下弱轴惯性矩只有真实值的一半
        z_bar = (2 * b * tf * b / 2 + hw * tw * tw / 2) / A
        Iy = (2 * (tf * b ** 3 / 12 + b * tf * (b / 2 - z_bar) ** 2)
              + hw * tw ** 3 / 12 + hw * tw * (tw / 2 - z_bar) ** 2)
        J = (2 * b * tf ** 3 + hw * tw ** 3) / 3
    elif section_type == "等边角钢":
        b, t = params["b"], params["t"]
        A = 2 * b * t - t ** 2
        # 对平行于两肢的形心轴取，即型钢表里的 Ix = Iy。两肢沿局部 y、z，
        # 这两根轴不是主轴（惯性积不为零，主轴斜 45°），内核只认主轴，
        # 这里忽略惯性积——仍是近似，所以本对话框不给角钢 cy/cz。
        c = (b * t * b / 2 + (b - t) * t * t / 2) / A   # 形心到肢背的距离
        Iy = Iz = (t * b ** 3 / 12 + b * t * (b / 2 - c) ** 2
                   + (b - t) * t ** 3 / 12 + (b - t) * t * (t / 2 - c) ** 2)
        J = (2 * b - t) * t ** 3 / 3
    elif section_type == "T 型钢":
        h, b, tw, tf = params["h"], params["b"], params["tw"], params["tf"]
        hw = h - tf
        A = b * tf + hw * tw
        # 强轴要对形心取：形心从腹板底端向翼缘偏到 ȳ。原来对翼缘外侧整块
        # 外框相减，默认尺寸下强轴惯性矩大了七成
        y_bar = (hw * tw * hw / 2 + b * tf * (hw + tf / 2)) / A
        Iz = (tw * hw ** 3 / 12 + hw * tw * (hw / 2 - y_bar) ** 2
              + b * tf ** 3 / 12 + b * tf * (hw + tf / 2 - y_bar) ** 2)
        # 关于 y 轴（腹板中面）对称，弱轴不受形心偏移影响
        Iy = (tf * b ** 3 + hw * tw ** 3) / 12
        J = (b * tf ** 3 + hw * tw ** 3) / 3
    else:
        A, Iy, Iz, J = 0.01, 1e-5, 1e-5, 1e-7

    # 不在模型层舍入。小截面的 m⁴ 数值本来就可能低于 1e-8，四舍五入
    # 会直接变成零，到求解阶段才报“截面惯性矩必须大于 0”。
    kinds = {"I 型钢": "i", "矩形管": "box", "圆管": "tube", "矩形": "rectangle",
             "圆钢": "circle", "槽钢": "channel", "等边角钢": "angle", "T 型钢": "tee"}
    geometry = None
    if section_type in kinds:
        geometry = {"type": kinds[section_type], "dimensions": dict(params)}
    if radius > 0.0 and geometry is not None:
        # 带根部圆角时 A、Iy、Iz 按真实轮廓精确计算（与节点实体用的是同一轮廓），
        # 上面的直角公式只在 r=0 时成立。J 仍按薄壁公式，未计圆角。
        geometry["dimensions"][ROOT_RADIUS_KEY] = radius
        exact = section_geometry.section_properties(geometry)
        A, Iy, Iz = exact["A"], exact["Iy"], exact["Iz"]
    out = {"A": float(A), "Iy": float(Iy), "Iz": float(Iz), "J": float(J)}
    if geometry is not None:
        out["geometry"] = geometry
    fibres = _SYMMETRIC_FIBRES.get(section_type)
    if fibres is not None:
        cy, cz, circular = fibres(params)
        out.update({"cy": float(cy), "cz": float(cz)})
        if circular:
            out["circular"] = True
    return out


class SectionDialog(QDialog):
    """梁截面对话框。"""

    def __init__(self, section: dict | None = None, parent=None,
                 units: str = "N-m-Pa"):
        super().__init__(parent)
        self.units = units
        self.length_unit = "mm" if units == "N-mm-MPa" else "m"
        size_scale = 1000.0 if units == "N-mm-MPa" else 1.0
        self.setWindowTitle("编辑截面" if section else "创建截面")
        # 预览块的行数随截面类型变（偏心截面要多说两句为什么不给 cy/cz），
        # 写死高度会把最后一行切掉——那一行恰恰是解释原因的。
        self.resize(440, 470)

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
                if key == ROOT_RADIUS_KEY:
                    sp.setRange(0.0, 1 * size_scale)
                    sp.setSpecialValueText("0（直角）")
                    sp.setToolTip("热轧型钢腹板与翼缘交接处的圆角半径，取型钢表中的 r；"
                                  "按型钢表填截面时必须填，否则面积比表值小，节点实体会拒绝。")
                # 先定小数位再赋值：默认两位小数时米制 tw=0.008 会被舍成 0.01。
                sp.setDecimals(3)
                sp.setValue(DEFAULT_DIMENSIONS[stype][key] * size_scale)
                sp.setSuffix(f" {self.length_unit}")
                sp.valueChanged.connect(self._update_preview)
                f.addRow(label, sp)
                spins[key] = sp
            self.param_widgets[stype] = w
            self.param_spins[stype] = spins
            self.stack.addWidget(w)
        layout.addWidget(self.stack)

        self.chk_shear = QCheckBox("考虑 Timoshenko 剪切变形（剪切面积按截面形状计算）")
        self.chk_shear.setChecked(True)
        self.chk_shear.setToolTip(
            "短梁/深梁建议启用；Ay、Az 按截面形状取 Cowper 剪切系数或薄壁剪力流公式，"
            "工字钢强轴约等于腹板面积。旧截面若没有 Ay/Az，内核自动保持 Euler-Bernoulli")
        self.chk_shear.toggled.connect(self._update_preview)
        layout.addWidget(self.chk_shear)

        # 预览区
        self.lbl_preview = QLabel()
        self.lbl_preview.setStyleSheet(
            f"background:{theme.PANEL_ALT}; padding:8px; border:1px solid {theme.BORDER};"
            f"color:{theme.INK}; font-family:monospace; font-size:8pt;")
        self.lbl_preview.setMinimumHeight(80)
        self.lbl_preview.setWordWrap(True)
        layout.addWidget(self.lbl_preview)

        btns = dialog_styles.button_box(self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        if section:
            self.txt_name.setText(section.get("name", ""))
            geometry = section.get("geometry", {})
            types = {"i":"I 型钢", "box":"矩形管", "tube":"圆管", "rectangle":"矩形",
                     "circle":"圆钢", "channel":"槽钢", "angle":"等边角钢", "tee":"T 型钢"}
            if geometry.get("type") in types:
                stype = types[geometry["type"]]
                self.cmb_type.setCurrentText(stype)
                for key, value in geometry.get("dimensions", {}).items():
                    if key in self.param_spins[stype]:
                        self.param_spins[stype][key].setValue(float(value))
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
        try:
            props = calc_section(stype, params)
        except ValueError as exc:
            # 输入过程中尺寸暂时不成立（如 r 大于翼缘外伸）是常态，预览说明原因即可。
            self.lbl_preview.setText(f"当前尺寸无法构成截面：{exc}")
            return
        if self.chk_shear.isChecked():
            areas = section_geometry.shear_areas(props["geometry"])
            shear = (f"\n  Ay = {areas['Ay']:.6e} {self.length_unit}²"
                     f"　Az = {areas['Az']:.6e} {self.length_unit}²"
                     f"\n  剪切面积：{areas['formula']}")
        else:
            shear = "\n  剪切变形：关闭"
        # 极端纤维距离决定这个截面能不能做强度验算。有就报出来，
        # 没有就**当场说清楚为什么**——不要等用户点了「强度验算」才被拒绝。
        if "cy" in props:
            fibre = (f"\n  cy = {props['cy']:.6e} {self.length_unit}"
                     f"　cz = {props['cz']:.6e} {self.length_unit}"
                     + ("（圆形，双向弯曲按平方和合成）" if props.get("circular")
                        else "")
                     + "\n  可用于强度验算")
        else:
            fibre = ("\n  形心偏置，本程序不给极端纤维距离 cy/cz："
                     "该截面的 I 本身是近似值，配一个精确的 c 算出来的应力"
                     "照样不对。\n  强度验算会明确拒绝，而不是给一个"
                     "来路不明的应力。")
        if params.get(ROOT_RADIUS_KEY):
            fibre += "\n  根部圆角已计入 A、Iy、Iz；J 按薄壁公式，未计圆角。"
        self.lbl_preview.setText(
            f"截面属性（自动计算）：\n"
            f"  A  = {props['A']:.6e} {self.length_unit}²\n"
            f"  Iy = {props['Iy']:.6e} {self.length_unit}⁴\n"
            f"  Iz = {props['Iz']:.6e} {self.length_unit}⁴\n"
            f"  J  = {props['J']:.6e} {self.length_unit}⁴" + shear + fibre
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
        if error is None:
            try:
                props = calc_section(stype, p)
            except ValueError as exc:
                error = f"{exc}"
        if error is None:
            # 只检查数值特性：结果里还有 geometry（dict）与 circular（bool），
            # 曾把 geometry 也送进 isfinite，点“确定”直接抛 TypeError、对话框关不掉。
            numbers = [props[k] for k in ("A", "Iy", "Iz", "J", "cy", "cz") if k in props]
            if not all(math.isfinite(v) and v > 0 for v in numbers):
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
            # 曾一律取 5A/6：只对实心矩形成立，默认工字钢强轴剪切面积因此大 2.2 倍。
            areas = section_geometry.shear_areas(props["geometry"])
            result["Ay"], result["Az"] = areas["Ay"], areas["Az"]
        return result
