"""多参数多目标截面优化（桌面端面板）。

同时变化多个截面参数，同时优化多个目标（重量/位移/成本），
满足约束（位移/最小面积）。多目标时返回 Pareto 前沿。

优化过程走 Runner 后台线程，不阻塞界面。
"""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from . import theme


# 截面类型对应的参数名
SECTION_PARAMS = {
    "工字形 / H 型钢": ["height", "flange_width", "web_thickness", "flange_thickness"],
    "矩形": ["width", "height"],
    "圆管": ["outer_diameter", "thickness"],
    "实心圆": ["diameter"],
}

# 参数默认范围 (min, max, num_points)
PARAM_DEFAULTS = {
    "height": (0.2, 1.0, 5),
    "flange_width": (0.1, 0.4, 4),
    "web_thickness": (0.005, 0.02, 3),
    "flange_thickness": (0.008, 0.03, 3),
    "width": (0.1, 0.5, 4),
    "outer_diameter": (0.1, 0.5, 4),
    "thickness": (0.005, 0.02, 3),
    "diameter": (0.1, 0.5, 5),
}


class SectionOptPanel(QWidget):
    """截面优化面板。"""

    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self._var_inputs: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox, QSpinBox]] = {}
        self._result = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 标题
        title = QLabel("多参数多目标截面优化")
        title.setStyleSheet(f"font-weight:650; font-size:10pt; color:{theme.INK};")
        layout.addWidget(title)

        # 基本配置
        cfg_row = QHBoxLayout()
        self.cmb_section = QComboBox()
        self.cmb_section.setToolTip("要优化的截面（从当前模型的截面中选）")
        self.cmb_type = QComboBox()
        self.cmb_type.addItems(SECTION_PARAMS.keys())
        self.cmb_type.currentTextChanged.connect(self._rebuild_var_inputs)
        self.cmb_algo = QComboBox()
        self.cmb_algo.addItems(["grid_search", "random_search"])
        cfg_row.addWidget(QLabel("截面"))
        cfg_row.addWidget(self.cmb_section)
        cfg_row.addWidget(QLabel("类型"))
        cfg_row.addWidget(self.cmb_type)
        cfg_row.addWidget(QLabel("算法"))
        cfg_row.addWidget(self.cmb_algo)
        layout.addLayout(cfg_row)

        # 变量范围配置区
        self.var_container = QWidget()
        self.var_layout = QGridLayout(self.var_container)
        self.var_layout.setContentsMargins(0, 0, 0, 0)
        self.var_layout.setSpacing(4)
        layout.addWidget(QLabel("变量范围（min / max / 点数）"))
        layout.addWidget(self.var_container)
        self._rebuild_var_inputs()

        # 目标和约束
        obj_row = QHBoxLayout()
        self.chk_weight = QCheckBox("最小重量")
        self.chk_weight.setChecked(True)
        self.chk_disp = QCheckBox("最小位移")
        self.chk_disp.setChecked(True)
        self.chk_cost = QCheckBox("最小成本")
        obj_row.addWidget(self.chk_weight)
        obj_row.addWidget(self.chk_disp)
        obj_row.addWidget(self.chk_cost)
        obj_row.addStretch()
        layout.addLayout(obj_row)

        con_row = QHBoxLayout()
        self.chk_disp_limit = QCheckBox("位移限值")
        self.spn_disp_limit = QDoubleSpinBox()
        self.spn_disp_limit.setRange(0.001, 1.0)
        self.spn_disp_limit.setValue(0.02)
        self.spn_disp_limit.setSingleStep(0.005)
        self.spn_disp_limit.setDecimals(4)
        self.spn_disp_limit.setEnabled(False)
        self.chk_disp_limit.toggled.connect(self.spn_disp_limit.setEnabled)
        self.chk_area_limit = QCheckBox("最小面积")
        self.spn_area_limit = QDoubleSpinBox()
        self.spn_area_limit.setRange(0.0001, 0.1)
        self.spn_area_limit.setValue(0.005)
        self.spn_area_limit.setSingleStep(0.001)
        self.spn_area_limit.setDecimals(5)
        self.spn_area_limit.setEnabled(False)
        self.chk_area_limit.toggled.connect(self.spn_area_limit.setEnabled)
        con_row.addWidget(self.chk_disp_limit)
        con_row.addWidget(self.spn_disp_limit)
        con_row.addWidget(QLabel("m"))
        con_row.addWidget(self.chk_area_limit)
        con_row.addWidget(self.spn_area_limit)
        con_row.addWidget(QLabel("m²"))
        con_row.addStretch()
        layout.addLayout(con_row)

        # 运行按钮
        self.btn_run = QPushButton("运行截面优化")
        self.btn_run.clicked.connect(self._run)
        self.btn_run.setStyleSheet(
            f"background:{theme.ACCENT}; color:white; font-weight:600;")
        layout.addWidget(self.btn_run)

        # 状态
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color:{theme.INK_MUTED}; font-size:8pt;")
        layout.addWidget(self.lbl_status)

        # 结果统计
        stat_row = QHBoxLayout()
        self.lbl_total = QLabel("—")
        self.lbl_feasible = QLabel("—")
        self.lbl_pareto = QLabel("—")
        self.lbl_time = QLabel("—")
        for lbl, name in [(self.lbl_total, "评估数"), (self.lbl_feasible, "可行"),
                           (self.lbl_pareto, "Pareto"), (self.lbl_time, "耗时")]:
            c = QVBoxLayout()
            c.addWidget(QLabel(name))
            lbl.setStyleSheet(f"font-weight:650; font-size:10pt; color:{theme.INK};")
            c.addWidget(lbl)
            stat_row.addLayout(c)
        layout.addLayout(stat_row)

        # 结果表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["截面参数", "重量", "位移", "可行"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(200)
        layout.addWidget(self.table)

        layout.addStretch()

    def refresh_sections(self):
        """从当前模型刷新截面列表。"""
        self.cmb_section.clear()
        for s in self.session.model.get("sections", []):
            name = s.get("name", "")
            if name:
                self.cmb_section.addItem(name)

    def _rebuild_var_inputs(self):
        """根据截面类型重建变量输入框。"""
        # 清空
        while self.var_layout.count():
            item = self.var_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._var_inputs.clear()

        section_type = self.cmb_type.currentText()
        params = SECTION_PARAMS.get(section_type, [])
        for col, pname in enumerate(params):
            dmin, dmax, dnum = PARAM_DEFAULTS.get(pname, (0.1, 0.5, 3))
            sp_min = QDoubleSpinBox()
            sp_min.setRange(0.0001, 10.0)
            sp_min.setValue(dmin)
            sp_min.setDecimals(4)
            sp_min.setSingleStep(0.01)
            sp_max = QDoubleSpinBox()
            sp_max.setRange(0.0001, 10.0)
            sp_max.setValue(dmax)
            sp_max.setDecimals(4)
            sp_max.setSingleStep(0.01)
            sp_num = QSpinBox()
            sp_num.setRange(2, 10)
            sp_num.setValue(dnum)

            col_layout = QVBoxLayout()
            col_layout.addWidget(QLabel(pname))
            row = QHBoxLayout()
            row.addWidget(sp_min)
            row.addWidget(QLabel("~"))
            row.addWidget(sp_max)
            col_layout.addLayout(row)
            num_row = QHBoxLayout()
            num_row.addWidget(QLabel("点数"))
            num_row.addWidget(sp_num)
            col_layout.addLayout(num_row)

            self.var_layout.addLayout(col_layout, 0, col)
            self._var_inputs[pname] = (sp_min, sp_max, sp_num)

    def _run(self):
        if self.runner.busy:
            return
        section_name = self.cmb_section.currentText()
        if not section_name:
            self.lbl_status.setText("当前模型没有截面，先到前处理建模")
            return

        objectives = []
        if self.chk_weight.isChecked(): objectives.append("weight")
        if self.chk_disp.isChecked(): objectives.append("max_displacement")
        if self.chk_cost.isChecked(): objectives.append("material_cost")
        if not objectives:
            self.lbl_status.setText("至少选一个优化目标")
            return

        variables = {}
        for pname, (sp_min, sp_max, sp_num) in self._var_inputs.items():
            variables[pname] = (sp_min.value(), sp_max.value(), sp_num.value())

        constraints = {}
        if self.chk_disp_limit.isChecked():
            constraints["max_displacement_limit"] = self.spn_disp_limit.value()
        if self.chk_area_limit.isChecked():
            constraints["min_section_area"] = self.spn_area_limit.value()

        section_type = self.cmb_type.currentText()
        algo = self.cmb_algo.currentText()
        model = self.session.model

        self.btn_run.setEnabled(False)
        self.lbl_status.setText("正在遍历参数组合并求解……")

        def _job():
            from section_optimizer import SectionOptimizer
            opt = SectionOptimizer(
                model=model, section_name=section_name,
                section_type=section_type, variables=variables,
                objectives=objectives, constraints=constraints)
            if algo == "random_search":
                n = 1
                for v in variables.values():
                    n *= v[2]
                return opt.random_search(n_samples=n)
            return opt.grid_search()

        self.runner.submit(_job, on_done=self._on_done, on_failed=self._on_failed)

    def _on_done(self, result):
        self.btn_run.setEnabled(True)
        self._result = result
        self.lbl_total.setText(str(result.total_calls))
        self.lbl_feasible.setText(str(result.feasible_count))
        self.lbl_pareto.setText(str(len(result.pareto)))
        self.lbl_time.setText(f"{result.duration_ms:.0f}ms")
        self.lbl_status.setText("✅ 优化完成")
        self.lbl_status.setStyleSheet(f"color:{theme.ACCENT}; font-size:8pt;")

        # 填充表格：Pareto 前沿 + 最优解
        points = result.pareto if result.pareto else sorted(
            [e for e in result.evaluations if e.feasible],
            key=lambda e: e.objectives.get("weight", float("inf")))[:10]
        self.table.setRowCount(len(points))
        for row, pt in enumerate(points):
            params_str = ", ".join(f"{k}={v:.4g}" for k, v in pt.params.items())
            self.table.setItem(row, 0, QTableWidgetItem(params_str))
            w = pt.objectives.get("weight")
            self.table.setItem(row, 1, QTableWidgetItem(f"{w:.4g}" if w else "—"))
            d = pt.objectives.get("max_displacement")
            self.table.setItem(row, 2, QTableWidgetItem(f"{d:.4g}" if d else "—"))
            self.table.setItem(row, 3, QTableWidgetItem("✓" if pt.feasible else "✗"))
        self.table.resizeColumnsToContents()

    def _on_failed(self, exc_type, msg):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"❌ 优化失败：{exc_type}: {msg}")
        self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
