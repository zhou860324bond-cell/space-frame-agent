"""主窗口的结果显示选项、探针与极值定位。"""

from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QMessageBox

from . import theme
from .result_inspector import global_extreme, probe_member, show_stress_dialog


class WindowResultsMixin:
    def show_coordinate_query(self) -> None:
        unit = ("mm（实体结果）" if self.mode == "实体云图" else
                "mm" if self.session.model.get("units") == "N-mm-MPa" else "m")
        self.results.coordinate.setPlaceholderText(f"原始全局坐标 X, Y, Z（{unit}）")
        self.results_dock.show()
        self.results_dock.raise_()
        self.results.coordinate.setFocus()
        self.statusBar().showMessage(
            "输入原始全局坐标（当前模型长度单位）；交叉位置请填写杆件号。", 8000)

    def query_coordinate(self, text: str, member_text: str = "") -> None:
        if self.runner.busy:
            return
        self._sync_result_validity()
        from coordinates import parse_coordinate
        from .result_inspector import coordinate_target

        try:
            point = parse_coordinate(text)
            if self.mode == "实体云图" and self._solid_grid is not None:
                import numpy as np
                grid = self._solid_grid
                distances = np.linalg.norm(grid.points - point, axis=1)
                index = int(np.argmin(distances))
                if distances[index] <= max(float(grid.length) * 1e-9, 1e-9):
                    self.query_solid_point(index, focus=True)
                else:
                    cell = int(grid.find_containing_cell(point))
                    if cell < 0:
                        raise ValueError("坐标不在当前实体网格内；请检查坐标（mm）。")
                    self.query_solid_cell(cell, focus=True)
                    self.results.notes.setPlainText(
                        "坐标位于所选实体单元内；当前显示该单元原始应力和连接节点位移，"
                        "未将节点位移插值为输入坐标的位移。")
                    self.results.notes.show()
                return
            if not self._needs_solution():
                return
            elements = None
            if member_text.strip():
                physical = int(member_text.strip())
                mapping = self.session.compilation.mapping
                if physical not in mapping.physical_to_elements:
                    raise ValueError("杆件号不存在；请检查模型树中的杆件编号。")
                elements = mapping.element_ids(physical)
            kind, ident = coordinate_target(self.session.frame, point, elements)
            self._selected_kind, self._selected_id = kind, ident
            if kind == "node":
                self.query_node_displacement(ident)
                self.viewport.set_result_marker(point, f"NODE {ident}")
            else:
                self.probe_member_result(ident, point)
        except ValueError as exc:
            QMessageBox.warning(self, "坐标查询未完成", str(exc))

    def _remember_analysis(self, name, payload):
        from copy import deepcopy

        self._analysis_exports = getattr(self, "_analysis_exports", {})
        self._analysis_exports[name] = (self.session._model_key(), deepcopy(payload))

    def _remember_optimization(self, fingerprint, payload):
        self._analysis_exports = getattr(self, "_analysis_exports", {})
        self._analysis_exports["section_optimizer"] = (fingerprint, payload)

    def export_all_csv(self) -> None:
        if self.runner.busy:
            return
        self._sync_result_validity()
        path, _ = QFileDialog.getSaveFileName(
            self, "导出全部模块 CSV", "全部模块.csv", "CSV 文件 (*.csv)")
        if not path:
            return
        from csv_export import export_csv
        try:
            text = export_csv(self.session, analyses=getattr(self, "_analysis_exports", {}),
                              solid=self._solid_grid)
            with open(path, "w", newline="", encoding="utf-8") as stream:
                stream.write(text)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "CSV 导出失败", f"{exc}\n请检查模型或文件路径后重试。")
            return
        self.statusBar().showMessage(f"全部模块已导出：{path}", 8000)

    def query_node_displacement(self, node_id: int) -> None:
        """读取当前工况的真实节点自由度，显示值不乘变形放大系数。"""
        if self.runner.busy:
            return
        self._sync_result_validity()
        frame, solution = self.session.frame, self.session.solution
        if frame is None or solution is None or node_id not in frame.nodes:
            return
        self._last_probe = None
        from units import of

        name = self.case or solution.primary
        system = of(frame)
        values = solution[name].U[frame.node_dofs(node_id)]
        rows = [[component, float(value) * scale, unit, "全局坐标系"]
                for component, value, scale, unit in zip(
                    ("Ux", "Uy", "Uz", "URx", "URy", "URz"), values,
                    (system.disp_scale,) * 3 + (1.0,) * 3,
                    (system.disp_unit,) * 3 + ("rad",) * 3, strict=True)]
        magnitude = float(sum(float(v) ** 2 for v in values[:3]) ** 0.5)
        rows.append(["位移幅值", magnitude * system.disp_scale,
                     system.disp_unit, "平移向量的长度"])
        self._remember_analysis("node_query", {"node": node_id, "case": name, "rows": rows})
        self.results.show_rows(
            f"节点位移：节点 {node_id} · 工况/组合 {name}（实际值，未放大）",
            ["结果量", "值", "单位", "约定"], rows,
            [("node", node_id)] * len(rows))
        self.results_dock.show()
        self.results_dock.raise_()

    def _on_result_display_changed(self, options: dict) -> None:
        """更新视口选项；无结果时也记住配色与光照。"""
        self.result_display_options = dict(options)
        self.viewport.set_contour_palette(
            options.get("palette", theme.DEFAULT_PALETTE))
        self.viewport.set_contour_shading(options.get("shading", True))
        if not self.runner.busy and self.mode == "云图" and self.session.solution is not None:
            self.redraw()
        elif not self.runner.busy and self.mode == "实体云图":
            self.redraw()

    def probe_member_result(self, member_id: int, point) -> None:
        """将视口点击位置变成可审查的杆件截面结果表。"""
        if not self._needs_solution():
            return
        data = probe_member(self.session.frame, self.session.solution,
                            member_id, point, self.case)
        self._last_probe = data
        from copy import deepcopy
        exported = deepcopy(data)
        if exported["stress"]:
            grid = exported["stress"]["grid"]
            unit = "mm" if self.session.frame.units == "N-mm-MPa" else "m"
            exported["stress"]["grid"] = {
                key: {"unit": unit if key in {"y", "z"} else "MPa", "values": values}
                for key, values in grid.items() if key != "circular"}
        self._remember_analysis("member_query", exported)
        rows = []
        for component in ("N", "Vy", "Vz", "T", "My", "Mz"):
            unit = (data["moment_unit"] if component in {"T", "My", "Mz"}
                    else data["force_unit"])
            rows.append([component, data["forces"][component], unit,
                         "杆件局部分量"])
        for prefix, values in (("U(global)", data["global_displacement"]),
                               ("U(local)", data["local_displacement"])):
            for axis, value in zip("xyz", values, strict=True):
                rows.append([f"{prefix}.{axis}", float(value),
                             data["displacement_unit"], "中心线位移"])
        if data["stress"] is not None:
            rows += [["sigma_min", data["stress"]["min"], "MPa", "受压端"],
                     ["sigma_max", data["stress"]["max"], "MPa", "受拉端"]]
        else:
            rows.append(["截面正应力", "不可用", "", data["stress_error"]])
        self.results_dock.setVisible(True)
        self.results_dock.raise_()
        # 对外报物理杆件编号；点中的分析杆段只作补充（用户在模型里找不到杆段号）
        physical = self.viewport._owner(member_id)
        x, length = data["x"], data["length"]
        segment = ""
        if physical != member_id or len(self.viewport._elements([physical])) > 1:
            # 剖分过的杆件：位置按整根物理杆件量，不按点中的那一段
            import numpy as np
            nodes = {int(n["id"]): n for n in self.session.model.get("nodes") or []}
            m = next((m for m in self.session.model.get("members") or []
                      if int(m["id"]) == physical), None)
            if m is not None:
                pi, pj = (np.array([nodes[int(m[e])][k] for k in "xyz"], float) for e in ("i", "j"))
                length = float(np.linalg.norm(pj - pi))
                x = float(np.dot(np.asarray(data["point"], float) - pi, pj - pi) / length)
            segment = f"（分析杆段 {member_id}）"
        self.results.show_rows(
            f"结果探针：杆件 {physical}{segment}，工况 {data['case']}，"
            f"距 i 端 x={x:.4g}/{length:.4g}"
            "\n位置为未变形杆件中心线；位移由内力曲率积分恢复并插值，未乘显示放大系数。"
            "应力为截面极端纤维正应力（轴力与双向弯曲），不含剪力和扭转剪应力。",
            ["结果量", "值", "单位", "约定"], rows,
            [("member", physical)] * len(rows))
        self.viewport.set_result_marker(
            data["point"], f"探针 · 杆件 {physical} · x = {data['x']:.3g}")

    def clear_solid_query(self) -> None:
        """换档、失效或主动清除时撤去旧单元标记和查询表。"""
        self.viewport.select_solid("cell", None)
        self.solid_panel.query_status.setText("尚未选择单元。序号按当前结果文件排列，从 1 开始。")
        if getattr(self, "_solid_query_table", False):
            self.results.show_message("实体查询已清除", "请点选当前网格的单元或定位实体极值。")
        self._solid_query_table = False

    def _show_solid_query(self, title, rows, locators) -> None:
        self._solid_query_table = True
        self.results.show_rows(title, ["结果量", "值", "单位", "位置"], rows, locators)
        self.results_dock.show()
        self.solid_panel.query_status.setText(title)

    def query_solid_cell(self, index: int, *, focus: bool = False) -> None:
        """原始单元应力及十个连接节点的坐标与位移；节点位移不平均成单元位移。"""
        if self._solid_grid is None or self.runner.busy:
            return
        from .solid_result import query_cell
        data = query_cell(self._solid_grid, index)
        self.viewport.select_solid("cell", index, focus=focus)
        title = f"实体单元序号 {index + 1} · 原始结果（应力不跨单元平均）"
        rows = [["Mises", data["Mises_MPa"], "MPa", f"单元 {index + 1}"],
                ["最大绝对主应力", data["AbsPrincipal_MPa"], "MPa", f"单元 {index + 1}"]]
        locators = [("solid_cell", index + 1)] * 2
        for node, coordinates, displacement in zip(data["nodes"], data["coordinates"],
                                                  data["displacement"], strict=True):
            for prefix, values in (("坐标", coordinates), ("U", displacement)):
                for axis, value in zip("xyz", values, strict=True):
                    rows.append([f"{prefix}.{axis}", float(value), "mm", f"节点序号 {node + 1}"])
                    locators.append(("solid_point", int(node) + 1))
        self._show_solid_query(title, rows, locators)

    def query_solid_point(self, index: int, *, focus: bool = False) -> None:
        grid = self._solid_grid
        if grid is None or self.runner.busy or not 0 <= index < grid.n_points:
            return
        self.viewport.select_solid("point", index, focus=focus)
        rows = [["位移幅值", float(grid.point_data["DisplacementMagnitude_mm"][index]),
                 "mm", f"节点序号 {index + 1}"]]
        for prefix, values in (("坐标", grid.points[index]),
                               ("U", grid.point_data["Displacement_mm"][index])):
            rows.extend([[f"{prefix}.{axis}", float(value), "mm", f"节点序号 {index + 1}"]
                         for axis, value in zip("xyz", values, strict=True)])
        self._show_solid_query(f"实体节点序号 {index + 1} · 原始坐标及位移", rows,
                               [("solid_point", index + 1)] * len(rows))

    def locate_solid_extreme(self) -> None:
        """按全部原始结果定位当前结果量的真实峰值，不取截断色标上限，也不需要整体求解。"""
        if self._solid_grid is None or self.runner.busy:
            return
        from .solid_result import result_extreme
        if self.mode != "实体云图":
            self.set_mode("实体云图")
        association, index, value = result_extreme(self._solid_grid,
                                                   self.solid_panel.field.currentData())
        if association == "cell":
            self.query_solid_cell(index, focus=True)
        else:
            self.query_solid_point(index, focus=True)
        self.statusBar().showMessage(f"已定位实体真实峰值 {value:.6g}，按全部原始结果取值。", 5000)

    def locate_current_extreme(self) -> None:
        """定位当前梁内力分量的全结构绝对极值；实体云图下定位实体峰值。"""
        if self.mode == "实体云图":
            self.locate_solid_extreme()
            return
        if not self._needs_solution():
            return
        extreme = global_extreme(self.session.frame, self.session.solution,
                                 self.component, self.case)
        if extreme["member"] is None:
            return
        extreme = {**extreme, "member": self.viewport._owner(extreme["member"])}
        self.locate("member", extreme["member"])
        from .viewport import peak_tag
        self.viewport.set_result_marker(
            extreme["point"],
            peak_tag(f"|{self.component}|max = ", extreme["value"], extreme["unit"],
                     extreme["member"], extreme["x"]))
        self.results.show_rows(
            f"{self.component} 全结构绝对极值，工况 {extreme['case']}",
            ["杆件", "距 i 端 x", "值", "单位"],
            [[extreme["member"], extreme["x"], extreme["value"],
              extreme["unit"]]], [("member", extreme["member"])])

    def show_section_stress(self) -> None:
        """打开当前探针截面的轴力与双向弯曲正应力图。"""
        if not self._needs_solution():
            return
        data = self._last_probe
        selected = getattr(self, "_selected_id", None)
        if data is None or (selected is not None
                            and self.viewport._owner(data["member"]) != selected):
            if getattr(self, "_selected_kind", None) != "member":
                QMessageBox.information(
                    self, "请选择杆件", "先在视口中选择或探测一根杆件。")
                return
            segments = self.viewport._elements([selected]) or [selected]
            element = segments[len(segments) // 2]          # 取物理杆件中段的杆段
            member = self.session.frame.members[element]
            midpoint = 0.5 * (self.session.frame.nodes[member.i].xyz
                              + self.session.frame.nodes[member.j].xyz)
            data = probe_member(self.session.frame, self.session.solution,
                                element, midpoint, self.case)
            self._last_probe = data
        if data["stress"] is None:
            QMessageBox.information(self, "截面正应力不可用", data["stress_error"])
            return
        show_stress_dialog(self, data)
