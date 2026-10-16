"""主窗口的结果显示选项、探针与极值定位。"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from . import theme
from .result_inspector import global_extreme, probe_member, show_stress_dialog


class WindowResultsMixin:
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
            f"距 i 端 x={x:.4g}/{length:.4g}",
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
