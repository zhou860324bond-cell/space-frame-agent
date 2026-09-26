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
        if self.mode == "云图" and self.session.solution is not None:
            self.redraw()

    def probe_member_result(self, member_id: int, point) -> None:
        """将视口点击位置变成可审查的杆件截面结果表。"""
        if self.session.solution is None or self.session.frame is None:
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
        self.results.show_rows(
            f"结果探针：杆件 {member_id}，工况 {data['case']}，"
            f"距 i 端 x={data['x']:.4g}/{data['length']:.4g}",
            ["结果量", "值", "单位", "约定"], rows,
            [("member", member_id)] * len(rows))
        self.viewport.set_result_marker(
            data["point"], f"PROBE M{member_id} x={data['x']:.3g}")

    def locate_current_extreme(self) -> None:
        """定位当前梁内力分量的全结构绝对极值。"""
        if self.session.solution is None or self.session.frame is None:
            QMessageBox.information(self, "尚无分析结果", "请先求解。")
            return
        extreme = global_extreme(self.session.frame, self.session.solution,
                                 self.component, self.case)
        if extreme["member"] is None:
            return
        self.locate("member", extreme["member"])
        self.viewport.set_result_marker(
            extreme["point"],
            f"MAX |{self.component}|={extreme['value']:+.3g} {extreme['unit']}")
        self.results.show_rows(
            f"{self.component} 全结构绝对极值，工况 {extreme['case']}",
            ["杆件", "距 i 端 x", "值", "单位"],
            [[extreme["member"], extreme["x"], extreme["value"],
              extreme["unit"]]], [("member", extreme["member"])])

    def show_section_stress(self) -> None:
        """打开当前探针截面的轴力与双向弯曲正应力图。"""
        if self.session.solution is None or self.session.frame is None:
            QMessageBox.information(self, "尚无分析结果", "请先求解。")
            return
        data = self._last_probe
        selected = getattr(self, "_selected_id", None)
        if data is None or (selected is not None and data["member"] != selected):
            if getattr(self, "_selected_kind", None) != "member":
                QMessageBox.information(
                    self, "请选择杆件", "先在视口中选择或探测一根杆件。")
                return
            member = self.session.frame.members[selected]
            midpoint = 0.5 * (self.session.frame.nodes[member.i].xyz
                              + self.session.frame.nodes[member.j].xyz)
            data = probe_member(self.session.frame, self.session.solution,
                                selected, midpoint, self.case)
            self._last_probe = data
        if data["stress"] is None:
            QMessageBox.information(self, "截面正应力不可用", data["stress_error"])
            return
        show_stress_dialog(self, data)
