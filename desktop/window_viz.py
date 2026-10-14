"""主窗口的 Visualization 命令：显示组、动画、路径曲线。

* 显示组（Abaqus 的 Display Group）：在视口选杆件后「仅显示」「隐藏」，或「全部显示」。
  大模型里看一榀框架、一层楼时，不必再透过整栋楼找那几根杆；隐藏的杆件也不参与拾取。
* 动画（Animate: Scale Factor）：变形图与振型按 -1…1 倍来回播放。振型是往复运动，
  静态图只能看到一个瞬间，分不清是平动、扭转还是局部振动。
* 路径曲线（XY Data from path）：见 path_plot.py。
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox


class WindowVizMixin:
    """结果显示命令。见模块 docstring。"""

    # --- 显示组 ---

    def display_only(self) -> None:
        self.region_pick.start("member", "仅显示所选杆件", self._finish_display_only,
                               self._initial_region("member"))

    def _finish_display_only(self, region: dict) -> None:
        keep = set(self._region_members(region))
        everyone = {int(m["id"]) for m in self.session.model.get("members") or []}
        self.viewport.set_hidden_members(everyone - keep)
        self._after_display_change(f"仅显示 {len(keep)} 根杆件；「全部显示」恢复。")

    def display_hide(self) -> None:
        self.region_pick.start("member", "隐藏所选杆件", self._finish_display_hide,
                               self._initial_region("member"))

    def _finish_display_hide(self, region: dict) -> None:
        hide = set(self._region_members(region))
        self.viewport.set_hidden_members(self.viewport.hidden_members | hide)
        self._after_display_change(f"已隐藏 {len(hide)} 根杆件；「全部显示」恢复。")

    def display_all(self) -> None:
        self.viewport.set_hidden_members(set())
        self._after_display_change("已显示全部杆件。")

    def _region_members(self, region: dict) -> list[int]:
        if "set" in region:
            return list((self.session.model.get("sets") or {}).get(region["set"], {})
                        .get("members") or [])
        return list(region.get("members") or [])

    def _after_display_change(self, message: str) -> None:
        self.statusBar().showMessage(message, 6000)
        self.redraw()

    # --- 动画 ---

    def toggle_animation(self) -> None:
        action = self.actions_by_name["animate"]
        if not action.isChecked():
            self.redraw()
            return
        if self.mode == "变形" and self.session.solution is not None and self.case:
            frame = self.session.frame
            scale = self.scale or self._auto_scale(frame)
            self.viewport.animate_deformed(frame, self.session.solution, self.case, scale)
            self.statusBar().showMessage("变形动画播放中；再点一次「动画」停止。", 6000)
        elif self.mode == "模态" and self.session.modal_view is not None:
            _key, _n, frame, result = self.session.modal_view
            f1 = result.frequencies[0]
            self.viewport.animate_mode(frame, result.shapes, 0,
                                       f"第 1 阶　{f1:.3f} Hz　周期 {1 / f1:.4f} s")
            self.statusBar().showMessage("振型动画播放中；再点一次「动画」停止。", 6000)
        else:
            action.setChecked(False)
            QMessageBox.information(self, "无法播放动画",
                                    "动画用于变形图与振型：请先求解并切到「变形」，或打开「模态」后再播放。")

    # --- 路径曲线 ---

    def path_plot(self) -> None:
        if self.session.solution is None or self.session.compilation is None:
            QMessageBox.information(self, "还没有结果", "路径曲线画的是求解结果：请先求解，再选路径。")
            return
        self.region_pick.start("member", "路径曲线：选一串首尾相连的杆件", self._finish_path,
                               self._initial_region("member"))

    def _finish_path(self, region: dict) -> None:
        from .path_plot import PathPlotDialog, order_path
        members = {int(m["id"]): (int(m["i"]), int(m["j"]))
                   for m in self.session.model.get("members") or []}
        try:
            ordered = order_path(members, self._region_members(region))
        except ValueError as exc:
            QMessageBox.warning(self, "路径不成立", f"{exc}。请重新选择一串首尾相连、不分叉的杆件。")
            return
        PathPlotDialog(self, ordered).exec()
