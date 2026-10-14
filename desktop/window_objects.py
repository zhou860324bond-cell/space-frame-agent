"""主窗口的建模对象命令：Abaqus 式创建、区域选择、编辑与管理器。

流程与 Abaqus/CAE 相同：

* 创建边界条件 / 创建载荷：创建对话框（名称、分析步或工况、类型）→ 提示区选区域
  （点选、Shift 加选、Ctrl 减选、拖框，或使用集合）→ 编辑对话框填数值；
* 截面指派：先选区域，再选截面和材料；
* 管理器：同一组操作——创建、编辑、编辑区域、复制、改名、删除、抑制、恢复。

所有修改都调 session 的对象工具，撤销、校验、再生成在那里统一完成；这里只编排
对话框与拾取。原先这几个命令要求先单选一个对象，选多个只能重复整套流程。
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QInputDialog, QMessageBox

from .object_dialogs import (BC_TYPES, LOAD_FIELD, LOAD_REGION, LOAD_TYPES,
                             AssignSectionDialog, CreateObjectDialog, EditBCDialog,
                             EditLoadDialog, ObjectManagerDialog, OrientationDialog,
                             next_name)

_PRESET_FIX = {"ENCASTRE": [1] * 6, "PINNED": [1, 1, 1, 0, 0, 0],
               "XSYMM": [1, 0, 0, 0, 1, 1], "YSYMM": [0, 1, 0, 1, 0, 1],
               "ZSYMM": [0, 0, 1, 1, 1, 0], "SPRING": [0] * 6}
_KIND_ALIAS = {"section_assignments": "section_assignment",
               "beam_orientations": "beam_orientation", "bcs": "bc", "loads": "load"}


class WindowObjectsMixin:
    """建模对象命令。见模块 docstring。"""

    # --- 公用 ---

    def _feature_names(self, kind: str) -> set[str]:
        return {f["name"] for f in (self.session.model.get("features") or {}).get(kind) or []}

    def _initial_region(self, kind: str) -> list[int]:
        """命令前已选中的同类对象，作为默认区域带进来（保留先选后点的用法）。"""
        if getattr(self, "_selected_kind", None) == kind and self._selected_id is not None:
            return [self._selected_id]
        return []

    def _object_result(self, result, done: str) -> bool:
        if not result.ok:
            QMessageBox.warning(self, "对象未写入", self._explain(result.payload))
            return False
        self._after_manual_edit(done)
        return True

    # --- 边界条件 ---

    def create_bc(self) -> None:
        """Abaqus 式：创建边界条件 → 选节点区域 → 编辑约束。"""
        steps = ["Initial"] + [s["name"] for s in self.session.model.get("steps") or []]
        dialog = CreateObjectDialog("创建边界条件", next_name("BC", self._feature_names("bcs")),
                                    "分析步", steps, BC_TYPES, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, step, preset = dialog.values()
        self.region_pick.start("node", f"创建边界条件「{name}」",
                               lambda region: self._finish_bc(name, step, preset, region),
                               self._initial_region("node"))

    def _finish_bc(self, name: str, step: str, preset, region: dict) -> None:
        editor = EditBCDialog(name, _PRESET_FIX[preset or "SPRING"], parent=self)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        fix, spring = editor.values()
        result = self.session.create_boundary_condition(name, region, fix=fix, spring=spring,
                                                        step=step)
        self._object_result(result, f"已创建边界条件「{name}」。")

    # --- 荷载 ---

    def create_load(self) -> None:
        """Abaqus 式：创建载荷 → 选区域（节点或杆件，按类型）→ 编辑分量。"""
        cases = [c["name"] for c in self.session.model.get("load_cases") or []]
        dialog = CreateObjectDialog("创建载荷", next_name("Load", self._feature_names("loads")),
                                    "工况", cases or ["Load-1"], LOAD_TYPES, self,
                                    editable_scope=True)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, case, load_type = dialog.values()
        if case not in cases:
            # 输入了一个新工况名：就地建好，不必先去别处建工况
            created = self.session.add_load_case(case)
            if not created.ok:
                QMessageBox.warning(self, "工况未创建", self._explain(created.payload))
                return
        kind = LOAD_REGION[load_type]
        self.region_pick.start(kind, f"创建载荷「{name}」",
                               lambda region: self._finish_load(name, case, load_type, region),
                               self._initial_region(kind))

    def _finish_load(self, name: str, case: str, load_type: str, region: dict) -> None:
        default = {"gravity": [0.0, 0.0, -self.session.units.gravity]}.get(load_type,
                                                                            [0.0, 0.0, 0.0])
        editor = EditLoadDialog(name, load_type, default,
                                self.session.model.get("units", "N-m-Pa"), self)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        result = self.session.create_load(name, case, load_type, region,
                                          **{LOAD_FIELD[load_type]: editor.vector()})
        self._object_result(result, f"已创建载荷「{name}」。")

    # --- 截面指派 ---

    def assign_section(self) -> None:
        """Abaqus 式：先选杆件区域，再选截面与材料。"""
        sections = [s["name"] for s in self.session.model.get("sections") or []]
        materials = [m["name"] for m in self.session.model.get("materials") or []]
        if not sections or not materials:
            self.statusBar().showMessage("请先创建材料和截面，再进行截面指派。", 5000)
            self._show_ribbon_page("属性")
            return
        self.region_pick.start("member", "截面指派",
                               lambda region: self._finish_assignment(region, sections,
                                                                       materials),
                               self._initial_region("member"))

    def _finish_assignment(self, region: dict, sections: list[str],
                           materials: list[str]) -> None:
        count = (f"区域：集合「{region['set']}」" if "set" in region
                 else f"区域：{len(region['members'])} 根杆件")
        dialog = AssignSectionDialog(next_name("SA", self._feature_names("section_assignments")),
                                     sections, materials, count, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, section, material = dialog.values()
        result = self.session.create_section_assignment(name, region, section, material)
        self._object_result(result, f"已创建截面指派「{name}」。")

    # --- 模型树 ---

    def _on_feature_action(self, kind: str, action: str, name) -> None:
        """模型树右键菜单：与管理器同一组动作。"""
        if action == "manager":
            self.open_object_manager(kind)
        else:
            self.manage_object(kind, action, name)

    def _on_feature_selected(self, kind: str, name: str) -> None:
        """单击模型树里的对象：在视口里高亮它的区域，状态栏说明范围。"""
        alias = _KIND_ALIAS[kind]
        listed = self.session.list_features(alias).payload["features"][kind]
        feature = next((f for f in listed if f["name"] == name), None)
        if feature is None:
            return
        what = "node" if "nodes" in feature else "member"
        ids = feature.get("nodes") or feature.get("members") or []
        self.viewport.highlight_region(what, ids)
        noun = "节点" if what == "node" else "杆件"
        self.statusBar().showMessage(
            f"「{name}」作用在 {len(ids)} 个{noun}上" +
            (f"：{feature['problem']}" if feature.get("problem") else ""), 5000)

    # --- 梁方向 ---

    def assign_beam_orientation(self) -> None:
        """Abaqus 式 Assign Beam Orientation：先选杆件区域，再给 n1。"""
        if not self.session.model.get("members"):
            self.statusBar().showMessage("还没有杆件：请先建立几何，再指定梁方向。", 5000)
            return
        self.region_pick.start("member", "梁方向", self._finish_orientation,
                               self._initial_region("member"))

    def _finish_orientation(self, region: dict) -> None:
        count = (f"区域：集合「{region['set']}」" if "set" in region
                 else f"区域：{len(region['members'])} 根杆件")
        dialog = OrientationDialog(next_name("Ori", self._feature_names("beam_orientations")),
                                   [0.0, 0.0, 1.0], count, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, n1 = dialog.values()
        self._object_result(self.session.create_beam_orientation(name, region, n1),
                            f"已指定梁方向「{name}」。")

    def open_orientation_manager(self) -> None:
        self.open_object_manager("beam_orientations")

    # --- 管理器 ---

    def open_object_manager(self, kind: str) -> None:
        ObjectManagerDialog(self, kind).exec()
        self.refresh()

    def open_bc_manager(self) -> None:
        self.open_object_manager("bcs")

    def open_load_manager(self) -> None:
        self.open_object_manager("loads")

    def open_assignment_manager(self) -> None:
        self.open_object_manager("section_assignments")

    def manage_object(self, kind: str, action: str, name: str | None, manager=None) -> None:
        """管理器与模型树右键菜单共用的动作。"""
        alias = _KIND_ALIAS[kind]
        session = self.session
        if action == "create":
            if manager is not None:
                manager.accept()
            {"bcs": self.create_bc, "loads": self.create_load,
             "section_assignments": self.assign_section,
             "beam_orientations": self.assign_beam_orientation}[kind]()
            return
        if name is None:
            return
        feature = next(f for f in session.model["features"][kind] if f["name"] == name)
        if action == "edit":
            self._edit_object(kind, feature)
        elif action == "region":
            if manager is not None:
                manager.accept()
            what = "node" if kind == "bcs" or LOAD_REGION.get(feature.get("type")) == "node" \
                else "member"
            listed = session.list_features(alias).payload["features"][kind]
            current = next(f for f in listed if f["name"] == name)
            self.region_pick.start(
                what, f"编辑「{name}」的区域",
                lambda region: self._object_result(
                    session.edit_feature(alias, name, {"region": region}),
                    f"「{name}」的区域已更新。"),
                current.get("nodes") or current.get("members") or [])
        elif action == "copy":
            new, ok = QInputDialog.getText(self, "复制", "副本名称：",
                                           text=f"{name}-副本")
            if ok and new.strip():
                self._object_result(session.copy_feature(alias, name, new.strip()),
                                    f"已复制为「{new.strip()}」。")
        elif action == "rename":
            new, ok = QInputDialog.getText(self, "改名", "新名称：", text=name)
            if ok and new.strip() and new.strip() != name:
                self._object_result(session.rename_feature(alias, name, new.strip()),
                                    f"已改名为「{new.strip()}」。")
        elif action == "delete":
            if QMessageBox.question(self, "删除", f"删除「{name}」？由它派生的约束、荷载或指派会一并去掉。"
                                    ) == QMessageBox.StandardButton.Yes:
                self._object_result(session.delete_feature(alias, name), f"已删除「{name}」。")
        elif action in ("suppress", "resume"):
            self._object_result(session.suppress_feature(alias, name, action == "suppress"),
                                f"「{name}」已{'抑制' if action == 'suppress' else '恢复'}。")

    def _edit_object(self, kind: str, feature: dict) -> None:
        alias, name = _KIND_ALIAS[kind], feature["name"]
        if kind == "bcs":
            editor = EditBCDialog(name, feature["fix"], feature.get("spring"), self)
            if editor.exec() == QDialog.DialogCode.Accepted:
                fix, spring = editor.values()
                self._object_result(self.session.edit_feature(
                    alias, name, {"fix": fix, "spring": spring}), f"「{name}」已更新。")
        elif kind == "loads":
            field = LOAD_FIELD[feature["type"]]
            editor = EditLoadDialog(name, feature["type"], feature[field],
                                    self.session.model.get("units", "N-m-Pa"), self)
            if editor.exec() == QDialog.DialogCode.Accepted:
                self._object_result(self.session.edit_feature(
                    alias, name, {field: editor.vector()}), f"「{name}」已更新。")
        elif kind == "beam_orientations":
            dialog = OrientationDialog(name, feature["n1"], "修改方向", self)
            dialog.name.setEnabled(False)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                _, n1 = dialog.values()
                self._object_result(self.session.edit_feature(alias, name, {"n1": n1}),
                                    f"「{name}」已更新。")
        else:
            sections = [s["name"] for s in self.session.model.get("sections") or []]
            materials = [m["name"] for m in self.session.model.get("materials") or []]
            dialog = AssignSectionDialog(name, sections, materials, "修改截面或材料", self)
            dialog.name.setEnabled(False)
            dialog.section.setCurrentText(feature["section"])
            dialog.material.setCurrentText(feature["material"])
            if dialog.exec() == QDialog.DialogCode.Accepted:
                _, section, material = dialog.values()
                self._object_result(self.session.edit_feature(
                    alias, name, {"section": section, "material": material}),
                    f"「{name}」已更新。")
