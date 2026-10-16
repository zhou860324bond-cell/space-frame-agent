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

from .object_dialogs import (BC_TYPES, CONSTRAINT_TYPES, LOAD_FIELD, LOAD_REGION, LOAD_TYPES,
                             SPAN_LOADS, AssignSectionDialog, CreateObjectDialog, DisplacementBCDialog,
                             EditBCDialog,
                             EditConstraintDialog, EditLoadDialog, ObjectManagerDialog,
                             OrientationDialog, next_name)

_PRESET_FIX = {"ENCASTRE": [1] * 6, "PINNED": [1, 1, 1, 0, 0, 0],
               "XSYMM": [1, 0, 0, 0, 1, 1], "YSYMM": [0, 1, 0, 1, 0, 1],
               "ZSYMM": [0, 0, 1, 1, 1, 0], "SPRING": [0] * 6}
_KIND_ALIAS = {"section_assignments": "section_assignment",
               "beam_orientations": "beam_orientation", "bcs": "bc", "loads": "load",
               "constraints": "constraint"}


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
        if preset == "DISPLACEMENT":
            self._finish_displacement_bc(name, step, region)
            return
        editor = EditBCDialog(name, _PRESET_FIX[preset or "SPRING"], parent=self)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        fix, spring = editor.values()
        result = self.session.create_boundary_condition(name, region, fix=fix, spring=spring,
                                                        step=step)
        self._object_result(result, f"已创建边界条件「{name}」。")

    def direct_supports(self) -> list[dict]:
        """没有对象来源的支座条目（框架生成器与旧模型写的「直接定义」）。"""
        return [s for s in self.session.model.get("supports") or [] if not s.get("owners")]

    def convert_direct_supports(self, confirm: bool = True) -> bool:
        """把「直接定义」的支座转成边界条件对象，之后统一在管理器里编辑。

        旧的「边界条件」表单删掉以后，生成器建出来的支座只有这一条路能改：
        管理器只管对象，看不见直接条目。转换必须在模型层一次完成——界面里
        先删后建会被「支座不能为空」挡住，先建后删又会被「直接定义与对象重叠」
        挡住，所以这里只调用 Session.convert_direct_supports。
        """
        direct = self.direct_supports()
        if not direct:
            return False
        convert = getattr(self.session, "convert_direct_supports", None)
        if convert is None:
            QMessageBox.information(
                self, "暂不能转换",
                f"这 {len(direct)} 个支座来自旧式「直接定义」，需要模型层的转换功能才能交给"
                "边界条件管理器编辑；该功能尚未合入。可以先在「模型表格」的支座表里修改。")
            return False
        if confirm and QMessageBox.question(
                self, "转成边界条件对象",
                f"这 {len(direct)} 个支座来自旧式「直接定义」（框架生成器或旧模型），"
                "边界条件管理器看不到它们。\n\n转成边界条件对象后，可以在管理器里编辑、复制、"
                "删除，并按分析步修改或失活；约束方向保持不变，可以撤销。现在转换吗？"
        ) != QMessageBox.StandardButton.Yes:
            return False
        return self._object_result(convert(), f"已把 {len(direct)} 个直接定义的支座转成边界条件对象。")

    # --- 约束（Interaction 模块） ---

    def create_constraint_ui(self) -> None:
        """Abaqus 式：创建约束（名称、类型）→ 选主节点 → 选从节点区域 → 编辑。"""
        if not self.session.model.get("nodes"):
            self.statusBar().showMessage("还没有节点：请先建立几何，再创建约束。", 5000)
            return
        dialog = CreateObjectDialog("创建约束", next_name("Constraint", self._feature_names("constraints")),
                                    "", None, CONSTRAINT_TYPES, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, _, kind = dialog.values()
        self.region_pick.start("node", f"约束「{name}」：选主节点（控制点）",
                               lambda region: self._constraint_control(name, kind, region),
                               self._initial_region("node"))

    def _constraint_control(self, name: str, kind: str, region: dict) -> None:
        ids = region.get("nodes") or []
        if "set" in region or len(ids) != 1:
            QMessageBox.information(self, "创建约束", "主节点只能有一个：请在视口里点选一个节点，不用集合。")
            return
        control = ids[0]
        self.region_pick.start("node", f"约束「{name}」：选从节点区域（主节点 {control}）",
                               lambda slaves: self._finish_constraint(name, kind, control, slaves))

    def _finish_constraint(self, name: str, kind: str, control: int, region: dict) -> None:
        nodes = sorted(int(n["id"]) for n in self.session.model.get("nodes") or [])
        editor = EditConstraintDialog(name, kind, control,
                                      [1, 1, 0, 0, 0, 1] if kind == "coupling" else None,
                                      nodes, self)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        kind, control, dofs = editor.values()
        result = self.session.create_constraint(name, kind, control, region, dofs=dofs)
        if self._object_result(result, f"已创建约束「{name}」。"):
            self._sync_datums()
            if result.payload.get("warning"):
                QMessageBox.information(self, "主节点需要边界条件", result.payload["warning"])

    def open_constraint_manager(self) -> None:
        self.open_object_manager("constraints")

    # --- 荷载 ---

    def create_load(self, load_type: str | None = None) -> None:
        """Abaqus 式：创建载荷 → 选区域（节点或杆件，按类型）→ 编辑分量。

        ``load_type`` 预选类型（重力按钮用）；重力且没选杆件时区域默认全部杆件。
        """
        cases = [c["name"] for c in self.session.model.get("load_cases") or []]
        prefix = "Gravity" if load_type == "gravity" else "Load"
        dialog = CreateObjectDialog("创建载荷", next_name(prefix, self._feature_names("loads")),
                                    "工况", cases or ["Load-1"], LOAD_TYPES, self,
                                    editable_scope=True)
        if load_type is not None and hasattr(dialog, "kind"):
            dialog.kind.setCurrentIndex([v for _, v in LOAD_TYPES].index(load_type))
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
        initial = self._initial_region(kind)
        if load_type == "gravity" and not initial:
            initial = [int(m["id"]) for m in self.session.model.get("members") or []]
        self.region_pick.start(kind, f"创建载荷「{name}」",
                               lambda region: self._finish_load(name, case, load_type, region),
                               initial)

    def _finish_displacement_bc(self, name: str, step: str, region: dict) -> None:
        """给定位移（支座沉降、强迫转角）：Abaqus 的 Displacement/Rotation 边界条件给非零值。"""
        cases = [c["name"] for c in self.session.model.get("load_cases") or []]
        editor = DisplacementBCDialog(name, cases, self.session.model.get("units", "N-m-Pa"), self)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        fix, displacement, case = editor.values()
        if case not in cases:
            created = self.session.add_load_case(case)
            if not created.ok:
                QMessageBox.warning(self, "工况未创建", self._explain(created.payload))
                return
        note = "" if step == "Initial" else "（给定位移的边界条件在 Initial 创建，随工况激活）"
        result = self.session.create_boundary_condition(name, region, fix=fix,
                                                        displacement=displacement, case=case)
        self._object_result(result, f"已创建给定位移「{name}」，随工况 {case} 施加{note}。")

    def _finish_load(self, name: str, case: str, load_type: str, region: dict) -> None:
        default = {"gravity": [0.0, 0.0, -self.session.units.gravity]}.get(load_type,
                                                                            [0.0, 0.0, 0.0])
        editor = EditLoadDialog(name, load_type, default,
                                self.session.model.get("units", "N-m-Pa"), self,
                                members=self._region_member_ends(load_type, region))
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        values = {LOAD_FIELD[load_type]: editor.vector()}
        if load_type in SPAN_LOADS:
            values.update(editor.span())
        result = self.session.create_load(name, case, load_type, region, **values)
        self._object_result(result, f"已创建载荷「{name}」。")

    def _region_member_ends(self, load_type: str, region: dict) -> dict:
        """沿杆定位荷载的区域杆件 → {编号: (i 端坐标, j 端坐标)}，供对话框换算位置。"""
        if load_type not in SPAN_LOADS:
            return {}
        import features
        from coordinates import model_member_ends
        model = self.session.model
        ids, _problem = features.region_ids(model, region, "members")
        return {mid: model_member_ends(model, mid) for mid in ids}

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
        if kind == "constraints":
            ids = [*ids, int(feature["control"])]
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

    # --- Part 模块：分割、基准点、测量 ---

    def partition_member_ui(self) -> None:
        if not self.session.model.get("members"):
            self.statusBar().showMessage("还没有杆件：请先建立几何。", 5000)
            return
        self.region_pick.start("member", "分割杆件：选一根杆件", self._finish_partition,
                               self._initial_region("member"))

    def _finish_partition(self, region: dict) -> None:
        ids = region.get("members") or []
        if "set" in region or len(ids) != 1:
            QMessageBox.information(self, "分割杆件", "一次分割一根杆件：请只选一根，不用集合。")
            return
        ratio, ok = QInputDialog.getDouble(self, "分割杆件", "顶点位置（从 i 端量起的比例，0~1）：",
                                           0.5, 0.0001, 0.9999, 4)
        if ok:
            got = self.session.partition_member(ids[0], ratio=ratio)
            if self._object_result(got, f"已在杆件 {ids[0]} 上加顶点：节点 {got.payload.get('node')}。"):
                self._sync_datums()

    def datum_point_ui(self) -> None:
        way, ok = QInputDialog.getItem(self, "基准点", "创建方式：",
                                       ["两节点中点（在视口选两个节点）", "输入坐标"], 0, False)
        if not ok:
            return
        name = next_name("Datum", set(self.session.datum_points()))
        if way.startswith("两节点"):
            self.region_pick.start("node", "基准点：选两个节点取中点",
                                   lambda region: self._finish_datum(name, region))
            return
        text, ok = QInputDialog.getText(self, "基准点", "坐标 x, y, z（模型单位）：", text="0, 0, 0")
        if not ok:
            return
        try:
            xyz = [float(v) for v in text.replace("，", ",").split(",")]
        except ValueError:
            QMessageBox.warning(self, "坐标格式不对", "请输入三个数，用逗号分隔，例如 3, 0, 4.5。")
            return
        self._object_result(self.session.create_datum_point(name, xyz=xyz), f"已创建基准点「{name}」。")

    def _finish_datum(self, name: str, region: dict) -> None:
        ids = region.get("nodes") or []
        if len(ids) != 2:
            QMessageBox.information(self, "基准点", "取中点要恰好选两个节点。")
            return
        self._object_result(self.session.create_datum_point(name, midpoint_of=ids),
                            f"已创建基准点「{name}」。")

    def measure_ui(self) -> None:
        way, ok = QInputDialog.getItem(self, "测量", "测什么：",
                                       ["两节点距离", "杆长或两杆夹角"], 0, False)
        if not ok:
            return
        kind = "node" if way.startswith("两节点") else "member"
        self.region_pick.start(kind, f"测量：{way}", self._finish_measure)

    def _finish_measure(self, region: dict) -> None:
        if "set" in region:
            QMessageBox.information(self, "测量", "测量请在视口里直接选对象，不用集合。")
            return
        got = (self.session.measure(nodes=region["nodes"]) if "nodes" in region
               else self.session.measure(members=region["members"]))
        if not got.ok:
            QMessageBox.information(self, "测量", got.payload.get("error", ""))
            return
        p = got.payload
        if "distance" in p:
            dx, dy, dz = p["components"]
            text = (f"节点 {p['nodes'][0]} → {p['nodes'][1]}：距离 {p['distance']:.6g} {p['unit']}"
                    f"（Δx {dx:.6g}，Δy {dy:.6g}，Δz {dz:.6g}）")
        elif "angle_deg" in p:
            text = f"杆件 {p['members'][0]} 与 {p['members'][1]} 的夹角 {p['angle_deg']:.4g}°"
        else:
            text = f"杆件 {p['members'][0]} 长度 {p['length']:.6g} {p['unit']}"
        self.set_prompt(text)
        self.statusBar().showMessage(text, 15000)

    # --- 布种（Mesh 模块） ---

    def _ask_divisions(self, title: str, label: str) -> int | None:
        current = int((self.session.model.get("mesh") or {}).get("divisions") or 1)
        value, ok = QInputDialog.getInt(self, title, label, current, 1, 200, 1)
        return value if ok else None

    def seed_part(self) -> None:
        if not self.session.model.get("members"):
            self.statusBar().showMessage("还没有杆件：请先建立几何，再布种。", 5000)
            return
        n = self._ask_divisions("全局布种", "每根杆件剖成几段：")
        if n is not None:
            self._object_result(self.session.seed_mesh(divisions=n),
                                f"全局布种：每杆 {n} 段。")

    def seed_edges(self) -> None:
        if not self.session.model.get("members"):
            self.statusBar().showMessage("还没有杆件：请先建立几何，再布种。", 5000)
            return
        self.region_pick.start("member", "杆件布种", self._finish_seed,
                               self._initial_region("member"))

    def _finish_seed(self, region: dict) -> None:
        n = self._ask_divisions("杆件布种", "所选杆件各剖成几段（覆盖全局布种）：")
        if n is None:
            return
        target = region["set"] if "set" in region else region["members"]
        self._object_result(self.session.seed_mesh(divisions=n, members=target),
                            f"已给所选杆件布种：每杆 {n} 段。")

    def clear_seeds(self) -> None:
        self._object_result(self.session.seed_mesh(clear=True), "已清除布种。")

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

    def manage_object(self, kind: str, action: str, name: str | None, manager=None,
                      step: str | None = None) -> None:
        """管理器与模型树右键菜单共用的动作。

        ``step`` 是边界条件管理器里选中的分析步列：编辑变成「在该步修改」，
        失活从该步起生效。在创建步或没选步列时，编辑改的是对象本身。
        """
        alias = _KIND_ALIAS[kind]
        session = self.session
        if action == "create":
            if manager is not None:
                manager.accept()
            {"bcs": self.create_bc, "loads": self.create_load,
             "section_assignments": self.assign_section,
             "beam_orientations": self.assign_beam_orientation,
             "constraints": self.create_constraint_ui}[kind]()
            return
        if name is None:
            return
        feature = next(f for f in session.model["features"][kind] if f["name"] == name)
        created = feature.get("step") or "Initial"
        if action == "edit" and kind == "bcs" and step not in (None, created):
            current = (feature.get("modifications") or {}).get(step, feature)
            editor = EditBCDialog(f"{name}（{step}）", current["fix"], current.get("spring"), self)
            if editor.exec() == QDialog.DialogCode.Accepted:
                fix, spring = editor.values()
                self._object_result(session.modify_boundary_condition(name, step, fix=fix,
                                                                       spring=spring),
                                    f"「{name}」在「{step}」已修改。")
        elif action == "edit":
            self._edit_object(kind, feature)
        elif action == "deactivate":
            if step in (None, created):
                QMessageBox.information(self, "失活", "请先在表格里点选要失活的那一步（创建步之后的列）。")
                return
            self._object_result(session.deactivate_boundary_condition(name, step),
                                f"「{name}」从「{step}」起失活。")
        elif action == "activate":
            self._object_result(session.deactivate_boundary_condition(name, None),
                                f"「{name}」已重新激活。")
        elif action == "region":
            if manager is not None:
                manager.accept()
            what = "node" if kind in ("bcs", "constraints") \
                or LOAD_REGION.get(feature.get("type")) == "node" else "member"
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
                                    self.session.model.get("units", "N-m-Pa"), self,
                                    span={k: feature[k] for k in ("a", "b", "w2") if k in feature},
                                    members=self._region_member_ends(feature["type"],
                                                                     feature["region"]))
            if editor.exec() == QDialog.DialogCode.Accepted:
                self._object_result(self.session.edit_feature(
                    alias, name, {field: editor.vector(),
                                  **(editor.span() if feature["type"] in SPAN_LOADS else {})}),
                    f"「{name}」已更新。")
        elif kind == "constraints":
            nodes = sorted(int(n["id"]) for n in self.session.model.get("nodes") or [])
            editor = EditConstraintDialog(name, feature["type"], int(feature["control"]),
                                          feature.get("dofs"), nodes, self)
            if editor.exec() == QDialog.DialogCode.Accepted:
                kind_value, control, dofs = editor.values()
                self._object_result(self.session.edit_feature(
                    alias, name, {"type": kind_value, "control": control, "dofs": dofs}),
                    f"「{name}」已更新。")
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
