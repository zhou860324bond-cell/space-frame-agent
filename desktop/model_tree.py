"""左侧模型树。

Abaqus/CAE 的左侧树是它信息架构的骨干：模型有什么、算出了什么，一眼看全。
这里是同一个东西——材料、截面、几何、约束、荷载工况、组合，加上算完之后的
结果分支。

树只**显示**，不改模型。双击某一项发出信号，由主窗口决定打开哪个对话框——
树不该知道对话框长什么样。

分支顺序与 Abaqus/CAE 的模型树一致：材料、截面、截面指派、集合、几何、分析步、
边界条件、荷载、组合、结果。截面指派、边界条件、荷载下面是**对象**（见
src/features.py）：单击在视口里高亮它的区域，右键是管理器的同一组操作——
编辑、编辑区域、复制、改名、抑制或恢复、删除；在分支上右键可以创建或打开管理器。
旧模型里没有对象来源的条目归在「直接定义」下，照原样显示。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem

from . import theme

# 双击某类节点要打开什么，由主窗口接住这个动作名再决定开哪个对话框
ACTION = Qt.ItemDataRole.UserRole + 1
PAYLOAD = Qt.ItemDataRole.UserRole + 2


def _branch_key(text: str) -> str:
    """顶层分支的身份。标题里带着计数（"材料（2）"），数量一变就成了另一个
    名字，展开状态会莫名其妙地丢——所以按括号前的部分认。"""
    return text.split("（")[0].strip()


class ModelTree(QTreeWidget):
    """模型 + 结果的树。"""

    activated_item = Signal(str, object)          # (action, payload)
    #: 对象操作：(类别, 动作, 对象名或 None)。动作同管理器：create / manager / edit /
    #: region / copy / rename / suppress / resume / delete
    feature_action = Signal(str, str, object)
    #: 单击一个对象：(类别, 对象名)，主窗口据此高亮区域
    feature_selected = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(14)
        self.setAlternatingRowColors(False)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setStyleSheet(f"""
            QTreeWidget {{
                background: {theme.PANEL_ALT};
                border: none;
                outline: none;
            }}
            QTreeWidget::item {{
                padding: 3px 4px;
            }}
            QTreeWidget::item:hover {{
                background: {theme.PANEL_HOVER};
            }}
            QTreeWidget::item:selected {{
                background: {theme.SELECTION};
                color: {theme.INK};
            }}
            QTreeWidget::branch {{
                background: {theme.PANEL_ALT};
            }}
        """)
        self.itemDoubleClicked.connect(self._on_double_click)
        self.itemClicked.connect(self._on_click)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        action = item.data(0, ACTION)
        if action == "feature":
            kind, name, _suppressed = item.data(0, PAYLOAD)
            self.feature_action.emit(kind, "edit", name)
        elif action:
            self.activated_item.emit(action, item.data(0, PAYLOAD))

    def _on_click(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, ACTION) == "feature":
            kind, name, _suppressed = item.data(0, PAYLOAD)
            self.feature_selected.emit(kind, name)

    def context_actions(self, item: QTreeWidgetItem) -> list[tuple[str, str]]:
        """某一项的右键菜单：(动作, 文字)。单独成函数，便于无头测试。"""
        action = item.data(0, ACTION)
        if action == "feature_branch":
            return [("create", "创建…"), ("manager", "管理器…")]
        if action == "feature":
            _kind, _name, suppressed = item.data(0, PAYLOAD)
            return [("edit", "编辑…"), ("region", "编辑区域…"), ("copy", "复制…"),
                    ("rename", "改名…"),
                    ("resume", "恢复") if suppressed else ("suppress", "抑制"),
                    ("delete", "删除")]
        return []

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        actions = self.context_actions(item)
        if not actions:
            return
        payload = item.data(0, PAYLOAD)
        if item.data(0, ACTION) == "feature_branch":
            kind, name = payload, None
        else:
            kind, name = payload[0], payload[1]
        menu = QMenu(self)
        for action, text in actions:
            menu.addAction(text).triggered.connect(
                lambda _=False, a=action: self.feature_action.emit(kind, a, name))
        menu.exec(self.viewport().mapToGlobal(pos))

    # --- 构树 ---

    @staticmethod
    def _child(parent, text: str, action: str | None = None,
               payload: Any = None, tip: str | None = None) -> QTreeWidgetItem:
        node = QTreeWidgetItem(parent, [text])
        # 面板窄的时候文字会被省略号吃掉——"COLUMN　A=0.012　Iz…"看不出任何东西。
        # 所以每一项都挂上完整文本作为悬停提示：**显示可以省略，数据不可以**。
        node.setToolTip(0, tip or text)
        if action:
            node.setData(0, ACTION, action)
            node.setData(0, PAYLOAD, payload)
        return node

    def _feature_branch(self, label: str, kind: str, features: dict):
        items = features.get(kind) or []
        branch = self._child(self, f"{label}（{len(items)}）", "feature_branch", kind,
                             tip=f"{label}：右键创建或打开管理器；单击对象高亮区域")
        for feature in items:
            self._feature_item(branch, kind, feature)
        return branch

    def _feature_item(self, parent, kind: str, feature: dict) -> QTreeWidgetItem:
        region = feature["region"]
        if "set" in region:
            where = f"集合「{region['set']}」"
        else:
            noun = "节点" if "nodes" in region else "杆件"
            where = f"{len(region.get('nodes') or region.get('members') or [])} 个{noun}"
        if kind == "section_assignments":
            what = f"{feature['section']}/{feature['material']}"
        elif kind == "beam_orientations":
            what = "n1=(" + ",".join(f"{v:g}" for v in feature["n1"]) + ")"
        elif kind == "bcs":
            what = "".join(str(v) for v in feature["fix"])
        else:
            what = feature["type"]
        suppressed = bool(feature.get("suppressed"))
        text = f"{feature['name']}　{what}　{where}" + ("　（已抑制）" if suppressed else "")
        node = self._child(parent, text, "feature", (kind, feature["name"], suppressed))
        if suppressed:
            node.setForeground(0, QColor(theme.INK_MUTED))
        return node

    def rebuild(self, session, result=None) -> None:
        """整棵重建。刚架模型小，重建比增量维护简单可靠得多。

        重建会丢掉展开状态，所以先记下来再恢复——不然每求解一次
        树就全折起来，用起来很别扭。

        记的是**收起过哪些**，不是展开过哪些。这两者对"新出现的分支"处理
        完全不同：求解之后才出现的"结果"分支，按前一种做法会因为"上次不在
        展开集合里"而默认收起——而它恰恰是用户此刻最想看的东西。
        新分支一律展开，只有用户亲手收起过的才保持收起。
        """
        collapsed = {_branch_key(self.topLevelItem(k).text(0))
                     for k in range(self.topLevelItemCount())
                     if not self.topLevelItem(k).isExpanded()}
        self.clear()
        model = session.model

        mats = self._child(self, f"材料（{len(model.get('materials') or [])}）",
                           "materials")
        for m in model.get("materials") or []:
            rho = m.get("density")
            extra = "　".join(
                f"{label}={m[key]:.6g}" for key, label in
                (("yield_stress", "屈服应力"), ("allow_tension", "许用拉应力"),
                 ("allow_compression", "许用压应力"), ("alpha", "线膨胀系数"))
                if m.get(key))
            self._child(mats, f"{m['name']}　E={m['E']:.4g}　ν={m['nu']}"
                              + (f"　ρ={rho:g}" if rho else "　（无密度）"),
                        tip=f"{m['name']}　E={m['E']:.6g}　ν={m['nu']}"
                            + (f"　ρ={rho:g}" if rho else "　未给密度，无法计入自重")
                            + (f"　{extra}" if extra else ""))

        secs = self._child(self, f"截面（{len(model.get('sections') or [])}）",
                           "sections")
        for s in model.get("sections") or []:
            detail = "　".join(
                f"{k}={s[k]:.6g}" for k in ("A", "Iy", "Iz", "J", "cy", "cz")
                if s.get(k) is not None)
            self._child(secs, f"{s['name']}　A={s['A']:.4g}　Iz={s['Iz']:.4g}",
                        tip=f"{s['name']}　{detail}")

        features = model.get("features") or {}
        self._feature_branch("截面指派", "section_assignments", features)
        self._feature_branch("梁方向", "beam_orientations", features)

        sets = model.get("sets") or {}
        set_node = self._child(self, f"集合（{len(sets)}）", "sets")
        for name, entry in sets.items():
            bits = [f"{len(entry[key])} {label}" for key, label in
                    (("nodes", "节点"), ("members", "杆件")) if entry.get(key)]
            self._child(set_node, f"{name}　" + "，".join(bits), "sets", name)

        geo = self._child(self, "几何", "geometry")
        self._child(geo, f"节点　{len(model.get('nodes') or [])}", "nodes")
        released = [m for m in model.get("members") or [] if m.get("releases")]
        self._child(geo, f"杆件　{len(model.get('members') or [])}"
                         + (f"（{len(released)} 根带端部释放）" if released else ""),
                    "members")

        steps = model.get("steps") or []
        step_node = self._child(self, f"分析步（{len(steps) + 1}）", "steps")
        self._child(step_node, "Initial", "steps")
        for step in steps:
            self._child(step_node, f"{step['name']}　{step.get('analysis', 'linear')}",
                        "steps", step["name"])

        sup = self._feature_branch("边界条件", "bcs", features)
        direct = [s for s in model.get("supports") or [] if not s.get("owners")]
        if direct:
            sup = self._child(sup, f"直接定义（{len(direct)}）", "supports")
        kinds: dict[str, list[int]] = {}
        for entry in direct:
            from viz_symbols import classify_support
            # 用 get 而不是 []：一条缺 fix 的支座（例如识别草稿只给了 kind）
            # 不该让整棵模型树重建失败，那会连带整个界面刷新一起崩。
            mask = entry.get("fix")
            kind = (classify_support(mask)
                    if isinstance(mask, (list, tuple)) and len(mask) == 6
                    else "约束未知")
            k = f"{entry.get('name') or '未命名'} · {kind}"
            kinds.setdefault(k, []).append(int(entry["node"]))
        for k, nodes in kinds.items():
            group = self._child(sup, f"{k} × {len(nodes)}")
            for node_id in nodes:
                self._child(group, f"节点 {node_id}", "support_object", node_id)

        cases = model.get("load_cases") or []
        loads = self._child(self, f"荷载（{len(cases)} 工况）", "feature_branch", "loads")
        for c in cases:
            bits = []
            for key, label in (("nodal_loads", "节点"), ("member_loads", "均布"),
                               ("member_spans", "梯形/集中"),
                               ("settlements", "沉降")):
                if c.get(key):
                    bits.append(f"{label}{len(c[key])}")
            case_node = self._child(
                loads, f"{c['name']}　" + ("·".join(bits) or "（空）"),
                "loads", c["name"])
            for feature in features.get("loads") or []:
                if feature["case"] == c["name"]:
                    self._feature_item(case_node, "loads", feature)
            for entry in c.get("nodal_loads") or []:
                if entry.get("owner"):
                    continue
                name = entry.get("name") or "未命名节点力"
                self._child(case_node, f"{name} · 节点 {entry['node']}",
                            "load_object", {"case": c["name"], "kind": "node",
                                            "id": int(entry["node"])})
            for entry in c.get("member_loads") or []:
                if entry.get("owner"):
                    continue
                name = entry.get("name") or "未命名均布力"
                self._child(case_node, f"{name} · 杆件 {entry['member']} · 均布",
                            "load_object", {"case": c["name"], "kind": "member",
                                            "id": int(entry["member"])})
            span_labels = {"uniform": "均布", "trapezoid": "梯形", "point": "集中"}
            for entry in c.get("member_spans") or []:
                if entry.get("owner"):
                    continue
                name = entry.get("name") or "未命名杆间力"
                shape = span_labels.get(entry.get("kind"), str(entry.get("kind")))
                self._child(case_node,
                            f"{name} · 杆件 {entry['member']} · {shape}",
                            "load_object", {"case": c["name"], "kind": "member",
                                            "id": int(entry["member"])})
            for entry in c.get("settlements") or []:
                name = entry.get("name") or "未命名给定位移"
                self._child(case_node, f"{name} · 节点 {entry['node']} · 给定位移",
                            "load_object", {"case": c["name"], "kind": "node",
                                            "id": int(entry["node"])})

        combos = model.get("combos") or []
        if combos:
            node = self._child(self, f"荷载组合（{len(combos)}）", "loads")
            for c in combos:
                terms = " ".join(f"{v:+g}×{k}" for k, v in c["factors"].items())
                self._child(node, f"{c['name']} = {terms}")

        if result is not None and result.ok:
            res = self._child(self, "结果", "results")
            for name in result.payload["cases"]:
                self._child(res, name, "results", name)

        steps = len(session.history)
        if steps:
            self._child(self, f"建模过程（{steps} 步）", "history")

        for k in range(self.topLevelItemCount()):
            item = self.topLevelItem(k)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, QColor(theme.INK))
            item.setExpanded(_branch_key(item.text(0)) not in collapsed)
