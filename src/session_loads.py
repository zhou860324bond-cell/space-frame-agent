"""Session 的荷载域：工况、节点荷载、杆件荷载、强制位移、初应变、自重。

从 agent.py 搬出来的，代码一个字节没改。

单独成文件的理由：荷载是唯一一处"工况"这个概念贯穿始终的地方，而工况的
重复与覆盖语义（同名工况怎么合、自重反复调用怎么替换）正是这个项目踩过坑
的地方——SELF_WEIGHT_NOTE 那个标记就是为此存在的。放在一起才看得出这套
规则是一致的。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from frame3d import self_weight_loads
from model_io import from_dict, validate_payload
from session_base import SELF_WEIGHT_NOTE, ToolResult, _records


class LoadsMixin:
    """荷载：工况、节点/杆件荷载、位移与初应变。见模块 docstring。"""
    #: 荷载强度的参照系。三种写法描述的是**不同的物理荷载**，混用不会报错，
    #: 只会给出一个看着很正常的错数：
    #:
    #:   global     w 是全局分量，强度按**沿杆长**分布。默认，也是内核唯一认识的形式。
    #:   local      w 是杆件局部分量，按沿杆长分布。风压垂直于杆轴时用它，
    #:              不必自己拆 sinθ/cosθ——那一步手算最容易反号。
    #:   projected  w 是全局分量，强度按**水平投影长度**分布。斜屋面的雪载、
    #:              活载按规范就是这么给的（kN/m 是"平面上每米"）。
    #:
    #: 后两种在**写入模型前**就换算成 global，模型里存的永远是内核认识的那一种。
    #: 这样求解器、编译器、内力回算一概不必知道有这回事；代价是几何变了之后
    #: 荷载不会自己跟着变，需要重新施加——与 add_self_weight 的约定一致。
    LOAD_REFERENCES = ("global", "local", "projected")

    def _to_global_intensity(self, member: dict, w: list[float],
                             reference: str) -> tuple[list[float], dict]:
        """把 local / projected 的强度换算成全局分量、沿杆长分布。

        返回 (全局强度, 换算说明)。说明会原样进 ToolResult——换算过程必须
        看得见，否则用户拿到一个和自己输入不一样的数字却不知道为什么。
        """
        from frame3d import local_axes

        nodes = {int(n["id"]): n for n in self.model.get("nodes") or []}
        ni, nj = nodes[int(member["i"])], nodes[int(member["j"])]
        pi = np.array([ni["x"], ni["y"], ni["z"]], dtype=float)
        pj = np.array([nj["x"], nj["y"], nj["z"]], dtype=float)
        length, rot = local_axes(pi, pj, member.get("ref_vector"))
        vector = np.asarray(w, dtype=float)

        if reference == "local":
            # rot 的行是局部轴的全局分量，所以局部→全局是 rot 的转置
            out = rot.T @ vector
            return [float(v) for v in out], {
                "reference": "local",
                "given_local": [float(v) for v in vector],
                "as_global": [round(float(v), 6) for v in out],
                "explain": "按杆件局部轴给出的强度已换算成全局分量；"
                           "局部 x 沿 i→j，局部 y 由参考向量定（水平杆件指向全局 +Z）。",
            }

        # projected：强度是"每米水平投影"，沿杆长的强度要乘 L_水平 / L
        horizontal = float(np.linalg.norm((pj - pi)[:2]))
        if horizontal <= 0.0:
            return list(vector), {
                "reference": "projected",
                "explain": "杆件是竖直的，水平投影为零——按水平投影给的荷载在它"
                           "身上没有意义，强度原样采用。请确认这是你要的。",
                "warning": True,
            }
        scale = horizontal / length
        out = vector * scale
        return [float(v) for v in out], {
            "reference": "projected",
            "given_per_horizontal_metre": [float(v) for v in vector],
            "member_length": round(length, 6),
            "horizontal_projection": round(horizontal, 6),
            "scale": round(scale, 6),
            "as_global_per_member_metre": [round(float(v), 6) for v in out],
            "explain": f"按水平投影给的强度乘以 {scale:.6g}（= 水平投影 / 杆长）"
                       "换算成沿杆长分布。总合力不变，这正是投影荷载的定义。",
        }


    @_records
    def set_load_cases(self, cases: list[dict], combos: list[dict] | None = None) -> ToolResult:
        """一次性定义全部荷载工况与组合，覆盖此前的荷载定义。

        多工况务必用这个，不要把几个工况的荷载先加起来当单一工况——
        那样组合结果虽对，却拿不到各工况的分项内力，也无法做包络。
        """
        if not cases:
            return ToolResult(False, {"error": "至少要给一个工况"})
        from copy import deepcopy

        candidate = deepcopy(self.model)
        for key in ("nodal_loads", "member_loads", "member_spans", "settlements"):
            candidate.pop(key, None)
        candidate["load_cases"] = deepcopy(cases)
        candidate["combos"] = deepcopy(combos or [])
        errors = validate_payload(candidate)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "荷载定义未写入，原模型保持不变"})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"cases": [c.get("name") for c in cases],
                                 "combos": [c.get("name") for c in (combos or [])]})

    @_records
    def add_load_case(self, name: str) -> ToolResult:
        """新建一个空荷载工况。"""
        from copy import deepcopy

        name = str(name).strip()
        if not name:
            return ToolResult(False, {"error": "工况名称不能为空"})
        candidate = deepcopy(self.model)
        candidate.setdefault("load_cases", [])
        if any(c.get("name") == name for c in candidate["load_cases"]):
            return ToolResult(False, {"error": f"工况 {name!r} 已存在"})
        candidate["load_cases"].append({"name": name, "nodal_loads": [],
                                          "member_loads": [], "member_spans": [],
                                          "settlements": []})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"case": name, "cases": [c["name"] for c in self.model["load_cases"]]})

    @_records
    def set_nodal_load(self, node_id: int, load: list[float],
                       case_name: str | None = None,
                       name: str | None = None) -> ToolResult:
        """设置节点集中力。load 是 [Fx,Fy,Fz,Mx,My,Mz]，单位为 N 与 N·当前长度单位。
        传全零列表表示删除该节点的荷载。"""
        from copy import deepcopy

        if len(load) != 6:
            return ToolResult(False, {"error": "load 必须是六个数 [Fx,Fy,Fz,Mx,My,Mz]"})
        try:
            values = [float(value) for value in load]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "集中力六个分量必须都是数字"})
        if not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "集中力分量必须是有限数"})
        node_id = int(node_id)
        if node_id not in {int(n["id"]) for n in self.model.get("nodes", [])}:
            return ToolResult(False, {"error": f"节点 {node_id} 不存在"})
        candidate = deepcopy(self.model)
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                         "member_loads": [], "member_spans": [],
                                         "settlements": []}]
        case_name = case_name or candidate["load_cases"][0]["name"]
        case = next((c for c in candidate["load_cases"] if c["name"] == case_name), None)
        if case is None:
            return ToolResult(False, {"error": f"工况 {case_name!r} 不存在"})
        same_target = [entry for entry in case.get("nodal_loads", [])
                       if int(entry["node"]) == node_id]
        if name is None and len(same_target) > 1:
            return ToolResult(False, {
                "error": f"节点 {node_id} 上有多条载荷，请给出要编辑的载荷名称",
                "names": [entry.get("name") for entry in same_target]})
        old = same_target[0] if same_target else None
        load_name = str(name or (old or {}).get("name") or
                        f"CF-Node-{node_id}").strip()
        if not load_name:
            return ToolResult(False, {"error": "载荷名称不能为空"})
        entries = [entry for entry in case.get("nodal_loads", [])
                   if not (entry.get("name") == load_name
                           or (old is entry and not entry.get("name")))]
        for key in ("member_loads", "member_spans"):
            case[key] = [entry for entry in case.get(key, [])
                         if entry.get("name") != load_name]
        if any(value != 0 for value in values):
            entries.append({"name": load_name, "node": node_id, "load": values})
        case["nodal_loads"] = entries
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": load_name,
                                 "node": node_id, "case": case_name, "load": values})

    @_records
    def set_member_load(self, member_id, load: list[float],
                        case_name: str | None = None,
                        name: str | None = None,
                        reference: str = "global") -> ToolResult:
        """设置杆件均布荷载。load 是三个数 [wx,wy,wz]，单位为 N/当前长度单位。
        传全零列表表示删除该杆件的荷载。

        ``member_id`` 可以是一个编号、一串编号，或**集合名**（甚至混写）。
        一句"给顶层所有梁加 5 kN/m"以前要拆成几十次调用，而漏掉其中两根
        没有任何人会发现——工具报成功、校验通过、结果看着也正常。

        ``reference`` 见 LOAD_REFERENCES：斜梁上的雪载、活载按规范是"每米水平
        投影"，用 projected；风压垂直于杆轴，用 local。两者都在写入前换算成
        全局分量，换算过程原样写进返回值。
        """
        from copy import deepcopy

        if len(load) != 3:
            return ToolResult(False, {"error": "load 必须是三个数 [wx,wy,wz]"})
        try:
            values = [float(value) for value in load]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "均布荷载三个分量必须都是数字"})
        if not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "均布荷载分量必须是有限数"})
        reference = str(reference)
        if reference not in self.LOAD_REFERENCES:
            return ToolResult(False, {
                "error": f"reference 只能是 {list(self.LOAD_REFERENCES)}"})

        try:
            ids = self._expand_members(member_id)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        if not ids:
            return ToolResult(False, {"error": "至少选择一根杆件"})
        known = {int(m["id"]): m for m in self.model.get("members") or []}
        missing = sorted(set(ids) - set(known))
        if missing:
            return ToolResult(False, {"error": f"杆件 {missing} 不存在"})

        candidate = deepcopy(self.model)
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                         "member_loads": [], "member_spans": [],
                                         "settlements": []}]
        case_name = case_name or candidate["load_cases"][0]["name"]
        case = next((c for c in candidate["load_cases"]
                     if c["name"] == case_name), None)
        if case is None:
            return ToolResult(False, {"error": f"工况 {case_name!r} 不存在"})

        single = len(ids) == 1
        applied: list[dict[str, Any]] = []
        for mid in ids:
            member_values = list(values)
            conversion = None
            if reference != "global" and any(member_values):
                member_values, conversion = self._to_global_intensity(
                    known[mid], member_values, reference)

            same_target = [entry for entry in case.get("member_loads", [])
                           if int(entry["member"]) == mid]
            if single and name is None and len(same_target) > 1:
                return ToolResult(False, {
                    "error": f"杆件 {mid} 上有多条均布荷载，请给出要编辑的载荷名称",
                    "names": [entry.get("name") for entry in same_target]})
            old = same_target[0] if same_target else None
            if single:
                load_name = str(name or (old or {}).get("name")
                                or f"Line-Member-{mid}").strip()
            else:
                # 多根时每根必须有**自己的**名字：共用一个名字的话，
                # 按名替换会把前面几根刚写进去的又删掉，只剩最后一根。
                load_name = str(f"{name}-{mid}" if name
                                else f"Line-Member-{mid}").strip()
            if not load_name:
                return ToolResult(False, {"error": "载荷名称不能为空"})

            entries = [entry for entry in case.get("member_loads", [])
                       if not (entry.get("name") == load_name
                               or (old is entry and not entry.get("name")))]
            for key in ("nodal_loads", "member_spans"):
                case[key] = [entry for entry in case.get(key, [])
                             if entry.get("name") != load_name]
            if any(value != 0 for value in member_values):
                entries.append({"name": load_name, "member": mid,
                                "w": member_values})
            case["member_loads"] = entries
            applied.append({"name": load_name, "member": mid,
                            "load": member_values,
                            **({"conversion": conversion} if conversion else {})})

        self.model = candidate
        self._invalidate()
        if single:
            # 单根时的返回值一个字节没变——既有调用方和测试都靠它
            return ToolResult(True, {**applied[0], "case": case_name})
        return ToolResult(True, {
            "case": case_name, "members": ids, "count": len(ids),
            "applied": applied,
            "note": "每根杆件各自一条命名荷载，可按名单独编辑或删除。",
        })

    #: 导荷方式。板把面荷载传给周边梁的方式取决于板的长宽比：
    #:
    #:   one_way        单向板。板只沿一个方向传力，梁按**从属宽度**承受均布。
    #:   two_way_short  双向板的**短边**梁：三角形，跨中峰值 q·Lx/2。
    #:   two_way_long   双向板的**长边**梁：梯形，两端各升 Lx/2 后进入平台。
    #:
    #: 后两种是按"45° 分角线划分板块"得到的经典分配，混凝土教材里的标准做法。
    LOAD_PATHS = ("one_way", "two_way_short", "two_way_long")

    @_records
    def apply_area_load(self, members, q: float, width: float,
                        load_path: str = "one_way",
                        case_name: str | None = None,
                        name: str | None = None,
                        direction: list[float] | None = None) -> ToolResult:
        """把**面荷载**（力/面积）导成梁上的线荷载。

        真实输入永远是 kN/m²——楼面恒载 3.5、雪载 0.5、活载 2.0。而求解器只
        认 kN/m，中间那步"从属宽度、双向板分配"以前全靠手算，**算错了结果
        看着完全正常**：量级对、图形也像那么回事，只是总重差一截。

        ``q`` 是面荷载强度（正值表示大小），默认沿全局 −Z（重力方向），
        可用 ``direction`` 改。``width`` 在 one_way 下是从属宽度，
        在两种 two_way 下是板的**短跨** Lx。

        三角形与梯形都用 partial_trapezoid 精确生成，**不走等效均布**——
        等效均布只保证跨中弯矩相等，支座附近的剪力是另一回事。
        """
        from copy import deepcopy

        try:
            q = float(q)
            width = float(width)
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "q 与 width 必须是数字"})
        if not (np.isfinite(q) and np.isfinite(width)):
            return ToolResult(False, {"error": "q 与 width 必须是有限数"})
        if width <= 0:
            return ToolResult(False, {"error": "width 必须为正"})
        load_path = str(load_path)
        if load_path not in self.LOAD_PATHS:
            return ToolResult(False, {
                "error": f"load_path 只能是 {list(self.LOAD_PATHS)}"})

        unit = np.array([0.0, 0.0, -1.0]) if direction is None else np.asarray(
            direction, dtype=float)
        norm = float(np.linalg.norm(unit))
        if norm <= 0:
            return ToolResult(False, {"error": "direction 不能是零向量"})
        unit = unit / norm

        try:
            ids = self._expand_members(members)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        if not ids:
            return ToolResult(False, {"error": "至少选择一根杆件"})
        known = {int(m["id"]): m for m in self.model.get("members") or []}
        absent = sorted(set(ids) - set(known))
        if absent:
            return ToolResult(False, {"error": f"杆件 {absent} 不存在"})

        nodes = {int(n["id"]): n for n in self.model.get("nodes") or []}
        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})

        base = str(name or "Area").strip() or "Area"
        report: list[dict[str, Any]] = []
        fresh: list[dict[str, Any]] = []
        for mid in ids:
            member = known[mid]
            ni, nj = nodes[int(member["i"])], nodes[int(member["j"])]
            length = float(np.linalg.norm(np.array(
                [nj[k] - ni[k] for k in ("x", "y", "z")], dtype=float)))
            pieces, note = self._area_load_pieces(q, width, length, load_path,
                                                  unit)
            if pieces is None:
                return ToolResult(False, {"error": note, "member": mid,
                                          "member_length": round(length, 6)})
            for index, piece in enumerate(pieces, 1):
                fresh.append({"name": f"{base}-{mid}-{index}", "member": mid,
                              **piece})
            report.append({"member": mid, "length": round(length, 6),
                           "pieces": len(pieces), "note": note})

        names = {item["name"] for item in fresh}
        for key in ("nodal_loads", "member_loads", "member_spans"):
            case[key] = [entry for entry in case.get(key, [])
                         if entry.get("name") not in names]
        case["member_spans"] = list(case.get("member_spans", [])) + fresh

        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        total = abs(q) * width * (len(ids) if load_path == "one_way" else 0)
        return ToolResult(True, {
            "case": chosen, "load_path": load_path, "members": ids,
            "count": len(ids), "entries": len(fresh), "detail": report,
            "analysis_ready": not errors, "warnings": errors,
            **({"total_force_per_member_hint": round(total / max(len(ids), 1), 6)}
               if load_path == "one_way" else {}),
            "note": "三角形与梯形用 partial_trapezoid 精确生成，不是等效均布——"
                    "等效均布只保证跨中弯矩相等，支座附近的剪力是另一回事。",
        })

    def _area_load_pieces(self, q: float, width: float, length: float,
                          load_path: str, unit) -> tuple[list | None, str]:
        """一根梁上该生成哪几段荷载。返回 (段列表, 说明)；不成立时段列表为 None。"""
        if load_path == "one_way":
            w = (q * width) * unit
            return ([{"kind": "uniform", "w1": [float(v) for v in w]}],
                    f"单向板从属宽度 {width:g}，线荷载 = q×宽度")

        peak = (q * width / 2.0) * unit          # 两种双向板峰值都是 q·Lx/2
        zero = [0.0, 0.0, 0.0]
        if load_path == "two_way_short":
            half = length / 2.0
            return ([{"kind": "partial_trapezoid", "w1": zero,
                      "w2": [float(v) for v in peak], "a": 0.0, "b": half},
                     {"kind": "partial_trapezoid",
                      "w1": [float(v) for v in peak], "w2": zero,
                      "a": half, "b": length}],
                    f"双向板短边梁：三角形，跨中峰值 q·Lx/2 = {abs(q) * width / 2:g}")

        rise = width / 2.0
        if 2.0 * rise >= length:
            return (None,
                    f"长边梁的梯形需要 Lx/2 = {rise:g} 在两端各升一段，"
                    f"而杆长只有 {length:g}——这根梁其实不比短跨长，"
                    "说明它不是长边，或者 width 传的不是短跨。")
        return ([{"kind": "partial_trapezoid", "w1": zero,
                  "w2": [float(v) for v in peak], "a": 0.0, "b": rise},
                 {"kind": "partial", "w1": [float(v) for v in peak],
                  "a": rise, "b": length - rise},
                 {"kind": "partial_trapezoid", "w1": [float(v) for v in peak],
                  "w2": zero, "a": length - rise, "b": length}],
                f"双向板长边梁：梯形，两端各升 {rise:g}，平台 q·Lx/2 = "
                f"{abs(q) * width / 2:g}")

    #: 分项系数与组合值系数。**这些数字是有出处的，不是默认值。**
    #:
    #: GB 50068-2018《建筑结构可靠性设计统一标准》把承载能力极限状态的
    #: 分项系数从 1.2/1.4 提到了 1.3/1.5；不少既有项目仍按 2012 版校核，
    #: 所以两套都留着，由调用方明确选。
    #:
    #: ψ_c（组合值系数）与 ψ_q（准永久值系数）**随建筑类别变**：这里给的是
    #: 一般民用建筑（住宅、办公）的常见取值。商业、库房、机房的活载系数
    #: 不同，必须由调用方覆盖——用错了不会报错，只会让组合悄悄偏小。
    COMBINATION_STANDARDS = {
        "GB50068-2018": {
            "gamma_G_unfavourable": 1.3, "gamma_G_favourable": 1.0,
            "gamma_Q": 1.5,
            "psi_c": {"live": 0.7, "wind": 0.6, "snow": 0.7, "crane": 0.7},
            "psi_q": {"live": 0.5, "wind": 0.0, "snow": 0.2, "crane": 0.6},
        },
        "GB50009-2012": {
            "gamma_G_unfavourable": 1.2, "gamma_G_favourable": 1.0,
            "gamma_Q": 1.4,
            "psi_c": {"live": 0.7, "wind": 0.6, "snow": 0.7, "crane": 0.7},
            "psi_q": {"live": 0.5, "wind": 0.0, "snow": 0.2, "crane": 0.6},
        },
    }
    VARIABLE_KINDS = ("live", "wind", "snow", "crane")

    @_records
    def generate_combinations(self, dead=None, live=None, wind=None,
                              snow=None, crane=None,
                              standard: str = "GB50068-2018",
                              include_serviceability: bool = True,
                              psi_c: dict | None = None,
                              replace: bool = True) -> ToolResult:
        """按规范生成荷载组合，写进模型的 combos。

        以前 combos 要手写系数字典，于是"漏了一个组合"这种错完全看不出来——
        包络只在给定的组合里取极值，少一个就是少一个，而结果看着完全正常。

        生成三类：

        * **基本组合**（承载能力极限状态）：每个可变荷载**轮流**当控制荷载，
          取 γ_Q；其余可变荷载取 γ_Q·ψ_c。轮流这一步最容易漏——只算
          "活载控制"而不算"风控制"，风控制的那些杆件就永远查不出来。
        * **恒载有利的基本组合**：有风荷载时另生成一组 γ_G=1.0。风吸把柱子
          往上拔时恒载是有利的，这时候用 1.3 反而不保守。
          软件判不了"哪根杆件上恒载算有利"——那是逐杆逐截面的事，
          所以两组都生成，交给包络逐点去挑。
        * **标准组合与准永久组合**（正常使用极限状态）：验挠度、裂缝用。

        各参数接受单个工况名或名字列表。
        """
        from copy import deepcopy

        spec = self.COMBINATION_STANDARDS.get(str(standard))
        if spec is None:
            return ToolResult(False, {
                "error": f"standard 只能是 {list(self.COMBINATION_STANDARDS)}"})

        def names(value) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            return [str(v) for v in value]

        groups = {"live": names(live), "wind": names(wind),
                  "snow": names(snow), "crane": names(crane)}
        dead_cases = names(dead)
        known = {c.get("name") for c in self.model.get("load_cases") or []}
        missing = sorted({n for n in dead_cases + [x for g in groups.values()
                                                   for x in g]} - known)
        if missing:
            return ToolResult(False, {
                "error": f"工况 {missing} 不存在", "available": sorted(known)})
        if not dead_cases and not any(groups.values()):
            return ToolResult(False, {"error": "至少要给一个工况"})

        factors_c = dict(spec["psi_c"])
        factors_c.update({str(k): float(v) for k, v in (psi_c or {}).items()})
        gu, gf = spec["gamma_G_unfavourable"], spec["gamma_G_favourable"]
        gq = spec["gamma_Q"]

        combos: list[dict] = []

        def add(name: str, factors: dict[str, float], basis: str) -> None:
            if factors:
                combos.append({"name": name, "factors": factors,
                               "basis": basis})

        variable = [(kind, case) for kind in self.VARIABLE_KINDS
                    for case in groups[kind]]

        # --- 基本组合：每个可变荷载轮流控制 ---
        for kind, lead in variable:
            factors = {case: gu for case in dead_cases}
            factors[lead] = factors.get(lead, 0.0) + gq
            for other_kind, other in variable:
                if other == lead:
                    continue
                psi = factors_c.get(other_kind, 0.7)
                factors[other] = factors.get(other, 0.0) + gq * psi
            add(f"基本-{lead}控制", factors,
                f"{standard} 基本组合，{lead} 为控制荷载："
                f"γ_G={gu}、γ_Q={gq}，其余可变荷载取 γ_Q·ψ_c")
        if dead_cases and not variable:
            add("基本-仅恒载", {case: gu for case in dead_cases},
                f"{standard} 基本组合，只有永久荷载")

        # --- 恒载有利：风吸上拔时用 ---
        for case in groups["wind"]:
            factors = {d: gf for d in dead_cases}
            factors[case] = factors.get(case, 0.0) + gq
            add(f"基本-{case}控制(恒载有利)", factors,
                f"{standard} 基本组合，永久荷载按有利取 γ_G={gf}。"
                "风吸把构件往上拔时恒载是有利的，这时用 1.3 反而不保守")

        # --- 正常使用极限状态 ---
        if include_serviceability:
            for kind, lead in variable:
                factors = {case: 1.0 for case in dead_cases}
                factors[lead] = factors.get(lead, 0.0) + 1.0
                for other_kind, other in variable:
                    if other == lead:
                        continue
                    factors[other] = factors.get(other, 0.0) + factors_c.get(
                        other_kind, 0.7)
                add(f"标准-{lead}控制", factors,
                    f"{standard} 标准组合，验挠度用")
            quasi = {case: 1.0 for case in dead_cases}
            for kind, case in variable:
                psi = spec["psi_q"].get(kind, 0.5)
                if psi:
                    quasi[case] = quasi.get(case, 0.0) + psi
            add("准永久", quasi, f"{standard} 准永久组合，验长期变形用")

        candidate = deepcopy(self.model)
        existing = [] if replace else list(candidate.get("combos") or [])
        keep = [c for c in existing
                if c.get("name") not in {x["name"] for x in combos}]
        candidate["combos"] = keep + [
            {"name": c["name"], "factors": c["factors"]} for c in combos]
        errors = validate_payload(candidate)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "组合未写入，原模型保持不变"})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "standard": standard, "count": len(combos),
            "combos": [{"name": c["name"], "factors": c["factors"],
                        "basis": c["basis"]} for c in combos],
            "gamma": {"G_unfavourable": gu, "G_favourable": gf, "Q": gq},
            "psi_c": factors_c,
            "note": "ψ 系数随建筑类别变，这里给的是一般民用建筑的常见取值。"
                    "商业、库房、机房不同，用错不会报错，只会让组合悄悄偏小——"
                    "请核对后用 psi_c 覆盖。组合一经写入，包络与强度校核"
                    "自动以它们为对象。",
        })

    #: 活载布置方式。混凝土规范里的"最不利荷载位置"简化做法：
    #:
    #:   full      满布。支座负弯矩通常由它控制
    #:   odd       奇数跨布置。跨中正弯矩最大值常出在这一组
    #:   even      偶数跨布置。与 odd 互补
    #:   adjacent  相邻两跨布置。某个支座的负弯矩峰值出在这一组，
    #:             跨数多时它会生成 N−1 个工况，按需开
    LIVE_PATTERNS = ("full", "odd", "even", "adjacent")

    def _spans_of(self, member_ids: list[int]) -> tuple[list, str]:
        """把一组梁按**跨**归类，返回 [(跨区间, [杆件号]), ...] 与说明。

        同一跨的梁在多层框架里分布在各层，但它们在水平方向占的区间相同，
        所以按"沿主导水平轴的起止坐标"归组，再从左到右排序。

        归组结果会原样写进返回值——**这一步是几何推断，用户必须能核对**。
        杆件方向不一致（比如混进了柱或另一方向的梁）时归出来的跨会很怪，
        与其猜不如让人看见。
        """
        nodes = {int(n["id"]): n for n in self.model.get("nodes") or []}
        members = {int(m["id"]): m for m in self.model.get("members") or []}
        groups: dict[tuple, list[int]] = {}
        axis_votes = {"x": 0, "y": 0}
        for mid in member_ids:
            m = members[mid]
            ni, nj = nodes[int(m["i"])], nodes[int(m["j"])]
            dx, dy = abs(nj["x"] - ni["x"]), abs(nj["y"] - ni["y"])
            axis = "x" if dx >= dy else "y"
            axis_votes[axis] += 1
        main = "x" if axis_votes["x"] >= axis_votes["y"] else "y"
        for mid in member_ids:
            m = members[mid]
            ni, nj = nodes[int(m["i"])], nodes[int(m["j"])]
            lo, hi = sorted((round(ni[main], 6), round(nj[main], 6)))
            groups.setdefault((lo, hi), []).append(mid)
        ordered = [(key, sorted(groups[key])) for key in sorted(groups)]
        note = (f"按沿 {main.upper()} 轴的起止坐标归出 {len(ordered)} 跨，"
                "从左到右编号。同一跨在各层的梁归在一起——"
                "布置是整片楼层同时切换的，不是逐层切换。")
        return ordered, note

    @_records
    def generate_live_patterns(self, members, load: list[float],
                               patterns=("full", "odd", "even"),
                               prefix: str = "LL",
                               replace: bool = True) -> ToolResult:
        """生成活载的**最不利布置**工况：满布、隔跨、相邻跨。

        连续梁与框架的跨中正弯矩不是满布时最大，而是**隔跨布置**时最大；
        支座负弯矩则常由满布或相邻跨控制。只算满布会把跨中弯矩算小——
        而且算小多少取决于跨数与刚度比，看不出来。

        生成的是**工况**不是组合。接着把它们一起传给 generate_combinations
        的 live 参数，每一种布置都会轮流当控制荷载。

        这是规范里的简化做法。严格的"最不利荷载位置"要画影响线逐点找，
        这里不冒充那个。
        """
        from copy import deepcopy

        if len(load) != 3:
            return ToolResult(False, {"error": "load 必须是三个数 [wx,wy,wz]"})
        try:
            values = [float(v) for v in load]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "荷载三个分量必须都是数字"})
        if not all(np.isfinite(v) for v in values):
            return ToolResult(False, {"error": "荷载分量必须是有限数"})
        wanted = [str(p) for p in (patterns or ())]
        unknown = [p for p in wanted if p not in self.LIVE_PATTERNS]
        if unknown:
            return ToolResult(False, {
                "error": f"布置方式只能取 {list(self.LIVE_PATTERNS)}，"
                         f"收到 {unknown}"})
        if not wanted:
            return ToolResult(False, {"error": "至少选一种布置方式"})

        try:
            ids = self._expand_members(members)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        known = {int(m["id"]) for m in self.model.get("members") or []}
        absent = sorted(set(ids) - known)
        if absent:
            return ToolResult(False, {"error": f"杆件 {absent} 不存在"})
        if not ids:
            return ToolResult(False, {"error": "至少选择一根杆件"})

        spans, note = self._spans_of(ids)
        if len(spans) < 2 and any(p != "full" for p in wanted):
            return ToolResult(False, {
                "error": f"只归出 {len(spans)} 跨，隔跨与相邻跨布置无从谈起。"
                         "单跨结构直接用满布即可。",
                "spans": [{"range": list(k), "members": v} for k, v in spans]})

        plans: list[tuple[str, list[int]]] = []
        for pattern in wanted:
            if pattern == "full":
                plans.append((f"{prefix}-满布",
                              [m for _, group in spans for m in group]))
            elif pattern == "odd":
                plans.append((f"{prefix}-奇数跨",
                              [m for k, (_, group) in enumerate(spans)
                               for m in group if k % 2 == 0]))
            elif pattern == "even":
                plans.append((f"{prefix}-偶数跨",
                              [m for k, (_, group) in enumerate(spans)
                               for m in group if k % 2 == 1]))
            else:                                   # adjacent
                for k in range(len(spans) - 1):
                    plans.append((f"{prefix}-相邻{k + 1}{k + 2}",
                                  sorted(spans[k][1] + spans[k + 1][1])))

        candidate = deepcopy(self.model)
        cases = list(candidate.get("load_cases") or [])
        made = {name for name, _ in plans}
        if replace:
            cases = [c for c in cases if c.get("name") not in made]
        elif {c.get("name") for c in cases} & made:
            return ToolResult(False, {
                "error": f"工况 {sorted({c.get('name') for c in cases} & made)} "
                         "已存在；要覆盖请传 replace=True"})
        for name, targets in plans:
            cases.append({"name": name, "nodal_loads": [], "member_spans": [],
                          "settlements": [],
                          "member_loads": [{"name": f"{name}-{mid}",
                                            "member": mid, "w": values}
                                           for mid in targets]})
        candidate["load_cases"] = cases
        errors = validate_payload(candidate)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "工况未写入，原模型保持不变"})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "spans": [{"range": list(k), "members": v} for k, v in spans],
            "span_count": len(spans),
            "cases": [{"name": name, "members": targets,
                       "loaded_spans": len({k for k, (_, g) in enumerate(spans)
                                            if set(g) & set(targets)})}
                      for name, targets in plans],
            "count": len(plans),
            "grouping_note": note,
            "note": "生成的是**工况**不是组合。把这些名字一起传给 "
                    "generate_combinations 的 live 参数，每种布置都会轮流当"
                    "控制荷载。跨中正弯矩常由隔跨布置控制，只算满布会算小。",
        })

    def _load_case_in(self, candidate: dict[str, Any],
                      case_name: str | None) -> tuple[dict[str, Any] | None, str]:
        """在候选模型中取分析步；增量载荷工具共用，避免默认工况逻辑漂移。"""
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                          "member_loads": [], "member_spans": [],
                                          "settlements": []}]
        chosen = case_name or candidate["load_cases"][0]["name"]
        case = next((item for item in candidate["load_cases"]
                     if item.get("name") == chosen), None)
        return case, str(chosen)

    @_records
    def set_member_span_load(self, member_id, kind: str, w1: list[float],
                             w2: list[float] | None = None,
                             a: float | None = None,
                             case_name: str | None = None,
                             name: str | None = None,
                             b: float | None = None,
                             reference: str = "global") -> ToolResult:
        """创建梯形、三角形、杆中集中力或部分跨均布，按名称编辑而不是重复叠加。

        ``partial`` 用 a、b 圈出受载区间 [a, b]，强度写在 w1 里。
        砌体墙、局部堆载、只压半跨的活载都是这个形状；以前只能拿几个集中力
        硬凑，凑出来的弯矩图在荷载区内是折线而不是抛物线。
        """
        from copy import deepcopy

        kind = str(kind)
        if kind not in {"uniform", "trapezoid", "point", "partial",
                        "partial_trapezoid", "moment"}:
            return ToolResult(False, {
                "error": "kind 只能是 uniform / trapezoid / point / partial / "
                         "partial_trapezoid / moment"})
        try:
            start = [float(value) for value in w1]
            end = [float(value) for value in (w2 if w2 is not None else w1)]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "w1/w2 必须是三个数字"})
        if len(start) != 3 or len(end) != 3 or not all(
                np.isfinite(value) for value in start + end):
            return ToolResult(False, {"error": "w1/w2 必须是三个有限数字"})
        try:
            ids = self._expand_members(member_id)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        if not ids:
            return ToolResult(False, {"error": "至少选择一根杆件"})
        known = {int(item["id"]): item for item in self.model.get("members") or []}
        absent = sorted(set(ids) - set(known))
        if absent:
            return ToolResult(False, {"error": f"杆件 {absent} 不存在"})
        if len(ids) > 1 and kind in ("point", "partial", "moment",
                                     "partial_trapezoid"):
            # a / b 是**沿杆长的绝对位置**，一组长短不一的杆件共用同一个 a
            # 没有意义：短杆上可能超出杆长，长杆上位置完全不对应。
            # 与其按比例自作主张，不如说清楚。
            return ToolResult(False, {
                "error": f"{kind} 的位置 a/b 是沿杆长的绝对距离，"
                         "一组杆件共用同一个位置没有意义（短杆会超界、"
                         "长杆位置也对不上）。请逐根施加。",
                "members": ids})
        member_id = int(ids[0])
        member = known[member_id]
        # 下面这一大段（位置校验、参照系换算、写入）是按**一根**杆件写的。
        # 多根时在同一份 candidate 上逐根走一遍，不能只取第一根——那正是
        # 这一轮反复在修的"静默丢弃"。
        position = None
        if kind in ("point", "moment"):
            try:
                position = float(a)
            except (TypeError, ValueError):
                return ToolResult(False, {
                    "error": f"{kind} 必须给出距 i 端位置 a"})
            nodes = {int(node["id"]): node for node in self.model.get("nodes") or []}
            ni, nj = nodes[int(member["i"])], nodes[int(member["j"])]
            length = float(np.linalg.norm(np.array(
                [nj[key] - ni[key] for key in ("x", "y", "z")], dtype=float)))
            if not np.isfinite(position) or not 0 <= position <= length:
                return ToolResult(False, {
                    "error": f"位置 a 必须在 0 到杆长 {length:g} 之间（当前模型长度单位）"})
        # 名字不能叫 end —— 上文的 end 是 w2 向量，覆盖掉会让梯形荷载
        # 静默丢掉 j 端强度。
        span_end = None
        if kind in ("partial", "partial_trapezoid"):
            nodes = {int(node["id"]): node for node in self.model.get("nodes") or []}
            ni, nj = nodes[int(member["i"])], nodes[int(member["j"])]
            length = float(np.linalg.norm(np.array(
                [nj[key] - ni[key] for key in ("x", "y", "z")], dtype=float)))
            try:
                position, span_end = float(a), float(b)
            except (TypeError, ValueError):
                return ToolResult(False, {
                    "error": "部分跨均布必须同时给出起点 a 与终点 b"})
            if not (np.isfinite(position) and np.isfinite(span_end)):
                return ToolResult(False, {"error": "a 与 b 必须是有限数"})
            if kind == "partial_trapezoid" and w2 is None:
                return ToolResult(False, {
                    "error": "区间梯形荷载必须给出 w2（区间终点处的强度）；"
                             "两端相同的话用 kind=partial 更省事"})
            if not 0 <= position < span_end <= length:
                # 不允许 a==b：那是个零长度的"荷载"，合力为零却照样占着一条
                # 记录，以后查"为什么合力对不上"会白费很多时间。
                return ToolResult(False, {
                    "error": f"必须满足 0 ≤ a < b ≤ 杆长 {length:g}，"
                             f"收到 a={position:g}、b={span_end:g}"})
        reference = str(reference)
        if reference not in self.LOAD_REFERENCES:
            return ToolResult(False, {
                "error": f"reference 只能是 {list(self.LOAD_REFERENCES)}"})
        conversion = None
        if reference != "global":
            # 集中力（point）的 w1 是**力**不是强度，投影换算对它没有意义：
            # 一个集中力不会因为杆件倾斜而变大变小。局部坐标则照样适用。
            if kind in ("point", "moment") and reference == "projected":
                return ToolResult(False, {
                    "error": "杆中集中力是力不是强度，没有「按水平投影分布」这回事。"
                             "要换方向用 reference=local，要减小数值请自己算。"})
            if any(start) or any(end):
                start, conversion = self._to_global_intensity(
                    member, start, reference)
                if kind in ("trapezoid", "partial_trapezoid"):
                    end, _ = self._to_global_intensity(member, end, reference)
                else:
                    end = list(start)

        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})

        single = len(ids) == 1
        applied: list[dict[str, Any]] = []
        for target in ids:
            target = int(target)
            member_start, member_end = list(start), list(end)
            member_conversion = conversion
            if reference != "global" and not single:
                # 每根杆件的倾角不同，换算系数也不同——不能沿用第一根的
                if any(member_start) or any(member_end):
                    member_start, member_conversion = self._to_global_intensity(
                        known[target], member_start, reference)
                    if kind in ("trapezoid", "partial_trapezoid"):
                        member_end, _ = self._to_global_intensity(
                            known[target], member_end, reference)
                    else:
                        member_end = list(member_start)
            load_name = str(
                (name if single else (f"{name}-{target}" if name else None))
                or f"{kind.title()}-Member-{target}").strip()
            if not load_name:
                return ToolResult(False, {"error": "载荷名称不能为空"})
            applied.append({"name": load_name, "member": target,
                            "start": member_start, "end": member_end,
                            "conversion": member_conversion})

        for item in applied:
            load_name = item["name"]
            target = item["member"]
            member_start, member_end = item["start"], item["end"]
            entries = [entry for entry in case.get("member_spans", [])
                       if entry.get("name") != load_name]
            for key in ("nodal_loads", "member_loads"):
                case[key] = [entry for entry in case.get(key, [])
                             if entry.get("name") != load_name]
            values_for_delete = member_start + (
                member_end if kind in ("trapezoid", "partial_trapezoid") else [])
            if any(value != 0 for value in values_for_delete):
                entry: dict[str, Any] = {"name": load_name, "member": target,
                                         "kind": kind, "w1": member_start}
                if kind == "trapezoid":
                    entry["w2"] = member_end
                if kind in ("point", "moment"):
                    entry["a"] = position
                if kind in ("partial", "partial_trapezoid"):
                    entry["a"] = position
                    entry["b"] = span_end
                if kind == "partial_trapezoid":
                    entry["w2"] = member_end
                entries.append(entry)
            case["member_spans"] = entries

        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        if single:
            first = applied[0]
            return ToolResult(True, {
                "name": first["name"], "member": first["member"],
                "case": chosen, "kind": kind,
                "analysis_ready": not errors, "warnings": errors,
                **({"conversion": first["conversion"]}
                   if first["conversion"] else {})})
        return ToolResult(True, {
            "case": chosen, "kind": kind,
            "members": [item["member"] for item in applied],
            "count": len(applied),
            "applied": [{k: v for k, v in item.items() if k != "end"}
                        for item in applied],
            "analysis_ready": not errors, "warnings": errors,
            "note": "每根杆件各自一条命名荷载，可按名单独编辑或删除。"})

    @_records
    def set_prescribed_displacement(self, node_id: int, d: list[float],
                                    case_name: str | None = None,
                                    name: str | None = None) -> ToolResult:
        """创建分析步内给定位移；只有 Initial 中已约束的方向会参与求解。"""
        from copy import deepcopy

        try:
            values = [float(value) for value in d]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "d 必须是六个数字"})
        if len(values) != 6 or not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "d 必须是六个有限数字"})
        node_id = int(node_id)
        if node_id not in {int(node["id"]) for node in self.model.get("nodes") or []}:
            return ToolResult(False, {"error": f"节点 {node_id} 不存在"})
        bc_name = str(name or f"Displacement-Node-{node_id}").strip()
        if not bc_name:
            return ToolResult(False, {"error": "边界条件名称不能为空"})
        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})
        entries = [entry for entry in case.get("settlements", [])
                   if entry.get("name") != bc_name and int(entry["node"]) != node_id]
        if any(value != 0 for value in values):
            entries.append({"name": bc_name, "node": node_id, "d": values})
        case["settlements"] = entries
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": bc_name, "node": node_id,
                                 "case": chosen, "d": values,
                                 "analysis_ready": not errors, "warnings": errors,
                                 "note": "给定位移属于分析步边界条件，不计入外力。"})

    @_records
    def set_member_strain(self, member_id: int,
                          lack_of_fit: float | None = None,
                          delta_t: float | None = None,
                          case_name: str | None = None,
                          name: str | None = None,
                          gradient_t: float | None = None) -> ToolResult:
        """装配误差与温度变化（讲义 §3-9 五、六）。

        讲义把温度应力"转化为装配内力问题"，这里就照这个思路做成一个入口：
        两者都只是给杆件一个初始长度变化，差别仅在于 Δl 是直接给的还是由
        α·ΔT 算出来的。
        """
        from copy import deepcopy

        member_id = int(member_id)
        members = {int(m["id"]): m for m in self.model.get("members") or []}
        if member_id not in members:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})
        for label, value in (("lack_of_fit", lack_of_fit), ("delta_t", delta_t),
                            ("gradient_t", gradient_t)):
            if value is not None and not np.isfinite(float(value)):
                return ToolResult(False, {"error": f"{label} 必须是有限数"})
        if gradient_t:
            # 梯度算 κ=α·ΔT/h 要截面高度，而 h 只有按尺寸建的截面才有
            section = next(
                (item for item in self.model.get("sections") or []
                 if item["name"] == members[member_id]["section"]), None)
            if not (section or {}).get("cy"):
                return ToolResult(False, {
                    "error": f"截面 {members[member_id]['section']!r} 没有极端纤维"
                             "距离 cy，算不出温度梯度引起的曲率 κ=α·ΔT/h",
                    "hint": "用按尺寸定义的截面（矩形/工字形/圆管/实心圆），"
                            "或直接给出 cy"})
        if delta_t or gradient_t:
            material = next(
                (item for item in self.model.get("materials") or []
                 if item["name"] == members[member_id]["material"]), None)
            if not (material or {}).get("alpha"):
                return ToolResult(False, {
                    "error": f"材料 {members[member_id]['material']!r} 没有线膨胀系数 alpha，"
                             "算不了温度应力",
                    "hint": "在 define_materials_and_sections 里给该材料补 alpha（1/℃），"
                            "钢约 1.2e-5"})

        entry_name = str(name or f"Strain-Member-{member_id}").strip()
        if not entry_name:
            return ToolResult(False, {"error": "名称不能为空"})
        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})
        entries = [entry for entry in case.get("member_strains", [])
                   if entry.get("name") != entry_name
                   and int(entry["member"]) != member_id]
        payload = {"name": entry_name, "member": member_id}
        if lack_of_fit:
            payload["lack_of_fit"] = float(lack_of_fit)
        if delta_t:
            payload["delta_t"] = float(delta_t)
        if gradient_t:
            payload["gradient_t"] = float(gradient_t)
        if len(payload) > 2:
            entries.append(payload)
        case["member_strains"] = entries
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "name": entry_name, "member": member_id, "case": chosen,
            "lack_of_fit": lack_of_fit, "delta_t": delta_t,
            "gradient_t": gradient_t,
            "analysis_ready": not errors, "warnings": errors,
            "note": "初应变不是外力：杆件能自由伸缩时轴力为零，被约束住才产生"
                    "内力。**均匀温度与温度梯度是两回事**：delta_t 让杆整体"
                    "伸缩、产生轴力，gradient_t（截面上下温差）让杆想要弯、"
                    "产生弯矩，κ=α·ΔT/h。静定结构两者都不产生内力。"})

    @_records
    def add_self_weight(self, case: str | None = None,
                        factor: float = 1.0) -> ToolResult:
        """按 ρ·A·g 生成各杆自重，作为均布荷载追加到某个工况。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "模型不合法，先修好再加自重"})
        frame = from_dict(self.model)
        loads = self_weight_loads(frame, factor=factor)
        if not loads:
            missing = sorted({m["material"] for m in self.model["members"]})
            return ToolResult(False, {
                "error": "没有一种材料定义了 density，自重为零",
                "hint": f"给材料 {missing} 加上 density（kg/m³，钢约 7850）"
                        "再调一次；define_materials_and_sections 可以带这个字段"})
        cases = self.model.setdefault("load_cases", [])
        if not cases:
            cases.append({"name": case or "SW"})
        name = case or cases[0]["name"]
        target = next((c for c in cases if c.get("name") == name), None)
        if target is None:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况",
                                      "available": [c.get("name") for c in cases]})
        spans = target.setdefault("member_spans", [])
        # 反复调用会把自重叠加好几遍，先把上一次生成的清掉
        kept = [e for e in spans if e.get("note") != SELF_WEIGHT_NOTE]
        replaced = len(spans) - len(kept)
        spans[:] = kept
        for mid, load in sorted(loads.items()):
            spans.append({"member": mid, **load.to_dict(),
                          "note": SELF_WEIGHT_NOTE})
        self._invalidate()
        total = sum(abs(load.w1[2]) for load in loads.values())
        return ToolResult(True, {
            "case": name, "members": len(loads),
            "replaced_previous": replaced,
            "total_vertical_kN_per_m": round(total * self.units.line_load_scale, 6),
            "note": "自重已写成显式均布荷载，可在模型里看到。改截面后需重新调用。"})
