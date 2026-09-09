"""草稿模型：允许「还没画完」的中间状态。

**这是手工建模缺的那块地基。**

现在的模型只有一种状态：合法。`validate_model` 一票否决，任何不合法的
中间状态都进不来。可手工画图天然就要经过不合法的中间状态——点第一个
节点时，它悬空；画第一根杆时，还没有支座；删一根杆重画时，模型一瞬间
是断开的。拿求解器的标准去卡编辑动作，用户第一步就被拦住。

FEM-Python 的做法是把类型分成两级，我们照搬这个分法：

* **草稿类型（这里）** —— 宽松。悬空节点、没有支座、结构断成两片，
  都只是提示，不阻止你继续画。
* **领域类型（`model_io` + `frame3d.check_model`）** —— 严格。
  能进这一级的，一定能求解。

对应两级诊断：

* `editing_diagnostics()` —— 编辑过程中显示，语气是「还差什么」。
* `finish_diagnostics()` —— 提交时显示，语气是「为什么不能算」。

两者用的是同一套检查，差别只在**同一个问题在哪一级算错**。分成两套
各写一遍的话，迟早会出现「编辑时说没问题、提交时说不行」这种自相矛盾。

有两类问题例外，**在草稿里就直接拒绝**：零长度杆件和重复杆件。
它们不是「还没画完」，是画错了；留到提交时才说，用户已经不记得
是哪一步画的了。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

# 节点重合判定的默认容差（米）。**鼠标点出来的坐标永远不会精确相等**，
# 不合并的话会得到两个相距 1e-9 的节点，杆件各连一个，
# 结构看着连着、算起来是散的——刚度矩阵奇异，而图上完全看不出来
SNAP_TOL = 1e-6

# 这几项草稿不碰，原样带着走：草稿只管几何与拓扑
CARRIED = ("units", "materials", "sections", "nodal_loads", "member_loads",
           "member_spans", "settlements", "load_cases", "combos", "sets")


@dataclass(frozen=True)
class Note:
    """一条诊断。

    `target` 让界面能把问题**指到图上**——「节点 7 悬空」远不如
    图上那个点闪一下有用。
    """

    level: str                     # "info" | "warn" | "error"
    message: str
    target: tuple[str, int] | None = None   # ("node" | "member", 编号)

    def __str__(self) -> str:      # 打印时就是给人看的那句话
        return self.message


class DraftError(ValueError):
    """草稿层直接拒绝的操作：画错了，不是还没画完。"""


@dataclass
class FrameDraft:
    """可以处于半成品状态的模型。

    内部就是模型字典的几何部分，不额外造一套数据结构——多一层映射，
    就多一个「界面显示的和模型里的不一致」的机会。
    """

    nodes: list[dict[str, Any]] = field(default_factory=list)
    members: list[dict[str, Any]] = field(default_factory=list)
    supports: list[dict[str, Any]] = field(default_factory=list)
    carried: dict[str, Any] = field(default_factory=dict)

    # --- 与正式模型互转 ---

    @classmethod
    def from_model(cls, model: dict[str, Any]) -> "FrameDraft":
        """从当前模型开一份草稿。**深拷贝**——草稿改坏了要能整个丢掉。"""
        m = copy.deepcopy(model or {})
        return cls(
            nodes=m.get("nodes") or [],
            members=m.get("members") or [],
            supports=m.get("supports") or [],
            carried={k: m[k] for k in CARRIED if k in m},
        )

    def to_model(self) -> dict[str, Any]:
        """装回模型字典。不做校验——校验是 `Session.apply_draft` 的事。"""
        out = dict(self.carried)
        out["nodes"] = copy.deepcopy(self.nodes)
        out["members"] = copy.deepcopy(self.members)
        out["supports"] = copy.deepcopy(self.supports)
        return out

    def snapshot(self) -> dict[str, Any]:
        """存一份现状，供「取消」回退。"""
        return copy.deepcopy({"nodes": self.nodes, "members": self.members,
                              "supports": self.supports})

    def restore(self, snap: dict[str, Any]) -> None:
        self.nodes = copy.deepcopy(snap["nodes"])
        self.members = copy.deepcopy(snap["members"])
        self.supports = copy.deepcopy(snap["supports"])

    # --- 查 ---

    def node(self, nid: int) -> dict[str, Any] | None:
        return next((n for n in self.nodes if int(n["id"]) == int(nid)), None)

    def member(self, mid: int) -> dict[str, Any] | None:
        return next((m for m in self.members if int(m["id"]) == int(mid)), None)

    def _next(self, entries: Iterable[dict[str, Any]]) -> int:
        ids = [int(e["id"]) for e in entries]
        return (max(ids) + 1) if ids else 1

    def find_node_at(self, x: float, y: float, z: float,
                     tol: float = SNAP_TOL) -> int | None:
        """这个位置上已经有节点了吗。"""
        for n in self.nodes:
            if (math.isclose(n["x"], x, abs_tol=tol)
                    and math.isclose(n["y"], y, abs_tol=tol)
                    and math.isclose(n["z"], z, abs_tol=tol)):
                return int(n["id"])
        return None

    def length(self, mid: int) -> float | None:
        m = self.member(mid)
        if m is None:
            return None
        a, b = self.node(int(m["i"])), self.node(int(m["j"]))
        if a is None or b is None:
            return None
        return math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))

    # --- 改 ---

    def add_node(self, x: float, y: float, z: float,
                 snap: bool = True, tol: float = SNAP_TOL) -> int:
        """加一个节点。**同一位置已有节点时返回原来那个**，不新建。

        这是画图建模最关键的一条：画完柱再画梁，梁的起点就落在柱顶。
        不吸附的话，两根杆各连一个几乎重合的节点，结构在图上是连着的，
        在矩阵里是散的。
        """
        if snap:
            hit = self.find_node_at(x, y, z, tol)
            if hit is not None:
                return hit
        nid = self._next(self.nodes)
        self.nodes.append({"id": nid, "x": float(x), "y": float(y), "z": float(z)})
        return nid

    def move_node(self, nid: int, x: float, y: float, z: float) -> None:
        n = self.node(nid)
        if n is None:
            raise DraftError(f"节点 {nid} 不存在")
        n["x"], n["y"], n["z"] = float(x), float(y), float(z)

    def delete_node(self, nid: int) -> list[int]:
        """删节点，连带删掉挂在它上面的杆件与支座。返回被连带删掉的杆件。

        连带删是唯一诚实的做法：留下引用了不存在节点的杆件，
        问题会在下一步以完全无关的面貌出现。
        """
        if self.node(nid) is None:
            raise DraftError(f"节点 {nid} 不存在")
        dropped = [int(m["id"]) for m in self.members
                   if int(m["i"]) == int(nid) or int(m["j"]) == int(nid)]
        gone = set(dropped)
        self.members = [m for m in self.members if int(m["id"]) not in gone]
        self.nodes = [n for n in self.nodes if int(n["id"]) != int(nid)]
        self.supports = [s for s in self.supports if int(s["node"]) != int(nid)]
        return dropped

    def add_member(self, i: int, j: int, section: str = "",
                   material: str | None = None, **extra: Any) -> int:
        """连一根杆。零长度和重复连接**当场拒绝**——那是画错，不是没画完。"""
        if self.node(i) is None or self.node(j) is None:
            missing = [n for n in (i, j) if self.node(n) is None]
            raise DraftError(f"节点 {missing} 不存在")
        if int(i) == int(j):
            raise DraftError("杆件的两端不能是同一个节点")
        a, b = self.node(i), self.node(j)
        if math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"])) < SNAP_TOL:
            raise DraftError(f"节点 {i} 与节点 {j} 位置重合，连出来的杆件长度为零")
        key = (min(int(i), int(j)), max(int(i), int(j)))
        for m in self.members:
            if (min(int(m["i"]), int(m["j"])), max(int(m["i"]), int(m["j"]))) == key:
                raise DraftError(f"节点 {i} 与节点 {j} 之间已经有杆件 {m['id']} 了")
        known_materials = self.carried.get("materials") or []
        mat = material
        if mat is None and self.members:
            mat = self.members[0].get("material")
        if mat is None and len(known_materials) == 1:
            mat = known_materials[0].get("name")
        mat = mat or ""
        mid = self._next(self.members)
        entry: dict[str, Any] = {"id": mid, "i": int(i), "j": int(j),
                                 "section": section, "material": mat}
        entry.update({k: v for k, v in extra.items() if v})
        self.members.append(entry)
        return mid

    def delete_member(self, mid: int) -> None:
        if self.member(mid) is None:
            raise DraftError(f"杆件 {mid} 不存在")
        self.members = [m for m in self.members if int(m["id"]) != int(mid)]

    def set_support(self, nid: int, fix: list[int] | None) -> None:
        """设支座。`fix` 传 None 或空表示取消约束。"""
        if self.node(nid) is None:
            raise DraftError(f"节点 {nid} 不存在")
        if fix and (len(fix) != 6 or any(v not in (0, 1) for v in fix)):
            raise DraftError("约束必须是六个 0 或 1，顺序 ux uy uz rx ry rz")
        self.supports = [s for s in self.supports if int(s["node"]) != int(nid)]
        if fix and any(fix):
            self.supports.append({"node": int(nid), "fix": [int(v) for v in fix]})
        self.supports.sort(key=lambda s: s["node"])

    # --- 清理 ---

    def coincident_nodes(self, tol: float = 1e-6) -> list[list[int]]:
        """位置重合的节点分组。**导入或多次生成后最常见的隐患。**"""
        groups: list[list[int]] = []
        taken: set[int] = set()
        for a in self.nodes:
            ida = int(a["id"])
            if ida in taken:
                continue
            same = [ida]
            for b in self.nodes:
                idb = int(b["id"])
                if idb == ida or idb in taken:
                    continue
                if math.dist((a["x"], a["y"], a["z"]),
                             (b["x"], b["y"], b["z"])) <= tol:
                    same.append(idb)
            if len(same) > 1:
                groups.append(sorted(same))
                taken.update(same)
        return groups

    def merge_coincident(self, tol: float = 1e-6) -> dict[str, Any]:
        """把重合节点并成一个，杆件改连保留下来的那个。

        并完可能出现重复杆件（两根杆本来连着两个重合点），一并去掉——
        重复杆件会让刚度翻倍，而结果看着完全正常。
        """
        groups = self.coincident_nodes(tol)
        if not groups:
            return {"merged": 0, "removed_members": []}
        remap: dict[int, int] = {}
        for g in groups:
            keep = g[0]
            for other in g[1:]:
                remap[other] = keep
        for m in self.members:
            m["i"] = remap.get(int(m["i"]), int(m["i"]))
            m["j"] = remap.get(int(m["j"]), int(m["j"]))
        # 支座跟着并；同一个保留节点上有多个支座定义时，取「约束更多」的那个
        merged_sup: dict[int, list[int]] = {}
        for s in self.supports:
            nid = remap.get(int(s["node"]), int(s["node"]))
            cur = merged_sup.get(nid)
            fix = [int(v) for v in s["fix"]]
            merged_sup[nid] = fix if cur is None else [max(a, b) for a, b in zip(cur, fix)]
        self.supports = [{"node": k, "fix": v} for k, v in sorted(merged_sup.items())]

        gone = set(remap)
        self.nodes = [n for n in self.nodes if int(n["id"]) not in gone]

        removed: list[int] = []
        seen: dict[tuple[int, int], int] = {}
        kept: list[dict[str, Any]] = []
        for m in self.members:
            key = (min(int(m["i"]), int(m["j"])), max(int(m["i"]), int(m["j"])))
            if int(m["i"]) == int(m["j"]) or key in seen:
                removed.append(int(m["id"]))
                continue
            seen[key] = int(m["id"])
            kept.append(m)
        self.members = kept
        return {"merged": len(gone), "removed_members": removed,
                "groups": groups}

    # --- 诊断 ---

    def _problems(self) -> list[tuple[str, Note]]:
        """一次查完，每条附上「它属于哪一类」。

        **两级诊断共用这一份**。分开写两套，早晚会漂成
        「编辑时说没问题、提交时说不行」。这里的分类键决定了
        同一个问题在编辑期是提示还是在提交期是错误。
        """
        found: list[tuple[str, Note]] = []

        known_sections = {s["name"] for s in self.carried.get("sections") or []}
        known_materials = {m["name"] for m in self.carried.get("materials") or []}
        used: set[int] = set()

        for m in self.members:
            mid = int(m["id"])
            used.update((int(m["i"]), int(m["j"])))
            if not m.get("section"):
                found.append(("undefined", Note(
                    "error", f"杆件 {mid} 尚未指派截面", ("member", mid))))
            elif m.get("section") not in known_sections:
                found.append(("undefined", Note(
                    "error", f"杆件 {mid} 用了未定义的截面「{m.get('section')}」",
                    ("member", mid))))
            if not m.get("material"):
                found.append(("undefined", Note(
                    "error", f"杆件 {mid} 尚未指派材料", ("member", mid))))
            elif m.get("material") not in known_materials:
                found.append(("undefined", Note(
                    "error", f"杆件 {mid} 用了未定义的材料「{m.get('material')}」",
                    ("member", mid))))

        for n in self.nodes:
            nid = int(n["id"])
            if nid not in used:
                found.append(("dangling", Note(
                    "info", f"节点 {nid} 还没有连上任何杆件", ("node", nid))))

        if not self.nodes:
            found.append(("empty", Note("info", "还没有任何节点")))
        elif not self.members:
            found.append(("empty", Note("info", "已有节点，但还没连出杆件")))

        if not self.supports:
            found.append(("no_support", Note("info", "还没有设置任何支座")))

        parts = self._components()
        if self.members and len(parts) > 1:
            sizes = "、".join(str(len(p)) for p in parts)
            found.append(("split", Note(
                "warn", f"结构分成了 {len(parts)} 片（各含 {sizes} 个节点），"
                        "彼此没有连接")))
        for part in parts:
            if self.members and not any(int(s["node"]) in part for s in self.supports):
                sample = sorted(part)[:3]
                found.append(("floating", Note(
                    "warn", f"有一片结构（如节点 {sample}）上没有任何支座，"
                            "求解时会当作机构",
                    ("node", sample[0]) if sample else None)))
        return found

    def _components(self) -> list[set[int]]:
        """连通分量。**断成两片是画图时最难自己看出来的错误**——
        图上两片可能挨得很近，看起来就是连着的。"""
        # **只看连上了杆件的节点。** 把孤立节点也算成一片的话，
        # 刚点了两个点还没连线时会报「结构分成了 2 片」——
        # 那不是断开，那是还没开始画
        adj: dict[int, set[int]] = {}
        for m in self.members:
            i, j = int(m["i"]), int(m["j"])
            if self.node(i) is None or self.node(j) is None:
                continue
            adj.setdefault(i, set()).add(j)
            adj.setdefault(j, set()).add(i)
        seen: set[int] = set()
        parts: list[set[int]] = []
        for start in adj:
            if start in seen:
                continue
            stack, part = [start], set()
            while stack:
                cur = stack.pop()
                if cur in part:
                    continue
                part.add(cur)
                stack.extend(adj[cur] - part)
            seen |= part
            parts.append(part)
        return parts

    def editing_diagnostics(self) -> list[Note]:
        """编辑过程中的提示。**没画完不是错误。**"""
        return [note for _, note in self._problems()]

    def finish_diagnostics(self) -> list[Note]:
        """提交前的检查：这些没解决就算不出来。

        和编辑期同一批问题，只是级别升上来了——悬空节点在画的时候
        无所谓，提交时它就是一行零刚度。
        """
        raise_to_error = {"dangling", "empty", "no_support", "floating",
                          "undefined"}
        out: list[Note] = []
        for kind, note in self._problems():
            if kind in raise_to_error:
                out.append(Note("error", note.message, note.target))
            elif note.level == "error":
                out.append(note)
        return out

    def can_finish(self) -> bool:
        return not self.finish_diagnostics()
