"""画图的状态机：一次点击在当前模式下意味着什么。

**这里不碰 Qt，也不碰视口。** 输入是「工作平面上的一个点」，
输出是「草稿变成了什么样、状态栏该说什么」。画图逻辑里最容易错的
恰恰是这一层——第二次点击点在了第一次那个节点上、点空白处该不该
自动建节点、画到一半按 Esc 退到哪里——这些都该能在没有窗口的情况下测。

模式只有四个。**再多用户就记不住了**，而这四个覆盖了手工建模的全部：
放点、连杆、设支座、删。

画图过程中的撤销退**一次点击**，不是退整段。整段的撤销在提交之后
由 `Session.undo` 负责——两种撤销的粒度不同是故意的：画的时候你想退
上一根杆，画完之后你想退掉整次修改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from draft import DraftError, FrameDraft
from workplane import WorkPlane

MODES = ("node", "member", "support", "delete")

MODE_LABELS = {
    "node": "放节点",
    "member": "连杆件",
    "support": "设支座",
    "delete": "删除",
}

# 每个模式在状态栏上的提示。**光标形状变了但没人告诉你现在能干什么**，
# 是这类工具最常见的失败方式
MODE_HINTS = {
    "node": "在工作平面上点击放置节点，会自动吸附到网格或已有节点",
    "member": "点两个节点连一根杆件；点空白处会先自动放一个节点",
    "support": "点节点设置支座，形式在右侧选",
    "delete": "点节点或杆件删除；删节点会连带删掉挂在它上面的杆件",
}


@dataclass
class ClickResult:
    """一次点击的结果。`ok` 为假表示这一下没生效，`message` 说明原因。"""

    ok: bool
    message: str
    created: tuple[str, int] | None = None      # 新建了什么
    removed: list[tuple[str, int]] = field(default_factory=list)
    changed: bool = False                       # 草稿是否真的变了


@dataclass
class DrawingSession:
    """一次手工画图。开始于「进入画图」，结束于提交或取消。"""

    draft: FrameDraft
    plane: WorkPlane = field(default_factory=WorkPlane)
    mode: str = "node"
    section: str = ""
    material: str | None = None
    support_fix: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])

    # 连杆模式下已经点了第一个节点
    pending_node: int | None = None
    # 连成一根之后，终点自动成为下一根的起点。**刚架就是一串连着的杆**，
    # 每根都要重点一次起点的话，画一榀门架要八下而不是四下；
    # 而且重点起点时点偏了，就会多出一个几乎重合的节点
    chain: bool = True
    # 每次点击前的现场，供画图过程中逐步撤销
    _undo: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # --- 模式 ---

    def set_mode(self, mode: str) -> str:
        if mode not in MODES:
            raise ValueError(f"未知模式 {mode!r}")
        self.mode = mode
        self.pending_node = None        # **换模式必须清掉半根杆**，
        return MODE_HINTS[mode]         # 否则下一模式的第一次点击会莫名连出一根

    def cancel_pending(self) -> bool:
        """Esc：放弃正在连的这根杆。返回是否真的有东西可放弃。"""
        had = self.pending_node is not None
        self.pending_node = None
        return had

    # --- 点击 ---

    def click(self, point) -> ClickResult:
        """在工作平面上点了一下。`point` 已经是三维坐标。"""
        snapped, hit, how = self.plane.snap(point, self.draft.nodes)
        before = self.draft.snapshot()

        if self.mode == "node":
            result = self._place_node(snapped, hit, how)
        elif self.mode == "member":
            result = self._connect(snapped, hit, how)
        elif self.mode == "support":
            result = self._support(hit)
        else:
            result = self._delete(snapped, hit)

        if result.changed:
            self._undo.append(before)
        return result

    def _place_node(self, p, hit, how) -> ClickResult:
        if hit is not None:
            # 已经有节点了就说清楚，别悄悄什么都不做——
            # 用户会以为点击丢了，然后反复点同一个地方
            return ClickResult(True, f"这里已经有节点 {hit} 了", ("node", hit))
        nid = self.draft.add_node(*p)
        return ClickResult(True, f"节点 {nid} @ ({p[0]:g}, {p[1]:g}, {p[2]:g}) m，{how}",
                           ("node", nid), changed=True)

    def _connect(self, p, hit, how) -> ClickResult:
        nid = hit if hit is not None else self.draft.add_node(*p)
        made_node = hit is None
        # **新建节点时必须把吸附结果说出来。** 网格 0.5 m 时点 3.6 会落到 3.5，
        # 不说的话用户画完才发现层高不对，而且完全不知道是哪一步偏的
        where = (f"，{how}（{p[0]:g}, {p[1]:g}, {p[2]:g}）" if made_node else "")

        if self.pending_node is None:
            self.pending_node = nid
            return ClickResult(True,
                               f"从节点 {nid} 开始{where}，再点一个节点连成杆件",
                               ("node", nid), changed=made_node)

        if nid == self.pending_node:
            # 原地点两下不是连杆。**保持起点不变**，用户多半只是点偏了
            return ClickResult(False, "起点和终点是同一个节点，换一个点",
                               changed=made_node)
        start = self.pending_node
        try:
            mid = self.draft.add_member(start, nid, self.section, self.material)
        except DraftError as exc:
            self.pending_node = None
            return ClickResult(False, str(exc), changed=made_node)
        # 接着往下画。Esc 断链——**必须有个明确的出口**，
        # 否则想在别处另起一根时，第一下会从上一根的终点连过去
        self.pending_node = nid if self.chain else None
        tail = "，接着画下一根（Esc 结束）" if self.chain else ""
        length = self.draft.length(mid) or 0.0
        return ClickResult(True,
                           f"杆件 {mid}：节点 {start} → {nid}，长 {length:.3f} m，"
                           f"截面 {self.section}{where}{tail}",
                           ("member", mid), changed=True)

    def _support(self, hit) -> ClickResult:
        if hit is None:
            return ClickResult(False, "支座只能加在节点上，点准一点")
        was = next((s["fix"] for s in self.draft.supports
                    if int(s["node"]) == hit), None)
        if was == list(self.support_fix):
            self.draft.set_support(hit, [])         # 再点一次就取消，符合直觉
            return ClickResult(True, f"取消了节点 {hit} 的支座", ("node", hit),
                               changed=True)
        self.draft.set_support(hit, list(self.support_fix))
        return ClickResult(True, f"节点 {hit} 设为 {self._fix_name()}", ("node", hit),
                           changed=True)

    def _fix_name(self) -> str:
        f = list(self.support_fix)
        if f == [1] * 6:
            return "固接"
        if f == [1, 1, 1, 0, 0, 0]:
            return "铰接"
        return "自定义约束"

    def _delete(self, p, hit) -> ClickResult:
        if hit is not None:
            dropped = self.draft.delete_node(hit)
            extra = f"，连带删掉杆件 {dropped}" if dropped else ""
            return ClickResult(True, f"删除节点 {hit}{extra}", None,
                               [("node", hit)] + [("member", m) for m in dropped],
                               changed=True)
        mid = self.nearest_member(p)
        if mid is None:
            return ClickResult(False, "这里没有可删的东西")
        self.draft.delete_member(mid)
        return ClickResult(True, f"删除杆件 {mid}", None, [("member", mid)],
                           changed=True)

    def nearest_member(self, point, tol: float = 0.4) -> int | None:
        """离这个点最近的杆件。用点到线段的距离，不是到中点的距离——
        长杆的中点可能离得很远，而用户点的是杆身。"""
        import numpy as np

        p = np.asarray(point, dtype=float)
        best, best_d = None, tol
        for m in self.draft.members:
            a, b = self.draft.node(int(m["i"])), self.draft.node(int(m["j"]))
            if a is None or b is None:
                continue
            u = np.array([a["x"], a["y"], a["z"]], dtype=float)
            v = np.array([b["x"], b["y"], b["z"]], dtype=float)
            seg = v - u
            L2 = float(seg @ seg)
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, float((p - u) @ seg) / L2))
            d = float(np.linalg.norm(p - (u + t * seg)))
            if d <= best_d:
                best, best_d = int(m["id"]), d
        return best

    # --- 画图过程中的撤销 ---

    def undo_click(self) -> bool:
        """退回上一次点击之前。**这和提交后的撤销是两码事**：
        画的时候你想退上一根杆，画完之后你想退掉整次修改。"""
        if not self._undo:
            return False
        self.draft.restore(self._undo.pop())
        self.pending_node = None
        return True

    @property
    def steps(self) -> int:
        return len(self._undo)

    # --- 状态 ---

    def status(self) -> str:
        """状态栏那一行：现在在哪个平面、什么模式、下一步该干什么。"""
        bits = [self.plane.label(), MODE_LABELS[self.mode]]
        if self.mode == "member":
            bits.append(f"截面 {self.section}" if self.section else "未选截面")
            if self.pending_node is not None:
                bits.append(f"已选起点 节点 {self.pending_node}，按 Esc 取消")
        if self.mode == "support":
            bits.append(self._fix_name())
        return " · ".join(bits)
