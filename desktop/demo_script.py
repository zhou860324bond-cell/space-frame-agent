"""离线演示后端。

没有密钥、没有网的场合（答辩现场很常见）也得能把完整链路演一遍。

**这不是假装在跟大模型说话。** 它按关键词认意图，然后调**真的工具**，
数从**真的求解器**出来——被替掉的只有"把中文映射成工具调用"这一步。
所以面板上显示的调用链和数字，跟联网时是一样的。

认不出来的话就直说认不出来。假装听懂了然后随便调点什么，
比不懂更糟：那会让人以为模型理解了它其实没理解的东西。

固定脚本在这里行不通——用户随便打一句就"脚本已用尽"了。
所以这里做的是意图分派，不是回放。
"""

from __future__ import annotations

import json
from typing import Any

import credentials

DEMO_PROMPT = ("单跨 24 米门式刚架，檐口高 7.5 米，屋脊比檐口高 1.2 米，"
               "柱脚铰接，Q355。屋面恒载 8 kN/m、活载 5 kN/m、风吸 3 kN/m，"
               "建模求解并看看挠跨比。")

SPAN, EAVE, RISE = 24.0, 7.5, 1.2

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]


def _i_section(name: str, h: float, b: float, tw: float, tf: float) -> dict:
    from sections import i_section
    return i_section(name, h, b, tw, tf)


# 关键词 → 意图。顺序有讲究：先匹配更具体的
_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("report",    ("报告", "计算书", "导出")),
    ("buckling",  ("稳定", "屈曲", "失稳", "临界")),
    ("modal",     ("自振", "模态", "频率", "周期", "振型")),
    ("envelope",  ("组合", "包络", "控制")),
    ("deflection", ("挠度", "位移", "挠跨比", "变形", "软")),
    ("build",     ("建模", "求解", "刚架", "门式", "算一下", "算算", "跨")),
)


def _intent(text: str) -> str | None:
    for name, words in _INTENTS:
        if any(w in text for w in words):
            return name
    return None


def _call(cid: str, name: str, **args) -> dict:
    """一个待发的工具调用。注意这里返回的是**调用本身**，不是一条消息——
    消息由 `_message()` 把若干个调用打包成一条。"""
    return {"id": cid, "name": name, "arguments": args}


def _message(calls: list[dict]) -> dict:
    """把若干工具调用打成一条 assistant 消息。

    **一次只发一个是很贵的。** `Conversation.ask` 每轮发一条消息、执行其中
    所有调用、再进下一轮；接真实大模型时一轮就是一个网络往返。建模那五步
    逐个发就是五个往返，用户等的是秒。互不依赖的调用打成一条，五轮变一轮。
    这也是提示词第 9 条要求真实模型做的事——离线后端没理由做得更差。
    """
    return {"tool_calls": calls}


class DemoProvider:
    """按意图分派的离线后端。接口与真后端一致，`Conversation` 分不出区别。"""

    name = "离线演示"

    def __init__(self, session=None) -> None:
        # 拿着 Session 是为了知道"模型建了没有"。查询类意图落在空模型上
        # 只会得到一句没头没脑的报错，而正确的反应是**先把模型建起来**
        self.session = session
        self._pending: list[dict] = []
        self._closing = ""

    # --- Provider 接口 ---

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        # 上一条是工具结果，说明这一轮的调用已经跑完，该给结论了
        if messages and messages[-1].get("role") == "tool":
            if self._pending:
                return self._pending.pop(0)
            return {"content": self._closing or "完成。"}

        text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                text = str(m.get("content") or "")
                break

        has_model = bool(self.session and self.session.model.get("nodes"))
        plan, closing = self._plan(text, has_model)
        if not plan:
            self._pending, self._closing = [], ""
            return {"content": closing}
        self._pending = plan[1:]
        self._closing = closing
        return plan[0]

    # --- 计划 ---

    def _plan(self, text: str, has_model: bool) -> tuple[list[dict], str]:
        intent = _intent(text)

        # 模型还不存在时，任何查询都得先建模。
        # 这不是取巧——**在空模型上查挠度，得到的是一句没头没脑的报错**，
        # 而用户真正想要的显然是"建起来再看"。真的 Agent 也该这么做。
        if intent is not None and intent != "build" and not has_model:
            plan, closing = self._plan(text, True)
            return self._build_plan() + plan, (
                f"已建好单跨 {SPAN:g} m 门式刚架并求解。" + closing)

        if intent is None:
            return [], (
                "离线演示模式只认得几类问题：建模求解、挠度、控制组合、"
                "稳定、自振、出报告。你可以点工具栏的「演示问题」填一句试试。\n"
                f"配置密钥（新建 {credentials.where_to_put_it()}）之后"
                "就能随便说了。")

        if intent == "build":
            return self._build_plan(), (
                f"已建好单跨 {SPAN:g} m 门式刚架并求解三个荷载组合，"
                "静力平衡校核通过。左侧模型树和视口里能看到结果。")
        if intent == "deflection":
            return ([_message([_call("d1", "query_results",
                                     what="max_deflection", span=SPAN)])],
                    f"挠度见上。判据取 L/400 = {SPAN / 400 * 1000:.0f} mm——"
                    "注意斜梁是两根杆，跨度必须显式传，"
                    "否则算出来的是杆长比而不是挠跨比。")
        if intent == "envelope":
            # 两个包络彼此无关，一条消息发完
            return ([_message([_call("e1", "query_envelope", component="Mz"),
                               _call("e2", "query_envelope", component="N")])],
                    "包络与控制组合见上。留意「控制全局极值」和"
                    "「在某处控制」是两回事——后者不能删。")
        if intent == "buckling":
            return ([_message([_call("b1", "buckling_analysis",
                                     num_modes=3)])],
                    "屈曲因子见上。这是线弹性屈曲，只给失稳的量级参考。")
        if intent == "modal":
            return ([_message([_call("m1", "modal_analysis",
                                     num_modes=6)])],
                    "自振频率见上。一致质量阵的特征值从上方收敛，"
                    "所以这些值是偏高一点点的。")
        if intent == "report":
            return ([_message([_call("r1", "write_report", fmt="both",
                                     filename="门式刚架分析报告")])],
                    "报告已生成，在仓库根目录。")
        return [], "没看懂。"

    def _build_plan(self) -> list[dict]:
        """建模这一步要先知道斜梁编号，所以拿一个探针 Session 先生成一遍。"""
        from agent import Session

        column = _i_section("COLUMN", 0.700, 0.250, 0.010, 0.016)
        rafter = _i_section("RAFTER", 0.900, 0.250, 0.010, 0.020)

        probe = Session()
        probe.define_materials_and_sections(MATERIALS, [column, rafter])
        gen = probe.generate_portal_frame(
            spans=[SPAN], eave_height=EAVE, ridge_rise=RISE,
            column_section="COLUMN", rafter_section="RAFTER",
            material="Q355", base="pinned")
        rafters = gen.payload["rafter_member_ids"]

        def on_rafters(w: float) -> list[dict]:
            return [{"member": m, "w": [0.0, 0.0, w]} for m in rafters]

        # 四步顺序固定，但都不需要读上一步的返回值（斜梁编号是探针先算好的），
        # 所以打成一条消息发出去，一轮跑完
        return [_message([
            _call("t1", "define_materials_and_sections",
                  materials=MATERIALS, sections=[column, rafter]),
            _call("t2", "generate_portal_frame", spans=[SPAN],
                  eave_height=EAVE, ridge_rise=RISE, column_section="COLUMN",
                  rafter_section="RAFTER", material="Q355", base="pinned"),
            _call("t3", "set_load_cases",
                  cases=[{"name": "D", "member_loads": on_rafters(-8e3)},
                         {"name": "L", "member_loads": on_rafters(-5e3)},
                         {"name": "W", "member_loads": on_rafters(+3e3)}],
                  combos=[{"name": "1.3D+1.5L",
                           "factors": {"D": 1.3, "L": 1.5}},
                          {"name": "1.3D+1.5W",
                           "factors": {"D": 1.3, "W": 1.5}},
                          {"name": "1.3D+1.5L+0.9W",
                           "factors": {"D": 1.3, "L": 1.5, "W": 0.9}}]),
            _call("t4", "solve_model"),
        ])]


def demo_provider(session=None) -> DemoProvider:
    return DemoProvider(session)
