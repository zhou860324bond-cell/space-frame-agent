"""单位制。

内核本身是量纲无关的：只要 E、几何、荷载用同一套单位，刚度方程就成立，
所以求解过程一个换算都不需要。真正跟单位制有关的只有两处：

1. **重力加速度**——自重要用 ρ·A·g，而 g 的数值随长度单位变。
   这一处最阴：`units` 早先只是 schema 里一个没人读的字段，
   自重却硬编码了 9.80665 m/s²。选 N-mm-MPa 的话自重小 1000 倍，
   而且不报错、不告警，结果看着完全正常。
2. **显示换算**——位移习惯用 mm、力用 kN、弯矩用 kN·m 报，
   而模型内部存的是 m/N 或 mm/N。

所以这里做的不是"把模型换算成另一套单位"，而是把这两处**唯一的**
单位相关常数集中起来，谁需要谁来取。模型的数值一个字节都不动。

约定：

    N-m-Pa      长度 m   力 N   E 用 Pa    密度 kg/m³      g = 9.80665 m/s²
    N-mm-MPa    长度 mm  力 N   E 用 MPa   密度 t/mm³      g = 9806.65 mm/s²

两套都是自洽的工程常用制。混着用（比如 m 配 MPa）不在支持之列，
也没法自动检出——量纲无关的代价就在这里。
"""

from __future__ import annotations

from dataclasses import dataclass

SI = "N-m-Pa"
MM = "N-mm-MPa"


@dataclass(frozen=True)
class UnitSystem:
    """一套单位制下的常数与显示换算。

    `*_scale` 是"模型内部数值 × scale = 显示数值"。位移一律报 mm、
    力报 kN、弯矩报 kN·m——报告和界面上的单位因此与单位制无关，
    读的人不用先想"这是哪套制"。
    """
    name: str
    length: str                 # 模型内部的长度单位
    gravity: float              # 与 length 一致的重力加速度
    density: str                # 密度单位，写在提示里
    length_to_m: float          # 模型长度 → m（结果中的 *_m 字段专用）
    disp_scale: float           # 位移 → mm
    force_scale: float          # 力 → kN
    moment_scale: float         # 弯矩 → kN·m
    line_load_scale: float      # 线荷载 → kN/m

    disp_unit: str = "mm"
    force_unit: str = "kN"
    moment_unit: str = "kN·m"
    line_load_unit: str = "kN/m"
    # 应力一律报 MPa。工程上没人拿 Pa 读应力：Q355 的许用值写成
    # 215000000 谁也认不出来，写成 215 一眼就知道。
    stress_scale: float = 1.0
    stress_unit: str = "MPa"


_SYSTEMS = {
    SI: UnitSystem(name=SI, length="m", gravity=9.80665, density="kg/m³",
                   length_to_m=1.0,
                   disp_scale=1e3,      # m  → mm
                   force_scale=1e-3,    # N  → kN
                   moment_scale=1e-3,   # N·m → kN·m
                   line_load_scale=1e-3,    # N/m → kN/m
                   stress_scale=1e-6),      # Pa  → MPa
    MM: UnitSystem(name=MM, length="mm", gravity=9806.65, density="t/mm³",
                   length_to_m=1e-3,
                   disp_scale=1.0,      # mm → mm
                   force_scale=1e-3,    # N  → kN
                   moment_scale=1e-6,   # N·mm → kN·m
                   line_load_scale=1.0,     # N/mm → kN/m
                   stress_scale=1.0),       # MPa → MPa
}

NAMES = tuple(_SYSTEMS)


def system(name: str | None) -> UnitSystem:
    """按名字取单位制。留空按 N-m-Pa。"""
    if name is None:
        return _SYSTEMS[SI]
    try:
        return _SYSTEMS[str(name)]
    except KeyError:
        raise ValueError(
            f"未知单位制 {name!r}，只支持 {'、'.join(NAMES)}") from None


def of(model) -> UnitSystem:
    """取模型的单位制。model 可以是 dict（JSON 模型）或 Frame。"""
    if isinstance(model, dict):
        return system(model.get("units"))
    return system(getattr(model, "units", None))


# --------------------------------------------------------------- 模型换算

# 从 N-m-Pa 换到 N-mm-MPa 时，每一类量要乘的系数。
# 反向就取倒数——所以只需要写一份，不会出现两套系数对不上的情况。
_TO_MM = {
    "length": 1e3,          # m   → mm
    "area": 1e6,            # m²  → mm²
    "inertia": 1e12,        # m⁴  → mm⁴
    "first_moment": 1e9,    # m³  → mm³（半截面静矩 S）
    "curvature": 1e-3,      # 1/m → 1/mm（初曲率 κ）
    "modulus": 1e-6,        # Pa  → MPa
    "density": 1e-12,       # kg/m³ → t/mm³
    "force": 1.0,           # N   → N
    "moment": 1e3,          # N·m → N·mm
    "line_load": 1e-3,      # N/m → N/mm
    # 弹簧刚度：平动是 力/长度，转动是 力·长度/弧度。**两者方向相反**，
    # 用同一个系数会让转动弹簧差 10⁶，而且只在换过单位的模型上现形。
    "spring_translation": 1e-3,   # N/m     → N/mm
    "spring_rotation": 1e3,       # N·m/rad → N·mm/rad
}


def _factors(src: str, dst: str) -> dict[str, float]:
    if src == dst:
        return {k: 1.0 for k in _TO_MM}
    if src == SI and dst == MM:
        return dict(_TO_MM)
    if src == MM and dst == SI:
        return {k: 1.0 / v for k, v in _TO_MM.items()}
    raise ValueError(f"不支持从 {src} 换到 {dst}")


def convert_model(model: dict, to: str) -> dict:
    """把整份模型换算到另一套单位制，返回新模型，原模型不动。

    换的是**数值**，物理量不变：同一个结构换算前后算出来的位移（以 mm 计）
    和内力（以 kN 计）必须逐位相同。测试就是拿这一条当判据的——
    比逐个系数去核对可靠得多，漏掉任何一类量都会立刻露馅。
    """
    src = str(model.get("units", SI))
    dst = str(to)
    system(src), system(dst)                      # 先确认两边都认识
    f = _factors(src, dst)
    out = {k: v for k, v in model.items()}
    out["units"] = dst

    out["materials"] = [{**m, "E": m["E"] * f["modulus"],
                         **({"density": m["density"] * f["density"]}
                            if "density" in m else {}),
                         **({"yield_stress": m["yield_stress"] * f["modulus"]}
                            if "yield_stress" in m else {}),
                         # 许用应力也是应力。漏掉它 Pa↔MPa 差 10⁶，
                         # 而且只在换过单位的模型上才现形。
                         **({"allow_tension": m["allow_tension"] * f["modulus"]}
                            if "allow_tension" in m else {}),
                         **({"allow_compression": m["allow_compression"] * f["modulus"]}
                            if "allow_compression" in m else {})}
                        for m in model.get("materials", [])]
    out["sections"] = [{**s, "A": s["A"] * f["area"],
                        "Iy": s["Iy"] * f["inertia"], "Iz": s["Iz"] * f["inertia"],
                        "J": s["J"] * f["inertia"],
                        **({"Ay": s["Ay"] * f["area"]} if "Ay" in s else {}),
                        **({"Az": s["Az"] * f["area"]} if "Az" in s else {}),
                        # 极端纤维距离是长度。漏掉这两项会让正应力差 1000 倍，
                        # 而且只在换过单位的模型上才出现。
                        **({"cy": s["cy"] * f["length"]} if "cy" in s else {}),
                        **({"cz": s["cz"] * f["length"]} if "cz" in s else {}),
                        # 静矩是长度³、中性轴宽度是长度。漏掉 S 会让剪应力
                        # 差 10⁹，和 cy/cz 一样只在换过单位的模型上才现形。
                        **({"Sz": s["Sz"] * f["first_moment"]} if "Sz" in s else {}),
                        **({"Sy": s["Sy"] * f["first_moment"]} if "Sy" in s else {}),
                        **({"bz": s["bz"] * f["length"]} if "bz" in s else {}),
                        **({"by": s["by"] * f["length"]} if "by" in s else {}),
                        **({"S_flange": s["S_flange"] * f["first_moment"]}
                           if "S_flange" in s else {}),
                        **({"c_web": s["c_web"] * f["length"]}
                           if "c_web" in s else {})}
                       for s in model.get("sections", [])]
    def spring(values):
        """六个方向的支承刚度：前三个是力/长度，后三个是力·长度/弧度。"""
        return [v * (f["spring_translation"] if k < 3 else f["spring_rotation"])
                for k, v in enumerate(values)]

    out["supports"] = [
        {**s, **({"spring": spring(s["spring"])} if "spring" in s else {})}
        for s in model.get("supports", [])]
    # 分析步里可以改写支座，那里的弹簧刚度同样要换。**同一种量出现在两个
    # 地方**，只换一处的话，用了分析步的模型在毫米制下就会带着一个没换过的
    # 弹簧——顶层那个换了、步里那个没换，两者还会同时存在。
    if model.get("steps"):
        out["steps"] = [
            {**st, **({"supports": {
                node: {**entry,
                       **({"spring": spring(entry["spring"])}
                          if "spring" in entry else {})}
                for node, entry in st["supports"].items()}}
                if st.get("supports") else {})}
            for st in model["steps"]]
    out["nodes"] = [{**n, "x": n["x"] * f["length"], "y": n["y"] * f["length"],
                     "z": n["z"] * f["length"]}
                    for n in model.get("nodes", [])]
    def connection(table):
        """半刚性连接刚度，键是局部自由度名：平动与转动的量纲不同。

        漏掉它，毫米制下连接刚度会差好几百倍——实测悬臂挠度 −60.17 mm
        变成 −21638 mm，而模型校验照样通过。
        """
        return {dof: value * (f["spring_rotation"] if dof.startswith("r")
                              else f["spring_translation"])
                for dof, value in table.items()}

    out["members"] = [
        {**m,
         **({"offset_i": [v * f["length"] for v in m["offset_i"]]}
            if "offset_i" in m else {}),
         **({"offset_j": [v * f["length"] for v in m["offset_j"]]}
            if "offset_j" in m else {}),
         **({"connections": {end: connection(table)
                             for end, table in m["connections"].items()}}
            if "connections" in m else {})}
        for m in model.get("members", [])]

    def case(block: dict) -> dict:
        got = {k: v for k, v in block.items()}
        if block.get("nodal_loads"):
            # 前三个是力、后三个是弯矩，系数不同
            got["nodal_loads"] = [
                {**e, "load": [v * (f["force"] if k < 3 else f["moment"])
                               for k, v in enumerate(e["load"])]}
                for e in block["nodal_loads"]]
        if block.get("member_loads"):
            got["member_loads"] = [
                {**e, "w": [v * f["line_load"] for v in e["w"]]}
                for e in block["member_loads"]]
        if block.get("member_spans"):
            spans = []
            for e in block["member_spans"]:
                # 集中力是力，梯形/均布是线荷载 —— 同一个字段两种量纲
                s = f["force"] if e.get("kind") == "point" else f["line_load"]
                item = {**e, "w1": [v * s for v in e["w1"]]}
                if "w2" in e:
                    item["w2"] = [v * s for v in e["w2"]]
                if "a" in e:
                    item["a"] = e["a"] * f["length"]
                # **b 和 a 一样是沿杆的位置。** 漏掉它，部分跨荷载在毫米制下
                # 变成"从 1000 mm 起到 4 mm 止"——区间反过来了。实测悬臂挠度
                # 从 −12.47 mm 变成 +0.228 mm，连符号都反，而且不报任何错。
                if "b" in e:
                    item["b"] = e["b"] * f["length"]
                spans.append(item)
            got["member_spans"] = spans
        if block.get("member_strains"):
            # lack_of_fit 是长度，要换；delta_t 是温度，**不能换**。
            # 同一条目里两种量纲，漏掉任一边都不会报错，只会静默算错。
            got["member_strains"] = [
                {**e, **({"lack_of_fit": e["lack_of_fit"] * f["length"]}
                         if e.get("lack_of_fit") is not None else {})}
                for e in block["member_strains"]]
        if block.get("settlements"):
            # 前三个是位移、后三个是转角（弧度，不换算）
            got["settlements"] = [
                {**e, "d": [v * (f["length"] if k < 3 else 1.0)
                            for k, v in enumerate(e["d"])]}
                for e in block["settlements"]]
        return got

    for key in ("nodal_loads", "member_loads", "member_spans", "settlements",
                "member_strains"):
        if model.get(key):
            out.update({k: v for k, v in case({key: model[key]}).items()})
    if model.get("load_cases"):
        out["load_cases"] = [case(c) for c in model["load_cases"]]
    return out
