"""沿杆长的内力分布。

**在单元内解析恢复，不靠网格加密。** 杆端力已经算出来了，跨间荷载的贡献是
封闭形式：均布荷载下弯矩是二次、剪力是线性。所以单跨划一个单元，弯矩图
就已经是精确解——不需要为了画图去加密网格。

（对照：FEM-Python 每根单元只在中点取一个积分点值，提高分辨率只能加密。
那是有限元的通用做法，但刚架程序不必这样。）

符号约定与杆端力一致：截面内力取**j 侧部分作用在截面上的合力**，
于是 x=L 处等于 f[6..11]，x=0 处等于 -f[0..5]。轴力以受拉为正。

推导（左段 [0, x] 的平衡）：

    N(x)  = -f0 - wx·x
    Vy(x) = -f1 - wy·x
    Vz(x) = -f2 - wz·x
    T(x)  = -f3
    My(x) = -f4 - f2·x - wz·x²/2
    Mz(x) = -f5 + f1·x + wy·x²/2

弯矩两项的符号来自 r × F：截面左侧的力 F 作用在相对截面 r = (-x, 0, 0) 处，
局部 y 向力对 z 轴的矩是 -x·Fy，移到等式右边即 +f1·x；局部 z 向力对 y 轴的矩
是 +x·Fz，移过去是 -f2·x。两项符号相反，第一版把它们弄成同号，
悬臂梁的自由端弯矩就没有归零——这是最容易验出来的判据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frame3d import (DEFAULT_CASE, Frame, local_axes, member_endpoints,
                     member_local_displacements, span_loads_of)
from span_loads import POINT, span_force

COMPONENTS = ("N", "Vy", "Vz", "T", "My", "Mz")
# 局部 y/z 轴随杆件方向转动。空间刚架跨杆件比较时，单看 My 或 Mz 很容易
# 把“局部轴不同”误读成“内力突变”。合剪力、合弯矩是坐标旋转不变量，专供
# 全结构云图与筛选使用；六个有符号基本分量仍保留给杆件设计与端力核对。
DERIVED_COMPONENTS = ("V", "M")
DISPLAY_COMPONENTS = COMPONENTS + DERIVED_COMPONENTS


@dataclass
class MemberDiagram:
    """一根杆件沿长度的内力分布。x 从 i 端量起，单位米；内力用 N 与 N·m。"""
    member: int
    case: str
    length: float
    x: np.ndarray
    N: np.ndarray
    Vy: np.ndarray
    Vz: np.ndarray
    T: np.ndarray
    My: np.ndarray
    Mz: np.ndarray

    def component(self, name: str) -> np.ndarray:
        if name == "V":
            return np.hypot(self.Vy, self.Vz)
        if name == "M":
            return np.hypot(self.My, self.Mz)
        if name not in COMPONENTS:
            raise KeyError(f"内力分量只能取 {DISPLAY_COMPONENTS}，收到 {name!r}")
        return getattr(self, name)

    def extreme(self, name: str) -> tuple[float, float]:
        """返回该分量绝对值最大的 (x, 值)。画图标注和查最不利截面都用它。"""
        values = self.component(name)
        k = int(np.argmax(np.abs(values)))
        return float(self.x[k]), float(values[k])


def _span_contribution(frame: Frame, member, case: str, L: float,
                       x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """该工况下所有杆间荷载对 [0, x] 段的合力与合力矩，局部坐标。

    公式与求解时的固端力共用 span_loads 一个模块——两边用同一份公式，
    弯矩图才不会和求解器各说各话。
    """
    S = np.zeros((3, len(x)))
    M = np.zeros((3, len(x)))
    load_case = frame.load_cases.get(case)
    if load_case is None:
        return S, M
    pi, pj = member_endpoints(frame, member)
    _, rot = local_axes(pi, pj, member.ref_vector)
    for item in span_loads_of(load_case, member.id):
        s, m = span_force(item, L, rot, x)
        S += s
        M += m
    return S, M


def point_load_stations(frame: Frame, member_id: int, case: str,
                        L: float) -> list[float]:
    """集中力所在位置。剪力在那里是跳跃的，画图要在两侧各取一个点，
    否则图上会画出一条根本不存在的斜线。"""
    load_case = frame.load_cases.get(case)
    if load_case is None:
        return []
    return [item.a for item in span_loads_of(load_case, member_id)
            if item.kind == POINT and 0.0 < item.a < L]


def member_length(frame: Frame, member_id: int) -> float:
    m = frame.members[member_id]
    pi, pj = member_endpoints(frame, m)
    return float(np.linalg.norm(pj - pi))


def _result_local_displacements(frame: Frame, solution, member, name: str) -> np.ndarray:
    """恢复静力结果的真实杆端位移；组合按基础工况逐项恢复后叠加。"""
    if name in frame.combos:
        return sum(
            factor * member_local_displacements(
                frame, member, solution[base].U, frame.load_cases[base])
            for base, factor in frame.combos[name].items())
    case = frame.load_cases.get(name)
    return member_local_displacements(frame, member, solution[name].U, case)


def diagram_stations(frame: Frame, solution, member_id: int,
                     case: str | None = None, stations: int = 21) -> np.ndarray:
    """某工况下该杆件的采样点，含集中力处的跳跃点。

    包络要把各工况放到**同一套**测点上比，所以先各自取一遍再取并集——
    否则两个工况的 x 数组长度都不同，逐点比根本无从谈起。
    """
    name = case or solution.primary
    length = member_length(frame, member_id)
    x = np.linspace(0.0, length, stations)
    sources = (frame.combos[name].items() if name in frame.combos
               else ((name, 1.0),))
    jumps: list[float] = []
    for base, _ in sources:
        jumps += point_load_stations(frame, member_id, base, length)
    if jumps:
        eps = max(length * 1e-9, 1e-12)
        # 荷载点本身也要采到：弯矩在那里连续且正是极值点，
        # 只取两侧的话读出来的峰值永远差一点点
        extra = [v for a in jumps for v in (a - eps, a, a + eps)]
        x = np.unique(np.clip(np.concatenate([x, extra]), 0.0, length))
    return x


def member_diagram(frame: Frame, solution, member_id: int,
                   case: str | None = None, stations: int = 21,
                   at_x: np.ndarray | None = None) -> MemberDiagram:
    """算一根杆件的内力分布。stations 只影响画图的采样密度，不影响精度。

    at_x 给定时直接在这些位置求值，不再自行插入跳跃点——包络用它把
    各工况对齐到同一套测点。
    """
    if member_id not in frame.members:
        raise KeyError(f"没有编号为 {member_id} 的杆件")
    if at_x is None and stations < 2:
        raise ValueError("stations 至少为 2")
    name = case or solution.primary
    member = frame.members[member_id]
    f = solution[name].member_forces[member_id]

    length = member_length(frame, member_id)
    sources = (frame.combos[name].items() if name in frame.combos
               else ((name, 1.0),))
    if at_x is not None:
        x = np.asarray(at_x, dtype=float)
    else:
        x = diagram_stations(frame, solution, member_id, name, stations)

    S = np.zeros((3, len(x)))
    M = np.zeros((3, len(x)))
    for base, factor in sources:
        s, m = _span_contribution(frame, member, base, length, x)
        S += factor * s
        M += factor * m

    def clean_roundoff(values: np.ndarray) -> np.ndarray:
        """去掉相对峰值处于机器舍入量级的假残值。

        线性组合与矩阵缩放会让理论零点留下约 1e-11 N·m 的尾数。若直接送进
        云图，全零/过零区域会显示一条不存在的细色带。只按本分量自身峰值
        清理，因此整体很小但真实存在的内力不会被抹掉。
        """
        values = np.asarray(values, dtype=float)
        peak = float(np.abs(values).max(initial=0.0))
        if peak == 0.0:
            return values
        out = values.copy()
        out[np.abs(out) <= 1e-13 * peak] = 0.0
        return out

    return MemberDiagram(
        member=member_id, case=name, length=length, x=x,
        N=clean_roundoff(-f[0] - S[0]),
        Vy=clean_roundoff(-f[1] - S[1]),
        Vz=clean_roundoff(-f[2] - S[2]),
        T=clean_roundoff(np.full_like(x, -f[3])),
        My=clean_roundoff(-f[4] - f[2] * x - M[2]),
        Mz=clean_roundoff(-f[5] + f[1] * x + M[1]),
    )


def physical_member_diagram(frame: Frame, solution, mapping,
                            physical_member_id: int, case: str | None = None,
                            stations: int = 21) -> MemberDiagram:
    """把若干分析单元的内力图按编译顺序接回一根物理构件。"""
    element_ids = mapping.element_ids(physical_member_id)
    diagrams = [member_diagram(frame, solution, element_id, case, stations)
                for element_id in element_ids]
    offsets = np.cumsum([0.0] + [diagram.length for diagram in diagrams[:-1]])

    def joined(name: str) -> np.ndarray:
        return np.concatenate([diagram.component(name) for diagram in diagrams])

    x = np.concatenate([diagram.x + offset
                        for diagram, offset in zip(diagrams, offsets)])
    name = case or solution.primary
    return MemberDiagram(
        member=physical_member_id, case=name,
        length=float(sum(diagram.length for diagram in diagrams)), x=x,
        N=joined("N"), Vy=joined("Vy"), Vz=joined("Vz"), T=joined("T"),
        My=joined("My"), Mz=joined("Mz"),
    )


def all_diagrams(frame: Frame, solution, case: str | None = None,
                 stations: int = 21) -> dict[int, MemberDiagram]:
    name = case or solution.primary
    return {mid: member_diagram(frame, solution, mid, name, stations)
            for mid in sorted(frame.members)}


def envelope(frame: Frame, solution, component: str,
             case: str | None = None, stations: int = 21) -> dict:
    """全结构该分量的极值，以及出现在哪根杆件的哪个位置。"""
    diagrams = all_diagrams(frame, solution, case, stations)
    best = {"component": component, "value": 0.0, "member": None, "x": 0.0}
    for mid, d in diagrams.items():
        pos, value = d.extreme(component)
        if abs(value) > abs(best["value"]) or best["member"] is None:
            best = {"component": component, "value": value, "member": mid, "x": pos}
    return best


# ---------------------------------------------------------------- 单元内挠度

def member_deflection(frame: Frame, solution, member_id: int,
                      case: str | None = None, stations: int = 101):
    """沿杆长的横向位移，局部坐标。返回 (x, v, w)。

    **为什么必须有这个**：`query_results` 的最大位移只看**节点**。单跨划一个
    单元时，跨中挠度所在的位置根本没有节点——8 m 梁 25 kN/m 的真实跨中挠度
    4.2 mm，只查节点会报成 0.16 mm。拿它去校核 L/400 会得出完全错误的结论。
    内力早就做了单元内解析恢复，位移不能落下。

    做法是复用已经算好的**精确内力**：弯曲转角由 ``EI·θ′ = M`` 恢复；
    Timoshenko 截面再叠加 ``γ = V/(G·As)`` 的剪切转角。对它们累积积分，
    两个积分常数由两端的真实杆端位移定。这样每一种荷载
    （均布、梯形、集中力、组合）都自动覆盖，不必逐个推闭合解。

    **符号是靠端部转角钉死的，不是靠闭合解。** 这一点值得说清楚：
    两端位移是强加的，所以"跨中挠度对上闭合解"这类检验对符号翻转**不敏感**
    ——曲率取反时整条曲线翻过来，两端仍被拉回正确值，峰值大小也不变。
    第一版就把符号写反了，那几条闭合解测试全是绿的。

    真正能定住符号的是端部转角：它不参与定常数，是独立的。悬臂根部的
    v′(0) 必须为零、自由端的 v′(L) 必须等于该端的节点转角。符号一反，
    这两条立刻就红。`test_deflection.py` 里那条测试就是干这个的。
    """
    member = frame.members[member_id]
    name = case or solution.primary
    d = member_diagram(frame, solution, member_id, name, stations=stations)
    sec = frame.sections[member.section]
    mat = frame.materials[member.material]

    pi, pj = member_endpoints(frame, member)
    _, rot = local_axes(pi, pj, member.ref_vector)
    res = solution[name]
    q = _result_local_displacements(frame, solution, member, name)
    ui, uj = q[:3], q[6:9]

    x = d.x
    L = d.length

    def integrate(moment: np.ndarray, shear: np.ndarray, EI: float,
                  shear_stiffness: float | None, v0: float, vL: float,
                  sign: float) -> np.ndarray:
        # θ′ = sign·M/EI；Timoshenko 梁的中心线斜率还包含 γ=V/(GAs)。
        curvature = sign * moment / EI
        slope = np.concatenate([[0.0], np.cumsum(
            0.5 * (curvature[1:] + curvature[:-1]) * np.diff(x))])
        if shear_stiffness is not None:
            slope = slope + shear / shear_stiffness
        raw = np.concatenate([[0.0], np.cumsum(
            0.5 * (slope[1:] + slope[:-1]) * np.diff(x))])
        # 两个常数由端点位移定：raw(0)=0，再线性修正到两端的真实位移
        return raw + v0 + (vL - v0 - raw[-1]) * (x / L if L else 0.0)

    # 局部 y 向位移由 Mz 控制，局部 z 向由 My 控制，两者符号相反——
    # 与刚度阵、质量阵、几何刚度阵、固端力里那几处同源
    v = integrate(d.Mz, d.Vy, mat.E * sec.Iz,
                  mat.G * sec.Ay if sec.Ay is not None else None,
                  float(ui[1]), float(uj[1]), 1.0)
    w = integrate(d.My, d.Vz, mat.E * sec.Iy,
                  mat.G * sec.Az if sec.Az is not None else None,
                  float(ui[2]), float(uj[2]), -1.0)
    return x, v, w


def member_displacement(frame: Frame, solution, member_id: int,
                        case: str | None = None, stations: int = 101):
    """杆件中心线的完整全局位移，返回 ``(x, displacement[n,3])``。

    横向 v/w 来自单元内曲率积分，已经包含两端节点横移；轴向位移在线性梁
    单元内插值。所有后处理都应走这个入口，避免桌面、PNG、Plotly 各自拼一遍
    位移后出现“某个视图对、另一个视图错”。
    """
    member = frame.members[member_id]
    name = case or solution.primary
    x, v, w = member_deflection(frame, solution, member_id, name, stations)
    pi, pj = member_endpoints(frame, member)
    _, rot = local_axes(pi, pj, member.ref_vector)
    q = _result_local_displacements(frame, solution, member, name)
    ui, uj = q[:3], q[6:9]
    ratio = x / x[-1] if x[-1] else np.zeros_like(x)
    axial = (1.0 - ratio) * ui[0] + ratio * uj[0]
    displacement = (np.outer(axial, rot[0])
                    + np.outer(v, rot[1])
                    + np.outer(w, rot[2]))
    return x, displacement


def max_centerline_displacement(frame: Frame, solution,
                                case: str | None = None,
                                stations: int = 101) -> dict:
    """全结构杆件中心线最大合位移，包含节点之间的挠曲。"""
    name = case or solution.primary
    best: dict | None = None
    for mid in sorted(frame.members):
        x, displacement = member_displacement(
            frame, solution, mid, name, stations)
        magnitude = np.linalg.norm(displacement, axis=1)
        k = int(np.argmax(magnitude))
        got = {"member": mid, "x": float(x[k]),
               "value": float(magnitude[k]),
               "vector": displacement[k].copy()}
        if best is None or got["value"] > best["value"]:
            best = got
    return best or {"member": None, "x": 0.0, "value": 0.0,
                    "vector": np.zeros(3)}


def max_deflection(frame: Frame, solution, case: str | None = None,
                   stations: int = 101, mapping=None) -> dict:
    """全结构沿杆长的最大横向挠度，含单元内部。

    与只看节点的查询相比，这个数才是校核 L/400 时该用的。
    """
    name = case or solution.primary
    # 初值必须取自**第一根杆件**，不能用 {"value": 0.0} 起头：纯轴向受力的
    # 结构横向挠度处处为零，"大于 0" 永远不成立，best 会一直是空的，
    # 上层就会报"没有杆件"——一个既错又难懂的消息
    best: dict | None = None
    member_ids = (sorted(mapping.physical_to_elements) if mapping is not None
                  else sorted(frame.members))
    for mid in member_ids:
        if mapping is None:
            x, v, w = member_deflection(frame, solution, mid, name, stations)
        else:
            element_ids = mapping.element_ids(mid)
            parts = [member_deflection(frame, solution, element_id, name, stations)
                     for element_id in element_ids]
            offsets = np.cumsum([0.0] + [part[0][-1] for part in parts[:-1]])
            x = np.concatenate([part[0] + offset
                                for part, offset in zip(parts, offsets)])
            v = np.concatenate([part[1] for part in parts])
            w = np.concatenate([part[2] for part in parts])
        mag = np.hypot(v, w)
        k = int(np.argmax(mag))
        got = {"member": mid, "x": float(x[k]), "value": float(mag[k]),
               "length": float(x[-1])}
        if best is None or got["value"] > best["value"]:
            best = got
    return best or {"member": None, "x": 0.0, "value": 0.0, "length": 0.0}
