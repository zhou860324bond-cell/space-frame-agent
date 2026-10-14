"""初始几何缺陷：把屈曲模态按幅值叠到分析模型的节点坐标上。

线性屈曲给的是完善结构的分岔荷载，是上限；真实钢结构有初始弯曲、安装偏差，
承载力明显低于它。工程上的通行做法（Abaqus 的 ``*IMPERFECTION``、GB 50017
的直接分析法）是：取最低阶屈曲模态作缺陷形状，缩放到给定幅值叠加到几何上，
再做二阶分析。缺陷沿着最容易失稳的方向，所以这是最不利的那一种。

只改**分析模型**的节点坐标，用户的物理模型不动；节点编号、单元连接与编译映射
都不变，结果照常聚合回物理杆件。二阶分析报的位移是相对于有缺陷构形的**附加**
位移，与 Abaqus 的约定一致。

幅值是长度，单位跟模型走。规范对整体缺陷和构件缺陷各有取值，按所用规范自行
确定后填入；这里不替用户选规范。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from buckling import buckling
from frame3d import DOF_PER_NODE, Frame, solve


#: 带缺陷二阶分析的最少分段数，比屈曲、频率用的 4 段更多。
#:
#: 缺陷形状靠折线逼近，二阶分析对它比特征值分析敏感：两端铰支压杆、正弦缺陷、
#: P = 0.5·Pcr 时，跨中附加挠度与闭合解之比随每杆段数为 4 段 0.951、8 段 0.987、
#: 16 段 0.997、32 段 0.999（O(h²) 收敛，见 tests/test_imperfection.py）。偏小即
#: 偏不安全，所以默认 8 段；要更准用 seed_mesh 加密。
MIN_DIVISIONS = 8


def reference_case(model: Frame, solution, case: str | None) -> str:
    """缺陷模态用哪个工况的屈曲模态：指定了就用它，否则取内力最大的工况或组合。"""
    if case:
        return case
    return max(solution.all_results().values(),
               key=lambda r: max(abs(f).max() for f in r.member_forces.values())).name


def apply_mode_imperfection(model: Frame, mode: int, amplitude: float,
                            case: str | None = None) -> tuple[Frame, dict[str, Any]]:
    """返回 (叠加了缺陷的分析模型副本, 说明)。

    模态按**平动**分量的最大节点合位移归一到 ``|amplitude|``；``amplitude`` 取负
    即反向（模态本身的正负号是任意的）。转动分量不改坐标——坐标没有转角。
    """
    if mode < 1:
        raise ValueError(f"缺陷模态阶数从 1 开始，收到 {mode}")
    if amplitude == 0 or not np.isfinite(amplitude):
        raise ValueError(f"缺陷幅值必须是非零有限数，收到 {amplitude}")
    linear = solve(model)
    name = reference_case(model, linear, case)
    result = buckling(model, name, num_modes=max(mode, 1), solution=linear)
    if result.shapes.shape[1] < mode:
        raise ValueError(f"只求出 {result.shapes.shape[1]} 阶屈曲模态，没有第 {mode} 阶")

    shape = result.shapes[:, mode - 1]
    index = model.index_of()
    moves = {nid: shape[DOF_PER_NODE * k:DOF_PER_NODE * k + 3]
             for nid, k in index.items()}
    peak_node, peak = max(((nid, float(np.linalg.norm(v))) for nid, v in moves.items()),
                          key=lambda item: item[1])
    if peak <= 0.0:
        raise ValueError(f"第 {mode} 阶屈曲模态没有平动分量，不能用作几何缺陷")
    scale = float(amplitude) / peak

    perturbed = deepcopy(model)
    for nid, delta in moves.items():
        node = perturbed.nodes[nid]
        x, y, z = node.xyz + scale * delta
        perturbed.nodes[nid] = replace(node, x=float(x), y=float(y), z=float(z))
    info = {"mode": int(mode), "case": name,
            "buckling_factor": float(result.factors[mode - 1]),
            "amplitude": float(amplitude), "peak_node": int(peak_node)}
    return perturbed, info
