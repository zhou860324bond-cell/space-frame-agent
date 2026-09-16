"""节点编号与带宽（讲义 §3-9 八）。

讲义的原话是：节点编号要让**一根杆两端的节点号相差尽量小**，因为半带宽
``ibdw = max[2(j−i)+1]``（那是每节点 2 个自由度的平面桁架写法）直接由这个差值
决定，而带状/变带宽存储的存储量与计算量都正比于带宽。

这里做三件事：

1. 按讲义的判据算出当前编号的**最大节点号差**与**半带宽**；
2. 用 Cuthill–McKee 逆序（RCM）给出一个更好的编号，并算出改善了多少；
3. 把编号作用在**矩阵**上，供 ``skyline.py`` 的变带宽求解使用。

**不动用户的节点号。** 这是刻意的：节点号是用户和大模型在对话里一直引用的东西
（"3 号节点加个铰"），求解器擅自换掉，用户后面每一句话都会对错人。所以重编号
只发生在求解内部——排列作用在自由度上，算完再排回来，模型本身一个字节不改。
需要一份真正重编号的模型时，请显式生成，而不是让它悄悄发生。

关于 6 自由度
------------
讲义的 ``2(j−i)+1`` 是 2 自由度/节点的写法。本项目每节点 6 个自由度，
同一件事写成 ``6(j−i)+6``（含对角元）。更要紧的是：实际带宽应当以**约束消元后**
的自由度编号为准，支座把一些自由度整行整列去掉了，按节点号估出来的带宽只是
上界。所以 :func:`matrix_bandwidth` 直接看矩阵，:func:`node_number_span`
才是讲义那个可以在编号阶段就算出来的估计值。
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse

from frame3d import DOF_PER_NODE, Frame


# --------------------------------------------------------------- 图与带宽

def node_adjacency(frame: Frame) -> dict[int, set[int]]:
    """节点邻接表。杆件就是边——这就是编号问题里的全部拓扑。"""
    graph: dict[int, set[int]] = {nid: set() for nid in frame.nodes}
    for m in frame.members.values():
        if m.i == m.j:
            continue
        graph.setdefault(m.i, set()).add(m.j)
        graph.setdefault(m.j, set()).add(m.i)
    return graph


def node_number_span(frame: Frame) -> int:
    """讲义判据：一根杆两端节点号差的最大值 max|j−i|。

    编号优化要压小的就是这个数。按**位次**算而不是按 id 的数值：自由度是按
    节点号排序后的位次编排的（``Frame.order()``），所以 id 从 1 跳到 100
    并不会让带宽变大，真正要紧的是相邻杆件两端的位次差。
    """
    order = frame.index_of()          # 自由度按节点号排序后的位次编排
    return max((abs(order[m.i] - order[m.j]) for m in frame.members.values()),
               default=0)


def estimated_half_bandwidth(frame: Frame) -> int:
    """由节点号差估计的半带宽 ``6(j−i)+6``（含对角元）。

    这是**上界**：支座消元后真实带宽只会更小。讲义在编号阶段用的就是这个估计。
    """
    if not frame.members:
        return DOF_PER_NODE
    return DOF_PER_NODE * (node_number_span(frame) + 1)


def matrix_bandwidth(K) -> int:
    """矩阵的实际半带宽（含对角元）：max(j − i) + 1，只看非零元。"""
    A = K.tocoo() if issparse(K) else csr_matrix(np.asarray(K)).tocoo()
    if A.nnz == 0:
        return 1 if A.shape[0] else 0
    mask = A.data != 0.0
    if not mask.any():
        return 1
    return int(np.max(np.abs(A.row[mask] - A.col[mask]))) + 1


def profile(K) -> int:
    """变带宽存储量：各列列高之和（含对角元）。等价于 skyline 的 entries。"""
    A = (K.tocsc() if issparse(K) else csr_matrix(np.asarray(K)).tocsc())
    total = 0
    for j in range(A.shape[1]):
        rows = A.indices[A.indptr[j]:A.indptr[j + 1]]
        vals = A.data[A.indptr[j]:A.indptr[j + 1]]
        rows = rows[vals != 0.0]
        top = int(rows.min()) if rows.size else j
        total += j - min(top, j) + 1
    return total


# --------------------------------------------------------------- RCM

def _adjacency_from_matrix(K) -> list[np.ndarray]:
    A = K.tocsr() if issparse(K) else csr_matrix(np.asarray(K))
    return [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(A.shape[0])]


def _pseudo_peripheral(neighbours, degree, start: int) -> int:
    """找一个"边缘"起点：从任意点出发反复取最远层里度最小的点。

    起点选得好，层次结构才窄，带宽才小。这一步不做也能跑，只是效果差些。
    """
    current = start
    for _ in range(4):
        levels = _bfs_levels(neighbours, current)
        last = levels[-1]
        nxt = min(last, key=lambda v: (degree[v], v))
        if nxt == current:
            break
        current = nxt
    return current


def _bfs_levels(neighbours, root: int) -> list[list[int]]:
    seen = {root}
    levels = [[root]]
    while levels[-1]:
        nxt = []
        for v in levels[-1]:
            for w in neighbours[v]:
                if w not in seen:
                    seen.add(w)
                    nxt.append(int(w))
        if not nxt:
            break
        levels.append(nxt)
    return levels


def rcm_order(K) -> np.ndarray:
    """Cuthill–McKee 逆序。返回新次序下依次是哪个旧下标。

    自己写而不是调 SciPy：讲义要求的就是这一步，而且算法本身只有二十行——
    BFS，每层内按度数从小到大加入，最后整体反转。反转这一下是 RCM 相对
    朴素 CM 的全部区别，通常能再小一截带宽（对变带宽存储尤其明显）。
    """
    neighbours = _adjacency_from_matrix(K)
    n = len(neighbours)
    degree = np.array([len(neighbours[i]) for i in range(n)])
    seen = np.zeros(n, dtype=bool)
    order: list[int] = []

    for seed in np.argsort(degree, kind="stable"):
        if seen[seed]:
            continue
        root = _pseudo_peripheral(neighbours, degree, int(seed))
        if seen[root]:
            root = int(seed)
        seen[root] = True
        queue = deque([root])
        order.append(root)
        while queue:
            v = queue.popleft()
            fresh = [int(w) for w in neighbours[v] if not seen[w]]
            for w in sorted(fresh, key=lambda x: (degree[x], x)):
                if seen[w]:
                    continue
                seen[w] = True
                order.append(w)
                queue.append(w)
    return np.array(order[::-1], dtype=int)


def permute(K, order: np.ndarray):
    """把矩阵按新次序重排：返回 ``K[order][:, order]``。"""
    order = np.asarray(order, dtype=int)
    A = K.tocsr() if issparse(K) else csr_matrix(np.asarray(K))
    return A[order][:, order]


# --------------------------------------------------------------- 报告

def bandwidth_report(K) -> dict[str, Any]:
    """重编号前后的带宽与存储量对比。给报告和 Agent 用。"""
    order = rcm_order(K)
    after = permute(K, order)
    n = K.shape[0]
    before_bw, after_bw = matrix_bandwidth(K), matrix_bandwidth(after)
    before_pf, after_pf = profile(K), profile(after)
    return {
        "n": n,
        "order": order,
        "half_bandwidth": {"before": before_bw, "after": after_bw},
        "profile": {"before": before_pf, "after": after_pf},
        "full": n * n,
        "improvement": {
            "half_bandwidth": (before_bw / after_bw) if after_bw else 1.0,
            "profile": (before_pf / after_pf) if after_pf else 1.0,
        },
    }


def frame_report(frame: Frame) -> dict[str, Any]:
    """讲义视角的编号报告：只看节点号，不需要先装配矩阵。"""
    return {"nodes": len(frame.nodes), "members": len(frame.members),
            "node_number_span": node_number_span(frame),
            "estimated_half_bandwidth": estimated_half_bandwidth(frame),
            "note": "6(j−i)+6，含对角元；支座消元后实际带宽只会更小"}
