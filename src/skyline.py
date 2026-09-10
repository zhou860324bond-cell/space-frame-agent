"""总刚一维变带宽存储与配套求解（讲义 §3-10、§4-6）。

讲义用两节课讲存储：等带宽带状矩阵（§3-10）与一维变带宽存储（§4-6）。
本项目主链路用的是 SciPy 稀疏 LU——工程上更好，但**「有更好的方法」不等于
「实现了讲授内容」**。所以这里把讲义那条路原样实现一遍，作为**教学对照路径**，
不替换主求解器：

    frame3d.solve(model)                       ← 稀疏 LU（默认）
    frame3d.solve(model, factorize=factory())  ← 一维变带宽 LDLᵀ（本模块）

两条路径共用同一份装配、约束消元、沉降处理与反力回算，**只有分解这一步不同**。
这一点是刻意的：否则比出来的差异分不清是解法的还是装配的。

存储格式（Bathe 的 COLSOL 约定，也就是讲义里的 A 与 MAXA）
--------------------------------------------------------
对称矩阵只存上三角，且**按列存到该列最高的非零元为止**——列高不必相等，
这就是「变带宽」相对「等带宽」的全部区别：

    a[maxa[j] + k] = K[j-k, j]       k = 0 … 列高_j

``maxa[j]`` 是第 j 列对角元在一维数组里的下标，``maxa[j+1] - maxa[j]``
就是该列存了几个数。空结构 n=0 时 maxa 只有一个元素。

为什么值得做
------------
零元素也占位（列内的零照存不误），但**分解不会产生列高以外的填充**——
这正是这套存储成立的前提，也是讲义要讲它的原因。对刚架这种带状明显的问题，
存储量通常只有满阵的百分之几。``storage()`` 会把三种方案的存储量一起给出来。

不做的事
--------
不做主元交换。对称正定矩阵的 LDLᵀ 本来就不需要选主元；而结构刚度阵约束
充分时就是正定的。**遇到非正主元直接抛异常**，因为那意味着约束不足或存在
机构——此时安静地算下去只会得到一个看不出问题的错误答案。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import issparse

_PIVOT_RATIO = 1e-12


class SkylineSingular(np.linalg.LinAlgError):
    """分解过程中出现非正主元：矩阵不是正定的。"""


@dataclass
class Skyline:
    """一维变带宽存储的对称矩阵。

    ``a`` 一维值数组，``maxa`` 长度 n+1 的列指针（讲义的 MAXA）。
    """

    n: int
    a: np.ndarray
    maxa: np.ndarray
    factored: bool = False

    # ----------------------------------------------------------- 构造

    @classmethod
    def from_matrix(cls, K) -> "Skyline":
        """由稠密或稀疏矩阵建立。只读上三角——**默认矩阵是对称的**。"""
        dense = np.asarray(K.todense() if issparse(K) else K, dtype=float)
        if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
            raise ValueError("只接受方阵")
        n = dense.shape[0]
        heights = np.zeros(n, dtype=int)
        for j in range(n):
            column = dense[: j + 1, j]
            nz = np.nonzero(column)[0]
            # 列高 = 对角元到本列最高非零元的距离；整列全零时只存对角元
            heights[j] = j - int(nz[0]) if nz.size else 0
        maxa = np.zeros(n + 1, dtype=int)
        np.cumsum(heights + 1, out=maxa[1:])
        a = np.zeros(int(maxa[n]), dtype=float)
        for j in range(n):
            for k in range(heights[j] + 1):
                a[maxa[j] + k] = dense[j - k, j]
        return cls(n=n, a=a, maxa=maxa)

    # ----------------------------------------------------------- 访问

    def column_height(self, j: int) -> int:
        return int(self.maxa[j + 1] - self.maxa[j]) - 1

    def __getitem__(self, key: tuple[int, int]) -> float:
        i, j = key
        if i > j:
            i, j = j, i                      # 对称
        k = j - i
        if k > self.column_height(j):
            return 0.0
        return float(self.a[self.maxa[j] + k])

    def to_dense(self) -> np.ndarray:
        out = np.zeros((self.n, self.n))
        for j in range(self.n):
            for k in range(self.column_height(j) + 1):
                out[j - k, j] = out[j, j - k] = self.a[self.maxa[j] + k]
        return out

    # ----------------------------------------------------------- 规模

    @property
    def entries(self) -> int:
        """一维数组实际长度，也就是讲义说的「存储量」。"""
        return int(self.maxa[self.n])

    @property
    def half_bandwidth(self) -> int:
        """最大列高 + 1（含对角元），即等带宽方案必须取的半带宽。"""
        if self.n == 0:
            return 0
        return int(np.max(np.diff(self.maxa)))

    def storage(self) -> dict[str, Any]:
        """三种方案的存储量对比。等带宽必须按**最大**列高铺满每一列。"""
        n, nb = self.n, self.half_bandwidth
        full = n * n
        banded = n * nb
        return {"n": n, "half_bandwidth": nb,
                "full": full, "banded": banded, "skyline": self.entries,
                "skyline_over_full": (self.entries / full) if full else 0.0,
                "skyline_over_banded": (self.entries / banded) if banded else 0.0}

    # ----------------------------------------------------------- 分解与回代

    def factor(self) -> "Skyline":
        """就地 LDLᵀ 分解（讲义 §4-6 的分解子函数）。

        分解后 ``a`` 里对角位置存 D，非对角位置存 Lᵀ 的元素。
        """
        if self.factored:
            return self
        a, maxa = self.a, self.maxa
        for j in range(self.n):
            kj = int(maxa[j])
            hj = self.column_height(j)
            top_j = j - hj                       # 本列最高的行号
            # 先把第 j 列（对角元以上部分）化成 L 的元素。
            # 内层 Σ_r a[r,i]·a[r,j] 写成点积：本存储里行号 r 增大对应下标
            # 减小，两个切片都是**同向反转**的，反转对点积没有影响，
            # 于是直接拿正序切片相乘即可。
            for i in range(top_j, j):
                ki = int(maxa[i])
                start = max(top_j, i - self.column_height(i))
                if i > start:
                    a[kj + j - i] -= (a[ki + 1: ki + i - start + 1]
                                      @ a[kj + j - i + 1: kj + j - start + 1])
            # 再消对角元，并把 L 归一化：a[r,j] ← a[r,j]/d_r
            d = float(a[kj])
            if j > top_j:
                seg = a[kj + 1: kj + j - top_j + 1]          # r = j-1 … top_j
                diag = a[maxa[top_j:j]][::-1]                # 对齐到同样的次序
                c = seg.copy()
                seg /= diag
                d -= float(seg @ c)
            if not np.isfinite(d) or d <= _PIVOT_RATIO * max(abs(a[kj]), 1.0):
                raise SkylineSingular(
                    f"第 {j} 个自由度出现非正主元 {d:.3e}："
                    "刚度矩阵不是正定的，约束不足或存在机构。")
            a[kj] = d
        self.factored = True
        return self

    def solve(self, b) -> np.ndarray:
        """回代求解。未分解时先分解。``b`` 可以是向量或按列排的多个右端项。"""
        if not self.factored:
            self.factor()
        rhs = np.asarray(b, dtype=float)
        if rhs.ndim == 2:
            return np.column_stack([self.solve(rhs[:, c])
                                    for c in range(rhs.shape[1])])
        if rhs.shape[0] != self.n:
            raise ValueError(f"右端项长度 {rhs.shape[0]} 与阶数 {self.n} 不符")
        a, maxa = self.a, self.maxa
        v = rhs.astype(float).copy()
        for j in range(self.n):                        # 前代 L v = b
            kj, h = int(maxa[j]), self.column_height(j)
            if h:
                v[j] -= a[kj + 1: kj + h + 1] @ v[j - h: j][::-1]
        v /= a[maxa[: self.n]]                         # 对角 D
        for j in range(self.n - 1, -1, -1):            # 回代 Lᵀ u = v
            kj, h = int(maxa[j]), self.column_height(j)
            if h:
                v[j - h: j] -= a[kj + 1: kj + h + 1][::-1] * v[j]
        return v


@dataclass
class _Reordered:
    """重编号后的分解结果。对外仍然只是一个 ``.solve``。

    排列只活在这个对象里：右端项进来先按新次序重排，解出来再排回去。
    调用方（``frame3d.solve``）完全不知道发生过重编号，模型的节点号也没动过——
    这正是「不擅自改用户的编号」那条约定的落点。
    """

    inner: Skyline
    order: np.ndarray

    def solve(self, b) -> np.ndarray:
        rhs = np.asarray(b, dtype=float)
        if rhs.ndim == 2:
            return np.column_stack([self.solve(rhs[:, c])
                                    for c in range(rhs.shape[1])])
        out = np.empty_like(rhs)
        out[self.order] = self.inner.solve(rhs[self.order])
        return out


def factory(reorder: bool = True):
    """给 ``frame3d.solve(model, factorize=...)`` 用的工厂。

    用法::

        from skyline import factory
        solution = solve(model, factorize=factory())

    ``reorder=True`` 时先做 RCM 重编号再存储——讲义 §3-9 八与 §4-6 本来就是
    配套的两件事：编号决定带宽，带宽决定存储量。``reorder=False`` 保留原编号，
    用来演示「编号没排好时变带宽存储有多亏」。
    """
    def make(K):
        if not reorder:
            return Skyline.from_matrix(K).factor()
        from numbering import permute, rcm_order
        order = rcm_order(K)
        return _Reordered(Skyline.from_matrix(permute(K, order)).factor(), order)
    return make
