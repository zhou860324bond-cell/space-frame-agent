"""内核提速路径的等价性。

三处提速都不该改变任何数值结果，这里逐一钉住：

1. 节点位次缓存——任何增删改都必须作废，否则自由度错位、结果安静地错；
2. 批量散射装配——与逐项累加的原写法逐位一致，重复位置要相加；
3. 大模型的稀疏特征值路径——与稠密 eigh 给出同样的频率、屈曲因子与振型。
"""

import copy
import pickle

import numpy as np
import pytest

import buckling as buckling_mod
import modal as modal_mod
from frame3d import (DEFAULT_CASE, Frame, Material, Member, Node, Section,
                     scatter_blocks, solve)


def small_building(nx=2, ny=2, nz=3) -> Frame:
    """规则多层框架：柱底固接、梁上满跨竖向荷载，含一根端铰与刚域杆。"""
    f = Frame()
    f.sections["S"] = Section("S", A=1e-2, Iy=2e-4, Iz=1e-4, J=5e-5)
    f.materials["Q"] = Material("Q", E=2.06e11, nu=0.3, density=7850.0)

    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                f.nodes[nid(i, j, k)] = Node(nid(i, j, k), 4.0 * i, 5.0 * j, 3.0 * k)
                if k == 0:
                    f.supports[nid(i, j, k)] = (1,) * 6
    mid = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                pairs = []
                if k < nz:
                    pairs.append((nid(i, j, k), nid(i, j, k + 1), False))
                if k > 0 and i < nx:
                    pairs.append((nid(i, j, k), nid(i + 1, j, k), True))
                if k > 0 and j < ny:
                    pairs.append((nid(i, j, k), nid(i, j + 1, k), True))
                for a, b, beam in pairs:
                    mid += 1
                    f.members[mid] = Member(mid, a, b, "S", "Q")
                    if beam:
                        f.member_loads[mid] = (0.0, 0.0, -2e4)
    top = nid(0, 0, nz)
    f.members[mid + 1] = Member(mid + 1, top, nid(nx, ny, nz), "S", "Q",
                                releases_j=("ry", "rz"), offset_i=(0.2, 0.0, 0.0))
    return f


# --------------------------------------------------------------------------- 位次缓存

def test_inserting_a_lower_node_id_shifts_the_dofs_of_later_nodes():
    f = Frame(nodes={10: Node(10, 0, 0, 0), 20: Node(20, 1, 0, 0)})
    assert f.node_dofs(20) == list(range(6, 12))
    f.nodes[5] = Node(5, -1, 0, 0)
    assert f.node_dofs(20) == list(range(12, 18))
    del f.nodes[5]
    assert f.node_dofs(20) == list(range(6, 12))
    f.nodes.pop(10)
    assert f.node_dofs(20) == list(range(0, 6))


@pytest.mark.parametrize("mutate", [
    lambda d: d.update({1: Node(1, 0, 0, 0)}),
    lambda d: d.setdefault(1, Node(1, 0, 0, 0)),
    lambda d: d.__ior__({1: Node(1, 0, 0, 0)}),
])
def test_every_dict_mutator_invalidates_the_rank_cache(mutate):
    f = Frame(nodes={10: Node(10, 0, 0, 0)})
    assert f.node_dofs(10) == list(range(6))
    mutate(f.nodes)
    assert f.node_dofs(10) == list(range(6, 12))


def test_clear_and_popitem_invalidate_the_rank_cache():
    f = Frame(nodes={1: Node(1, 0, 0, 0), 2: Node(2, 1, 0, 0)})
    f.node_dofs(1)
    f.nodes.popitem()
    assert f.index_of() == {1: 0}
    f.nodes.clear()
    assert f.index_of() == {}


def test_reassigning_the_node_dict_keeps_the_cache_correct():
    f = Frame(nodes={10: Node(10, 0, 0, 0)})
    f.node_dofs(10)
    f.nodes = {3: Node(3, 0, 0, 0), 10: Node(10, 1, 0, 0)}
    assert f.node_dofs(10) == list(range(6, 12))


def test_copies_do_not_share_a_stale_cache():
    f = Frame(nodes={10: Node(10, 0, 0, 0)})
    f.node_dofs(10)
    for twin in (copy.deepcopy(f), pickle.loads(pickle.dumps(f))):
        twin.nodes[1] = Node(1, 0, 0, 0)
        assert twin.node_dofs(10) == list(range(6, 12))
        assert f.node_dofs(10) == list(range(6))


def test_index_of_hands_out_a_copy():
    f = Frame(nodes={1: Node(1, 0, 0, 0)})
    f.index_of()[1] = 99
    assert f.node_dofs(1) == list(range(6))


# --------------------------------------------------------------------------- 散射装配

def test_scatter_matches_the_elementwise_loop_including_shared_dofs():
    rng = np.random.default_rng(0)
    n = 30
    all_dofs = [list(rng.choice(n, 12, replace=False)) for _ in range(8)]
    all_dofs.append(list(all_dofs[0]))          # 整块重叠，必须相加
    blocks = [rng.standard_normal((12, 12)) for _ in all_dofs]
    expected = np.zeros((n, n))
    for dofs, ke in zip(all_dofs, blocks):
        for a in range(12):
            for b in range(12):
                expected[dofs[a], dofs[b]] += ke[a, b]
    assert np.allclose(scatter_blocks(n, all_dofs, blocks).toarray(), expected,
                       rtol=0, atol=1e-12)


def test_scatter_of_nothing_is_an_empty_matrix():
    K = scatter_blocks(7, [], [])
    assert K.shape == (7, 7) and K.nnz == 0


# --------------------------------------------------------------------------- 稀疏特征值

def _mac(a: np.ndarray, b: np.ndarray) -> float:
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_sparse_modal_path_matches_dense(monkeypatch):
    f = small_building()
    dense = modal_mod.modal(f, 6)
    monkeypatch.setattr(modal_mod, "DENSE_EIGEN_LIMIT", 0)
    sparse = modal_mod.modal(f, 6)
    assert np.allclose(sparse.frequencies, dense.frequencies, rtol=1e-9)
    assert np.allclose(sparse.effective_mass, dense.effective_mass,
                       rtol=0, atol=1e-8 * dense.total_mass)
    for k in range(6):
        assert _mac(sparse.shapes[:, k], dense.shapes[:, k]) > 1 - 1e-8
        # 与 eigh(K, M) 同样按质量归一化，振型幅值才能跨路径比较
        assert np.linalg.norm(sparse.shapes[:, k]) == pytest.approx(
            np.linalg.norm(dense.shapes[:, k]), rel=1e-8)


def test_sparse_buckling_path_matches_dense(monkeypatch):
    f = small_building()
    sol = solve(f)
    dense = buckling_mod.buckling(f, DEFAULT_CASE, num_modes=3, solution=sol)
    monkeypatch.setattr(buckling_mod, "DENSE_EIGEN_LIMIT", 0)
    sparse = buckling_mod.buckling(f, DEFAULT_CASE, num_modes=3, solution=sol)
    assert np.allclose(sparse.factors, dense.factors, rtol=1e-9)
    for k in range(3):
        assert _mac(sparse.shapes[:, k], dense.shapes[:, k]) > 1 - 1e-8
