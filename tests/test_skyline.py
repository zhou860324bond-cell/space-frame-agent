"""一维变带宽存储、配套求解与节点编号优化（讲义 §3-9 八、§3-10、§4-6）。

主链路用 SciPy 稀疏 LU，这一条是**教学对照路径**：把讲义的存储与解法原样实现，
再证明两条路给出同一个答案。所以这一组测试的核心判据只有一句——

    换了存储和解法，位移、反力、杆端力必须一致。

其余各条盯的是存储格式本身：MAXA 指针、列高、存储量，都用手算得出的小例子钉住。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from frame3d import assemble, constrained_dofs, solve
from model_compiler import compile_model
from numbering import (bandwidth_report, estimated_half_bandwidth,
                       matrix_bandwidth, node_number_span, permute, profile,
                       rcm_order)
from skyline import Skyline, SkylineSingular, factory


# --------------------------------------------------------------- 存储格式

# 一个手算得出的 4×4 变带宽例子。列高分别是 0、1、0、2：
#   第 3 列（从 0 数）最高非零元在第 1 行，所以要存 3 个数。
_SAMPLE = np.array([
    [4.0, 1.0, 0.0, 0.0],
    [1.0, 5.0, 0.0, 2.0],
    [0.0, 0.0, 6.0, 3.0],
    [0.0, 2.0, 3.0, 7.0],
])


def test_maxa_and_entries_match_the_hand_worked_example():
    s = Skyline.from_matrix(_SAMPLE)
    assert [s.column_height(j) for j in range(4)] == [0, 1, 0, 2]
    # MAXA 是各列对角元在一维数组里的下标：0, 1, 3, 4, 7
    assert list(s.maxa) == [0, 1, 3, 4, 7]
    assert s.entries == 7                      # 满阵 16，等带宽 4×3=12
    assert s.half_bandwidth == 3


def test_storage_comparison_orders_the_three_schemes():
    got = Skyline.from_matrix(_SAMPLE).storage()
    assert got["skyline"] <= got["banded"] <= got["full"]
    assert got == {**got, "n": 4, "half_bandwidth": 3,
                   "full": 16, "banded": 12, "skyline": 7}


def test_the_stored_matrix_reads_back_element_by_element():
    s = Skyline.from_matrix(_SAMPLE)
    assert np.array_equal(s.to_dense(), _SAMPLE)
    for i in range(4):
        for j in range(4):
            assert s[i, j] == _SAMPLE[i, j]


def test_a_sparse_matrix_gives_the_same_storage_as_the_dense_one():
    a = Skyline.from_matrix(_SAMPLE)
    b = Skyline.from_matrix(csr_matrix(_SAMPLE))
    assert list(a.maxa) == list(b.maxa)
    assert np.array_equal(a.a, b.a)


# --------------------------------------------------------------- 分解与回代

def _spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, n))
    return b @ b.T + n * np.eye(n)


@pytest.mark.parametrize("n", [1, 2, 5, 17, 40])
def test_ldlt_solves_the_same_system_as_numpy(n):
    K = _spd(n, seed=n)
    rhs = np.random.default_rng(n).normal(size=n)
    got = Skyline.from_matrix(K).solve(rhs)
    assert got == pytest.approx(np.linalg.solve(K, rhs), rel=1e-10, abs=1e-12)


def test_multiple_right_hand_sides_share_one_factorisation():
    K = _spd(12, seed=3)
    rhs = np.random.default_rng(3).normal(size=(12, 4))
    s = Skyline.from_matrix(K)
    got = s.solve(rhs)
    assert s.factored
    assert got == pytest.approx(np.linalg.solve(K, rhs), rel=1e-10, abs=1e-12)


def test_a_singular_matrix_raises_instead_of_returning_garbage():
    K = np.array([[1.0, 1.0], [1.0, 1.0]])       # 秩亏，第二个主元为零
    with pytest.raises(SkylineSingular, match="非正主元"):
        Skyline.from_matrix(K).factor()


def test_variable_bandwidth_really_is_variable():
    """一根链 + 一个远端连接：列高不等，等带宽方案会浪费一大片。"""
    n = 30
    K = np.eye(n) * 4.0
    for i in range(n - 1):
        K[i, i + 1] = K[i + 1, i] = -1.0
    K[0, n - 1] = K[n - 1, 0] = -0.5             # 一根长弦，把最大带宽顶满
    s = Skyline.from_matrix(K)
    assert s.half_bandwidth == n                 # 等带宽被这一根弦拖到满阵
    assert s.entries < n * s.half_bandwidth      # 变带宽只为那一列多花钱
    rhs = np.arange(n, dtype=float)
    assert Skyline.from_matrix(K).solve(rhs) == pytest.approx(
        np.linalg.solve(K, rhs), rel=1e-10)


# --------------------------------------------------------------- 编号

def test_rcm_never_makes_a_shuffled_numbering_worse():
    rng = np.random.default_rng(7)
    n = 50
    A = np.eye(n)
    for _ in range(120):
        i, j = rng.integers(0, n, 2)
        A[i, j] = A[j, i] = 1.0
    K = csr_matrix(A)
    order = rcm_order(K)
    assert sorted(order.tolist()) == list(range(n)), "必须是一个排列"
    assert matrix_bandwidth(permute(K, order)) <= matrix_bandwidth(K)


def test_rcm_recovers_the_natural_order_of_a_shuffled_chain():
    """一条链被打乱编号后，RCM 应当把它排回（正或反）自然次序。"""
    n = 12
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    A = np.eye(n)
    for i in range(n - 1):
        a, b = perm[i], perm[i + 1]
        A[a, b] = A[b, a] = 1.0
    K = csr_matrix(A)
    assert matrix_bandwidth(K) > 2               # 打乱后带宽很大
    assert matrix_bandwidth(permute(K, rcm_order(K))) == 2


def test_the_lecture_side_report_counts_node_number_differences():
    payload = _portal()
    frame = compile_model(payload).analysis_model
    assert node_number_span(frame) >= 1
    assert estimated_half_bandwidth(frame) == 6 * (node_number_span(frame) + 1)


def test_bandwidth_report_shows_both_bandwidth_and_profile():
    K = csr_matrix(_spd(20, seed=5))
    got = bandwidth_report(K)
    assert got["n"] == 20 and got["full"] == 400
    assert got["profile"]["before"] == profile(K)
    assert set(got["half_bandwidth"]) == {"before", "after"}


# --------------------------------------------------------------- 与主求解器对表

def _portal() -> dict:
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": 2.06e11, "nu": 0.3}],
        "sections": [{"name": "S", "A": 1.0e-2, "Iy": 1.0e-4, "Iz": 8.0e-5,
                      "J": 1.6e-4}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 0, "y": 0, "z": 4.0},
                  {"id": 3, "x": 6.0, "y": 0, "z": 4.0},
                  {"id": 4, "x": 6.0, "y": 0, "z": 0},
                  {"id": 5, "x": 6.0, "y": 5.0, "z": 4.0},
                  {"id": 6, "x": 6.0, "y": 5.0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M",
                     "ref_vector": [1, 0, 0]},
                    {"id": 2, "i": 2, "j": 3, "section": "S", "material": "M"},
                    {"id": 3, "i": 4, "j": 3, "section": "S", "material": "M",
                     "ref_vector": [1, 0, 0]},
                    {"id": 4, "i": 3, "j": 5, "section": "S", "material": "M"},
                    {"id": 5, "i": 6, "j": 5, "section": "S", "material": "M",
                     "ref_vector": [1, 0, 0]}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": 4, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": 6, "fix": [1, 1, 1, 1, 1, 1]}],
        "load_cases": [
            {"name": "D", "member_loads": [{"member": 2, "w": [0, 0, -20e3]},
                                           {"member": 4, "w": [0, 0, -15e3]}]},
            {"name": "W", "nodal_loads": [{"node": 2,
                                           "load": [30e3, 0, 0, 0, 0, 0]}]}],
    }


@pytest.mark.parametrize("reorder", [True, False])
def test_the_skyline_path_reproduces_the_sparse_answer(reorder):
    """本组的核心判据：换存储换解法，答案不变。

    位移、反力、杆端力全部对表——只对位移是不够的，反力和杆端力是在解出位移
    之后再算的，装配或消元哪里被动了手脚，它们才会露出来。
    """
    frame = compile_model(_portal()).analysis_model
    reference = solve(frame)
    got = solve(frame, factorize=factory(reorder=reorder))

    for name in reference.all_results():
        a, b = reference[name], got[name]
        scale = max(np.max(np.abs(a.U)), 1e-30)
        assert np.max(np.abs(a.U - b.U)) / scale < 1e-9
        rscale = max(np.max(np.abs(a.R)), 1e-30)
        assert np.max(np.abs(a.R - b.R)) / rscale < 1e-9
        for mid in frame.members:
            fa, fb = a.member_forces[mid], b.member_forces[mid]
            fscale = max(np.max(np.abs(fa)), 1e-30)
            assert np.max(np.abs(fa - fb)) / fscale < 1e-9


def test_reordering_does_not_touch_the_users_node_numbers():
    """重编号只活在求解内部。用户的节点号一个都不许变。"""
    frame = compile_model(_portal()).analysis_model
    before = list(frame.nodes)
    solve(frame, factorize=factory(reorder=True))
    assert list(frame.nodes) == before


def test_the_skyline_stores_far_less_than_the_full_matrix():
    frame = compile_model(_portal()).analysis_model
    K, _, _ = assemble(frame)
    fixed = constrained_dofs(frame)
    free = np.setdiff1d(np.arange(frame.num_dofs), fixed)
    Kff = K[free][:, free]
    got = Skyline.from_matrix(permute(Kff, rcm_order(Kff))).storage()
    assert got["skyline"] < got["banded"] <= got["full"]
    assert got["skyline_over_full"] < 1.0


# ------------------------------------------- 编号到底值多少钱（讲义 §3-9 八）

def _renumber(payload: dict, mapping: dict[int, int]) -> dict:
    """换一套节点号。结构、荷载、答案全不变，只有自由度编号变了。"""
    out = dict(payload)
    out["nodes"] = [{**n, "id": mapping[int(n["id"])]} for n in payload["nodes"]]
    out["members"] = [{**m, "i": mapping[int(m["i"])], "j": mapping[int(m["j"])]}
                      for m in payload["members"]]
    out["supports"] = [{**s, "node": mapping[int(s["node"])]}
                       for s in payload["supports"]]
    out["load_cases"] = [
        {**c, **({"nodal_loads": [{**e, "node": mapping[int(e["node"])]}
                                  for e in c["nodal_loads"]]}
                 if c.get("nodal_loads") else {})}
        for c in payload.get("load_cases", [])]
    return out


def _reduced(payload: dict):
    frame = compile_model(payload).analysis_model
    K, _, _ = assemble(frame)
    fixed = constrained_dofs(frame)
    free = np.setdiff1d(np.arange(frame.num_dofs), fixed)
    return frame, K[free][:, free]


def _tower() -> dict:
    """两跨五层、带一个进深的框架。

    要够大才看得出编号的影响：门式那六个节点、甚至单榀平面框架都太小，
    打乱编号也就差个百分之几十，试不出讲义那条判据。
    """
    from generator import generate_frame
    g = generate_frame(spans=[6.0, 6.0], storeys=[3.6] * 5, bays=[5.0],
                       column_section="C", beam_section="B", material="M")
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": 2.06e11, "nu": 0.3}],
        "sections": [{"name": "C", "A": 1.2e-2, "Iy": 1.6e-4, "Iz": 1.6e-4,
                      "J": 3.2e-4},
                     {"name": "B", "A": 8.0e-3, "Iy": 1.0e-4, "Iz": 6.0e-5,
                      "J": 1.2e-4}],
        "nodes": g["nodes"], "members": g["members"], "supports": g["supports"],
        "load_cases": [{"name": "D", "member_loads":
                        [{"member": m["id"], "w": [0, 0, -12e3]}
                         for m in g["members"]]}],
    }


def _shuffle_ids(payload: dict, seed: int = 3) -> dict:
    ids = [int(n["id"]) for n in payload["nodes"]]
    mixed = list(ids)
    np.random.default_rng(seed).shuffle(mixed)
    return _renumber(payload, dict(zip(ids, mixed, strict=True)))


def test_a_bad_numbering_costs_real_storage_and_rcm_gets_it_back():
    """讲义 §3-9 八的判据，用真结构量一遍。

    换一套节点号，结构和答案一个数都不变，但变带宽存储量能差好几倍。
    RCM 应当把它基本拉回到自然编号的水平——这就是「编号优化」的全部价值。
    """
    good = _tower()
    bad = _shuffle_ids(good)

    frame_good, K_good = _reduced(good)
    frame_bad, K_bad = _reduced(bad)

    # 先确认这真的是同一个结构：位移的**集合**必须一样（只是排列不同）
    a = np.sort(np.abs(solve(frame_good)["D"].U))
    b = np.sort(np.abs(solve(frame_bad)["D"].U))
    assert a == pytest.approx(b, rel=1e-9, abs=1e-15)

    assert profile(K_bad) > 1.5 * profile(K_good), "打乱编号本该明显更亏"
    fixed = profile(permute(K_bad, rcm_order(K_bad)))
    assert fixed < profile(K_bad)
    assert fixed <= profile(K_good) * 1.2, "RCM 应当基本拉回自然编号的水平"


def test_the_answer_does_not_depend_on_the_numbering():
    """换编号只该改存储，不该改答案。这条一挂，说明装配里有对编号的隐含依赖。"""
    good = _tower()
    bad = _shuffle_ids(good)
    ref = solve(compile_model(good).analysis_model, factorize=factory())
    got = solve(compile_model(bad).analysis_model, factorize=factory())
    # 杆件号没变，杆端力应当逐根对得上
    for mid in sorted(compile_model(good).analysis_model.members):
        fa, fb = ref["D"].member_forces[mid], got["D"].member_forces[mid]
        assert np.max(np.abs(fa - fb)) / max(np.max(np.abs(fa)), 1e-30) < 1e-9
