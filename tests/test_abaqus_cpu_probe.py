"""对照网格生成器的验证。

上一个探针给出了错误结论，因为**探针自己有问题**（工作目录引入了中文路径
这个额外变量）。这次的探针靠一个程序生成的规则网格当对照：如果那个网格本身
是坏的，"P3 也崩了"就会被误读成"CPU 有问题"，而真相只是我造了个烂网格。

所以生成器必须先自证：四面体不重不漏地填满立方体、没有负体积、中间点确实
在边中点上、所有单元全等。这些都不需要装 Abaqus。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from abaqus_cpu_probe import kuhn_c3d10_box   # noqa: E402

# C3D10 的中间点对应的角点对：5=(1,2) 6=(2,3) 7=(1,3) 8=(1,4) 9=(2,4) 10=(3,4)
MIDSIDE_PAIRS = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))


def _parse(deck: str):
    coordinates: dict[int, tuple[float, float, float]] = {}
    elements: list[list[int]] = []
    mode = None
    for line in deck.splitlines():
        if line.startswith("*"):
            head = line.split(",")[0].strip().upper()
            # 注意 '*NODE PRINT' 也以 *NODE 开头，必须整词比较
            mode = "N" if head == "*NODE" else ("E" if head == "*ELEMENT" else None)
            continue
        if mode == "N":
            parts = [item.strip() for item in line.split(",")]
            coordinates[int(parts[0])] = tuple(float(v) for v in parts[1:4])
        elif mode == "E":
            parts = [int(item) for item in line.split(",")]
            elements.append(parts[1:])
    return coordinates, elements


def _volume(points) -> float:
    a, b, c, d = (np.array(p) for p in points)
    return float(np.dot(b - a, np.cross(c - a, d - a)) / 6.0)


@pytest.fixture(scope="module")
def mesh():
    deck, nodes, elements = kuhn_c3d10_box()
    coordinates, connectivity = _parse(deck)
    assert len(coordinates) == nodes
    assert len(connectivity) == elements
    return deck, coordinates, connectivity


def test_no_element_has_a_negative_or_zero_volume(mesh):
    """负体积单元 Abaqus 直接报错，那样这个对照就白做了。"""
    _deck, coordinates, elements = mesh
    volumes = [_volume([coordinates[n] for n in element[:4]])
               for element in elements]
    assert min(volumes) > 0.0


def test_every_element_is_congruent(mesh):
    """对照的全部价值就在"零畸变"上。体积不全等就说明剖分不规则，
    那它就不再是"形状完美"的对照了。"""
    _deck, coordinates, elements = mesh
    volumes = [_volume([coordinates[n] for n in element[:4]])
               for element in elements]
    assert max(volumes) - min(volumes) == pytest.approx(0.0, abs=1e-9)


def test_the_tetrahedra_tile_the_box_exactly(mesh):
    """不重不漏：体积之和必须精确等于立方体体积。
    少了是有洞，多了是重叠——两种都会让结果没法解释。"""
    _deck, coordinates, elements = mesh
    volumes = [_volume([coordinates[n] for n in element[:4]])
               for element in elements]
    assert sum(volumes) == pytest.approx(105.0 ** 3, rel=1e-12)


def test_midside_nodes_really_sit_at_the_edge_midpoints(mesh):
    """二次单元的中间点位置错了，单元照样能算，但结果是错的——
    而且不会报任何错。"""
    _deck, coordinates, elements = mesh
    for element in elements:
        corners = [np.array(coordinates[n]) for n in element[:4]]
        for index, (i, j) in enumerate(MIDSIDE_PAIRS):
            expected = 0.5 * (corners[i] + corners[j])
            got = np.array(coordinates[element[4 + index]])
            assert np.allclose(expected, got, atol=1e-6)


def test_the_deck_is_the_same_size_as_the_job_that_crashed(mesh):
    """对照要有意义，规模必须和失败的算例相当。
    失败算例是 3725 节点 / 1837 单元 / 10731 方程。"""
    _deck, coordinates, elements = mesh
    assert 2500 <= len(coordinates) <= 5000
    assert 3 * len(coordinates) == pytest.approx(10731, rel=0.25)


def test_no_element_references_a_node_that_does_not_exist(mesh):
    _deck, coordinates, elements = mesh
    for element in elements:
        assert len(element) == 10
        for node in element:
            assert node in coordinates


def test_node_numbering_is_dense_and_starts_at_one(mesh):
    """Abaqus 允许稀疏编号，但空洞往往是生成器串位的征兆。"""
    _deck, coordinates, _elements = mesh
    assert sorted(coordinates) == list(range(1, len(coordinates) + 1))


def test_the_deck_constrains_one_face_and_loads_another(mesh):
    """没有约束就是刚体位移，求解器会报奇异——那也不是我们想测的东西。"""
    deck, _coordinates, _elements = mesh
    assert "*BOUNDARY" in deck and "BASE, 1, 3" in deck
    assert "*CLOAD" in deck and "TOP, 1," in deck
    assert deck.count("*NSET, NSET=BASE") == 1
    assert deck.count("*NSET, NSET=TOP") == 1
