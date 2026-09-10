"""杆件截面朝向（局部 y/z 轴）的确定性。

这一组测试盯的是一个**不报错**的缺陷：竖直杆件的参考向量退化，兜底方向依赖于
"有多竖直"。杆件差 0.01° 和差 0.1° 会落到两个不同的分支，局部 y 反号、
局部 z 跟着反号，于是同一根柱子报出来的 My/Mz/Vy/Vz 全部反号。

方程照样有解，位移照样对，只有内力的符号悄悄变了——图画反、配筋配反侧。
这个不连续是本质的（竖直杆没有"向上投影"），所以修法不是让它连续，
而是让生成器把这个选择**显式写进模型**。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from frame3d import local_axes
from generator import _COLUMN_REF, generate_frame, generate_portal_frame
from model_tables import from_tables, to_tables


def _axes(tilt_deg: float, ref=None) -> np.ndarray:
    """一根近似竖直的杆，倾斜 tilt_deg 度，返回方向余弦阵。"""
    t = math.radians(tilt_deg)
    pi = np.array([0.0, 0.0, 0.0])
    pj = np.array([math.sin(t) * 3.0, 0.0, math.cos(t) * 3.0])
    return local_axes(pi, pj, ref)[1]


def test_the_fallback_really_does_flip_between_hundredths_of_a_degree():
    """先把缺陷本身钉住：不给参考向量时，0.01° 与 0.1° 的局部 y 是反的。

    这条是**特征化测试**，不是期望行为。它在这里的作用是：将来谁去动
    `_VERTICAL_TOL` 或兜底逻辑，这条会失败并把人带到这段注释。
    """
    y_small = _axes(0.01)[1]
    y_large = _axes(0.1)[1]
    assert float(y_small @ y_large) < -0.9, "兜底分支若已改成连续的，请更新这条"


def test_an_explicit_reference_vector_survives_a_tenth_of_a_degree():
    """给了参考向量就不再走兜底，同样两个倾角的局部 y 几乎重合。"""
    ref = _COLUMN_REF
    for a, b in ((0.0, 0.01), (0.01, 0.1), (0.1, 1.0)):
        ya, yb = _axes(a, ref)[1], _axes(b, ref)[1]
        assert float(ya @ yb) > 0.999, f"{a}° 与 {b}° 之间朝向跳变了"


def test_generate_frame_writes_the_choice_down_for_every_column():
    g = generate_frame(spans=[6.0], storeys=[3.6, 3.6],
                       column_section="C", beam_section="B", material="M")
    by_id = {n["id"]: n for n in g["nodes"]}
    columns = wrote = 0
    for m in g["members"]:
        a, b = by_id[m["i"]], by_id[m["j"]]
        upright = abs(a["x"] - b["x"]) < 1e-9 and abs(a["y"] - b["y"]) < 1e-9
        if upright:
            columns += 1
            assert m.get("ref_vector") == list(_COLUMN_REF)
            wrote += 1
        else:
            # 横杆的参考向量不退化，不该平白多写一项脏字段
            assert "ref_vector" not in m
    assert columns > 0 and wrote == columns


def test_generate_portal_frame_writes_it_too():
    g = generate_portal_frame(spans=[12.0], eave_height=4.0, ridge_rise=2.0,
                              column_section="C", rafter_section="B", material="M")
    by_id = {n["id"]: n for n in g["nodes"]}
    uprights = [m for m in g["members"]
                if abs(by_id[m["i"]]["x"] - by_id[m["j"]]["x"]) < 1e-9
                and abs(by_id[m["i"]]["y"] - by_id[m["j"]]["y"]) < 1e-9]
    assert uprights, "门式刚架总该有柱子"
    assert all(m.get("ref_vector") == list(_COLUMN_REF) for m in uprights)


# --------------------------------------------------------------- 表格往返不许吃字段

def _tiny_model() -> dict:
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": 2.06e11, "G": 7.9e10}],
        "sections": [{"name": "S", "A": 1e-2, "Iy": 1e-4, "Iz": 1e-4, "J": 2e-4}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 0, "y": 0, "z": 4.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M",
                     "ref_vector": [1.0, 0.0, 0.0],
                     "offset_i": [0.0, 0.0, 0.3], "offset_j": [0.0, 0.0, -0.3]}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
    }


@pytest.mark.parametrize("key", ["ref_vector", "offset_i", "offset_j"])
def test_editing_a_table_does_not_eat_fields_the_table_never_showed(key):
    """表格里根本没有这几列，改一行别的东西不该把它们抹掉。

    抹掉 `ref_vector` 只是内力反号（还能看出来）；抹掉 `offset_i/offset_j`
    是**刚域没了、结构本身变了**，位移和内力一起变，而且一声不吭。
    """
    base = _tiny_model()
    tables = to_tables(base)
    tables["members"][0]["section"] = "S"        # 一次"什么也没改"的编辑
    out = from_tables(base, tables)
    assert out["members"][0][key] == base["members"][0][key]


def test_a_deleted_member_takes_its_extra_fields_with_it():
    """带回来的字段按杆件号对齐；表里没有的杆件不该凭空复活。"""
    base = _tiny_model()
    base["members"].append({"id": 2, "i": 1, "j": 2, "section": "S",
                            "material": "M", "ref_vector": [0.0, 1.0, 0.0]})
    tables = to_tables(base)
    tables["members"] = [r for r in tables["members"] if int(r["id"]) == 1]
    out = from_tables(base, tables)
    assert [m["id"] for m in out["members"]] == [1]
    assert out["members"][0]["ref_vector"] == [1.0, 0.0, 0.0]
