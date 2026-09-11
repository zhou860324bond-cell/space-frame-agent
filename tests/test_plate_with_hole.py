"""带孔板：自研 C3D10 **整条链路**对着教科书答案的验证。

`test_solid3d.py` 验的是单元本身（形函数、分片检验、刚体模态、制造解），
这一条验的是**装配起来之后**：网格生成 → 荷载等效 → 求解 → 应力恢复 →
峰值提取。这一串原来一次都没对过外部参照——`run_native_joint_analysis`
在测试里是被 mock 掉的，而第 1 题要的正是"节点单独实体单元计算应力集中"。

中心带圆孔的受拉板是这个问题的标准算例：Kt 有闭式解，而且**峰值是收敛的**
（孔边不是奇异点），正好和项目里那个发散的相贯线算例配成一对。

这里跑的是**粗网格**（几秒），只验"链路通、量级对、峰值位置对"。
完整的三档收敛研究在 `tools/verify_plate_with_hole.py`，结果写在
`docs/带孔板验证.md`。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh", reason="未安装 gmsh，跳过实体网格验证")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _tool():
    spec = importlib.util.spec_from_file_location(
        "verify_plate_with_hole", ROOT / "tools" / "verify_plate_with_hole.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def coarse():
    """跑一次，四条测试共用。**这一档要几秒**，每条各跑一次是浪费。"""
    tool = _tool()
    return tool, tool.run_one(3.0)


def test_the_whole_solid_pipeline_lands_on_the_textbook_answer(coarse):
    """粗网格上 Kt 就该落在闭式解附近。

    偏大是预期的：网格粗、而且 t/d=0.5 的板不是"薄板"，中面应力比平面应力
    的经典值高几个百分点（完整的沿厚度分布见 docs/带孔板验证.md）。
    这里给一个宽的窗口，只拦"链路错了"这种量级上的问题。
    """
    tool, got = coarse
    reference = tool.heywood_kt()
    assert reference == pytest.approx(2.512, abs=1e-3)
    assert 0.9 * reference <= got["kt"] <= 1.25 * reference, got["kt"]


def test_the_peak_sits_on_the_hole_edge_at_ninety_degrees(coarse):
    """位置比数值更能说明链路对不对。

    受拉板孔边的 σxx 峰值在 90° 处（垂直于受力方向的那两点）。峰值若跑到
    别处——夹持端、孔的 0° 位置——那是荷载、约束或应力恢复错了，
    而这种错**不会报错**，只会给一个看着合理的数。
    """
    tool, got = coarse
    x, y, z = got["at"]
    assert x == pytest.approx(tool.LENGTH / 2.0, abs=0.12 * tool.HOLE_D)
    assert abs(abs(y - tool.WIDTH / 2.0) - tool.HOLE_D / 2.0) < 0.05 * tool.HOLE_D
    assert 0.0 <= z <= tool.THICKNESS


def test_the_nodal_average_is_below_the_unaveraged_peak(coarse):
    """不平均的峰值系统性偏高，且偏高多少取决于网格。

    两个数都报出来，它们的差就是"网格够不够细"的一个指标。反过来（平均值
    比不平均还大）说明平均或外推写错了。
    """
    tool, got = coarse
    assert got["kt"] < got["kt_unaveraged"]
    assert got["kt_unaveraged"] / got["kt"] < 1.15


def test_the_solve_residual_is_negligible(coarse):
    """自由自由度上的残差要小到可以忽略，否则前面的数都不用看。"""
    tool, got = coarse
    force = tool.GROSS_STRESS * tool.WIDTH * tool.THICKNESS
    assert got["residual"] / force < 1e-8
