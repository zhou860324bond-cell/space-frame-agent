"""`.dat` 解析的验证。

这条路不需要 Abaqus，所以对比验证能在任何机器上做——比走 ODB 省事得多。
用真实 Abaqus 输出的片段做样本。
"""
import pytest

from read_dat import job_status, parse_dat

# 取自 cantilever_strong_axis_B33.dat 的真实片段：
# 全约束的节点 1 不出现在 U 表里，只出现在 RF 表里
SAMPLE = """
                                       N O D E   O U T P U T


   THE FOLLOWING TABLE IS PRINTED FOR NODES BELONGING TO NODE SET ALLN

       NODE FOOT-   U1          U2          U3          UR1         UR2         UR3
            NOTE

          2       0.000       0.000     -9.6261E-04   0.000      2.5112E-03   0.000
          9       0.000       0.000     -4.2857E-02   0.000      1.0714E-02   0.000


   THE FOLLOWING TABLE IS PRINTED FOR NODES BELONGING TO NODE SET ALLN

       NODE FOOT-   RF1         RF2         RF3         RM1         RM2         RM3
            NOTE

          1       0.000       0.000      5.0000E+04   0.000     -3.0000E+05   0.000


          THE ANALYSIS HAS BEEN COMPLETED
"""


@pytest.fixture
def dat(tmp_path):
    path = tmp_path / "job.dat"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


def test_displacements_are_read(dat):
    r = parse_dat(dat)
    assert r[9]["u3"] == pytest.approx(-0.042857)
    assert r[9]["ur2"] == pytest.approx(0.010714)
    assert r[2]["u3"] == pytest.approx(-9.6261e-4)


def test_reactions_are_read(dat):
    r = parse_dat(dat)
    assert r[1]["rf3"] == pytest.approx(5.0e4)
    assert r[1]["rm2"] == pytest.approx(-3.0e5)


def test_fully_fixed_nodes_get_zero_displacement_not_a_missing_key(dat):
    """Abaqus 不在 U 表里打印全约束节点。补 0 是物理事实，也免得调用方踩空。"""
    r = parse_dat(dat)
    assert r[1]["u3"] == 0.0
    assert set(r[1]) == set(r[9]), "每个节点的字段集合应当一致"


def test_free_nodes_get_zero_reaction(dat):
    r = parse_dat(dat)
    assert r[9]["rf3"] == 0.0


def test_every_node_carries_the_full_field_set(dat):
    r = parse_dat(dat)
    expected = {"u1", "u2", "u3", "ur1", "ur2", "ur3",
                "rf1", "rf2", "rf3", "rm1", "rm2", "rm3"}
    assert all(set(row) == expected for row in r.values())


def test_job_status_detects_completion(dat):
    assert job_status(dat)["completed"]
    assert job_status(dat)["error_lines"] == 0


def test_job_status_detects_an_incomplete_run(tmp_path):
    path = tmp_path / "bad.dat"
    path.write_text(" ***ERROR: BAD INPUT\n", encoding="utf-8")
    status = job_status(path)
    assert not status["completed"] and status["error_lines"] == 1
