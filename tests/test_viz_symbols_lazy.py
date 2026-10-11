"""`viz_symbols` 不该为了两个纯几何函数去依赖一个绘图后端。

桌面端的 `desktop/scene.py` 只从这里拿 `model_size` 和 `classify_support`，
两个都只用 numpy。plotly 是网页端那三个 `*_traces` 函数要的。

平时看不出来，打包成 exe 时暴露：排掉 plotly（桌面端用不到，而且它会拖进
pyarrow、tornado 一大串）之后，程序一启动就
`ModuleNotFoundError: No module named 'plotly'`，栈顶指向 viz_symbols，
而那一行 import 看上去理所当然。
"""

from __future__ import annotations

import builtins
import importlib
import sys



def test_importing_viz_symbols_does_not_need_plotly(monkeypatch):
    """把 plotly 彻底挡掉，模块仍要 import 得进来。"""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "plotly" or name.startswith("plotly."):
            raise ModuleNotFoundError("No module named 'plotly'")
        return real_import(name, *args, **kwargs)

    for cached in [m for m in sys.modules if m.startswith("plotly")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.delitem(sys.modules, "viz_symbols", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    module = importlib.import_module("viz_symbols")
    # 桌面端真正用到的两个，必须照常可用
    assert module.classify_support([1, 1, 1, 1, 1, 1]) == "固接"
    assert callable(module.model_size)


def test_the_plotly_traces_still_work_when_plotly_is_there():
    """延迟导入不能把网页端那条路径弄坏。"""
    import viz_symbols

    assert viz_symbols._go() is not None
    assert hasattr(viz_symbols._go(), "Scatter3d")


def test_partial_load_traces_keep_uniform_intensity_range_and_separate_point_force():
    """防止网页把区间均布沿全杆衰减到零，并保留图IV-5杆中集中力的独立作用点。"""
    import numpy as np

    from agent import Session
    from model_io import from_dict
    import viz_symbols

    session = Session()
    assert session.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 8, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"}],
        "materials": [{"name": "M", "E": 210e9, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 1e-5, "Iz": 1e-5, "J": 2e-5}],
        "supports": [{"node": 1, "fix": [1]*6}],
    }).ok
    for kind, a, w2 in (("partial", 0.0, None), ("partial", 1.0, None),
                        ("partial_trapezoid", 1.0, [0, 0, -1600])):
        entry = {"member": 1, "kind": kind, "w1": [0, 0, -800], "a": a, "b": 3.0}
        if w2 is not None:
            entry["w2"] = w2
        point = {"member": 1, "kind": "point", "w1": [4000, 0, 0], "a": 3.0}
        assert session.set_load_cases(cases=[{
            "name": "D", "member_spans": [entry, point]}]).ok
        traces, meta = viz_symbols.load_traces(from_dict(session.model), "D", size=1.0)
        trace = next(t for t in traces if t.name == "杆间荷载")
        np.testing.assert_allclose(trace.x, [a, (a + 3.0)/2, 3.0, 3.0])
        np.testing.assert_allclose(trace.y, 0)
        np.testing.assert_allclose(trace.z, 0)
        assert meta["member"] == 4 and trace.u[-1] > 0 and trace.w[-1] == 0
        intensities = np.asarray(trace.w[:3])
        assert np.all(intensities < 0)
        if kind == "partial":
            np.testing.assert_allclose(intensities, intensities[0])
        else:
            assert intensities[0] > intensities[1] > intensities[2]


def test_plotly_is_not_imported_at_module_level():
    """源码层面钉住：顶层不许再出现 plotly 的 import。

    只查断言不够——有人"顺手整理 import"把它挪回顶层，
    上面那条测试确实会红，但这条的报错更直指问题。
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "src" / "viz_symbols.py").read_text(encoding="utf-8"))
    top_level = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = []
    for node in top_level:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif node.module:
            names.append(node.module)
    assert not any(n.startswith("plotly") for n in names), \
        "plotly 又回到顶层了：桌面端会因此依赖一个它一行都不用的绘图后端"
