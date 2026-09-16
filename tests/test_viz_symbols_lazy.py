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
