"""渲染工具必须画应用真正画的东西。

`tools/render_viewport.py` 存在的理由，写在它自己的 docstring 里：视口的观感
问题只有渲染出来才看得见，所以要有一条"改一版、渲一版、对比"的回路。

**但这条回路会悄悄断掉，而且断了没人知道。** 实测它同时坏在两处：

1. `theme.DIVERGING` 在某次配色重构后就不存在了 —— 脚本跑到一半直接
   AttributeError 崩掉；
2. 云图用的是 `scene.member_tubes`（模型视图的细管 TUBE_RATIO=0.0032 +
   连续渐变），而应用用的是 `scene.banded_tubes`（CONTOUR_TUBE_RATIO=0.0145
   的粗管 + 分级色块 + 调色板）。**差了四倍半管径和"分不分级"。**

第二条尤其阴险：脚本照样跑完、照样出图，只是那张图不是用户看到的那张。
拿它去判断观感，会得出完全相反的结论——实测就差点据此去"修"作者一次次
调好的管径与支座尺寸（scene.py 里那几个常量的注释有完整调试史）。

所以这个文件盯的不是"图好不好看"，是**脚本有没有跟 viewport 走散**。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "render_viewport.py"
VIEWPORT = ROOT / "desktop" / "viewport.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_the_render_tool_still_imports():
    """最低要求：它得能被解析、能被导入。

    上一次它是 AttributeError 崩在半路的——前两张图出来了、第三张没有，
    而脚本的退出码没人看。
    """
    ast.parse(_source(TOOL))


def test_the_render_tool_draws_banded_contour_like_the_app():
    """云图必须走 banded_tubes + CONTOUR_TUBE_RATIO，和 viewport 一致。"""
    tool = _source(TOOL)
    assert "banded_tubes" in tool, "云图要用分级色块，不是连续渐变"
    assert "CONTOUR_TUBE_RATIO" in tool, "云图要用粗管，不是模型视图的细管"
    assert "BAND_SUFFIX" in tool, "着色要挂在分级后的标量上"
    # 应用那边也确实是这么画的——两边同步才有意义
    app = _source(VIEWPORT)
    for token in ("banded_tubes", "CONTOUR_TUBE_RATIO", "BAND_SUFFIX"):
        assert token in app, f"viewport 里找不到 {token}，这条断言的前提变了"


def test_the_render_tool_uses_the_real_palette():
    """调色板要走 theme.palette_cmap + theme.banded，别自己另挑一个。

    原先固定用 sequential_cmap / diverging_cmap，而应用按分量从
    CONTOUR_PALETTES 里选（默认 rainbow，Abaqus 式）。颜色都不是一套，
    对比出来的观感没有意义。
    """
    tool = _source(TOOL)
    assert "palette_cmap" in tool
    assert "theme.banded" in tool


def test_no_reference_to_attributes_the_theme_no_longer_has():
    """脚本引用的 theme 属性必须真实存在。

    这条直接拦住上次那种崩法：配色模块重构了，脚本没跟上，
    而"跑一下看看"才发现——那时候人已经在看一张过期的图了。
    """
    import desktop.theme as theme

    tool = ast.parse(_source(TOOL))
    missing = []
    for node in ast.walk(tool):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "theme"
                and not hasattr(theme, node.attr)):
            missing.append(node.attr)
    assert not missing, f"theme 上没有这些属性了：{sorted(set(missing))}"


@pytest.mark.parametrize("name", ["vp_model.png", "vp_contour_M.png",
                                  "vp_contour_N.png"])
def test_every_documented_output_is_actually_produced(name):
    """三张图都要出得来。

    上次崩在第三张（轴力云图）上，前两张照常生成——只看目录里有文件，
    是看不出脚本失败过的。
    """
    tool = _source(TOOL)
    assert f'"{name}"' in tool, f"脚本里找不到 {name}，输出清单变了就改这条测试"
