"""打包规格的闸。

打包这件事没法在这里端到端验证——构建必须在 Windows 上跑，而且要十几分钟。
能守的是**规格本身别指向不存在的东西**：入口脚本被改名、运行时钩子被删、
排除清单里混进桌面端真正需要的包，这三类错都要等到构建失败甚至等到
用户双击闪退才暴露，而它们在这里一秒就能查出来。

目录叫 `build_tools/` 不叫 `packaging/`：后者会盖掉第三方的 `packaging` 包，
而 matplotlib 起手就 `from packaging.version import parse`——整条绘图链路
连同一半测试会以 `ModuleNotFoundError` collapse，报错还指向 matplotlib，
跟这个目录看上去毫无关系。（这不是设想，是刚刚踩过一次。）
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "build_tools"
SPEC = PACKAGING / "frame_lab.spec"


def _spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_the_spec_and_its_two_scripts_exist():
    for path in (SPEC,
                 PACKAGING / "frame_lab_launcher.py",
                 PACKAGING / "runtime_hook.py",
                 ROOT / "build_exe.bat"):
        assert path.exists(), f"打包少了 {path.relative_to(ROOT)}"


@pytest.mark.parametrize("path", [
    "build_tools/frame_lab.spec",
    "build_tools/frame_lab_launcher.py",
    "build_tools/runtime_hook.py",
])
def test_the_spec_scripts_parse(path):
    """spec 也是 Python。语法错要在这里红，不是在十分钟的构建之后。"""
    ast.parse((ROOT / path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("package", ["pyvista", "vtkmodules", "pyvistaqt", "gmsh"])
def test_the_big_runtime_packages_are_collected_whole(package):
    """这四个都在运行时动态 import 子模块，静态分析抓不全。

    漏一个的表现是**界面起来了、一点云图就闪退**——构建期一声不吭。
    """
    assert f'"{package}"' in _spec_text()


@pytest.mark.parametrize("package", ["PySide6", "pyvista", "vtkmodules",
                                     "pyvistaqt", "numpy", "scipy",
                                     "matplotlib"])
def test_nothing_the_desktop_needs_is_excluded(package):
    """排除清单里不能混进桌面端真正要用的包。"""
    text = _spec_text()
    start = text.index("EXCLUDES = [")
    excludes = text[start:text.index("]", start)]
    assert f'"{package}"' not in excludes, f"{package} 是桌面端要用的，不能排除"


def test_the_build_refuses_to_ship_a_key():
    """密钥进包必须让构建失败，而不是打出一个带密钥的包。"""
    text = _spec_text()
    assert "deepseek" in text and "SystemExit" in text, \
        "spec 里那道拦密钥的断言不见了"
    bat = (ROOT / "build_exe.bat").read_text(encoding="utf-8", errors="replace")
    assert "*.key" in bat, "build_exe.bat 构建完不再扫一遍密钥了"


def test_the_launcher_pins_the_qt_binding_before_importing_qt():
    """QT_API 必须在 import 任何 Qt 之前钉死。

    机器上装过 PyQt5 时 qtpy 会选它，一个进程两套 Qt 绑定直接闪退，
    报错还看不出根源——这是"桌面端打不开"最常见的一种原因。
    """
    for path in (PACKAGING / "frame_lab_launcher.py",
                 PACKAGING / "runtime_hook.py"):
        source = path.read_text(encoding="utf-8")
        qt_api = source.index("QT_API")
        imports = [source.index(name) for name in ("PySide6", "pyvista", "qtpy")
                   if name in source]
        assert all(qt_api < where for where in imports), \
            f"{path.name} 里 QT_API 设晚了"


def test_a_frozen_build_is_documented_as_a_folder_not_a_single_file():
    """单文件版启动要解压近 1 GB，用户会以为卡死。这条决定要写下来。"""
    notes = (PACKAGING / "分发说明.md").read_text(encoding="utf-8")
    assert "onefile" in _spec_text() or "单文件" in _spec_text()
    assert "整个文件夹" in notes, "分发说明没写清楚不能只发 exe"


def test_the_workflow_hook_override_is_in_place():
    """`src/workflow.py` 和 PyPI 上一个不相干的 `workflow` 包重名。

    pyinstaller-hooks-contrib 为那个包写了 stdhook，内容是
    `copy_metadata('workflow')`；PyInstaller 按模块名匹配钩子，
    不看这个模块是谁的，于是构建以 `PackageNotFoundError` 中止，
    报错里一个字都没提"你有个同名模块"。

    这条守两件事：空钩子还在，spec 还把 hooks 目录挂上了。
    少任何一件，构建在跑满两分半之后才失败。
    """
    override = PACKAGING / "hooks" / "hook-workflow.py"
    assert override.exists(), "hook-workflow.py 让路钩子不见了"
    # 只看代码不看文档字符串——说明里当然要写清楚绕的是哪一句
    tree = ast.parse(override.read_text(encoding="utf-8"))
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "copy_metadata" not in calls, \
        "让路钩子不能真去找包元数据，那正是要绕开的东西"
    assert (ROOT / "src" / "workflow.py").exists(), \
        "src/workflow.py 没了的话这个让路钩子也该一并删掉"
    assert 'hookspath=[str(ROOT / "build_tools" / "hooks")]' in _spec_text(), \
        "spec 没挂 hooks 目录，让路钩子不会生效"


@pytest.mark.parametrize("module", ["unittest", "test", "email", "logging",
                                    "json", "ctypes", "importlib"])
def test_no_stdlib_module_that_numpy_or_scipy_needs_is_excluded(module):
    """排除清单里不能有这些标准库。

    `unittest` 是踩出来的：它看着只是测试用的，实际上 `numpy.testing` 在
    模块顶层就 `import unittest`，而 scipy 的 array_api_compat 会 clone
    整个 numpy 命名空间、顺手碰到 `numpy.testing`。排掉之后打包照样成功，
    **一启动就 ModuleNotFoundError**，栈顶还指向 scipy——
    跟这份排除清单看上去毫无关系，很难想到是自己排掉的。
    """
    text = _spec_text()
    start = text.index("EXCLUDES = [")
    excludes = text[start:text.index("]", start)]
    assert f'"{module}"' not in excludes, \
        f"{module} 是标准库且被运行时依赖，排掉会让打包体一启动就崩"


def test_the_word_report_backend_is_bundled():
    """`report.to_docx` 是延迟导入，静态分析看不见它。

    不在 spec 里点名的话，打包一切正常，**点「报告」才说缺 python-docx**——
    而拿到 exe 的人没有 pip，这个提示对他毫无用处。
    """
    text = _spec_text()
    start = text.index("EXCLUDES = [")
    excludes = text[start:text.index("]", start)]
    assert '"docx"' not in excludes, "python-docx 被排掉了，Word 报告会点不动"
    assert '"docx"' in text, "spec 没点名 docx，延迟导入的它不会被收进去"


def test_the_zip_step_refuses_to_ship_a_key():
    """打包体里放过密钥是最容易犯、后果最重的一个错。

    `build_exe.bat` 只在**构建时**扫一遍，而"为了自己试用把 key 放到
    exe 旁边"恰恰发生在构建之后、压包之前——那一步没人拦就会直接发出去。
    """
    zipper = ROOT / "package_zip.bat"
    assert zipper.exists(), "package_zip.bat 不见了"
    text = zipper.read_text(encoding="utf-8", errors="replace")
    assert "*.key" in text and "REFUSED" in text, \
        "压包脚本不再拦密钥了"
    assert "DEEPSEEK_API_KEY" in text, \
        "拦下之后要告诉用户改用环境变量，否则他只会把文件挪来挪去"
