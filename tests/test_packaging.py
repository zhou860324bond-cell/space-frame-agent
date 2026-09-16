"""打包、依赖与 README 里那些**会过期的数字**。

这个文件盯的不是功能，是**别人 clone 下来能不能跑**，以及**README 有没有
在说过时的话**。两者都不会让任何测试变红，但都会让看的人对整个项目
打折扣——README 写着 45 个工具、实际 49 个，读的人只会想"还有多少处是旧的"。
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_readme_tool_count_matches_reality():
    """**这条已经过期过一次**：README 写 45 个工具，实际已经 49 个。

    工具是一个个加上去的，每次都要记得回头改 README——记不住，所以让测试记。
    """
    from agent import TOOLS

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    claimed = re.search(r"注册 \*\*(\d+) 个确定性工具\*\*", text)
    assert claimed, "README 里那句工具数量的话被改动了，测试跟不上了"
    assert int(claimed.group(1)) == len(TOOLS), (
        f"README 说 {claimed.group(1)} 个工具，实际 {len(TOOLS)} 个")


def test_the_packaging_metadata_is_readable_and_declares_a_python_floor():
    data = _pyproject()
    assert data["project"]["name"]
    floor = data["project"]["requires-python"]
    from desktop import doctor

    major, minor = doctor.MIN_PYTHON
    assert floor == f">={major}.{minor}", (
        "pyproject 的 Python 下限要和自检里的 MIN_PYTHON 一致，"
        "否则装得上却过不了自检")


def test_every_runtime_import_is_declared_somewhere():
    """内核直接 import 的第三方包，必须出现在依赖声明里。

    漏一个的表现是：自己机器上早就装过所以一直没发现，别人 clone 下来
    在 import 那一行就挂了。
    """
    data = _pyproject()
    declared = " ".join(data["project"]["dependencies"])
    for group in data["project"]["optional-dependencies"].values():
        declared += " " + " ".join(group)
    declared = declared.lower()
    for package in ("numpy", "scipy", "jsonschema", "matplotlib",
                    "pyvista", "pyside6", "gmsh", "streamlit"):
        assert package in declared, f"{package} 没有出现在 pyproject 的依赖里"


def test_the_lock_file_pins_exact_versions():
    """锁文件的意义就在于**确切**。留一个 `>=` 进去，它就不再是锁文件。"""
    lines = [line.strip() for line in
             (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()]
    pins = [line for line in lines if line and not line.startswith("#")]
    assert pins, "锁文件是空的"
    for pin in pins:
        assert "==" in pin, f"锁文件里这一行不是确切版本：{pin}"


def test_requirements_and_the_lock_file_talk_about_the_same_packages():
    """两个文件各说各的包，就等于没有锁。"""
    def names(path: pathlib.Path) -> set[str]:
        out = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                out.add(re.split(r"[><=]", line)[0].strip().lower())
        return out

    loose = names(ROOT / "requirements.txt")
    locked = names(ROOT / "requirements-lock.txt")
    missing = sorted(loose - locked - {"pytest"})
    assert not missing, f"requirements.txt 里这些包没有被锁定：{missing}"


def test_the_ci_workflow_sets_the_headless_qt_platform():
    """无头环境不设 offscreen，桌面端测试一碰 Qt 就崩——而且是 C++ 层的崩，
    Python 抓不住，流水线只会给一个没有上下文的退出码。"""
    text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "QT_QPA_PLATFORM: offscreen" in text
    assert "libglu1-mesa" in text, "VTK / Gmsh 要的系统库不能少"
    assert "pytest" in text
