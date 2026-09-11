# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包规格。

    build_exe.bat            （在项目根目录双击或在命令行跑）

产物是**文件夹版**（onedir）不是单文件，这是有意的：单文件版每次启动都要把
近 1 GB 解压到临时目录，VTK 冷启动要 30~60 秒，用户会以为程序卡死了。
文件夹版首次启动 3~5 秒，之后更快。

体积主要来自三块，都砍不掉：VTK（云图渲染）、Qt（界面）、gmsh（实体网格）。
能砍的都写在 EXCLUDES 里了。
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import (collect_all, collect_dynamic_libs,
                                     collect_submodules)

ROOT = Path(SPECPATH).parent            # noqa: F821  (SPECPATH 由 PyInstaller 注入)

# --- 必须整包收进去的三个大件 ---------------------------------------------
# 它们都在运行时动态 import 子模块，静态分析抓不全：漏一个的表现是
# 界面起来了、一点云图就闪退，而不是构建期报错。
datas, binaries, hiddenimports = [], [], []
for package in ("pyvista", "vtkmodules", "pyvistaqt"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

# gmsh 单独处理：它的 wheel 装出来是**一个顶层模块 `gmsh.py` 加一个原生库**，
# 不是包。`collect_all` 对模块会抛，抛了就整块跳过，结果是"构建成功、
# 一点节点实体就 ImportError"。所以失败时退回只收原生库，并且把
# 跳过与否打出来——静默丢掉一整块功能是最难查的一类打包问题。
try:
    d, b, h = collect_all("gmsh")
    datas += d
    binaries += b
    hiddenimports += h
    print("[spec] gmsh: collect_all 成功")
except Exception as exc:
    try:
        binaries += collect_dynamic_libs("gmsh")
        hiddenimports += ["gmsh"]
        print(f"[spec] gmsh: collect_all 失败（{exc}），已退回只收原生库")
    except Exception as exc2:
        print(f"[spec] gmsh 完全收不进来（{exc2}）——"
              "打出来的程序将没有节点实体子模型功能")

hiddenimports += collect_submodules("scipy.sparse.linalg")
hiddenimports += ["matplotlib.backends.backend_qtagg",
                  "matplotlib.backends.backend_agg"]

# --- 不需要的东西 -----------------------------------------------------------
# streamlit / plotly 是网页版那条路径，桌面端用不到，但它们会拖进
# pyarrow、tornado 一大串，光这两个就上百兆。
EXCLUDES = [
    "streamlit", "plotly", "pyarrow", "tornado",
    "pytest", "_pytest", "pluggy",
    "tkinter",
    # **不要排 unittest，也不要排 test。** 它们看着只是测试用的，实际是
    # numpy/scipy 的运行时依赖：`numpy.testing` 在模块顶层 `import unittest`，
    # 而 scipy 的 array_api_compat 会 clone 整个 numpy 命名空间、顺手碰到
    # `numpy.testing`。排掉的表现是打包成功、**一启动就 ModuleNotFoundError**，
    # 而且栈顶指向 scipy，跟这份排除清单看上去毫无关系。
    "PyQt5", "PyQt6", "PySide2", "shiboken2",
    "IPython", "jupyter", "notebook", "nbformat",
    "docx",          # 出 Word 报告用；留着会把 lxml 拖进来，需要时再放开
]

a = Analysis(                                       # noqa: F821
    [str(ROOT / "build_tools" / "frame_lab_launcher.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # 自带钩子里有按**模块名**匹配的地雷：PyPI 上有个不相干的包也叫
    # workflow，而我们有 src/workflow.py。详见 hooks/hook-workflow.py。
    # hookspath 的优先级高于自带钩子，同名放一个空的即可让开。
    hookspath=[str(ROOT / "build_tools" / "hooks")],
    runtime_hooks=[str(ROOT / "build_tools" / "runtime_hook.py")],
    excludes=EXCLUDES,
    noarchive=False,
)

# --- 安全闸：密钥绝不能进包 -------------------------------------------------
# 打进去等于把 API 额度送给每一个拿到文件的人。这里是**兜底**，
# 正常情况下 deepseek.key 根本不会被 Analysis 收进来；
# 真收进来了就直接让构建失败，而不是打出一个带密钥的包。
_LEAKED = [entry for entry in a.datas
           if "deepseek" in entry[0].lower() or entry[0].lower().endswith(".key")]
if _LEAKED:
    raise SystemExit(f"构建中止：密钥文件被收进了打包体 {_LEAKED}")

pyz = PYZ(a.pure)                                   # noqa: F821

exe = EXE(                                          # noqa: F821
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="FrameLab",
    debug=False,
    strip=False,
    upx=False,          # UPX 压 Qt/VTK 的 dll 会让部分杀软直接判毒，得不偿失
    # 平时不弹黑窗；`build_exe.bat --debug` 打出的版本保留控制台，
    # 因为第一次打包八成会在启动时缺模块，而无控制台版只会一闪而过。
    console=bool(os.environ.get("FRAMELAB_CONSOLE")),
    icon=None,
)

coll = COLLECT(                                     # noqa: F821
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="FrameLab",
)
