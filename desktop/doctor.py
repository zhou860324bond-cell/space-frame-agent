"""桌面端自检。

    python -m desktop.app --doctor

"打不开"这三个字里可能藏着七八种原因：依赖没装、Qt 绑定选错、
显卡没有可用的 OpenGL、Python 版本太老……而 Windows 上双击 .bat 时
错误往往一闪而过，什么都留不下。所以单写这么一个东西：**逐项检查，
用人话说出哪一项挂了、下一步做什么**，而不是丢一个堆栈让人自己猜。

每一项返回 (通过?, 说明)。检查顺序是有讲究的——前面的挂了后面必然跟着挂，
所以报完根因就停，免得一屏红字里找不着重点。
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

# 检查 Qt 之前先钉死绑定，理由见 app.py
os.environ.setdefault("QT_API", "pyside6")

# 自检要能**单独**跑：`python -m desktop.doctor`。走 app.py 时 sys.path 是
# 它铺好的，直接跑这个模块则没有——那样求解器和云图两项会报 "No module
# named 'agent'"，而真正的问题只是路径。自检工具自己把路铺好，
# 免得它自己成为要排查的对象。
for _extra in (Path(__file__).resolve().parent.parent,
               Path(__file__).resolve().parent.parent / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

MIN_PYTHON = (3, 10)


def _python() -> tuple[bool, str]:
    v = sys.version_info
    ok = v[:2] >= MIN_PYTHON
    return ok, (f"{v.major}.{v.minor}.{v.micro}"
                + ("" if ok else f"　太老，需要 {MIN_PYTHON[0]}.{MIN_PYTHON[1]} 以上"))


def _numeric() -> tuple[bool, str]:
    missing = []
    for name in ("numpy", "scipy", "jsonschema"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        return False, ("缺少 " + "、".join(missing)
                       + "　→　pip install -r requirements.txt")
    return True, "numpy / scipy / jsonschema 齐全"


def _kernel() -> tuple[bool, str]:
    """真算一个单跨框架。依赖装齐了不等于内核是好的。"""
    try:
        from agent import Session
        s = Session()
        s.define_materials_and_sections(
            [{"name": "S", "E": 2.1e11, "nu": 0.3}],
            [{"name": "B", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}])
        g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="B",
                             beam_section="B", material="S")
        s.set_load_cases(cases=[{"name": "D", "member_loads":
                                 [{"member": m, "w": [0, 0, -20e3]}
                                  for m in g.payload["beam_member_ids"]]}])
        got = s.solve_model()
    except Exception as exc:                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"
    if not got.ok:
        return False, f"求解没通过：{str(got.payload)[:160]}"
    return True, "求解正常（算了一个单跨单层框架）"


def _contour() -> tuple[bool, str]:
    """把云图这条链路真跑一遍：求解 → 内力 → 应力 → 分级切分 → 生成管。

    OpenGL 正常不等于云图正常。这一条链路上任何一步坏了，在有屏幕的机器上
    表现都只是"图看着怪"——没人查得出来是哪一步。所以单独检一次，
    而且**连应力一起检**：应力多依赖截面的极端纤维距离，是另一处会断的地方。
    """
    try:
        import numpy as np

        from agent import Session
        from desktop import scene, theme

        s = Session()
        s.define_materials_and_sections(
            [{"name": "S", "E": 2.1e11, "nu": 0.3}],
            [{"name": "B", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7,
              "cy": 0.15, "cz": 0.08}])
        g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="B",
                             beam_section="B", material="S")
        s.set_load_cases(cases=[{"name": "D", "member_loads":
                                 [{"member": m, "w": [0, 0, -20e3]}
                                  for m in g.payload["beam_member_ids"]]}])
        if not s.solve_model().ok:
            return False, "云图要用的静力解没算出来"
        levels = theme.CONTOUR_LEVELS
        notes = []
        for component, scale in (("Mz", 1e-3), (scene.STRESS, 1e-6)):
            line = scene.contour_line(s.frame, s.solution, "D", component,
                                      value_scale=scale)
            clim = scene.contour_clim(line, component, percentile=None)
            tube = scene.banded_tubes(line, component, clim, levels,
                                      radius=0.02)
            name = component + scene.BAND_SUFFIX
            if name not in tube.cell_data or tube.n_cells == 0:
                return False, f"{component} 云图没生成出带颜色的网格"
            shades = len(np.unique(np.round(tube.cell_data[name], 9)))
            notes.append(f"{component} {shades} 级/{levels}")
    except Exception as exc:                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"
    return True, "内力与应力云图链路正常（" + "，".join(notes) + "）"


def _qt() -> tuple[bool, str]:
    try:
        import PySide6
    except ImportError:
        return False, "没装　→　pip install PySide6 pyvista pyvistaqt"
    except Exception as exc:                      # noqa: BLE001
        return False, (f"装了但加载不了：{type(exc).__name__}: {str(exc)[:140]}　"
                       "常见于缺 VC++ 运行库，装一个 "
                       "Microsoft Visual C++ Redistributable 试试")
    return True, f"PySide6 {PySide6.__version__}"


def _binding() -> tuple[bool, str]:
    """qtpy 选中了哪个绑定。

    选错是"打不开"里最常见、也最难从报错看出来的一种：pyvistaqt 经 qtpy
    加载 PyQt5，我们的窗口用 PySide6，两套 Qt 在同一个进程里必然出事。
    """
    try:
        import qtpy
    except ImportError:
        return True, "未安装 qtpy（装 pyvistaqt 时会带上）"
    others = []
    for name in ("PyQt5", "PySide2", "PyQt6"):
        try:
            __import__(name)
            others.append(name)
        except ImportError:
            pass
    if qtpy.API_NAME != "PySide6":
        return False, (f"qtpy 选中了 {qtpy.API_NAME}，本程序用的是 PySide6。"
                       "一个进程里两套 Qt 绑定必然出问题。"
                       "启动前设环境变量 QT_API=pyside6（run_desktop.bat 已经设了），"
                       "或者卸载 " + "、".join(others or ["其它绑定"]))
    note = f"qtpy → {qtpy.API_NAME}"
    if others:
        note += f"（机器上还装着 {'、'.join(others)}，已被 QT_API 钉死，没事）"
    return True, note


def _pyvista() -> tuple[bool, str]:
    try:
        import vtk
        import pyvista
    except ImportError as exc:
        return False, (f"缺 {exc.name}　→　pip install pyvista pyvistaqt")
    try:
        import pyvistaqt                          # noqa: F401
    except ImportError:
        return False, "缺 pyvistaqt　→　pip install pyvistaqt"
    except Exception as exc:                      # noqa: BLE001
        return False, f"pyvistaqt 加载失败：{type(exc).__name__}: {str(exc)[:140]}"
    return True, f"PyVista {pyvista.__version__}　VTK {vtk.VTK_VERSION}"


def _opengl() -> tuple[bool, str]:
    """真去建一个渲染窗口画个球。

    装得上不等于画得出——远程桌面、虚拟机、没装显卡驱动的机器，
    都是"依赖全齐但一开就黑屏或闪退"。这一项挂了，问题不在代码。

    **两级检测**：先试真实 OpenGL 渲染窗口（vtkOpenGLRenderWindow + Render），
    这才是桌面端 MainWindow 实际走的路；失败再降级到 off_screen 离屏模式。
    离屏能画不代表真实窗口能开——早先只测 off_screen 会在无 GL 机器上
    误报"离屏正常"，但真实启动照样崩在 'failed to get valid pixel format'。
    """
    # 第一级：真实 OpenGL 渲染窗口。必须真调一次 Render()，只建窗口不渲染
    # 在某些环境下会"假成功"。
    try:
        import vtk
        ren_win = vtk.vtkOpenGLRenderWindow()
        ren_win.SetSize(64, 64)
        renderer = vtk.vtkRenderer()
        ren_win.AddRenderer(renderer)
        sphere = vtk.vtkSphereSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        ren_win.Render()                          # 真正触发 OpenGL 上下文初始化
        ren_win.Finalize()
        return True, f"OpenGL 渲染正常（VTK {vtk.VTK_VERSION}）"
    except Exception:
        pass                                       # 真实窗口失败，降级到离屏

    # 第二级：离屏模式。能画说明软件渲染/ Mesa 可用，但真实窗口仍可能失败。
    try:
        import pyvista as pv
        plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
        plotter.add_mesh(pv.Sphere())
        plotter.screenshot(None, return_img=True)
        plotter.close()
    except Exception as exc:                      # noqa: BLE001
        return False, (f"{type(exc).__name__}: {str(exc)[:150]}　"
                       "多见于远程桌面或没装显卡驱动的机器。"
                       "可以试试设环境变量 LIBGL_ALWAYS_SOFTWARE=1；"
                       "实在不行就用网页版 run_gui.bat，功能一样全")
    return False, ("仅离屏渲染可用，真实 OpenGL 窗口无法初始化——"
                   "桌面端启动会崩。可以试试设 LIBGL_ALWAYS_SOFTWARE=1；"
                   "或直接用网页版 run_gui.bat")


CHECKS = (("Python", _python), ("内核依赖", _numeric), ("求解器", _kernel),
          ("Qt", _qt), ("Qt 绑定", _binding), ("PyVista", _pyvista),
          ("云图链路", _contour), ("OpenGL", _opengl))
CI_CHECKS = tuple(item for item in CHECKS if item[0] != "OpenGL")

# 这几项挂了，后面的必然跟着挂，报完根因就停
FATAL = {"Python", "Qt", "PyVista"}


def report(checks=None) -> int:
    """逐项检查并打印。全通过返回 0，否则返回 1。"""
    checks = CHECKS if checks is None else checks
    print("空间刚架智能计算　桌面端自检")
    print(f"  {platform.platform()}")
    print(f"  {sys.executable}")
    print()
    failed = []
    for label, check in checks:
        try:
            ok, note = check()
        except Exception as exc:                  # noqa: BLE001
            ok, note = False, f"检查本身出错：{type(exc).__name__}: {exc}"
        print(f"  [{'OK' if ok else '!!'}] {label:<8} {note}")
        if not ok:
            failed.append(label)
            if label in FATAL:
                print("       （这一项挂了后面必然跟着挂，先解决它）")
                break
    print()
    if failed:
        print("没通过：" + "、".join(failed))
        print("照上面那条提示处理。桌面端实在跑不起来，"
              "网页版 run_gui.bat 功能是一样全的。")
        return 1
    print("全部通过。桌面端应当能正常启动：python -m desktop.app")
    return 0


if __name__ == "__main__":
    raise SystemExit(report(CI_CHECKS if "--ci" in sys.argv[1:] else None))
