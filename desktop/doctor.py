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

import contextlib
import os
import platform
import subprocess
import sys
import tempfile
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


# OpenGL 探测脚本。单独拎出来，是因为它必须在**子进程**里跑，见 _opengl。
#
# 约定：往 stdout 打一行结论，父进程按那一行判断。真崩了就没有那一行，
# 父进程看返回码即可。
_OPENGL_PROBE = """
import sys
try:
    import vtk
    w = vtk.vtkOpenGLRenderWindow()
    w.SetSize(64, 64)
    r = vtk.vtkRenderer()
    w.AddRenderer(r)
    s = vtk.vtkSphereSource()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    r.AddActor(a)
    w.Render()
    w.Finalize()
    print("REAL " + vtk.VTK_VERSION)
    sys.exit(0)
except Exception:
    pass
try:
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=(64, 64))
    p.add_mesh(pv.Sphere())
    p.screenshot(None, return_img=True)
    p.close()
    print("OFFSCREEN")
except Exception as exc:
    print("FAIL %s: %s" % (type(exc).__name__, str(exc)[:150]))
sys.exit(0)
"""

_NO_GL_HINT = ("可以试试设环境变量 LIBGL_ALWAYS_SOFTWARE=1；"
               "实在不行就用网页版 run_gui.bat，功能一样全")


def _opengl() -> tuple[bool, str]:
    """真去建一个渲染窗口画个球。

    装得上不等于画得出——远程桌面、虚拟机、没装显卡驱动的机器，
    都是"依赖全齐但一开就黑屏或闪退"。这一项挂了，问题不在代码。

    **两级检测**：先试真实 OpenGL 渲染窗口（vtkOpenGLRenderWindow + Render），
    这才是桌面端 MainWindow 实际走的路；失败再降级到 off_screen 离屏模式。
    离屏能画不代表真实窗口能开——早先只测 off_screen 会在无 GL 机器上
    误报"离屏正常"，但真实启动照样崩在 'failed to get valid pixel format'。

    **两级都在子进程里跑。** 原先是在本进程里 try/except 包着的，兜不住：
    在没有 GPU 的机器上 Render() 不抛 Python 异常，而是原生崩溃，
    except Exception 对 access violation 无能为力。于是这个**专门用来诊断
    "这台机器能不能渲染"的自检**，在最该给出诊断的那种机器上把自己搞崩了。
    CI 第一次在 windows runner 上跑就是这样，整个 pytest 被带走（exit 139）。

    子进程崩了，父进程只是读到一个非零返回码——那正好就是"真实窗口开不了"。
    tests/conftest.py 的 opengl_available 出于同样的理由也是子进程探测；
    两处分开是因为问的问题不同（那边只问"能不能离屏渲染"）。
    """
    try:
        done = subprocess.run([sys.executable, "-c", _OPENGL_PROBE],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"探测进程起不来：{type(exc).__name__}"

    verdict = (done.stdout or "").strip().splitlines()
    verdict = verdict[-1] if verdict else ""

    if verdict.startswith("REAL"):
        return True, f"OpenGL 渲染正常（VTK {verdict[5:].strip()}）"
    if verdict.startswith("OFFSCREEN"):
        return False, ("仅离屏渲染可用，真实 OpenGL 窗口无法初始化——"
                       "桌面端启动会崩。" + _NO_GL_HINT)
    if verdict.startswith("FAIL "):
        return False, (verdict[5:] + "　多见于远程桌面或没装显卡驱动的机器。"
                       + _NO_GL_HINT)
    # 没有结论行 = 探测进程自己崩了。这正是最该报出来的那种机器。
    return False, (f"探测进程异常退出（返回码 {done.returncode}），"
                   "说明这台机器连建渲染窗口都会崩——桌面端一定起不来。"
                   + _NO_GL_HINT)


CHECKS = (("Python", _python), ("内核依赖", _numeric), ("求解器", _kernel),
          ("Qt", _qt), ("Qt 绑定", _binding), ("PyVista", _pyvista),
          ("云图链路", _contour), ("OpenGL", _opengl))

# 这几项挂了，后面的必然跟着挂，报完根因就停
FATAL = {"Python", "Qt", "PyVista"}


def _force_utf8_stdout() -> None:
    """把 stdout/stderr 定死成 UTF-8。

    这个工具的输出**全是中文**，而 Windows 上 stdout 默认跟系统代码页走。
    中文 Windows 是 cp936，打得出来；英文 Windows 是 cp1252，第一行就

        UnicodeEncodeError: 'charmap' codec can't encode characters
        in position 0-13: character maps to <undefined>

    ——整个自检一个字都没打出来就挂了。也就是说，**越是环境不标准的机器
    越需要这个自检，而它恰恰在那种机器上先死**。

    是 CI 的 windows runner（英文 locale）第一次跑时暴露的。本机是中文
    Windows，永远撞不到。

    errors="replace" 兜底：真遇到打不出的字符就显示成 ?，也好过整个崩掉。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass                     # 被重定向成不支持 reconfigure 的对象，算了


@contextlib.contextmanager
def _scratch_capsules():
    """自检期间的求解不留胶囊。

    每次 solve_model 成功都会往 `capsules/` 存一份存档。自检算的是固定的
    样例，存下来只会把用户自己的存档淹掉——所以指到一个用完即删的临时目录。
    依赖没装齐时 capsule 导入不了，那就什么也不做，让后面的检查去报错。
    """
    try:
        import capsule
    except Exception:                             # noqa: BLE001
        yield
        return
    original = capsule.DEFAULT_CAPSULE_DIR
    with tempfile.TemporaryDirectory(prefix="framelab-doctor-") as scratch:
        capsule.DEFAULT_CAPSULE_DIR = Path(scratch)
        try:
            yield
        finally:
            capsule.DEFAULT_CAPSULE_DIR = original


def report() -> int:
    """逐项检查并打印。全通过返回 0，否则返回 1。"""
    _force_utf8_stdout()
    print("空间刚架智能计算　桌面端自检")
    print(f"  {platform.platform()}")
    print(f"  {sys.executable}")
    print()
    failed = []
    with _scratch_capsules():
        for label, check in CHECKS:
            try:
                ok, note = check()
            except Exception as exc:              # noqa: BLE001
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
    raise SystemExit(report())
