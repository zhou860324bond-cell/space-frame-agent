"""真渲染冒烟：开一个真正的窗口，把每种显示模式真画一遍，逐项检查。

**为什么需要它。** 自动测试在无头环境（QT_QPA_PLATFORM=offscreen）里跑，
视口的绘制路径整段跳过——只验证"送进渲染器的参数"。而这几轮前端的问题，
几乎全出在真渲染这条路上，都是截图肉眼才看出来的：

* 应力比图的色标参数写错（``categories``），真渲染时抛异常，图只画了一半；
* 底部抽屉被某一页的最小尺寸撑大，越出视口、压住右抽屉；
* 色标写死在视口比例坐标上，结果抽屉一开下半截被盖住；
* 右缘的 AI 按钮被 VTK 原生窗口盖住，根本看不见。

这里把这些检查成条写下来：绘制中不许抛异常、每种模式画面非空、带色标的
模式确实有彩色且色标存在、抽屉都在视口内且互不重叠、色标落在未被抽屉
盖住的区域。

用法::

    python tools/render_smoke.py [报告.json]

退出码 0 表示全部通过。需要能开窗口、有 OpenGL 的环境；由
``tests/test_render_smoke.py`` 在子进程里调用，CI（无显卡）上跳过。
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0,
              "allow_tension": 3.05e8, "allow_compression": 3.05e8,
              "yield_stress": 3.55e8}]
SECTIONS = [{"name": "COL", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7,
             "cy": 0.15, "cz": 0.1},
            {"name": "BEAM", "A": 0.0186, "Iy": 5.2e-5, "Iz": 2.47e-3, "J": 1.4e-6,
             "cy": 0.2, "cz": 0.1}]


def build_session():
    from agent import Session

    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0, 6.0], storeys=[3.6, 3.6], bays=[6.0],
                     beam_load=3e4, column_section="COL", beam_section="BEAM",
                     material="Q355")
    return s


def _colourful_fraction(image) -> float:
    """画面里"有颜色"的像素占比。灰阶背景、灰色杆件、白字都不算。"""
    import numpy as np

    rgb = np.asarray(image, dtype=float)[..., :3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    return float((spread > 60).mean())


def main(report_path: Path | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    errors: list[str] = []
    sys.excepthook = lambda kind, value, tb: errors.append(
        "".join(traceback.format_exception(kind, value, tb)))

    app = QApplication.instance() or QApplication([])
    from desktop import app as desktop_app
    from desktop import qt_style, theme
    desktop_app._normalise_application_font(app)
    desktop_app._load_cjk_font(app)
    qt_style.install(app)
    app.setStyleSheet(theme.STYLESHEET)

    import desktop.viewport as viewport_module
    from desktop.main_window import MainWindow

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("真渲染已启用", viewport_module.CAN_RENDER,
          "QT_QPA_PLATFORM 是 offscreen 时视口不画，本脚本没有意义")

    window = MainWindow(build_session())
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    window.solve()
    window.runner.wait(60000)
    app.processEvents()
    check("求解成功", window.result is not None and window.result.ok)

    plotter = window.viewport.plotter

    def settle() -> None:
        for _ in range(5):
            app.processEvents()

    def shot():
        plotter.render()
        return plotter.screenshot(return_img=True)

    coloured_modes = {"云图", "内力图", "应力比"}
    for mode in ("模型", "变形", "云图", "内力图", "应力比", "模态"):
        before = len(errors)
        if mode == "应力比":
            window.show_utilization()
            window.runner.wait(60000)
        else:
            window.set_mode(mode)
        settle()
        image = shot()
        check(f"{mode}：绘制无异常", len(errors) == before,
              errors[-1][-400:] if len(errors) > before else "")
        check(f"{mode}：画面非空", float(image.std()) > 5.0,
              f"像素标准差 {float(image.std()):.1f}")
        if mode in coloured_modes:
            fraction = _colourful_fraction(image)
            check(f"{mode}：画面有彩色", fraction > 0.003, f"彩色像素 {fraction:.2%}")
            try:
                bar = plotter.scalar_bar
            except (AttributeError, IndexError, KeyError, StopIteration):
                bar = None
            check(f"{mode}：有色标", bar is not None)

    # 分级云图走的是另一条几何（按色带切段），单独画一遍
    window.results.levels.setCurrentIndex(window.results.levels.findData(12))
    window.set_mode("云图")
    settle()
    check("分级云图：画面有彩色", _colourful_fraction(shot()) > 0.003)

    # 抽屉全开：都在视口内、互不重叠；色标躲开被盖住的部分
    window.tree_dock.show()
    window.chat_dock.show()
    window.results_dock.show()
    settle()
    area = window.drawers.viewport_rect()
    geometry = {side: d.geometry() for side, d in window.drawers.drawers.items()}
    for side, rect in geometry.items():
        check(f"{side} 抽屉在视口内", area.contains(rect),
              f"抽屉 {rect.getRect()} / 视口 {area.getRect()}")
        check(f"{side} 抽屉就在分给它的位置", rect == window.drawers.geometry_for(side))
    check("底部抽屉不压两侧抽屉",
          not geometry["bottom"].intersects(geometry["left"])
          and not geometry["bottom"].intersects(geometry["right"]))
    left, right, bottom = window.drawers.insets()
    width = max(1, window.viewport.width())
    height = max(1, window.viewport.height())
    try:
        bar = plotter.scalar_bar
        x, y = bar.GetPosition()
        bar_right = (x + bar.GetWidth()) * width
        bar_bottom = y * height
        check("色标不在右抽屉底下", bar_right <= width - right + 1,
              f"色标右沿 {bar_right:.0f} px，右抽屉占 {right} px")
        check("色标不在底部抽屉底下", bar_bottom >= bottom - 1,
              f"色标下沿 {bar_bottom:.0f} px，底部抽屉占 {bottom} px")
    except (AttributeError, IndexError, KeyError, StopIteration):
        check("抽屉打开后仍有色标", False)
    check("右侧 AI 按钮看得见", window.agent_button.isVisible())

    check("全程无未处理异常", not errors, "\n\n".join(e[-600:] for e in errors))
    window.close()

    failed = [c for c in checks if not c["ok"]]
    report = {"ok": not failed, "checks": checks}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if report_path is not None:
        Path(report_path).write_text(text, encoding="utf-8")
    for item in checks:
        mark = "OK" if item["ok"] else "!!"
        print(f"[{mark}] {item['check']}" + (f"  — {item['detail']}"
                                              if item["detail"] and not item["ok"] else ""))
    return 0 if not failed else 1


if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM", None)
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
