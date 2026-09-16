"""离屏渲染桌面端三维视口，把结果存成 PNG。

**为什么要有这个脚本**：视口的观感问题（管子太细、支座太抢眼、色标底端溶进
背景）只有渲染出来才看得见，而 `desktop/scene.py` 和 `desktop/theme.py` 都
不依赖 Qt——只用 PyVista。所以不必启动整个界面，也不必人工截图，
一条命令就能把几张关键视图落成文件，改完再跑一次即可对比。

用法（在项目根目录）：

    .venv\\Scripts\\python.exe tools\\render_viewport.py

输出写到 `.pytest-viz\\`（已被 .gitignore 忽略）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in ("src", "desktop", str(ROOT)):
    path = str(ROOT / extra) if extra in ("src", "desktop") else extra
    if path not in sys.path:
        sys.path.insert(0, path)

import pyvista as pv                                        # noqa: E402

import sections as sec                                      # noqa: E402
from agent import Session                                   # noqa: E402
from desktop import scene, theme                            # noqa: E402

OUT = ROOT / ".pytest-viz"


def build() -> Session:
    """两层两跨刚架：三种支座、节点力、节点力矩、均布荷载、一个杆端铰。"""
    col = sec.i_section("COL", 0.4, 0.2, 0.010, 0.016)
    beam = sec.rectangle("BEAM", 0.2, 0.45)
    xs, zs = [0.0, 6.0, 12.0], [0.0, 3.6, 7.2]
    nid, nodes, key = {}, [], 1
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            nid[(ix, iz)] = key
            nodes.append({"id": key, "x": x, "y": 0.0, "z": z})
            key += 1
    members, mid = [], 1
    for ix in range(3):
        for iz in range(2):
            members.append({"id": mid, "i": nid[(ix, iz)], "j": nid[(ix, iz + 1)],
                            "section": "COL", "material": "Q355"})
            mid += 1
    for iz in (1, 2):
        for ix in range(2):
            member = {"id": mid, "i": nid[(ix, iz)], "j": nid[(ix + 1, iz)],
                      "section": "BEAM", "material": "Q355"}
            if iz == 2 and ix == 1:
                member["releases"] = {"j": ["rz"]}
            members.append(member)
            mid += 1

    fixes = {n["id"]: [0, 1, 0, 1, 0, 1] for n in nodes}     # 面外约束
    for node, mask in ((nid[(0, 0)], [1, 1, 1, 1, 1, 1]),    # 固接
                       (nid[(1, 0)], [1, 1, 1, 0, 0, 0]),    # 铰接
                       (nid[(2, 0)], [0, 1, 1, 0, 0, 0])):   # 滚动
        fixes[node] = [max(a, b) for a, b in zip(fixes[node], mask, strict=True)]

    model = {
        "units": "N-m-Pa",
        "materials": [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}],
        "sections": [col, beam], "nodes": nodes, "members": members,
        "supports": [{"node": k, "fix": v} for k, v in fixes.items()],
        "load_cases": [{
            "name": "D",
            "nodal_loads": [{"node": nid[(0, 2)], "load": [30e3, 0, 0, 0, 0, 0]},
                            {"node": nid[(2, 2)], "load": [0, 0, 0, 0, 25e3, 0]}],
            "member_loads": [{"member": 7, "w": [0, 0, -18e3]},
                             {"member": 8, "w": [0, 0, -18e3]},
                             {"member": 9, "w": [0, 0, -12e3]}],
        }],
    }
    session = Session()
    result = session.set_model(model=model)
    assert result.ok, result.payload
    assert session.solve_model().ok
    return session


def plotter() -> pv.Plotter:
    p = pv.Plotter(off_screen=True, window_size=(1400, 900))
    p.set_background(theme.VIEWPORT_BG)
    return p


def shot(p: pv.Plotter, name: str) -> None:
    p.camera_position = "xz"
    p.camera.azimuth = -35
    p.camera.elevation = 18
    p.reset_camera()
    p.camera.zoom(1.25)              # reset_camera 留白太多；再大支座会被裁掉
    path = OUT / name
    p.screenshot(str(path))
    p.close()
    print("已输出", path)


def _bar_title(tubes, component: str, clim, base: str) -> str:
    """与 viewport.show_contour 同一条披露规则：裁剪了就写在色标上。"""
    if scene.clim_is_clipped(tubes, component, clim):
        return f"{base}  [clip p{scene.CONTOUR_PERCENTILE:.0f}]"
    return base


def _scaled_tubes(frame, solution, component: str):
    """带工程显示单位的云图管。与 viewport.show_contour 同一套换算——
    脚本若略过它，核对的就不是用户真正看到的那张图。"""
    from units import of as unit_system
    tubes = scene.member_tubes(frame, solution, "D", scalars=component)
    system = unit_system(frame)
    tubes[component] = tubes[component] * (
        system.moment_scale if component in {"T", "My", "Mz", "M"}
        else system.force_scale)
    clim = scene.contour_clim(tubes, component)
    return tubes, clim, system


def main() -> None:
    OUT.mkdir(exist_ok=True)
    session = build()
    frame, solution = session.frame, session.solution

    # 1. 模型视图：杆件管 + 节点 + 铰 + 支座 + 荷载，与 viewport.show_model 一致
    p = plotter()
    p.add_mesh(scene.member_tubes(frame), color=theme.MEMBER, smooth_shading=True)
    p.add_mesh(scene.node_points(frame), color=theme.INK,
               point_size=6, render_points_as_spheres=True)
    hinges = scene.hinge_glyphs(frame)
    if hinges.n_points:
        p.add_mesh(hinges, color=theme.HINGE)
    for mesh in scene.support_glyphs(frame).values():
        p.add_mesh(mesh, color=theme.SUPPORT)
    for label, mesh in scene.load_arrows(frame, "D").items():
        color = (theme.ACCENT if label in {"节点力矩", "给定位移", "给定转角"}
                 else theme.LOAD)
        p.add_mesh(mesh, color=color)
    shot(p, "vp_model.png")

    # 2. 合弯矩云图：顺序色标，从零起
    p = plotter()
    tubes, clim, system = _scaled_tubes(frame, solution, "M")
    p.add_mesh(tubes, scalars="M", cmap=theme.sequential_cmap(), clim=clim,
               smooth_shading=True,
               scalar_bar_args=dict(title=_bar_title(tubes, "M", clim,
                                                     f"|M| ({system.moment_unit})"),
                                    color=theme.INK_MUTED,
                                    title_font_size=14, label_font_size=12,
                                    n_labels=5, width=0.30, height=0.045,
                                    position_x=0.66, position_y=0.03))
    for mesh in scene.support_glyphs(frame).values():
        p.add_mesh(mesh, color=theme.SUPPORT)
    shot(p, "vp_contour_M.png")

    # 3. 轴力云图：发散色标，关于零对称
    p = plotter()
    tubes, clim, system = _scaled_tubes(frame, solution, "N")
    p.add_mesh(tubes, scalars="N", cmap=theme.DIVERGING, clim=clim,
               smooth_shading=True,
               scalar_bar_args=dict(title=_bar_title(tubes, "N", clim,
                                                     f"N ({system.force_unit})"),
                                    color=theme.INK_MUTED,
                                    title_font_size=14, label_font_size=12,
                                    n_labels=5, width=0.30, height=0.045,
                                    position_x=0.66, position_y=0.03))
    for mesh in scene.support_glyphs(frame).values():
        p.add_mesh(mesh, color=theme.SUPPORT)
    shot(p, "vp_contour_N.png")


if __name__ == "__main__":
    main()
