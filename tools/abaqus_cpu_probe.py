"""AMD Zen 4 上 standard.exe 崩溃的第二轮探针：把"规模"和"网格质量"拆开。

上一轮我用"单个 C3D10 单元跑通了"推出"这台机器能跑实体单元"，这个推理
是错的：那个算例只有 30 个自由度，**根本进不了稀疏求解器的向量化内核**。

这台机器是 AMD Ryzen 9 7940HX（Zen 4，2023），而 Abaqus 6.14 是 2014 年的，
带的 Intel MKL 11.x 比它早九年。MKL 在 AMD 上的 CPU 分派路径和 Intel 完全
不同，而 6.14 发布时 Zen 架构还不存在。所以"P2 崩了"完全可能不是网格问题，
而是它是第一个大到进得了向量化路径的算例。

四个探针，逐个消掉一个变量：

  P3  **形状完美**的 C3D10 网格，规模与失败算例相当（~10k 自由度）
      它崩 -> 与网格质量无关，是规模/CPU 路径
      它过 -> 网格质量确实是原因，按壁厚重新定档的修复方向正确
  P4  失败的那个 .inp + MKL_DEBUG_CPU_TYPE=5
      MKL <= 2020 上强制 AMD 走 AVX2 路径的老开关
  P5  失败的那个 .inp + MKL_ENABLE_INSTRUCTIONS=SSE4_2
      最保守的指令集，绕开一切 AVX
  P6  失败的那个 .inp + memory=4gb
      默认 memory='90%'，在大内存机器上有已知的分配问题

P3 的网格是程序生成的 Kuhn 剖分：每个立方体切成 6 个全等四面体，再加边中点
升成二次单元。**所有单元形状完全一样**，不存在任何畸变——这正是它作为
对照的价值。

用法：run_abaqus_cpu_probe.bat
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WORK = ROOT / "results" / "abaqus_cpu_probe"
LOG_PATH = WORK / "probe_log.txt"
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


# 立方体的 6-四面体 Kuhn 剖分：按三个轴的每种排列取一条从 (0,0,0) 走到
# (1,1,1) 的路径，四个顶点就是一个四面体。六种排列不重不漏地填满立方体，
# 而且六个四面体全等——没有任何一个是薄片。
_PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def _kuhn_tets(cell: tuple[int, int, int]) -> list[tuple[tuple[int, int, int], ...]]:
    tets = []
    for order in _PERMUTATIONS:
        point = [0, 0, 0]
        path = [tuple(point)]
        for axis in order:
            point[axis] += 1
            path.append(tuple(point))
        tets.append(tuple(
            (cell[0] + p[0], cell[1] + p[1], cell[2] + p[2]) for p in path))
    return tets


def kuhn_c3d10_box(cells: int = 7, size_mm: float = 105.0) -> tuple[str, int, int]:
    """生成一个形状完全规则的 C3D10 立方体网格输入文件。

    返回 (输入文件文本, 节点数, 单元数)。
    """
    step = size_mm / cells
    node_id: dict[tuple[float, float, float], int] = {}
    coordinates: list[tuple[float, float, float]] = []

    def node(point: tuple[float, float, float]) -> int:
        key = tuple(round(v, 6) for v in point)
        if key not in node_id:
            coordinates.append(key)
            node_id[key] = len(coordinates)
        return node_id[key]

    def xyz(grid: tuple[int, int, int]) -> tuple[float, float, float]:
        return (grid[0] * step, grid[1] * step, grid[2] * step)

    def middle(a: tuple[float, float, float],
               b: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(0.5 * (a[i] + b[i]) for i in range(3))

    def volume(pts) -> float:
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) = pts
        a = (x2 - x1, y2 - y1, z2 - z1)
        b = (x3 - x1, y3 - y1, z3 - z1)
        c = (x4 - x1, y4 - y1, z4 - z1)
        cross = (b[1] * c[2] - b[2] * c[1],
                 b[2] * c[0] - b[0] * c[2],
                 b[0] * c[1] - b[1] * c[0])
        return (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0

    elements: list[tuple[int, ...]] = []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                for tet in _kuhn_tets((i, j, k)):
                    pts = [xyz(g) for g in tet]
                    if volume(pts) < 0.0:
                        # 负体积的单元 Abaqus 直接报错。交换两个角点翻转定向。
                        pts[2], pts[3] = pts[3], pts[2]
                    corner = [node(p) for p in pts]
                    # C3D10 的中间点顺序：5=(1,2) 6=(2,3) 7=(1,3)
                    #                     8=(1,4) 9=(2,4) 10=(3,4)
                    pairs = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
                    mids = [node(middle(pts[a], pts[b])) for a, b in pairs]
                    elements.append(tuple(corner + mids))

    lines = ["*HEADING", "regular C3D10 box, no distorted elements", "*NODE"]
    for index, point in enumerate(coordinates, 1):
        lines.append(f"{index}, {point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}")
    lines.append("*ELEMENT, TYPE=C3D10, ELSET=EALL")
    for index, nodes in enumerate(elements, 1):
        lines.append(str(index) + ", " + ", ".join(str(n) for n in nodes))
    lines += [
        "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        "206000., 0.3",
        "*NSET, NSET=BASE, GENERATE",
        f"1, {len(coordinates)}, 1",
    ]
    base = [i for i, p in enumerate(coordinates, 1) if abs(p[2]) < 1e-9]
    top = [i for i, p in enumerate(coordinates, 1) if abs(p[2] - size_mm) < 1e-9]
    lines = [line for line in lines if not line.startswith("*NSET")
             and not line.startswith(f"1, {len(coordinates)}, 1")]
    lines.append("*NSET, NSET=BASE")
    for chunk in range(0, len(base), 8):
        lines.append(", ".join(str(n) for n in base[chunk:chunk + 8]))
    lines.append("*NSET, NSET=TOP")
    for chunk in range(0, len(top), 8):
        lines.append(", ".join(str(n) for n in top[chunk:chunk + 8]))
    lines += [
        "*STEP",
        "*STATIC",
        "*BOUNDARY",
        "BASE, 1, 3",
        "*CLOAD",
        "TOP, 1, 100.0",
        "*NODE PRINT, NSET=TOP",
        "U,",
        "*END STEP",
    ]
    return "\n".join(lines) + "\n", len(coordinates), len(elements)


def run_job(executable: str, directory: Path, job: str, keep: Path,
            env_extra: dict[str, str] | None = None,
            extra_args: list[str] | None = None,
            timeout: float = 1800.0) -> bool:
    say(f"命令：{executable} job={job} "
        + " ".join(extra_args or []) + " interactive")
    if env_extra:
        say(f"环境变量：{env_extra}")
    say(f"目录：{directory}")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(env_extra or {})
    try:
        proc = subprocess.run(
            [executable, f"job={job}"] + (extra_args or []) + ["interactive"],
            cwd=str(directory), capture_output=True, text=True,
            timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        say(f"!! 超过 {timeout:.0f} 秒未结束")
        return False
    say(f"返回码 = {proc.returncode}")
    tail = (proc.stdout or "").strip().splitlines()[-14:]
    say("--- stdout 尾 ---")
    say("\n".join(tail) or "(空)")
    if (proc.stderr or "").strip():
        say("--- stderr ---")
        say((proc.stderr or "").strip()[-800:])

    keep.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            try:
                shutil.copy2(path, keep / path.name)
            except OSError:
                pass
    status = directory / (job + ".sta")
    ok = status.is_file() and "COMPLETED SUCCESSFULLY" in status.read_text(
        encoding="utf-8", errors="replace")
    say(">>> " + ("跑通了" if ok else "没跑通"))
    return ok


def main() -> int:
    import abaqus_backend

    say(f"Abaqus / AMD 探针　{_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    say("CPU：AMD Ryzen 9 7940HX（Zen 4, 2023）　Abaqus 6.14（2014）")
    executable = abaqus_backend.find_abaqus()
    say(f"find_abaqus() -> {executable!r}")
    if executable is None:
        say("找不到 Abaqus，无法继续。")
        return 1
    WORK.mkdir(parents=True, exist_ok=True)
    ascii_root = Path(tempfile.mkdtemp(prefix="abqcpu_"))
    say(f"ASCII 临时目录：{ascii_root}")

    say()
    say("=" * 70)
    say("# P3　形状完美的 C3D10 网格，规模与失败算例相当")
    say("=" * 70)
    say("它崩 => 与网格质量无关，是规模或 CPU 路径的问题。")
    say("它过 => 网格质量确实是原因，按壁厚重新定档的方向正确。")
    deck, nodes, elements = kuhn_c3d10_box()
    say(f"生成：{nodes} 节点 / {elements} 单元 ≈ {3 * nodes} 自由度")
    say("（失败算例是 3725 节点 / 1837 单元 / 10731 方程）")
    say("所有单元由 Kuhn 剖分得到，彼此全等，畸变为零。")
    say()
    p3_dir = ascii_root / "p3"
    p3_dir.mkdir(parents=True)
    (p3_dir / "regular_box.inp").write_text(deck, encoding="ascii")
    p3 = run_job(executable, p3_dir, "regular_box", WORK / "p3_regular_box")

    source = ROOT / "results" / "solid_joint" / "node_2_LC1" / "solid_joint_0.inp"
    variants: list[tuple[str, str, dict[str, str], list[str]]] = [
        ("P4　+ MKL_DEBUG_CPU_TYPE=5",
         "MKL <= 2020 上强制 AMD 走 AVX2 路径的老开关。",
         {"MKL_DEBUG_CPU_TYPE": "5"}, []),
        ("P5　+ MKL_ENABLE_INSTRUCTIONS=SSE4_2",
         "最保守的指令集，绕开一切 AVX。",
         {"MKL_ENABLE_INSTRUCTIONS": "SSE4_2"}, []),
        ("P6　+ memory=4gb",
         "默认 memory='90%'，在大内存机器上有已知的分配问题。",
         {}, ["memory=4gb"]),
    ]
    outcomes: dict[str, bool | None] = {}
    if not source.is_file():
        say()
        say(f"找不到 {source}，P4/P5/P6 跳过。")
        for title, _why, _env, _args in variants:
            outcomes[title] = None
    else:
        for index, (title, why, env_extra, extra_args) in enumerate(variants):
            say()
            say("=" * 70)
            say(f"# {title}")
            say("=" * 70)
            say(why)
            say()
            directory = ascii_root / f"v{index}"
            directory.mkdir(parents=True)
            shutil.copy2(source, directory / "solid_joint_0.inp")
            outcomes[title] = run_job(
                executable, directory, "solid_joint_0",
                WORK / f"v{index}", env_extra, extra_args)

    say()
    say("=" * 70)
    say("# 结论")
    say("=" * 70)
    say(f"P3（形状完美 / 同等规模）{'通过' if p3 else '失败'}")
    for title, value in outcomes.items():
        say(f"{title}　{'跳过' if value is None else ('通过' if value else '失败')}")
    say()
    if not p3:
        say("P3 也崩了 —— 单元形状完美、规模相当，仍然崩。")
        say("那就与我们的网格无关：这台 AMD Zen 4 上的 Abaqus 6.14 求解器")
        say("在这个规模上本身就跑不了。上一轮按壁厚重新定档的修复仍然")
        say("是对的（用 66 mm 单元铺 8 mm 壁本来就不该做），但它治不了这个。")
        good = [t for t, v in outcomes.items() if v]
        if good:
            say(f"而 {good[0]} 跑通了 —— 这就是可用的绕法，应该固化进代码。")
        else:
            say("三个开关都没救回来。下一步要么换更新的 Abaqus，")
            say("要么承认这台机器跑不了实体子模型，报告里如实写明。")
    else:
        say("P3 通过 —— 同等规模的规则网格能跑，说明 CPU 和求解器没问题，")
        say("崩溃确实来自我们那套布尔合并网格的畸变单元。")
        say("按壁厚重新定档的修复方向正确，直接重跑 run_joint.bat 即可。")
    say()
    say(f"各作业产物已保存到 {WORK}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        WORK.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        print(f"\n日志已写入 {LOG_PATH}")
    sys.exit(code)
