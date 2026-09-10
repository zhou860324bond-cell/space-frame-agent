"""节点局部实体分析的首跑冒烟脚本。

`src/solid_joint.py` 分两半：建规格（纯 Python，已被 tests/test_solid_joint.py
按解析解钉死）和跑 Abaqus（需要本机 6.14，CI 里跑不了）。这个脚本专门跑后一半，
用一个自带的小算例，而不是用户的模型——**第一次要验的是管线本身通不通，不是
你的数据对不对**。两件事混在一起，出错时分不清是谁的问题。

它把每一步的结果和完整 traceback 都写进日志，出错也不中断后面的步骤，
这样一次运行就能拿到全部诊断信息，不用来回试。

用法：run_joint.bat
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LOG_PATH = ROOT / "results" / "solid_joint" / "smoke_log.txt"
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


def step(title: str, fn):
    """跑一步，失败不抛出——把 traceback 记下来继续。

    第一次跑通常会连着踩好几个坑。中断在第一个上，就得来回跑很多次；
    一次跑完拿到全部错误要省事得多。
    """
    say()
    say("=" * 70)
    say(f"# {title}")
    say("=" * 70)
    try:
        return fn()
    except BaseException:                       # noqa: BLE001 - 诊断脚本，全接
        say("!! 失败：")
        say(traceback.format_exc())
        return None


def main() -> int:
    say(f"节点局部实体冒烟测试　{_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    say(f"python {sys.version.split()[0]}　工作目录 {ROOT}")

    def find():
        import abaqus_backend
        found = abaqus_backend.find_abaqus()
        say(f"find_abaqus() -> {found!r}")
        if found is None:
            say("在下列位置找过：")
            for hint in abaqus_backend._WINDOWS_HINTS:
                exists = Path(hint).is_dir()
                say(f"  {'有' if exists else '无'}　{hint}")
                if exists:
                    for path in sorted(Path(hint).glob("*.bat"))[:20]:
                        say(f"      {path.name}")
            say("如果上面列出了 abq6xxx.bat 却仍返回 None，就是名字匹配的问题，"
                "把这段日志发回来。")
        return found

    executable = step("1. 找 Abaqus", find)

    def build():
        import sections as S
        from agent import Session

        session = Session()
        # L 形节点：柱 (1)->(2) 3 m，梁 (2)->(3) 4 m，梁端 20 kN 向下。
        # 圆管 P219x8——选圆管是因为实体子模型第一版只支持圆截面。
        assert session.add_nodes([[0, 0, 0], [0, 0, 3.0], [4.0, 0, 3.0]]).ok
        assert session.define_materials_and_sections(
            [{"name": "Q355", "E": 2.06e11, "nu": 0.3}],
            [S.circular_tube("P219x8", 0.219, 0.008)]).ok
        assert session.add_members([[1, 2], [2, 3]],
                                   section="P219x8", material="Q355").ok
        assert session.set_supports([1], fix=[1, 1, 1, 1, 1, 1]).ok
        assert session.set_load_cases([{"name": "LC1"}]).ok
        assert session.set_nodal_load(3, [0, 0, -20000.0, 0, 0, 0],
                                      case_name="LC1").ok
        result = session.solve_model()
        assert result.ok, result.payload
        say("梁模型求解完成。节点 2 是柱与梁的交点。")
        return session

    session = step("2. 建并求解梁模型（不需要 Abaqus）", build)
    if session is None:
        return 1

    def dry():
        result = session.analyze_joint_solid(node_id=2, dry_run=True)
        say(f"ok = {result.ok}")
        say(json.dumps(result.payload, ensure_ascii=False, indent=2, default=str))
        if result.ok:
            import math
            inertia = math.pi * (0.219 ** 4 - 0.203 ** 4) / 64.0
            expected = 20000.0 * 4.0 * 0.1095 / inertia / 1e6
            got = result.payload["nominal_normal_mpa"]
            say(f"\n手算校核　M·c/I = {expected:.4f} MPa　程序给出 {got:.4f} MPa"
                f"　相对误差 {abs(got - expected) / expected:.2e}")
        return result

    step("3. dry_run：只建规格，不调 Abaqus", dry)

    if executable is None:
        say()
        say("没找到 Abaqus，跳过实体分析。前面的规格部分仍然有效。")
        return 1

    def solid():
        # 用默认档位，不再手动传粗网格。上一次我为了"先看通不通别等太久"
        # 传了 [60,45,35] mm——而管壁只有 8 mm，网格器只能铺出退化四面体，
        # 1837 个单元里 725 个畸变，standard.exe 直接崩。
        # 现在默认档位由壁厚决定（t、t/1.25、t/1.55），最细一档单元数不少，
        # 慢是正常的。
        import solid_joint
        spec = solid_joint.prepare_joint_spec(session, 2)
        reference = solid_joint.mesh_reference_length(spec)
        say(f"网格参考尺寸 = {reference:.3g} mm（由最薄管壁定，不是由直径定）")
        say(f"档位：{[round(reference / r, 3) for r in (1.0, 1.25, 1.55)]} mm")
        say("最细一档单元数可能上十万，请给它几分钟。")
        result = session.analyze_joint_solid(node_id=2)
        say(f"ok = {result.ok}")
        say(json.dumps(result.payload, ensure_ascii=False, indent=2, default=str))
        if result.ok:
            payload = result.payload
            say()
            say(f"峰值收敛判定：{payload['peak_convergence']['verdict']}")
            say(f"　　理由：{payload['peak_convergence']['reason']}")
            kt = payload.get("stress_concentration_factor")
            if kt is None:
                say(f"应力集中系数：未给出——{payload['stress_concentration_refused']}")
            else:
                say(f"应力集中系数 Kt = {kt:.3f}"
                    f"（{payload['stress_concentration_basis']}）")
            for member, item in (payload.get("hot_spot_extrapolation") or {}).items():
                say(f"　　臂 {member}: {item}")
            for mesh in payload.get("meshes") or []:
                error = mesh.get("surface_samples_error")
                if error:
                    say(f"!! 网格 {mesh['mesh_size_mm']} mm 取样失败：{error}")
        return result

    step("4. 实体子模型（调用 Abaqus，可能要几分钟）", solid)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        print(f"\n日志已写入 {LOG_PATH}")
    sys.exit(code)
