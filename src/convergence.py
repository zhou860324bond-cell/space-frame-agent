"""把不同物理含义的收敛记录整理成同一套可审查表。

这里不启动求解，也不把云图显示采样当成分析网格。调用方可以把已经完成的
Abaqus B31 细化记录、节点实体多档网格结果或非线性增量历史交进来；本模块只
负责统一字段、沿用上游诊断结论，并明确每类数据能说明什么、不能说明什么。
"""

from __future__ import annotations

import math
from typing import Any, Iterable


COLUMNS = ["专项", "档位", "响应量", "响应值", "变化 / 差异 (%)", "结论", "适用限制"]


def solid_mesh_plan(reference_mm: float, *, backend: str = "native",
                    mode: str = "convergence") -> list[float]:
    """返回节点实体任务的保守网格档位，不在这里启动网格或求解。"""
    reference = _number(reference_mm)
    if reference is None or reference <= 0.0:
        raise ValueError("实体网格参考尺寸必须是正数")
    if backend == "native":
        factors = (2.5,) if mode == "quick" else (5.0, 3.75, 2.5)
    elif backend == "abaqus":
        if mode == "quick":
            raise ValueError("Abaqus 实体后端只接受三档收敛任务")
        # Abaqus 的全局四面体尺寸不能粗于最薄壁厚；最细档为 0.5t，
        # 使最粗到最细达到两倍，满足热点稳定性门禁。
        factors = (1.0, 0.7, 0.5)
    else:
        raise ValueError(f"未知实体后端 {backend!r}")
    if mode not in {"quick", "convergence"}:
        raise ValueError(f"未知网格方案 {mode!r}")
    return [round(reference * factor, 6) for factor in factors]


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _relative_change(previous: Any, current: Any) -> float | None:
    a, b = _number(previous), _number(current)
    if a is None or b is None:
        return None
    return abs(b - a) / max(abs(a), abs(b), 1e-12)


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _verdict_label(verdict: str) -> tuple[str, str]:
    return {
        "converging": ("已进入稳定区", "pass"),
        "diverging": ("峰值疑似奇异发散", "fail"),
        "inconclusive": ("证据不足", "unclear"),
    }.get(verdict, ("尚未诊断", "unclear"))


def _generic_scalar_verdict(values: list[float], tolerance: float) -> tuple[str, str]:
    """给 B31 这类平滑响应一个保守门禁；两档不能正式盖章。"""
    if len(values) < 3:
        return "inconclusive", "少于三档网格，只能看变化，不能确认趋势"
    changes = [_relative_change(a, b) for a, b in zip(values, values[1:])]
    last = changes[-1]
    if last is not None and last <= tolerance:
        return "converging", f"最后两档响应变化 {last:.2%}，不超过 {tolerance:.0%}"
    return "inconclusive", (
        f"最后两档响应变化 {_percent(last)}，尚未进入 {tolerance:.0%} 稳定区")


def b31_rows(records: Iterable[dict[str, Any]], *, case: str | None = None,
             tolerance: float = 0.05) -> tuple[list[list[Any]], list[str | None], list[str]]:
    """整理 ``abaqus_bench.b31_refinement.run`` 返回的记录。"""
    selected = [row for row in records if case is None or row.get("case") == case]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(str(row.get("case") or "未命名算例"), []).append(row)

    rows: list[list[Any]] = []
    marks: list[str | None] = []
    notes: list[str] = []
    for name, series in grouped.items():
        series.sort(key=lambda row: float(row.get("factor") or 0.0))
        values = [_number(row.get("abaqus_mm")) for row in series]
        clean = [value for value in values if value is not None]
        verdict, reason = _generic_scalar_verdict(clean, tolerance)
        label, mark = _verdict_label(verdict)
        notes.append(f"B31 {name}：{reason}。")
        previous = None
        for index, (record, value) in enumerate(zip(series, values)):
            factor = record.get("factor")
            level = f"{factor}× / {record.get('members', '—')} 单元"
            change = _relative_change(previous, value)
            final = index == len(series) - 1
            rows.append([
                f"B31 · {name}", level, "最大节点位移 (mm)", value,
                None if change is None else 100.0 * change,
                label if final else "细化记录",
                "只验证 B31 离散响应；梁理论、剪切面积和截面参数必须另行对齐。",
            ])
            marks.append(mark if final else None)
            if value is not None:
                previous = value
    return rows, marks, notes


def solid_rows(payload: dict[str, Any]) -> tuple[list[list[Any]], list[str | None], list[str]]:
    """优先展示热点外推收敛；没有热点路径时退回奇异峰值诊断。"""
    rows: list[list[Any]] = []
    marks: list[str | None] = []
    notes: list[str] = []
    hotspot = payload.get("hot_spot_convergence") or {}
    for arm, diagnosis in hotspot.items():
        values = diagnosis.get("values") or []
        if not values:
            continue
        label, mark = _verdict_label(str(diagnosis.get("verdict") or ""))
        notes.append(f"实体热点臂 {arm}：{diagnosis.get('reason') or label}。")
        previous = None
        for index, item in enumerate(values):
            value = _number(item.get("hot_spot_mpa"))
            size = item.get("hotspot_mesh_size_mm")
            change = _relative_change(previous, value)
            final = index == len(values) - 1
            rows.append([
                f"实体热点 · 臂 {arm}", f"h={size} mm",
                "0.4t/1.0t 热点应力 (MPa)",
                value, None if change is None else 100.0 * change,
                label if final else "细化记录",
                "热点外推可用于 Kt 门禁；相贯线节点峰值仍可能因几何奇异而发散。",
            ])
            marks.append(mark if final else None)
            if value is not None:
                previous = value

    if rows:
        return rows, marks, notes

    meshes = sorted(payload.get("meshes") or [],
                    key=lambda row: -float(row.get("mesh_size_mm") or 0.0))
    diagnosis = payload.get("peak_convergence") or {}
    label, mark = _verdict_label(str(diagnosis.get("verdict") or ""))
    notes.append(f"实体峰值：{diagnosis.get('reason') or label}。")
    previous = None
    for index, mesh in enumerate(meshes):
        value = _number(mesh.get("max_abs_principal_mpa"))
        change = _relative_change(previous, value)
        final = index == len(meshes) - 1
        rows.append([
            "实体峰值", f"h={mesh.get('mesh_size_mm')} mm",
            "最大绝对主应力 (MPa)",
            value, None if change is None else 100.0 * change,
            label if final else "细化记录",
            "节点峰值只用于识别奇异性；未稳定时不得据此给出正式 Kt。",
        ])
        marks.append(mark if final else None)
        if value is not None:
            previous = value
    return rows, marks, notes


def _backend_family(payload: dict[str, Any]) -> str | None:
    name = str(payload.get("backend") or "").lower()
    if "native" in name:
        return "native"
    if "abaqus" in name:
        return "abaqus"
    return None


def _arm_series(payload: dict[str, Any], arm: str) -> list[tuple[float, float]]:
    diagnosis = (payload.get("hot_spot_convergence") or {}).get(arm) or {}
    out = []
    for item in diagnosis.get("values") or []:
        size = _number(item.get("hotspot_mesh_size_mm"))
        stress = _number(item.get("hot_spot_mpa"))
        if size is not None and size > 0.0 and stress is not None:
            out.append((size, stress))
    return sorted(out, reverse=True)


def _difference_mark(relative: float, tolerance: float) -> tuple[str, str]:
    if relative <= tolerance:
        return f"差异不超过 {tolerance:.0%}", "pass"
    if relative <= 2.0 * tolerance:
        return "存在可见差异，建议复核网格与取样路径", "unclear"
    return "差异超过对标阈值", "fail"


def solid_backend_rows(payloads: Iterable[dict[str, Any]], *,
                       tolerance: float = 0.05,
                       size_tolerance: float = 0.05
                       ) -> tuple[list[list[Any]], list[str | None], list[str]]:
    """比较同一节点的 native/Abaqus 热点结果，不比较奇异节点峰值。"""
    by_backend = {_backend_family(payload): payload for payload in payloads
                  if _backend_family(payload) is not None}
    if not {"native", "abaqus"}.issubset(by_backend):
        present = "、".join(sorted(by_backend)) or "无"
        return [], [], [f"实体双后端对标尚未形成：已有 {present}，还需另一后端结果。"]
    native, abaqus = by_backend["native"], by_backend["abaqus"]
    signature_n = (native.get("node_id"), native.get("case"))
    signature_a = (abaqus.get("node_id"), abaqus.get("case"))
    if signature_n != signature_a:
        return [[
            "实体双后端对标", "不可比较", "节点 / 工况签名",
            f"native {signature_n} / Abaqus {signature_a}", None,
            "输入不一致", "必须使用同一节点、同一工况和同一整体梁模型。",
        ]], ["fail"], ["实体双后端结果的节点或工况不一致，已拒绝数值比较。"]

    nominal_n = _number(native.get("nominal_normal_mpa"))
    nominal_a = _number(abaqus.get("nominal_normal_mpa"))
    nominal_difference = _relative_change(nominal_n, nominal_a)
    if nominal_difference is None or nominal_difference > 1e-6:
        return [[
            "实体双后端对标", "不可比较", "名义正应力 (MPa)",
            f"{nominal_n} / {nominal_a}",
            None if nominal_difference is None else 100.0 * nominal_difference,
            "输入基准不一致", "Kt 分母不同，继续比较会把模型差异误认为求解器差异。",
        ]], ["fail"], ["两个后端的名义应力不一致，已拒绝 Kt 对标。"]

    rows: list[list[Any]] = []
    marks: list[str | None] = []
    notes: list[str] = []
    arms = sorted(set((native.get("hot_spot_convergence") or {}))
                  | set((abaqus.get("hot_spot_convergence") or {})), key=str)
    matched = 0
    for arm in arms:
        native_series = _arm_series(native, arm)
        abaqus_series = _arm_series(abaqus, arm)
        used_abaqus: set[int] = set()
        for native_size, native_stress in native_series:
            choices = [(abs(abaqus_size - native_size) /
                        max(abaqus_size, native_size), index,
                        abaqus_size, abaqus_stress)
                       for index, (abaqus_size, abaqus_stress)
                       in enumerate(abaqus_series) if index not in used_abaqus]
            if not choices:
                continue
            size_difference, index, abaqus_size, abaqus_stress = min(choices)
            if size_difference > size_tolerance:
                continue
            used_abaqus.add(index)
            difference = _relative_change(native_stress, abaqus_stress) or 0.0
            rows.append([
                f"实体同尺度对标 · 臂 {arm}",
                f"h={native_size:g} / {abaqus_size:g} mm",
                "热点应力 native / Abaqus (MPa)",
                f"{native_stress:.6g} / {abaqus_stress:.6g}",
                100.0 * difference, "同尺度记录",
                "只比较实际热点局部尺寸相近的档位；单档一致不替代各自的收敛检查。",
            ])
            marks.append(None)
            matched += 1

        native_diag = (native.get("hot_spot_convergence") or {}).get(arm) or {}
        abaqus_diag = (abaqus.get("hot_spot_convergence") or {}).get(arm) or {}
        stable = (native_diag.get("verdict") == "converging"
                  and abaqus_diag.get("verdict") == "converging"
                  and native_series and abaqus_series)
        if not stable:
            rows.append([
                f"实体收敛解对标 · 臂 {arm}", "证据不足",
                "0.4t/1.0t 热点应力", "—", None, "暂不可比较",
                "两个后端都必须先完成各自三档热点收敛，不能拿未稳定结果判断求解器差异。",
            ])
            marks.append("unclear")
            continue
        native_size, native_stress = native_series[-1]
        abaqus_size, abaqus_stress = abaqus_series[-1]
        difference = _relative_change(native_stress, abaqus_stress) or 0.0
        label, mark = _difference_mark(difference, tolerance)
        rows.append([
            f"实体收敛解对标 · 臂 {arm}",
            f"各自最细稳定档 h={native_size:g} / {abaqus_size:g} mm",
            "热点应力 native / Abaqus (MPa)",
            f"{native_stress:.6g} / {abaqus_stress:.6g}",
            100.0 * difference, label,
            "比较的是各自已收敛热点解；网格不必同构，节点奇异峰值不参与后端判定。",
        ])
        marks.append(mark)

    kt_n = _number(native.get("stress_concentration_factor"))
    kt_a = _number(abaqus.get("stress_concentration_factor"))
    if kt_n is not None and kt_a is not None:
        difference = _relative_change(kt_n, kt_a) or 0.0
        label, mark = _difference_mark(difference, tolerance)
        rows.append([
            "实体 Kt 对标", "正式发布值", "Kt native / Abaqus",
            f"{kt_n:.6g} / {kt_a:.6g}", 100.0 * difference, label,
            "两边都通过全部候选杆臂门禁后才比较；该差异不包含焊缝几何建模误差。",
        ])
        marks.append(mark)
    else:
        rows.append([
            "实体 Kt 对标", "未形成", "Kt native / Abaqus",
            f"{kt_n if kt_n is not None else '—'} / {kt_a if kt_a is not None else '—'}",
            None, "至少一侧未发布", "先补齐全部候选杆臂的热点收敛证据。",
        ])
        marks.append("unclear")
    notes.append(
        f"实体双后端对标：匹配到 {matched} 个同尺度热点档位；正式判断采用各自已收敛解。")
    return rows, marks, notes


def nonlinear_rows(analysis: dict[str, Any], *, case: str | None = None
                   ) -> tuple[list[list[Any]], list[str | None], list[str]]:
    """整理非线性求解的增量末次迭代记录，不冒充网格收敛。"""
    histories = analysis.get("convergence") or {}
    selected = histories.items() if case is None else [(case, histories.get(case) or [])]
    tolerance = _number(analysis.get("tolerance")) or 0.0
    rows: list[list[Any]] = []
    marks: list[str | None] = []
    notes: list[str] = []
    for name, history in selected:
        if not history:
            continue
        for item in history:
            ratio = _number(item.get("relative_residual"))
            quantity = "末次迭代相对残差"
            if ratio is None:
                ratio = _number(item.get("relative_change"))
                quantity = "末次迭代位移变化"
            ok = ratio is not None and ratio <= tolerance
            rows.append([
                f"非线性增量 · {name}",
                f"增量 {item.get('increment')} / "
                f"λ={(_number(item.get('load_factor')) or 0.0):.3g}",
                f"{quantity} (无量纲)", ratio, "不同荷载级，不横向比较",
                f"{item.get('iterations', '—')} 次迭代收敛" if ok else "需要复核",
                "这是增量迭代收敛，不代表空间网格无关，也不证明本构模型适用。",
            ])
            marks.append("pass" if ok else "unclear")
        notes.append(
            f"非线性 {name}：共 {len(history)} 个荷载增量；记录的是每个增量的最终迭代状态。")
    return rows, marks, notes


def build_workspace(*, analysis: dict[str, Any] | None = None,
                    solid_payload: dict[str, Any] | None = None,
                    solid_payloads: Iterable[dict[str, Any]] | None = None,
                    b31_records: Iterable[dict[str, Any]] | None = None,
                    case: str | None = None) -> dict[str, Any]:
    """汇总已有记录；没有专项记录时明确显示“不适用”，不制造假数据。"""
    rows: list[list[Any]] = []
    marks: list[str | None] = []
    notes: list[str] = []
    if b31_records is not None:
        part_rows, part_marks, part_notes = b31_rows(b31_records, case=case)
        rows.extend(part_rows)
        marks.extend(part_marks)
        notes.extend(part_notes)
    if solid_payload:
        part_rows, part_marks, part_notes = solid_rows(solid_payload)
        rows.extend(part_rows)
        marks.extend(part_marks)
        notes.extend(part_notes)
    if solid_payloads is not None:
        part_rows, part_marks, part_notes = solid_backend_rows(solid_payloads)
        rows.extend(part_rows)
        marks.extend(part_marks)
        notes.extend(part_notes)
    if analysis and analysis.get("convergence"):
        part_rows, part_marks, part_notes = nonlinear_rows(analysis, case=case)
        rows.extend(part_rows)
        marks.extend(part_marks)
        notes.extend(part_notes)

    if not rows:
        rows = [[
            "当前线性梁结果", "当前分析网格", "节点位移与杆端内力", "已完成",
            "—", "尚未执行专项细化",
            "云图的 41 个显示点不参与刚度组装。需要网格证据时，应运行 B31 多档细化或实体三档局部网格。",
        ]]
        marks = ["unclear"]
        notes = ["当前只有单档线性梁分析结果，因此不能给出网格收敛结论。"]

    if "fail" in marks:
        label, severity = "存在发散迹象", "fail"
    elif "unclear" in marks:
        label, severity = "需要更多证据", "unclear"
    else:
        label, severity = "专项记录已通过", "pass"
    return {
        "label": label, "severity": severity, "columns": list(COLUMNS),
        "rows": rows, "marks": marks,
        "summary": " ".join(notes),
    }
