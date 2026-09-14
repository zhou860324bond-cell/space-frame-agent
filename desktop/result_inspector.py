"""结果探针、极值定位与截面正应力图所需的纯计算。"""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

import numpy as np


FORCES = {"N", "Vy", "Vz", "V"}
MOMENTS = {"T", "My", "Mz", "M"}


def solid_result_provenance(payload: dict) -> dict:
    """把实体摘要压成同页可读的来源证据，不暴露原始 JSON。"""
    backend = str(payload.get("backend") or "未标明后端")
    origin = {
        "computed": "本次计算",
        "disk": "磁盘恢复",
        "memory": "会话内复用",
    }.get(payload.get("_result_origin"), "历史结果")

    raw_time = payload.get("generated_at")
    try:
        generated = datetime.fromisoformat(str(raw_time)).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        generated = "时间未记录"

    identity = payload.get("cache_identity")
    if isinstance(identity, dict):
        model_hash = str(identity.get("model_digest") or "")
        joint_hash = str(identity.get("joint_input_digest") or "")
        fingerprint = (f"模型 {model_hash[:10]}… / 节点 {joint_hash[:10]}…"
                       if model_hash and joint_hash else "指纹不完整")
        fingerprint_tip = (f"模型 SHA-256：{model_hash}\n"
                           f"节点输入 SHA-256：{joint_hash}")
        plan = identity.get("mesh_plan_mm") or []
    else:
        fingerprint = "旧版结果，无输入指纹"
        fingerprint_tip = "该摘要生成于输入指纹功能之前；不会用于跨重启自动续跑。"
        plan = []
    try:
        mesh_text = (" / ".join(f"{float(value):g}" for value in plan) + " mm"
                     if isinstance(plan, list) and plan else "网格计划未记录")
        plan_ok = isinstance(plan, list) and bool(plan)
    except (TypeError, ValueError):
        mesh_text, plan_ok = "网格计划损坏", False

    files = payload.get("files") or {}
    is_abaqus = "abaqus" in backend.lower()
    artifact_key = "finest_odb" if is_abaqus else "finest_vtu"
    artifact_kind = "ODB" if is_abaqus else "VTU"
    raw_path = files.get(artifact_key) if isinstance(files, dict) else None
    path = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
    try:
        size = path.stat().st_size if path is not None and path.is_file() else 0
    except OSError:
        size = 0
    if size:
        if size >= 1024 ** 2:
            size_text = f"{size / 1024 ** 2:.1f} MB"
        elif size >= 1024:
            size_text = f"{size / 1024:.1f} KB"
        else:
            size_text = f"{size} B"
        artifact = f"{artifact_kind} 完整 · {size_text}"
    else:
        artifact = f"{artifact_kind} 缺失或为空"
    converged = payload.get("hot_spot_all_converged") is True
    hashes_ok = all(len(value) == 64
                    and all(char in "0123456789abcdef" for char in value.lower())
                    for value in (model_hash, joint_hash)) if isinstance(identity, dict) else False
    identity_ok = (isinstance(identity, dict)
                   and identity.get("schema") == "solid-joint-cache/v1"
                   and hashes_ok and plan_ok)
    traceable = identity_ok and size > 0
    if traceable and converged:
        status, severity = "证据链完整", "pass"
    elif traceable:
        status, severity = "结果可追溯，热点未通过", "unclear"
    else:
        status, severity = "溯源信息不完整", "unclear"
    return {
        "source": f"{origin} · {backend}",
        "generated": generated,
        "fingerprint": fingerprint,
        "mesh": mesh_text,
        "artifact": artifact,
        "status": status,
        "severity": severity,
        "tooltips": {
            "source": "说明当前界面结果是刚计算、从磁盘恢复，还是会话内复用。",
            "generated": str(raw_time or "摘要没有记录生成时间"),
            "fingerprint": fingerprint_tip,
            "mesh": "提交给当前后端的全局目标网格尺寸，由粗到细排列。",
            "artifact": str(path) if path is not None else "摘要没有记录主结果文件路径。",
        },
    }


def _solid_target_prefix(payload: dict, include_target: bool) -> str:
    if not include_target:
        return ""
    node = payload.get("node_id", "?")
    case = str(payload.get("case") or "未命名工况")
    return f"节点 {node} / {case} · "


def solid_history_label(payload: dict, *, include_target: bool = False) -> str:
    """历史选择框的一行摘要：时间、后端、Kt，避免暴露文件名噪声。"""
    raw_time = str(payload.get("generated_at") or "时间未记录")
    try:
        stamp = datetime.fromisoformat(raw_time).astimezone().strftime(
            "%m-%d %H:%M:%S")
    except ValueError:
        stamp = raw_time[:19]
    kt = payload.get("stress_concentration_factor")
    kt_text = f"Kt {float(kt):.4g}" if isinstance(kt, (int, float)) else "Kt 未发布"
    return (f"{_solid_target_prefix(payload, include_target)}{stamp} · "
            f"{payload.get('backend') or '未知后端'} · {kt_text}")


def solid_orphan_label(payload: dict, *, include_target: bool = False) -> str:
    """不完整运行在清理框中的短标签，不伪装成可比较的正式结果。"""
    raw_time = str(payload.get("generated_at") or "时间未记录")
    try:
        stamp = datetime.fromisoformat(raw_time).astimezone().strftime(
            "%m-%d %H:%M:%S")
    except ValueError:
        stamp = raw_time[:19]
    return (f"{_solid_target_prefix(payload, include_target)}{stamp} · "
            f"{payload.get('backend') or '未知后端'} · "
            f"不完整：{payload.get('incomplete_stage') or '阶段未知'}")


def format_storage_size(size: int) -> str:
    value = max(0, int(size))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "0 B"


def build_solid_history_management(history: list[dict],
                                   protected_ids: set[str],
                                   orphans: list[dict] | None = None, *,
                                   include_target: bool = False) -> dict:
    """生成同页历史占用表，并给出通过底层 latest 门禁的清理候选。"""
    from solid_cache import ORPHAN_MIN_AGE_SECONDS, solid_history_storage

    rows = []
    marks = []
    removable_ids: set[str] = set()
    total_bytes = 0
    removable_bytes = 0
    for payload in history:
        identifier = str(payload.get("_history_id") or "")
        storage = solid_history_storage(payload)
        size = int(storage.get("bytes") or 0)
        total_bytes += size
        reasons = []
        if identifier in protected_ids:
            reasons.append("当前对比版本")
        if storage.get("latest") is True:
            reasons.append("后端最新成功结果")
        if not storage.get("valid") or storage.get("latest") is None:
            reasons.append(str(storage.get("reason") or "目录不可核验"))
        removable = bool(identifier and not reasons)
        if removable:
            removable_ids.add(identifier)
            removable_bytes += size
            state = "可移入回收站"
            mark = None
        else:
            state = "受保护：" + "；".join(reasons)
            mark = "pass" if storage.get("latest") is True else "unclear"
        identity = payload.get("cache_identity") or {}
        plan = identity.get("mesh_plan_mm") if isinstance(identity, dict) else None
        try:
            mesh = (" / ".join(f"{float(value):g}" for value in plan) + " mm"
                    if isinstance(plan, list) and plan else "未记录")
        except (TypeError, ValueError):
            mesh = "记录损坏"
        kt = payload.get("stress_concentration_factor")
        kt_text = f"{float(kt):.4g}" if isinstance(kt, (int, float)) else "未发布"
        rows.append([
            solid_history_label(payload, include_target=include_target),
            payload.get("backend") or "未知后端",
            mesh, kt_text, solid_result_provenance(payload)["artifact"],
            format_storage_size(size), int(storage.get("files") or 0), state,
        ])
        marks.append(mark)
    incomplete = list(orphans or [])
    for payload in incomplete:
        identifier = str(payload.get("_history_id") or "")
        size = int(payload.get("bytes") or 0)
        total_bytes += size
        eligible = payload.get("cleanup_eligible") is True
        if payload.get("latest") is True:
            state = "受保护：被后端 latest 引用"
        elif eligible:
            hours = int(ORPHAN_MIN_AGE_SECONDS / 3600)
            state = f"可清理：不完整运行已超过 {hours} 小时"
            removable_ids.add(identifier)
            removable_bytes += size
        else:
            age = max(0.0, float(payload.get("age_seconds") or 0.0))
            remaining = max(
                1, math.ceil((ORPHAN_MIN_AGE_SECONDS - age) / 3600.0))
            state = f"受保护：最近仍有写入，约 {remaining} 小时后可清理"
        rows.append([
            solid_orphan_label(payload, include_target=include_target),
            f"{payload.get('backend') or '未知后端'}（不完整）",
            "未形成有效摘要", "未发布",
            payload.get("incomplete_stage") or "阶段未知",
            format_storage_size(size), int(payload.get("files") or 0), state,
        ])
        marks.append("unclear")
    return {
        "label": "实体运行历史与磁盘占用",
        "summary": (
            f"共 {len(rows)} 个运行目录，其中 {len(incomplete)} 个不完整，"
            f"占用 {format_storage_size(total_bytes)}；"
            f"当前可清理 {len(removable_ids)} 次，"
            f"约 {format_storage_size(removable_bytes)}。删除操作移入 Windows 回收站。"),
        "columns": ["运行", "后端", "提交网格", "正式 Kt", "主结果文件",
                    "目录占用", "文件数", "清理状态"],
        "rows": rows, "marks": marks, "removable_ids": removable_ids,
        "total_bytes": total_bytes, "removable_bytes": removable_bytes,
    }


def solid_storage_provenance(records: dict, management: dict) -> dict:
    """把全局扫描范围明确写在来源卡上，避免伪装成某一次求解结果。"""
    targets = records.get("targets") or []
    history = records.get("history") or []
    orphans = records.get("orphans") or []
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    target_text = f"{len(targets)} 个节点 / 工况目录"
    size_text = format_storage_size(int(management.get("total_bytes") or 0))
    return {
        "source": "全局实体存储扫描",
        "generated": generated,
        "fingerprint": target_text,
        "mesh": f"{len(history)} 个正式结果 / {len(orphans)} 个不完整运行",
        "artifact": f"{len(history) + len(orphans)} 个目录 · {size_text}",
        "status": "存储索引已刷新",
        "severity": "pass" if not orphans else "unclear",
        "tooltips": {
            "source": "扫描 native 与 Abaqus 两个受管理结果根目录，不依赖当前模型选择。",
            "generated": "本次只读扫描完成时间；没有启动求解器。",
            "fingerprint": "按 node_<编号>_<工况> 直接子目录识别，符号链接与其它目录会跳过。",
            "mesh": "正式摘要和未形成有效摘要的运行严格分开统计。",
            "artifact": "受管理运行目录的合计数量与磁盘占用。",
        },
    }


def compare_solid_history(left: dict, right: dict) -> dict:
    """比较两次实体运行；输入不同时明确阻止求解器归因。"""
    left_identity = left.get("cache_identity") or {}
    right_identity = right.get("cache_identity") or {}

    def short(identity: dict, key: str) -> str:
        value = str(identity.get(key) or "")
        return value[:12] + "…" if value else "未记录"

    def mesh(identity: dict) -> str:
        values = identity.get("mesh_plan_mm") or []
        try:
            return " / ".join(f"{float(value):g}" for value in values) + " mm"
        except (TypeError, ValueError):
            return "记录损坏"

    def hotspot(payload: dict) -> float | None:
        governing = payload.get("governing_arm")
        item = (payload.get("hot_spot_extrapolation") or {}).get(str(governing), {})
        value = item.get("hot_spot_mpa") if isinstance(item, dict) else None
        if isinstance(value, (int, float)):
            return float(value)
        diagnosis = (payload.get("hot_spot_convergence") or {}).get(
            str(governing), {})
        values = diagnosis.get("values") if isinstance(diagnosis, dict) else []
        final = values[-1].get("hot_spot_mpa") if values else None
        return float(final) if isinstance(final, (int, float)) else None

    def difference(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return "数据不足"
        scale = max(abs(a), abs(b), 1e-12)
        return f"{100.0 * abs(a - b) / scale:.2f}%"

    model_same = (left_identity.get("model_digest")
                  and left_identity.get("model_digest") ==
                  right_identity.get("model_digest"))
    joint_same = (left_identity.get("joint_input_digest")
                  and left_identity.get("joint_input_digest") ==
                  right_identity.get("joint_input_digest"))
    plan_same = (left_identity.get("mesh_plan_mm")
                 and left_identity.get("mesh_plan_mm") ==
                 right_identity.get("mesh_plan_mm"))
    left_hotspot, right_hotspot = hotspot(left), hotspot(right)
    left_kt = left.get("stress_concentration_factor")
    right_kt = right.get("stress_concentration_factor")
    left_kt = float(left_kt) if isinstance(left_kt, (int, float)) else None
    right_kt = float(right_kt) if isinstance(right_kt, (int, float)) else None
    rows = [
        ["运行", solid_history_label(left), solid_history_label(right), "—"],
        ["模型输入", short(left_identity, "model_digest"),
         short(right_identity, "model_digest"), "一致" if model_same else "不同"],
        ["节点六分量输入", short(left_identity, "joint_input_digest"),
         short(right_identity, "joint_input_digest"), "一致" if joint_same else "不同"],
        ["提交网格", mesh(left_identity), mesh(right_identity),
         "一致" if plan_same else "不同"],
        ["控制热点应力 (MPa)", left_hotspot, right_hotspot,
         difference(left_hotspot, right_hotspot)],
        ["正式 Kt", left_kt, right_kt, difference(left_kt, right_kt)],
        ["主结果文件", solid_result_provenance(left)["artifact"],
         solid_result_provenance(right)["artifact"], "仅核验存在与大小"],
    ]
    comparable = bool(model_same and joint_same)
    marks = [None, "pass" if model_same else "unclear",
             "pass" if joint_same else "unclear",
             "pass" if plan_same else "unclear", None, None, None]
    return {
        "label": "两次实体运行对比",
        "summary": ("模型与节点六分量输入一致，可审查网格、热点应力和 Kt 变化。"
                    if comparable else
                    "输入已经变化；响应差异只能作为版本记录，不能归因于求解器或网格。"),
        "columns": ["对比项", "版本 A", "版本 B", "差异 / 结论"],
        "rows": rows,
        "marks": marks,
    }


def assess_result_trust(session, display_stations: int = 41) -> dict:
    """汇总已求解模型的基础可信度检查，不把显示采样冒充分析网格。"""
    from frame3d import check_equilibrium
    from model_io import validate_payload

    rows: list[list[str]] = []
    marks: list[str] = []
    errors = validate_payload(session.model)
    if errors:
        rows.append(["模型合法性", "未通过", f"发现 {len(errors)} 个模型问题"])
        marks.append("fail")
    else:
        rows.append(["模型合法性", "通过", "材料、截面、连接、边界和荷载引用有效"])
        marks.append("pass")

    solution = session.solution
    frame = session.frame
    if solution is None or frame is None:
        rows.append(["求解状态", "无结果", "当前模型尚未形成可检查的求解结果"])
        marks.append("unclear")
        return {"label": "尚未求解", "severity": "idle",
                "summary": "需要先完成求解", "rows": rows, "marks": marks,
                "physical_members": len(session.model.get("members") or []),
                "analysis_elements": 0, "display_stations": display_stations}

    equilibrium_failed = 0
    for case in solution.all_results():
        check = check_equilibrium(frame, solution, case)
        ok = bool(check["ok"])
        equilibrium_failed += int(not ok)
        rows.append([
            f"静力平衡：{case}", "通过" if ok else "未通过",
            f"归一化残差 {float(check['relative']):.3e}",
        ])
        marks.append("pass" if ok else "fail")

    mapping = getattr(getattr(session, "compilation", None), "mapping", None)
    physical = (len(mapping.physical_to_elements) if mapping is not None
                else len(session.model.get("members") or []))
    analysis = (len(mapping.element_to_physical) if mapping is not None
                else len(frame.members))
    generated = (len(mapping.generated_nodes) if mapping is not None else 0)
    rows.append([
        "分析离散", "已记录",
        f"{physical} 根物理杆件编译为 {analysis} 个分析单元，"
        f"自动生成 {generated} 个切分节点",
    ])
    marks.append("pass")
    rows.append([
        "显示细分", "仅影响显示",
        f"每根杆件采用 {display_stations} 个显示点；不改变刚度矩阵、"
        "分析单元或求解结果",
    ])
    marks.append("pass")

    analysis_type = str(solution.analysis.get("type", "linear_static"))
    if analysis_type == "linear_static":
        rows.append([
            "网格收敛适用性", "无需用显示细分判断",
            "线性等截面梁采用解析单元刚度和跨内结果恢复；显示点加密不是分析网格收敛。"
            "Abaqus B31、实体应力和非线性分析应另做网格细化。",
        ])
        marks.append("pass")
    else:
        rows.append([
            "网格收敛适用性", "需要专项复核",
            "当前不是线性静力结果，应结合增量收敛记录和分析网格细化单独判断。",
        ])
        marks.append("unclear")

    failed = bool(errors) or equilibrium_failed > 0
    review = analysis_type != "linear_static"
    if failed:
        label, severity = "基础检查未通过", "fail"
    elif review:
        label, severity = "需要专项复核", "unclear"
    else:
        label, severity = "基础检查通过", "pass"
    return {
        "label": label, "severity": severity,
        "summary": ("模型合法性和静力平衡已检查；该结论不代替规范验算、"
                    "外部对标或实体热点收敛检查。"),
        "rows": rows, "marks": marks,
        "physical_members": physical, "analysis_elements": analysis,
        "generated_nodes": generated, "display_stations": display_stations,
    }


def assess_solid_result(payload: dict) -> dict:
    """审查实体节点结果的发布证据，不沿用整体梁结果的绿灯。"""
    meshes = payload.get("meshes") or []
    convergence = payload.get("hot_spot_convergence") or {}
    governing = payload.get("governing_arm")
    kt = payload.get("stress_concentration_factor")
    rows: list[list[str]] = [[
        "实体求解记录", "已完成" if meshes else "无记录",
        f"{payload.get('backend') or '未标明后端'}；节点 {payload.get('node_id')}；"
        f"工况 {payload.get('case') or '未标明'}；共 {len(meshes)} 档网格",
    ]]
    marks: list[str] = ["pass" if meshes else "fail"]

    converged_arms: list[str] = []
    for arm, diagnosis in convergence.items():
        verdict = str(diagnosis.get("verdict") or "inconclusive")
        if verdict == "converging":
            status, mark = "热点已稳定", "pass"
            converged_arms.append(str(arm))
        elif verdict == "diverging":
            status, mark = "热点发散", "fail"
        else:
            status, mark = "证据不足", "unclear"
        rows.append([
            f"候选杆臂 {arm}", status,
            str(diagnosis.get("reason") or "没有可用的热点收敛说明"),
        ])
        marks.append(mark)

    governing_key = None if governing is None else str(governing)
    contract_ok = (
        isinstance(kt, (int, float))
        and governing_key in converged_arms
        and payload.get("hot_spot_all_converged") is True
    )
    if isinstance(kt, (int, float)):
        gate_status = "允许发布" if contract_ok else "发布契约异常"
        gate_mark = "pass" if contract_ok else "fail"
        gate_detail = (
            f"Kt={float(kt):.3g}；控制杆臂 {governing_key}；"
            "所有候选杆臂均通过热点稳定门禁。"
            if contract_ok else
            "结果包含 Kt，但没有完整的候选杆臂收敛证据；不得作为正式结果。"
        )
    else:
        gate_status, gate_mark = "未发布", "unclear"
        gate_detail = str(payload.get("stress_concentration_refused") or
                          "没有满足正式 Kt 所需的完整热点收敛证据。")
    rows.append(["Kt 发布门禁", gate_status, gate_detail])
    marks.append(gate_mark)

    peak = payload.get("peak_convergence") or {}
    rows.append([
        "相贯线峰值", "仅作诊断",
        (str(peak.get("reason") or "未形成峰值趋势结论") +
         "；最大值和 P99 不作为 Kt 分子。"),
    ])
    # 无倒圆相贯线的峰值发散是预期诊断，不应把一个合格的热点 Kt 染成红色；
    # 但它也不是一项“通过”，保持琥珀色的仅供诊断状态更准确。
    marks.append("unclear")

    if "fail" in marks:
        label, severity = "实体证据存在异常", "fail"
    elif contract_ok:
        label, severity = "Kt 证据已通过", "pass"
    else:
        label, severity = "实体结果仅供诊断", "unclear"
    return {
        "label": label, "severity": severity,
        "summary": ("实体结果按候选杆臂逐项审查；相贯线奇异峰值与 IIW "
                    "0.4t/1.0t 热点应力分开评价。"),
        "rows": rows, "marks": marks,
    }


def _display_system(frame):
    from units import of
    return of(frame)


def point_on_member(frame, member_id: int, x: float) -> np.ndarray:
    from frame3d import member_endpoints

    member = frame.members[member_id]
    pi, pj = member_endpoints(frame, member)
    length = float(np.linalg.norm(pj - pi))
    ratio = 0.0 if length == 0.0 else float(np.clip(x / length, 0.0, 1.0))
    return pi + ratio * (pj - pi)


def projected_station(frame, member_id: int, point) -> float:
    """将拾取点投影到杆件中心线，返回距 i 端位置。"""
    from frame3d import member_endpoints

    member = frame.members[member_id]
    pi, pj = member_endpoints(frame, member)
    axis = pj - pi
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        return 0.0
    return float(np.clip(np.dot(np.asarray(point) - pi, axis) / length,
                         0.0, length))


def section_stress_grid(section, N: float, My: float, Mz: float,
                        samples: int = 41) -> dict:
    """截面包络内的线性正应力场；受拉为正。"""
    from stress import StressUnavailable

    if section.cy is None or section.cz is None:
        raise StressUnavailable(
            f"截面 {section.name!r} 缺少 cy/cz，无法绘制截面正应力分布。")
    y = np.linspace(-section.cy, section.cy, samples)
    z = np.linspace(-section.cz, section.cz, samples)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    sigma = N / section.A - Mz * yy / section.Iz + My * zz / section.Iy
    mask = ((yy / section.cy) ** 2 + (zz / section.cz) ** 2 <= 1.0
            if section.circular else np.ones_like(sigma, dtype=bool))
    return {"y": y, "z": z, "sigma": np.where(mask, sigma, np.nan),
            "circular": bool(section.circular)}


def probe_member(frame, solution, member_id: int, point,
                 case: str | None = None) -> dict:
    """读取任意杆件截面的六内力、全局/局部位移与极端纤维正应力。"""
    from frame3d import local_axes, member_endpoints
    from internal_forces import member_diagram, member_displacement
    from stress import StressUnavailable, extreme_normal_stress

    name = case or solution.primary
    member = frame.members[member_id]
    pi, pj = member_endpoints(frame, member)
    length, rotation = local_axes(pi, pj, member.ref_vector)
    x = projected_station(frame, member_id, point)
    diagram = member_diagram(frame, solution, member_id, name,
                             at_x=np.asarray([x]))
    raw = {component: float(diagram.component(component)[0])
           for component in ("N", "Vy", "Vz", "T", "My", "Mz")}
    stations, displacement = member_displacement(
        frame, solution, member_id, name, stations=401)
    global_u = np.array([np.interp(x, stations, displacement[:, axis])
                         for axis in range(3)])
    local_u = rotation @ global_u
    system = _display_system(frame)
    values = {
        component: raw[component] * (system.moment_scale
                                     if component in MOMENTS
                                     else system.force_scale)
        for component in raw
    }
    stress = None
    stress_error = None
    section = frame.sections[member.section]
    try:
        low, high = extreme_normal_stress(
            section, raw["N"], raw["My"], raw["Mz"])
        stress_scale = 1e-6 if system.name == "N-m-Pa" else 1.0
        stress = {"min": float(low) * stress_scale,
                  "max": float(high) * stress_scale, "unit": "MPa",
                  "grid": section_stress_grid(
                      section, raw["N"], raw["My"], raw["Mz"])}
        stress["grid"]["sigma"] *= stress_scale
    except StressUnavailable as exc:
        stress_error = str(exc)
    return {
        "member": member_id, "case": name, "section": section.name,
        "i": member.i, "j": member.j, "x": x,
        "length": length, "point": point_on_member(frame, member_id, x),
        "forces": values, "force_unit": system.force_unit,
        "moment_unit": system.moment_unit,
        "global_displacement": global_u * system.disp_scale,
        "local_displacement": local_u * system.disp_scale,
        "displacement_unit": system.disp_unit,
        "stress": stress, "stress_error": stress_error,
    }


def global_extreme(frame, solution, component: str,
                   case: str | None = None, stations: int = 101) -> dict:
    """全结构指定分量的绝对极值及其中心线位置。"""
    from internal_forces import member_diagram

    import numpy as _np

    from .scene import STRESS, member_scalar

    name = case or solution.primary
    best = None
    for member_id in sorted(frame.members):
        diagram = member_diagram(frame, solution, member_id, name,
                                 stations=stations)
        if component == STRESS:
            # σ 不是内力分量，diagram.extreme 认不出它；按同一条标量口径
            # （scene.member_scalar）自己取极值，免得云图和标注各算各的。
            values = member_scalar(frame, frame.members[member_id],
                                   diagram, STRESS)
            k = int(_np.argmax(_np.abs(values)))
            x, value = float(diagram.x[k]), float(values[k])
        else:
            x, value = diagram.extreme(component)
        if best is None or abs(value) > abs(best["raw_value"]):
            best = {"member": member_id, "x": x, "raw_value": value}
    if best is None:
        return {"member": None, "x": 0.0, "value": 0.0, "unit": ""}
    system = _display_system(frame)
    if component == STRESS:
        scale, unit = system.stress_scale, system.stress_unit
    elif component in MOMENTS:
        scale, unit = system.moment_scale, system.moment_unit
    else:
        scale, unit = system.force_scale, system.force_unit
    best.update({"component": component, "case": name,
                 "value": best.pop("raw_value") * scale, "unit": unit,
                 "point": point_on_member(frame, best["member"], best["x"])})
    return best


def show_stress_dialog(parent, probe: dict) -> None:
    """显示探针截面的线性正应力分布；剪应力不在当前截面契约内。"""
    from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    stress = probe.get("stress")
    if stress is None:
        raise ValueError(probe.get("stress_error") or "该截面无法计算正应力。")
    dialog = QDialog(parent)
    dialog.setWindowTitle(
        f"截面正应力 — 杆件 {probe['member']}，x={probe['x']:.3g}")
    dialog.resize(560, 480)
    box = QVBoxLayout(dialog)
    note = QLabel(
        f"截面 {probe['section']}　σmin={stress['min']:+.3f} MPa　"
        f"σmax={stress['max']:+.3f} MPa\n"
        "轴力 + 双向弯曲线性正应力，受拉为正；不含剪力和扭转剪应力。")
    note.setWordWrap(True)
    box.addWidget(note)
    figure = Figure(figsize=(5.2, 3.8))
    canvas = FigureCanvasQTAgg(figure)
    box.addWidget(canvas, 1)
    axis = figure.add_subplot(111)
    grid = stress["grid"]
    image = axis.pcolormesh(grid["z"], grid["y"], grid["sigma"],
                            shading="auto", cmap="coolwarm")
    axis.set_aspect("equal")
    axis.set_xlabel("local z")
    axis.set_ylabel("local y")
    axis.set_title("Normal stress sigma (MPa)")
    figure.colorbar(image, ax=axis, label="MPa")
    figure.tight_layout()
    dialog.exec()
