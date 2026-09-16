"""静默失败检测 — 求解跑完了，但结果可能是错的。

工程仿真里最危险的不是报错，而是「作业正常结束、数值看起来合理、
但模型本身有问题」。这个模块在每次求解后自动扫描 8 类常见静默失败，
每类给出严重度、检测结果、具体数值和修复建议。

用法：
    from silent_failures import detect_silent_failures

    findings = detect_silent_failures(model, sol)
    for f in findings:
        if f["status"] != "pass":
            print(f"[{f['severity']}] {f['name']}: {f['message']}")
            print(f"  建议: {f['suggestion']}")

也可以只跑某几项：
    findings = detect_silent_failures(model, sol, only=["excessive_displacement", "zero_reaction_with_load"])

检测项清单见 docs/SILENT_FAILURES.md。
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from frame3d import Frame, Solution

# ---------------------------------------------------------------------------
# 检测结果数据结构
# ---------------------------------------------------------------------------

SEVERITY_CRITICAL = "critical"   # 结果几乎肯定是错的，必须修
SEVERITY_WARNING = "warning"     # 可能有问题，建议检查
SEVERITY_INFO = "info"           # 提示性信息，不一定是问题

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"


def _finding(
    check_id: str,
    name: str,
    severity: str,
    status: str,
    message: str,
    detail: dict[str, Any] | None = None,
    suggestion: str = "",
) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "severity": severity,
        "status": status,
        "message": message,
        "detail": detail or {},
        "suggestion": suggestion,
    }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _structure_dimension(model: Frame) -> float:
    """结构特征尺寸 = 节点坐标的最大范围（对角线长度的近似）。"""
    if not model.nodes:
        return 1.0
    coords = np.array([n.xyz for n in model.nodes.values()])
    ranges = coords.max(axis=0) - coords.min(axis=0)
    return float(np.linalg.norm(ranges))


def _total_applied_load(model: Frame, case: str) -> np.ndarray:
    """某个工况的外载荷合力（平动 3 分量，全局坐标）。"""
    total = np.zeros(3)
    lc = model.case(case)
    for nid, f in lc.nodal_loads.items():
        total += np.array(f[:3])
    # 均布载荷合力 = 线荷载 * 杆长
    for mid, w in lc.member_loads.items():
        m = model.members[mid]
        length = float(np.linalg.norm(model.nodes[m.j].xyz - model.nodes[m.i].xyz))
        total += np.array(w[:3]) * length
    return total


def _total_reaction(model: Frame, sol: Solution, case: str) -> np.ndarray:
    """某个工况的支座反力合力（平动 3 分量）。"""
    total = np.zeros(3)
    result = sol[case]
    for nid in model.supports:
        dofs = model.node_dofs(nid)
        total += result.R[dofs[:3]]
    return total


def _count_support_dofs(model: Frame) -> int:
    """统计支座约束的自由度数（1=约束，0=自由，求和）。"""
    total = 0
    for sup in model.supports.values():
        total += sum(int(x) for x in sup)
    return total


# ---------------------------------------------------------------------------
# 检测项 1: 位移过大
# ---------------------------------------------------------------------------

def check_excessive_displacement(model: Frame, sol: Solution) -> dict[str, Any]:
    """最大位移超过结构特征尺寸的 10% — 通常意味着约束不足或单位错误。"""
    dim = _structure_dimension(model)
    threshold = dim * 0.10  # 10%

    worst_case, worst_disp, worst_node = None, 0.0, None
    for name, result in sol.all_results().items():
        for nid in model.order():
            dofs = model.node_dofs(nid)
            mag = float(np.linalg.norm(result.U[dofs[:3]]))
            if mag > worst_disp:
                worst_disp, worst_node, worst_case = mag, nid, name

    if worst_disp > threshold:
        ratio = worst_disp / dim * 100
        return _finding(
            "excessive_displacement",
            "位移过大",
            SEVERITY_CRITICAL,
            STATUS_FAIL,
            f"工况 {worst_case} 节点 {worst_node} 位移 {worst_disp*1000:.2f} mm，"
            f"达结构尺寸 {dim*1000:.0f} mm 的 {ratio:.1f}%（阈值 10%）",
            {"max_displacement_m": worst_disp, "structure_dimension_m": dim,
             "ratio_pct": ratio, "node": worst_node, "case": worst_case},
            "检查支座约束是否完整（6个刚体自由度是否都被限制）、单位是否一致（mm vs m）、"
            "载荷量级是否合理。大位移通常意味着结构处于机构运动状态而非弹性变形。",
        )
    return _finding(
        "excessive_displacement", "位移过大", SEVERITY_CRITICAL, STATUS_PASS,
        f"最大位移 {worst_disp*1000:.4f} mm，结构尺寸 {dim*1000:.0f} mm，"
        f"占比 {worst_disp/dim*100:.3f}%（< 10%）",
    )


# ---------------------------------------------------------------------------
# 检测项 2: 有载荷但反力为零
# ---------------------------------------------------------------------------

def check_zero_reaction_with_load(model: Frame, sol: Solution) -> dict[str, Any]:
    """有外载荷但所有支座反力为零 — 约束完全失效，载荷没有传到基础。"""
    bad_cases = []
    for name in sol.all_results():
        applied = _total_applied_load(model, name)
        reaction = _total_reaction(model, sol, name)
        applied_mag = float(np.linalg.norm(applied))
        reaction_mag = float(np.linalg.norm(reaction))
        if applied_mag > 1e-10 and reaction_mag < applied_mag * 1e-6:
            bad_cases.append({
                "case": name,
                "applied_load_N": applied_mag,
                "reaction_force_N": reaction_mag,
            })

    if bad_cases:
        return _finding(
            "zero_reaction_with_load",
            "有载荷但反力为零",
            SEVERITY_CRITICAL,
            STATUS_FAIL,
            f"{len(bad_cases)} 个工况有外载荷但支座反力几乎为零："
            + ", ".join(f"{b['case']}({b['applied_load_N']:.1f}N)" for b in bad_cases),
            {"cases": bad_cases},
            "支座约束可能全部为 0（自由端），或约束方向定义错误。"
            "检查 model.supports 中每个支座的 6 元组（1=约束，0=自由），"
            "确保至少有一个节点的 3 个平动自由度被约束。",
        )
    return _finding(
        "zero_reaction_with_load", "有载荷但反力为零", SEVERITY_CRITICAL, STATUS_PASS,
        "所有有载荷的工况都有非零支座反力",
    )


# ---------------------------------------------------------------------------
# 检测项 3: 有载荷但内力为零
# ---------------------------------------------------------------------------

def check_zero_internal_force(model: Frame, sol: Solution) -> dict[str, Any]:
    """有外载荷但所有杆件内力为零 — 载荷没有传递到杆件，可能是机构或载荷作用点错误。"""
    bad_cases = []
    for name, result in sol.all_results().items():
        applied = _total_applied_load(model, name)
        if float(np.linalg.norm(applied)) < 1e-10:
            continue
        max_force = 0.0
        for f in result.member_forces.values():
            max_force = max(max_force, float(np.abs(f).max()))
        if max_force < 1e-10:
            bad_cases.append({"case": name, "applied_load_N": float(np.linalg.norm(applied))})

    if bad_cases:
        return _finding(
            "zero_internal_force",
            "有载荷但内力为零",
            SEVERITY_CRITICAL,
            STATUS_FAIL,
            f"{len(bad_cases)} 个工况有外载荷但所有杆件内力为零："
            + ", ".join(b["case"] for b in bad_cases),
            {"cases": bad_cases},
            "结构可能是机构（可刚体运动），或载荷作用在自由节点上未与杆件连接。"
            "检查节点是否都被至少一根杆件连接、载荷节点 ID 是否正确。",
        )
    return _finding(
        "zero_internal_force", "有载荷但内力为零", SEVERITY_CRITICAL, STATUS_PASS,
        "所有有载荷的工况都有非零杆件内力",
    )


# ---------------------------------------------------------------------------
# 检测项 4: 刚度矩阵接近奇异
# ---------------------------------------------------------------------------

def check_no_applied_load(model: Frame, sol: Solution) -> dict[str, Any]:
    """一个荷载都没有，却照样求出了一套（全零的）结果。

    **这是 zero_reaction_with_load / zero_internal_force 两项的盲区。**
    那两条的前提都是"有外载荷"——没有载荷时它们直接跳过，于是最基本的一种
    静默失败反而没人管：求解返回 ok=True，内力弯矩全零，界面上看不出异常。

    真实的触发路径至少有两条，都不罕见：

    * `set_load_cases` 因为杆件号写错而**失败**（它会如实报错并保持原模型
      不变），但调用方没看返回值就接着 solve；
    * 建了工况却没往里放荷载——空工况是被接受的。

    全零结果"肉眼看就是错的"，但只有人看才看得出来。这条检测就是替人看。
    """
    loaded = []
    for name in sol.all_results():
        magnitude = float(np.linalg.norm(_total_applied_load(model, name)))
        if magnitude > 1e-10:
            loaded.append(name)

    if loaded:
        return _finding(
            "no_applied_load", "求解时没有任何荷载", SEVERITY_CRITICAL, STATUS_PASS,
            f"{len(loaded)} 个工况有非零外载荷",
            {"loaded_cases": loaded},
        )
    return _finding(
        "no_applied_load",
        "求解时没有任何荷载",
        SEVERITY_CRITICAL,
        STATUS_FAIL,
        f"全部 {len(sol.all_results())} 个工况的外载荷合力都是零——"
        "这次求解的结果必然是全零位移、全零内力，不代表任何受力状态",
        {"cases": sorted(sol.all_results())},
        "检查两件事：一是 set_load_cases / set_member_load 是否返回了 ok=False"
        "（杆件号写错时它会报错并保持原模型不变，此时荷载根本没写进去）；"
        "二是工况里是不是只有名字、没有荷载条目。",
    )


def check_near_singular_stiffness(model: Frame, sol: Solution) -> dict[str, Any]:
    """刚度矩阵条件数过大 — 接近奇异，可能有机构位移模式或数值不稳定。

    **必须算施加边界条件【之后】的矩阵。** `sol.K` 是原始总刚，里面还留着
    没被约束住的刚体位移模式——空间问题恒有 6 个，所以它**按定义就奇异**，
    条件数恒在 1e17~1e19。

    原先直接拿 sol.K 算，后果是这一项对**任何模型**都报 critical：一个柱底
    固接的门式刚架和一个真正的机构给出同一个数。一个 100% 误报的 critical
    比没有更糟——它让人学会忽略整块告警，真信号跟着一起被埋掉。
    （实测：正常门式刚架 24×24、秩 18、亏秩正好 6，条件数 1.21e+19。）

    取自由自由度的子矩阵，口径和 frame3d.solve 里那一步完全一致。
    """
    from frame3d import constrained_dofs

    full = sol.K
    fixed = constrained_dofs(model)
    free = np.setdiff1d(np.arange(full.shape[0]), fixed)
    if free.size == 0:
        return _finding(
            "near_singular_stiffness", "刚度矩阵接近奇异", SEVERITY_WARNING,
            STATUS_WARN, "所有自由度都被约束，没有可解的方程",
            suggestion="检查是否把整个结构都固定住了。")
    K = full[free][:, free]
    n = K.shape[0]

    # 小矩阵转稠密算精确条件数；大矩阵用范数估计
    if n <= 2000:
        K_dense = K.toarray()
        try:
            cond = float(np.linalg.cond(K_dense))
        except np.linalg.LinAlgError:
            cond = float("inf")
    else:
        # 大矩阵：用 1-范数和无穷范数的乘积估计（上界）
        try:
            from scipy.sparse.linalg import onenormest
            cond = float(onenormest(K) * onenormest(sp.linalg.inv(K.tocsc())))
        except Exception:  # noqa: BLE001  条件数估不出来就记 -1.0，下游 `if cond < 0` 会出一条 finding
            cond = -1.0  # 无法估计

    if cond < 0:
        return _finding(
            "near_singular_stiffness", "刚度矩阵接近奇异", SEVERITY_WARNING, STATUS_WARN,
            "无法估计刚度矩阵条件数（矩阵过大或求解失败）",
            suggestion="手动检查是否有未约束的刚体自由度，或减小模型规模后重试。",
        )

    if cond > 1e12:
        return _finding(
            "near_singular_stiffness",
            "刚度矩阵接近奇异",
            SEVERITY_CRITICAL,
            STATUS_FAIL,
            f"刚度矩阵条件数 {cond:.2e}（> 1e12），数值高度不稳定",
            {"condition_number": cond, "dof": n},
            "条件数过大通常意味着存在接近零能的位移模式（机构运动）。"
            "检查支座约束是否完整、是否有仅铰接的不稳定子结构、"
            "是否有零刚度杆件（截面面积为零）。",
        )
    if cond > 1e8:
        return _finding(
            "near_singular_stiffness",
            "刚度矩阵接近奇异",
            SEVERITY_WARNING,
            STATUS_WARN,
            f"刚度矩阵条件数 {cond:.2e}（> 1e8），建议关注数值精度",
            {"condition_number": cond, "dof": n},
            "条件数偏高但尚未失控。检查是否有刚度差异极大的杆件（如极柔杆与极刚杆共存），"
            "或单位不一致导致的量级差异。",
        )
    return _finding(
        "near_singular_stiffness", "刚度矩阵接近奇异", SEVERITY_CRITICAL, STATUS_PASS,
        f"刚度矩阵条件数 {cond:.2e}（< 1e8），数值稳定",
        {"condition_number": cond, "dof": n},
    )


# ---------------------------------------------------------------------------
# 检测项 5: 支座约束不足
# ---------------------------------------------------------------------------

def check_insufficient_supports(model: Frame, sol: Solution) -> dict[str, Any]:
    """支座约束不足以限制刚体运动 — 空间结构至少需要 6 个约束方向。"""
    num_support_dofs = _count_support_dofs(model)
    num_support_nodes = len(model.supports)

    # 空间刚体有 6 个自由度（3 平动 + 3 转动）
    if num_support_dofs < 6:
        return _finding(
            "insufficient_supports",
            "支座约束不足",
            SEVERITY_CRITICAL,
            STATUS_FAIL,
            f"支座约束共 {num_support_dofs} 个自由度（{num_support_nodes} 个支座节点），"
            f"少于空间刚体所需的 6 个",
            {"support_dofs": num_support_dofs, "support_nodes": num_support_nodes,
             "required": 6},
            "空间结构至少需要约束 6 个刚体自由度（3 平动 + 3 转动）。"
            "典型配置：一个固定端（6 约束），或一个铰支座（3 约束）+ 一个滚动支座（1-2 约束）"
            "+ 适当的抗扭约束。",
        )

    # 约束数够了，但可能都集中在同一个方向（比如 6 个约束全是 x 方向）
    # 检查 3 个平动方向是否都有约束
    trans_constrained = [False, False, False]
    rot_constrained = [False, False, False]
    for sup in model.supports.values():
        for i in range(3):
            if sup[i]:
                trans_constrained[i] = True
        for i in range(3, 6):
            if sup[i]:
                rot_constrained[i - 3] = True

    missing_trans = [["x", "y", "z"][i] for i in range(3) if not trans_constrained[i]]
    missing_rot = [["rx", "ry", "rz"][i] for i in range(3) if not rot_constrained[i]]

    if missing_trans:
        return _finding(
            "insufficient_supports",
            "支座约束不足",
            SEVERITY_CRITICAL,
            STATUS_FAIL,
            f"平动方向 {', '.join(missing_trans)} 完全没有约束，结构可沿该方向刚体运动",
            {"missing_translation": missing_trans, "missing_rotation": missing_rot},
            f"确保 {', '.join(missing_trans)} 方向至少有一个支座节点被约束。",
        )
    if missing_rot:
        return _finding(
            "insufficient_supports",
            "支座约束不足",
            SEVERITY_WARNING,
            STATUS_WARN,
            f"转动方向 {', '.join(missing_rot)} 没有约束（可能导致转动机构，取决于结构形式）",
            {"missing_rotation": missing_rot},
            "如果结构是平面刚架且载荷在平面内，缺少面外转动约束可能是正常的。"
            "但如果是空间结构或有面外载荷，需要补充转动约束。",
        )
    return _finding(
        "insufficient_supports", "支座约束不足", SEVERITY_CRITICAL, STATUS_PASS,
        f"支座约束 {num_support_dofs} 个自由度，6 个刚体运动方向均有约束",
        {"support_dofs": num_support_dofs, "support_nodes": num_support_nodes},
    )


# ---------------------------------------------------------------------------
# 检测项 6: 载荷量级异常
# ---------------------------------------------------------------------------

def check_load_magnitude_anomaly(model: Frame, sol: Solution) -> dict[str, Any]:
    """载荷量级与结构尺寸/材料不匹配 — 可能是单位错误（N vs kN）。"""
    if not model.materials:
        return _finding("load_magnitude_anomaly", "载荷量级异常", SEVERITY_INFO, STATUS_PASS,
                        "无材料定义，跳过载荷量级检查")

    # 取典型材料的弹性模量
    typical_E = max(m.E for m in model.materials.values())
    dim = _structure_dimension(model)

    anomalies = []
    for name in sol.all_results():
        applied = _total_applied_load(model, name)
        applied_mag = float(np.linalg.norm(applied))
        if applied_mag < 1e-10:
            continue
        # 典型应力量级 = 载荷 / 典型截面积（假设 0.01 m²）
        typical_area = 0.01  # m²，约 10cm x 10cm
        stress_estimate = applied_mag / typical_area
        # 如果估算应力超过材料屈服强度的 100 倍（钢约 235MPa，取 10GPa 为阈值）
        # 或者应力极小（< 1Pa），都可能是单位问题
        if stress_estimate > typical_E * 0.1:  # 应力 > 0.1 E，物理上不可能
            anomalies.append({
                "case": name, "applied_load_N": applied_mag,
                "estimated_stress_Pa": stress_estimate,
                "issue": "载荷过大，估算应力超过 0.1E，可能单位是 kN 但当作 N 使用",
            })
        elif stress_estimate < 1e-3:  # 应力 < 0.001 Pa
            anomalies.append({
                "case": name, "applied_load_N": applied_mag,
                "estimated_stress_Pa": stress_estimate,
                "issue": "载荷过小，可能单位是 N 但实际应该是 kN",
            })

    if anomalies:
        return _finding(
            "load_magnitude_anomaly",
            "载荷量级异常",
            SEVERITY_WARNING,
            STATUS_WARN,
            f"{len(anomalies)} 个工况载荷量级与材料不匹配："
            + "; ".join(f"{a['case']}: {a['issue']}" for a in anomalies),
            {"anomalies": anomalies, "typical_E_Pa": typical_E, "structure_dimension_m": dim},
            "检查载荷单位是否与材料单位一致。如果材料 E 用 Pa（N/m²），"
            "载荷应该用 N、尺寸用 m；如果载荷用 kN，材料 E 应该用 kPa（kN/m²）。",
        )
    return _finding(
        "load_magnitude_anomaly", "载荷量级异常", SEVERITY_WARNING, STATUS_PASS,
        "所有工况载荷量级与材料弹性模量匹配",
    )


# ---------------------------------------------------------------------------
# 检测项 7: 截面主轴方向风险
# ---------------------------------------------------------------------------

def check_section_orientation(model: Frame, sol: Solution) -> dict[str, Any]:
    """弱轴惯性矩大于强轴 — 可能 ref_vector 方向定义错误，导致 Iy/Iz 搞反。"""
    suspicious = []
    for sname, sec in model.sections.items():
        if sec.Iy > sec.Iz * 1.5:  # Iy 明显大于 Iz
            # 统计使用该截面的杆件
            used_by = [mid for mid, m in model.members.items() if m.section == sname]
            suspicious.append({
                "section": sname,
                "Iy": sec.Iy,
                "Iz": sec.Iz,
                "Iy_Iz_ratio": sec.Iy / sec.Iz if sec.Iz > 0 else float("inf"),
                "used_by_members": used_by[:10],
                "used_count": len(used_by),
            })

    if suspicious:
        return _finding(
            "section_orientation",
            "截面主轴方向风险",
            SEVERITY_WARNING,
            STATUS_WARN,
            f"{len(suspicious)} 个截面 Iy > Iz（弱轴惯性矩大于强轴）："
            + ", ".join(f"{s['section']}(Iy/ Iz={s['Iy_Iz_ratio']:.2f})" for s in suspicious),
            {"suspicious_sections": suspicious},
            "通常 Iz（绕强轴）应大于 Iy（绕弱轴）。如果 Iy > Iz，可能是："
            "(1) 截面定义时 Iy/Iz 写反了；(2) 杆件 ref_vector 方向错误导致局部坐标系旋转。"
            "检查截面特性和杆件的 ref_vector 定义。",
        )
    return _finding(
        "section_orientation", "截面主轴方向风险", SEVERITY_WARNING, STATUS_PASS,
        "所有截面 Iz >= Iy，主轴方向正常",
    )


# ---------------------------------------------------------------------------
# 检测项 8: 反力与外载荷不平衡
# ---------------------------------------------------------------------------

def check_reaction_load_balance(model: Frame, sol: Solution) -> dict[str, Any]:
    """反力合力与外载荷合力方向/大小不一致 — 平衡校核的补充检查。"""
    bad_cases = []
    for name in sol.all_results():
        applied = _total_applied_load(model, name)
        reaction = _total_reaction(model, sol, name)
        applied_mag = float(np.linalg.norm(applied))
        if applied_mag < 1e-10:
            continue
        # 反力应该与外载荷大小相等、方向相反
        residual = applied + reaction  # 应该接近零
        residual_mag = float(np.linalg.norm(residual))
        ratio = residual_mag / applied_mag
        if ratio > 0.01:  # 残差 > 1%
            bad_cases.append({
                "case": name,
                "applied_load_N": applied_mag,
                "reaction_force_N": float(np.linalg.norm(reaction)),
                "residual_N": residual_mag,
                "residual_ratio_pct": ratio * 100,
            })

    if bad_cases:
        return _finding(
            "reaction_load_balance",
            "反力与外载荷不平衡",
            SEVERITY_CRITICAL,
            STATUS_FAIL,
            f"{len(bad_cases)} 个工况反力合力与外载荷合力残差 > 1%："
            + ", ".join(f"{b['case']}({b['residual_ratio_pct']:.1f}%)" for b in bad_cases),
            {"cases": bad_cases},
            "反力与外载荷不平衡意味着：(1) 有载荷没有被约束抵抗（机构运动）；"
            "(2) 均布载荷/跨间载荷的合力计算遗漏；(3) 约束反力提取方向错误。"
            "结合 check_equilibrium 的详细残差信息定位问题。",
        )
    return _finding(
        "reaction_load_balance", "反力与外载荷不平衡", SEVERITY_CRITICAL, STATUS_PASS,
        "所有工况反力合力与外载荷合力平衡（残差 < 1%）",
    )


# ---------------------------------------------------------------------------
# 注册表与主入口
# ---------------------------------------------------------------------------

ALL_CHECKS: list[tuple[str, Callable[[Frame, Solution], dict[str, Any]]]] = [
    ("insufficient_supports", check_insufficient_supports),
    ("excessive_displacement", check_excessive_displacement),
    ("zero_reaction_with_load", check_zero_reaction_with_load),
    ("zero_internal_force", check_zero_internal_force),
    ("no_applied_load", check_no_applied_load),
    ("near_singular_stiffness", check_near_singular_stiffness),
    ("reaction_load_balance", check_reaction_load_balance),
    ("load_magnitude_anomaly", check_load_magnitude_anomaly),
    ("section_orientation", check_section_orientation),
]


def detect_silent_failures(
    model: Frame,
    sol: Solution,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[dict[str, Any]]:
    """运行全部静默失败检测，返回检测结果列表。

    Args:
        model: 已求解的 Frame 模型
        sol: solve() 返回的 Solution
        only: 只运行指定 ID 的检测（可选）
        exclude: 排除指定 ID 的检测（可选）

    Returns:
        检测结果列表，每项包含 id/name/severity/status/message/detail/suggestion
    """
    results = []
    for check_id, check_fn in ALL_CHECKS:
        if only and check_id not in only:
            continue
        if exclude and check_id in exclude:
            continue
        try:
            result = check_fn(model, sol)
        except Exception as e:  # noqa: BLE001  检测器本身挂了也要出一条 WARN——静默失败检测器自己静默失败最讽刺
            result = _finding(
                check_id, check_id, SEVERITY_INFO, STATUS_WARN,
                f"检测执行出错: {type(e).__name__}: {e}",
                suggestion="这是检测工具自身的问题，不代表模型有问题。",
            )
        results.append(result)
    return results


def summarize_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总检测结果，给出总体状态。"""
    critical = [f for f in findings if f["severity"] == SEVERITY_CRITICAL and f["status"] != STATUS_PASS]
    warnings = [f for f in findings if f["severity"] == SEVERITY_WARNING and f["status"] != STATUS_PASS]
    passed = [f for f in findings if f["status"] == STATUS_PASS]

    if critical:
        overall = "fail"
    elif warnings:
        overall = "warn"
    else:
        overall = "pass"

    return {
        "overall": overall,
        "total_checks": len(findings),
        "passed": len(passed),
        "critical_issues": len(critical),
        "warnings": len(warnings),
        "critical_details": critical,
        "warning_details": warnings,
    }


def format_findings(findings: list[dict[str, Any]], verbose: bool = False) -> str:
    """把检测结果格式化成可读的文本报告。"""
    lines = []
    summary = summarize_findings(findings)
    status_icon = {"pass": "通过", "warn": "警告", "fail": "失败"}
    lines.append(f"静默失败检测: {status_icon.get(summary['overall'], summary['overall'])} "
                 f"({summary['passed']}/{summary['total_checks']} 通过, "
                 f"{summary['critical_issues']} 严重, {summary['warnings']} 警告)")
    lines.append("")

    for f in findings:
        if f["status"] == STATUS_PASS and not verbose:
            continue
        icon = {"pass": "[OK]", "warn": "[!]", "fail": "[X]"}.get(f["status"], "[?]")
        lines.append(f"{icon} [{f['severity']}] {f['name']}")
        lines.append(f"    {f['message']}")
        if f["suggestion"]:
            lines.append(f"    建议: {f['suggestion']}")
        lines.append("")

    return "\n".join(lines)
