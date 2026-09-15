"""把工具返回的 payload 摊成表格行。

**纯函数，不碰 Qt。** 分出来有两个好处：一是主窗口不必知道每种结果长什么样，
二是这一层能像内核一样逐条测。

---

**这个文件返工过一次，教训值得写下来。**

第一版是**照着我猜的 payload 结构写的**，键名全靠想当然（以为包络会返回
一个 members 列表）。配套测试也是我自己编一份 payload 喂进去——
于是测试在拿我的猜测验证我的猜测，全绿，却什么都没验到。
真实的包络返回的是 `peak / at_member / at_x_m / governing_case`，
根本没有 members 列表，界面上显示"共 0 根杆件"。

**是截图抓到的，不是测试。**

所以现在的测试一律用"真的建模、真的求解、真的调工具"拿到的 payload。
自己编的输入只能验格式，验不了接口对不对。

---

三条规矩：

1. **每一列都带单位**，而且优先用 payload 自己给的 `unit`，不要写死。
2. **数值保持数值类型**，不要提前格式化成字符串——表格排序按字符串排的话，
   "9" 会排在 "10" 后面。
3. **能定位的行要给出定位对象。** 只报"杆件 27"是没用的，
   用户不知道 27 在哪；行要能点，点了在视口里高亮。

另外，内核给的 `note` 往往写着这个结果最容易被误读的地方（比如挠跨比的
分母是杆长不是跨度）。**那句话要带到界面上**，不能在这一层丢掉。
"""

from __future__ import annotations

import re
from typing import Any

Rows = tuple[str, list[str], list[list[Any]], list[tuple[str, int] | None]]


def _r(x: Any, digits: int = 3) -> Any:
    """统一的数值位数。非数值原样返回，整数保持整数。"""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return x
    if isinstance(x, int):
        return x
    return round(float(x), digits)


# note 是写给**大模型**看的，里面会出现字段名和调用方式。原样印到界面上，
# 就是把内部接口漏给了用户。
#
# 第一版的做法是"整句丢掉"，结果更糟：包络那句变成"它为 0 并不等于……"，
# **代词没了指代对象**；挠度那条"分母是杆长不是设计跨度"的警告
# 整条消失——而那是这份结果里最容易被误用的一点。
#
# 所以改成**把字段名译成中文**。丢句子会丢掉意思，换词不会。
_TRANSLATE = {
    "controls_global_extreme": "「控制全结构极值」",
    "never_governs_anywhere": "「一处都不控制」",
    "member_length_over_deflection": "「杆长/挠度」",
    "span_over_deflection": "「跨度/挠度」",
    "max_displacement": "「最大节点位移」",
    "effective_mass_ratio_xyz": "「有效质量比」",
    "most_compressed_member": "「最大受压杆件」",
    "num_modes": "阶数",
    "传 member": "指定杆件",
    "传进来": "填进来",
    # 校核类结果里的字段名。这几个是被截图抓出来的：强度验算的警告原文里
    # 写着"请在杆件上显式给 mu_y / mu_z"，直接印到界面上就是把内部接口
    # 漏给了用户——和当初"算前守门"漏出去是同一类错误。
    "mu_y / mu_z": "「计算长度系数 μy / μz」",
    "mu_y": "「计算长度系数 μy」",
    "mu_z": "「计算长度系数 μz」",
    "buckling_analysis": "特征值屈曲分析",
    "check_strength": "强度验算",
    "allow_tension": "「许用拉应力」",
    "allow_compression": "「许用压应力」",
    "yield_stress": "「屈服应力」",
    "define_materials_and_sections": "材料与截面定义",
    "solve_model": "求解",
}

# 译完还剩下的标识符：说明有没覆盖到的字段名，那一句宁可不显示
_LEFTOVER = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)+")


def clean(text: Any, limit: int = 150) -> str:
    """把内核写给大模型的说明**清洗**成能印在界面上的句子。

    内核的说明里常常带着字段名和调用方式——那是写给模型看的。原样印出来
    就是把内部接口漏给了用户。所以先把认识的字段名译成中文；译完还剩下
    标识符的那一句，说明有没覆盖到的字段，宁可从那里断开也不显示。

    断开而不是跳过：跳过会让后面的句子失去指代对象（"它为 0 并不等于……"
    的"它"没了）。这一条是踩过的坑。
    """
    text = str(text or "").replace("**", "")
    if not text:
        return ""
    for src, dst in _TRANSLATE.items():
        text = text.replace(src, dst)
    kept: list[str] = []
    for sentence in re.split(r"(?<=[。；])", text):
        s = sentence.strip()
        if not s:
            continue
        if _LEFTOVER.search(s):
            break                   # 断在这儿，别让后面的句子失去上下文
        kept.append(s)
        if sum(len(x) for x in kept) >= limit:
            break
    return "".join(kept)


def _note(payload: dict, limit: int = 150) -> str:
    """内核给的 note，清洗后带到界面上。

    它写的通常正是这个结果最容易被误读的地方（挠跨比的分母是杆长不是
    设计跨度、λ 只是上限不能当承载力）。丢掉的话，用户就只剩一个
    没有上下文的数字。
    """
    body = clean(payload.get("note"), limit)
    return f"\n{body}" if body else ""


def envelope(payload: dict) -> Rows:
    """包络。分两种：全结构汇总，或指定 member 时的单杆逐点包络。"""
    comp = payload.get("component", "")
    unit = payload.get("unit", "")

    if "member" in payload:                     # 单根杆件的逐点包络
        mid = payload["member"]
        i_end = payload.get("at_i_end") or {}
        j_end = payload.get("at_j_end") or {}
        cols = ["位置", f"上包线 ({unit})", f"下包线 ({unit})"]
        rows = [["i 端", _r(i_end.get("upper")), _r(i_end.get("lower"))],
                ["j 端", _r(j_end.get("upper")), _r(j_end.get("lower"))],
                [f"峰值 @ x={_r(payload.get('at_x_m'))} m",
                 _r(payload.get("peak")), None]]
        loc = [("member", mid)] * len(rows)
        govern = "、".join(payload.get("cases_that_govern_somewhere") or [])
        title = (f"杆件 {mid} 的 {comp} 包络　杆长 "
                 f"{_r(payload.get('length_m'))} m　"
                 f"峰值 {_r(payload.get('peak'))} {unit}"
                 f"（{payload.get('governing_case', '')} 控制）")
        if govern:
            title += f"　在某处控制的组合：{govern}"
        return title + _note(payload), cols, rows, loc

    # 全结构汇总
    controls = payload.get("controls_global_extreme") or {}
    never = payload.get("never_governs_anywhere") or []
    cols = ["工况/组合", "拿下几个分量的全结构极值", "评价"]
    rows, loc = [], []
    for name, count in controls.items():
        rows.append([name, count,
                     "控制全结构极值" if count else "未拿下全结构极值，"
                     "但可能在某根杆的端部说了算"])
        loc.append(None)
    for name in never:
        rows.append([name, 0, "一处都不控制，可考虑删除或检查分项系数"])
        loc.append(None)
    title = (f"{comp} 全结构包络　峰值 {_r(payload.get('peak'))} {unit}"
             f"　位于杆件 {payload.get('at_member')} 的 "
             f"x={_r(payload.get('at_x_m'))} m"
             f"（{payload.get('governing_case', '')} 控制）")
    at = payload.get("at_member")
    if at is not None and rows:
        loc[0] = ("member", at)
    return title + _note(payload), cols, rows, loc


def buckling(payload: dict) -> Rows:
    """屈曲：各阶失稳因子 λ。"""
    factors = payload.get("factors") or []
    cols = ["阶次", "屈曲因子 λ", "说明"]
    rows = [[i + 1, _r(f, 4),
             "λ<1：当前荷载已超过线弹性失稳临界值" if isinstance(f, (int, float))
             and f < 1 else ""]
            for i, f in enumerate(factors)]
    title = (f"线弹性屈曲（工况 {payload.get('case', '')}）　"
             f"临界因子 {_r(payload.get('critical_factor'), 4)}　"
             f"最大受压杆件 {payload.get('most_compressed_member')}"
             f"（轴力 {_r(payload.get('its_axial_kN'))} kN）")
    loc = [None] * len(rows)
    mid = payload.get("most_compressed_member")
    if isinstance(mid, int) and rows:
        loc[0] = ("member", mid)
    return title + _note(payload), cols, rows, loc


def modal(payload: dict) -> Rows:
    """模态：各阶频率与周期。"""
    modes = payload.get("modes") or []
    cols = ["阶次", "频率 (Hz)", "周期 (s)"]
    rows = [[m.get("order", i + 1), _r(m.get("frequency_Hz"), 4),
             _r(m.get("period_s"), 5)] for i, m in enumerate(modes)]
    ratio = payload.get("effective_mass_ratio_xyz") or []
    title = (f"自振特性，共 {len(rows)} 阶　"
             f"总质量 {_r(payload.get('total_mass_kg'), 1)} kg")
    if ratio:
        title += ("　有效质量比 X/Y/Z："
                  + " / ".join(f"{_r(v, 3)}" for v in ratio))
    return title + _note(payload), cols, rows, [None] * len(rows)


def deflection(payload: dict) -> Rows:
    """最大挠度。"""
    cols = ["工况/组合", "杆件", "最大挠度 (mm)", "位置 x (m)",
            "杆长 (m)", "杆长/挠度"]
    rows = [[payload.get("case", ""), payload.get("at_member"),
             _r(payload.get("magnitude_mm")), _r(payload.get("at_x_m")),
             _r(payload.get("member_length_m")),
             _r(payload.get("member_length_over_deflection"), 1)]]
    span_ratio = payload.get("span_over_deflection")
    if span_ratio is not None:
        cols.append("跨度/挠度")
        rows[0].append(_r(span_ratio, 1))
    mid = payload.get("at_member")
    loc = [("member", mid)] if isinstance(mid, int) else [None]
    return "最大挠度" + _note(payload), cols, rows, loc


def solid_joint(payload: dict) -> Rows:
    """节点实体结果：主表同时交代网格代价、热点响应和发布状态。"""
    meshes = payload.get("meshes") or []
    hotspot = payload.get("hot_spot_convergence") or {}
    governing = payload.get("governing_arm")
    display_arm = str(governing) if governing is not None else (
        sorted(hotspot, key=str)[0] if hotspot else None)
    diagnosis = hotspot.get(display_arm, {}) if display_arm else {}
    fallback_values = {
        round(float(item["hotspot_mesh_size_mm"]), 9): item.get("hot_spot_mpa")
        for item in diagnosis.get("values") or []
        if item.get("hotspot_mesh_size_mm") is not None
    }
    changes = diagnosis.get("relative_changes") or []
    published = (isinstance(payload.get("stress_concentration_factor"),
                            (int, float))
                 and payload.get("hot_spot_all_converged") is True)
    cols = ["档位", "全局网格 (mm)", "热点网格 (mm)", "节点数",
            "C3D10 单元数", "最大 Mises (MPa)",
            "控制热点应力 (MPa)", "相邻变化率 (%)", "证据状态"]
    rows = []
    for index, mesh in enumerate(meshes):
        local_size = mesh.get("hotspot_mesh_size_mm", mesh.get("mesh_size_mm"))
        extrapolation = mesh.get("hot_spot_extrapolation") or {}
        hot_value = (extrapolation.get(display_arm) or {}).get("hot_spot_mpa")
        if hot_value is None and local_size is not None:
            hot_value = fallback_values.get(round(float(local_size), 9))
        final = index == len(meshes) - 1
        if final and published:
            state = "Kt 已发布"
        elif final and diagnosis.get("verdict") == "converging":
            state = "热点已稳定，Kt 未发布"
        elif final:
            state = "证据不足"
        else:
            state = "细化记录"
        rows.append([
            f"第 {index + 1} 档",
            _r(mesh.get("global_mesh_size_mm", mesh.get("mesh_size_mm"))),
            _r(local_size), mesh.get("nodes"), mesh.get("elements"),
            _r(mesh.get("max_mises_mpa")), _r(hot_value),
            _r(100.0 * changes[index - 1], 2)
            if index and index - 1 < len(changes) else None,
            state,
        ])
    peak_diagnosis = payload.get("peak_convergence") or {}
    verdict = {"converging": "峰值趋于收敛", "diverging": "峰值呈奇异发散",
               "inconclusive": "峰值收敛性未定"}.get(
                   peak_diagnosis.get("verdict"), "峰值收敛性未知")
    kt = payload.get("stress_concentration_factor")
    kt_text = f"Kt={_r(kt, 3)}" if isinstance(kt, (int, float)) else "Kt 未给出"
    backend = payload.get("backend") or "未标明后端"
    arm_text = f"控制杆臂 {governing}" if governing is not None else (
        f"诊断杆臂 {display_arm}" if display_arm is not None else "没有热点路径")
    title = (f"{backend}　节点 {payload.get('node_id')}，工况 {payload.get('case', '')}　"
             f"名义正应力 {_r(payload.get('nominal_normal_mpa'))} MPa　"
             f"{kt_text}　{arm_text}　{verdict}")
    refused = payload.get("stress_concentration_refused")
    if refused:
        title += f"\n{clean(refused, limit=260)}"
    title += "\n相贯线最大值只用于识别奇异性；正式 Kt 只采用三档稳定的 0.4t/1.0t 热点外推。"
    contour = (payload.get("files") or {}).get("contour_png")
    if contour:
        title += f"\nMises 云图：{contour}"
    return title, cols, rows, [("node", int(payload["node_id"]))] * len(rows)


def strength(payload: dict) -> Rows:
    """强度验算：一杆一行。

    **「判不了」必须与「不合格」分开显示。** 粗短杆的欧拉临界力没有物理意义，
    把它标成"不合格"会让人以为结构有问题，标成"合格"则是拿一个虚高的临界力
    盖章。所以「结论」这一列原样用内核给的三态，标题里也分开点名。
    """
    cols = ["杆件", "截面", "应力比 σ/[σ]", "控制状态", "轴力 N (kN)",
            "Euler 临界力 Pcr (kN)", "稳定利用率 N/Pcr", "长细比 λ",
            "计算长度系数 μ", "计算长度依据", "校核结论"]
    rows, loc = [], []
    for r in payload.get("members") or []:
        rows.append([r.get("member"), r.get("section"),
                     _r(r.get("stress_ratio"), 4), r.get("governs"),
                     _r(r.get("axial_kN")), _r(r.get("P_cr_kN")),
                     _r(r.get("buckling_ratio"), 4),
                     _r(r.get("slenderness"), 1), _r(r.get("mu"), 3),
                     r.get("mu_source", ""), r.get("verdict")])
        mid = r.get("member")
        loc.append(("member", int(mid)) if isinstance(mid, int) else None)

    worst = payload.get("worst_strength") or {}
    bits = [f"共 {payload.get('count', len(rows))} 根杆件",
            f"工况 {'、'.join(payload.get('cases') or [])}"]
    if worst:
        bits.append(f"最大应力比 {_r(worst.get('ratio'), 4)}"
                    f"（杆件 {worst.get('member')}，{worst.get('governs', '')}）")
    failed = payload.get("failed_members") or []
    unclear = payload.get("inconclusive_members") or []
    bits.append("全部通过" if not failed else
                "超限：" + "、".join(str(v) for v in failed))
    if unclear:
        bits.append("欧拉公式不适用（既非通过也非超限）："
                    + "、".join(str(v) for v in unclear))
    title = "　".join(bits)
    for w in payload.get("warnings") or []:
        cleaned = clean(w, limit=200)
        if cleaned:
            title += "\n" + cleaned
    limitation = clean(payload.get("limitation"), limit=200)
    if limitation:
        title += "\n" + limitation
    return title, cols, rows, loc


def symmetry(payload: dict) -> Rows:
    """对称性：每个对称面一行，附各工况的对称/反对称判定。"""
    cols = ["对称面", "结构", "各工况", "可用于校核的工况"]
    rows = []
    for p in payload.get("planes") or []:
        cases = "　".join(f"{k}：{v}" for k, v in (p.get("cases") or {}).items())
        rows.append([p.get("plane"), "对称", cases,
                     "、".join(p.get("usable_cases") or []) or "（无）"])
    check = payload.get("response_check") or {}
    for c in check.get("checks") or []:
        rows.append([c.get("plane"), f"位移镜像校核（{check.get('case', '')}）",
                     f"{c.get('kind', '')}　最大相对偏差 "
                     f"{c.get('max_relative_difference')}",
                     "通过" if c.get("ok") else "未通过"])
    title = clean(payload.get("advice"), limit=200)
    if check.get("checks"):
        title += ("\n对称结构 + 对称荷载，对称位置的位移必须互为镜像。"
                  "这条校核不需要任何外部参照。")
    elif check.get("note"):
        title += "\n" + clean(check["note"], limit=200)
    return title, cols, rows, [None] * len(rows)


def numbering(payload: dict) -> Rows:
    """编号与总刚存储：四种方案各一行，按存储量排。"""
    e = payload.get("storage_entries") or {}
    full = e.get("full") or 0

    def share(v) -> Any:
        return _r(100.0 * v / full, 1) if full else None

    cols = ["存储方案", "存储量（个数）", "相对满阵 (%)"]
    rows = [["满阵 n²", e.get("full"), share(e.get("full", 0))],
            ["等带宽 n·b（讲义 §3-10）", e.get("banded"),
             share(e.get("banded", 0))],
            ["一维变带宽（讲义 §4-6，当前编号）", e.get("skyline_current"),
             share(e.get("skyline_current", 0))],
            ["一维变带宽（RCM 重编号后）", e.get("skyline_after_rcm"),
             share(e.get("skyline_after_rcm", 0))],
            ["稀疏，只存非零元（本程序默认）", e.get("sparse_nonzeros"),
             share(e.get("sparse_nonzeros", 0))]]
    bw = payload.get("half_bandwidth") or {}
    title = (f"自由度 {payload.get('dofs')} 个　"
             f"最大节点号差 {payload.get('node_number_span')}　"
             f"半带宽 {bw.get('current')} → RCM 重编号后 {bw.get('after_rcm')}")
    v = payload.get("verification") or {}
    if v:
        title += (f"\n用讲义的一维变带宽 LDLᵀ 再解一遍，与默认稀疏解的最大相对"
                  f"偏差 {v.get('max_relative_difference')}"
                  f"（{'一致' if v.get('agrees') else '不一致，需要排查'}）。")
    lecture = clean(payload.get("lecture"), limit=220)
    if lecture:
        title += "\n" + lecture
    title += "\n" + clean(payload.get("note"), limit=120)
    return title, cols, rows, [None] * len(rows)


def generic(payload: dict) -> Rows:
    """认不出结构时的兜底：键值两列。

    **不要因为格式没对上就什么都不显示。** 摊平至少让人看得见数据本身，
    比一个空面板或一句"无法显示"有用——第一版的包络就是因为
    键名猜错显示成了空表，而空表看不出是"没有结果"还是"读错了"。
    """
    rows = []
    for key, value in (payload or {}).items():
        if isinstance(value, list):
            value = f"（列表，{len(value)} 项）"
        elif isinstance(value, dict):
            value = "　".join(f"{k}={_r(v)}" for k, v in list(value.items())[:6])
        rows.append([key, _r(value)])
    return "结果", ["项", "值"], rows, [None] * len(rows)


HANDLERS = {"envelope": envelope, "buckling": buckling, "modal": modal,
            "deflection": deflection, "solid_joint": solid_joint,
            "strength": strength, "symmetry": symmetry, "numbering": numbering}


# 行的严重度。表越长，"逐格读结论列"越不现实——超限的行要自己跳出来。
FAIL = "fail"          # 真的不合格
UNCLEAR = "unclear"    # 判不了（欧拉公式不适用那一档），既非通过也非不通过
PASS = "pass"


def _strength_marks(payload: dict) -> list[str | None]:
    failed = {int(v) for v in payload.get("failed_members") or []}
    unclear = {int(v) for v in payload.get("inconclusive_members") or []}
    out: list[str | None] = []
    for r in payload.get("members") or []:
        mid = r.get("member")
        out.append(FAIL if mid in failed else
                   UNCLEAR if mid in unclear else PASS)
    return out


def _symmetry_marks(payload: dict) -> list[str | None]:
    out: list[str | None] = [None for _ in (payload.get("planes") or [])]
    for c in (payload.get("response_check") or {}).get("checks") or []:
        out.append(PASS if c.get("ok") else FAIL)
    return out


def _solid_marks(payload: dict) -> list[str | None]:
    meshes = payload.get("meshes") or []
    if not meshes:
        return []
    kt = payload.get("stress_concentration_factor")
    if isinstance(kt, (int, float)):
        final = PASS if payload.get("hot_spot_all_converged") is True else FAIL
    else:
        final = UNCLEAR
    return [None] * (len(meshes) - 1) + [final]


_MARKERS = {"strength": _strength_marks, "symmetry": _symmetry_marks,
            "solid_joint": _solid_marks}


def row_marks(kind: str, payload: dict) -> list[str | None]:
    """每行的严重度，供界面着色。认不出的种类一律不标——
    **宁可全表素色，也不要把好行标成红的**。"""
    try:
        return _MARKERS[kind](payload or {})
    except Exception:                           # noqa: BLE001
        return []


def to_rows(kind: str, payload: dict) -> Rows:
    """按种类摊成表。种类不认识就走兜底，**绝不抛异常**——
    界面拿不到表只是难看，抛异常是整块面板空白。"""
    try:
        return HANDLERS.get(kind, generic)(payload or {})
    except Exception as exc:                    # noqa: BLE001
        return (f"结果解析失败：{type(exc).__name__}（数据本身在下面）",
                ["项", "值"],
                [[k, str(v)[:100]] for k, v in (payload or {}).items()], [])


def engineering_summary(kind: str, payload: dict) -> dict:
    """把结构化结果压成顶部工程结论，不从标题文本或表格列名猜语义。"""
    data = payload or {}
    summary: dict[str, Any] = {"solve_status": "已完成"}
    if kind == "deflection":
        value = data.get("magnitude_mm")
        summary.update(
            maximum=f"{_r(value, 4)}" if isinstance(value, (int, float)) else "—",
            minimum="不适用",
            location=(f"杆件 {data.get('at_member')} · 距 i 端 "
                      f"{_r(data.get('at_x_m'))} m"),
            case=data.get("case") or "—", unit="mm")
    elif kind == "envelope":
        peak = data.get("peak")
        summary.update(
            maximum=f"{_r(peak, 4)}" if isinstance(peak, (int, float)) else "—",
            minimum="见上下包线",
            location=(f"杆件 {data.get('at_member')} · 距 i 端 "
                      f"{_r(data.get('at_x_m'))} m"
                      if data.get("at_member") is not None else
                      "未提供控制杆件"),
            case=data.get("governing_case") or "多工况包络",
            unit=data.get("unit") or "—")
    elif kind == "strength":
        ratios = [row.get("stress_ratio") for row in data.get("members") or []
                  if isinstance(row.get("stress_ratio"), (int, float))]
        worst = data.get("worst_strength") or {}
        failed = data.get("failed_members") or []
        unclear = data.get("inconclusive_members") or []
        if failed:
            severity, status = "fail", "已完成 · 存在超限"
        elif unclear:
            severity, status = "unclear", "已完成 · 存在待复核项"
        else:
            severity, status = "pass", "已完成 · 未发现超限"
        summary.update(
            maximum=f"{_r(max(ratios), 4)}" if ratios else "—",
            minimum=f"{_r(min(ratios), 4)}" if ratios else "—",
            location=(f"杆件 {worst.get('member')} · "
                      f"{worst.get('governs') or '控制截面'}"
                      if worst else "未发现控制杆件"),
            case="、".join(data.get("cases") or []) or "—",
            unit="应力比",
            severity=severity, solve_status=status)
    elif kind == "buckling":
        factors = [v for v in data.get("factors") or []
                   if isinstance(v, (int, float))]
        summary.update(
            maximum=f"{_r(max(factors), 4)}" if factors else "—",
            minimum=f"{_r(min(factors), 4)}" if factors else "—",
            location=(f"杆件 {data.get('most_compressed_member')} · 最大受压"
                      if data.get("most_compressed_member") is not None
                      else "未提供控制杆件"),
            case=data.get("case") or "—", unit="屈曲因子 λ")
    elif kind == "modal":
        frequencies = [mode.get("frequency_Hz")
                       for mode in data.get("modes") or []
                       if isinstance(mode.get("frequency_Hz"), (int, float))]
        summary.update(
            maximum=f"{_r(max(frequencies), 4)}" if frequencies else "—",
            minimum=f"{_r(min(frequencies), 4)}" if frequencies else "—",
            location="第 1 阶振型" if frequencies else "未生成振型",
            case="与荷载工况无关", unit="Hz")
    elif kind == "solid_joint":
        stresses = [mesh.get("max_mises_mpa")
                    for mesh in data.get("meshes") or []
                    if isinstance(mesh.get("max_mises_mpa"), (int, float))]
        summary.update(
            maximum=f"{_r(max(stresses), 4)}" if stresses else "—",
            minimum=f"{_r(min(stresses), 4)}" if stresses else "—",
            location=(f"节点 {data.get('node_id')} · 控制杆臂 "
                      f"{data.get('governing_arm')}"),
            case=data.get("case") or "—", unit="MPa",
            severity="pass" if data.get("hot_spot_all_converged") is True
            else "unclear",
            solve_status=("已完成 · 热点已收敛"
                          if data.get("hot_spot_all_converged") is True
                          else "已完成 · 热点待复核"))
    return summary
