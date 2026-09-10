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
}

# 译完还剩下的标识符：说明有没覆盖到的字段名，那一句宁可不显示
_LEFTOVER = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)+")


def _note(payload: dict, limit: int = 150) -> str:
    """内核给的说明，**清洗后**带到界面上。

    它写的通常正是这个结果最容易被误读的地方（挠跨比的分母是杆长不是
    设计跨度、λ 只是上限不能当承载力）。丢掉的话，用户就只剩一个
    没有上下文的数字。
    """
    text = str(payload.get("note") or "").replace("**", "")
    if not text:
        return ""
    for src, dst in _TRANSLATE.items():
        text = text.replace(src, dst)
    kept = []
    for sentence in re.split(r"(?<=[。；])", text):
        s = sentence.strip()
        if not s:
            continue
        if _LEFTOVER.search(s):
            break                   # 断在这儿，别让后面的句子失去上下文
        kept.append(s)
        if sum(len(x) for x in kept) >= limit:
            break
    body = "".join(kept)
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
    """节点实体结果：每档网格一行，结论与可复核文件留在标题里。"""
    meshes = payload.get("meshes") or []
    cols = ["网格尺寸 (mm)", "节点数", "C3D10 单元数",
            "最大 Mises (MPa)", "最大绝对主应力 (MPa)", "P99 主应力 (MPa)"]
    rows = [[_r(m.get("mesh_size_mm")), m.get("nodes"), m.get("elements"),
             _r(m.get("max_mises_mpa")), _r(m.get("max_abs_principal_mpa")),
             _r(m.get("p99_abs_principal_mpa"))] for m in meshes]
    diagnosis = payload.get("peak_convergence") or {}
    verdict = {"converging": "峰值趋于收敛", "diverging": "峰值呈奇异发散",
               "inconclusive": "峰值收敛性未定"}.get(
                   diagnosis.get("verdict"), "峰值收敛性未知")
    kt = payload.get("stress_concentration_factor")
    kt_text = f"Kt={_r(kt, 3)}" if isinstance(kt, (int, float)) else "Kt 未给出"
    backend = payload.get("backend") or "未标明后端"
    title = (f"{backend}　节点 {payload.get('node_id')}，工况 {payload.get('case', '')}　"
             f"名义正应力 {_r(payload.get('nominal_normal_mpa'))} MPa　"
             f"{kt_text}　{verdict}")
    refused = payload.get("stress_concentration_refused")
    if refused:
        title += f"\n{refused}"
    contour = (payload.get("files") or {}).get("contour_png")
    if contour:
        title += f"\nMises 云图：{contour}"
    return title, cols, rows, [("node", int(payload["node_id"]))] * len(rows)


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
            "deflection": deflection, "solid_joint": solid_joint}


def to_rows(kind: str, payload: dict) -> Rows:
    """按种类摊成表。种类不认识就走兜底，**绝不抛异常**——
    界面拿不到表只是难看，抛异常是整块面板空白。"""
    try:
        return HANDLERS.get(kind, generic)(payload or {})
    except Exception as exc:                    # noqa: BLE001
        return (f"结果解析失败：{type(exc).__name__}（数据本身在下面）",
                ["项", "值"],
                [[k, str(v)[:100]] for k, v in (payload or {}).items()], [])
