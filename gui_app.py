r"""空间刚架智能计算 —— 图形界面。

**它只是一层壳。** 所有计算都走 `agent.Session` 里那套工具——和命令行、和 Agent
用的是同一条链路，没有任何"界面专用"的计算分支。界面上的每个数字，都和评测集里
跑出来的是同一份代码算的（`tests/test_gui.py` 守这一条）。

界面按有限元的三个阶段分层：**前处理 → 分析 → 后处理**。
早先是六个平级标签页加一条长侧边栏，"建模的"和"看结果的"混在一层，
标签一多就找不着东西。现在顶层只有三个，各自再分二级页；
参数表单从侧边栏搬进了前处理，页面因此宽出一大截，三维视图和表格都好读了。

配色取自 dataviz 参考调色板，与 `plot3d.py` 出的报告图共用 `viz_theme`，
静态图和界面不会各说各话。

启动：

    .venv\Scripts\activate.bat
    set PYTHONPATH=src
    streamlit run gui_app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

import pandas as pd

import model_tables as MT
import sections as S
import viz_symbols as VS
import viz_theme as T
from agent import DeepSeekProvider, Session, run_turn
from conversation import Conversation
from model_io import validate_payload
from model_router import ModelConfig, ModelRouter
from section_optimizer import SectionOptimizer
from sketch_parser import SketchParser
from plot3d import plot_axial, plot_deformed, plot_diagram
from plot3d_interactive import (figure_axial, figure_deformed,
                                figure_diagram_3d, figure_member_diagram,
                                figure_member_envelope, figure_model)

sys.path.insert(0, str(Path(__file__).resolve().parent / "examples"))

from console import use_utf8            # 见 src/console.py：别让一个字符打死一次成功的运行
use_utf8()
try:
    from agent_live import load_api_key
except ImportError:                      # examples 不在时也别让界面挂掉
    def load_api_key():
        return os.environ.get("DEEPSEEK_API_KEY", "").strip() or None

st.set_page_config(page_title="空间刚架智能计算", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""<style>
  /* ---- 整体：向 CAE 前处理器靠拢 ----
     参照 Abaqus/CAE 与 FEM-Python 的观感：字号小一档、行距紧、
     每块内容都有边框和小标题、数字用等宽数位。信息密度高才像专业工具，
     松散的大字排版看着像演示页。 */
  html, body, [class*="css"] {{ font-size: 14px; }}
  .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem;
                      max-width: 1560px; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .stApp {{ background: {T.PAGE}; }}

  /* 标题条：产品名 + 版本状态，像软件的标题栏 */
  .titlebar {{ display: flex; align-items: baseline; gap: .8rem;
               border-bottom: 1px solid {T.GRID}; padding-bottom: .55rem;
               margin-bottom: .9rem; }}
  .app-title {{ font-size: 1.15rem; font-weight: 650; letter-spacing: -.01em;
                color: {T.INK}; margin: 0; }}
  .app-sub {{ color: {T.INK_MUTED}; font-size: .78rem; margin: 0; }}

  /* 数字一律等宽数位——表格里的小数点要对齐 */
  div[data-testid="stMetricValue"], .stDataFrame, code, .badge b {{
      font-variant-numeric: tabular-nums; }}

  /* 指标卡：压扁一档，去掉圆角的"卡片感"，改成工具栏读数 */
  div[data-testid="stMetric"] {{
      background: {T.SURFACE}; border: 1px solid {T.GRID};
      border-radius: 3px; padding: .5rem .7rem;
  }}
  div[data-testid="stMetricLabel"] p {{
      color: {T.INK_MUTED}; font-size: .7rem; letter-spacing: .04em;
      text-transform: uppercase;
  }}
  div[data-testid="stMetricValue"] {{ color: {T.INK}; font-size: 1.18rem; }}
  div[data-testid="stMetricDelta"] {{ font-size: .7rem; }}

  /* 一级标签页做成模块选择器（Abaqus 的 Module 下拉在这里横过来） */
  .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {T.GRID}; }}
  .stTabs [data-baseweb="tab"] {{
      padding: .45rem 1.1rem; color: {T.INK_SECONDARY}; font-size: .86rem;
      border: 1px solid transparent; border-bottom: none; border-radius: 3px 3px 0 0;
  }}
  .stTabs [aria-selected="true"] {{
      color: {T.INK}; font-weight: 600; background: {T.SURFACE};
      border-color: {T.GRID};
  }}
  /* 二级标签页：更轻，明显低一层 */
  .stTabs .stTabs [data-baseweb="tab"] {{
      padding: .28rem .7rem; font-size: .8rem; }}
  .stTabs .stTabs [data-baseweb="tab-list"] {{
      border-bottom: 1px dotted {T.GRID}; margin-bottom: .5rem; }}
  .stTabs .stTabs [aria-selected="true"] {{ background: transparent; }}

  /* 三维视口：深底，四周一圈细边，和页面明确分开 */
  .stPlotlyChart {{ border: 1px solid {T.GRID}; border-radius: 3px;
                    overflow: hidden; }}

  /* 模型树：等宽、紧凑，像树控件而不是段落 */
  .tree-title {{ font-size: .72rem; letter-spacing: .06em; color: {T.INK_MUTED};
                 text-transform: uppercase; margin: 0 0 .3rem; }}
  div[data-testid="stExpander"] details {{ border: 1px solid {T.GRID};
      border-radius: 3px; background: {T.SURFACE}; margin-bottom: .25rem; }}
  div[data-testid="stExpander"] summary {{ font-size: .82rem; padding: .3rem .6rem; }}
  div[data-testid="stExpander"] p {{ font-size: .76rem; }}

  /* 表格：细边、小字 */
  .stDataFrame {{ font-size: .8rem; }}

  /* 底部状态栏：CAE 的 message area */
  .statusbar {{ position: sticky; bottom: 0; margin-top: 1.2rem;
                border-top: 1px solid {T.GRID}; background: {T.SURFACE};
                padding: .4rem .7rem; font-size: .75rem;
                color: {T.INK_SECONDARY};
                display: flex; gap: 1.4rem; flex-wrap: wrap; }}
  .statusbar .k {{ color: {T.INK_MUTED}; letter-spacing: .04em; }}

  .badge {{ display: inline-flex; align-items: center; gap: .35rem;
            font-size: .74rem; padding: .16rem .5rem; border-radius: 3px;
            border: 1px solid {T.GRID}; background: {T.SURFACE};
            color: {T.INK_SECONDARY}; }}
  .badge b {{ font-weight: 600; }}
</style>""", unsafe_allow_html=True)

PRESET_COL = {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6}
PRESET_BEAM = {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}


def _session() -> Session:
    if "fem" not in st.session_state:
        st.session_state.fem = Session()
    return st.session_state.fem


def _model_summary(fem: Session, case: str) -> str:
    n = len(fem.frame.nodes)
    m = len(fem.frame.members)
    fixed = sum(1 for mask in fem.frame.supports.values() if all(mask))
    kinds = len({mem.section for mem in fem.frame.members.values()})
    return (f"{n} 节点 · {m} 杆件 · {len(fem.frame.supports)} 支座"
            f"（{fixed} 个全固接）· {kinds} 种截面 · 工况 {case}")


def _model_tree(fem: Session, result) -> None:
    """左侧那种模型树。FEM-Python 的界面靠它一眼看清模型有什么，
    这里用嵌套展开栏做出同样的层级——比翻七张表快得多。"""
    model = fem.model
    st.markdown("**模型树**")
    with st.expander(f"材料（{len(model.get('materials', []))}）", expanded=False):
        for m in model.get("materials", []):
            rho = m.get("density")
            st.caption(f"{m['name']}　E = {m['E']:.4g}　ν = {m['nu']}"
                       + (f"　ρ = {rho:g}" if rho else "　（无密度，不能做模态）"))
    with st.expander(f"截面（{len(model.get('sections', []))}）", expanded=False):
        for sec in model.get("sections", []):
            st.caption(f"{sec['name']}　A = {sec['A']:.4g}　Iy = {sec['Iy']:.4g}"
                       f"　Iz = {sec['Iz']:.4g}　J = {sec['J']:.4g}")
    with st.expander(f"几何（{len(model.get('nodes', []))} 节点 / "
                     f"{len(model.get('members', []))} 杆件）", expanded=False):
        released = [m for m in model.get("members", []) if m.get("releases")]
        st.caption(f"其中 {len(released)} 根杆件带端部释放"
                   if released else "所有杆端均为刚接")
    with st.expander(f"约束（{len(model.get('supports', []))}）", expanded=False):
        kinds: dict[str, int] = {}
        for sup in model.get("supports", []):
            k = VS.classify_support(sup["fix"])
            kinds[k] = kinds.get(k, 0) + 1
        for k, n in kinds.items():
            st.caption(f"{k} × {n}")
    cases = model.get("load_cases", [])
    with st.expander(f"荷载工况（{len(cases)}）", expanded=False):
        for c in cases:
            bits = []
            for key, label in (("nodal_loads", "节点荷载"),
                               ("member_loads", "均布"),
                               ("member_spans", "梯形/集中"),
                               ("settlements", "沉降")):
                if c.get(key):
                    bits.append(f"{label} {len(c[key])}")
            st.caption(f"{c['name']}　" + ("　·　".join(bits) or "（空）"))
    combos = model.get("combos", [])
    if combos:
        with st.expander(f"荷载组合（{len(combos)}）", expanded=False):
            for c in combos:
                terms = "　".join(f"{v:+g}×{k}" for k, v in c["factors"].items())
                st.caption(f"{c['name']}　=　{terms}")
    if result is not None and result.ok:
        with st.expander("结果", expanded=False):
            st.caption(f"已求解 {len(result.payload['cases'])} 个工况 / 组合")
            st.caption("位移 · 反力 · 杆端力 · 内力图 · 包络")
            for label, key in (("模态", "modal_cache"), ("屈曲", "buckling_cache")):
                got = st.session_state.get(key)
                if got:
                    st.caption(f"{label}：{got}")


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.replace("，", ",").split(",") if x.strip()]


def _forget_model_state() -> None:
    """换模型时，编辑区的旧内容和上一次的 Abaqus 对比结果都作废。"""
    for key in ("edit_tables", "edit_csv", "cmp_result"):
        st.session_state.pop(key, None)
    for name in MT.TABLE_NAMES:
        st.session_state.pop(f"editor_{name}", None)


def _section_input(label: str, name: str, default_kind: str,
                   key: str) -> dict[str, float]:
    """按尺寸填，特性由公式算。直接填 A/Iy/Iz/J 太容易填错，也看不出强弱轴。"""
    st.markdown(f"**{label}**")
    kinds = list(S.BUILDERS) + ["直接输入特性"]
    kind = st.selectbox("截面类型", kinds, index=kinds.index(default_kind),
                        key=f"{key}_kind", label_visibility="collapsed")
    if kind == "直接输入特性":
        preset = PRESET_COL if key == "col" else PRESET_BEAM
        out = {"name": name}
        cols = st.columns(4)
        for c, k in zip(cols, ("A", "Iy", "Iz", "J")):
            with c:
                out[k] = st.number_input(k, value=float(preset[k]), format="%.3g",
                                         key=f"{key}_{k}")
        return out

    builder, keys, labels = S.BUILDERS[kind]
    values = []
    cols = st.columns(len(keys))
    for c, k, lab, default in zip(cols, keys, labels, S.DEFAULT_DIMENSIONS[kind]):
        with c:
            values.append(st.number_input(lab, value=float(default), step=0.005,
                                          format="%.3f", key=f"{key}_{k}"))
    try:
        props = builder(name, *values)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.caption(f"A = {props['A']:.4g} m²　Iy = {props['Iy']:.4g}　"
               f"Iz = {props['Iz']:.4g} m⁴　J = {props['J']:.4g} m⁴"
               f"　（Iz 为强轴，平面内弯曲用它）")
    return props


def _parametric_form():
    """参数化建模表单。原来在侧边栏，现在归到前处理里 —— 它本来就是前处理。"""
    form = st.radio("结构形式", ["规整框架", "坡屋面门式刚架"], horizontal=True,
                    help="规整框架是平屋面等跨等层；门式刚架跨中起脊，"
                         "柱脚常规做法是铰接")
    geo, sec = st.columns([1, 1])
    with geo:
        st.markdown("**几何**")
        spans_text = st.text_input("X 向各跨跨度 (m)",
                                   "6, 6, 6" if form == "规整框架" else "24")
        if form == "规整框架":
            storeys_text = st.text_input("各层层高 (m)，自下而上", "3.6, 3.6")
            eave_height = ridge_rise = 0.0
        else:
            storeys_text = ""
            eave_height = st.number_input("檐口高度 (m)", value=6.0, step=0.5)
            ridge_rise = st.number_input("屋脊升高 (m)", value=2.0, step=0.25,
                                         help="坡度 i = 屋脊升高 ÷ (跨度/2)")
        bays_text = st.text_input("Y 向各开间宽度 (m)，留空为平面刚架",
                                  "6" if form == "规整框架" else "6, 6, 6, 6")

        st.markdown("**约束与荷载**")
        base = st.radio("柱底", ["固接", "铰接"], horizontal=True,
                        index=0 if form == "规整框架" else 1)
        beam_release = (st.checkbox("梁两端铰接") if form == "规整框架" else False)
        load_kn = st.number_input(
            "梁上均布荷载 (kN/m，向下)" if form == "规整框架"
            else "斜梁上均布荷载 (kN/m，向下)",
            value=20.0 if form == "规整框架" else 12.0, step=1.0)
        if not bays_text.strip():
            st.caption("平面刚架的面外自由度会被自动约束——否则柱脚铰接时它是机构。")

    with sec:
        st.markdown("**材料**")
        m1, m2 = st.columns(2)
        E = m1.number_input("弹性模量 E (Pa)", value=2.1e11, format="%.4g")
        nu = m2.number_input("泊松比", value=0.3, min_value=0.0, max_value=0.5)
        st.divider()
        col = _section_input("柱截面", "COLUMN", "工字形 / H 型钢", "col")
        st.divider()
        beam = _section_input("梁截面", "BEAM", "工字形 / H 型钢", "beam")

    run = st.button("生成并求解", type="primary")
    return run, dict(form=form, spans_text=spans_text, storeys_text=storeys_text,
                     bays_text=bays_text, eave_height=eave_height,
                     ridge_rise=ridge_rise, base=base, beam_release=beam_release,
                     load_kn=load_kn, E=E, nu=nu, col=col, beam=beam)


def _generate(p) -> tuple[Session | None, str | None]:
    """按表单参数建一个新 Session。失败时返回错误文字，**不动现有模型** ——
    生成失败还把手上能算的模型清掉，是最让人恼火的一种行为。"""
    fem = Session()
    fem.define_materials_and_sections(
        [{"name": "STEEL", "E": float(p["E"]), "nu": float(p["nu"])}],
        [p["col"], p["beam"]])
    fixity = "fixed" if p["base"] == "固接" else "pinned"
    try:
        if p["form"] == "规整框架":
            gen = fem.generate_frame(
                spans=_floats(p["spans_text"]), storeys=_floats(p["storeys_text"]),
                bays=_floats(p["bays_text"]), column_section=p["col"]["name"],
                beam_section=p["beam"]["name"], material="STEEL",
                base=fixity, beam_release=p["beam_release"])
            loaded_key = "beam_member_ids"
        else:
            gen = fem.generate_portal_frame(
                spans=_floats(p["spans_text"]), eave_height=float(p["eave_height"]),
                ridge_rise=float(p["ridge_rise"]), bays=_floats(p["bays_text"]),
                column_section=p["col"]["name"], rafter_section=p["beam"]["name"],
                material="STEEL", base=fixity)
            loaded_key = "rafter_member_ids"
    except ValueError as exc:
        return None, f"参数有误：{exc}"
    if not gen.ok:
        return None, gen.payload.get("error", "生成失败")
    fem.set_load_cases(cases=[{"name": "DL", "member_loads": [
        {"member": m, "w": [0.0, 0.0, -abs(p["load_kn"]) * 1000.0]}
        for m in gen.payload[loaded_key]]}])
    return fem, None


_MM = st.column_config.NumberColumn(format="%.4f")
_KN = st.column_config.NumberColumn(format="%.3f")
_M = st.column_config.NumberColumn(format="%.3f")

_COMPONENTS = {"弯矩 Mz": "Mz", "剪力 Vy": "Vy", "轴力 N": "N",
               "弯矩 My": "My", "剪力 Vz": "Vz", "扭矩 T": "T"}

# ------------------------------------------------------------------ 页头

st.markdown(
    '<div class="titlebar">'
    '<p class="app-title">空间刚架智能计算</p>'
    '<p class="app-sub">静力 · 模态 · 屈曲　|　自研求解器 + Abaqus 对标</p>'
    '</div>', unsafe_allow_html=True)

# 状态条要显示在最上面，内容却依赖下面才算出来的结果。先占个位，算完再填回来。
status_slot = st.container()

tab_pre, tab_run, tab_post = st.tabs(
    ["① 前处理 · 建模", "② 分析 · 求解与校核", "③ 后处理 · 结果"])

# ------------------------------------------------------------------ 前处理

with tab_pre:
    sub_param, sub_agent, sub_sketch, sub_view, sub_steps, sub_edit, sub_json = st.tabs(
        ["参数化建模", "自然语言建模", "手绘草图", "模型视图", "建模过程",
         "模型编辑", "模型 JSON"])

    with sub_param:
        st.caption("不联网、不花钱、结果确定可复现。规整框架和门式刚架用这个最快；"
                   "不规整的地方到「模型编辑」里逐行改。")
        _run, _params = _parametric_form()

    if _run:
        _fresh, _err = _generate(_params)
        if _err:
            with sub_param:
                st.error(_err)
        else:
            st.session_state.fem = _fresh
            st.session_state.agent_case = None
            _forget_model_state()
            st.session_state.solve_result = _fresh.solve_model()

    with sub_agent:
        st.caption("多轮对话：模型状态跨轮保留。建完之后可以接着提要求，"
                   "比如让它把柱子加大一点再算一遍——它会在现有模型上改，"
                   "不会推倒重建。")
        api_key = load_api_key()
        if not api_key:
            st.info("未找到 API 密钥。在仓库根目录建 `deepseek.key`，里面只写一行密钥，"
                    "然后刷新页面。参数化建模不需要密钥。")
        else:
            model_name = st.text_input("主模型", "deepseek-v4-flash", key="agent_model")
            _use_router = st.checkbox("启用备用模型降级（主模型失败时自动切换）",
                                       key="agent_use_router")
            if _use_router:
                _bk_col1, _bk_col2, _bk_col3 = st.columns(3)
                _bk_name = _bk_col1.text_input("备用模型名", "gpt-4o-mini", key="bk_model")
                _bk_url = _bk_col2.text_input("备用 base_url", "https://api.openai.com/v1",
                                                key="bk_url")
                _bk_key = _bk_col3.text_input("备用 API 密钥", type="password", key="bk_key",
                                                help="不填则用主模型密钥")
                st.caption("主模型调用失败（429/500/网络错误）时自动降级到备用模型，"
                           "调用历史可在下方查看。")

            for _k, _row in enumerate(st.session_state.get("chat_log", [])):
                with st.chat_message("user"):
                    st.write(_row["user"])
                with st.chat_message("assistant"):
                    st.write(_row["reply"])
                    if _row["tools"]:
                        st.caption(f"{_row['rounds']} 轮 · 调用了 "
                                   + " → ".join(_row["tools"]))

            _prompt = st.chat_input("描述结构，或对上一轮的结果提要求")
            _cols = st.columns([1, 1, 4])
            if _cols[0].button("清空对话", help="只清对话，模型保留"):
                if "chat" in st.session_state:
                    st.session_state.chat.reset_history()
                st.session_state.chat_log = []
                st.rerun()
            if _cols[1].button("从头开始", help="对话和模型一起清掉"):
                for _k in ("chat", "chat_log", "fem", "solve_result", "agent_case"):
                    st.session_state.pop(_k, None)
                _forget_model_state()
                st.rerun()

            # 模型调用状态展示
            _prov = st.session_state.get("chat_provider")
            if isinstance(_prov, ModelRouter) and _prov.call_history:
                _last = _prov.call_history[-1]
                _status = "✅ 成功" if _last.success else f"❌ {_last.error[:40] if _last.error else '失败'}"
                _fallback = " ⚠️ 已降级" if _last.was_fallback else ""
                st.caption(f"最近调用：{_last.model} · {_status}{_fallback} · "
                           f"{_last.duration_ms:.0f}ms · 共 {len(_prov.call_history)} 次调用")
            elif _prov is not None:
                st.caption(f"当前模型：{model_name}")

            if _prompt:
                if "chat" not in st.session_state:
                    if _use_router:
                        _configs = [
                            ModelConfig(name=model_name, base_url="https://api.deepseek.com",
                                        api_key=api_key, priority=1),
                            ModelConfig(name=_bk_name, base_url=_bk_url,
                                        api_key=_bk_key or api_key, priority=2),
                        ]
                        _provider = ModelRouter(_configs)
                    else:
                        _provider = DeepSeekProvider(api_key=api_key, model=model_name)
                    st.session_state.chat = Conversation(
                        _provider, session=_session())
                    st.session_state.chat_provider = _provider
                chat = st.session_state.chat
                with st.spinner("Agent 正在思考并调用工具……"):
                    try:
                        out = chat.ask(_prompt)
                    except Exception as exc:
                        st.error(f"调用失败：{type(exc).__name__}: {exc}")
                        out = None
                if out is not None:
                    st.session_state.fem = chat.session
                    st.session_state.solve_result = (
                        chat.session.solve_model()
                        if chat.session.solution is not None else None)
                    st.session_state.agent_case = (
                        chat.session._controlling_case()
                        if chat.session.solution is not None else None)
                    _forget_model_state()
                    st.session_state.chat_log = chat.transcript()
                    st.rerun()

    with sub_sketch:
        st.caption("上传手绘结构草图，多模态 LLM 自动识别节点、杆件、支座和荷载，"
                   "转换成可求解的模型。LLM 只产结构拓扑，数值仍由确定性求解器计算。")
        _sk_col1, _sk_col2 = st.columns([1, 2])
        with _sk_col1:
            _sk_provider = st.selectbox("多模态模型", ["openai", "anthropic", "deepseek"],
                                         key="sketch_provider")
            _sk_model = st.text_input("模型名称",
                                       {"openai": "gpt-4o", "anthropic": "claude-3-5-sonnet-20241022",
                                        "deepseek": "deepseek-vl"}[_sk_provider],
                                       key="sketch_model")
            _sk_key = st.text_input("API 密钥", type="password", key="sketch_key",
                                     help="不填则从环境变量读取（OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY）")
            _sk_retries = st.slider("自修复重试次数", 1, 5, 3, key="sketch_retries")
        with _sk_col2:
            _sk_file = st.file_uploader("上传手绘草图（PNG/JPG）", type=["png", "jpg", "jpeg", "webp"],
                                         key="sketch_file")
            if _sk_file is not None:
                st.image(_sk_file, caption="上传的草图", use_container_width=True)
                # 保存到临时文件
                import tempfile
                _sk_tmp = tempfile.NamedTemporaryFile(suffix=_sk_file.name[-4:], delete=False)
                _sk_tmp.write(_sk_file.getvalue())
                _sk_tmp.close()
                _sk_path = _sk_tmp.name

                if st.button("识别草图 → 模型", type="primary", key="sketch_run"):
                    with st.spinner("多模态 LLM 正在识别草图结构……"):
                        try:
                            if _sk_key:
                                _parser = SketchParser(provider=_sk_provider, api_key=_sk_key,
                                                       model=_sk_model)
                            else:
                                _parser = SketchParser.from_env(_sk_provider)
                            _sk_result = _parser.parse_with_retry(_sk_path, max_retries=_sk_retries)
                            if _sk_result.success:
                                st.session_state.sketch_model = _sk_result.model
                                st.success(f"识别成功！{_sk_result.attempts} 次尝试，"
                                           f"{len(_sk_result.model['nodes'])} 个节点，"
                                           f"{len(_sk_result.model['members'])} 根杆件")
                            else:
                                st.error(f"识别失败（{_sk_result.attempts} 次尝试）：")
                                for _e in _sk_result.errors:
                                    st.text(_e)
                        except Exception as _exc:
                            st.error(f"调用失败：{type(_exc).__name__}: {_exc}")

                if st.session_state.get("sketch_model"):
                    st.divider()
                    st.write("**识别结果预览**")
                    _sm = st.session_state.sketch_model
                    _c1, _c2, _c3 = st.columns(3)
                    _c1.metric("节点", len(_sm.get("nodes", [])))
                    _c2.metric("杆件", len(_sm.get("members", [])))
                    _c3.metric("支座", len(_sm.get("supports", [])))
                    with st.expander("查看识别出的模型 JSON"):
                        st.json(_sm)
                    if st.button("加载为当前模型", type="primary", key="sketch_load"):
                        _fresh = Session()
                        _fresh.set_model(_sm)
                        st.session_state.fem = _fresh
                        st.session_state.agent_case = None
                        _forget_model_state()
                        st.session_state.solve_result = _fresh.solve_model()
                        st.rerun()

    fem = _session()
    result = st.session_state.get("solve_result")
    has_model = bool(fem.model.get("nodes"))

    with sub_view:
        if not has_model:
            st.info("还没有模型。先到「参数化建模」生成一个，或用自然语言建一个。")
        else:
            _tree_col, _fig_col = st.columns([1, 3])
            with _tree_col:
                _model_tree(fem, result)
            with _fig_col:
                _view_cases = [c.get("name") for c in fem.model.get("load_cases", [])]
                _bar = st.columns([2, 1, 1])
                _vcase = (_bar[0].selectbox("工况", _view_cases, key="view_case")
                          if _view_cases else None)
                _show_sup = _bar[1].checkbox("约束符号", value=True)
                _show_load = _bar[2].checkbox("荷载符号", value=True)
                try:
                    # preview_frame 不需要先求解——看模型正是求解**之前**该做的事
                    _fig, _meta = figure_model(fem.preview_frame(), _vcase,
                                               height=520, supports=_show_sup,
                                               loads=_show_load)
                except ValueError as _exc:
                    st.warning(f"模型还画不出来：{_exc}")
                except Exception as _exc:      # noqa: BLE001
                    st.warning(f"画不出来：{type(_exc).__name__}: {_exc}")
                else:
                    st.plotly_chart(_fig, width='stretch')
                    _bits = []
                    if _meta.get("supports"):
                        _sp = _meta["supports"]
                        _bits.append(f"支座符号 {_sp['drawn']} 个")
                        if _sp["skipped"]:
                            _bits.append(f"另有 {_sp['skipped']} 个节点只约束了一个"
                                         "平动方向（多半是平面刚架自动加的面外约束），"
                                         "不画符号")
                    if _meta.get("loads"):
                        _ld = _meta["loads"]
                        _bits.append(f"荷载箭头 {_ld['nodal']} 支节点、"
                                     f"{_ld['member']} 支杆间")
                    st.caption("　·　".join(_bits))
                    st.caption("**箭头长度在节点荷载与杆间荷载各自的范围内归一**，"
                               "两组之间不可比——荷载量级常跨几个数量级，"
                               "共用一把尺子会让小的那类彻底消失。"
                               "看这张图主要是确认**方向**有没有加反。")

    with sub_steps:
        steps = fem.history
        if not len(steps):
            st.info("还没有建模操作。参数化生成、自然语言建模、手工编辑都会记在这里。")
        else:
            st.caption("**建模过程**　每一次改动模型的操作都记了一步，"
                       "包括那一步之后模型长什么样。拖动下面的滑块回看。")
            _pick = st.slider("回看到第几步", 1, len(steps), len(steps),
                              key="history_step")
            _step = steps[_pick - 1]

            _l, _r = st.columns([1, 2])
            with _l:
                st.dataframe(pd.DataFrame(steps.timeline()), hide_index=True,
                             width='stretch', height=min(320, 40 * len(steps) + 40))
            with _r:
                st.markdown(f"**第 {_step.index + 1} 步　{_step.summary}**")
                st.caption(f"工具 `{_step.tool}`　·　{_step.changed}"
                           + ("" if _step.ok else f"　·　未通过：{_step.error}"))
                _cols = st.columns(4)
                for _c, (_k, _v) in zip(_cols * 2, list(_step.stats.items())[:4]):
                    _c.metric(_k, _v)
                try:
                    # 用那一步的快照重建，画出当时的模型——这才是"过程"
                    _snap = Session()
                    _snap.model = _step.model
                    _frame = _snap.preview_frame()
                except ValueError as _exc:
                    st.info(f"这一步的模型还不完整，画不出来：{_exc}")
                else:
                    _cases = [c.get("name")
                              for c in _step.model.get("load_cases", [])]
                    _fig, _ = figure_model(_frame, _cases[0] if _cases else None,
                                           height=360)
                    st.plotly_chart(_fig, width='stretch',
                                    key=f"history_fig_{_pick}")

    with sub_edit:
        if not has_model:
            st.info("还没有模型。先到「参数化建模」生成一个，或用自然语言建一个。")
        else:
            st.markdown("**逐行改模型**")
            st.caption("左侧那些参数只能生成规整框架和门式刚架。真实的刚架总有不规整的地方 —— "
                       "抽柱、错层、局部加撑、某一跨单独加载。这里可以直接改节点坐标、增删杆件、"
                       "调约束和荷载。单位与模型一致（m、N、N·m、N/m），表格里不做换算。")

            _EDITOR_SPECS = (
                ("nodes", "节点", MT.NODE_COLUMNS,
                 "id 是节点号，x/y/z 是坐标（m）"),
                ("members", "杆件", MT.MEMBER_COLUMNS,
                 "i/j 是两端节点号；releases 填局部自由度名，如 rz 表示该端不传弯矩，留空即刚接"),
                ("supports", "约束", MT.SUPPORT_COLUMNS,
                 "勾上表示该方向被约束住。六个全勾是固接，只勾前三个是铰接"),
                ("nodal_loads", "节点荷载", MT.NODAL_LOAD_COLUMNS,
                 "F 是力（N），M 是弯矩（N·m），全局坐标；case 是工况名"),
                ("member_loads", "杆件均布荷载", MT.MEMBER_LOAD_COLUMNS,
                 "满跨均布，w 是全局坐标下的线荷载（N/m）。竖直向下就是 wz 填负值"),
                ("member_spans", "梯形与集中荷载", MT.SPAN_LOAD_COLUMNS,
                 "kind=trapezoid 时 w1 是 i 端强度、w2 是 j 端强度（N/m，三角形把一端填 0）；"
                 "kind=point 时 w1 是集中力（N）、a 是距 i 端的距离（m）"),
                ("settlements", "支座沉降", MT.SETTLEMENT_COLUMNS,
                 "给定位移（m 与 rad），不是荷载。只有被约束住的方向才生效；"
                 "向下沉降 10 mm 在 uz 填 -0.01"),
            )

            _COLUMNS = {name: cols for name, _, cols, _ in _EDITOR_SPECS}
            _LABEL = {name: label for name, label, _, _ in _EDITOR_SPECS}

            def _forget_editor(name: str) -> None:
                """换掉表格内容时必须连 widget 状态一起丢。

                data_editor 把用户的改动按行号存在 key 里。底下数据换了、key 还留着，
                旧的改动会盖到新数据的同一行上 —— 看起来就是导入之后数据莫名其妙。
                """
                st.session_state.pop(f"editor_{name}", None)

            unit_col, _rest = st.columns([1, 3])
            with unit_col:
                systems = list(MT.UNIT_SYSTEMS)
                now = fem.model.get("units", systems[0])
                pick = st.selectbox(
                    "单位制", systems, index=systems.index(now),
                    help="切换会把整份模型按物理量等价地换算过去——数值全变、"
                         "结果不变。位移仍报 mm、力仍报 kN。")
            if pick != now:
                converted = MT.convert_model(fem.model, pick)
                if validate_payload(converted):
                    st.error("换算后的模型不合法，已保留原模型")
                else:
                    fem.set_model(model=converted)
                    st.session_state.solve_result = fem.solve_model()
                    st.session_state.edit_tables = MT.to_tables(fem.model)
                    for _n in MT.TABLE_NAMES:
                        st.session_state.pop(f"editor_{_n}", None)
                    st.session_state.pop("cmp_result", None)
                    st.rerun()

            current = MT.to_tables(fem.model)
            pending = st.session_state.setdefault("edit_tables", current)

            up = st.file_uploader("从 CSV 导入一张表（按表头自动识别是哪一张）", type="csv")
            if up is not None and st.session_state.get("edit_csv") != up.file_id:
                text = up.getvalue().decode("utf-8-sig")
                got = {h.strip() for h in (text.splitlines() or [""])[0].split(",")}
                hit = next((n for n, cols in _COLUMNS.items() if set(cols) <= got), None)
                st.session_state["edit_csv"] = up.file_id
                if hit is None:
                    st.error("认不出这是哪张表。表头需要和下面某张表的列名一致。")
                else:
                    try:
                        pending[hit] = MT.from_csv(text, _COLUMNS[hit])
                    except MT.TableError as exc:
                        st.error(str(exc))
                    else:
                        _forget_editor(hit)
                        st.success(f"已导入{_LABEL[hit]}表（{len(pending[hit])} 行），"
                                   "点下面的「应用修改并重算」才会生效")
                        st.rerun()

            edited: dict[str, list[dict]] = {}
            for name, label, columns, help_text in _EDITOR_SPECS:
                st.markdown(f"**{label}**")
                st.caption(help_text)
                frame = pd.DataFrame(pending[name], columns=list(columns))
                out = st.data_editor(frame, num_rows="dynamic", width='stretch',
                                     hide_index=True, key=f"editor_{name}",
                                     height=min(360, 40 * (len(frame) + 2) + 40))
                edited[name] = out.to_dict("records")
                # 导出的是屏幕上这一刻的内容，不是模型里的旧内容 ——
                # 改完再导出却拿到改前的数据，是最容易让人白干一轮的那种坑
                st.download_button(f"导出{label} CSV", MT.to_csv(edited[name], columns),
                                   file_name=f"{name}.csv", mime="text/csv",
                                   key=f"dl_{name}")

            apply_col, reset_col, _ = st.columns([2, 2, 4])
            if apply_col.button("应用修改并重算", type="primary", width='stretch'):
                try:
                    new_model = MT.from_tables(fem.model, edited)
                except MT.TableError as exc:
                    st.error(str(exc))          # 读不成：报出是哪张表哪一行
                else:
                    # 先验后交。set_model 是"先存下再报错"——那是给 Agent 用的：
                    # 模型要能拿着坏模型逐条改。界面的契约不一样，驳回就该什么都没变，
                    # 否则用户点一次"应用"就把手上能算的模型弄丢了。
                    errors = validate_payload(new_model)
                    if errors:
                        st.error("模型没通过校验，已保留原模型")
                        for e in errors:
                            st.write("·", e)
                    else:
                        fem.set_model(model=new_model)
                        st.session_state.edit_tables = MT.to_tables(fem.model)
                        st.session_state.solve_result = fem.solve_model()
                        # 模型变了，上一次的 Abaqus 对比结果就作废了，别留在页面上误导人
                        st.session_state.pop("cmp_result", None)
                        for name in _COLUMNS:
                            _forget_editor(name)
                        st.rerun()
            if reset_col.button("放弃修改", width='stretch'):
                st.session_state.edit_tables = MT.to_tables(fem.model)
                for name in _COLUMNS:
                    _forget_editor(name)
                st.rerun()

    with sub_json:
        if not has_model:
            st.info("还没有模型。")
        else:
            st.caption("这份 JSON 就是大模型要产出的东西 —— 界面和 Agent 走同一个契约")
            st.code(json.dumps(fem.model, ensure_ascii=False, indent=2), language="json")
            st.download_button("下载模型 JSON",
                               json.dumps(fem.model, ensure_ascii=False, indent=2),
                               file_name="model.json", mime="application/json")

fem = _session()
result = st.session_state.get("solve_result")

# ------------------------------------------------------------------ 状态条

case = None
if result is not None and result.ok:
    cases = result.payload["cases"]
    default_case = st.session_state.get("agent_case") or (
        "DL" if "DL" in cases else next(iter(cases)))
    case = default_case

with status_slot:
    if result is None:
        st.info("从「① 前处理」开始：填参数生成，或用一句话描述结构。")
    elif not result.ok:
        st.error("求解未通过 —— 这正是「算前守门」在起作用，详见「② 分析」")
    else:
        # 先选工况再取数。反过来的话，切换工况那一次刷新里，
        # 指标卡和徽章会一半是新工况、一半还是旧工况的
        all_cases = list(result.payload["cases"])
        k0, k1, k2, k3, k4 = st.columns(5)
        with k0:
            if len(all_cases) > 1:
                case = st.selectbox("工况 / 组合", all_cases,
                                    index=all_cases.index(case))
            else:
                st.metric("工况", case)
        info = result.payload["cases"][case]
        disp = fem.query_results(what="max_displacement", case=case).payload
        reac = fem.query_results(what="reactions", case=case).payload
        ok = info["equilibrium_ok"]
        k1.metric("节点 / 杆件", f"{len(fem.frame.nodes)} / {len(fem.frame.members)}")
        k2.metric("最大合位移", f"{info['max_displacement_mm']:.4f} mm",
                  f"节点 {disp['node']}", delta_color="off")
        k3.metric("竖向反力合计", f"{reac['vertical_total_kN']:.2f} kN")
        k4.metric("静力平衡", "通过" if ok else "未通过",
                  f"残差 {info['equilibrium_residual']:.1e}", delta_color="off")
        st.markdown(
            f'<div style="margin:.5rem 0 1.2rem">'
            f'<span class="badge" style="border-color:{T.GOOD if ok else T.CRITICAL};'
            f'color:{T.GOOD if ok else T.CRITICAL}">'
            f'{"✓" if ok else "✕"} 静力平衡{"通过" if ok else "未通过"}</span>　'
            f'<span class="badge">{_model_summary(fem, case)}</span></div>',
            unsafe_allow_html=True)
        if info.get("warning"):
            st.warning(info["warning"])

        # 静默失败检测摘要 + 实验胶囊链接（solve_model 自动跑的两项质检）
        _sf = result.payload.get("silent_failures")
        if _sf and _sf.get("findings"):
            _bad = [f for f in _sf["findings"] if f["status"] != "pass"]
            _color = T.CRITICAL if any(f["severity"] == "critical" for f in _bad) else (
                T.WARN if _bad else T.GOOD)
            _text = f'{"✕" if _bad else "✓"} 静默检测 {len(_sf["findings"]) - len(_bad)}/{len(_sf["findings"])} 通过'
            if _bad:
                _text += f'（{len(_bad)} 项异常，见求解状态页）'
            st.markdown(
                f'<div style="margin:.2rem 0 1rem">'
                f'<span class="badge" style="border-color:{_color};color:{_color}">{_text}</span>',
                unsafe_allow_html=True)
        _cap = result.payload.get("capsule")
        if _cap:
            st.caption(f"实验胶囊已存档：{_cap}（可用 capsule diff 对比历史运行）")

# ------------------------------------------------------------------ 分析

with tab_run:
    if result is None:
        st.info("还没有求解结果。先到「① 前处理」建模。")
    elif not result.ok:
        st.error("求解未通过 —— 这正是「算前守门」在起作用")
        for _key in ("error", "errors", "hint"):
            if _key in result.payload:
                st.write(result.payload[_key])
        if result.payload.get("diagnosis"):
            st.subheader("奇异诊断")
            for _mode in result.payload["diagnosis"]:
                _who = "、".join(f"节点 {p['node']} {p['direction']}"
                                 for p in _mode["participants"][:4])
                st.warning(f"检出刚体模态：{_who}")
    else:
        sub_status, sub_cmp, sub_opt = st.tabs(["求解状态", "求解器对比", "截面优化"])
        with sub_status:
            st.markdown("**自研求解器**")
            st.caption("模型先过两级校验（结构 + 语义）才求解；刚度矩阵奇异会被"
                       "主元检查拦下并定位到具体节点方向。上面的图与内力都出自这里。")
            _rc1, _rc2, _rc3 = st.columns([2, 2, 4])
            if _rc1.button("生成分析报告", type="secondary", width='stretch',
                           help="模型、校验、位移、反力、控制组合、模态、稳定，"
                                "外加变形图与内力图，写到 results/ 目录"):
                with st.spinner("正在汇总并出图……"):
                    _rep = fem.write_report(case=case)
                if _rep.ok:
                    st.session_state.report_result = _rep.payload
                else:
                    st.session_state.report_result = None
                    st.error(_rep.payload.get("error", "报告生成失败"))
            _got = st.session_state.get("report_result")
            if _got:
                st.success("报告已生成：" + "、".join(_got["files"].values()))
                for _label, _path in _got["files"].items():
                    _p = Path(_path)
                    if _p.exists():
                        _rc2.download_button(
                            f"下载 {_p.suffix.lstrip('.')}", _p.read_bytes(),
                            file_name=_p.name, width='stretch',
                            key=f"dl_report_{_label}")
                if _got.get("partial"):
                    st.warning("部分格式没写成：" + str(_got["partial"]))
                if _got.get("skipped"):
                    st.caption("报告里明确跳过的内容：" + "；".join(
                        f"{k}（{v}）" for k, v in _got["skipped"].items()))
            st.divider()

            _rows = [{"工况 / 组合": name,
                      "最大合位移 (mm)": v["max_displacement_mm"],
                      "静力平衡": "通过" if v["equilibrium_ok"] else "未通过",
                      "平衡残差": f"{v['equilibrium_residual']:.2e}"}
                     for name, v in result.payload["cases"].items()]
            st.dataframe(pd.DataFrame(_rows), hide_index=True, width='stretch',
                         column_config={"最大合位移 (mm)": _MM})

            # 静默失败检测详细结果（solve_model 自动跑的 8 项质检）
            _sf = result.payload.get("silent_failures")
            if _sf and _sf.get("findings"):
                st.markdown("**静默失败检测**")
                st.caption("求解能跑通不代表结果可信——这 8 项检查把"
                           "\"算出来了但可疑\"的情况挑出来。异常项点开看修复建议。")
                _sf_rows = []
                for _f in _sf["findings"]:
                    _status_map = {"pass": "✓ 通过", "fail": "✕ 异常", "warn": "⚠ 警告"}
                    _sev_map = {"critical": "严重", "warning": "警告", "info": "提示"}
                    _sf_rows.append({
                        "检测项": _f["name"],
                        "级别": _sev_map.get(_f["severity"], _f["severity"]),
                        "结果": _status_map.get(_f["status"], _f["status"]),
                        "说明": _f["message"],
                    })
                st.dataframe(pd.DataFrame(_sf_rows), hide_index=True, width='stretch')
                _abnormal = [f for f in _sf["findings"] if f["status"] != "pass"]
                if _abnormal:
                    with st.expander(f"异常项修复建议（{len(_abnormal)} 项）", expanded=False):
                        for _f in _abnormal:
                            st.markdown(f"**{_f['name']}** — {_f['message']}")
                            if _f.get("suggestion"):
                                st.info(_f["suggestion"])
                            if _f.get("detail"):
                                st.caption(f"详情：{_f['detail']}")
            # 实验胶囊信息
            _cap = result.payload.get("capsule")
            if _cap:
                st.caption(f"实验胶囊：{_cap} — 完整输入与结果摘要已存档，"
                           f"可用 `python -m capsule list/show/diff` 查看或对比")

            st.markdown("**控制组合**")
            st.caption("各内力分量的全结构最不利点，以及由哪个组合控制。"
                       "定义了组合就用组合，否则用工况——组合才是设计校核的对象。")
            try:
                from envelope import governing_summary
                _sum = governing_summary(fem.frame, fem.solution, stations=101)
            except (ImportError, ValueError) as _exc:
                st.info(f"暂时算不了包络：{_exc}")
            else:
                _U = fem.units
                _labels = {"N": "轴力 N", "Vy": "剪力 Vy", "Vz": "剪力 Vz",
                           "T": "扭矩 T", "My": "弯矩 My", "Mz": "弯矩 Mz"}
                _rows = []
                for _c, _w in _sum["worst"].items():
                    _moment = _c in ("T", "My", "Mz")
                    _sc = _U.moment_scale if _moment else _U.force_scale
                    _rows.append({
                        "分量": _labels[_c],
                        "最不利值": round(_w["value"] * _sc, 4),
                        "单位": _U.moment_unit if _moment else _U.force_unit,
                        "杆件": _w["member"],
                        "位置 x (m)": round(_w["x"], 3),
                        "控制组合": _w["case"]})
                st.dataframe(pd.DataFrame(_rows), hide_index=True, width='stretch')
                if _sum["never_governs_anywhere"]:
                    st.warning(
                        "这些组合在任何杆件的任何一点都不控制："
                        + "、".join(_sum["never_governs_anywhere"])
                        + "。要么可以删掉，要么它们的系数或荷载写错了。")
                st.caption("注意「拿到全局极值」和「在某处控制」不是一回事："
                           "一个组合可能在某根梁端部说了算，却不是全结构最大的那点。"
                           "上表只列全局极值，逐点的控制情况在后处理的内力图页看。")
        with sub_cmp:
            st.markdown("**同一个模型，两个后端各算一遍**")
            st.caption("上面所有的图与内力都由自研求解器给出。这一页把同一份模型导成 "
                       "Abaqus 输入文件再算一次，逐分量比位移与反力 —— 结论是"
                       "「自研的结果对不对」，不是换一个后端来出图。")

            try:
                import abaqus_backend
                have_abaqus = abaqus_backend.abaqus_available()
            except ImportError:
                abaqus_backend, have_abaqus = None, False

            left, right = st.columns([3, 2])
            element = left.radio(
                "Abaqus 单元格式", ["B33", "B31"], horizontal=True,
                captions=["Euler-Bernoulli，与本程序同一套理论 —— 对标用这个",
                          "Timoshenko，含剪切变形 —— 杆件越粗差得越多"],
                disabled=not have_abaqus)
            go = right.button("用 Abaqus 再算一遍", type="primary",
                              width='stretch', disabled=not have_abaqus)

            if not have_abaqus:
                st.info("本机没有找到 Abaqus，这一页只能看不能跑。"
                        "自研求解器不依赖它，其余功能不受影响。")
                st.caption("如果这台机器其实装了 Abaqus：它只把 abaqus 命令放进开始菜单里那个 "
                           "「Abaqus Command」终端的 PATH，从普通命令行启动本程序是找不到的。"
                           "换到那个终端里执行 run_gui.bat 即可。"
                           "　已有的离线对标结果见 abaqus_bench/benchmark_B33.md 与 benchmark_B31.md。")

            if go:
                with st.spinner("Abaqus 求解中，通常几十秒……"):
                    st.session_state.cmp_result = fem.compare_solvers(case=case, element=element)

            cmp_res = st.session_state.get("cmp_result")
            if cmp_res is not None:
                if not cmp_res.ok:
                    st.error(cmp_res.payload.get("error", "对比失败"))
                    if cmp_res.payload.get("detail"):
                        st.write(cmp_res.payload["detail"])
                else:
                    p = cmp_res.payload
                    worst = p["worst_error"]
                    # B33 同理论，1e-5 以上说明有实质分歧；B31 本来就该差，不设阈值
                    good = p["element"] == "B31" or (worst is not None and worst < 1e-5)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("参与比较的节点", p["nodes_compared"])
                    m2.metric("最大分量偏差",
                              "—" if worst is None else f"{worst:.2e}")
                    m3.metric("峰值位移 自研 / Abaqus",
                              f"{p['peak_displacement']['native_mm']:.4f} mm",
                              f"Abaqus {p['peak_displacement']['abaqus_mm']:.4f} mm",
                              delta_color="off")
                    st.markdown(
                        f'<div style="margin:.4rem 0 1rem"><span class="badge" '
                        f'style="border-color:{T.GOOD if good else T.CRITICAL};'
                        f'color:{T.GOOD if good else T.CRITICAL}">'
                        f'{"✓ 两个后端一致" if good else "✕ 偏差超出预期"}</span>　'
                        f'<span class="badge">{p["element"]}　工况 {p["case"]}</span></div>',
                        unsafe_allow_html=True)

                    rows = [{"分量": k,
                             "归一化偏差": ("参考解为零" if v is None else f"{v:.3e}")}
                            for k, v in p["errors"].items()]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
                    st.caption(p["metric"])
                    st.info(p["note"])

        with sub_opt:
            st.markdown("**多参数多目标截面优化**")
            st.caption("同时变化多个截面参数（高度/翼缘宽/腹板厚/翼缘厚），"
                       "同时优化多个目标（最小重量/最小位移/最小成本），"
                       "满足约束（应力/位移/最小面积）。多目标时返回 Pareto 前沿。")

            _sections = fem.model.get("sections", [])
            if not _sections:
                st.info("当前模型没有截面。先到「前处理」建模。")
            else:
                _sec_names = [s.get("name", "") for s in _sections]
                _opt_c1, _opt_c2, _opt_c3 = st.columns([1, 1, 2])
                _opt_sec = _opt_c1.selectbox("优化目标截面", _sec_names, key="opt_sec")
                _opt_type = _opt_c2.selectbox("截面类型",
                                               ["工字形 / H 型钢", "矩形", "圆管", "实心圆"],
                                               key="opt_type")
                _opt_algo = _opt_c3.selectbox("搜索算法",
                                               ["grid_search（网格搜索，确定可复现）",
                                                "random_search（随机搜索，适合多参数）"],
                                               key="opt_algo")

                # 变量范围配置
                st.markdown("**变量范围**")
                _var_params = {
                    "工字形 / H 型钢": ["height", "flange_width", "web_thickness", "flange_thickness"],
                    "矩形": ["width", "height"],
                    "圆管": ["outer_diameter", "thickness"],
                    "实心圆": ["diameter"],
                }[_opt_type]
                _var_defaults = {
                    "height": (0.2, 1.0, 5), "flange_width": (0.1, 0.4, 4),
                    "web_thickness": (0.005, 0.02, 3), "flange_thickness": (0.008, 0.03, 3),
                    "width": (0.1, 0.5, 4), "outer_diameter": (0.1, 0.5, 4),
                    "thickness": (0.005, 0.02, 3), "diameter": (0.1, 0.5, 5),
                }
                _variables = {}
                _var_cols = st.columns(len(_var_params))
                for _i, _vp in enumerate(_var_params):
                    with _var_cols[_i]:
                        _dmin, _dmax, _dnum = _var_defaults.get(_vp, (0.1, 0.5, 3))
                        _vmin = st.number_input(f"{_vp} min", value=_dmin, format="%.4f",
                                                 key=f"opt_{_vp}_min")
                        _vmax = st.number_input(f"{_vp} max", value=_dmax, format="%.4f",
                                                 key=f"opt_{_vp}_max")
                        _vnum = st.slider(f"{_vp} 点数", 2, 10, _dnum, key=f"opt_{_vp}_num")
                        _variables[_vp] = (_vmin, _vmax, _vnum)

                # 目标和约束
                _obj_c1, _obj_c2 = st.columns(2)
                with _obj_c1:
                    st.markdown("**优化目标（可多选）**")
                    _obj_weight = st.checkbox("最小重量", value=True, key="obj_weight")
                    _obj_disp = st.checkbox("最小最大位移", value=True, key="obj_disp")
                    _obj_cost = st.checkbox("最小材料成本", value=False, key="obj_cost")
                with _obj_c2:
                    st.markdown("**约束**")
                    _use_disp_limit = st.checkbox("最大位移限值", value=False, key="use_disp_limit")
                    _disp_limit = st.number_input("位移限值 (m)", value=0.02, format="%.4f",
                                                   disabled=not _use_disp_limit, key="disp_limit")
                    _use_area_limit = st.checkbox("最小截面面积", value=False, key="use_area_limit")
                    _area_limit = st.number_input("最小面积 (m²)", value=0.005, format="%.5f",
                                                   disabled=not _use_area_limit, key="area_limit")

                _objectives = []
                if _obj_weight: _objectives.append("weight")
                if _obj_disp: _objectives.append("max_displacement")
                if _obj_cost: _objectives.append("material_cost")
                _constraints = {}
                if _use_disp_limit: _constraints["max_displacement_limit"] = _disp_limit
                if _use_area_limit: _constraints["min_section_area"] = _area_limit

                if st.button("运行截面优化", type="primary", key="opt_run",
                             disabled=not _objectives):
                    with st.spinner("正在遍历参数组合并求解……"):
                        try:
                            _optimizer = SectionOptimizer(
                                model=fem.model,
                                section_name=_opt_sec,
                                section_type=_opt_type,
                                variables=_variables,
                                objectives=_objectives,
                                constraints=_constraints,
                            )
                            if "random" in _opt_algo:
                                _n_random = 1  # 网格搜索的总组合数作为随机采样数
                                for _v in _variables.values():
                                    _n_random *= _v[2]
                                _opt_result = _optimizer.random_search(n_samples=_n_random)
                            else:
                                _opt_result = _optimizer.grid_search()
                            st.session_state.opt_result = _opt_result
                        except Exception as _exc:
                            st.error(f"优化失败：{type(_exc).__name__}: {_exc}")
                            st.session_state.opt_result = None

                _opt_res = st.session_state.get("opt_result")
                if _opt_res is not None:
                    st.divider()
                    _or1, _or2, _or3, _or4 = st.columns(4)
                    _or1.metric("评估组合数", _opt_res.total_calls)
                    _or2.metric("可行解数", _opt_res.feasible_count)
                    _or3.metric("Pareto 前沿点数", len(_opt_res.pareto))
                    _or4.metric("耗时", f"{_opt_res.duration_ms:.0f}ms")

                    # 最优解：单目标取最优，多目标取 Pareto 前沿
                    _best_pts = _opt_res.pareto if len(_objectives) > 1 else sorted(
                        [e for e in _opt_res.evaluations if e.feasible],
                        key=lambda e: e.objectives.get(_objectives[0], float("inf"))
                    )[:5]
                    if _best_pts:
                        st.markdown("**最优解**" + ("（Pareto 前沿）" if len(_objectives) > 1 else ""))
                        _best_rows = []
                        for _pt in _best_pts:
                            _row = {"截面参数": ", ".join(f"{k}={v:.4g}" for k, v in _pt.params.items())}
                            for _obj in _objectives:
                                _val = _pt.objectives.get(_obj)
                                _row[_obj] = f"{_val:.4g}" if _val is not None else "—"
                            _row["可行"] = "✓" if _pt.feasible else "✗"
                            _best_rows.append(_row)
                        st.dataframe(pd.DataFrame(_best_rows), hide_index=True, width='stretch')

                    # Pareto 前沿散点图（双目标时）
                    if len(_objectives) >= 2 and len(_opt_res.pareto) > 1:
                        st.markdown(f"**Pareto 前沿（{_objectives[0]} vs {_objectives[1]}）**")
                        _pareto_x = [p.objectives.get(_objectives[0], 0) for p in _opt_res.pareto]
                        _pareto_y = [p.objectives.get(_objectives[1], 0) for p in _opt_res.pareto]
                        _all_x = [p.objectives.get(_objectives[0], 0) for p in _opt_res.evaluations]
                        _all_y = [p.objectives.get(_objectives[1], 0) for p in _opt_res.evaluations]
                        _chart_df = pd.DataFrame({
                            _objectives[0]: _all_x + _pareto_x,
                            _objectives[1]: _all_y + _pareto_y,
                            "类型": ["所有可行解"] * len(_all_x) + ["Pareto 前沿"] * len(_pareto_x),
                        })
                        st.scatter_chart(_chart_df, x=_objectives[0], y=_objectives[1],
                                         color="类型", width='stretch')

                    with st.expander("查看所有评估点"):
                        _all_rows = []
                        for _pt in _opt_res.evaluations:
                            _row = {"截面参数": ", ".join(f"{k}={v:.4g}" for k, v in _pt.params.items())}
                            for _obj in _objectives:
                                _val = _pt.objectives.get(_obj)
                                _row[_obj] = f"{_val:.4g}" if _val is not None else "—"
                            _row["可行"] = "✓" if _pt.feasible else "✗"
                            _all_rows.append(_row)
                        st.dataframe(pd.DataFrame(_all_rows), hide_index=True, width='stretch')

# ------------------------------------------------------------------ 后处理

with tab_post:
    if result is None or not result.ok:
        st.info("还没有可看的结果。先到「① 前处理」建模、在「② 分析」里确认求解通过。")
    else:
        sub_view, sub_force, sub_num = st.tabs(["三维视图", "内力图", "数值结果"])
        with sub_view:
            auto = figure_deformed(fem.frame, fem.solution, case)[1]["scale"]
            bar_kind, bar_scale, bar_export = st.columns([2, 3, 1])
            with bar_kind:
                kind = st.radio("显示", ["变形图", "轴力图"], horizontal=True,
                                label_visibility="collapsed")
            with bar_scale:
                scale = st.slider("变形放大倍数", 1.0, float(auto) * 3.0, float(auto),
                                  step=max(1.0, float(auto) / 50),
                                  disabled=(kind != "变形图"))
            with bar_export:
                st.write("")
                if st.button("导出 PNG", width='stretch', help="报告用的静态图，写入 results/"):
                    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
                    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in case)
                    plot_deformed(fem.frame, fem.solution, case, out_dir / f"deformed_{safe}.png")
                    plot_axial(fem.frame, fem.solution, case, out_dir / f"axial_{safe}.png")
                    st.toast(f"已写入 results/deformed_{safe}.png 与 axial_{safe}.png")

            _opt = st.columns([1, 1, 4])
            _overlay = _opt[0].checkbox("叠加未变形", value=True,
                                        help="虚线是原始位置。关掉之后只看变形后的形状，"
                                             "适合结构很密、虚线成了干扰的时候")
            _sym = _opt[1].checkbox("显示约束符号", value=False,
                                    help="支座画成固接方框或铰接三角，"
                                         "与「前处理 · 模型视图」里的一致")

            if kind == "变形图":
                fig, meta = figure_deformed(fem.frame, fem.solution, case, scale=scale)
                if not _overlay:
                    # 第 0 条 trace 是"原始位置"那条虚线
                    fig.data[0].visible = False
                if _sym:
                    for _t in VS.support_traces(fem.frame)[0]:
                        fig.add_trace(_t)
                st.plotly_chart(fig, width='stretch')
                note = (f'<span class="badge">放大 <b>{meta["scale"]:.0f}</b> 倍</span>　'
                        f'<span class="badge">峰值 <b>{meta["max_displacement_mm"]:.4f} mm</b>'
                        f' @ 节点 {meta["at_node"]}</span>')
            else:
                fig, meta = figure_axial(fem.frame, fem.solution, case)
                st.plotly_chart(fig, width='stretch')
                note = (f'<span class="badge">最大 |N| <b>{meta["max_abs_axial_kN"]:.2f} kN</b>'
                        f' @ 杆件 {meta["at_member"]}</span>　'
                        f'<span class="badge">受拉为正，受压为负</span>')
            st.markdown(note + '　<span class="badge">拖动旋转 · 滚轮缩放 · 悬停读数</span>',
                        unsafe_allow_html=True)

        with sub_force:
            st.caption("内力在单元内解析恢复：单跨划一个单元，弯矩图也是精确解，"
                       "不需要为了画图去加密网格。弯矩按工程习惯画在受拉侧。")
            pick, span = st.columns([1, 3])
            with pick:
                label = st.radio("分量", list(_COMPONENTS), index=0)
                comp = _COMPONENTS[label]
                members = sorted(fem.frame.members)
                member = st.selectbox("查看单根杆件", ["（整体）"] + members)
                show_env = st.checkbox(
                    "画各组合包络", value=False,
                    help="逐点取所有组合的上下界，并标出控制组合。"
                         "只对单根杆件有效。")
                if st.button("导出内力图 PNG", width='stretch'):
                    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
                    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in case)
                    info = plot_diagram(fem.frame, fem.solution, comp, case,
                                        out_dir / f"{comp}_{safe}.png")
                    st.toast(f"已写入 {info['path']}")
            with span:
                fig3d, meta = figure_diagram_3d(fem.frame, fem.solution, comp, case)
                st.plotly_chart(fig3d, width='stretch')
                st.markdown(
                    f'<span class="badge">最大 <b>{meta["peak"]:.3f}</b>'
                    f'{" kN·m" if comp in ("T", "My", "Mz") else " kN"}</span>　'
                    f'<span class="badge">杆件 {meta["at_member"]} 的 x = {meta["at_x"]:.3f} m 处</span>',
                    unsafe_allow_html=True)

            if member != "（整体）":
                if show_env:
                    try:
                        fig2d, info = figure_member_envelope(
                            fem.frame, fem.solution, int(member), comp)
                    except ValueError as exc:
                        st.warning(f"画不了包络：{exc}")
                    else:
                        st.plotly_chart(fig2d, width='stretch')
                        st.markdown(
                            f'<span class="badge">杆件 {info["member"]}</span>　'
                            f'<span class="badge">{len(info["cases"])} 个组合</span>　'
                            f'<span class="badge">最不利 <b>{info["peak"]:.3f}</b>'
                            f' @ x = {info["at_x"]:.3f} m</span>　'
                            f'<span class="badge">控制组合 '
                            f'<b>{info["governing_case"]}</b></span>',
                            unsafe_allow_html=True)
                        if len(info["governing_list"]) > 1:
                            st.caption(
                                "这根杆上不止一个组合在控制："
                                + "、".join(info["governing_list"])
                                + "。跨中与端部由不同组合控制是常态，"
                                  "引用结论时要连位置一起说。")
                else:
                    fig2d, info = figure_member_diagram(
                        fem.frame, fem.solution, int(member), comp, case)
                    st.plotly_chart(fig2d, width='stretch')
                    st.markdown(
                        f'<span class="badge">杆件 {info["member"]}　长 {info["length"]:.3f} m</span>　'
                        f'<span class="badge">峰值 <b>{info["peak"]:.3f}</b> @ x = {info["at_x"]:.3f} m</span>',
                        unsafe_allow_html=True)

        with sub_num:
            st.markdown("**最大位移**")
            c = disp["components_mm"]
            st.dataframe(pd.DataFrame([{
                "节点": disp["node"],
                "x (m)": disp["coordinates_xyz"][0], "y (m)": disp["coordinates_xyz"][1],
                "z (m)": disp["coordinates_xyz"][2],
                "ux (mm)": c["ux"], "uy (mm)": c["uy"], "uz (mm)": c["uz"],
                "合位移 (mm)": disp["magnitude_mm"]}]),
                hide_index=True, width='stretch',
                column_config={"x (m)": _M, "y (m)": _M, "z (m)": _M,
                               "ux (mm)": _MM, "uy (mm)": _MM, "uz (mm)": _MM,
                               "合位移 (mm)": _MM})

            st.markdown("**支座反力**")
            st.caption("单位 kN，全局坐标")
            st.dataframe(pd.DataFrame(
                [{"节点": int(nid), "x (m)": v["xyz"][0], "y (m)": v["xyz"][1],
                  "z (m)": v["xyz"][2], "Fx (kN)": v["R"][0], "Fy (kN)": v["R"][1],
                  "Fz (kN)": v["R"][2]} for nid, v in reac["reactions"].items()]),
                hide_index=True, width='stretch',
                column_config={"x (m)": _M, "y (m)": _M, "z (m)": _M,
                               "Fx (kN)": _KN, "Fy (kN)": _KN, "Fz (kN)": _KN})

            st.markdown("**杆端力**")
            forces = fem.query_results(what="member_forces", case=case).payload
            st.caption(f"单位 kN 与 kN·m　·　{forces['sign_convention']}")
            st.dataframe(pd.DataFrame(
                [{"杆件": int(mid), "N (kN)": row["N"],
                  "Mz(i) (kN·m)": row["Mz_i"], "Mz(j) (kN·m)": row["Mz_j"]}
                 for mid, row in forces["all_members"].items()]),
                hide_index=True, width='stretch', height=320,
                column_config={"N (kN)": _KN, "Mz(i) (kN·m)": _KN, "Mz(j) (kN·m)": _KN})


# ------------------------------------------------------------------ 状态栏
# CAE 的 message area：当前模型规模、求解状态、最近一步操作、单位制。
# 切到哪个阶段都在，不用回去翻。

_bits: list[str] = []
if fem.model.get("nodes"):
    _bits.append(f'<span class="k">模型</span> {len(fem.model["nodes"])} 节点 / '
                 f'{len(fem.model["members"])} 杆件')
    _bits.append(f'<span class="k">单位</span> {fem.model.get("units", "N-m-Pa")}')
else:
    _bits.append('<span class="k">模型</span> 空')

if result is None:
    _bits.append('<span class="k">求解</span> 未求解')
elif result.ok:
    _n = len(result.payload["cases"])
    _bad = [k for k, v in result.payload["cases"].items() if not v["equilibrium_ok"]]
    _bits.append(f'<span class="k">求解</span> {_n} 个工况/组合　'
                 + ("全部通过静力平衡" if not _bad else f'{len(_bad)} 个未通过平衡'))
else:
    _bits.append('<span class="k">求解</span> 未通过校验')

if len(fem.history):
    _last = fem.history[len(fem.history) - 1]
    _bits.append(f'<span class="k">最近操作</span> 第 {_last.index + 1} 步　'
                 f'{_last.summary}')

st.markdown('<div class="statusbar">' + "".join(
    f'<span>{b}</span>' for b in _bits) + '</div>', unsafe_allow_html=True)
