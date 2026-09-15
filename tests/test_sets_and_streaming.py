"""命名集合与对话流式反馈。

两件事回答的是同一类困扰：**用户看不见、也说不出模型内部的编号**。

* **命名集合**——编号规则由生成器决定，谁也不知道 27 号在哪。
  没有名字，"给顶层所有梁加 5 kN/m" 这句话就没有落点。
  集合是纯粹的命名层：求解器看不到它，工具在调用时就展开成编号了。

* **流式反馈**——一轮对话十几秒，期间界面上什么都没有的话用户只能干等。
  逐个报出来不只是"看着不慌"：他能在第三行就发现模型理解错了跨度，
  立刻按停止，而不是等它把报告都写完。
"""

from __future__ import annotations

import os

import pytest

from agent import Session
from sections import i_section

MAT = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SEC = [i_section("COLUMN", .4, .2, .008, .012),
       i_section("BEAM", .5, .2, .008, .014),
       i_section("BRACE", .15, .15, .006, .008)]


def built() -> Session:
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    s.generate_bent(profile=[[0, 7.2], [12, 7.2]], columns=[0, 6, 12],
                    levels=[3.6], column_section="COLUMN",
                    beam_section="BEAM", material="Q355")
    s.extrude_bents(bays=[6, 6], tie_section="BEAM")
    return s


# ------------------------------------------------- 命名集合

def test_a_set_can_be_defined_by_geometry():
    """按几何条件挑，而不是让用户报编号。"""
    s = built()
    r = s.define_set(name="顶层梁", orientation="horizontal",
                     z_range=[7.0, 7.4], note="屋面层")
    assert r.ok, r.payload
    assert r.payload["members"], "一根都没挑到"
    assert s.model["sets"]["顶层梁"]["note"] == "屋面层"


def test_a_set_can_be_defined_by_explicit_ids():
    s = built()
    r = s.define_set(name="两根", member_ids=[1, 2])
    assert r.ok and s.model["sets"]["两根"]["members"] == [1, 2]


def test_an_empty_set_is_refused_not_stored():
    """**空集合不要存。** 存下来之后引用它的地方会静默什么都不做，
    比当场报错难查得多。"""
    s = built()
    r = s.define_set(name="空的", x_range=[100, 200])
    assert not r.ok
    assert "一根杆件都没挑到" in r.payload["error"]
    assert "空的" not in (s.model.get("sets") or {})


def test_unknown_ids_are_refused():
    s = built()
    r = s.define_set(name="乱写", member_ids=[99999])
    assert not r.ok and "不存在" in r.payload["error"]


def test_a_set_name_can_be_used_wherever_ids_are_expected():
    """定义完之后，remove_members / retaper 直接写名字。"""
    s = built()
    s.define_set(name="中柱", orientation="vertical", x_range=[5.9, 6.1],
                 z_range=[0, 3.6])
    n = len(s.model["members"])
    r = s.remove_members(ids="中柱")
    assert len(s.model["members"]) < n
    assert r.payload["removed"]


def test_ids_and_names_can_be_mixed():
    """**允许混着写**：用户脑子里本来就是"那一组，再加这两根"。"""
    s = built()
    s.define_set(name="中柱", orientation="vertical", x_range=[5.9, 6.1],
                 z_range=[0, 3.6])
    expanded = s._expand_members(["中柱", 1])
    assert 1 in expanded and len(expanded) > 1


def test_an_unknown_set_name_lists_what_does_exist():
    """只说"没有这个集合"用户还得自己回想定义过什么。"""
    s = built()
    s.define_set(name="中柱", orientation="vertical", x_range=[5.9, 6.1])
    with pytest.raises(ValueError, match="中柱"):
        s._expand_members(["顶层梁"])


def test_expansion_removes_duplicates_but_keeps_order():
    s = built()
    s.define_set(name="甲", member_ids=[3, 1])
    s.define_set(name="乙", member_ids=[1, 2])
    assert s._expand_members(["甲", "乙"]) == [1, 3, 2]


def test_sets_are_pruned_when_members_are_removed():
    """**不剪的话，集合还指着已经不存在的编号**，下一次引用它时才报错——
    而那时用户早就忘了是哪一步删的。"""
    s = built()
    cols = s.select_members(orientation="vertical",
                            x_range=[5.9, 6.1]).payload["member_ids"]
    s.define_set(name="中柱全段", member_ids=cols)
    s.define_set(name="其中一段", member_ids=cols[:1])
    r = s.remove_members(ids=cols[:1])
    assert r.ok, r.payload
    assert r.payload.get("pruned_sets"), "该报出修剪了哪些集合"
    assert cols[0] not in s.model["sets"]["中柱全段"]["members"]


def test_a_set_emptied_by_deletion_is_dropped_entirely():
    """被删空的集合整个去掉——空集合被引用时会静默什么都不做。"""
    s = built()
    cols = s.select_members(orientation="vertical",
                            x_range=[5.9, 6.1]).payload["member_ids"]
    s.define_set(name="只有这些", member_ids=cols)
    r = s.remove_members(ids=cols)
    assert "只有这些" in (r.payload.get("emptied_sets") or [])
    assert "只有这些" not in (s.model.get("sets") or {})


def test_sets_survive_model_validation():
    """集合存在模型里，必须过 schema。"""
    from model_io import validate_payload

    s = built()
    s.define_set(name="顶层梁", orientation="horizontal", z_range=[7.0, 7.4])
    assert validate_payload(s.model) == []


def test_the_solver_never_sees_sets():
    """**集合是纯粹的命名层。** 求解器只认编号；
    集合混进去会让 from_dict 多认一个它不该认的字段。"""
    from model_io import from_dict

    s = built()
    s.define_set(name="顶层梁", orientation="horizontal", z_range=[7.0, 7.4])
    frame = from_dict(s.model)
    assert not hasattr(frame, "sets")
    beams = s.select_members(orientation="horizontal").payload["member_ids"]
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -15e3]} for m in beams]}])
    assert s.solve_model().ok


def test_listing_sets_reports_sizes_not_the_ids():
    """列清单是给人看的概览，不该刷一屏编号。"""
    s = built()
    s.define_set(name="顶层梁", orientation="horizontal", z_range=[7.0, 7.4])
    r = s.list_sets()
    assert r.ok and r.payload["count"] == 1
    assert isinstance(r.payload["sets"]["顶层梁"]["members"], int)


def test_redefining_a_set_says_it_replaced_one():
    s = built()
    s.define_set(name="甲", member_ids=[1])
    r = s.define_set(name="甲", member_ids=[2])
    assert r.payload["replaced"] is True
    assert s.model["sets"]["甲"]["members"] == [2]


# ------------------------------------------------- 流式反馈

def test_ask_reports_each_tool_as_it_finishes():
    """**每跑完一个就报一次**，不是等一整轮结束才一起给。"""
    from conversation import Conversation
    from desktop.demo_script import demo_provider

    s = Session()
    seen: list[tuple[str, bool]] = []
    chat = Conversation(demo_provider(s), session=s)
    result = chat.ask("建个 24 米门式刚架并求解",
                      on_tool=lambda name, args, ok: seen.append((name, ok)))
    assert len(seen) == len(result.tool_calls), "报出来的条数和实际调用数对不上"
    assert [n for n, _ in seen] == [n for n, _ in result.tool_calls]


def test_the_report_arrives_before_the_turn_ends():
    """这是流式的全部意义：**回调必须在最终回答之前就到**，
    否则用户还是等到最后才看见东西。"""
    from conversation import Conversation
    from desktop.demo_script import demo_provider

    s = Session()
    order: list[str] = []
    chat = Conversation(demo_provider(s), session=s)
    chat.ask("建个 24 米门式刚架并求解",
             on_tool=lambda name, args, ok: order.append("tool"))
    order.append("reply")
    assert order[0] == "tool" and order[-1] == "reply"
    assert order.count("tool") >= 3


def test_a_broken_callback_does_not_break_the_turn():
    """回调是界面的事。它出问题不该把这一轮对话带垮——
    模型已经改过的东西不会因为显示不出来就撤销。"""
    from conversation import Conversation
    from desktop.demo_script import demo_provider

    s = Session()
    chat = Conversation(demo_provider(s), session=s)

    def explodes(*_a, **_k):
        raise RuntimeError("界面这边炸了")

    result = chat.ask("建个 24 米门式刚架并求解", on_tool=explodes)
    assert result.tool_calls, "对话该照常跑完"
    assert s.model.get("nodes"), "模型该照常建起来"


def test_failed_calls_are_reported_too():
    """**模型自己改正错误是常事。** 藏起失败的那几步，
    用户就看不出它试过什么、为什么改了主意。"""
    from conversation import Conversation

    class Stumbles:
        """先调一个必然失败的，再调一个成功的。"""
        def __init__(self):
            self.n = 0

        def complete(self, messages, tools):
            self.n += 1
            if self.n == 1:
                return {"tool_calls": [{"id": "t1", "name": "remove_members",
                                        "arguments": {"ids": [999]}}]}
            if self.n == 2:
                return {"tool_calls": [
                    {"id": "t2", "name": "define_materials_and_sections",
                     "arguments": {"materials": MAT, "sections": SEC}}]}
            return {"content": "好了"}

        seen: list = []

    s = Session()
    seen: list[tuple[str, bool]] = []
    chat = Conversation(Stumbles(), session=s)
    chat.ask("试试", on_tool=lambda n, a, ok: seen.append((n, ok)))
    assert (False in [ok for _, ok in seen]), "失败的调用没报出来"
    assert (True in [ok for _, ok in seen])


# ------------------------------------------------- 界面层的流式
#
# **Qt 的守卫要按测试标记，不能用模块级的 importorskip。**
# 后者在没有 PySide6 的机器上会把**整个模块**跳过——包括上面那些
# 纯逻辑测试。而那些测试本来不需要 Qt，白白少测一半，
# 而且从输出上看只是"skipped"，很难注意到。

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    HAS_QT = True
except ImportError:                                          # pragma: no cover
    HAS_QT = False

needs_qt = pytest.mark.skipif(not HAS_QT, reason="未安装 PySide6")


@pytest.fixture(scope="module")
def qt_app():
    if not HAS_QT:
        pytest.skip("未安装 PySide6")
    yield QApplication.instance() or QApplication([])


@needs_qt
def test_progress_is_delivered_on_the_main_thread(qt_app):
    """**进度必须走信号投递。** 让工作线程直接回调界面就是跨线程碰控件，
    之前已经因此崩过一次（收尾函数在工作线程里等自己）。
    """
    import threading

    from desktop.worker import Runner

    main = threading.get_ident()
    where: list[int] = []
    runner = Runner()

    def work(report):
        for k in range(3):
            report(f"step{k}", {"k": k}, True)
        return "done"

    runner.submit(work, on_progress=lambda *_: where.append(threading.get_ident()))
    assert runner.wait()
    assert len(where) == 3
    assert all(t == main for t in where), "进度回调跑在了工作线程上"


@needs_qt
def test_a_job_without_progress_still_works(qt_app):
    """没给 on_progress 的活儿不该被要求接 report 参数。"""
    from desktop.worker import Runner

    got = []
    runner = Runner()
    runner.submit(lambda: 42, on_done=got.append)
    assert runner.wait()
    assert got == [42]


@needs_qt
def test_the_panel_streams_instead_of_dumping_at_the_end(qt_app, monkeypatch):
    """面板收到进度就画一行；收尾时**不再重复打印**一遍调用链。"""
    from conftest import opengl_available
    if not opengl_available():
        pytest.skip("无可用 OpenGL，MainWindow 的 VTK 视口无法初始化")
    import credentials
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: None)

    import pyvista as pv
    pv.OFF_SCREEN = True
    from desktop.main_window import MainWindow

    w = MainWindow()
    w.chat.entry.setText("建个 24 米门式刚架并求解")
    w.chat.send()
    assert w.runner.wait(60000)
    text = w.chat.view.toPlainText()
    assert w.chat._streamed > 0, "一条流式进度都没收到"
    # 每个中文动作只出现一次——出现两次就是收尾又补打了一遍；内部函数名
    # 只进开发日志，不能重新泄漏到正常界面。
    assert text.count("生成门式刚架") == 1, text
    assert text.count("运行结构求解") == 1
    assert "generate_portal_frame" not in text
    assert "solve_model" not in text
    w.close()
