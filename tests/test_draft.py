"""草稿模型：允许半成品，但不许放行算不出来的东西。

**这里盯的是两级类型之间的那道缝。**

草稿宽松、正式模型严格，两级各有一套检查——一旦两套判断漂开，就会出现
最难查的一类故障：草稿说「可以提交」，提交后正式校验说「不行」，
而用户面前的图看不出任何问题。所以最后一个测试专门盯这条不变量：
**`can_finish()` 放行的，`validate_payload` 必须也放行。**
"""

from __future__ import annotations

import pytest

from agent import Session
from draft import SNAP_TOL, DraftError, FrameDraft
from model_io import validate_payload
from sections import i_section

MAT = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SEC = [i_section("COLUMN", 0.4, 0.2, 0.008, 0.012),
       i_section("BEAM", 0.5, 0.2, 0.008, 0.014)]


def empty() -> FrameDraft:
    return FrameDraft(carried={"units": "N-m-Pa", "materials": MAT,
                               "sections": SEC})


def solved() -> Session:
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                         beam_section="BEAM", material="Q355")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    s.solve_model()
    return s


def levels(notes) -> set[str]:
    return {n.level for n in notes}


# --------------------------------------------------------- 半成品是允许的

def test_a_half_drawn_model_is_allowed_while_editing():
    """**画到一半不是错误。** 这是整个草稿层存在的理由。"""
    d = empty()
    a = d.add_node(0, 0, 0)
    assert "error" not in levels(d.editing_diagnostics()), "刚点一个点就报错，第一步就走不下去"

    b = d.add_node(0, 0, 3.6)
    d.add_member(a, b, "COLUMN")
    assert "error" not in levels(d.editing_diagnostics()), "还没画支座，编辑期不该算错"

    # 但是提交不了——同一批问题，级别升上来
    assert not d.can_finish()
    assert all(n.level == "error" for n in d.finish_diagnostics())
    assert any("支座" in n.message for n in d.finish_diagnostics())

    d.set_support(a, [1, 1, 1, 1, 1, 1])
    assert d.can_finish()


def test_the_two_levels_talk_about_the_same_problems():
    """编辑期提示的和提交期拦截的，必须是同一批问题的两种语气。

    分成两套各写一遍，早晚会漂成「编辑时说没问题、提交时说不行」。
    """
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    d.add_member(a, b, "COLUMN")
    d.add_node(9, 0, 0)                       # 悬空节点

    editing = {n.message for n in d.editing_diagnostics()}
    finishing = {n.message for n in d.finish_diagnostics()}
    assert finishing <= editing, "提交期拦下了编辑期从没提过的问题"


def test_a_dangling_node_is_a_hint_now_and_an_error_at_submit():
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    d.add_member(a, b, "COLUMN")
    d.set_support(a, [1] * 6)
    lonely = d.add_node(9, 9, 9)

    hint = [n for n in d.editing_diagnostics() if n.target == ("node", lonely)]
    assert hint and hint[0].level == "info"
    assert not d.can_finish()
    assert any(n.target == ("node", lonely) for n in d.finish_diagnostics())


# --------------------------------------------------------- 画错是当场拒绝的

def test_drawing_on_top_of_an_existing_node_reuses_it():
    """**吸附是画图建模最关键的一条。**

    不吸附的话，柱顶和梁端会是两个相距 1e-9 的不同节点，
    图上看是连着的，矩阵里是散的——求解报奇异，而图完全正常。
    """
    d = empty()
    top = d.add_node(0, 0, 3.6)
    assert d.add_node(0, 0, 3.6) == top
    assert d.add_node(0, 0, 3.6 + SNAP_TOL / 10) == top
    assert d.add_node(0, 0, 3.7) != top
    assert len(d.nodes) == 2


def test_zero_length_and_duplicate_members_are_refused_on_the_spot():
    """这两个是画错，不是没画完——留到提交时说，用户已经忘了是哪一步。"""
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    d.add_member(a, b, "COLUMN")

    with pytest.raises(DraftError):
        d.add_member(a, a, "COLUMN")
    with pytest.raises(DraftError, match="已经有杆件"):
        d.add_member(b, a, "BEAM")            # 反向也是同一对节点
    with pytest.raises(DraftError):
        d.add_member(a, 999, "COLUMN")
    assert len(d.members) == 1, "被拒的操作不该留下半个杆件"


def test_deleting_a_node_takes_its_members_with_it():
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    c = d.add_node(6, 0, 3.6)
    m1 = d.add_member(a, b, "COLUMN")
    m2 = d.add_member(b, c, "BEAM")
    d.set_support(a, [1] * 6)

    dropped = d.delete_node(b)
    assert sorted(dropped) == sorted([m1, m2])
    assert d.members == [], "留下引用了不存在节点的杆件，错误会在别处冒出来"

    d.delete_node(a)
    assert d.supports == [], "支座要跟着节点一起走"


# --------------------------------------------------------- 看不出来的那几种错

def test_a_structure_split_in_two_is_flagged():
    """图上两片可能挨得很近，看起来就是连着的。"""
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    d.add_member(a, b, "COLUMN")
    d.set_support(a, [1] * 6)
    c, e = d.add_node(9, 0, 0), d.add_node(9, 0, 3.6)
    d.add_member(c, e, "COLUMN")
    d.set_support(c, [1] * 6)

    notes = d.editing_diagnostics()
    assert any("2 片" in n.message for n in notes)
    assert not any(n.level == "error" for n in notes), "两片各有支座，能算，不是错误"
    assert d.can_finish()


def test_a_floating_part_without_supports_blocks_submission():
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    d.add_member(a, b, "COLUMN")
    d.set_support(a, [1] * 6)
    c, e = d.add_node(9, 0, 0), d.add_node(9, 0, 3.6)
    d.add_member(c, e, "COLUMN")              # 这一片没支座

    assert any("机构" in n.message for n in d.editing_diagnostics())
    assert not d.can_finish()


def test_placing_two_points_is_not_a_split_structure():
    """还没连线时说「结构分成了 2 片」，是在报告一件没发生的事。"""
    d = empty()
    d.add_node(0, 0, 0)
    d.add_node(9, 0, 0)
    assert not any("片" in n.message for n in d.editing_diagnostics())


def test_an_undefined_section_is_an_error_even_while_editing():
    """截面名打错和「还没画完」不是一回事，编辑期就该是红的。"""
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    d.add_member(a, b, "没有这个截面")
    assert any(n.level == "error" and "截面" in n.message
               for n in d.editing_diagnostics())


def test_merging_coincident_nodes_also_kills_the_duplicate_members():
    """重合节点合并后会冒出重复杆件，**刚度翻倍而结果看着完全正常**。"""
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    d.add_member(a, b, "COLUMN")
    # 绕过吸附造两个重合点，模拟从外部导入的模型
    c = d.add_node(0, 0, 0, snap=False)
    e = d.add_node(0, 0, 3.6, snap=False)
    d.add_member(c, e, "COLUMN")
    d.set_support(c, [1, 1, 1, 0, 0, 0])
    d.set_support(a, [1, 1, 1, 1, 1, 1])

    assert len(d.coincident_nodes()) == 2
    report = d.merge_coincident()
    assert report["merged"] == 2
    assert len(report["removed_members"]) == 1, "两根重合的柱要并成一根"
    assert len(d.nodes) == 2 and len(d.members) == 1
    # 两个支座并到一个节点上时取约束更严的那个，不能悄悄放松
    assert d.supports[0]["fix"] == [1, 1, 1, 1, 1, 1]


def test_snapshot_and_restore_make_cancel_possible():
    d = empty()
    a, b = d.add_node(0, 0, 0), d.add_node(0, 0, 3.6)
    d.add_member(a, b, "COLUMN")
    snap = d.snapshot()

    d.add_node(9, 0, 0)
    d.delete_member(1)
    d.restore(snap)

    assert len(d.nodes) == 2 and len(d.members) == 1
    d.nodes[0]["x"] = 99.0
    assert snap["nodes"][0]["x"] == 0.0, "快照必须是深拷贝，否则回退回不去"


# --------------------------------------------------------- 提交这道门

def test_apply_draft_goes_through_the_same_door_as_everything_else():
    """手工画的图必须和 Agent 改的走同一条路：校验、失效、记账、可撤销。"""
    s = solved()
    assert s.solution is not None
    d = s.open_draft()
    top = next(n for n in d.nodes if n["z"] > 3.0)
    n1 = d.add_node(12.0, 0.0, 0.0)
    n2 = d.add_node(12.0, 0.0, 3.6)
    d.add_member(n1, n2, "COLUMN")
    d.add_member(int(top["id"]), n2, "BEAM")
    d.set_support(n1, [1] * 6)

    before = len(s.model["members"])
    r = s.apply_draft(d, note="手工加了一跨")
    assert r.ok
    assert len(s.model["members"]) == before + 2
    assert s.solution is None, "几何变了，旧结果必须作废"
    assert s.history.steps[-1].summary == "手工加了一跨"
    assert s.undo() and len(s.model["members"]) == before


def test_a_whole_drawing_session_is_one_undo_step():
    """画了十根杆，Ctrl+Z 应该退回画之前，而不是退掉最后一根。"""
    s = solved()
    d = s.open_draft()
    steps = len(s.history.steps)
    for k in range(3):
        a = d.add_node(20.0 + k, 0.0, 0.0)
        b = d.add_node(20.0 + k, 0.0, 3.0)
        d.add_member(a, b, "COLUMN")
        d.set_support(a, [1] * 6)
    assert s.apply_draft(d).ok
    assert len(s.history.steps) == steps + 1


def test_an_unfinished_draft_is_committed_as_editing_state_with_reasons():
    s = solved()
    d = s.open_draft()
    d.add_node(30.0, 0.0, 0.0)          # 悬空
    r = s.apply_draft(d)
    assert r.ok
    assert r.payload["warnings"], "草稿提交后仍必须说清楚还缺什么"
    # 中央清理器会删掉不参与拓扑的孤立节点，避免其污染后续编号与求解。
    assert all(node["x"] != 30.0 for node in s.model["nodes"])


def test_deleting_a_loaded_member_in_a_draft_cleans_its_loads():
    """删掉带荷载的杆件，荷载要跟着走——否则校验会在下一步才炸。"""
    s = solved()
    loaded = s.model["load_cases"][0]["member_loads"][0]["member"]
    d = s.open_draft()
    d.delete_member(loaded)
    # 删了梁之后顶上的节点会悬空，一并删掉
    for nid in [int(n["id"]) for n in list(d.nodes)]:
        if not any(int(m["i"]) == nid or int(m["j"]) == nid for m in d.members):
            d.delete_node(nid)
    r = s.apply_draft(d)
    assert r.ok, r.payload
    left = [e["member"] for e in s.model["load_cases"][0]["member_loads"]]
    assert loaded not in left


def test_whatever_can_finish_accepts_the_solver_also_accepts():
    """**这条不变量是两级类型之间那道缝的唯一防线。**

    草稿说能提交、正式校验说不行，是最难查的一类故障：
    用户面前的图看不出任何问题。
    """
    s = solved()
    shapes = []

    d1 = s.open_draft()                                   # 原样
    shapes.append(d1)

    d2 = s.open_draft()                                   # 加一跨
    top = next(n for n in d2.nodes if n["z"] > 3.0)
    a = d2.add_node(12.0, 0.0, 0.0)
    b = d2.add_node(12.0, 0.0, 3.6)
    d2.add_member(a, b, "COLUMN")
    d2.add_member(int(top["id"]), b, "BEAM")
    d2.set_support(a, [1] * 6)
    shapes.append(d2)

    d3 = s.open_draft()                                   # 悬臂
    left_top = min((n for n in d3.nodes if n["z"] > 3.0), key=lambda n: n["x"])
    tip = d3.add_node(-3.0, 0.0, 3.6)
    d3.add_member(int(left_top["id"]), tip, "BEAM")
    shapes.append(d3)

    d4 = empty()                                          # 从零画一个门架
    p1, p2 = d4.add_node(0, 0, 0), d4.add_node(0, 0, 4.0)
    p3, p4 = d4.add_node(8, 0, 4.0), d4.add_node(8, 0, 0)
    d4.add_member(p1, p2, "COLUMN")
    d4.add_member(p2, p3, "BEAM")
    d4.add_member(p3, p4, "COLUMN")
    d4.set_support(p1, [1] * 6)
    d4.set_support(p4, [1] * 6)
    shapes.append(d4)

    for k, d in enumerate(shapes):
        assert d.can_finish(), f"第 {k} 个草稿自己就说不能提交"
        errors = validate_payload(d.to_model())
        assert not errors, f"第 {k} 个草稿骗过了自己的检查：{errors}"


def test_the_invariant_survives_random_drawing_sequences():
    """随机乱画一千次，**只要草稿说能提交，正式校验就必须也通过**。

    上面那几个形状是我手挑的，手挑的例子只能证明我想到的情况没问题。
    这里用固定种子随机画：加点、连杆、设支座、删东西，
    每一步都问一次「现在能提交吗」，说能就当场拿正式校验对一遍。

    反过来那半边不测——草稿可以比求解器更严（比如它拒绝重复杆件），
    严一点只会少放行，不会放行错的。
    """
    import random

    rng = random.Random(20260903)
    grid = [(x * 3.0, y * 3.0, z * 3.0)
            for x in range(3) for y in range(2) for z in range(3)]
    checked = 0

    for _ in range(200):
        d = empty()
        for _ in range(rng.randint(1, 12)):
            move = rng.random()
            try:
                if move < 0.35 or len(d.nodes) < 2:
                    d.add_node(*rng.choice(grid))
                elif move < 0.75:
                    a, b = rng.sample([int(n["id"]) for n in d.nodes], 2)
                    d.add_member(a, b, rng.choice(["COLUMN", "BEAM"]))
                elif move < 0.9:
                    d.set_support(rng.choice([int(n["id"]) for n in d.nodes]),
                                  rng.choice([[1] * 6, [1, 1, 1, 0, 0, 0], []]))
                elif d.members:
                    d.delete_member(rng.choice([int(m["id"]) for m in d.members]))
                else:
                    d.delete_node(rng.choice([int(n["id"]) for n in d.nodes]))
            except DraftError:
                pass                    # 画错被拒是正常的，接着画

            if d.can_finish():
                checked += 1
                errors = validate_payload(d.to_model())
                assert not errors, (
                    f"草稿放行了求解器不收的模型：{errors}\n"
                    f"节点 {d.nodes}\n杆件 {d.members}\n支座 {d.supports}")

    assert checked > 50, f"只有 {checked} 次真正走到可提交状态，这个测试没测到什么"
