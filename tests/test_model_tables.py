"""模型 ↔ 表格的转换。

界面上手改模型，最怕的不是报错，而是**不报错却改错了**：
一行被静悄悄丢掉、重号后一行盖了前一行、释放自由度拼错到组装时才炸、
荷载归错工况。这里逐条盯这些。
"""

import pytest

from agent import Session
from frame3d import LOCAL_DOF_NAMES
from model_io import validate_payload
from model_tables import (MEMBER_COLUMNS, MEMBER_LOAD_COLUMNS, NODAL_LOAD_COLUMNS,
                          NODE_COLUMNS, SETTLEMENT_COLUMNS, SPAN_LOAD_COLUMNS,
                          SUPPORT_COLUMNS, TableError, from_csv, from_tables,
                          to_csv, to_tables)

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7},
]


@pytest.fixture
def session():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6, 3.6], bays=[6.0],
                             column_section="COLUMN", beam_section="BEAM",
                             material="STEEL", beam_release=True).payload["beam_member_ids"]
    # 风荷载要加在自由节点上。加在支座上位移恒为零，这条夹具就废了——
    # 往返测试会在"零 == 零"上通过，什么也没验到。
    top = max(s.model["nodes"], key=lambda n: (n["z"], n["x"]))["id"]
    assert top not in {sup["node"] for sup in s.model["supports"]}
    s.set_load_cases(cases=[
        {"name": "DL", "member_loads": [{"member": m, "w": [0, 0, -20e3]}
                                        for m in beams]},
        {"name": "WL", "nodal_loads": [{"node": top, "load": [5e3, 0, 0, 0, 0, 0]}]},
    ])
    s.solve_model()
    return s


# ------------------------------ 往返 ------------------------------

def test_round_trip_preserves_the_model(session):
    """摊成表再装回去，模型必须一模一样 —— 这是全部编辑功能的地基。"""
    model = session.model
    back = from_tables(model, to_tables(model))
    assert back["nodes"] == model["nodes"]
    assert back["members"] == model["members"]
    assert back["supports"] == model["supports"]
    assert {c["name"] for c in back["load_cases"]} == {
        c["name"] for c in model["load_cases"]}
    for a in back["load_cases"]:
        b = next(c for c in model["load_cases"] if c["name"] == a["name"])
        assert a.get("nodal_loads", []) == b.get("nodal_loads", [])
        assert a.get("member_loads", []) == b.get("member_loads", [])


def test_round_trip_result_still_solves_to_the_same_numbers(session):
    """更硬的一条：往返之后算出来的数必须逐位相同。

    结构对上了不等于数对得上 —— 释放、工况归属这些错法都能骗过结构比对。
    """
    def snapshot() -> dict:
        return {case: session.query_results(what="max_displacement",
                                            case=case).payload
                for case in ("DL", "WL")}

    before = snapshot()
    assert all(s["magnitude_mm"] > 0 for s in before.values()), \
        "两个工况都得算出非零位移，否则这条测试是在比较两个零"
    assert session.set_model(
        model=from_tables(session.model, to_tables(session.model))).ok
    assert session.solve_model().ok
    assert snapshot() == before


def test_round_trip_keeps_member_end_releases(session):
    """梁端铰接是"结构一样、数不一样"的典型来源，单独盯一条。"""
    released = [m for m in session.model["members"] if m.get("releases")]
    assert released, "夹具本身就该带释放，否则这条测试是空的"
    back = from_tables(session.model, to_tables(session.model))
    assert [m for m in back["members"] if m.get("releases")] == released


def test_tables_carry_every_row(session):
    t = to_tables(session.model)
    assert len(t["nodes"]) == len(session.model["nodes"])
    assert len(t["members"]) == len(session.model["members"])
    assert len(t["supports"]) == len(session.model["supports"])
    assert len(t["member_loads"]) == sum(
        len(c.get("member_loads", [])) for c in session.model["load_cases"])


def test_supports_become_booleans_not_a_six_list(session):
    """约束在表格里是六个勾选框；6 个 0/1 让人手填是自找麻烦。"""
    row = to_tables(session.model)["supports"][0]
    assert set(SUPPORT_COLUMNS) == set(row)
    assert all(isinstance(row[d], bool) for d in LOCAL_DOF_NAMES)


def test_top_level_loads_are_normalised_into_a_case():
    """顶层 nodal_loads 读进来记作 default 工况，写回去落在 load_cases 里。

    否则表格里会出现一批"不属于任何工况"的荷载行。
    """
    model = {"units": "N-m-Pa", "materials": MATERIALS, "sections": SECTIONS,
             "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                       {"id": 2, "x": 0, "y": 0, "z": 3}],
             "members": [{"id": 1, "i": 1, "j": 2,
                          "section": "COLUMN", "material": "STEEL"}],
             "supports": [{"node": 1, "fix": [1] * 6}],
             "nodal_loads": [{"node": 2, "load": [1e3, 0, 0, 0, 0, 0]}]}
    tables = to_tables(model)
    assert tables["nodal_loads"][0]["case"] == "default"
    back = from_tables(model, tables)
    assert "nodal_loads" not in back
    assert back["load_cases"][0]["name"] == "default"
    assert back["load_cases"][0]["nodal_loads"] == model["nodal_loads"]
    assert not validate_payload(back)


# ------------------------------ 编辑 ------------------------------

def test_an_edited_coordinate_reaches_the_model(session):
    tables = to_tables(session.model)
    tables["nodes"][0]["z"] = 1.25
    back = from_tables(session.model, tables)
    assert back["nodes"][0]["z"] == 1.25


def test_a_new_row_becomes_a_new_member(session):
    tables = to_tables(session.model)
    n = len(tables["members"])
    tables["members"].append({"id": 999, "i": 1, "j": 2, "section": "BEAM",
                              "material": "STEEL", "releases_i": "",
                              "releases_j": ""})
    back = from_tables(session.model, tables)
    assert len(back["members"]) == n + 1
    assert back["members"][-1]["id"] == 999


def test_blank_rows_are_dropped_not_parsed(session):
    """data_editor 新增的空行会带着一行 None 过来，不该报错也不该进模型。"""
    tables = to_tables(session.model)
    n = len(tables["nodes"])
    tables["nodes"].append(dict.fromkeys(NODE_COLUMNS))
    tables["nodes"].append({c: "" for c in NODE_COLUMNS})
    tables["nodes"].append({c: float("nan") for c in NODE_COLUMNS})
    assert len(from_tables(session.model, tables)["nodes"]) == n


def test_materials_and_sections_survive_editing(session):
    """表格里没有材料和截面，它们必须原样带过来，不能被清空。"""
    back = from_tables(session.model, to_tables(session.model))
    assert back["materials"] == session.model["materials"]
    assert back["sections"] == session.model["sections"]


def test_combos_referring_to_a_deleted_case_are_dropped():
    """删掉一个工况，引用它的组合必须一起走，否则模型直接非法。"""
    base = {"units": "N-m-Pa", "materials": MATERIALS, "sections": SECTIONS,
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                      {"id": 2, "x": 0, "y": 0, "z": 3}],
            "members": [{"id": 1, "i": 1, "j": 2,
                         "section": "COLUMN", "material": "STEEL"}],
            "supports": [{"node": 1, "fix": [1] * 6}],
            "load_cases": [{"name": "DL", "nodal_loads":
                            [{"node": 2, "load": [0, 0, -1e3, 0, 0, 0]}]},
                           {"name": "WL", "nodal_loads":
                            [{"node": 2, "load": [1e3, 0, 0, 0, 0, 0]}]}],
            "combos": [{"name": "1.3DL", "factors": {"DL": 1.3}},
                       {"name": "DL+WL", "factors": {"DL": 1.3, "WL": 1.5}}]}
    tables = to_tables(base)
    tables["nodal_loads"] = [r for r in tables["nodal_loads"] if r["case"] != "WL"]
    back = from_tables(base, tables)
    assert [c["name"] for c in back["combos"]] == ["1.3DL"]
    assert not validate_payload(back)


# ------------------------------ 错误 ------------------------------

def test_a_duplicate_node_id_is_refused(session):
    """重号不拦，后一行会静悄悄盖掉前一行 —— 模型看着还合法。"""
    tables = to_tables(session.model)
    tables["nodes"].append(dict(tables["nodes"][0]))
    with pytest.raises(TableError, match="重复"):
        from_tables(session.model, tables)


def test_a_duplicate_member_id_is_refused(session):
    tables = to_tables(session.model)
    tables["members"].append(dict(tables["members"][0]))
    with pytest.raises(TableError, match="重复"):
        from_tables(session.model, tables)


def test_text_in_a_number_column_names_the_row(session):
    tables = to_tables(session.model)
    tables["nodes"][2]["x"] = "六米"
    with pytest.raises(TableError) as exc:
        from_tables(session.model, tables)
    assert "第 3 行" in str(exc.value) and "x" in str(exc.value)


def test_a_fractional_node_id_is_refused(session):
    tables = to_tables(session.model)
    tables["nodes"][0]["id"] = 1.5
    with pytest.raises(TableError, match="整数"):
        from_tables(session.model, tables)


def test_a_half_filled_row_names_the_empty_column(session):
    """整行空是新增行，半行空是填漏了 —— 后者必须报出来。"""
    tables = to_tables(session.model)
    tables["nodes"].append({"id": 500, "x": 1.0, "y": None, "z": None})
    with pytest.raises(TableError, match="y 是空的"):
        from_tables(session.model, tables)


def test_a_misspelt_release_is_caught_here_not_at_assembly(session):
    """写错的自由度名，等到组装单元刚度才炸的话，错误信息就跟表格对不上了。"""
    tables = to_tables(session.model)
    tables["members"][0]["releases_j"] = "mz"
    with pytest.raises(TableError) as exc:
        from_tables(session.model, tables)
    assert "mz" in str(exc.value) and "rz" in str(exc.value)


def test_a_chinese_comma_between_releases_still_works(session):
    """中文逗号是中文输入法下最容易打出来的分隔符，别为这个报错。"""
    tables = to_tables(session.model)
    tables["members"][0]["releases_j"] = "ry，rz"
    back = from_tables(session.model, tables)
    assert back["members"][0]["releases"]["j"] == ["ry", "rz"]


def test_an_empty_release_cell_leaves_no_key_behind(session):
    """没有释放就不该写 releases 键，否则模型 JSON 里全是空壳。"""
    tables = to_tables(session.model)
    for row in tables["members"]:
        row["releases_i"] = row["releases_j"] = ""
    back = from_tables(session.model, tables)
    assert all("releases" not in m for m in back["members"])


def test_an_edited_model_still_goes_through_the_normal_validation(session):
    """表格层只管读得成读不成；悬空节点这类问题仍由 validate_payload 抓。

    两级校验的分工必须清楚，不能在表格层再抄一遍模型规则。
    """
    tables = to_tables(session.model)
    tables["members"] = tables["members"][:1]
    back = from_tables(session.model, tables)   # 读得成
    assert validate_payload(back)               # 但模型不合法


# ------------------------------ CSV ------------------------------

@pytest.mark.parametrize("name, columns", [
    ("nodes", NODE_COLUMNS), ("members", MEMBER_COLUMNS),
    ("supports", SUPPORT_COLUMNS), ("nodal_loads", NODAL_LOAD_COLUMNS),
    ("member_loads", MEMBER_LOAD_COLUMNS)])
def test_csv_round_trips_every_table(session, name, columns):
    rows = to_tables(session.model)[name]
    back = from_csv(to_csv(rows, columns), columns)
    assert len(back) == len(rows)
    rebuilt = from_tables(session.model, {**to_tables(session.model), name: back})
    original = from_tables(session.model, to_tables(session.model))
    assert rebuilt == original


def test_csv_with_a_missing_column_is_refused():
    """少一列就静默补默认值的话，用户会以为导入成功了。"""
    with pytest.raises(TableError, match="缺少列"):
        from_csv("id,x,y\n1,0,0\n", NODE_COLUMNS)


def test_csv_booleans_accept_the_spellings_people_actually_type():
    text = "node,ux,uy,uz,rx,ry,rz\n1,1,true,是,0,false,\n"
    row = from_csv(text, SUPPORT_COLUMNS)[0]
    assert [row[d] for d in LOCAL_DOF_NAMES] == [True, True, True, False, False, False]


def test_csv_ignores_trailing_blank_lines():
    assert len(from_csv("id,x,y,z\n1,0,0,0\n\n,,,\n", NODE_COLUMNS)) == 1


# ------------------------------ 梯形/集中荷载与沉降 ------------------------------

@pytest.fixture
def rich():
    """带上全部新荷载类型的模型，用于验往返。"""
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "STEEL", "E": 2.1e11, "nu": 0.3, "density": 7850}],
        "sections": SECTIONS,
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 6.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1] * 6}, {"node": 2, "fix": [1] * 6}],
        "load_cases": [{
            "name": "DL",
            "member_spans": [
                {"member": 1, "kind": "point", "w1": [0.0, 0.0, -30e3], "a": 2.0},
                {"member": 1, "kind": "trapezoid", "w1": [0.0, 0.0, 0.0],
                 "w2": [0.0, 0.0, -10e3], "note": "雪荷载"},
                {"member": 1, "kind": "uniform", "w1": [0.0, 0.0, -2e3]},
            ],
            "settlements": [{"node": 2, "d": [0.0, 0.0, -0.01, 0.0, 0.0, 0.0]}],
        }],
    }


def test_span_loads_and_settlements_round_trip(rich):
    back = from_tables(rich, to_tables(rich))
    case = back["load_cases"][0]
    assert case["member_spans"] == rich["load_cases"][0]["member_spans"]
    assert case["settlements"] == rich["load_cases"][0]["settlements"]
    assert not validate_payload(back)


def test_a_uniform_span_keeps_no_w2_or_a(rich):
    """均布项写回去不该带上无意义的 w2 和 a —— 模型 JSON 会变脏，
    而且梯形与均布在表格里就分不清了。"""
    back = from_tables(rich, to_tables(rich))
    uniform = [e for e in back["load_cases"][0]["member_spans"]
               if e["kind"] == "uniform"][0]
    assert "w2" not in uniform and "a" not in uniform


def test_the_note_column_survives(rich):
    back = from_tables(rich, to_tables(rich))
    notes = [e.get("note") for e in back["load_cases"][0]["member_spans"]]
    assert "雪荷载" in notes


def test_an_unknown_span_kind_is_refused(rich):
    tables = to_tables(rich)
    tables["member_spans"][0]["kind"] = "triangle"
    with pytest.raises(TableError, match="无效"):
        from_tables(rich, tables)


def test_a_point_load_without_a_position_is_refused(rich):
    """集中力少了 a，默认成 0 就悄悄变成了作用在 i 端 —— 必须报出来。"""
    tables = to_tables(rich)
    row = next(r for r in tables["member_spans"] if r["kind"] == "point")
    row["a"] = None
    with pytest.raises(TableError, match="a 是空的"):
        from_tables(rich, tables)


def test_settlements_are_their_own_table(rich):
    """沉降是给定位移，不该和荷载混在一张表里。"""
    tables = to_tables(rich)
    assert set(SETTLEMENT_COLUMNS) == set(tables["settlements"][0])
    assert tables["settlements"][0]["uz"] == -0.01


@pytest.mark.parametrize("name, columns", [
    ("member_spans", SPAN_LOAD_COLUMNS), ("settlements", SETTLEMENT_COLUMNS)])
def test_csv_round_trips_the_new_tables(rich, name, columns):
    rows = to_tables(rich)[name]
    back = from_csv(to_csv(rows, columns), columns)
    rebuilt = from_tables(rich, {**to_tables(rich), name: back})
    assert rebuilt == from_tables(rich, to_tables(rich))


def test_settlement_csv_columns_are_numbers_not_checkboxes():
    """约束表和沉降表都有 ux…rz 这几列，但含义完全不同。

    按列名判断布尔列的话，沉降 -0.01 会被当成"假"读成 0 —— 不报错，
    只是悄悄把沉降抹平了。这条盯的就是这个。
    """
    text = "case,node,ux,uy,uz,rx,ry,rz\nDL,2,0,0,-0.01,0,0,0\n"
    row = from_csv(text, SETTLEMENT_COLUMNS)[0]
    assert row["uz"] == "-0.01"
    assert not isinstance(row["uz"], bool)
    # 约束表仍然按勾选框读
    fix = from_csv("node,ux,uy,uz,rx,ry,rz\n1,1,1,1,0,0,0\n", SUPPORT_COLUMNS)[0]
    assert fix["ux"] is True and fix["rx"] is False
