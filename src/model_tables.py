"""模型 JSON ↔ 平铺表格。

界面上要能逐行改节点、杆件、约束和荷载，而不是只能按参数重新生成一个规整框架。
转换逻辑放在这里、不放在界面里，是因为**出错的地方全在这一层**：
类型转换、空行、重号、释放自由度拼写、荷载归属工况。
这些用 pytest 验起来又快又彻底，塞进 Streamlit 回调里就只能靠手点。

约定两条：

* **单位** 与模型本身一致（看模型的 units 字段），表格里不做任何换算，
  免得"界面显示 kN、存回去还是 kN"这类错误无声无息地混进去。
  要换单位制走 `convert_model`，那是整份模型一起换、物理量不变。
* **工况** 一律归一化成 `load_cases`。模型顶层的 `nodal_loads` / `member_loads`
  读进来记作工况 `default`，写回去时统一落到 `load_cases`，
  这样表格里每一行荷载都明确属于某个工况，不存在"这行到底算哪个工况"的歧义。
* **支座沉降单独一张表**，不和荷载混在一起——它是给定位移，不是荷载。
"""

from __future__ import annotations

import math
from typing import Any

from frame3d import DEFAULT_CASE, LOCAL_DOF_NAMES
from span_loads import KINDS as SPAN_KINDS
from span_loads import POINT, TRAPEZOID
from units import NAMES as UNIT_SYSTEMS
from units import convert_model  # noqa: F401  界面从这里取，不必再 import units

NODE_COLUMNS = ("id", "x", "y", "z")
MEMBER_COLUMNS = ("id", "i", "j", "section", "material", "releases_i", "releases_j")
SUPPORT_COLUMNS = ("name", "node") + LOCAL_DOF_NAMES
NODAL_LOAD_COLUMNS = ("case", "name", "node", "Fx", "Fy", "Fz", "Mx", "My", "Mz")
MEMBER_LOAD_COLUMNS = ("case", "name", "member", "wx", "wy", "wz")
SPAN_LOAD_COLUMNS = ("case", "name", "member", "kind",
                     "w1x", "w1y", "w1z", "w2x", "w2y", "w2z", "a", "note")
SETTLEMENT_COLUMNS = ("case", "name", "node") + LOCAL_DOF_NAMES

TABLE_NAMES = ("nodes", "members", "supports", "nodal_loads", "member_loads",
               "member_spans", "settlements")


# 表名 → 列名。**必须显式给出**：`to_tables` 在表为空时返回空列表，
# 调用方就拿不到列名，只能靠猜——界面要先把空表的表头画出来，
# 用户才知道该往哪一列填什么。
COLUMNS: dict[str, tuple[str, ...]] = {
    "nodes": NODE_COLUMNS,
    "members": MEMBER_COLUMNS,
    "supports": SUPPORT_COLUMNS,
    "nodal_loads": NODAL_LOAD_COLUMNS,
    "member_loads": MEMBER_LOAD_COLUMNS,
    "member_spans": SPAN_LOAD_COLUMNS,
    "settlements": SETTLEMENT_COLUMNS,
}

_LABELS = {"nodes": "节点", "members": "杆件", "supports": "约束",
           "nodal_loads": "节点荷载", "member_loads": "杆件均布荷载",
           "member_spans": "梯形与集中荷载", "settlements": "支座沉降"}


# 哪些列是勾选框，按**表**决定而不是按列名。约束表和沉降表都叫 ux…rz，
# 但前者是 0/1 开关、后者是给定位移。按列名判断会把 -0.01 当成"假"读成 0——
# 不报错，只是悄悄把沉降抹平了
LABELS = _LABELS          # 公开名，界面用


_BOOLEAN_COLUMNS = {SUPPORT_COLUMNS: frozenset(LOCAL_DOF_NAMES)}


class TableError(ValueError):
    """某一行读不成模型。消息里一定带表名和行号。"""


# --------------------------------------------------------------- 模型 → 表格

def _releases_text(member: dict[str, Any], end: str) -> str:
    return ", ".join((member.get("releases") or {}).get(end, []))


def to_tables(model: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """把模型摊成几张平表。空表也返回空列表，不返回 None。"""
    nodes = [{"id": int(n["id"]), "x": float(n["x"]),
              "y": float(n["y"]), "z": float(n["z"])}
             for n in model.get("nodes", [])]

    members = [{"id": int(m["id"]), "i": int(m["i"]), "j": int(m["j"]),
                "section": str(m["section"]), "material": str(m["material"]),
                "releases_i": _releases_text(m, "i"),
                "releases_j": _releases_text(m, "j")}
               for m in model.get("members", [])]

    supports = []
    for s in model.get("supports", []):
        row: dict[str, Any] = {"name": str(s.get("name", "")),
                               "node": int(s["node"])}
        row.update({name: bool(int(v)) for name, v in zip(LOCAL_DOF_NAMES, s["fix"])})
        supports.append(row)

    nodal: list[dict[str, Any]] = []
    member: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    settle: list[dict[str, Any]] = []

    def collect(case_name: str, block: dict[str, Any]) -> None:
        for e in block.get("nodal_loads") or []:
            row = {"case": case_name, "name": str(e.get("name", "")),
                   "node": int(e["node"])}
            row.update({k: float(v) for k, v in
                        zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), e["load"])})
            nodal.append(row)
        for e in block.get("member_loads") or []:
            row = {"case": case_name, "name": str(e.get("name", "")),
                   "member": int(e["member"])}
            row.update({k: float(v) for k, v in
                        zip(("wx", "wy", "wz"), e["w"])})
            member.append(row)
        for e in block.get("member_spans") or []:
            row = {"case": case_name, "name": str(e.get("name", "")),
                   "member": int(e["member"]),
                   "kind": str(e.get("kind", "uniform")),
                   "a": float(e.get("a", 0.0)), "note": str(e.get("note", ""))}
            row.update({k: float(v) for k, v in zip(("w1x", "w1y", "w1z"), e["w1"])})
            row.update({k: float(v) for k, v in
                        zip(("w2x", "w2y", "w2z"), e.get("w2", (0.0, 0.0, 0.0)))})
            spans.append(row)
        for e in block.get("settlements") or []:
            row = {"case": case_name, "name": str(e.get("name", "")),
                   "node": int(e["node"])}
            row.update({k: float(v) for k, v in zip(LOCAL_DOF_NAMES, e["d"])})
            settle.append(row)

    if any(model.get(k) for k in ("nodal_loads", "member_loads",
                                  "member_spans", "settlements")):
        collect(DEFAULT_CASE, model)
    for case in model.get("load_cases") or []:
        collect(str(case["name"]), case)

    return {"nodes": nodes, "members": members, "supports": supports,
            "nodal_loads": nodal, "member_loads": member,
            "member_spans": spans, "settlements": settle}


# --------------------------------------------------------------- 表格 → 模型

def _blank(value: Any) -> bool:
    """空行判定。data_editor 新增的空行会填 None 或 NaN。"""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _row_is_blank(row: dict[str, Any]) -> bool:
    return all(_blank(v) for v in row.values())


def _num(row: dict[str, Any], key: str, table: str, lineno: int) -> float:
    value = row.get(key)
    if _blank(value):
        raise TableError(f"{_LABELS[table]}表第 {lineno} 行：{key} 是空的")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise TableError(
            f"{_LABELS[table]}表第 {lineno} 行：{key}={value!r} 不是数字") from None
    if math.isnan(out) or math.isinf(out):
        raise TableError(f"{_LABELS[table]}表第 {lineno} 行：{key} 不是有限数")
    return out


def _int(row: dict[str, Any], key: str, table: str, lineno: int) -> int:
    out = _num(row, key, table, lineno)
    if out != int(out):
        raise TableError(
            f"{_LABELS[table]}表第 {lineno} 行：{key}={out} 必须是整数")
    return int(out)


def _text(row: dict[str, Any], key: str, table: str, lineno: int) -> str:
    value = row.get(key)
    if _blank(value):
        raise TableError(f"{_LABELS[table]}表第 {lineno} 行：{key} 是空的")
    return str(value).strip()


def _parse_releases(text: Any, table: str, lineno: int, end: str) -> list[str]:
    """'rz' / 'ry, rz' / 空 → 释放自由度列表。

    拼错的自由度名在这里就拦下来，而不是等到组装单元刚度时才炸。
    """
    if _blank(text):
        return []
    names = [t.strip() for t in str(text).replace("，", ",").split(",") if t.strip()]
    bad = [n for n in names if n not in LOCAL_DOF_NAMES]
    if bad:
        raise TableError(
            f"{_LABELS[table]}表第 {lineno} 行：{end} 端释放自由度 {bad} 无效，"
            f"应取自 {', '.join(LOCAL_DOF_NAMES)}")
    if len(set(names)) != len(names):
        raise TableError(f"{_LABELS[table]}表第 {lineno} 行：{end} 端释放自由度有重复")
    return names


def _check_unique(ids: list[int], table: str, what: str) -> None:
    """重号必须在这里拦下。放过去的话后一行会静悄悄覆盖前一行。"""
    seen: set[int] = set()
    dup: set[int] = set()
    for i in ids:
        (dup if i in seen else seen).add(i)
    if dup:
        raise TableError(f"{_LABELS[table]}表里 {what} {sorted(dup)} 重复了")


def from_tables(base: dict[str, Any],
                tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """按表格重建模型。材料、截面、单位、组合从 base 原样带过来。

    只做"读得成不成"的检查；模型本身合不合理（悬空节点、机构等）
    交给 validate_payload，不在这里重复一遍。
    """
    model: dict[str, Any] = {
        "units": base.get("units", "N-m-Pa"),
        "materials": [dict(m) for m in base.get("materials", [])],
        "sections": [dict(s) for s in base.get("sections", [])],
        "nodes": [], "members": [], "supports": [],
    }

    for lineno, row in enumerate(tables.get("nodes") or [], 1):
        if _row_is_blank(row):
            continue
        model["nodes"].append({
            "id": _int(row, "id", "nodes", lineno),
            "x": _num(row, "x", "nodes", lineno),
            "y": _num(row, "y", "nodes", lineno),
            "z": _num(row, "z", "nodes", lineno)})
    _check_unique([n["id"] for n in model["nodes"]], "nodes", "节点号")

    for lineno, row in enumerate(tables.get("members") or [], 1):
        if _row_is_blank(row):
            continue
        m: dict[str, Any] = {
            "id": _int(row, "id", "members", lineno),
            "i": _int(row, "i", "members", lineno),
            "j": _int(row, "j", "members", lineno),
            "section": _text(row, "section", "members", lineno),
            "material": _text(row, "material", "members", lineno)}
        rel_i = _parse_releases(row.get("releases_i"), "members", lineno, "i")
        rel_j = _parse_releases(row.get("releases_j"), "members", lineno, "j")
        if rel_i or rel_j:
            # 只在真有释放时才写这一项——空的 releases 会让模型 JSON 变脏
            m["releases"] = {k: v for k, v in (("i", rel_i), ("j", rel_j)) if v}
        model["members"].append(m)
    _check_unique([m["id"] for m in model["members"]], "members", "杆件号")

    for lineno, row in enumerate(tables.get("supports") or [], 1):
        if _row_is_blank(row):
            continue
        fix = [1 if bool(row.get(name)) else 0 for name in LOCAL_DOF_NAMES]
        item = {"node": _int(row, "node", "supports", lineno), "fix": fix}
        if not _blank(row.get("name")):
            item["name"] = str(row["name"]).strip()
        model["supports"].append(item)
    _check_unique([s["node"] for s in model["supports"]], "supports", "节点号")

    cases: dict[str, dict[str, list]] = {}

    def case_of(row: dict[str, Any]) -> dict[str, list]:
        name = str(row.get("case") or DEFAULT_CASE).strip() or DEFAULT_CASE
        return cases.setdefault(name, {"nodal_loads": [], "member_loads": [],
                                       "member_spans": [], "settlements": []})

    for lineno, row in enumerate(tables.get("nodal_loads") or [], 1):
        if _row_is_blank(row):
            continue
        item = {
            "node": _int(row, "node", "nodal_loads", lineno),
            "load": [_num(row, k, "nodal_loads", lineno)
                     for k in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]}
        if not _blank(row.get("name")):
            item["name"] = str(row["name"]).strip()
        case_of(row)["nodal_loads"].append(item)

    for lineno, row in enumerate(tables.get("member_loads") or [], 1):
        if _row_is_blank(row):
            continue
        item = {
            "member": _int(row, "member", "member_loads", lineno),
            "w": [_num(row, k, "member_loads", lineno)
                  for k in ("wx", "wy", "wz")]}
        if not _blank(row.get("name")):
            item["name"] = str(row["name"]).strip()
        case_of(row)["member_loads"].append(item)

    for lineno, row in enumerate(tables.get("member_spans") or [], 1):
        if _row_is_blank(row):
            continue
        kind = _text(row, "kind", "member_spans", lineno)
        if kind not in SPAN_KINDS:
            raise TableError(
                f"{_LABELS['member_spans']}表第 {lineno} 行：类型 {kind!r} 无效，"
                f"应取自 {', '.join(SPAN_KINDS)}")
        item: dict[str, Any] = {
            "member": _int(row, "member", "member_spans", lineno),
            "kind": kind,
            "w1": [_num(row, k, "member_spans", lineno)
                   for k in ("w1x", "w1y", "w1z")]}
        if not _blank(row.get("name")):
            item["name"] = str(row["name"]).strip()
        if kind == TRAPEZOID:
            item["w2"] = [_num(row, k, "member_spans", lineno)
                          for k in ("w2x", "w2y", "w2z")]
        if kind == POINT:
            item["a"] = _num(row, "a", "member_spans", lineno)
        note = row.get("note")
        if not _blank(note):
            item["note"] = str(note).strip()
        case_of(row)["member_spans"].append(item)

    for lineno, row in enumerate(tables.get("settlements") or [], 1):
        if _row_is_blank(row):
            continue
        item = {
            "node": _int(row, "node", "settlements", lineno),
            "d": [_num(row, k, "settlements", lineno) for k in LOCAL_DOF_NAMES]}
        if not _blank(row.get("name")):
            item["name"] = str(row["name"]).strip()
        case_of(row)["settlements"].append(item)

    if cases:
        model["load_cases"] = [
            {"name": name,
             **{k: v for k, v in block.items() if v}}
            for name, block in cases.items()]
        kept = {c["name"] for c in model["load_cases"]}
        combos = [dict(c) for c in base.get("combos") or []
                  if set(c["factors"]) <= kept]
        if combos:
            model["combos"] = combos
    return model


# --------------------------------------------------------------- CSV

def to_csv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buf.getvalue()


def from_csv(text: str, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """读 CSV。表头缺列就直接报错——静默填默认值会让人以为导入成功了。"""
    import csv
    import io
    reader = csv.DictReader(io.StringIO(text))
    # name 是后来加入的 Abaqus 式可读标识；旧项目导出的 CSV 没有这一列，
    # 仍应能导回。其它列继续严格检查，避免把数值悄悄补成空值。
    missing = set(columns) - set(reader.fieldnames or []) - {"name"}
    if missing:
        raise TableError(f"CSV 缺少列：{', '.join(sorted(missing))}；"
                         f"需要的表头是 {', '.join(columns)}")
    bools = _BOOLEAN_COLUMNS.get(tuple(columns), frozenset())
    out = []
    for row in reader:
        clean = {c: (row.get(c) or "").strip() for c in columns}
        if all(not v for v in clean.values()):
            continue
        for c in bools:
            clean[c] = clean[c].lower() in {"1", "true", "yes", "是", "y"}
        out.append(clean)
    return out
