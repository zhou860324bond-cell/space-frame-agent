"""多份图纸证据 → 一份证据：图片、几个 DXF、补充说明混合输入。

每种入口各自产出 ``drawing_evidence`` 格式的证据，这里只做**结构上的合并**，不比较数值：
同一键的条目（同一对轴号的间距、同名标高、同一组编号的规格、单位、材料、支座）
放进主条目的 ``alternatives``。值是否相同由 ``drawing_assemble`` 在单位确定后判断：
相同即合并并列出全部出处，不同即报冲突让用户选，不自动以哪边为准。

* 轴号顺序：各份证据的轴号序列按先后关系合并；顺序不唯一或互相矛盾时写进
  ``order_problem`` 交给核对表，不猜。只有一个方向轴线的视图（立面、剖面）若轴号全属于
  平面图的另一方向，就改归那个方向——DXF 读取器按图上走向分族，字母轴的立面上
  轴线也是竖线；这类视图的轴号顺序与已有顺序正好相反时按反向视图读。两种调整都写进
  识别提示。
* 杆件不按键合并，原样并列，由组装按位置去重（同位置同编号即同一根）。
* ``previous``：之前合并后用户处理过的条目按同一键保留（可传几次合并结果，新的在前、
  同一键以新的为准；加图后图片证据暂时撤出、重新识别回来时，用户在图片读数上的处理
  仍能找回）。新读数里出现用户处理时没见过的原文，用户的结果作为其中一项读数重新进入
  冲突，由用户再选一次。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from drawing_evidence import EVIDENCE_FORMAT, reading, section_labels, settled

FAMILY_NAMES = {"x": "x 向", "y": "y 向"}
_REGISTRIES = ("images", "files", "statements", "layers")


def _pair(item: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(item.get(key) or "").strip() for key in ("from", "to")))


def origin(source: Any) -> str:
    """一条出处属于哪份证据：图片号、DXF 文件号、说明号；用户填写为 user。"""
    if not isinstance(source, Mapping) or source.get("user"):
        return "user"
    for key in ("image", "file", "statement"):
        if source.get(key) is not None:
            return f"{key}:{source[key]}"
    return "unknown"


def member_key(item: Mapping[str, Any]) -> tuple:
    """杆件条目的身份：出处加上全部位置字段（只用来在重读后找回用户的确认或删除）。"""
    def names(value):
        return tuple(str(v) for v in value) if isinstance(value, list) else ()
    return (origin(item.get("source")), item.get("kind"), item.get("label"), item.get("from"),
            item.get("to"), item.get("along"), names(item.get("levels")), names(item.get("at")),
            item.get("from_level"), item.get("to_level"))


def keyed(evidence: Mapping[str, Any]) -> list[tuple[tuple, Mapping[str, Any]]]:
    """证据里可按键合并的条目：(键, 条目)。键的第一项是条目所在的位置。"""
    out: list[tuple[tuple, Any]] = []
    for key, item in (evidence.get("units") or {}).items():
        out.append((("units", key), item))
    for family in ("x", "y"):
        grid = (evidence.get("grids") or {}).get(family)
        if not isinstance(grid, Mapping):
            continue
        out += [(("spacings", family, _pair(item)), item) for item in grid.get("spacings") or []
                if isinstance(item, Mapping)]
        if isinstance(grid.get("total"), Mapping):
            out.append((("total", family), grid["total"]))
        out += [(("edges", family, index), item)
                for index, item in enumerate(grid.get("edges") or [])]
    out += [(("levels", str(item.get("name") or "").strip()), item)
            for item in evidence.get("levels") or [] if isinstance(item, Mapping)]
    out += [(("level_heights", _pair(item)), item)
            for item in evidence.get("level_heights") or [] if isinstance(item, Mapping)]
    out += [(("sections", tuple(section_labels(item))), item)
            for item in evidence.get("sections") or [] if isinstance(item, Mapping)]
    for key in ("material", "support"):
        out.append(((key,), evidence.get(key)))
    out += [(("members", member_key(item)), item)
            for item in evidence.get("members") or [] if isinstance(item, Mapping)]
    out += [(("notes", str(item.get("text")), origin(item.get("source")),
              str((item.get("source") or {}).get("sentence"))), item)
            for item in evidence.get("notes") or [] if isinstance(item, Mapping)]
    return [(key, item) for key, item in out if isinstance(item, Mapping)]


def _axis_labels(part: Mapping[str, Any], family: str) -> list[str]:
    grid = (part.get("grids") or {}).get(family) or {}
    return [str(axis.get("label")).strip() for axis in grid.get("axes") or []
            if isinstance(axis, Mapping) and str(axis.get("label") or "").strip()]


def _single_family(part: Mapping[str, Any]) -> str | None:
    families = [family for family in ("x", "y") if _axis_labels(part, family)]
    return families[0] if len(families) == 1 else None


def _part_name(part: Mapping[str, Any]) -> str:
    for key, label in (("files", "DXF"), ("images", "图"), ("statements", "说明")):
        records = [r for r in part.get(key) or [] if isinstance(r, Mapping)]
        if records:
            return f"{label}「{'、'.join(str(r.get('name', '')) for r in records)}」"
    return "一份证据"


def _hint(part: Mapping[str, Any], text: str) -> dict[str, Any]:
    note: dict[str, Any] = {"text": text}
    files = [r for r in part.get("files") or [] if isinstance(r, Mapping)]
    images = [r for r in part.get("images") or [] if isinstance(r, Mapping)]
    if files:
        note["source"] = {"file": files[0].get("id")}
    elif images:
        note["source"] = {"image": images[0].get("id")}
    return note


def _align_families(parts: list[dict[str, Any]], hints: list[dict[str, Any]]) -> None:
    """只有一个方向轴线的视图，轴号全属于平面图另一方向时改归那个方向。"""
    plans = [part for part in parts if _single_family(part) is None
             and all(_axis_labels(part, family) for family in ("x", "y"))]
    reference = {family: {label for part in plans for label in _axis_labels(part, family)}
                 for family in ("x", "y")}
    for part in parts:
        family = _single_family(part)
        if family is None or part in plans:
            continue
        other = "y" if family == "x" else "x"
        mine = set(_axis_labels(part, family))
        if mine <= reference[other] and not mine & reference[family]:
            grids = part["grids"]
            grids[other] = grids.pop(family)
            hints.append(_hint(part, f"{_part_name(part)}只有一个方向的轴线，轴号 "
                                     f"{'、'.join(_axis_labels(part, other))} 在平面图中是"
                                     f"{FAMILY_NAMES[other]}轴线，已按{FAMILY_NAMES[other]}合并。"))


def merge_order(sequences: list[list[str]]) -> tuple[list[str], str]:
    """几串有先后关系的名称 → 一串；返回 (顺序, 问题：'' / 'ambiguous' / 'cycle')。

    只按各串内部的相邻先后排序；同一时刻有不止一个名称可以排在前面，说明这几串
    之间的相对位置无法确定（ambiguous）；先后关系成环说明互相矛盾（cycle）。"""
    nodes = list(dict.fromkeys(name for seq in sequences for name in seq))
    rank = {name: index for index, name in enumerate(nodes)}
    after: dict[str, set[str]] = {name: set() for name in nodes}
    indegree = dict.fromkeys(nodes, 0)
    for seq in sequences:
        for a, b in zip(seq, seq[1:], strict=False):
            if b not in after[a]:
                after[a].add(b)
                indegree[b] += 1
    ready = [name for name in nodes if indegree[name] == 0]
    order, problem = [], ""
    while ready:
        if len(ready) > 1:
            problem = "ambiguous"
        name = ready.pop(0)
        order.append(name)
        for nxt in after[name]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=rank.get)
    if len(order) < len(nodes):
        order += [name for name in nodes if name not in order]
        problem = "cycle"
    return order, problem


def _orient(parts: list[dict[str, Any]], family: str,
            hints: list[dict[str, Any]]) -> list[list[str]]:
    """各份证据的轴号序列；单方向视图与已有顺序正好相反时按反向视图读。"""
    sequences: list[list[str]] = []
    for part in parts:
        seq = _axis_labels(part, family)
        if not seq:
            continue
        known = merge_order(sequences)[0] if sequences else []
        shared = [label for label in seq if label in known]
        expected = [label for label in known if label in shared]
        if _single_family(part) and len(shared) >= 2 and shared == expected[::-1]:
            seq = seq[::-1]
            hints.append(_hint(part, f"{_part_name(part)}的{FAMILY_NAMES[family]}轴号从"
                                     f"{seq[-1]}排到{seq[0]}，与平面顺序相反，按反向视图合并。"))
        sequences.append(seq)
    return sequences


def _order_message(noun: str, names: list[str], problem: str) -> str:
    listing = "、".join(names)
    if problem == "cycle":
        return (f"各份证据中{noun}的先后顺序互相矛盾（{listing}）；请对照原图检查编号是否抄错，"
                "或移除读错的那份图纸。")
    return (f"{noun} {listing} 分别来自不同证据，相对顺序无法确定；请加入同时含这些{noun}的"
            "图纸（如平面图），不按名称猜顺序。")


def _strip(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "alternatives"}


def _carry(groups: dict[tuple, list[Mapping[str, Any]]],
           history: list[Mapping[str, Any]]) -> None:
    """把之前合并后用户处理过的条目按键放回；同一键只取最近的一次。"""
    done: set[tuple] = set()
    for key, item in (entry for previous in history for entry in keyed(previous)):
        if key in done or not (settled(item) or item.get("deleted")):
            continue
        done.add(key)
        group = groups.get(key)
        if group is None:
            if (item.get("source") or {}).get("user") and key[0] not in ("members", "notes"):
                groups[key] = [item]        # 用户补填的条目
            continue
        seen = set(item.get("seen") or []) | {reading(item)}
        seen |= {reading(alt) for alt in item.get("alternatives") or []}
        unseen = [_strip(c) for c in group if reading(c) not in seen]
        groups[key] = [item] if not unseen else [_strip(item), *unseen]


def _registry(parts: Iterable[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    records, ids = [], set()
    for part in parts:
        for record in part.get(key) or []:
            if not isinstance(record, Mapping):
                continue
            ident = record.get("id") if key != "layers" else (record.get("file"),
                                                                record.get("name"))
            if key != "layers" and ident in ids:
                raise ValueError(f"几份证据的编号 {ident} 重复；每份证据的图号、文件号须唯一。")
            ids.add(ident)
            records.append(dict(record))
    return records


def merge_evidence(parts: list[Mapping[str, Any]],
                   previous: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None
                   ) -> dict[str, Any] | None:
    """几份证据合并成一份交给 ``drawing_assemble.assemble``；没有证据时返回 None。

    ``previous`` 是之前的一次或几次合并结果（新的在前），用来找回用户的处理。"""
    parts = [deepcopy(dict(part)) for part in parts if isinstance(part, Mapping)]
    history = [previous] if isinstance(previous, Mapping) else [
        item for item in previous or [] if isinstance(item, Mapping)]
    if not parts:
        return None
    if len(parts) == 1 and not history:
        return parts[0]
    hints: list[dict[str, Any]] = []
    _align_families(parts, hints)
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    owner: dict[tuple, int] = {}
    for index, part in enumerate(parts):
        for key, item in keyed(part):
            if key in groups and owner[key] == index:
                count = 1           # 同一份证据里的重复条目保持各自独立，与单份证据的行为一致
                while key + (count,) in groups:
                    count += 1
                key = key + (count,)
            groups.setdefault(key, []).append(item)
            owner.setdefault(key, index)
    if history:
        _carry(groups, history)

    out: dict[str, Any] = {"format": EVIDENCE_FORMAT}
    for key in _REGISTRIES:
        out[key] = _registry(parts, key)
    planes: dict[Any, Mapping[str, Any]] = {}
    for before in history:
        for record in before.get("files") or []:
            if isinstance(record, Mapping) and record.get("plane"):
                planes.setdefault(record.get("id"), record)
    for record in out["files"]:
        before = planes.get(record.get("id"))
        if before is not None and before.get("sha256") == record.get("sha256"):
            record["plane"] = before["plane"]
    out.update(units={}, grids={}, levels=[], level_heights=[], members=[], sections=[],
               material=None, support=None, notes=[])
    # 「几份立面是同一榀」的确认连同当时的视图清单一起保留；视图变了组装会重新问。
    views = next((before["views"] for before in history
                  if isinstance(before.get("views"), Mapping)), None)
    if views is not None:
        out["views"] = dict(views)
    for family in ("x", "y"):
        sequences = _orient(parts, family, hints)
        if not sequences and not any(key[:2] in (("spacings", family), ("total", family),
                                                 ("edges", family)) for key in groups):
            continue
        order, problem = merge_order(sequences)
        axes = {}
        for part in parts:
            for axis in ((part.get("grids") or {}).get(family) or {}).get("axes") or []:
                if isinstance(axis, Mapping):
                    axes.setdefault(str(axis.get("label") or "").strip(), dict(axis))
        grid: dict[str, Any] = {"axes": [axes[label] for label in order], "spacings": [],
                                "total": None, "edges": []}
        if problem:
            grid["order_problem"] = _order_message(f"{FAMILY_NAMES[family]}轴号", order, problem)
        out["grids"][family] = grid
    for key, candidates in groups.items():
        # 只有名字没有读数的条目（立面上只写层名、不写标高）不算一份读数，不参与比较。
        informative = [c for c in candidates if c.get("text") not in (None, "")
                       or c.get("value") is not None or c.get("kind") is not None]
        if key[0] not in ("members", "notes") and informative:
            candidates = informative
        item = dict(candidates[0])
        if len(candidates) > 1:
            item["alternatives"] = [_strip(c) for c in candidates[1:]]
        kind = key[0]
        if kind == "units":
            out["units"][key[1]] = item
        elif kind in ("spacings", "total", "edges"):
            grid = out["grids"].setdefault(key[1], {"axes": [], "spacings": [], "total": None,
                                                    "edges": []})
            if kind == "total":
                grid["total"] = item
            else:
                grid[kind].append(item)
        elif kind in ("material", "support"):
            out[kind] = item
        else:
            out[kind].append(item)
    level_seqs = [[str(item.get("name") or "").strip() for item in part.get("levels") or []
                   if isinstance(item, Mapping)] for part in parts]
    level_order, problem = merge_order([seq for seq in level_seqs if seq])
    rank = {name: index for index, name in enumerate(level_order)}
    out["levels"].sort(key=lambda item: rank.get(str(item.get("name") or "").strip(),
                                                 len(rank)))
    if problem and out["level_heights"]:
        out["level_order_problem"] = _order_message("层名", level_order, problem)
    out["notes"] += hints
    return out
