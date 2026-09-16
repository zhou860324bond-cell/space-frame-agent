r"""从 Abaqus 的 `.dat` 里读节点结果。

**这条路不需要 Abaqus。** `.dat` 是纯文本，只要 `.inp` 里写了 `*NODE PRINT`，
位移与反力就会打进去。相比走 ODB（必须用 `abaqus python`，且受版本限制），
解析 `.dat` 让对比可以在任何机器上做——队友、CI、答辩用的电脑都行。

代价是 `.dat` 只有请求过的量，而 ODB 里什么都有。所以两条路都留着：
`.dat` 用于对比验证，ODB 用于需要完整场量的场合。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# "       NODE FOOT-   U1          U2          U3          UR1 ..." 这一行给出列名
_HEADER = re.compile(r"^\s*NODE\s+FOOT-\s+(.+?)\s*$")
_ROW = re.compile(r"^\s*(\d+)\s+(.*)$")
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")

# .dat 里的列名 -> 我们统一用的字段名
_ALIAS = {"U1": "u1", "U2": "u2", "U3": "u3",
          "UR1": "ur1", "UR2": "ur2", "UR3": "ur3",
          "RF1": "rf1", "RF2": "rf2", "RF3": "rf3",
          "RM1": "rm1", "RM2": "rm2", "RM3": "rm3"}


def _to_float(token: str) -> float:
    # Fortran 有时用 D 表示指数
    return float(token.replace("D", "E").replace("d", "e"))


_FIELDS = tuple(_ALIAS.values())


def parse_dat(path: str | Path) -> dict[int, dict[str, float]]:
    """返回 {节点号: {u1..ur3, rf1..rm3}}，缺项补 0。同一节点的多张表会合并。

    **补 0 是有物理含义的，不是凑数**：Abaqus 不会在 U 表里打印全约束节点
    （位移恒为零，省略），也不会给自由节点打印反力（本来就没有）。
    不补的话，调用方直接索引 `d[node]["u3"]` 就会在支座节点上踩空。
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict[int, dict[str, float]] = {}
    columns: list[str] = []

    for line in text.splitlines():
        header = _HEADER.match(line)
        if header:
            columns = [_ALIAS[tok] for tok in header.group(1).split()
                       if tok in _ALIAS]
            continue
        if not columns:
            continue
        row = _ROW.match(line)
        if not row:
            if line.strip() and not line.strip().startswith("NOTE"):
                # 表结束（遇到说明文字或下一节），等下一个表头
                if not _NUMBER.match(line.strip()):
                    columns = []
            continue
        values = [_to_float(t) for t in _NUMBER.findall(row.group(2))]
        if len(values) < len(columns):
            continue
        node = int(row.group(1))
        record = out.setdefault(node, {})
        # 上面只挡了 values 比 columns 短的行；长出来的列是 .dat 常有的
        # 尾随字段，表头没给名字，截掉正是想要的
        for name, value in zip(columns, values, strict=False):
            record[name] = value
    for record in out.values():
        for field in _FIELDS:
            record.setdefault(field, 0.0)
    return out


def available_jobs(folder: str | Path) -> list[Path]:
    """列出该目录下所有有 .dat 的作业。"""
    return sorted(p for p in Path(folder).glob("*.dat") if p.is_file())


def job_status(path: str | Path) -> dict[str, Any]:
    """从 .dat / .sta 判断这次作业是不是真的算完了。"""
    dat = Path(path)
    sta = dat.with_suffix(".sta")
    text = dat.read_text(encoding="utf-8", errors="replace")
    completed = "THE ANALYSIS HAS BEEN COMPLETED" in text
    if sta.is_file():
        completed = completed or ("COMPLETED SUCCESSFULLY"
                                  in sta.read_text(encoding="utf-8", errors="replace"))
    errors = len(re.findall(r"\*\*\*ERROR", text))
    return {"job": dat.stem, "completed": completed, "error_lines": errors}
