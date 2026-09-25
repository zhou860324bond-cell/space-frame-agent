"""Session 各域共用的几样基础件。

从 agent.py 搬出来的，代码一个字节没改。

为什么要单独一个文件：Session 按域拆成了 session_*.py 那几个 mixin，
而这四样东西每个域都要用，又不属于任何一个域。留在 agent.py 里的话，
mixin 就得反过来 import agent——而 agent 正要 import 那些 mixin，成环。

所以这里是依赖图的底：本文件不 import 任何 session_* 或 agent。
往这里加东西之前先想一下，是不是真的"每个域都要用且不属于任何一个域"。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

from model_io import CURRENT_SCHEMA_VERSION

SELF_WEIGHT_NOTE = "self-weight"      # 自重荷载的标记，用于重复调用时替换
_UNSET = object()                       # 区分“没有修改”与“清除可选字段”


@dataclass
class ToolResult:
    ok: bool
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, **self.payload}, ensure_ascii=False)


def _name_unnamed_supports(model: dict) -> None:
    """给没有名字的支座补上 BC-n。

    **参数化生成器建的支座不带名字**，而边界条件管理器是按名字删的：
    列表里显示成"(未命名-节点1)"这种合成标签，照着它删会报"没有名为
    '(未命名-节点1)' 的边界条件"——**列得出来，一个也删不掉**。
    用 generate_frame 建的模型全都这样，而那是最常用的一条路径。

    名字按**约束掩码**分组给：同一组约束（比如六个固接柱脚）共用一个
    BC-n，这与 Abaqus 的 BC 概念一致，也与 list_boundary_conditions 的
    分组口径一致；逐个节点各给一个名字的话，删一条只删掉一个柱脚，
    而用户在管理器里看到的是一整条。
    """
    supports = model.get("supports") or []
    if not supports or all(item.get("name") for item in supports):
        return
    used = {str(item["name"]) for item in supports if item.get("name")}
    index = 1
    assigned: dict[tuple, str] = {}
    for item in supports:
        if item.get("name"):
            continue
        key = (tuple(item.get("fix") or ()), tuple(item.get("spring") or ()))
        if key not in assigned:
            while f"BC-{index}" in used:
                index += 1
            assigned[key] = f"BC-{index}"
            used.add(assigned[key])
        item["name"] = assigned[key]


def _records(fn: Callable) -> Callable:
    """把这个工具的调用记进构建历史。

    **必须挂在方法上，不能只挂在 dispatch 上。** 界面是直接调
    `fem.generate_frame(...)` 的，不走 dispatch；只在 dispatch 里记账的话，
    参数化建模这条路径的时间轴永远是空的——而那恰恰是用户最常看的一条。
    """
    import functools

    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        # Session 中的模型是持久化 Domain IR。旧模型允许读入，但任何一次成功
        # 的建模操作之后都写成当前版本，历史快照和保存文件便不再是无版本数据。
        if result.ok and self.model:
            self.model.setdefault("schema_version", CURRENT_SCHEMA_VERSION)
            _name_unnamed_supports(self.model)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        recorded = dict(bound.arguments)
        recorded.pop("self", None)
        # 几个工具的签名是 (self, **kwargs)，bind 会把参数整包塞进
        # arguments["kwargs"]。不展平的话摘要里全是 None——
        # "生成门式刚架：0 跨，檐口 None m"，第一版就是这样
        for name, param in signature.parameters.items():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                recorded.update(recorded.pop(name, {}) or {})
        self.history.record(fn.__name__, recorded, result.ok,
                            result.payload, self.model,
                            self.multimodal_provenance)
        return result

    return wrapper
