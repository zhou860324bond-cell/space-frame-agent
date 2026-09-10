"""Agent 层：把自然语言变成结构模型，再由确定性工具算出全部数字。

设计原则（这几条是整个项目立得住的根据，改代码时别破坏）：

1. **大模型只产结构，不产数值。** 它可以决定"三跨两层、柱底固接"，但位移、内力、
   反力必须来自工具返回。系统提示里明令禁止它自己算，报告里每个数字都能溯源到
   某次工具调用。
2. **拓扑由代码展开。** 多层多跨让模型逐个列节点坐标必然出错，所以给它
   `generate_frame(spans, storeys, bays, ...)` 这样的参数入口。不规则结构才走
   `set_model` 这条逃生舱。
3. **算前守门。** `solve_model` 内部先跑 `validate_model`，不通过就不算，把中文
   错误清单回给模型，让它自己改——这就是自修复闭环的一轮。
4. **信息不全就追问，不许编默认值。** 这是 Agent 区别于脚本的地方，也是评测集
   里专门要统计的一项指标。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from frame3d import (check_equilibrium, diagnose_singularity, self_weight_loads,
                     solve)
from generator import (GeneratorError, beam_member_ids, describe,
                       generate_frame, generate_portal_frame,
                       rafter_member_ids)
import bent as _bent
import changes as _changes
from frame3d import LOCAL_DOF_NAMES
from history import BuildHistory, MUTATING
from model_compiler import CompilationError, compile_model
from model_io import (CURRENT_SCHEMA_VERSION, from_dict, migrate_payload,
                      validate_definitions, validate_payload)
from result_db import from_solution as result_db_from_solution
from units import convert_model
from units import system as unit_system

# 求解后自动跑的两项质检：静默失败检测 + 实验胶囊存档。
# 这两项失败不影响求解结果本身——用 try/except 兜住，只记警告。
import capsule as _capsule
import silent_failures as _silent

MAX_ROUNDS = 24     # 一次完整建模+求解+查询，模型若逐个发调用能到十几轮。
                    # 真正的省时手段是让它一条消息发多个调用（见 SYSTEM_PROMPT 第 9 条），
                    # 上限只是最后那道防跑飞的闸
SELF_WEIGHT_NOTE = "self-weight"      # 自重荷载的标记，用于重复调用时替换
_UNSET = object()                       # 区分“没有修改”与“清除可选字段”

SYSTEM_PROMPT = """你是空间刚架结构分析助手。你的职责是把用户的自然语言描述转成结构模型，
调用工具完成计算，然后解释结果。

硬性规则，任何情况下都不得违反：

1. 你不做任何数值计算。位移、内力、反力、应力的每一个数字都必须来自工具返回值。
   如果你没有调用工具就说出一个数，那就是编的。禁止心算、估算、类比推测。
2. 建模按这个顺序挑工具，**不要自己逐个列节点坐标**——多层多跨你一定会数错：
   · 正交规则框架 → generate_frame
   · 坡屋面门式刚架 → generate_portal_frame
   · 其它形状（单坡、悬挑、错层、任意折线屋面）→ generate_bent，
     屋面形状全由 profile 折线决定；要做成空间结构再调 extrude_bents
   · 在已有模型上局部改动 → 算子：add_bracing 加支撑、remove_members 抽柱开洞、
     raise_nodes 抬高一片区域、retaper 换截面
   算子要的杆件编号**一律先用 select_members 按几何条件挑**，不许凭空猜编号——
   编号规则是生成器定的，你并不知道 27 号在哪。
   同一组杆件要反复引用时，用 define_set 给它起个工程上的名字（「顶层梁」「边柱」），
   之后 remove_members / retaper 直接写名字即可。回答里也用这个名字，
   比报一串编号好懂得多。
   只有以上都覆盖不了时才用 set_model 传完整 JSON。
3. 几何、属性、边界和载荷分阶段建立，顺序参照 Abaqus/CAE：可以先生成或手工绘制
   节点与杆件拓扑，暂不指派材料和截面；之后用 Property 阶段补齐属性，再定义工况、
   边界条件和载荷。只有 solve_model 前必须全部完整。几何尺寸缺失时必须追问，
   不许编造；用户只要求先建几何时，不要因材料、截面或载荷未给而阻止建模。
4. solve_model 返回错误清单时，读懂它、改模型、重试。同一个错误连续两次没改对，
   就停下来告诉用户你卡在哪里。
5. 默认单位制 N-m-Pa：长度米，力牛顿，弹性模量帕斯卡。用户说"20 kN/m"你要换算成
   20000 再传给工具。换算过程写在回答里。用户明确要求用毫米制时才调 set_units
   切到 N-mm-MPa，那会把整份模型按物理量等价地换算过去。
   查询结果一律以 mm、kN、kN·m 返回，与模型用哪套制无关。
6. 材料与截面调用 define_materials_and_sections 定义一次即可，generate_frame 和
   set_model 都会自动沿用。几何草稿的 analysis_ready=false 不是建模失败；继续指派属性、
   边界和载荷，最后调用 validate_model。不要因为同一报错反复重发模型。
   同一个工具连续两次返回同类错误，就停下来说明你卡在哪里，不要继续试。
7. 多个荷载工况必须用 set_load_cases 分别定义再做组合，**不许把几个工况的荷载先加起来
   当成单一工况**。线性叠加下组合结果虽然相同，但那样拿不到各工况的分项内力，也无法做包络。
8. 描述结果位置时只能引用工具返回的坐标。不要凭节点编号推测它在结构的什么位置——
   编号规则你并不知道。

9. **互不依赖的工具调用要放在同一条消息里一次发出。** 定义材料截面、生成框架、
   设定荷载工况这一串是固定顺序的，可以一次全发；查位移、查内力、查包络彼此无关，
   更该一次全发。一次只发一个会让每一步都多一个网络往返——用户等的是秒，不是毫秒。
   只有当后一步真的要读前一步的返回值时（比如要用生成的杆件编号去加荷载），
   才分两次发。

10. 删除已有节点或杆件必须先调用 preview_change，把结构化差异和预演编号告诉用户，
    然后停止本轮。只有用户在**下一条消息**明确确认后才调用 apply_preview；你无权替用户
    确认，也不得在同一轮或同一批工具调用中预演并执行。

回答用中文，简洁，先给结论再给数据。"""


# --------------------------------------------------------------------------- 工具定义

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "define_materials_and_sections",
            "description": "定义材料与截面。可在几何之前或之后调用；"
                           "Abaqus 式流程通常先建几何，再做属性指派。",
            "parameters": {
                "type": "object",
                "required": ["materials", "sections"],
                "properties": {
                    "materials": {
                        "type": "array",
                        "items": {"type": "object", "required": ["name", "E", "nu"],
                                  "properties": {"name": {"type": "string"},
                                                 "E": {"type": "number"},
                                                 "nu": {"type": "number"},
                                                 "density": {
                                                     "type": "number",
                                                     "description": "kg/m³，只有要算自重时才需要，"
                                                                    "钢约 7850、混凝土约 2500"},
                                                 "yield_stress": {"type": "number"},
                                                 "hardening_ratio": {"type": "number"}}},
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "object", "required": ["name", "A", "Iy", "Iz", "J"],
                                  "properties": {"name": {"type": "string"},
                                                 "A": {"type": "number"},
                                                 "Iy": {"type": "number"},
                                                 "Iz": {"type": "number"},
                                                 "J": {"type": "number"},
                                                 "Ay": {"type": "number"},
                                                 "Az": {"type": "number"}}},
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_frame",
            "description": "按参数生成规则空间刚架的节点与杆件。规则框架一律用这个，"
                           "不要自己列节点坐标。",
            "parameters": {
                "type": "object",
                "required": ["spans", "storeys"],
                "properties": {
                    "spans": {"type": "array", "items": {"type": "number"},
                              "description": "沿 X 方向各跨跨度，单位米"},
                    "storeys": {"type": "array", "items": {"type": "number"},
                                "description": "各层层高，自下而上，单位米"},
                    "bays": {"type": "array", "items": {"type": "number"},
                             "description": "沿 Y 方向各开间宽度；平面刚架留空"},
                    "column_section": {"type": "string"},
                    "beam_section": {"type": "string"},
                    "material": {"type": "string"},
                    "base": {"type": "string", "enum": ["free", "fixed", "pinned"],
                             "description": "free 只建几何；或柱底固接/铰接"},
                    "beam_release": {"type": "boolean", "description": "梁两端铰接"},
                    "beam_load": {"type": "number",
                                  "description": "所有梁上的均布线荷载，N/m，正值向下"},
                    "load_case": {"type": "string", "description": "荷载放进哪个工况"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_member",
            "description":
                "改一根杆件的截面、材料或端部释放。**只传要改的项**，不传的不动。"
                "端部释放传空数组表示取消释放（改回刚接）——"
                "这和「不传」是两件事，必须分得开，否则铰接梁改不回刚接。",
            "parameters": {
                "type": "object",
                "required": ["member_id"],
                "properties": {
                    "member_id": {"type": "integer"},
                    "section": {"type": "string"},
                    "material": {"type": "string"},
                    "releases_i": {"type": "array", "items": {"type": "string"},
                                   "description": "i 端释放的局部自由度，"
                                                  "空数组表示取消释放"},
                    "releases_j": {"type": "array", "items": {"type": "string"}},
                    "offset_i": {"type": "array", "minItems": 3, "maxItems": 3,
                                 "items": {"type": "number"}},
                    "offset_j": {"type": "array", "minItems": 3, "maxItems": 3,
                                 "items": {"type": "number"}},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_node",
            "description":
                "改一个节点的坐标或约束。**只传要改的项**。"
                "fix 是六个 0/1，顺序 ux uy uz rx ry rz；空数组表示取消全部约束。",
            "parameters": {
                "type": "object",
                "required": ["node_id"],
                "properties": {
                    "node_id": {"type": "integer"},
                    "x": {"type": "number"}, "y": {"type": "number"},
                    "z": {"type": "number"},
                    "fix": {"type": "array", "items": {"type": "integer",
                                                       "enum": [0, 1]},
                            "description": "六个 0/1；空数组表示取消约束"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "define_set",
            "description":
                "给一组杆件或节点起个名字，之后凡是要杆件编号的地方都能直接写这个名字。"
                "**这是让「给顶层所有梁加 5 kN/m」这句话有落点的前提**——"
                "编号规则由生成器决定，谁也不知道 27 号在哪。"
                "两种给法二选一：直接给 member_ids / node_ids，"
                "或者给几何条件（同 select_members），由代码挑出来。"
                "一根都没挑到时会报错，而不是存下一个空集合。",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string",
                             "description": "集合名，用工程上的叫法，"
                                            "如「顶层梁」「边柱」"},
                    "member_ids": {"type": "array", "items": {"type": "integer"}},
                    "node_ids": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": "string", "description": "这组是什么，给人看的"},
                    "section": {"type": "string"},
                    "orientation": {"type": "string",
                                    "enum": ["vertical", "horizontal", "inclined"]},
                    "x_range": {"type": "array", "items": {"type": "number"},
                                "minItems": 2, "maxItems": 2},
                    "y_range": {"type": "array", "items": {"type": "number"},
                                "minItems": 2, "maxItems": 2},
                    "z_range": {"type": "array", "items": {"type": "number"},
                                "minItems": 2, "maxItems": 2},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sets",
            "description": "列出已定义的命名集合及各自的规模。",
            "parameters": {"type": "object", "properties": {},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_bent",
            "description":
                "生成一榀平面刚架。**屋面形状完全由 profile 折线决定**："
                "平屋面两个点 [[0,7.5],[24,7.5]]；双坡三个点，中间那个是屋脊 "
                "[[0,7.5],[12,8.7],[24,7.5]]；单坡两端不等高；悬挑就把折线两端"
                "伸到柱子外面。柱顶标高由折线插值得到。"
                "多层框架填 levels；错层用 base_levels 逐柱给柱脚标高。"
                "**这一个工具覆盖了原来要写四个生成器的形状。** "
                "做成空间结构请接着调 extrude_bents。",
            "parameters": {
                "type": "object",
                "required": ["profile", "columns"],
                "properties": {
                    "profile": {"type": "array",
                                "items": {"type": "array",
                                          "items": {"type": "number"},
                                          "minItems": 2, "maxItems": 2},
                                "description": "顶层折线 [[x, z], ...]，按 x 递增，米"},
                    "columns": {"type": "array", "items": {"type": "number"},
                                "description": "落柱的 x 坐标，严格递增，米"},
                    "levels": {"type": "array", "items": {"type": "number"},
                               "description": "中间楼层标高，自下而上，米"},
                    "base_levels": {"type": "array", "items": {"type": "number"},
                                    "description": "各柱柱脚标高，米。做错层用；"
                                                   "长度必须等于柱数"},
                    "column_section": {"type": "string"},
                    "beam_section": {"type": "string",
                                     "description": "楼面梁与屋面梁的截面"},
                    "material": {"type": "string"},
                    "base": {"type": "string", "enum": ["free", "fixed", "pinned"]},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extrude_bents",
            "description":
                "把当前这一榀沿 Y 方向复制成多榀，并在柱顶与楼层节点之间生成连系梁。"
                "bays 是**各开间宽度**不是开间数量——山墙开间往往小一些。"
                "单榀时补的面外约束会自动去掉：拉伸后不再是平面刚架，"
                "留着会把结构在 Y 向钉死，横向刚度偏刚且不报错。",
            "parameters": {
                "type": "object",
                "required": ["bays"],
                "properties": {
                    "bays": {"type": "array", "items": {"type": "number"},
                             "description": "各开间宽度，米"},
                    "tie_section": {"type": "string",
                                    "description": "连系梁截面；留空沿用梁截面"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_members",
            "description":
                "按几何条件挑出杆件编号，供 add_bracing / remove_members / retaper 使用。"
                "**不要凭空猜编号**——编号规则由生成器决定，你并不知道 27 号在哪。"
                "区间判定用杆件中点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "只要这个截面的"},
                    "orientation": {"type": "string",
                                    "enum": ["vertical", "horizontal", "inclined"],
                                    "description": "vertical 柱 / horizontal 水平梁 / inclined 斜杆"},
                    "x_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                    "y_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                    "z_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_bracing",
            "description":
                "在竖向框格里加支撑。**这是钢框架最有效的抗侧手段**——"
                "加一道 X 撑对侧移的改善往往比把所有柱加大一号还明显，用钢量却少得多。"
                "支撑两端默认铰接（靠轴力工作）。"
                "评价效果要看**被撑那一榀的侧移**，不要看全结构最大位移："
                "只撑局部时最大值会跑到没撑的地方，看起来像没生效。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["X", "V", "single"],
                             "description": "X 交叉撑 / V 人字撑 / single 单斜撑"},
                    "x_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                    "y_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                    "z_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                    "section": {"type": "string"},
                    "pinned": {"type": "boolean",
                               "description": "两端是否铰接，默认 true"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "raise_nodes",
            "description":
                "把一片区域内的节点整体抬高 dz（负值即降低）。"
                "抬高中央区域能增加空间刚度、降低跨中位移，是很典型的一个 what-if。"
                "区域用 x/y/z_range 框，不指定的方向不限制。",
            "parameters": {
                "type": "object",
                "required": ["dz"],
                "properties": {
                    "dz": {"type": "number", "description": "抬高量，米"},
                    "x_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                    "y_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                    "z_range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "[下限, 上限]，按杆件或节点的位置筛选"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retaper",
            "description":
                "给一批杆件换截面：变截面、局部加强都用它。"
                "杆件编号请先用 select_members 挑出来。",
            "parameters": {
                "type": "object",
                "required": ["member_ids", "section"],
                "properties": {
                    "member_ids": {"type": "array", "items": {"type": "integer"}},
                    "section": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_portal_frame",
            "description": "生成坡屋面门式刚架（国内钢结构最常见的形式）。"
                           "每跨两柱两斜梁、跨中起脊；给了 bays 就沿进深重复布置并加纵向系杆。"
                           "平屋面的规整框架请用 generate_frame。",
            "parameters": {
                "type": "object",
                "required": ["spans", "eave_height", "ridge_rise"],
                "properties": {
                    "spans": {"type": "array", "items": {"type": "number"},
                              "description": "X 向各跨跨度，单位米"},
                    "eave_height": {"type": "number", "description": "檐口高度（柱高），米"},
                    "ridge_rise": {"type": "number",
                                   "description": "屋脊相对檐口的升高，米。"
                                                  "坡度 i = ridge_rise / (跨度/2)"},
                    "bays": {"type": "array", "items": {"type": "number"},
                             "description": "Y 向各开间宽度；留空则只生成一榀平面刚架"},
                    "column_section": {"type": "string"},
                    "rafter_section": {"type": "string"},
                    "tie_section": {"type": "string", "description": "纵向系杆截面"},
                    "material": {"type": "string"},
                    "base": {"type": "string", "enum": ["free", "fixed", "pinned"],
                             "description": "free 只建几何；需要同步定义柱脚时再选 pinned/fixed"},
                    "rafter_load": {"type": "number",
                                    "description": "所有斜梁上的均布线荷载，N/m，正值向下"},
                    "load_case": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_nodes",
            "description": "在当前模型上追加节点，返回分配到的编号。"
                           "用于生成器盖不住的局部改动。",
            "parameters": {
                "type": "object", "required": ["coordinates"],
                "properties": {"coordinates": {
                    "type": "array",
                    "description": "每项是 [x, y, z]，单位米",
                    "items": {"type": "array", "minItems": 3, "maxItems": 3,
                              "items": {"type": "number"}}}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_members",
            "description": "在已有节点之间追加杆件。斜撑、悬挑、连系杆都用这个，"
                           "不必重建整个模型。",
            "parameters": {
                "type": "object", "required": ["pairs"],
                "properties": {
                    "pairs": {"type": "array",
                              "description": "每项是 [起点节点, 终点节点]",
                              "items": {"type": "array", "minItems": 2, "maxItems": 2,
                                        "items": {"type": "integer"}}},
                    "section": {"type": "string",
                                "description": "可留空，稍后在 Property 阶段指派"},
                    "material": {"type": "string",
                                 "description": "可留空，稍后在 Property 阶段指派"},
                    "releases_i": {"type": "array", "items": {"type": "string"},
                                   "description": "起点释放的局部自由度，如 [\"rz\"]"},
                    "releases_j": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_supports",
            "description": "给节点或节点集合施加位移/转角边界条件。"
                           "边界条件属于 Initial 阶段，不属于荷载工况。",
            "parameters": {
                "type": "object", "required": ["node_ids", "fix"],
                "properties": {
                    "node_ids": {
                        "description": "节点编号列表或已定义的节点集合名",
                        "oneOf": [
                            {"type": "array", "items": {"type": "integer"}},
                            {"type": "string"},
                        ],
                    },
                    "fix": {"type": "array", "minItems": 6, "maxItems": 6,
                            "items": {"type": "integer", "enum": [0, 1]},
                            "description": "[ux,uy,uz,rx,ry,rz]，1 表示约束"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_properties",
            "description": "给一批杆件统一指派已经定义的材料与截面；"
                           "相当于 Abaqus 的 Section Assignment。",
            "parameters": {
                "type": "object", "required": ["member_ids", "section", "material"],
                "properties": {
                    "member_ids": {
                        "description": "杆件编号列表或已定义的集合名",
                        "oneOf": [
                            {"type": "array", "items": {"type": "integer"}},
                            {"type": "string"},
                        ],
                    },
                    "section": {"type": "string"},
                    "material": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_members",
            "description": "删除若干杆件。做缺跨、拆掉某根构件对比时用。"
                           "删完可能出现悬空节点，validate_model 会查出来。",
            "parameters": {
                "type": "object", "required": ["ids"],
                "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_model",
            "description": "直接提交完整 JSON 模型，提交时立即校验。"
                           "只用于不规则结构（斜杆、缺跨、非等距、特殊支座）；规则框架请用 generate_frame。已定义的材料与截面会自动带上，不必在 model 里重复书写。",
            "parameters": {
                "type": "object", "required": ["model"],
                "properties": {"model": {
                    "type": "object",
                    "description": (
                        "完整模型。supports 里的 fix 必须是长度为 6 的 0/1 整数数组，"
                        "顺序 (ux, uy, uz, rx, ry, rz)，1 表示约束住。"
                        "不能写成 true/false，不能写成 {\"ux\":1} 这种对象，"
                        "也不能写成 [\"ux\",\"uy\"] 这种名字数组。示例：\n"
                        '{"units":"N-m-Pa",'
                        '"nodes":[{"id":1,"x":0,"y":0,"z":0},{"id":2,"x":8,"y":0,"z":0}],'
                        '"members":[{"id":1,"i":1,"j":2,"section":"BEAM","material":"STEEL"}],'
                        '"supports":[{"node":1,"fix":[1,1,1,1,0,0]},'
                        '{"node":2,"fix":[0,1,1,0,0,0]}]}\n'
                        "上例是三维简支梁：两端放开绕 y、z 的转动，但**至少一端要约束 rx**，"
                        "否则整根梁能绕自身轴自由扭转，成为机构。"),
                }},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_load_case",
            "description": "创建一个空分析工况（相当于 Abaqus 的分析步载荷容器）。",
            "parameters": {
                "type": "object", "required": ["name"],
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_nodal_load",
            "description": "在指定工况中创建或替换一个节点集中力；全零表示删除。",
            "parameters": {
                "type": "object", "required": ["node_id", "load"],
                "properties": {
                    "node_id": {"type": "integer"},
                    "load": {"type": "array", "minItems": 6, "maxItems": 6,
                             "items": {"type": "number"},
                             "description": "[Fx,Fy,Fz,Mx,My,Mz]，N 与 N·m"},
                    "case_name": {"type": "string"},
                    "name": {"type": "string",
                             "description": "载荷名，例如 Roof-CF-1"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_member_load",
            "description": "在指定工况中创建或替换一个杆件全跨均布荷载；全零表示删除。",
            "parameters": {
                "type": "object", "required": ["member_id", "load"],
                "properties": {
                    "member_id": {"type": "integer"},
                    "load": {"type": "array", "minItems": 3, "maxItems": 3,
                             "items": {"type": "number"},
                             "description": "全局 [wx,wy,wz]，N/m"},
                    "case_name": {"type": "string"},
                    "name": {"type": "string",
                             "description": "载荷名，例如 Roof-Line-1"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_member_span_load",
            "description": "在杆件上创建或按名称替换非满跨荷载："
                           "均布、梯形/三角形或杆中集中力；w1 全零表示删除。",
            "parameters": {
                "type": "object", "required": ["member_id", "kind", "w1"],
                "properties": {
                    "member_id": {"type": "integer"},
                    "kind": {"type": "string",
                             "enum": ["uniform", "trapezoid", "point"]},
                    "w1": {"type": "array", "minItems": 3, "maxItems": 3,
                           "items": {"type": "number"}},
                    "w2": {"type": "array", "minItems": 3, "maxItems": 3,
                           "items": {"type": "number"}},
                    "a": {"type": "number",
                          "description": "距杆件 i 端位置，使用当前模型长度单位"},
                    "case_name": {"type": "string"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_prescribed_displacement",
            "description": "在分析步中创建或按名称替换节点给定位移/支座沉降；"
                           "它是边界条件，不是力。全零表示删除。",
            "parameters": {
                "type": "object", "required": ["node_id", "d"],
                "properties": {
                    "node_id": {"type": "integer"},
                    "d": {"type": "array", "minItems": 6, "maxItems": 6,
                          "items": {"type": "number"},
                          "description": "[ux,uy,uz,rx,ry,rz]，当前长度单位与 rad"},
                    "case_name": {"type": "string"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_load_cases",
            "description": "一次性定义全部荷载工况与组合。多工况必须用这个，"
                           "不要把几个工况的荷载加起来当单一工况。"
                           "梁的编号从 generate_frame 返回的 beam_member_ids 取。"
                           "注意 combos 是与 cases 平级的顶层参数，"
                           "不要写进某个工况对象里面。",
            "parameters": {
                "type": "object",
                "required": ["cases"],
                "properties": {
                    "cases": {
                        "type": "array",
                        "items": {
                            "type": "object", "required": ["name"],
                            "properties": {
                                "name": {"type": "string"},
                                "member_loads": {
                                    "type": "array",
                                    "description": "均布线荷载。w 必须是三个数 [wx, wy, wz]，"
                                                   "全局坐标分量，单位 N/m。"
                                                   "重力方向是全局 -Z，所以向下 20 kN/m 写成 "
                                                   "[0, 0, -20000]。写成 [0, -20000, 0] 是"
                                                   "水平方向，结果会离谱。",
                                    "items": {
                                        "type": "object",
                                        "required": ["member", "w"],
                                        "properties": {
                                            "member": {"type": "integer"},
                                            "w": {"type": "array", "minItems": 3, "maxItems": 3,
                                                  "items": {"type": "number"}}}}},
                                "member_spans": {
                                    "type": "array",
                                    "description": "不满跨的杆间荷载，全局坐标。"
                                                   "kind=trapezoid 时 w1 是 i 端强度、w2 是 j 端强度"
                                                   "（单位 N/m，三角形就把一端写 0）；"
                                                   "kind=point 时 w1 是集中力（单位 N）、"
                                                   "a 是距 i 端的距离（单位 m，必须在 0~杆长之间）。"
                                                   "满跨均布用上面的 member_loads 更省事。",
                                    "items": {
                                        "type": "object",
                                        "required": ["member", "kind", "w1"],
                                        "properties": {
                                            "member": {"type": "integer"},
                                            "kind": {"type": "string",
                                                     "enum": ["uniform", "trapezoid", "point"]},
                                            "w1": {"type": "array", "minItems": 3, "maxItems": 3,
                                                   "items": {"type": "number"}},
                                            "w2": {"type": "array", "minItems": 3, "maxItems": 3,
                                                   "items": {"type": "number"}},
                                            "a": {"type": "number"},
                                            "note": {"type": "string",
                                                     "description": "备注，可留空"}}}},
                                "settlements": {
                                    "type": "array",
                                    "description": "支座沉降。d 是六个给定位移 "
                                                   "[ux, uy, uz, rx, ry, rz]，单位 m 与 rad，"
                                                   "**只有被约束住的方向才有效**。"
                                                   "向下沉降 10 mm 写成 [0,0,-0.01,0,0,0]。"
                                                   "这是给定位移不是荷载，静定结构不会因此产生内力。",
                                    "items": {
                                        "type": "object",
                                        "required": ["node", "d"],
                                        "properties": {
                                            "node": {"type": "integer"},
                                            "d": {"type": "array", "minItems": 6, "maxItems": 6,
                                                  "items": {"type": "number"}}}}},
                                "nodal_loads": {
                                    "type": "array",
                                    "description": "节点荷载。load 必须是六个数 "
                                                   "[Fx, Fy, Fz, Mx, My, Mz]，全局坐标，"
                                                   "单位 N 与 N·m。向下 50 kN 写成 "
                                                   "[0, 0, -50000, 0, 0, 0]。",
                                    "items": {
                                        "type": "object",
                                        "required": ["node", "load"],
                                        "properties": {
                                            "node": {"type": "integer"},
                                            "load": {"type": "array", "minItems": 6, "maxItems": 6,
                                                     "items": {"type": "number"}}}}},
                            },
                        },
                    },
                    "combos": {
                        "type": "array",
                        "items": {"type": "object", "required": ["name", "factors"],
                                  "properties": {"name": {"type": "string"},
                                                 "factors": {"type": "object"}}},
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_learning_trace",
            "description": "把当前会话的建模工具调用、参数、结果、校验状态和最终模型"
                           "导出成可复用 JSON 轨迹，用于自己的 Agent 学习与回放。",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "include_snapshots": {"type": "boolean",
                                          "description": "是否保存每一步完整模型；默认否"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_units",
            "description": "把整份模型换算到另一套单位制。换的是数值，物理量不变——"
                           "换算前后算出来的结果完全相同。只在用户明确要求时才用；"
                           "默认 N-m-Pa 就够了。",
            "parameters": {
                "type": "object", "required": ["units"],
                "properties": {"units": {"type": "string",
                                         "enum": ["N-m-Pa", "N-mm-MPa"]}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_self_weight",
            "description": "按截面面积与材料密度生成各杆自重，追加到指定工况。"
                           "材料必须有 density（kg/m³，钢约 7850）。"
                           "生成的是显式的均布荷载，会出现在模型里；"
                           "改了截面要重新调一次，否则自重还是按旧截面算的。",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string", "description": "加到哪个工况；留空取第一个"},
                    "factor": {"type": "number", "description": "分项系数，默认 1.0"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_model",
            "description": "校验当前模型的结构与语义，返回中文错误清单。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_change",
            "description": "在当前模型的深拷贝上预演一个会修改模型的 Agent 工具，"
                           "返回实体增删改、模型哈希和校验错误，当前模型与结果不变。"
                           "对已有模型做删除、批量修改或整体替换前先调用；把差异告诉用户，"
                           "等待用户下一条消息确认后才能调用真正的修改工具，"
                           "不得在同一批工具调用里预演并执行。",
            "parameters": {
                "type": "object",
                "required": ["tool", "arguments"],
                "properties": {
                    "tool": {"type": "string",
                             "description": "要预演的写工具名称"},
                    "arguments": {"type": "object",
                                  "description": "原样传给该工具的参数"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_preview",
            "description": "应用一个已预演的变更。只有用户在预演之后的新消息中明确确认，"
                           "且当前模型哈希仍与预演一致时才会成功；模型不能自行授权。",
            "parameters": {
                "type": "object",
                "required": ["preview_id"],
                "properties": {
                    "preview_id": {"type": "string",
                                   "description": "preview_change 返回的 chg_ 编号"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_analysis_mesh",
            "description": "只读预览求解器实际使用的分析网格及物理构件映射。"
                           "杆间集中力会触发自动剖分；本工具显示新增节点和分析单元，"
                           "但不求解、也不改变当前模型或已有结果。",
            "parameters": {"type": "object", "properties": {},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_model",
            "description": "求解当前模型。支持线性静力、P-Delta 二阶弹性、双线性轴向材料非线性。",
            "parameters": {"type": "object", "properties": {
                "analysis": {"type": "string", "enum": ["linear", "pdelta", "material"]},
                "increments": {"type": "integer", "minimum": 1},
                "max_iter": {"type": "integer", "minimum": 1},
                "tolerance": {"type": "number", "exclusiveMinimum": 0}
            }, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_with_abaqus",
            "description": "用 Abaqus 求解同一个模型（可选后端，需要本机装有 Abaqus）。"
                           "默认的 solve_model 快几百倍且无依赖，"
                           "只在需要与商软对标、或用户明确要求时才用这个。"
                           "一次只算一个工况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string", "description": "工况名；留空取第一个"},
                    "element": {"type": "string", "enum": ["B33", "B31"],
                                "description": "截面未给 Ay/Az 时用 B33 对标；"
                                               "启用 Timoshenko 剪切面积时用 B31"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_solvers",
            "description": "同一个模型两个后端各算一遍，逐分量给出归一化偏差。"
                           "用户问“结果可不可信”“和 Abaqus 差多少”时用这个，"
                           "不要自己去分别调两个求解器再口算差值。需要本机装有 Abaqus。",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string", "description": "工况名；留空取第一个"},
                    "element": {"type": "string", "enum": ["B33", "B31"],
                                "description": "Euler-Bernoulli 分支用 B33；"
                                               "给 Ay/Az 的 Timoshenko 分支用 B31"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_results",
            "description": "查询已求解的结果。member_forces 不给 member_id 就一次返回全部杆件，"
                           "不要一根一根查。"
                           "**校核挠跨比（L/400 之类）必须用 max_deflection**："
                           "max_displacement 只看节点，单跨一个单元时跨中根本没有节点，"
                           "报出来的数会小一个量级。"
                           "算挠跨比时把设计跨度用 span 传进来，"
                           "否则工具只能给出杆长与挠度之比，那不是跨度比。",
            "parameters": {
                "type": "object",
                "required": ["what"],
                "properties": {
                    "what": {"type": "string",
                             "enum": ["max_displacement", "max_deflection",
                                      "reactions", "member_forces"]},
                    "case": {"type": "string", "description": "工况或组合名，留空取控制工况"},
                    "member_id": {"type": "integer"},
                    "span": {"type": "number",
                             "description": "设计跨度（米），只对 max_deflection 有用。"
                                            "给了才会返回 span_over_deflection。"
                                            "程序无法自己判断跨度——斜梁由两根杆组成、"
                                            "梁又常划成几个单元，杆长不等于跨度。"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_results",
            "description": "把结果画成图片存到 results/ 目录，返回文件路径与数值摘要。"
                           "图是给用户看的；你的结论仍然只能引用 query_results 或 "
                           "query_diagram 返回的数字。",
            "parameters": {
                "type": "object",
                "required": ["kind"],
                "properties": {
                    "kind": {"type": "string",
                             "enum": ["deformed", "axial", "moment", "shear"],
                             "description": "deformed 变形图；axial 轴力图；"
                                            "moment 弯矩图 Mz；shear 剪力图 Vy"},
                    "case": {"type": "string", "description": "工况或组合名，留空取控制工况"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_diagram",
            "description": "查沿杆长的内力分布（弯矩、剪力、轴力、扭矩）。"
                           "内力在单元内解析恢复，单跨一个单元也是精确解，不必加密网格。"
                           "不给 member 就返回全结构该分量的极值及其位置。",
            "parameters": {
                "type": "object",
                "required": ["component"],
                "properties": {
                    "component": {"type": "string",
                                  "enum": ["N", "Vy", "Vz", "T", "My", "Mz"],
                                  "description": "Mz 是平面内弯曲的弯矩，最常用"},
                    "member": {"type": "integer",
                               "description": "杆件编号；留空则给全结构极值"},
                    "case": {"type": "string"},
                    "stations": {"type": "integer",
                                 "description": "采样点数，只影响返回的分布密度，"
                                                "不影响精度；默认只给极值不给全分布"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sweep",
            "description": "参数扫描：把某个参数取一串值各算一遍，返回对照表。"
                           "用户问「截面取多大才够」「加大到多少能满足 L/400」"
                           "「荷载增大结果怎么变」时用这个，**不要自己反复改模型再一次次求解**——"
                           "那会耗光轮数。给了 limit 就同时告诉你哪些值满足、"
                           "以及满足所需的最小取值。扫描不改动当前模型。",
            "parameters": {
                "type": "object",
                "required": ["what", "target", "values", "metric"],
                "properties": {
                    "what": {"type": "string",
                             "enum": ["section_property", "material_E", "load_scale"],
                             "description": "扫什么：截面特性 / 材料弹性模量 / 某工况荷载倍数"},
                    "target": {"type": "string",
                               "description": "截面名、材料名或工况名"},
                    "prop": {"type": "string", "enum": ["A", "Iy", "Iz", "J"],
                             "description": "what=section_property 时必填"},
                    "values": {"type": "array", "items": {"type": "number"},
                               "minItems": 2, "maxItems": 20,
                               "description": "要试的一串取值"},
                    "metric": {"type": "string",
                               "enum": ["max_deflection", "max_displacement",
                                        "max_abs_Mz", "buckling_factor",
                                        "first_frequency"],
                               "description": "看哪个指标随之怎么变"},
                    "case": {"type": "string", "description": "指标取哪个工况；留空取控制工况"},
                    "limit": {"type": "number",
                              "description": "限值。位移用 mm、弯矩用 kN·m、"
                                             "屈曲因子与频率用原值。"
                                             "位移和弯矩是越小越好，屈曲因子和频率是越大越好，"
                                             "工具会按指标自动判断方向"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_envelope",
            "description": "各荷载组合的内力包络，以及**每个位置由哪个组合控制**。"
                           "用户问「最不利」「哪个组合控制」「按哪个组合设计」时用这个，"
                           "不要自己把各工况的 query_diagram 结果拿来比大小——"
                           "跨中和支座常常由不同组合控制，逐点包络才分辨得出来。"
                           "定义了 combos 就自动用组合，否则用工况。",
            "parameters": {
                "type": "object",
                "required": ["component"],
                "properties": {
                    "component": {"type": "string",
                                  "enum": ["N", "Vy", "Vz", "T", "My", "Mz"]},
                    "member": {"type": "integer",
                               "description": "留空则返回全结构各分量的最不利汇总"},
                    "cases": {"type": "array", "items": {"type": "string"},
                              "description": "指定参与包络的工况或组合；留空按默认"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modal_analysis",
            "description": "自振频率与振型。需要材料有 density。"
                           "用户问「自振周期」「基频」「动力特性」「共振」时用这个。"
                           "结果是结构固有属性，与荷载无关——不要传工况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_modes": {"type": "integer",
                                  "description": "要算几阶，默认 6"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buckling_analysis",
            "description": "线性屈曲：临界荷载因子 λ 与失稳模态。"
                           "λ=3 表示把该工况的荷载放大 3 倍结构才失稳。"
                           "用户问「稳定」「屈曲」「临界荷载」「安全储备」时用这个。"
                           "**这是线性特征值屈曲，给的是上限**——真实结构有初始缺陷，"
                           "实际承载力更低，回答时必须说明这一点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string",
                             "description": "参考荷载工况；留空取控制工况"},
                    "num_modes": {"type": "integer", "description": "默认 4"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": "把这次分析写成一份完整报告：模型概况、校验、位移、反力、"
                           "内力极值与控制组合、自振特性、稳定，外加变形图与内力图。"
                           "用户说「出个报告」「整理成文档」「交作业」时用这个，"
                           "**不要自己在回答里手写一遍表格**。"
                           "报告里的每个数都来自工具返回，与你看到的一致。",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string", "description": "主工况；留空取控制工况"},
                    "fmt": {"type": "string", "enum": ["markdown", "docx", "both"],
                            "description": "默认 both"},
                    "filename": {"type": "string",
                                 "description": "文件名（不含扩展名），默认 报告"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_supports",
            "description": "当求解报告结构成为机构时，定位是哪些节点的哪些方向缺少约束。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# --------------------------------------------------------------------------- 会话

@dataclass
class ToolResult:
    ok: bool
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, **self.payload}, ensure_ascii=False)


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


@dataclass
class Session:
    """一次会话持有的全部确定性状态。模型与结果只在这里，不在提示词里。"""
    model: dict[str, Any] = field(default_factory=dict)
    frame: Any = None
    solution: Any = None
    compilation: Any = None
    result_db: Any = None
    tool_log: list[tuple[str, dict, "ToolResult"]] = field(default_factory=list)
    # 建模过程。界面拖时间轴回看每一步靠它
    history: BuildHistory = field(default_factory=BuildHistory)
    # 图片识别证据不进入求解 Domain IR，但与模型共用撤销事务。
    multimodal_provenance: dict[str, Any] | None = None
    # 预演属于对话控制状态，不属于 Domain IR，也不进入撤销历史。
    pending_change: dict[str, Any] | None = field(default=None, repr=False)
    authorized_preview_id: str | None = field(default=None, repr=False)
    _applying_preview: bool = field(default=False, repr=False)

    # --- 撤销 / 重做 ---
    #
    # 这三个方法**不是工具**，不进 TOOLS，大模型调不到它们。
    # 撤销是用户对界面的操作，不是模型该自己决定的事——让模型能撤销
    # 自己刚做的事，只会让它在犯错时反复横跳。

    def undo(self) -> bool:
        """退一步。成功返回 True。"""
        model = self.history.undo()
        if model is None:
            return False
        self._restore(model)
        self.multimodal_provenance = self.history.provenance_at(self.history.cursor)
        return True

    def redo(self) -> bool:
        model = self.history.redo()
        if model is None:
            return False
        self._restore(model)
        self.multimodal_provenance = self.history.provenance_at(self.history.cursor)
        return True

    def goto_step(self, cursor: int) -> bool:
        """跳到第 cursor 步之后的状态。cursor = -1 表示回到空模型。"""
        model = self.history.goto(cursor)
        if model is None:
            return False
        self._restore(model)
        self.multimodal_provenance = self.history.provenance_at(self.history.cursor)
        return True

    def _restore(self, model: dict[str, Any]) -> None:
        """装回一个快照。

        **必须同时清掉 frame 和 solution。** 不清的话，模型退回去了而结果
        还是旧模型的——界面会拿着一份对不上的位移去画变形图，
        而且画得出来、看不出错。这是这个功能最危险的地方。
        """
        self.model = model
        self.frame = None
        self.solution = None
        self.compilation = None
        self.result_db = None
        self.pending_change = None
        self.authorized_preview_id = None

    def receive_user_confirmation(self, user_text: str) -> bool:
        """只接受一条新的、明确的用户消息；工具调用本身不能走这条路。"""
        if self.pending_change is None:
            return False
        text = str(user_text).strip().lower()
        if text in {"取消", "取消变更", "cancel"}:
            self.pending_change = None
            self.authorized_preview_id = None
            return False
        preview_id = self.pending_change["preview_id"]
        accepted = {"确认", "确认执行", "应用变更", "confirm", "apply",
                    f"确认 {preview_id.lower()}", f"confirm {preview_id.lower()}",
                    f"apply {preview_id.lower()}"}
        if text not in accepted:
            return False
        self.authorized_preview_id = preview_id
        return True

    def preview_change(self, tool: str,
                       arguments: dict[str, Any]) -> ToolResult:
        """在隔离的 Session 上执行写工具，返回紧凑 diff，不碰当前状态。"""
        from copy import deepcopy
        from change_preview import diff_models, preview_digest

        exposed = {item["function"]["name"] for item in TOOLS}
        if tool not in MUTATING or tool not in exposed:
            return ToolResult(False, {
                "error": f"{tool!r} 不是可预演的 Agent 写工具",
                "allowed_tools": sorted(MUTATING & exposed),
            })
        if not isinstance(arguments, dict):
            return ToolResult(False, {"error": "可预演工具的 arguments 必须是对象"})

        before = deepcopy(self.model)
        sandbox = Session(model=deepcopy(before))
        attempted = getattr(sandbox, tool)(**deepcopy(arguments))
        if not attempted.ok:
            detail = attempted.payload.get("error") or attempted.payload.get("errors")
            return ToolResult(False, {
                "schema": "agent-change-preview/v1",
                "tool": tool,
                "preview_error": str(detail or "预演未通过"),
            })

        delta = diff_models(before, sandbox.model)
        errors = [str(error)[:240]
                  for error in validate_payload(sandbox.model)[:10]]
        preview_id = preview_digest(tool, arguments, delta["before_hash"],
                                    delta["after_hash"])
        payload = {
            "schema": "agent-change-preview/v1",
            "preview_id": preview_id,
            "tool": tool,
            "diff": delta,
            "validation_errors": errors,
            "analysis_ready": not errors,
            "would_invalidate_results": bool(delta["changed"]),
            "preview_result": attempted.payload,
        }
        self.pending_change = {
            "preview_id": preview_id,
            "tool": tool,
            "arguments": deepcopy(arguments),
            "before_hash": delta["before_hash"],
            "diff": deepcopy(delta),
        }
        self.authorized_preview_id = None
        return ToolResult(True, payload)

    def apply_preview(self, preview_id: str) -> ToolResult:
        """应用已由用户消息授权的预演；哈希不一致时拒绝陈旧变更。"""
        from copy import deepcopy
        from change_preview import model_digest

        pending = self.pending_change
        if pending is None or pending.get("preview_id") != preview_id:
            return ToolResult(False, {
                "error": "没有这个待确认预演，可能已失效或已被新的预演替换",
                "confirmation_required": True,
            })
        if self.authorized_preview_id != preview_id:
            return ToolResult(False, {
                "error": "需要用户在预演之后发送一条新的明确确认消息",
                "confirmation_required": True,
                "preview_id": preview_id,
            })
        if model_digest(self.model) != pending["before_hash"]:
            self.pending_change = None
            self.authorized_preview_id = None
            return ToolResult(False, {
                "error": "预演后模型已变化，该预演已失效；请重新预演",
                "stale_preview": True,
            })

        tool = pending["tool"]
        self._applying_preview = True
        try:
            applied = getattr(self, tool)(**deepcopy(pending["arguments"]))
        finally:
            self._applying_preview = False
        if not applied.ok:
            return applied
        self.pending_change = None
        self.authorized_preview_id = None
        return ToolResult(True, {
            "schema": "agent-change-application/v1",
            "preview_id": preview_id,
            "applied_tool": tool,
            "result": applied.payload,
            "diff": pending["diff"],
        })

    # --- 工具实现 ---
    @_records
    def define_materials_and_sections(self, materials, sections) -> ToolResult:
        # 单位制是项目设置，不是“定义属性”的副作用。用户若先选了毫米制，
        # 创建材料时不能又被悄悄改回 SI。
        from copy import deepcopy

        if not isinstance(materials, list) or not isinstance(sections, list):
            return ToolResult(False, {"error": "材料和截面必须分别用列表提供"})
        new_materials = deepcopy(materials)
        new_sections = deepcopy(sections)
        candidate = deepcopy(self.model)
        candidate.setdefault("units", "N-m-Pa")
        candidate["materials"] = new_materials
        candidate["sections"] = new_sections

        definition_errors = validate_definitions(
            new_materials, new_sections, allow_incomplete=True)
        if definition_errors:
            return ToolResult(False, {
                "errors": definition_errors,
                "hint": "属性定义未写入，原模型保持不变",
            })
        if (self.model.get("materials", []) == new_materials
                and self.model.get("sections", []) == new_sections):
            return ToolResult(True, {
                "materials": [m["name"] for m in new_materials],
                "sections": [s["name"] for s in new_sections],
                "no_change": True,
            })
        self.model = candidate
        self._invalidate()
        errors = validate_payload(candidate)
        return ToolResult(True, {"materials": [m["name"] for m in new_materials],
                                 "sections": [s["name"] for s in new_sections],
                                 "analysis_ready": not errors,
                                 "warnings": errors})

    @_records
    def generate_frame(self, **kwargs) -> ToolResult:
        if not self.model.get("materials"):
            kwargs.setdefault("material", "")
        if not self.model.get("sections"):
            kwargs.setdefault("column_section", "")
            kwargs.setdefault("beam_section", "")
        try:
            generated = generate_frame(**kwargs)
        except (GeneratorError, TypeError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # 生成器的公开参数固定用 m / N / N·m，产物也是 SI。当前模型若已
        # 切到毫米制，必须先把新几何和荷载整体换到毫米制再沿用已有属性；
        # 只改 units 标签会把 6 m 的跨误成 6 mm，而且不会立即报错。
        target_units = self.model.get("units", "N-m-Pa")
        if target_units != generated.get("units", "N-m-Pa"):
            try:
                generated = convert_model(generated, target_units)
            except ValueError as exc:
                return ToolResult(False, {"error": str(exc)})
        keep = {k: self.model[k] for k in ("materials", "sections") if k in self.model}
        candidate = {**generated, **keep}
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        beams = beam_member_ids(self.model, kwargs.get("beam_section", "BEAM"))
        return ToolResult(True, {
            "summary": describe(self.model),
            "beam_member_ids": beams,
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "节点与杆件已按参数展开，不要再自行罗列坐标；"
                    + ("当前是几何草稿，请在 Property 模块指派材料和截面。"
                       if errors else
                       "要分工况加载就用 set_load_cases，并引用上面的梁编号"),
        })

    @_records
    def generate_portal_frame(self, **kwargs) -> ToolResult:
        if not self.model.get("materials"):
            kwargs.setdefault("material", "")
        if not self.model.get("sections"):
            kwargs.setdefault("column_section", "")
            kwargs.setdefault("rafter_section", "")
            kwargs.setdefault("tie_section", "")
        try:
            generated = generate_portal_frame(**kwargs)
        except (GeneratorError, TypeError) as exc:
            return ToolResult(False, {"error": str(exc)})
        target_units = self.model.get("units", "N-m-Pa")
        if target_units != generated.get("units", "N-m-Pa"):
            try:
                generated = convert_model(generated, target_units)
            except ValueError as exc:
                return ToolResult(False, {"error": str(exc)})
        keep = {k: self.model[k] for k in ("materials", "sections") if k in self.model}
        candidate = {**generated, **keep}
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "summary": describe(self.model),
            "rafter_member_ids": rafter_member_ids(self.model),
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "斜梁编号按几何判别，已排除纵向系杆；分工况加载请引用它",
        })



    # --- 命名集合 ---
    #
    # **编号规则是生成器定的，用户并不知道 27 号在哪。**
    # 没有名字，"给顶层所有梁加 5 kN/m" 这句话就没有落点。
    # 有了名字，这组杆件在整个会话里都能被稳定引用。
    #
    # 集合是纯粹的命名层：求解器看不到它，工具在调用时就展开成编号了。

    @_records
    def define_set(self, name: str, member_ids=None, node_ids=None,
                   note: str | None = None, **selector) -> ToolResult:
        """给一组杆件或节点起个名字。

        两种给法，二选一：
        · 直接给 `member_ids` / `node_ids`
        · 给几何条件（同 select_members 的参数），由代码挑出来
        """
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        name = str(name).strip()
        if not name:
            return ToolResult(False, {"error": "集合名不能为空"})

        entry: dict[str, Any] = {}
        if member_ids is None and node_ids is None:
            if not selector:
                return ToolResult(False, {
                    "error": "要么直接给 member_ids / node_ids，"
                             "要么给几何条件（section / orientation / x_range 等）"})
            try:
                picked = _bent.select_members(self.model, **selector)
            except (GeneratorError, TypeError, ValueError) as exc:
                return ToolResult(False, {"error": str(exc)})
            if not picked:
                return ToolResult(False, {
                    "error": "按这些条件一根杆件都没挑到。"
                             "**空集合不要存**——存下来之后引用它的地方会静默什么都不做，"
                             "比当场报错难查得多。检查 x/y/z_range 是否框对了位置"})
            entry["members"] = picked
        else:
            known_m = {int(m["id"]) for m in self.model.get("members") or []}
            known_n = {int(n["id"]) for n in self.model.get("nodes") or []}
            if member_ids is not None:
                bad = sorted({int(k) for k in member_ids} - known_m)
                if bad:
                    return ToolResult(False, {"error": f"杆件 {bad} 不存在"})
                entry["members"] = sorted({int(k) for k in member_ids})
            if node_ids is not None:
                bad = sorted({int(k) for k in node_ids} - known_n)
                if bad:
                    return ToolResult(False, {"error": f"节点 {bad} 不存在"})
                entry["nodes"] = sorted({int(k) for k in node_ids})
        if note:
            entry["note"] = str(note)

        sets = dict(self.model.get("sets") or {})
        replaced = name in sets
        sets[name] = entry
        self.model["sets"] = sets
        return ToolResult(True, {
            "name": name, **entry, "replaced": replaced,
            "sets": sorted(sets),
            "note": "之后凡是要杆件编号的地方，都可以直接写这个名字。",
        })

    def list_sets(self) -> ToolResult:
        sets = self.model.get("sets") or {}
        return ToolResult(True, {
            "sets": {k: {"members": len(v.get("members") or []),
                         "nodes": len(v.get("nodes") or []),
                         "note": v.get("note", "")}
                     for k, v in sets.items()},
            "count": len(sets),
        })

    def _expand_members(self, value) -> list[int]:
        """把"编号列表或集合名"统一展开成编号列表。

        **允许混着写**：`["顶层梁", 12, 13]` 是合法的——用户脑子里
        本来就是"那一组，再加这两根"。
        """
        sets = self.model.get("sets") or {}
        if isinstance(value, (str, int)):
            value = [value]
        out: list[int] = []
        for item in value or []:
            if isinstance(item, str):
                entry = sets.get(item)
                if entry is None:
                    raise ValueError(
                        f"没有名为「{item}」的集合。已有的集合："
                        + ("、".join(sorted(sets)) if sets else "（一个都没有）"))
                out.extend(entry.get("members") or [])
            else:
                out.append(int(item))
        seen, unique = set(), []
        for k in out:                       # 去重但保持顺序，便于人工核对
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique

    def _expand_nodes(self, value) -> list[int]:
        """把节点编号、编号列表或节点集合名展开为去重后的编号列表。"""
        sets = self.model.get("sets") or {}
        if isinstance(value, (str, int)):
            value = [value]
        out: list[int] = []
        for item in value or []:
            if isinstance(item, str):
                entry = sets.get(item)
                if entry is None:
                    raise ValueError(
                        f"没有名为「{item}」的集合。已有集合："
                        + ("、".join(sorted(sets)) if sets else "（无）"))
                out.extend(int(node) for node in entry.get("nodes") or [])
            else:
                out.append(int(item))
        return list(dict.fromkeys(out))

    # --- 一榀 / 拉伸 / 算子 ---
    #
    # **模板是加法，算子是乘法。** 两个模板给两种结构；一榀 + 拉伸 + 五个算子
    # 覆盖的是绝大多数刚架。而且每一个都是一句话能说清的，
    # 正好落在自然语言这一侧。

    def _length_factor_from_metres(self) -> float:
        """公开建模工具统一收米；这里换成当前模型内部长度单位。"""
        return 1000.0 if self.model.get("units") == "N-mm-MPa" else 1.0

    def _scaled_ranges(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """把工具参数里的 x/y/z 区间由米换到当前模型长度单位。"""
        out = dict(kwargs)
        factor = self._length_factor_from_metres()
        for name in ("x_range", "y_range", "z_range"):
            if out.get(name) is not None:
                out[name] = [float(value) * factor for value in out[name]]
        return out

    @_records
    def generate_bent(self, **kwargs) -> ToolResult:
        if not self.model.get("materials"):
            kwargs.setdefault("material", "")
        if not self.model.get("sections"):
            kwargs.setdefault("column_section", "")
            kwargs.setdefault("beam_section", "")
        try:
            model, ids = _bent.generate_bent(**kwargs)
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # generate_bent 的公开参数固定为米，底层产物也是 SI；不能只把
        # units 标签改成毫米制，否则 12 m 会静默变成 12 mm。
        model["units"] = "N-m-Pa"
        target_units = self.model.get("units", "N-m-Pa")
        if target_units != "N-m-Pa":
            try:
                model = convert_model(model, target_units)
            except ValueError as exc:
                return ToolResult(False, {"error": str(exc)})
        keep = {k: self.model[k] for k in ("materials", "sections")
                if k in self.model}
        candidate = {**model, **keep}
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "summary": describe(self.model), **ids,
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "一榀平面刚架。屋面形状全由 profile 折线决定："
                    "平屋面两个点，双坡三个点（中间那个是屋脊），"
                    "悬挑就把折线两端伸到柱子外面。要做成空间结构，"
                    "接着调 extrude_bents 沿纵向复制。",
        })

    @_records
    def extrude_bents(self, bays, tie_section=None) -> ToolResult:
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型，请先生成一榀刚架"})
        try:
            factor = self._length_factor_from_metres()
            widths = [float(value) * factor for value in bays]
            model, info = _bent.extrude_bents(self.model, widths,
                                              tie_section=tie_section)
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        errors = validate_payload(model)
        self.model = model
        self._invalidate()
        info.pop("node_map", None)              # 太大，不往回传
        return ToolResult(True, {
            "summary": describe(self.model), **info,
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "已沿 Y 方向复制成多榀并生成连系梁。"
                    "单榀时补的面外约束已经去掉——拉伸后不再是平面刚架，"
                    "留着会把结构在 Y 方向钉死，横向刚度算出来偏刚且不报错。",
        })

    def select_members(self, **kwargs) -> ToolResult:
        """按几何条件挑杆件。**不是让用户报编号**——编号规则是生成器定的。"""
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        try:
            ids = _bent.select_members(self.model, **self._scaled_ranges(kwargs))
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        return ToolResult(True, {
            "member_ids": ids, "count": len(ids),
            "note": "区间判定用的是杆件**中点**，这样「这一跨的梁」才有确定含义。",
        })

    @_records
    def add_bracing(self, **kwargs) -> ToolResult:
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        try:
            model, info = _bent.add_bracing(
                self.model, **self._scaled_ranges(kwargs))
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        errors = validate_payload(model)
        self.model = model
        self._invalidate()
        return ToolResult(True, {
            "summary": describe(self.model), **info,
            "analysis_ready": not errors, "warnings": errors,
            "note": "支撑两端默认铰接：它靠轴力工作，按刚接建会高估它对弯矩的贡献。"
                    "评价加撑效果时要看**被撑那一榀的侧移**，不要看全结构最大位移——"
                    "只撑局部时最大值会跑到没撑的地方去，看起来像没生效。",
        })

    @_records
    def raise_nodes(self, dz, **kwargs) -> ToolResult:
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        try:
            factor = self._length_factor_from_metres()
            model, info = _bent.raise_nodes(
                self.model, dz=float(dz) * factor,
                **self._scaled_ranges(kwargs))
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # 区域移动可能把两个节点压到同一点，或把一根杆压成零长度。
        # 这种不是“尚未建完”，必须当场拒绝，不能留到求解时才报奇异。
        from draft import FrameDraft, SNAP_TOL
        draft = FrameDraft.from_model(model)
        coincident = draft.coincident_nodes(tol=SNAP_TOL / self.units.length_to_m)
        zero = [int(member["id"]) for member in model.get("members") or []
                if (draft.length(int(member["id"])) is not None
                    and draft.length(int(member["id"])) <
                    SNAP_TOL / self.units.length_to_m)]
        if coincident or zero:
            return ToolResult(False, {
                "error": "节点移动会造成重合节点或零长度杆件，改动未写入",
                "coincident_nodes": coincident, "zero_length_members": zero})
        errors = validate_payload(model)
        self.model = model
        self._invalidate()
        info["dz"] = float(dz)       # 工具回报仍按公开接口的米
        return ToolResult(True, {"summary": describe(self.model), **info,
                                 "analysis_ready": not errors,
                                 "warnings": errors})

    @_records
    def retaper(self, member_ids, section) -> ToolResult:
        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        if section not in {str(item["name"]) for item in self.model.get("sections") or []}:
            return ToolResult(False, {"error": f"截面 {section!r} 未定义"})
        try:
            member_ids = self._expand_members(member_ids)
            model, info = _bent.retaper(self.model, member_ids, section)
        except (GeneratorError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        errors = validate_payload(model)
        self.model = model
        self._invalidate()
        return ToolResult(True, {"summary": describe(self.model), **info,
                                 "analysis_ready": not errors,
                                 "warnings": errors})

    @_records
    def add_nodes(self, coordinates: list[list[float]]) -> ToolResult:
        # 节点属于几何，不应偷偷创建 Q355/COL/BEAM。允许模型处于尚未指派
        # 属性的编辑态，等创建杆件时再明确要求材料与截面。
        from copy import deepcopy

        parsed = []
        for c in coordinates:
            try:
                valid_length = len(c) == 3
            except TypeError:
                valid_length = False
            if not valid_length:
                return ToolResult(False, {"error": f"坐标必须是三个数，收到 {c}"})
            try:
                point = [float(c[0]), float(c[1]), float(c[2])]
            except (TypeError, ValueError):
                return ToolResult(False, {"error": f"坐标必须是三个数，收到 {c}"})
            if not all(np.isfinite(value) for value in point):
                return ToolResult(False, {"error": f"坐标必须是有限数，收到 {c}"})
            parsed.append(point)
        candidate = deepcopy(self.model)
        candidate.setdefault("nodes", [])
        candidate.setdefault("units", "N-m-Pa")
        candidate.setdefault("materials", [])
        candidate.setdefault("sections", [])
        candidate.setdefault("supports", [])
        candidate.setdefault("load_cases", [])
        existing = {int(n["id"]) for n in candidate["nodes"]}
        next_id = (max(existing) + 1) if existing else 1
        snap_tol = 1e-6 / self.units.length_to_m
        added = []
        reused = []
        input_ids = []
        for x, y, z in parsed:
            hit = next((int(node["id"]) for node in candidate["nodes"]
                        if float(np.linalg.norm(
                            np.array([node["x"] - x, node["y"] - y, node["z"] - z],
                                     dtype=float))) <= snap_tol), None)
            if hit is not None:
                reused.append(hit)
                input_ids.append(hit)
                continue
            candidate["nodes"].append({"id": next_id, "x": x, "y": y, "z": z})
            added.append(next_id)
            input_ids.append(next_id)
            next_id += 1
        if not added:
            return ToolResult(True, {"added_node_ids": [],
                                     "reused_node_ids": reused,
                                     "input_node_ids": input_ids,
                                     "no_change": True,
                                     "summary": describe(self.model)})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"added_node_ids": added,
                                 "reused_node_ids": reused,
                                 "input_node_ids": input_ids,
                                 "summary": describe(self.model)})

    # --- 就地修改单个对象 ---
    #
    # 属性面板靠这两个。**面板不能自己去改 model 字典**——那样就绕开了
    # 建模过程记录，改完按 Ctrl+Z 什么也不会发生，而用户完全预期能撤销。
    # 所以走 @_records，和 Agent 改的是同一条链。

    @_records
    def edit_member(self, member_id: int, section: str | None = None,
                    material: str | None = None,
                    releases_i: list[str] | None = None,
                    releases_j: list[str] | None = None,
                    ref_vector: list[float] | None | object = _UNSET,
                    offset_i: list[float] | None | object = _UNSET,
                    offset_j: list[float] | None | object = _UNSET) -> ToolResult:
        """改一根杆件的截面、材料、局部轴向或端部释放。只传要改的项。

        释放传空列表表示**取消释放**（刚接）；不传表示不动它——
        这两件事必须分得开，否则用户没法把一根铰接梁改回刚接。

        ``ref_vector`` 是局部 y 轴的全局参考向量，用于非对称截面；传
        ``None`` 或空列表可恢复自动定向。参考向量与杆轴平行会在这里拒绝，
        不把一个求解时才爆出的错误留给用户。
        """
        from copy import deepcopy

        candidate = deepcopy(self.model)
        target = None
        for m in candidate.get("members") or []:
            if int(m["id"]) == int(member_id):
                target = m
                break
        if target is None:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})

        changed: dict[str, Any] = {}
        if section is not None:
            known = {s["name"] for s in candidate.get("sections") or []}
            if section not in known:
                return ToolResult(False, {
                    "error": f"截面 {section!r} 未定义",
                    "hint": f"已定义的截面：{'、'.join(sorted(known)) or '（无）'}"})
            if section != target.get("section"):
                changed["section"] = (target.get("section"), section)
                target["section"] = section
        if material is not None:
            known = {m["name"] for m in candidate.get("materials") or []}
            if material not in known:
                return ToolResult(False, {
                    "error": f"材料 {material!r} 未定义",
                    "hint": f"已定义的材料：{'、'.join(sorted(known)) or '（无）'}"})
            if material != target.get("material"):
                changed["material"] = (target.get("material"), material)
                target["material"] = material

        rel = dict(target.get("releases") or {})
        for end, value in (("i", releases_i), ("j", releases_j)):
            if value is None:
                continue
            bad = [k for k in value if k not in LOCAL_DOF_NAMES]
            if bad:
                return ToolResult(False, {
                    "error": f"{bad} 不是有效的自由度名",
                    "hint": f"可用：{'、'.join(LOCAL_DOF_NAMES)}"})
            before = list(rel.get(end) or [])
            if value:
                rel[end] = list(value)
            else:
                rel.pop(end, None)          # 空列表 = 取消释放
            if before != list(value):
                changed[f"releases_{end}"] = (before, list(value))
        if rel:
            target["releases"] = rel
        else:
            target.pop("releases", None)

        if ref_vector is not _UNSET:
            before = target.get("ref_vector")
            if ref_vector is None or ref_vector == []:
                vector = None
            else:
                try:
                    vector = [float(v) for v in ref_vector]
                except (TypeError, ValueError):
                    return ToolResult(False, {
                        "error": "梁方向参考向量必须是三个数字，例如 [0, 0, 1]"})
                if len(vector) != 3 or not all(np.isfinite(vector)):
                    return ToolResult(False, {
                        "error": "梁方向参考向量必须是三个有限数字，例如 [0, 0, 1]"})
                if float(np.linalg.norm(vector)) < 1e-12:
                    return ToolResult(False, {"error": "梁方向参考向量不能是零向量"})
                nodes = {int(n["id"]): n for n in candidate.get("nodes") or []}
                ni, nj = nodes.get(int(target["i"])), nodes.get(int(target["j"]))
                if ni is None or nj is None:
                    return ToolResult(False, {"error": "杆件端点不存在，无法设置梁方向"})
                axis = np.array([float(nj[k]) - float(ni[k]) for k in ("x", "y", "z")])
                if float(np.linalg.norm(np.cross(axis, vector))) < 1e-12:
                    return ToolResult(False, {
                        "error": "梁方向参考向量不能与杆件轴线平行",
                        "hint": "请换一个垂直于杆件轴线的方向，例如 [0, 0, 1]。"})
            if before != vector:
                changed["ref_vector"] = (before, vector)
                if vector is None:
                    target.pop("ref_vector", None)
                else:
                    target["ref_vector"] = vector

        for key, raw in (("offset_i", offset_i), ("offset_j", offset_j)):
            if raw is _UNSET:
                continue
            before = target.get(key)
            if raw is None or raw == []:
                vector = None
            else:
                try:
                    vector = [float(v) for v in raw]
                except (TypeError, ValueError):
                    return ToolResult(False, {"error": f"{key} 必须是三个数字"})
                if len(vector) != 3 or not all(np.isfinite(vector)):
                    return ToolResult(False, {"error": f"{key} 必须是三个有限数字"})
                if np.linalg.norm(vector) <= 1e-14:
                    vector = None
            if before != vector:
                changed[key] = (before, vector)
                if vector is None:
                    target.pop(key, None)
                else:
                    target[key] = vector

        if not changed:
            # **值没变就当没发生。** 属性面板每次重填表单都会把当前值
            # 再写一遍控件；照记不误的话，建模过程里会堆满"修改杆件 3："
            # 这种空步骤，撤销一次什么都不动，用户会以为撤销坏了
            return ToolResult(True, {"member": int(member_id), "changed": {},
                                     "summary": describe(self.model),
                                     "no_change": True,
                                     "note": "值与原来相同，未作改动"})
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "member": int(member_id),
            "changed": {k: {"was": a, "now": b} for k, (a, b) in changed.items()},
            "summary": describe(self.model),
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "改动已生效，原有分析结果已失效，请重新求解。",
        })

    @_records
    def edit_node(self, node_id: int, x: float | None = None,
                  y: float | None = None, z: float | None = None,
                  fix: list[int] | None = None) -> ToolResult:
        """改一个节点的坐标或约束。只传要改的项。

        `fix` 是六个 0/1，顺序 ux uy uz rx ry rz；传空列表表示**取消全部约束**。
        """
        from copy import deepcopy

        candidate = deepcopy(self.model)
        target = None
        for n in candidate.get("nodes") or []:
            if int(n["id"]) == int(node_id):
                target = n
                break
        if target is None:
            return ToolResult(False, {"error": f"节点 {node_id} 不存在"})

        changed: dict[str, Any] = {}
        for key, value in (("x", x), ("y", y), ("z", z)):
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                return ToolResult(False, {"error": f"节点坐标 {key} 必须是数字"})
            if not np.isfinite(number):
                return ToolResult(False, {"error": f"节点坐标 {key} 必须是有限数"})
            if number != float(target[key]):
                changed[key] = (float(target[key]), number)
                target[key] = number

        if any(value is not None for value in (x, y, z)):
            point = np.array([target[k] for k in ("x", "y", "z")], dtype=float)
            for other in candidate.get("nodes") or []:
                if int(other["id"]) == int(node_id):
                    continue
                other_point = np.array([other[k] for k in ("x", "y", "z")], dtype=float)
                if float(np.linalg.norm(point - other_point)) < 1e-9:
                    return ToolResult(False, {
                        "error": f"移动后会与节点 {other['id']} 重合，改动未写入"})

        if fix is not None:
            if fix and (len(fix) != 6 or any(v not in (0, 1) for v in fix)):
                return ToolResult(False, {
                    "error": "fix 必须是六个 0 或 1，顺序 ux uy uz rx ry rz"})
            supports = [s for s in candidate.get("supports") or []
                        if int(s["node"]) != int(node_id)]
            before_entry = next((s for s in candidate.get("supports") or []
                                 if int(s["node"]) == int(node_id)), None)
            before = before_entry["fix"] if before_entry else None
            if fix:
                entry = {"node": int(node_id), "fix": [int(v) for v in fix]}
                if before_entry and before_entry.get("name"):
                    entry["name"] = before_entry["name"]
                supports.append(entry)
            if before != (list(fix) if fix else None):
                changed["fix"] = (before, list(fix) if fix else None)
            candidate["supports"] = sorted(supports, key=lambda s: s["node"])

        if not changed:
            return ToolResult(True, {"node": int(node_id), "changed": {},
                                     "summary": describe(self.model),
                                     "no_change": True,
                                     "note": "值与原来相同，未作改动"})
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "node": int(node_id),
            "changed": {k: {"was": a, "now": b} for k, (a, b) in changed.items()},
            "summary": describe(self.model),
            "analysis_ready": not errors,
            "warnings": errors,
            "note": "改动已生效，原有分析结果已失效，请重新求解。",
        })

    @_records
    def add_members(self, pairs: list[list[int]], section: str | None = None,
                    material: str | None = None,
                    releases_i: list[str] | None = None,
                    releases_j: list[str] | None = None) -> ToolResult:
        from copy import deepcopy

        if not pairs:
            return ToolResult(False, {"error": "至少需要一对节点来创建杆件"})
        bad_releases = [value for value in (releases_i or []) + (releases_j or [])
                        if value not in LOCAL_DOF_NAMES]
        if bad_releases:
            return ToolResult(False, {
                "error": f"{bad_releases} 不是有效的局部自由度",
                "hint": f"可用：{'、'.join(LOCAL_DOF_NAMES)}"})
        known = {int(n["id"]) for n in self.model.get("nodes", [])}
        nodes = {int(n["id"]): n for n in self.model.get("nodes", [])}
        existing_members = self.model.get("members") or []
        known_sections = {str(s["name"]) for s in self.model.get("sections") or []}
        known_materials = {str(m["name"]) for m in self.model.get("materials") or []}
        if section and known_sections and section not in known_sections:
            return ToolResult(False, {
                "error": f"截面 {section!r} 未定义",
                "hint": "请选择已有截面，或留空后稍后统一指派"})
        if material and known_materials and material not in known_materials:
            return ToolResult(False, {
                "error": f"材料 {material!r} 未定义",
                "hint": "请选择已有材料，或留空后稍后统一指派"})
        if section is None:
            if existing_members and existing_members[0].get("section") in known_sections:
                section = existing_members[0]["section"]
            elif len(known_sections) == 1:
                section = next(iter(known_sections))
        if material is None:
            if existing_members and existing_members[0].get("material") in known_materials:
                material = existing_members[0]["material"]
            elif len(known_materials) == 1:
                material = next(iter(known_materials))
        section_name = str(section or "")
        mat = str(material or "")
        next_id = (max(int(m["id"]) for m in existing_members) + 1) if existing_members else 1
        seen = {(min(int(m["i"]), int(m["j"])), max(int(m["i"]), int(m["j"])))
                for m in existing_members}
        parsed_pairs = []
        for pair in pairs:
            try:
                valid_length = len(pair) == 2
            except TypeError:
                valid_length = False
            if not valid_length:
                return ToolResult(False, {"error": f"每项必须是两个节点编号，收到 {pair}"})
            try:
                i, j = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                return ToolResult(False, {"error": f"节点编号必须是整数，收到 {pair}"})
            missing = [n for n in (i, j) if n not in known]
            if missing:
                return ToolResult(False, {"error": f"节点 {missing} 不存在",
                                          "hint": "先用 add_nodes 建节点"})
            if i == j:
                return ToolResult(False, {"error": f"杆件起点和终点不能都是节点 {i}"})
            a, b = nodes[i], nodes[j]
            length = float(np.linalg.norm(np.array(
                [b[k] - a[k] for k in ("x", "y", "z")], dtype=float)))
            if length < 1e-6 / self.units.length_to_m:
                return ToolResult(False, {
                    "error": f"节点 {i} 与 {j} 重合或距离过小，不能创建零长度杆件"})
            key = (min(i, j), max(i, j))
            if key in seen:
                return ToolResult(False, {"error": f"节点 {i} 与 {j} 之间已经有杆件"})
            seen.add(key)
            parsed_pairs.append((i, j))

        candidate = deepcopy(self.model)
        candidate.setdefault("members", [])
        added = []
        for i, j in parsed_pairs:
            entry: dict[str, Any] = {"id": next_id, "i": i, "j": j,
                                     "section": section_name, "material": mat}
            rel = {}
            if releases_i:
                rel["i"] = list(releases_i)
            if releases_j:
                rel["j"] = list(releases_j)
            if rel:
                entry["releases"] = rel
            candidate["members"].append(entry)
            added.append(next_id)
            next_id += 1
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"added_member_ids": added,
                                 "summary": describe(self.model),
                                 "analysis_ready": not errors,
                                 "warnings": errors})

    @_records
    def set_supports(self, node_ids, fix: list[int],
                     name: str = "BC-1") -> ToolResult:
        """给节点或节点集合统一施加位移边界；边界属于 Initial 阶段。"""
        from copy import deepcopy

        try:
            ids = self._expand_nodes(node_ids)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        if not ids:
            return ToolResult(False, {"error": "至少选择一个节点"})
        if len(fix) != 6 or any(value not in (0, 1) for value in fix):
            return ToolResult(False, {
                "error": "fix 必须是六个 0 或 1，顺序 ux uy uz rx ry rz"})
        known = {int(node["id"]) for node in self.model.get("nodes") or []}
        missing = sorted(set(ids) - known)
        if missing:
            return ToolResult(False, {"error": f"节点 {missing} 不存在"})

        candidate = deepcopy(self.model)
        selected = set(ids)
        supports = [support for support in candidate.get("supports") or []
                    if int(support["node"]) not in selected]
        if any(fix):
            bc_name = str(name or "BC-1").strip() or "BC-1"
            supports.extend({"name": bc_name, "node": node,
                             "fix": [int(v) for v in fix]}
                            for node in ids)
        candidate["supports"] = sorted(supports, key=lambda item: int(item["node"]))
        if candidate.get("supports") == self.model.get("supports", []):
            return ToolResult(True, {"nodes": ids, "name": name,
                                     "step": "Initial", "no_change": True})
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "nodes": ids, "fix": list(fix), "name": str(name or "BC-1"),
            "step": "Initial", "analysis_ready": not errors,
            "warnings": errors, "summary": describe(self.model),
        })

    @_records
    def assign_properties(self, member_ids, section: str,
                          material: str) -> ToolResult:
        """给一批杆件统一指派属性；对应 Abaqus 的 Section Assignment。"""
        from copy import deepcopy

        try:
            ids = self._expand_members(member_ids)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        if not ids:
            return ToolResult(False, {"error": "至少选择一根要指派的杆件"})
        sections = {str(item["name"]) for item in self.model.get("sections") or []}
        materials = {str(item["name"]) for item in self.model.get("materials") or []}
        if section not in sections:
            return ToolResult(False, {"error": f"截面 {section!r} 未定义"})
        if material not in materials:
            return ToolResult(False, {"error": f"材料 {material!r} 未定义"})
        known = {int(item["id"]) for item in self.model.get("members") or []}
        missing = sorted(set(ids) - known)
        if missing:
            return ToolResult(False, {"error": f"杆件 {missing} 不存在"})

        candidate = deepcopy(self.model)
        changed = []
        selected = set(ids)
        for member in candidate.get("members") or []:
            if int(member["id"]) not in selected:
                continue
            if (member.get("section") != section
                    or member.get("material") != material):
                member["section"] = section
                member["material"] = material
                changed.append(int(member["id"]))
        if not changed:
            return ToolResult(True, {"members": sorted(selected), "no_change": True,
                                     "analysis_ready": not validate_payload(candidate)})
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {
            "members": changed, "section": section, "material": material,
            "analysis_ready": not errors, "warnings": errors,
            "summary": describe(self.model),
        })

    @_records
    def remove_members(self, ids) -> ToolResult:
        from copy import deepcopy

        if not self.model.get("members"):
            return ToolResult(False, {"error": "当前没有模型"})
        try:
            ids = self._expand_members(ids)     # 允许直接写集合名
        except (TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        try:
            drop = {int(k) for k in ids}
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "杆件编号必须是整数列表或集合名"})
        if not drop:
            return ToolResult(False, {"error": "至少选择一个要删除的杆件"})
        known = {int(m["id"]) for m in self.model["members"]}
        missing = sorted(drop - known)
        if missing:
            return ToolResult(False, {"error": f"杆件 {missing} 不存在"})
        candidate = deepcopy(self.model)
        candidate["members"] = [m for m in candidate["members"]
                                if int(m["id"]) not in drop]

        # 清理走 changes.apply，**不在这里手写**。
        #
        # 原来这里按记忆逐条清，只过滤了 member_loads——而荷载有四类
        # （nodal_loads / member_loads / member_spans / settlements），
        # 而且顶层和 load_cases 两处都能写。自重恰好写的是 member_spans，
        # 于是"加自重 → 删一根柱"会留下指向已删杆件的荷载，
        # 校验在下一步才报错，指向的却不是删除这一步。
        report = _changes.apply(candidate, _changes.Change.GEOMETRY,
                                dropped_members=drop)

        errors = validate_payload(candidate)
        payload = {"removed": sorted(drop),
                   "removed_orphan_nodes": report.get("removed_orphan_nodes", []),
                   "summary": describe(candidate), **report}
        self.model = candidate
        self._invalidate()
        payload["analysis_ready"] = not errors
        payload["warnings"] = errors
        bits = []
        if report.get("removed_orphan_nodes"):
            bits.append(f"清除了 {len(report['removed_orphan_nodes'])} 个孤立节点"
                        "（留着会让刚度矩阵出现整行零，求解报奇异）")
        if report.get("pruned_loads"):
            total = sum(report["pruned_loads"].values())
            bits.append(f"清除了 {total} 条指向已删对象的荷载")
        if report.get("emptied_sets"):
            bits.append(f"集合 {'、'.join(report['emptied_sets'])} 已被删空，一并移除")
        if bits:
            payload["note"] = "顺带" + "；".join(bits) + "。"
        return ToolResult(True, payload)

    @_records
    def remove_nodes(self, ids) -> ToolResult:
        """删除节点。连接到这些节点的杆件会一并删除（否则会留下指向不存在节点的杆）。"""
        from copy import deepcopy

        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "当前没有模型"})
        if isinstance(ids, (str, int)):
            ids = [ids]
        try:
            drop = {int(k) for k in ids}
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "节点编号必须是整数列表"})
        if not drop:
            return ToolResult(False, {"error": "至少选择一个要删除的节点"})
        known = {int(n["id"]) for n in self.model["nodes"]}
        missing = sorted(drop - known)
        if missing:
            return ToolResult(False, {"error": f"节点 {missing} 不存在"})
        candidate = deepcopy(self.model)
        connected = [int(m["id"]) for m in candidate.get("members", [])
                     if int(m["i"]) in drop or int(m["j"]) in drop]
        connected_set = set(connected)
        candidate["members"] = [m for m in candidate.get("members", [])
                                if int(m["id"]) not in connected_set]
        candidate["nodes"] = [n for n in candidate["nodes"]
                              if int(n["id"]) not in drop]
        candidate["supports"] = [s for s in candidate.get("supports", [])
                                  if int(s["node"]) not in drop]
        report = _changes.apply(
            candidate, _changes.Change.GEOMETRY,
            dropped_members=connected_set, dropped_nodes=drop)
        errors = validate_payload(candidate)
        payload = {"removed": sorted(drop), "removed_connected_members": connected,
                   "removed_orphan_nodes": report.get("removed_orphan_nodes", []),
                   "summary": describe(candidate), **report}
        self.model = candidate
        self._invalidate()
        payload["analysis_ready"] = not errors
        payload["warnings"] = errors
        return ToolResult(True, payload)

    @_records
    def set_load_cases(self, cases: list[dict], combos: list[dict] | None = None) -> ToolResult:
        """一次性定义全部荷载工况与组合，覆盖此前的荷载定义。

        多工况务必用这个，不要把几个工况的荷载先加起来当单一工况——
        那样组合结果虽对，却拿不到各工况的分项内力，也无法做包络。
        """
        if not cases:
            return ToolResult(False, {"error": "至少要给一个工况"})
        from copy import deepcopy

        candidate = deepcopy(self.model)
        for key in ("nodal_loads", "member_loads", "member_spans", "settlements"):
            candidate.pop(key, None)
        candidate["load_cases"] = deepcopy(cases)
        candidate["combos"] = deepcopy(combos or [])
        errors = validate_payload(candidate)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "荷载定义未写入，原模型保持不变"})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"cases": [c.get("name") for c in cases],
                                 "combos": [c.get("name") for c in (combos or [])]})

    @_records
    def add_load_case(self, name: str) -> ToolResult:
        """新建一个空荷载工况。"""
        from copy import deepcopy

        name = str(name).strip()
        if not name:
            return ToolResult(False, {"error": "工况名称不能为空"})
        candidate = deepcopy(self.model)
        candidate.setdefault("load_cases", [])
        if any(c.get("name") == name for c in candidate["load_cases"]):
            return ToolResult(False, {"error": f"工况 {name!r} 已存在"})
        candidate["load_cases"].append({"name": name, "nodal_loads": [],
                                          "member_loads": [], "member_spans": [],
                                          "settlements": []})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"case": name, "cases": [c["name"] for c in self.model["load_cases"]]})

    @_records
    def set_nodal_load(self, node_id: int, load: list[float],
                       case_name: str | None = None,
                       name: str | None = None) -> ToolResult:
        """设置节点集中力。load 是 [Fx,Fy,Fz,Mx,My,Mz]，单位为 N 与 N·当前长度单位。
        传全零列表表示删除该节点的荷载。"""
        from copy import deepcopy

        if len(load) != 6:
            return ToolResult(False, {"error": "load 必须是六个数 [Fx,Fy,Fz,Mx,My,Mz]"})
        try:
            values = [float(value) for value in load]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "集中力六个分量必须都是数字"})
        if not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "集中力分量必须是有限数"})
        node_id = int(node_id)
        if node_id not in {int(n["id"]) for n in self.model.get("nodes", [])}:
            return ToolResult(False, {"error": f"节点 {node_id} 不存在"})
        candidate = deepcopy(self.model)
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                         "member_loads": [], "member_spans": [],
                                         "settlements": []}]
        case_name = case_name or candidate["load_cases"][0]["name"]
        case = next((c for c in candidate["load_cases"] if c["name"] == case_name), None)
        if case is None:
            return ToolResult(False, {"error": f"工况 {case_name!r} 不存在"})
        same_target = [entry for entry in case.get("nodal_loads", [])
                       if int(entry["node"]) == node_id]
        if name is None and len(same_target) > 1:
            return ToolResult(False, {
                "error": f"节点 {node_id} 上有多条载荷，请给出要编辑的载荷名称",
                "names": [entry.get("name") for entry in same_target]})
        old = same_target[0] if same_target else None
        load_name = str(name or (old or {}).get("name") or
                        f"CF-Node-{node_id}").strip()
        if not load_name:
            return ToolResult(False, {"error": "载荷名称不能为空"})
        entries = [entry for entry in case.get("nodal_loads", [])
                   if not (entry.get("name") == load_name
                           or (old is entry and not entry.get("name")))]
        for key in ("member_loads", "member_spans"):
            case[key] = [entry for entry in case.get(key, [])
                         if entry.get("name") != load_name]
        if any(value != 0 for value in values):
            entries.append({"name": load_name, "node": node_id, "load": values})
        case["nodal_loads"] = entries
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": load_name,
                                 "node": node_id, "case": case_name, "load": values})

    @_records
    def set_member_load(self, member_id: int, load: list[float],
                        case_name: str | None = None,
                        name: str | None = None) -> ToolResult:
        """设置杆件均布荷载。load 是三个数 [wx,wy,wz]，单位为 N/当前长度单位。
        传全零列表表示删除该杆件的荷载。"""
        from copy import deepcopy

        if len(load) != 3:
            return ToolResult(False, {"error": "load 必须是三个数 [wx,wy,wz]"})
        try:
            values = [float(value) for value in load]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "均布荷载三个分量必须都是数字"})
        if not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "均布荷载分量必须是有限数"})
        member_id = int(member_id)
        if member_id not in {int(m["id"]) for m in self.model.get("members", [])}:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})
        candidate = deepcopy(self.model)
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                         "member_loads": [], "member_spans": [],
                                         "settlements": []}]
        case_name = case_name or candidate["load_cases"][0]["name"]
        case = next((c for c in candidate["load_cases"] if c["name"] == case_name), None)
        if case is None:
            return ToolResult(False, {"error": f"工况 {case_name!r} 不存在"})
        same_target = [entry for entry in case.get("member_loads", [])
                       if int(entry["member"]) == member_id]
        if name is None and len(same_target) > 1:
            return ToolResult(False, {
                "error": f"杆件 {member_id} 上有多条均布荷载，请给出要编辑的载荷名称",
                "names": [entry.get("name") for entry in same_target]})
        old = same_target[0] if same_target else None
        load_name = str(name or (old or {}).get("name") or
                        f"Line-Member-{member_id}").strip()
        if not load_name:
            return ToolResult(False, {"error": "载荷名称不能为空"})
        entries = [entry for entry in case.get("member_loads", [])
                   if not (entry.get("name") == load_name
                           or (old is entry and not entry.get("name")))]
        for key in ("nodal_loads", "member_spans"):
            case[key] = [entry for entry in case.get(key, [])
                         if entry.get("name") != load_name]
        if any(value != 0 for value in values):
            entries.append({"name": load_name, "member": member_id, "w": values})
        case["member_loads"] = entries
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": load_name,
                                 "member": member_id, "case": case_name, "load": values})

    def _load_case_in(self, candidate: dict[str, Any],
                      case_name: str | None) -> tuple[dict[str, Any] | None, str]:
        """在候选模型中取分析步；增量载荷工具共用，避免默认工况逻辑漂移。"""
        if not candidate.get("load_cases"):
            candidate["load_cases"] = [{"name": "Load-1", "nodal_loads": [],
                                          "member_loads": [], "member_spans": [],
                                          "settlements": []}]
        chosen = case_name or candidate["load_cases"][0]["name"]
        case = next((item for item in candidate["load_cases"]
                     if item.get("name") == chosen), None)
        return case, str(chosen)

    @_records
    def set_member_span_load(self, member_id: int, kind: str, w1: list[float],
                             w2: list[float] | None = None,
                             a: float | None = None,
                             case_name: str | None = None,
                             name: str | None = None) -> ToolResult:
        """创建梯形、三角形或杆中集中力，按名称编辑而不是重复叠加。"""
        from copy import deepcopy

        kind = str(kind)
        if kind not in {"uniform", "trapezoid", "point"}:
            return ToolResult(False, {"error": "kind 只能是 uniform / trapezoid / point"})
        try:
            start = [float(value) for value in w1]
            end = [float(value) for value in (w2 if w2 is not None else w1)]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "w1/w2 必须是三个数字"})
        if len(start) != 3 or len(end) != 3 or not all(
                np.isfinite(value) for value in start + end):
            return ToolResult(False, {"error": "w1/w2 必须是三个有限数字"})
        member_id = int(member_id)
        member = next((item for item in self.model.get("members") or []
                       if int(item["id"]) == member_id), None)
        if member is None:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})
        position = None
        if kind == "point":
            try:
                position = float(a)
            except (TypeError, ValueError):
                return ToolResult(False, {"error": "杆中集中力必须给出距 i 端位置 a"})
            nodes = {int(node["id"]): node for node in self.model.get("nodes") or []}
            ni, nj = nodes[int(member["i"])], nodes[int(member["j"])]
            length = float(np.linalg.norm(np.array(
                [nj[key] - ni[key] for key in ("x", "y", "z")], dtype=float)))
            if not np.isfinite(position) or not 0 <= position <= length:
                return ToolResult(False, {
                    "error": f"位置 a 必须在 0 到杆长 {length:g} 之间（当前模型长度单位）"})
        load_name = str(name or f"{kind.title()}-Member-{member_id}").strip()
        if not load_name:
            return ToolResult(False, {"error": "载荷名称不能为空"})

        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})
        entries = [entry for entry in case.get("member_spans", [])
                   if entry.get("name") != load_name]
        for key in ("nodal_loads", "member_loads"):
            case[key] = [entry for entry in case.get(key, [])
                         if entry.get("name") != load_name]
        values_for_delete = start + (end if kind == "trapezoid" else [])
        if any(value != 0 for value in values_for_delete):
            entry: dict[str, Any] = {"name": load_name, "member": member_id,
                                     "kind": kind, "w1": start}
            if kind == "trapezoid":
                entry["w2"] = end
            if kind == "point":
                entry["a"] = position
            entries.append(entry)
        case["member_spans"] = entries
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": load_name, "member": member_id,
                                 "case": chosen, "kind": kind,
                                 "analysis_ready": not errors, "warnings": errors})

    @_records
    def set_prescribed_displacement(self, node_id: int, d: list[float],
                                    case_name: str | None = None,
                                    name: str | None = None) -> ToolResult:
        """创建分析步内给定位移；只有 Initial 中已约束的方向会参与求解。"""
        from copy import deepcopy

        try:
            values = [float(value) for value in d]
        except (TypeError, ValueError):
            return ToolResult(False, {"error": "d 必须是六个数字"})
        if len(values) != 6 or not all(np.isfinite(value) for value in values):
            return ToolResult(False, {"error": "d 必须是六个有限数字"})
        node_id = int(node_id)
        if node_id not in {int(node["id"]) for node in self.model.get("nodes") or []}:
            return ToolResult(False, {"error": f"节点 {node_id} 不存在"})
        bc_name = str(name or f"Displacement-Node-{node_id}").strip()
        if not bc_name:
            return ToolResult(False, {"error": "边界条件名称不能为空"})
        candidate = deepcopy(self.model)
        case, chosen = self._load_case_in(candidate, case_name)
        if case is None:
            return ToolResult(False, {"error": f"工况 {chosen!r} 不存在"})
        entries = [entry for entry in case.get("settlements", [])
                   if entry.get("name") != bc_name and int(entry["node"]) != node_id]
        if any(value != 0 for value in values):
            entries.append({"name": bc_name, "node": node_id, "d": values})
        case["settlements"] = entries
        errors = validate_payload(candidate)
        self.model = candidate
        self._invalidate()
        return ToolResult(True, {"name": bc_name, "node": node_id,
                                 "case": chosen, "d": values,
                                 "analysis_ready": not errors, "warnings": errors,
                                 "note": "给定位移属于分析步边界条件，不计入外力。"})

    @_records
    def set_model(self, model: dict[str, Any]) -> ToolResult:
        """提交完整模型。已定义的材料与截面会自动带上，不必重复书写。

        （早期版本这里是整体替换，模型以为材料定义过一次就够了，结果被
        "materials is required" 反复打回，陷入死循环。与 generate_frame 保持一致。）
        """
        from copy import deepcopy

        merged = deepcopy(model)
        legacy = "schema_version" not in merged
        carried = []
        for key in ("materials", "sections"):
            if key not in merged and key in self.model:
                merged[key] = self.model[key]
                carried.append(key)
        merged.setdefault("units", self.model.get("units", "N-m-Pa"))
        # 立即校验。早期版本只存不验，模型提交坏模型却拿到"成功"，
        # 错误要到下一个工具才炸，它就以为是下一个工具的参数写错了，白转好几轮。
        errors = validate_payload(merged)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "完整模型未写入，原模型保持不变。"
                                              "也可改用 add_nodes/add_members 分阶段建几何"})
        self.model = migrate_payload(merged)
        self._invalidate()
        payload: dict[str, Any] = {
            "summary": describe(self.model),
            "schema_version": CURRENT_SCHEMA_VERSION,
        }
        if legacy:
            payload["migrated_from"] = 0
        if carried:
            payload["note"] = f"沿用了已定义的 {'、'.join(carried)}，不必重复提交"
        return ToolResult(True, payload)

    def open_draft(self) -> "Any":
        """从当前模型开一份草稿。**不记账**——开草稿本身什么都没改。

        草稿是「编辑中」的模型：可以悬空、可以没支座、可以断成两片。
        改完调 `apply_draft` 才落到正式模型上，中途丢掉不留任何痕迹。
        """
        from draft import FrameDraft
        return FrameDraft.from_model(self.model)

    @_records
    def apply_draft(self, draft: "Any", note: str | None = None) -> ToolResult:
        """把草稿提交为正式模型。**草稿进入模型只有这一道门。**

        门后面走的是和其它工具完全一样的一套：校验、失效、清理、记账。
        草稿要是能绕过去自己写 `self.model`，手工画的图就会既不进撤销链、
        也不触发结果失效——画完还显示着上一版的变形图。
        """
        blocking = draft.finish_diagnostics()

        before_m = {int(m["id"]) for m in self.model.get("members") or []}
        candidate = draft.to_model()
        after_m = {int(m["id"]) for m in candidate.get("members") or []}

        # 草稿只动几何，但删掉的杆件上可能挂着荷载。**清理走同一张影响表**，
        # 不在这里另写一遍——另写的那一份就是下一个漏掉某类荷载的地方
        report = _changes.apply(candidate, _changes.Change.GEOMETRY,
                                dropped_members=before_m - after_m)

        errors = validate_payload(candidate)

        self.model = candidate
        self._invalidate()
        payload: dict[str, Any] = {
            "summary": describe(self.model),
            "note": note or "手工修改已提交，原有分析结果已失效，请重新求解。",
            "analysis_ready": not errors,
            "warnings": [n.message for n in blocking] or errors,
        }
        payload.update(report)
        return ToolResult(True, payload)

    def apply_multimodal_draft(
            self, candidate_model: dict[str, Any], provenance: dict[str, Any],
            *, expected_baseline_hash: str) -> ToolResult:
        """原子写入识别模型与侧车状态；失败不改状态也不记历史。"""
        from copy import deepcopy
        from change_preview import model_digest
        from draft import FrameDraft

        if model_digest(self.model) != expected_baseline_hash:
            return ToolResult(False, {"error": "提交基线已变化，请重新预演"})
        before = {
            "model": deepcopy(self.model), "frame": self.frame,
            "solution": self.solution, "compilation": self.compilation,
            "result_db": self.result_db,
            "provenance": deepcopy(self.multimodal_provenance),
            "history": deepcopy(self.history),
        }
        try:
            draft = FrameDraft.from_model(candidate_model)
            # 调用未装饰的实现，避免内层先记一条 apply_draft 历史。
            result = Session.apply_draft.__wrapped__(
                self, draft, "图片识别草稿已提交；分析结果已失效，请按需求解。")
            if not result.ok:
                raise ValueError(result.payload.get("error") or "草稿提交失败")
            self.multimodal_provenance = deepcopy(provenance)
            self.history.record(
                "apply_multimodal_draft",
                {"expected_baseline_hash": expected_baseline_hash}, True,
                result.payload, self.model, self.multimodal_provenance)
            return result
        except Exception as exc:  # noqa: BLE001 - 原子边界必须兜住全部异常
            self.model = before["model"]
            self.frame = before["frame"]
            self.solution = before["solution"]
            self.compilation = before["compilation"]
            self.result_db = before["result_db"]
            self.multimodal_provenance = before["provenance"]
            self.history = before["history"]
            return ToolResult(False, {"error": str(exc)})

    @_records
    def set_units(self, units: str) -> ToolResult:
        """换算整份模型到另一套单位制。"""
        before = self.model.get("units", "N-m-Pa")
        source = self.model or {"units": before}
        try:
            converted = convert_model(source, units)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(False, {"error": str(exc)})
        # 编辑态允许只有节点、还没有杆件/属性；convert_model 对所有已存在
        # 的物理量逐项换算即可，完整合法性仍由提交/求解阶段统一把关。
        self.model = converted
        self._invalidate()
        U = self.units
        return ToolResult(True, {
            "units": units, "was": before,
            "note": f"长度用 {U.length}、密度用 {U.density}、"
                    f"自重 g 取 {U.gravity}。查询结果仍以 mm/kN/kN·m 返回，"
                    "换算前后的数完全相同。"})

    @_records
    def add_self_weight(self, case: str | None = None,
                        factor: float = 1.0) -> ToolResult:
        """按 ρ·A·g 生成各杆自重，作为均布荷载追加到某个工况。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "模型不合法，先修好再加自重"})
        frame = from_dict(self.model)
        loads = self_weight_loads(frame, factor=factor)
        if not loads:
            missing = sorted({m["material"] for m in self.model["members"]})
            return ToolResult(False, {
                "error": "没有一种材料定义了 density，自重为零",
                "hint": f"给材料 {missing} 加上 density（kg/m³，钢约 7850）"
                        "再调一次；define_materials_and_sections 可以带这个字段"})
        cases = self.model.setdefault("load_cases", [])
        if not cases:
            cases.append({"name": case or "SW"})
        name = case or cases[0]["name"]
        target = next((c for c in cases if c.get("name") == name), None)
        if target is None:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况",
                                      "available": [c.get("name") for c in cases]})
        spans = target.setdefault("member_spans", [])
        # 反复调用会把自重叠加好几遍，先把上一次生成的清掉
        kept = [e for e in spans if e.get("note") != SELF_WEIGHT_NOTE]
        replaced = len(spans) - len(kept)
        spans[:] = kept
        for mid, load in sorted(loads.items()):
            spans.append({"member": mid, **load.to_dict(),
                          "note": SELF_WEIGHT_NOTE})
        self._invalidate()
        total = sum(abs(load.w1[2]) for load in loads.values())
        return ToolResult(True, {
            "case": name, "members": len(loads),
            "replaced_previous": replaced,
            "total_vertical_kN_per_m": round(total * self.units.line_load_scale, 6),
            "note": "自重已写成显式均布荷载，可在模型里看到。改截面后需重新调用。"})

    def validate_model(self) -> ToolResult:
        errors = validate_payload(self.model)
        return ToolResult(not errors, {"errors": errors} if errors else {"summary": describe(self.model)})

    def preview_analysis_mesh(self) -> ToolResult:
        """返回物理构件到分析单元的映射，不改写会话的求解状态。"""
        try:
            compiled = compile_model(self.model)
        except CompilationError as exc:
            return ToolResult(False, {
                "errors": list(exc.diagnostics),
                "hint": "先修正模型，再预览分析网格",
            })

        frame = compiled.analysis_model
        mapping = compiled.mapping
        length_unit = "mm" if self.model.get("units") == "N-mm-MPa" else "m"
        generated_node_details = []
        for node_id, (physical_member, ratio) in sorted(mapping.generated_nodes.items()):
            node = frame.nodes[node_id]
            generated_node_details.append({
                "node": node_id,
                "physical_member": physical_member,
                "s": ratio,
                "coordinates": [float(value) for value in node.xyz],
            })

        analysis_element_details = []
        for element_id in sorted(frame.members):
            member = frame.members[element_id]
            analysis_element_details.append({
                "element": element_id,
                "physical_member": mapping.physical_member_id(element_id),
                "i": member.i,
                "j": member.j,
            })

        split_node_details = []
        for physical_member, element_ids in sorted(mapping.physical_to_elements.items()):
            lengths = []
            for element_id in element_ids:
                member = frame.members[element_id]
                lengths.append(float(np.linalg.norm(
                    frame.nodes[member.j].xyz - frame.nodes[member.i].xyz)))
            total = sum(lengths)
            distance = 0.0
            for index, element_id in enumerate(element_ids[:-1]):
                distance += lengths[index]
                node_id = frame.members[element_id].j
                node = frame.nodes[node_id]
                split_node_details.append({
                    "node": node_id,
                    "physical_member": physical_member,
                    "s": distance / total,
                    "origin": ("generated_for_point_load"
                               if node_id in mapping.generated_nodes
                               else "explicit_physical_node"),
                    "coordinates": [float(value) for value in node.xyz],
                })

        physical_nodes = len(compiled.source_ir.get("nodes", ()))
        physical_members = len(compiled.source_ir.get("members", ()))
        return ToolResult(True, {
            "physical_nodes": physical_nodes,
            "physical_members": physical_members,
            "analysis_nodes": len(frame.nodes),
            "analysis_elements": len(frame.members),
            "generated_nodes": len(mapping.generated_nodes),
            "split_nodes": len(split_node_details),
            "has_automatic_splits": len(frame.members) > physical_members,
            "length_unit": length_unit,
            "member_mapping": {
                str(member_id): list(element_ids)
                for member_id, element_ids in sorted(mapping.physical_to_elements.items())
            },
            "generated_node_details": generated_node_details,
            "split_node_details": split_node_details,
            "analysis_element_details": analysis_element_details,
            "diagnostics": list(compiled.diagnostics),
        })

    def export_learning_trace(self, label: str = "",
                              include_snapshots: bool = False) -> ToolResult:
        """把自己的确定性工具过程落成训练/回放样本。

        这里只记录工具事实，不把聊天模型的自然语言结论当真值；最终模型和严格
        校验结果一起保存，后续筛选样本时能排除尚未完成或未通过的轨迹。
        """
        import hashlib
        from datetime import datetime

        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                             for ch in str(label).strip())[:40]
        digest = hashlib.sha256(json.dumps(self.model, sort_keys=True,
                                           ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{stamp}_{digest}" + (f"_{safe_label}" if safe_label else "") + ".json"
        directory = Path("learning_traces")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        errors = validate_payload(self.model)
        document = {
            "format": "space-frame-agent-trace/v1",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "label": str(label),
            "verified": not errors,
            "validation_errors": errors,
            "steps": self.history.learning_trace(bool(include_snapshots)),
            "final_model": self.model,
        }
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return ToolResult(True, {"path": str(path.resolve()),
                                 "steps": len(document["steps"]),
                                 "verified": document["verified"],
                                 "format": document["format"],
                                 "note": "verified=true 的轨迹才适合作为已完成建模样本；"
                                         "未完成轨迹保留用于错误修复学习。"})

    def solve_model(self, analysis: str = "linear", increments: int = 10,
                    max_iter: int = 40, tolerance: float = 1e-7) -> ToolResult:
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "先按上面的清单修正模型，再重新求解"})
        self.compilation = compile_model(self.model)
        self.frame = self.compilation.analysis_model
        self.result_db = None
        try:
            if analysis == "linear":
                self.solution = solve(self.frame)
            elif analysis == "pdelta":
                from nonlinear import solve_pdelta
                self.solution = solve_pdelta(
                    self.frame, increments=increments, max_iter=max_iter,
                    tolerance=tolerance)
            elif analysis == "material":
                from nonlinear import solve_material_nonlinear
                self.solution = solve_material_nonlinear(
                    self.frame, increments=increments, max_iter=max_iter,
                    tolerance=tolerance)
            else:
                self.solution = None
                return ToolResult(False, {"error": f"未知分析类型 {analysis!r}"})
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            self.solution = None
            self.result_db = None
            payload: dict[str, Any] = {"error": str(exc)}
            diag = self.diagnose_supports()       # 直接把诊断带上，不必再调一次
            if diag.ok and diag.payload.get("modes"):
                payload["diagnosis"] = diag.payload["modes"]
                payload["hint"] = ("诊断已在上面的 diagnosis 里给出，不必再调 diagnose_supports。"
                                   "补上约束再求解，或向用户说明缺什么。注意三维梁两端都放开 rx "
                                   "会产生绕杆轴的扭转机构，至少要约束一端的 rx。")
            else:
                payload["hint"] = "检查约束是否足以消除全部刚体位移"
            return ToolResult(False, payload)
        self.result_db = result_db_from_solution(
            self.compilation.source_ir, self.frame, self.solution,
            self.compilation.mapping)
        span = self._reference_length()
        U = self.units
        results = {}
        for name, res in self.solution.all_results().items():
            node, mag = self._max_displacement(res)
            eq = check_equilibrium(self.frame, self.solution, name)
            entry = {"max_displacement_mm": round(mag * U.disp_scale, 6),
                     "at_node": node,
                     "equilibrium_ok": eq["ok"],
                     "equilibrium_residual": float(f"{eq['relative']:.3e}")}
            if mag <= 1e-12 and self._applied_load_magnitude(name) > 0.0:
                entry["warning"] = (
                    "有荷载但位移为零。常见原因：荷载全部作用在被约束死的方向上——"
                    "例如给平面刚架（bays 留空）加了面外荷载，"
                    "平面刚架的面外自由度是被自动约束的。检查荷载方向。")
            elif span > 0 and mag > span / 200.0:
                entry["warning"] = (
                    f"最大位移达到最短杆件长度的 1/{max(1, int(span / mag))}，量级异常。"
                    "常见原因：荷载方向写错（重力应为全局 -Z，即 w=[0,0,-w]）、"
                    "单位没换算成 N-m-Pa、或截面惯性矩填小了。")
            results[name] = entry

        # 求解成功后自动跑两项质检：静默失败检测 + 实验胶囊存档。
        # 这两项是"附加价值"，失败不影响求解结果本身——用 try/except 兜住，
        # 只在 payload 里记警告，不让主流程红。
        payload: dict[str, Any] = {
            "cases": results,
            "analysis": dict(self.solution.analysis),
            "compilation": {
                "physical_members": len(
                    self.compilation.mapping.physical_to_elements),
                "analysis_elements": len(
                    self.compilation.mapping.element_to_physical),
                "generated_nodes": len(self.compilation.mapping.generated_nodes),
                "diagnostics": list(self.compilation.diagnostics),
            },
        }

        # 1. 静默失败检测：8 项检查，把"算出来了但结果可疑"的情况挑出来
        try:
            findings = _silent.detect_silent_failures(self.frame, self.solution)
            summary = _silent.summarize_findings(findings)
            payload["silent_failures"] = {"summary": summary, "findings": findings}
        except Exception as exc:                      # noqa: BLE001
            payload["silent_failures"] = {
                "summary": {"status": "error",
                            "error": f"{type(exc).__name__}: {exc}"},
                "findings": [],
            }

        # 2. 实验胶囊存档：完整输入+结果摘要，便于复现和回归对比
        try:
            cap_path = _capsule.save_capsule(
                self.model, self.frame, self.solution, source="solve_model")
            payload["capsule"] = str(cap_path)
        except Exception as exc:                      # noqa: BLE001
            payload["capsule_error"] = f"{type(exc).__name__}: {exc}"

        return ToolResult(True, payload)

    def solve_with_abaqus(self, case: str | None = None,
                          element: str = "B33") -> ToolResult:
        """用 Abaqus 求解。结果格式与 solve_model 对齐，便于直接比对。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "模型都不合法，换求解器也没用"})
        try:
            import abaqus_backend
        except ImportError as exc:
            return ToolResult(False, {"error": f"Abaqus 后端不可用：{exc}"})
        out_dir = Path("results") / "abaqus"
        try:
            summary = abaqus_backend.solve(self.model, out_dir, case=case,
                                           element=element)
        except KeyError as exc:
            return ToolResult(False, {"error": str(exc)})
        except abaqus_backend.AbaqusError as exc:
            return ToolResult(False, {"error": str(exc),
                                      "hint": "自研求解器 solve_model 不需要 Abaqus，"
                                              "可以改用它"})
        summary["note"] = ("这是 Abaqus 的结果。与 solve_model 的结果比对时请注意单元格式："
                           "B33 与本程序同为 Euler-Bernoulli，误差应到数值精度量级；"
                           "B31 含剪切变形，偏差是格式差异不是错误。")
        return ToolResult(True, summary)

    def _native_rows(self, case: str) -> dict[int, dict[str, float]]:
        """把自研结果摊成与 Abaqus CSV 同一套字段，才能逐分量比。

        单位取当前模型的原生单位（m 或 mm、rad、N），两边同源，不做缩放。
        """
        res = self.solution[case]
        rows: dict[int, dict[str, float]] = {}
        keys = ("u1", "u2", "u3", "ur1", "ur2", "ur3")
        for nid in sorted(self.frame.nodes):
            d = self.frame.node_dofs(nid)
            row = {k: float(res.U[d[i]]) for i, k in enumerate(keys)}
            # 非支座节点反力恒为零，写进去让两侧字段对齐
            for i, k in enumerate(("rf1", "rf2", "rf3")):
                row[k] = float(res.R[d[i]]) if nid in self.frame.supports else 0.0
            rows[nid] = row
        return rows

    def compare_solvers(self, case: str | None = None,
                        element: str = "B33") -> ToolResult:
        """自研 + Abaqus 各算一遍，逐分量给归一化偏差。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "模型都不合法，比对没有意义"})
        native = self.solve_model()
        if not native.ok:
            return ToolResult(False, {"error": "自研求解器先失败了",
                                      "detail": native.payload})
        name = case or self.model["load_cases"][0]["name"]
        if self.solution is None or name not in set(self.solution.all_results()):
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况"})
        ab = self.solve_with_abaqus(case=name, element=element)
        if not ab.ok:
            return ToolResult(False, {"error": "Abaqus 侧失败，无法比对",
                                      "detail": ab.payload})
        try:
            import abaqus_backend
            rows_ab = abaqus_backend.read_result_csv(Path(ab.payload["files"]["csv"]))
            cmp = abaqus_backend.compare_rows(
                self._native_rows(name), rows_ab,
                displacement_scale=self.units.disp_scale)
        except Exception as exc:  # noqa: BLE001 — 比对失败不该让整轮对话崩掉
            return ToolResult(False, {"error": f"读取 Abaqus 结果失败：{exc}"})
        if "error" in cmp:
            return ToolResult(False, {"error": cmp["error"]})
        worst = max((v["e"] for v in cmp["errors"].values() if v["e"] is not None),
                    default=None)
        return ToolResult(True, {
            "case": name, "element": element,
            "nodes_compared": cmp["nodes_compared"],
            "errors": {k: (None if v["e"] is None else float(f"{v['e']:.3e}"))
                       for k, v in cmp["errors"].items()},
            "worst_error": None if worst is None else float(f"{worst:.3e}"),
            "peak_displacement": cmp["peak"],
            "metric": "全场归一化相对误差 sqrt(Σ(a−b)²)/sqrt(Σb²)，以 Abaqus 为参考；"
                      "该分量参考解整体为零时记 null（归一化无意义），不是没算",
            "note": ("B33 与本程序同为 Euler-Bernoulli，偏差应在 1e-5 以下；"
                     "B31 含剪切变形，偏差随杆件越粗越大，那是格式差异不是错误。"),
        })

    def query_results(self, what: str, case: str | None = None,
                      member_id: int | None = None,
                      span: float | None = None) -> ToolResult:
        if self.solution is None or self.result_db is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        name = case or self._controlling_case()
        U = self.units
        if name not in self.result_db.steps:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况或组合",
                                      "available": list(self.result_db.steps)})
        if what == "max_displacement":
            displacements = self.result_db.field(name, "U")
            node = max(displacements.values,
                       key=lambda item: float(np.linalg.norm(displacements.value(item))))
            vector = displacements.value(node)
            mag = float(np.linalg.norm(vector))
            n = self.frame.nodes[node]
            comp = [round(float(value) * U.disp_scale, 6) for value in vector]
            payload: dict[str, Any] = {
                "case": name, "node": node,
                "coordinates_xyz": [round(v * U.length_to_m, 6)
                                    for v in (n.x, n.y, n.z)],
                "coordinate_unit": "m",
                "magnitude_mm": round(mag * U.disp_scale, 6),
                "components_mm": {"ux": comp[0], "uy": comp[1], "uz": comp[2]},
                "note": "位置用上面的坐标描述，方向用 components_mm 判断，"
                        "两者都不要靠节点编号推测",
            }
            origin = self.compilation.mapping.generated_nodes.get(node)
            if origin is not None:
                payload.update({"node_kind": "generated_analysis_node",
                                "physical_member": origin[0],
                                "member_s": round(origin[1], 9)})
            else:
                payload["node_kind"] = "physical_node"
            return ToolResult(True, payload)
        if what == "max_deflection":
            try:
                from internal_forces import max_deflection
            except ImportError as exc:
                return ToolResult(False, {"error": f"内力模块不可用：{exc}"})
            # 梯形积分的误差随测点数二阶下降。101 点时单跨一个单元差 0.016%，
            # 201 点降到 0.004%——够了，再加只是烧时间
            best = max_deflection(
                self.frame, self.result_db.solution_view(self.frame), name,
                stations=201,
                mapping=self.compilation.mapping)
            if best["member"] is None:
                return ToolResult(False, {"error": "模型里没有杆件"})
            length = float(best["length"])
            value = abs(best["value"])
            payload: dict[str, Any] = {
                "case": name,
                "magnitude_mm": round(value * U.disp_scale, 6),
                "at_member": best["member"],
                "at_x_m": round(best["x"] * U.length_to_m, 4),
                "member_length_m": round(length * U.length_to_m, 4),
                "member_length_over_deflection": (round(length / value, 1)
                                                  if value > 0 else None),
            }
            if span is not None:
                payload["span_m"] = float(span)
                payload["span_over_deflection"] = (round(float(span) /
                                                          (value * U.length_to_m), 1)
                                                   if value > 0 else None)
            payload["note"] = (
                "这是**单元内**的最大横向挠度，由精确弯矩两次积分得到，与网格疏密无关。"
                "校核挠跨比用它，不要用 max_displacement——后者只看节点。"
                "注意 member_length_over_deflection 的分母是**这一根杆件的长度**，"
                "不是设计跨度：门式刚架的斜梁是两根杆，梁划成几个单元时更是差几倍。"
                "要和 L/400 这类限值比，请把设计跨度作为 span 传进来，"
                "工具会给出 span_over_deflection。")
            return ToolResult(True, payload)

        if what == "reactions":
            reactions = self.result_db.field(name, "RF")
            out = {}
            for nid in sorted(self.frame.supports):
                n = self.frame.nodes[nid]
                out[str(nid)] = {"xyz": [round(v * U.length_to_m, 6)
                                         for v in (n.x, n.y, n.z)],
                                 "R": [round(float(value) * U.force_scale, 6)
                                       for value in reactions.value(nid)]}
            total = U.force_scale * sum(float(reactions.value(nid)[2])
                                        for nid in self.frame.supports)
            return ToolResult(True, {"case": name, "unit": "kN",
                                     "coordinate_unit": "m",
                                     "reactions": out,
                                     "vertical_total_kN": round(total, 6)})
        if what == "member_forces":
            physical_ids = self.result_db.physical_to_elements
            if member_id is None:
                # 不给编号就一次返回全部，省得一根根查
                rows = {}
                for mid in sorted(physical_ids):
                    axial = self.result_db.physical_member_value(name, "N", mid)
                    mz = self.result_db.physical_member_value(name, "Mz", mid)
                    # 保留位数与 query_diagram 一致：同一个量在两处对不上，
                    # 模型会以为是两个不同的结果
                    rows[str(mid)] = {
                        "N": round(float(axial[1]) * U.force_scale, 6),
                        "Mz_i": round(float(mz[0]) * U.moment_scale, 6),
                        "Mz_j": round(float(mz[1]) * U.moment_scale, 6)}
                return ToolResult(True, {"case": name, "unit": "kN, kN·m",
                                         "sign_convention": "N 以受拉为正，受压为负",
                                         "all_members": rows})
            if member_id not in physical_ids:
                return ToolResult(False, {"error": f"没有编号为 {member_id} 的杆件"})
            axial = self.result_db.physical_member_value(name, "N", member_id)
            my = self.result_db.physical_member_value(name, "My", member_id)
            mz = self.result_db.physical_member_value(name, "Mz", member_id)
            return ToolResult(True, {"case": name, "member": member_id, "unit": "kN, kN·m",
                                     "sign_convention": "N 以受拉为正，受压为负",
                                     "N": round(float(axial[1]) * U.force_scale, 6),
                                     "Mz_i": round(float(mz[0]) * U.moment_scale, 6),
                                     "Mz_j": round(float(mz[1]) * U.moment_scale, 6),
                                     "My_i": round(float(my[0]) * U.moment_scale, 6),
                                     "My_j": round(float(my[1]) * U.moment_scale, 6)})
        return ToolResult(False, {"error": f"不支持的查询 {what!r}"})

    def plot_results(self, kind: str, case: str | None = None) -> ToolResult:
        """出图。只回路径与摘要，不回图像——结论仍须引用 query_results 的数字。"""
        if self.solution is None or self.result_db is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        try:
            from plot3d import plot_axial, plot_deformed, plot_diagram
        except ImportError as exc:
            return ToolResult(False, {"error": f"绘图依赖缺失：{exc}"})
        name = case or self._controlling_case()
        if name not in self.result_db.steps:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况或组合",
                                      "available": list(self.result_db.steps)})
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        result_view = self.result_db.solution_view(self.frame)
        try:
            if kind == "deformed":
                info = plot_deformed(self.frame, result_view, name,
                                     out_dir / f"deformed_{safe}.png",
                                     mapping=self.compilation.mapping)
            elif kind == "axial":
                info = plot_axial(self.frame, result_view, name,
                                  out_dir / f"axial_{safe}.png",
                                  mapping=self.compilation.mapping)
            elif kind in {"moment", "shear"}:
                comp = "Mz" if kind == "moment" else "Vy"
                info = plot_diagram(self.frame, result_view, comp, name,
                                    out_dir / f"{kind}_{safe}.png",
                                    mapping=self.compilation.mapping)
            else:
                return ToolResult(False, {"error": f"不支持的图类型 {kind!r}"})
        except Exception as exc:
            return ToolResult(False, {"error": f"绘图失败：{type(exc).__name__}: {exc}"})
        info["note"] = "图已保存供用户查看；引用数字请用 query_results，不要从图上读数"
        return ToolResult(True, info)

    def query_diagram(self, component: str, member: int | None = None,
                      case: str | None = None,
                      stations: int | None = None) -> ToolResult:
        """沿杆长的内力。解析恢复，与网格无关。"""
        if self.solution is None or self.result_db is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        try:
            from internal_forces import COMPONENTS, physical_member_diagram
        except ImportError as exc:
            return ToolResult(False, {"error": f"内力模块不可用：{exc}"})
        if component not in COMPONENTS:
            return ToolResult(False, {"error": f"内力分量只能取 {list(COMPONENTS)}"})
        name = case or self._controlling_case()
        if name not in self.result_db.steps:
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工况或组合",
                                      "available": list(self.result_db.steps)})
        U = self.units
        moment = component in {"T", "My", "Mz"}
        unit = U.moment_unit if moment else U.force_unit
        scale = U.moment_scale if moment else U.force_scale
        result_view = self.result_db.solution_view(self.frame)

        if member is None:
            worst = None
            for physical_id in sorted(self.compilation.mapping.physical_to_elements):
                diagram = physical_member_diagram(
                    self.frame, result_view, self.compilation.mapping,
                    physical_id, name, stations=201)
                x_at, value = diagram.extreme(component)
                if worst is None or abs(value) > abs(worst["value"]):
                    worst = {"value": value, "member": physical_id, "x": x_at}
            if worst is None:
                return ToolResult(False, {"error": "模型里没有杆件"})
            return ToolResult(True, {
                "case": name, "component": component, "unit": unit,
                "peak": round(worst["value"] * scale, 6),
                "at_member": worst["member"],
                "at_x_m": round(worst["x"] * U.length_to_m, 4),
                "note": "全结构极值。要看某根杆件的完整分布，传 member 与 stations。"})

        if member not in self.compilation.mapping.physical_to_elements:
            return ToolResult(False, {"error": f"没有编号为 {member} 的杆件"})
        d = physical_member_diagram(
            self.frame, result_view, self.compilation.mapping, member, name,
            stations=stations or 21)
        x_at, peak = d.extreme(component)
        payload: dict[str, Any] = {
            "case": name, "member": member, "component": component, "unit": unit,
            "length_m": round(d.length * U.length_to_m, 4),
            "peak": round(peak * scale, 6),
            "at_x_m": round(x_at * U.length_to_m, 4),
            "at_i_end": round(float(d.component(component)[0]) * scale, 6),
            "at_j_end": round(float(d.component(component)[-1]) * scale, 6)}
        if stations:
            payload["x_m"] = [round(float(v) * U.length_to_m, 4) for v in d.x]
            payload["values"] = [round(float(v) * scale, 6)
                                 for v in d.component(component)]
        return ToolResult(True, payload)

    # 扫描指标：名字 → (取值函数, 单位, 越大越好吗)
    _SWEEP_BIGGER_IS_BETTER = {"buckling_factor", "first_frequency"}

    def _sweep_metric(self, metric: str, case: str | None):
        """在当前 self.model 上算一个指标。失败时返回 (None, 原因)。"""
        solved = self.solve_model()
        if not solved.ok:
            return None, "求解失败"
        if metric in ("max_displacement", "max_deflection"):
            r = self.query_results(what=metric, case=case)
            return (r.payload["magnitude_mm"], None) if r.ok else (None, "查询失败")
        if metric == "max_abs_Mz":
            r = self.query_diagram(component="Mz", case=case)
            return (abs(r.payload["peak"]), None) if r.ok else (None, "查询失败")
        if metric == "buckling_factor":
            r = self.buckling_analysis(case=case, num_modes=1)
            return (r.payload["critical_factor"], None) if r.ok else \
                (None, r.payload.get("error", "屈曲失败"))
        if metric == "first_frequency":
            r = self.modal_analysis(num_modes=1)
            return (r.payload["modes"][0]["frequency_Hz"], None) if r.ok else \
                (None, r.payload.get("error", "模态失败"))
        return None, f"不认识的指标 {metric!r}"

    def sweep(self, what: str, target: str, values: list[float], metric: str,
              prop: str | None = None, case: str | None = None,
              limit: float | None = None) -> ToolResult:
        """参数扫描。一次调用跑完 N 个取值，返回对照表与结论。"""
        import copy

        if not self.model.get("nodes"):
            return ToolResult(False, {"error": "还没有模型"})
        if not values or len(values) < 2:
            return ToolResult(False, {"error": "至少要给两个取值才谈得上扫描"})
        if what == "section_property" and prop not in ("A", "Iy", "Iz", "J"):
            return ToolResult(False, {"error": "扫截面特性时 prop 必须是 A/Iy/Iz/J"})

        names = {"section_property": [s["name"] for s in self.model["sections"]],
                 "material_E": [m["name"] for m in self.model["materials"]],
                 "load_scale": [c.get("name")
                                for c in self.model.get("load_cases", [])]}.get(what)
        if names is None:
            return ToolResult(False, {"error": f"不认识的扫描类型 {what!r}"})
        if target not in names:
            return ToolResult(False, {"error": f"没有名为 {target!r} 的对象",
                                      "available": names})

        # 扫描**绝不能动**当前模型：用户扫完还要接着用原模型算别的
        original = copy.deepcopy(self.model)
        original_solution = self.solution
        rows: list[dict] = []
        try:
            for v in values:
                probe = copy.deepcopy(original)
                if what == "section_property":
                    for sec in probe["sections"]:
                        if sec["name"] == target:
                            sec[prop] = float(v)
                elif what == "material_E":
                    for mat in probe["materials"]:
                        if mat["name"] == target:
                            mat["E"] = float(v)
                else:
                    probe.setdefault("combos", [])
                    probe["combos"] = [c for c in probe["combos"]
                                       if c["name"] != "__sweep__"]
                    probe["combos"].append({"name": "__sweep__",
                                            "factors": {target: float(v)}})
                self.model = probe
                self._invalidate()
                use_case = "__sweep__" if what == "load_scale" else case
                got, why = self._sweep_metric(metric, use_case)
                rows.append({"value": float(v),
                             "metric": None if got is None else round(float(got), 6),
                             "failed": why})
        finally:
            # 一定要还原成扫描前的样子：模型、装配好的 frame、以及结果。
            # 只还原 model 不还原 frame/solution 的话，接下来的查询会拿
            # 最后一个探针的结果去回答原模型的问题——不报错，但数全是错的
            self.model = original
            self._invalidate()
            if original_solution is not None:
                self.solve_model()

        good = [r for r in rows if r["metric"] is not None]
        if not good:
            return ToolResult(False, {"error": "每个取值都没算出来",
                                      "rows": rows})

        payload: dict[str, Any] = {
            "what": what, "target": target, "prop": prop, "metric": metric,
            "unit": {"max_displacement": "mm", "max_deflection": "mm",
                     "max_abs_Mz": "kN·m", "buckling_factor": "倍",
                     "first_frequency": "Hz"}[metric],
            "rows": rows,
            "note": "扫描没有改动当前模型；要真正采用某个取值，"
                    "再单独调一次相应的工具把它写进模型。",
        }
        if limit is not None:
            bigger_better = metric in self._SWEEP_BIGGER_IS_BETTER
            ok_rows = [r for r in good
                       if (r["metric"] >= limit if bigger_better
                           else r["metric"] <= limit)]
            payload["limit"] = float(limit)
            payload["satisfying_values"] = [r["value"] for r in ok_rows]
            # 满足限值所需的最省取值。位移/弯矩类是"越大的截面越好"，
            # 所以取满足者里最小的那个值；两类指标方向相反，别写反
            payload["smallest_sufficient"] = (min(r["value"] for r in ok_rows)
                                              if ok_rows else None)
            if not ok_rows:
                payload["hint"] = ("给的这些取值都不满足限值，"
                                   "把范围往有利方向再扩一档再扫一次。")
        return ToolResult(True, payload)

    def query_envelope(self, component: str, member: int | None = None,
                       cases: list[str] | None = None) -> ToolResult:
        """内力包络与控制组合。"""
        if self.solution is None or self.result_db is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        try:
            from envelope import governing_summary, member_envelope
            from internal_forces import COMPONENTS
        except ImportError as exc:
            return ToolResult(False, {"error": f"包络模块不可用：{exc}"})
        if component not in COMPONENTS:
            return ToolResult(False, {"error": f"内力分量只能取 {list(COMPONENTS)}"})
        picked = tuple(cases) if cases else None
        U = self.units
        moment = component in {"T", "My", "Mz"}
        unit = U.moment_unit if moment else U.force_unit
        scale = U.moment_scale if moment else U.force_scale
        result_view = self.result_db.solution_view(self.frame)

        try:
            if member is None:
                s = governing_summary(
                    self.frame, result_view, picked, stations=101,
                    mapping=self.compilation.mapping)
                worst = s["worst"][component]
                return ToolResult(True, {
                    "component": component, "unit": unit,
                    "cases": s["cases"],
                    "peak": round(worst["value"] * scale, 6),
                    "at_member": worst["member"],
                    "at_x_m": round(worst["x"] * U.length_to_m, 4),
                    "governing_case": worst["case"],
                    "controls_global_extreme": s["controls_global_extreme"],
                    "never_governs_anywhere": s["never_governs_anywhere"],
                    "note": "全结构最不利。要看某根杆件的逐点包络，传 member。"
                            "controls_global_extreme 表示该组合拿下了几个分量的"
                            "全结构极值；它为 0 并不等于这个组合没用——"
                            "它完全可能在某根梁的端部说了算。"
                            "真正一处都不控制的在 never_governs_anywhere 里，"
                            "那种才可以考虑删掉，或者检查它的系数是不是写错了。"})
            if member not in self.compilation.mapping.physical_to_elements:
                return ToolResult(False, {"error": f"没有编号为 {member} 的杆件"})
            env = member_envelope(
                self.frame, result_view, member, picked, stations=101,
                mapping=self.compilation.mapping)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})

        worst = env.extreme(component)
        return ToolResult(True, {
            "member": member, "component": component, "unit": unit,
            "cases": list(env.cases),
            "length_m": round(env.length * U.length_to_m, 4),
            "peak": round(worst["value"] * scale, 6),
            "at_x_m": round(worst["x"] * U.length_to_m, 4),
            "governing_case": worst["case"],
            "at_i_end": {"upper": round(float(env.upper[component][0]) * scale, 6),
                         "lower": round(float(env.lower[component][0]) * scale, 6)},
            "at_j_end": {"upper": round(float(env.upper[component][-1]) * scale, 6),
                         "lower": round(float(env.lower[component][-1]) * scale, 6)},
            "cases_that_govern_somewhere": env.governing(component),
            "note": "上下包线是逐点取的。同一根杆上跨中与支座常由不同组合控制，"
                    "所以引用结论时要连位置和组合名一起说。"})

    def modal_analysis(self, num_modes: int = 6) -> ToolResult:
        """自振频率与振型。结构固有属性，与荷载无关。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        try:
            from modal import modal
            r = modal(from_dict(self.model), int(num_modes))
        except ImportError as exc:
            return ToolResult(False, {"error": f"模态模块不可用：{exc}"})
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ToolResult(False, {"error": str(exc)})
        share = r.effective_mass.sum(axis=0) / r.total_mass if r.total_mass else None
        return ToolResult(True, {
            "modes": [{"order": k + 1,
                       "frequency_Hz": round(float(f), 6),
                       "period_s": round(float(1.0 / f), 6) if f > 0 else None}
                      for k, f in enumerate(r.frequencies)],
            "total_mass_kg": round(r.total_mass, 6),
            "effective_mass_ratio_xyz": (None if share is None else
                                         [round(float(v), 4) for v in share]),
            "note": "一致质量矩阵，频率略高于精确解；每跨四个单元时误差 0.2% 以内。"
                    "有效质量比接近 1 才说明取的阶数够——差得远就加大 num_modes。",
        })

    def buckling_analysis(self, case: str | None = None,
                          num_modes: int = 4) -> ToolResult:
        """线性屈曲。λ 是该工况荷载的临界放大倍数。"""
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        if self.solution is None:
            solved = self.solve_model()
            if not solved.ok:
                return ToolResult(False, {"error": "静力求解先失败了，无法做屈曲分析",
                                          "detail": solved.payload})
        name = case or self._controlling_case()
        try:
            from buckling import buckling
            r = buckling(self.frame, name, int(num_modes), solution=self.solution)
        except ImportError as exc:
            return ToolResult(False, {"error": f"屈曲模块不可用：{exc}"})
        except (ValueError, np.linalg.LinAlgError) as exc:
            return ToolResult(False, {"error": str(exc)})
        U = self.units
        worst = min(r.axial, key=lambda m: r.axial[m])
        return ToolResult(True, {
            "case": r.case,
            "critical_factor": round(float(r.critical), 6),
            "factors": [round(float(v), 6) for v in r.factors],
            "most_compressed_member": worst,
            "its_axial_kN": round(r.axial[worst] * U.force_scale, 6),
            "note": "λ 是该工况荷载的临界放大倍数：λ=3 表示放大三倍才失稳。"
                    "这是**线性特征值屈曲**，假定失稳前保持线弹性、变形小、"
                    "轴力不随变形改变。真实结构有初始缺陷与残余应力，"
                    "实际承载力低于此值——只能当上限，不能直接当承载力用。",
        })

    def write_report(self, case: str | None = None, fmt: str = "both",
                     filename: str = "报告") -> ToolResult:
        """产出分析报告。"""
        if self.solution is None:
            return ToolResult(False, {"error": "还没有结果，请先调用 solve_model"})
        if fmt not in ("markdown", "docx", "both"):
            return ToolResult(False, {"error": "fmt 只能取 markdown / docx / both"})
        try:
            import report as _report
        except ImportError as exc:
            return ToolResult(False, {"error": f"报告模块不可用：{exc}"})

        out_dir = Path("results")
        safe = "".join(c if c.isalnum() or c in "-_（）()" else "_"
                       for c in str(filename)) or "报告"
        try:
            doc = _report.gather(self, case=case, out_dir=out_dir)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})

        written: dict[str, str] = {}
        problems: dict[str, str] = {}
        if fmt in ("markdown", "both"):
            path = out_dir / f"{safe}.md"
            _report.to_markdown(doc, path)
            written["markdown"] = str(path)
        if fmt in ("docx", "both"):
            try:
                written["docx"] = str(_report.to_docx(doc, out_dir / f"{safe}.docx"))
            except Exception as exc:      # noqa: BLE001 docx 失败不该毁掉 md
                problems["docx"] = f"{type(exc).__name__}: {str(exc)[:200]}"

        if not written:
            return ToolResult(False, {"error": "两种格式都没写成", "detail": problems})
        payload: dict[str, Any] = {
            "files": written,
            "case": doc["case"],
            "sections": ["模型概况", "求解与校验", "位移", "支座反力"]
                        + (["内力极值与控制组合"] if doc.get("envelope") else [])
                        + (["自振特性"] if doc.get("modal") else [])
                        + (["稳定"] if doc.get("buckling") else [])
                        + (["图"] if doc["figures"] else []),
            "figures": list(doc["figures"]),
            "note": "报告已写到 results/ 目录。里面的数与你查到的完全一致，"
                    "回答里可以直接引用，不必重述整张表。",
        }
        if doc["skipped"]:
            payload["skipped"] = doc["skipped"]
        if problems:
            payload["partial"] = problems
        return ToolResult(True, payload)

    def diagnose_supports(self) -> ToolResult:
        if not self.model:
            return ToolResult(False, {"error": "还没有模型"})
        errors = validate_payload(self.model)
        if errors:
            return ToolResult(False, {"errors": errors})
        frame = from_dict(self.model)
        modes = diagnose_singularity(frame)
        if not modes:
            return ToolResult(True, {"modes": [], "note": "约束充分，未发现刚体位移模态"})
        return ToolResult(True, {"modes": [
            {"eigenvalue": m["eigenvalue"],
             "participants": [{"node": p["node"], "direction": p["dof_name"]}
                              for p in m["participants"][:4]]}
            for m in modes
        ], "note": "上列节点与方向缺少约束，结构在该方向可自由运动"})

    # --- 内部 ---
    def preview_frame(self):
        """不求解也装配出一个 Frame，供"先看一眼再算"用。

        `self.frame` 要到 solve_model 才有值——那是刻意的，免得别处
        误用一个没算过的模型去读结果。但**看模型不该要求先求解**：
        荷载方向加反、支座漏了，正是应该在算之前就看出来的事。
        所以单开这个入口，语义明确：只装配，不算。
        """
        if self.frame is not None:
            return self.frame
        errors = validate_payload(self.model)
        if errors:
            raise ValueError("模型不合法：" + "；".join(errors[:3]))
        return compile_model(self.model).analysis_model

    def _invalidate(self) -> None:
        self.frame = None
        self.solution = None
        self.compilation = None
        self.result_db = None

    def _applied_load_magnitude(self, case: str) -> float:
        """该工况施加的荷载总量级，用来分辨"没有荷载"和"荷载被约束吃掉了"。"""
        load_case = self.frame.load_cases.get(case)
        if load_case is None:                      # 组合：按系数合成后再看
            factors = self.frame.combos.get(case, {})
            total = 0.0
            for name, f in factors.items():
                total += abs(f) * self._applied_load_magnitude(name)
            return total
        total = 0.0
        for load in load_case.nodal_loads.values():
            total += float(np.abs(np.asarray(load, dtype=float)).sum())
        for w in load_case.member_loads.values():
            total += float(np.abs(np.asarray(w, dtype=float)).sum())
        return total

    def _reference_length(self) -> float:
        """取最短杆件长度作为位移量级的参照。"""
        best = 0.0
        for m in self.frame.members.values():
            a, b = self.frame.nodes[m.i], self.frame.nodes[m.j]
            L = float(np.linalg.norm(b.xyz - a.xyz))
            best = L if best == 0.0 else min(best, L)
        return best

    def _max_displacement(self, res) -> tuple[int, float]:
        best_node, best = -1, -1.0
        for nid in self.frame.order():
            d = self.frame.node_dofs(nid)
            mag = float(np.linalg.norm(res.U[d[:3]]))
            if mag > best:
                best, best_node = mag, nid
        return best_node, best

    @property
    def units(self):
        """当前模型的单位制。位移一律报 mm、力报 kN、弯矩报 kN·m，
        与模型内部用 m 还是 mm 无关——读结果的人不用先想是哪套制。"""
        return unit_system(self.model.get("units"))

    def _controlling_case(self) -> str:
        return max(self.solution.all_results().values(),
                   key=lambda r: max(abs(f).max() for f in r.member_forces.values())).name

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        confirmation_tools = {"remove_members", "remove_nodes"}
        has_geometry = bool(self.model.get("nodes") or self.model.get("members"))
        destructive = name in confirmation_tools and has_geometry
        if name == "set_model" and has_geometry and isinstance(arguments, dict):
            replacement = arguments.get("model")
            if isinstance(replacement, dict):
                try:
                    old_nodes = {int(item["id"])
                                 for item in self.model.get("nodes") or []}
                    new_nodes = {int(item["id"])
                                 for item in replacement.get("nodes") or []}
                    old_members = {int(item["id"])
                                   for item in self.model.get("members") or []}
                    new_members = {int(item["id"])
                                   for item in replacement.get("members") or []}
                    destructive = bool((old_nodes - new_nodes)
                                       or (old_members - new_members))
                except (KeyError, TypeError, ValueError):
                    # 格式错误仍交给 set_model 自己返回原有的结构化校验信息。
                    destructive = False
        if destructive and not self._applying_preview:
            result = ToolResult(False, {
                "error": f"{name} 会删除已有模型实体，请先 preview_change 并等待用户确认",
                "confirmation_required": True,
                "required_tool": "preview_change",
            })
            self.tool_log.append((name, arguments, result))
            return result
        handler: Callable[..., ToolResult] | None = getattr(self, name, None)
        if handler is None or name.startswith("_"):
            return ToolResult(False, {"error": f"没有名为 {name!r} 的工具"})
        try:
            result = handler(**arguments)
        except TypeError as exc:
            result = ToolResult(False, {"error": f"参数不对：{exc}"})
        self.tool_log.append((name, arguments, result))
        return result


# --------------------------------------------------------------------------- 模型提供方

class Provider(Protocol):
    """只要能吃 messages + tools 吐回复，什么后端都行。"""

    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...


@dataclass
class ScriptedProvider:
    """按脚本回放的假后端：离线测试与演示都用它，不需要密钥也不怕断网。"""
    script: Sequence[dict]
    seen: list[list[dict]] = field(default_factory=list)
    _cursor: int = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        self.seen.append(list(messages))
        if self._cursor >= len(self.script):
            raise AssertionError("脚本已用尽，说明轮数比预期多")
        item = self.script[self._cursor]
        self._cursor += 1
        return item


class DeepSeekProvider:
    """OpenAI 兼容接口的云端后端。

    密钥从外部传入，**不落盘、不进日志、不进 repr**。调用方自己从环境变量读：

        DeepSeekProvider(api_key=os.environ["DEEPSEEK_API_KEY"])

    temperature 默认 0：评测集要可复现，演示也要和报告里的数字一致。
    """

    def __init__(self, api_key: str, model: str = "deepseek-v4-flash",
                 base_url: str = "https://api.deepseek.com", temperature: float = 0.0):
        from openai import OpenAI          # 延迟导入：没装 openai 也能跑离线测试
        if not api_key or not api_key.strip():
            raise ValueError("缺少 API 密钥。请设置环境变量 DEEPSEEK_API_KEY 后重试。")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                      "cache_hit_tokens": 0, "calls": 0}

    def __repr__(self) -> str:              # 防止密钥随对象打印泄露
        return f"DeepSeekProvider(model={self._model!r})"

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, tools=tools,
            temperature=self._temperature,
        )
        u = getattr(response, "usage", None)
        if u is not None:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            self.usage["cache_hit_tokens"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0

        message = response.choices[0].message
        if message.tool_calls:
            calls = []
            for c in message.tool_calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"__invalid_json__": c.function.arguments}
                calls.append({"id": c.id, "name": c.function.name, "arguments": args})
            return {"tool_calls": calls}
        return {"content": message.content or ""}


# --------------------------------------------------------------------------- 主循环

@dataclass
class TurnResult:
    reply: str
    rounds: int
    tool_calls: list[tuple[str, dict]]
    session: Session
    stopped_by_limit: bool = False
    # 由对话层填写的分段耗时。保留默认值，让单元测试、离线脚本和第三方调用
    # 不必为了新增观测能力同步修改构造代码。
    metrics: dict[str, Any] = field(default_factory=dict)


def run_turn(user_text: str, provider: Provider, session: Session | None = None,
             max_rounds: int = MAX_ROUNDS) -> TurnResult:
    """跑完一轮对话：模型调工具、读结果、再调，直到给出最终回答。"""
    from workflow import refresh_workflow_message, workflow_message

    session = session or Session()
    session.receive_user_confirmation(user_text)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            workflow_message(session),
                            {"role": "user", "content": user_text}]
    calls: list[tuple[str, dict]] = []

    for round_index in range(max_rounds):
        reply = provider.complete(messages, TOOLS)
        if not reply.get("tool_calls"):
            return TurnResult(reply.get("content", ""), round_index + 1, calls, session)

        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"],
                                         "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                           for c in reply["tool_calls"]],
        })
        for call in reply["tool_calls"]:
            result = session.dispatch(call["name"], call["arguments"])
            calls.append((call["name"], call["arguments"]))
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": result.to_json()})
        refresh_workflow_message(messages, session)

    return TurnResult("已达到最大工具调用轮数，仍未得到结论。请检查模型或缩小问题范围。",
                      max_rounds, calls, session, stopped_by_limit=True)
