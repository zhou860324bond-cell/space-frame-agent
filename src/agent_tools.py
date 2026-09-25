"""大模型看得见的工具清单。**只有数据，没有逻辑。**

从 agent.py 搬出来的，一个字节没改。搬的理由是比例：这份清单是
1180 行 JSON Schema，占了 agent.py 的四分之一强，而它既不执行也不被执行——
是给大模型读的接口说明书。把它和会话状态机放在同一个文件里，唯一的效果
就是每次想找一个方法都要先翻过一千行不会执行的数据。

改这里要记得：工具数量写进了 README、《课程报告》和
`tests/test_docs_current.py`，加减一个工具就要同步改那几处，否则测试会红。
清单和实现的对应关系由 `tests/test_tool_contracts.py` 守着——
这里声明的每个 name，`Session` 上都必须有同名方法。
"""

from __future__ import annotations

from typing import Any

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
                                                 "hardening_ratio": {"type": "number"},
                                                 "alpha": {
                                                     "type": "number",
                                                     "description": "线膨胀系数 1/℃，"
                                                                    "只有要算温度应力时才需要，"
                                                                    "钢约 1.2e-5"},
                                                 "allow_tension": {
                                                     "type": "number",
                                                     "description": "许用拉应力，"
                                                                    "只有要做强度验算时才需要"},
                                                 "allow_compression": {
                                                     "type": "number",
                                                     "description": "许用压应力；不给按钢材"
                                                                    "惯例取与拉相同。铸铁、砌体、"
                                                                    "木材抗压远大于抗拉，"
                                                                    "**必须单独给**"}}},
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
                                                 "Az": {"type": "number"},
                                                 "cy": {"type": "number",
                                                        "description": "局部 y 向极端纤维距离，"
                                                                       "算弯曲应力必需"},
                                                 "cz": {"type": "number",
                                                        "description": "局部 z 向极端纤维距离"},
                                                 "circular": {"type": "boolean",
                                                              "description": "圆形截面，"
                                                                             "两向弯曲按平方和合成"}}},
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
                    "mu_y": {"type": "number", "exclusiveMinimum": 0,
                             "description": "绕局部 y 轴屈曲的计算长度系数。"
                                            "不给则由杆端释放推定，而推定只对"
                                            "**无侧移**结构成立——有侧移的框架柱"
                                            "μ>1、悬臂柱 2.0，必须在这里显式给"},
                    "mu_z": {"type": "number", "exclusiveMinimum": 0,
                             "description": "绕局部 z 轴的计算长度系数"},
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
                "type": "object", "required": ["node_ids"],
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
                            "description": "[ux,uy,uz,rx,ry,rz]，1 表示约束。给了 bc_type 就不必给它"},
                    "bc_type": {"type": "string",
                                "enum": ["ENCASTRE", "PINNED", "XSYMM", "YSYMM",
                                         "ZSYMM", "XASYMM", "YASYMM", "ZASYMM",
                                         "FREE"],
                                "description":
                                    "Abaqus 那套命名边界条件，给了它就不用填六个 0/1。"
                                    "**对称面上该约束哪几个自由度没几个人记得住，"
                                    "而填错了不会报错**——结构照样算得出来，只是算的不是你想要的那个。"
                                    "ENCASTRE 完全固定、PINNED 三向铰接；"
                                    "XSYMM/YSYMM/ZSYMM 是对称面（字母是**法向**），"
                                    "XASYMM/… 是反对称面；FREE 解除约束。"
                                    "用对称边界可以只建半个或四分之一结构——实测半跨门式刚架加 XSYMM 与整跨模型逐项完全一致。"},
                    "spring": {"type": "array", "minItems": 6, "maxItems": 6,
                               "items": {"type": "number", "minimum": 0},
                               "description":
                                   "弹性支座刚度 [kx,ky,kz,krx,kry,krz]，"
                                   "0 表示该方向没有弹簧。平动单位 力/长度"
                                   "（N-m-Pa 下是 N/m），转动单位 力·长度/弧度"
                                   "（N·m/rad）。**同一方向不能既 fix=1 又给"
                                   "弹簧**，刚性约束会让弹簧完全失效，"
                                   "工具会当场拒绝。"},
                    "name": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_boundary_conditions",
            "description":
                "列出全部边界条件，相当于 Abaqus 的 BC Manager："
                "谁、在哪些节点、约束了哪几个自由度、是不是某个标准类型。"
                "\n边界条件是**最容易改错又最难看出来**的一类对象——多约束一个"
                "自由度，结构照样算得出来，只是算的不是用户想要的那个结构。"
                "用户问「现在有哪些边界条件」「支座怎么定的」时用这个。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_boundary_condition",
            "description":
                "按名字删掉一条边界条件。**这会让结构少掉约束**，"
                "工具删完会立刻跑一次奇异诊断；返回里出现 warning 与 modes 时"
                "必须转达——那意味着结构已经变成机构，再求解就会失败，"
                "而那时的错误信息指向的是刚度矩阵，不是刚才删掉的那一条。",
            "parameters": {
                "type": "object", "required": ["name"],
                "properties": {"name": {"type": "string",
                                        "description": "边界条件名称，"
                                                       "用 list_boundary_conditions 查"}},
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
                    "member_id": {
                        "description":
                            "杆件编号、编号列表，或已定义的**集合名**（可混写）。"
                            "一句「给顶层所有梁加 5 kN/m」直接传集合名即可——"
                            "以前要拆成几十次调用，而漏掉其中两根没有任何人会"
                            "发现：工具报成功、校验通过、结果看着也正常。"
                            "每根杆件各自一条命名荷载，可按名单独编辑。",
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "array", "items": {"type": "integer"}},
                            {"type": "string"},
                        ],
                    },
                    "load": {"type": "array", "minItems": 3, "maxItems": 3,
                             "items": {"type": "number"},
                             "description": "全局 [wx,wy,wz]，N/m"},
                    "reference": {"type": "string",
                                  "enum": ["global", "local", "projected"],
                                  "description":
                                      "强度的参照系，默认 global（全局分量、沿杆长）。"
                                      "**斜梁上这个选错，结果看着完全正常但是错的。**"
                                      "projected：强度按每米**水平投影**给——斜屋面的"
                                      "雪载、活载按规范就是这么给的；工具会乘 "
                                      "水平投影/杆长 换算，总合力不变。"
                                      "local：w 是杆件局部分量，风压垂直于杆轴时用它，"
                                      "不必自己拆 sinθ/cosθ。"
                                      "换算过程会原样写进返回值的 conversion 里。"},
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
                           "均布、梯形/三角形、杆中集中力或**部分跨均布**；"
                           "w1 全零表示删除。"
                           "partial 用 a、b 圈出受载区间 [a, b]，强度写在 w1 里——"
                           "砌体墙、局部堆载、只压半跨的活载都是这个形状。"
                           "以前只能拿几个集中力硬凑，凑出来的弯矩图在荷载区内"
                           "是折线而不是抛物线。"
                           "partial_trapezoid 在 [a,b] 内由 w1 线性变到 w2，"
                           "双向板的三角形/梯形分配靠它拼（也可以直接用 "
                           "apply_area_load，那个会替你算好）。"
                           "moment 是**跨间集中力偶**：w1 是力矩矢量（N·m），"
                           "a 是作用位置。预制构件的偏心支承、次梁传来的扭矩、"
                           "牛腿偏心都是它。弯矩图会在 a 处**跳跃**一个 w1，"
                           "那不是画错了。",
            "parameters": {
                "type": "object", "required": ["member_id", "kind", "w1"],
                "properties": {
                    "member_id": {
                        "description":
                            "杆件编号、编号列表，或已定义的**集合名**（可混写）。"
                            "一句「给顶层所有梁加 5 kN/m」直接传集合名即可——"
                            "以前要拆成几十次调用，而漏掉其中两根没有任何人会"
                            "发现：工具报成功、校验通过、结果看着也正常。"
                            "每根杆件各自一条命名荷载，可按名单独编辑。",
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "array", "items": {"type": "integer"}},
                            {"type": "string"},
                        ],
                    },
                    "kind": {"type": "string",
                             "enum": ["uniform", "trapezoid", "point", "partial",
                                      "partial_trapezoid", "moment"]},
                    "w1": {"type": "array", "minItems": 3, "maxItems": 3,
                           "items": {"type": "number"}},
                    "w2": {"type": "array", "minItems": 3, "maxItems": 3,
                           "items": {"type": "number"}},
                    "a": {"type": "number",
                          "description": "距杆件 i 端位置，使用当前模型长度单位。"
                                         "point 是作用点；partial 是受载区间起点"},
                    "b": {"type": "number",
                          "description": "partial 专用：受载区间终点，"
                                         "必须满足 0 ≤ a < b ≤ 杆长"},
                    "reference": {"type": "string",
                                  "enum": ["global", "local", "projected"],
                                  "description":
                                      "强度的参照系，默认 global（全局分量、沿杆长）。"
                                      "**斜梁上这个选错，结果看着完全正常但是错的。**"
                                      "projected：强度按每米**水平投影**给——斜屋面的"
                                      "雪载、活载按规范就是这么给的；工具会乘 "
                                      "水平投影/杆长 换算，总合力不变。"
                                      "local：w 是杆件局部分量，风压垂直于杆轴时用它，"
                                      "不必自己拆 sinθ/cosθ。"
                                      "换算过程会原样写进返回值的 conversion 里。"},
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
            "name": "set_member_strain",
            "description": "给杆件施加装配误差或温度变化（讲义 §3-9 五、六）。"
                           "两者是同一个机制：杆件有初始长度变化，转成沿杆轴的"
                           "等效节点力 EA·ε₀ 进入右端项，回算内力时再减掉，"
                           "所以自由伸缩的杆轴力为零、被约束住才产生内力。"
                           "lack_of_fit 是制造误差 Δl=实际长度−设计长度，"
                           "**正值表示做长了**，与升温同向。"
                           "delta_t 需要材料定义了线膨胀系数 alpha。"
                           "两者可同时给，叠加。全部留空或给零表示删除。",
            "parameters": {
                "type": "object", "required": ["member_id"],
                "properties": {
                    "member_id": {"type": "integer"},
                    "lack_of_fit": {"type": "number",
                                    "description": "制造误差，当前长度单位；正=做长了"},
                    "delta_t": {"type": "number",
                                "description": "温度变化 ℃；正=升温"},
                    "gradient_t": {"type": "number",
                                   "description":
                                       "截面**上下温差**（℃，沿局部 y 即截面高度方向）。"
                                       "与 delta_t 是两回事：delta_t 让杆整体伸缩、"
                                       "产生轴力；gradient_t 让杆想要弯、产生弯矩，"
                                       "κ=α·ΔT/h。日照下的屋面梁、蒸汽管道、"
                                       "大体积混凝土内外温差都是这一类。"
                                       "需要材料有 alpha、截面有 cy（按尺寸建的截面才有）。"},
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
                                                   "a 是距 i 端的距离（单位 m，必须在 0~杆长之间）；"
                                                   "kind=partial 时 w1 是强度（N/m），"
                                                   "a、b 圈出受载区间，0 ≤ a < b ≤ 杆长。"
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
                                            "b": {"type": "number",
                                                  "description": "partial 的区间终点"},
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
            "name": "generate_live_patterns",
            "description":
                "生成活载的**最不利布置**工况：满布、隔跨（奇数跨/偶数跨）、相邻跨。"
                "用户说「活载最不利布置」「隔跨布置」「棋盘布置」时用这个。"
                "\n\n**连续梁与框架的跨中正弯矩不是满布时最大，而是隔跨布置时最大。**"
                "实测三跨框架：跨中弯矩满布 28.7 kN·m、偶数跨布置 41.2 kN·m —— "
                "只算满布会算小 43%，而且算小多少取决于跨数与刚度比，看不出来。"
                "支座负弯矩则常由满布或相邻跨控制（实测相邻跨比满布大 6%）。"
                "\n\n生成的是**工况**不是组合。接着把这些名字一起传给 "
                "generate_combinations 的 live 参数，每种布置都会轮流当控制荷载。"
                "\n跨的归类由几何推断（按沿主导水平轴的起止坐标），返回值里会原样"
                "给出归组结果——**要核对**，混进了柱或另一方向的梁时归出来的跨会很怪。"
                "\n这是规范里的简化做法；严格的「最不利荷载位置」要画影响线逐点找，"
                "本工具不冒充那个，回答时要说明。",
            "parameters": {
                "type": "object",
                "required": ["members", "load"],
                "properties": {
                    "members": {
                        "description": "梁的编号、编号列表或集合名",
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "array", "items": {"type": "integer"}},
                            {"type": "string"},
                        ],
                    },
                    "load": {"type": "array", "minItems": 3, "maxItems": 3,
                             "items": {"type": "number"},
                             "description": "活载线荷载 [wx,wy,wz]，全局坐标，"
                                            "向下写成 [0,0,-w]"},
                    "patterns": {"type": "array",
                                 "items": {"type": "string",
                                           "enum": ["full", "odd", "even",
                                                    "adjacent"]},
                                 "description":
                                     "默认 full/odd/even。adjacent 会生成 N−1 个"
                                     "工况（N 是跨数），跨多时按需开"},
                    "prefix": {"type": "string",
                               "description": "工况名前缀，默认 LL"},
                    "replace": {"type": "boolean",
                                "description": "是否覆盖同名旧工况，默认是"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_combinations",
            "description":
                "按规范生成荷载组合并写入模型。用户说「按规范组合」「做荷载组合」"
                "「1.3 恒 + 1.5 活」时用这个，不要手写系数字典——"
                "**漏了一个组合完全看不出来**：包络只在给定的组合里取极值，"
                "少一个就是少一个，而结果看着完全正常。"
                "\n\n生成三类："
                "\n基本组合（承载能力）—— 每个可变荷载**轮流**当控制荷载，取 γ_Q，"
                "其余取 γ_Q·ψ_c。轮流这一步最容易漏：只算「活载控制」不算"
                "「风控制」，风控制的那些杆件就永远查不出来。"
                "\n恒载有利的基本组合 —— 有风荷载时另生成一组 γ_G=1.0。风吸把柱子"
                "往上拔时恒载是有利的，用 1.3 反而不保守。软件判不了「哪根杆件上"
                "恒载算有利」，所以两组都生成，交给包络逐点挑。"
                "\n标准组合与准永久组合 —— 验挠度、长期变形用。"
                "\n\n**ψ 系数随建筑类别变**：默认是一般民用建筑（住宅、办公）的"
                "常见取值，商业、库房、机房不同。用错不会报错，只会让组合悄悄偏小——"
                "回答时要把用了哪套系数说出来，并提醒核对。"
                "\n组合一经写入，query_envelope 与 check_strength 自动以它们为对象。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dead": {"description": "永久荷载工况名或名字列表",
                             "oneOf": [{"type": "string"},
                                       {"type": "array",
                                        "items": {"type": "string"}}]},
                    "live": {"description": "楼面/屋面活荷载工况",
                             "oneOf": [{"type": "string"},
                                       {"type": "array",
                                        "items": {"type": "string"}}]},
                    "wind": {"description": "风荷载工况。有多个方向就都列上，"
                                            "每个都会轮流当控制荷载",
                             "oneOf": [{"type": "string"},
                                       {"type": "array",
                                        "items": {"type": "string"}}]},
                    "snow": {"description": "雪荷载工况",
                             "oneOf": [{"type": "string"},
                                       {"type": "array",
                                        "items": {"type": "string"}}]},
                    "crane": {"description": "吊车荷载工况",
                              "oneOf": [{"type": "string"},
                                        {"type": "array",
                                         "items": {"type": "string"}}]},
                    "standard": {"type": "string",
                                 "enum": ["GB50068-2018", "GB50009-2012"],
                                 "description":
                                     "默认 GB50068-2018（γ_G=1.3、γ_Q=1.5）。"
                                     "2012 版是 1.2/1.4，既有项目校核才用"},
                    "include_serviceability": {"type": "boolean",
                                               "description": "是否生成标准组合与"
                                                              "准永久组合，默认是"},
                    "psi_c": {"type": "object",
                              "additionalProperties": {"type": "number"},
                              "description":
                                  "覆盖组合值系数，键取 live/wind/snow/crane。"
                                  "建筑类别与默认值不符时**必须**覆盖"},
                    "replace": {"type": "boolean",
                                "description": "是否清掉同名的旧组合，默认是"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_area_load",
            "description":
                "把**面荷载**（力/面积，如楼面恒载 3.5 kN/m²）导成梁上的线荷载。"
                "用户给的荷载是「每平方米多少」时用这个，不要自己去乘从属宽度——"
                "那一步算错结果看着完全正常：量级对、图形也像那么回事，只是总重差一截。"
                "\n\n三种导荷方式，取决于板的长宽比："
                "\none_way —— 单向板，width 是**从属宽度**，梁上均布 q×width；"
                "\ntwo_way_short —— 双向板**短边**梁，三角形，width 是板的短跨 Lx，"
                "跨中峰值 q·Lx/2；"
                "\ntwo_way_long —— 双向板**长边**梁，梯形，width 仍是短跨 Lx，"
                "两端各升 Lx/2 后进入平台 q·Lx/2。"
                "\n\n三角形与梯形是**精确生成**的（用 partial_trapezoid 拼），"
                "不是等效均布——等效均布只保证跨中弯矩相等，支座附近的剪力是另一回事。"
                "\n校核过：4×6 双向板的四边梁合计 96.000 kN，与板自身 q×4×6 分毫不差，"
                "既没漏也没重复计。",
            "parameters": {
                "type": "object",
                "required": ["members", "q", "width"],
                "properties": {
                    "members": {
                        "description": "杆件编号、编号列表或集合名（可混写）",
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "array", "items": {"type": "integer"}},
                            {"type": "string"},
                        ],
                    },
                    "q": {"type": "number", "exclusiveMinimum": 0,
                          "description": "面荷载强度，力/面积。N-m-Pa 下是 N/m²，"
                                         "4 kN/m² 写成 4000"},
                    "width": {"type": "number", "exclusiveMinimum": 0,
                              "description": "one_way 下是从属宽度；"
                                             "两种 two_way 下是板的**短跨** Lx"},
                    "load_path": {"type": "string",
                                  "enum": ["one_way", "two_way_short",
                                           "two_way_long"],
                                  "description": "导荷方式，默认 one_way"},
                    "direction": {"type": "array", "minItems": 3, "maxItems": 3,
                                  "items": {"type": "number"},
                                  "description": "荷载方向，默认全局 −Z（重力）。"
                                                 "只取方向，大小由 q 决定"},
                    "case_name": {"type": "string"},
                    "name": {"type": "string",
                             "description": "荷载名前缀；每段生成 前缀-杆号-序号"},
                },
                "additionalProperties": False,
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
            "name": "response_spectrum_analysis",
            "description":
                "振型分解反应谱法（地震作用）。用户说「算地震」「按抗规做」"
                "「7 度设防」「水平地震作用」时用这个。"
                "\n\n**返回的内力和位移没有符号。** SRSS 与 CQC 都是平方和开方，"
                "出来的只有大小。地震往复，与重力组合时必须按 ± 各算一次——"
                "直接当普通工况叠加进去，得到的只是两个方向里恰好同号的那一个。"
                "返回里的 sign 字段要原样转达。"
                "\n谱二选一：alpha_max + tg 走 GB 50011 设计谱（地震影响系数，"
                "多遇地震 7 度 0.08、8 度 0.16），或 spectrum_points 给自定义"
                "(周期, 加速度) 表。"
                "\nmass_ratio 低于 0.9 时返回里会有 mass_ratio_warning，"
                "要照做加大 num_modes——GB 50011 要求累计参与质量不小于 90%。",
            "parameters": {
                "type": "object", "properties": {
                    "direction": {"type": "string", "enum": ["x", "y", "z"]},
                    "num_modes": {"type": "integer", "minimum": 2},
                    "alpha_max": {"type": "number", "exclusiveMinimum": 0,
                                  "description": "地震影响系数最大值，"
                                                 "如多遇地震 7 度 0.08"},
                    "tg": {"type": "number", "exclusiveMinimum": 0,
                           "description": "特征周期 s"},
                    "spectrum_points": {
                        "type": "array", "minItems": 2,
                        "items": {"type": "array", "minItems": 2,
                                  "maxItems": 2,
                                  "items": {"type": "number"}},
                        "description": "自定义谱 [[周期, 加速度 m/s²], ...]"},
                    "combination": {"type": "string", "enum": ["CQC", "SRSS"],
                                    "description": "频率接近的振型不独立，"
                                                   "默认 CQC"},
                    "damping": {"type": "number", "exclusiveMinimum": 0,
                                "exclusiveMaximum": 1},
                    "gravity": {"type": "number", "exclusiveMinimum": 0},
                }, "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_member_connection",
            "description":
                "给杆端装半刚性连接（转动弹簧）。用户说「梁柱节点不是刚接」"
                "「端板连接」「角钢连接」「节点有转动刚度」时用这个。"
                "\n\n**真实的梁柱节点既不是铰也不是刚接。** 两头都按极限算，"
                "弯矩分布差得很远：两跨连续梁实测，刚接支座弯矩 90 kN·m，"
                "连接刚度取 3EI/L 时只剩 45 kN·m，差出来的那部分全跑到跨中去了"
                "——而两种算法都不会报错。回答时要把这个差别说出来。"
                "\n刚度优先用 connection_type 按**梁线刚度 EI/L 的倍数**给"
                "（端板 20、平端板 8、顶底角钢 3、腹板角钢 1）；绝对刚度离开"
                "截面和跨度就没有意义，同一个端板装在不同梁上相对刚度差几倍。"
                "\n理想铰是 k=0 的极限，用杆端释放表达，不要在这里填 0；"
                "刚接是 k=∞ 的极限，不装弹簧就是。",
            "parameters": {
                "type": "object", "required": ["member_ids"],
                "properties": {
                    "member_ids": {
                        "description": "杆件编号、编号列表或集合名",
                        "oneOf": [
                            {"type": "integer"},
                            {"type": "array", "items": {"type": "integer"}},
                            {"type": "string"},
                        ],
                    },
                    "end": {"type": "string", "enum": ["i", "j", "both"],
                            "description": "装在哪一端；默认 j"},
                    "dof": {"type": "string",
                            "enum": ["ux", "uy", "uz", "rx", "ry", "rz"],
                            "description": "局部自由度；梁端弯矩通常是 rz"},
                    "stiffness": {"type": "number", "exclusiveMinimum": 0,
                                  "description": "绝对连接刚度，力·长度/弧度"},
                    "connection_type": {
                        "type": "string",
                        "enum": ["端板", "平端板", "顶底角钢", "腹板角钢"],
                        "description": "按 EI/L 的倍数给刚度，比绝对值可用"},
                    "clear": {"type": "boolean",
                              "description": "去掉该端连接弹簧，恢复刚接"},
                }, "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_step",
            "description":
                "新增一个分析步（Abaqus 的 Step）。用户说「先满载再撤活载」"
                "「拆掉这个支座看看」「分几个工况阶段算」时用这个。"
                "\n\n**荷载与边界条件在步之间传播**：这一步只写新建或改写的东西，"
                "上一步有而这里没提的会自动沿用。要让它消失必须显式写进 "
                "deactivate_loads / deactivate_supports——**漏写不等于撤销**，"
                "这是最容易向用户解释错的一点，回答时要讲明白。"
                "\nloads 是 工况名 -> 幅值曲线名；supports 是 节点号 -> "
                "{fix: [...], spring: [...]}，用来在这一步改写某个支座。"
                "\n每一步都从未变形、无应力状态重解，不把上一步的状态带进来，"
                "杆件也不能在步之间生灭——所以这表达的是「同一结构的几种配置」，"
                "不是施工过程。用户问施工顺序时要说清楚这个区别。",
            "parameters": {"type": "object", "required": ["name"], "properties": {
                "name": {"type": "string", "description": "分析步名；Initial 是保留名"},
                "analysis": {"type": "string", "enum": ["linear", "pdelta"],
                             "description": "材料非线性暂不支持分析步"},
                "loads": {"type": "object",
                          "additionalProperties": {"type": "string"},
                          "description": "本步新建或改写的荷载：工况名 -> 幅值曲线名"},
                "deactivate_loads": {"type": "array", "items": {"type": "string"},
                                     "description": "本步起失活的工况名"},
                "supports": {"type": "object",
                             "additionalProperties": {"type": "object"},
                             "description": "本步新建或改写的支座：节点号 -> {fix, spring}"},
                "deactivate_supports": {"type": "array", "items": {"type": "integer"},
                                        "description": "本步起拆掉的支座所在节点号"},
                "increments": {"type": "integer", "minimum": 1},
                "after": {"type": "string",
                          "description": "插在这一步之后；留空则追加到末尾"}
            }, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_steps",
            "description":
                "列出全部分析步，声明与**实际生效**分两栏。"
                "\n传播是隐式的：某一步写着一行「失活活载」，生效的却是前面传下来"
                "的一整套减掉活载。只看声明会把这件事看漏，所以回答用户"
                "「第几步在算什么」时一律看 effective 那一栏，不要看 declared。",
            "parameters": {"type": "object", "properties": {},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_step",
            "description":
                "删除一个分析步。**后面的步会跟着变**：删掉建立某个荷载的那一步，"
                "后面的「失活」就失去对象，整串分析步可能结算不出来——"
                "那种情况下工具会拒绝删除并保持模型原样，把错误原样转达给用户。",
            "parameters": {"type": "object", "required": ["name"], "properties": {
                "name": {"type": "string"}
            }, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_steps",
            "description":
                "按顺序求解模型里声明的全部分析步，每一步给一行结果。"
                "有分析步时用这个，不要退回 solve_model 一步步手动算——"
                "手动算会丢掉传播，而漏掉一个上一步的荷载看不出来。"
                "\n会话一次只端得住一份结果供后处理（画图、验算、查内力）；"
                "默认留最后一步，用 inspect 指定留哪一步。返回里的 limitation "
                "要原样转达：每一步都从未变形、无应力状态重解，杆件不能在步之间"
                "生灭，所以这不是施工过程分析。",
            "parameters": {"type": "object", "properties": {
                "inspect": {"type": "string",
                            "description": "哪一步的结果留在会话里；留空取最后一步"}
            }, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "define_amplitude",
            "description":
                "定义一条幅值曲线（Abaqus 的 Amplitude），给 solve_model(analysis=\"step\") 用。"
                "曲线是 (伪时间, 系数) 表，线性插值，表外取端点值不外推。"
                "伪时间是分析步内部进度 0→1，不是真实时间——本内核是静力分析。"
                "\n内置 RAMP（0→1 斜坡）与 STEP（全程为 1）已够用大多数情形，"
                "只有需要「前 30% 就加满再保持」这类非线性路径时才定义新的。",
            "parameters": {"type": "object", "required": ["name"], "properties": {
                "name": {"type": "string", "description": "曲线名；不能叫 RAMP 或 STEP"},
                "points": {"type": "array", "minItems": 2,
                           "items": {"type": "array", "minItems": 2, "maxItems": 2,
                                     "items": {"type": "number"}},
                           "description": "[[伪时间, 系数], ...]，时间必须严格递增"},
                "times": {"type": "array", "items": {"type": "number"},
                          "description": "与 values 配对给，替代 points"},
                "values": {"type": "array", "items": {"type": "number"}}
            }, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_amplitudes",
            "description": "列出可用的幅值曲线，含内置的 RAMP 与 STEP 及其含义。"
                           "用户问「有哪些加载方式」或要做推覆时先看这个。",
            "parameters": {"type": "object", "properties": {},
                           "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_amplitude",
            "description": "删除一条自定义幅值曲线。内置的 RAMP 与 STEP 删不掉。",
            "parameters": {"type": "object", "required": ["name"], "properties": {
                "name": {"type": "string"}
            }, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_model",
            "description":
                "求解当前模型。支持线性静力、P-Delta 二阶弹性、双线性轴向材料非线性。"
                "\n\n**报位移时用 max_deflection_mm，不要用 max_displacement_mm。**"
                "后者只扫节点，而满跨均布、梯形荷载不触发杆件剖分，跨中根本没有节点"
                "可查——实测 87 杆三层框架，节点值 4.30 mm、真实挠度 9.27 mm，差 2.16 "
                "倍；简支梁上后者直接是 0.0。前者含单元内部，校核挠跨比用它。"
                "\n返回里若有 note 或 warning，原样转达。note 常常是在解释「节点位移为零"
                "是正常的」，漏掉它用户会以为荷载加错了。"
                "\n\n**analysis=\"step\" 是非比例加载**：几个工况同时施加，"
                "各按自己的幅值曲线随分析步变化。推覆分析要它——"
                "{\"DL\":\"STEP\",\"WX\":\"RAMP\"} 表示重力全程加满、侧力线性上升。"
                "**它不改变弹性分析的终点**（二阶弹性解与加载路径无关），"
                "改变的是路径；别把它当成更准的算法。用户要能力曲线时看返回的 "
                "analysis.convergence，每个增量带 max_displacement。",
            "parameters": {"type": "object", "properties": {
                "analysis": {"type": "string",
                             "enum": ["linear", "pdelta", "material", "step"],
                             "description": "step 是非比例加载的 P-Delta 分析步，需要配 amplitudes"},
                "amplitudes": {"type": "object",
                               "additionalProperties": {"type": "string"},
                               "description": "analysis=step 时必填：工况名 -> 幅值曲线名，例如 {\"DL\": \"STEP\", \"WX\": \"RAMP\"}"},
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
            "description": "【实验性】用 Abaqus 求解同一个模型（可选后端，需要本机装有 Abaqus）。"
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
            "name": "analyze_joint_solid",
            "description": "【实验性】对某个节点做局部实体子模型，观察杆件相交处的应力集中。"
                           "整体仍用梁模型，只把相连杆件截成短臂拼成 C3D10 实体，"
                           "切割面传入梁模型的六分量截面力。"
                           "默认由自研 native-solid 有限元内核求解，Gmsh只负责网格；"
                           "也可指定 Abaqus 作为独立商软对标。"
                           "只支持圆钢/圆管、同一种材料、无刚域偏移的节点，"
                           "其余情况明确拒绝，不做几何猜测。"
                           "几何不含焊缝倒圆，相贯线是应力奇异点，因此峰值应力只用于"
                           "观察分布；应力集中系数一律以 IIW 0.4t/1.0t 线性外推的"
                           "热点应力为分子，外推拿不到就不给系数。"
                           "dry_run 只建规格、返回杆端力与名义应力，不需要本机装 Abaqus。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "integer",
                                "description": "要放大观察的节点编号"},
                    "case": {"type": "string",
                             "description": "工况或组合名；留空取控制工况"},
                    "anchor_member": {"type": "integer",
                                      "description": "作为固定端的杆件；"
                                                     "留空优先取通向支座的那根"},
                    "mesh_sizes_mm": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 1,
                        "description": "网格档位（mm）；一档可求解，至少三档才能判断收敛与发散",
                    },
                    "backend": {"type": "string", "enum": ["native", "abaqus"],
                                "description": "默认 native；abaqus 用于商软对标"},
                    "dry_run": {"type": "boolean",
                                "description": "只建规格，不调用实体求解器"},
                },
                "required": ["node_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_solvers",
            "description": "【实验性】同一个模型两个后端各算一遍，逐分量给出归一化偏差。"
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
                           "结果是结构固有属性，与荷载无关——不要传工况。"
                           "\n返回里出现 mesh_warning 时**必须转达**：一致质量阵靠形函数"
                           "装配，网格越粗频率报得越高。实测 8 m 简支梁一跨一个单元时"
                           "基频偏高 11%，四个单元降到 0.026%。"
                           "\n\n**参与质量比的分母是 participable_mass，不是 total_mass。**"
                           "压在支座上的质量永远不参与振动，两者的差随网格变粗而变大"
                           "——一根剖成 4 段的悬臂柱，按 total_mass 算把振型取满也只有"
                           "84%。所以看 cumulative_mass_ratio_xyz，不要自己拿有效质量"
                           "去除 total_mass；那样算出来的比值永远到不了 1，而"
                           "「加大 num_modes」这条建议在那里是无效的。",
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
                           "实际承载力更低，回答时必须说明这一点。"
                           "\n\n**返回里出现 mesh_warning 时必须转达，它比上面那句更要命。**"
                           "几何刚度阵靠形函数装配，网格越粗 λ 报得越高。实测单跨门式刚架"
                           "每构件一个单元时 λ=1.66，收敛值 0.82——偏高 102%：前者在说"
                           "「还有 66% 余量」，后者意味着它已经失稳。generate_frame 生成的"
                           "恰好是每构件一个单元，所以这是最容易撞上的情形。"
                           "\n失败时若有 likely_cause，先讲它，不要照搬「检查荷载方向」。",
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
            "name": "check_strength",
            "description":
                "强度验算 + 逐杆稳定校核（讲义 §3-9 三）。"
                "用户问「够不够」「安全吗」「应力比」「会不会压屈」「校核一下」时用这个。"
                "逐杆给出：最不利截面的应力比 σ/[σ]（**拉压分别对各自的许用值**）、"
                "受压杆的欧拉临界力 Pcr=π²EI/(μl)²、长细比 λ、回转半径。"
                "需要材料定义了 allow_tension（许用拉应力）、截面定义了 cy/cz。"
                "同时给出**折算应力 √(σ²+3τ²)**（第四强度理论，也是 GB 50017 §6.1.5 "
                "的式子）与材力四个强度理论的相当应力 σr1~σr4，以及 **GB 50017 稳定"
                "系数 φ** 和 N/(φA)≤f 的规范校核。"
                "\n\n**正应力与折算应力是两条独立结论，都要看。** 只看正应力会漏掉剪切"
                "控制的构件：实测 L/h=2.5 的短深梁，正应力比 0.064「安全得很」，折算"
                "应力却是它的 2.78 倍，控制点在中性轴——那里弯曲 σ=0 而 τ 最大。"
                "\n**欧拉与规范法冲突时以规范法为准。** 欧拉在中小柔度段不适用，而实际"
                "钢柱大多落在那里：实测 λ=24.9 的粗短柱，欧拉 N/Pcr=0.023、规范 "
                "N/(φA)/f=0.822，差 35 倍且偏不安全。"
                "\n能力边界：**仍不含扭转剪应力**，所以还不是完整的规范承载力验算；"
                "回答时要把这一条说出来，也要把返回的 warnings 原样转达。",
            "parameters": {
                "type": "object",
                "properties": {
                    "members": {"type": "array", "items": {"type": "integer"},
                                "description": "要验算的杆件号；留空表示全部"},
                    "cases": {"type": "array", "items": {"type": "string"},
                              "description": "参与包络的工况；留空表示全部"},
                    "slenderness_limit": {"type": "number", "exclusiveMinimum": 0,
                                          "description": "允许的最大长细比 [λ]，"
                                                         "如钢压杆常取 150；留空不查"},
                    "buckling_curve": {"type": "string", "enum": ["a", "b", "c", "d"],
                                       "description":
                                           "GB 50017 截面类别，只影响稳定系数 φ。"
                                           "默认 b（热轧工字钢、H 型钢绕强轴的常见归类）。"
                                           "**不是可以忽略的细节**：λ=80 时 a 类 φ=0.783、"
                                           "d 类 0.493，差 59%。拿不准就沿用默认并说明。"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_symmetry",
            "description":
                "对称性检测与对称性校核（讲义 §3-9 四）。"
                "判断结构关于哪个坐标面镜像对称、各工况荷载是对称还是反对称；"
                "已求解时还会检查对称位置的位移是否真的互为镜像——"
                "**这是一条不需要任何外部参照的自校核**，能查出装配、坐标转换、"
                "等效节点荷载里的一大类错误。"
                "用户问「对称吗」「能不能取半结构」「结果对不对」时用这个。"
                "本程序直接求解全结构，**不做半结构简化**，原因见返回的 note。",
            "parameters": {
                "type": "object",
                "properties": {
                    "case": {"type": "string",
                             "description": "要做位移镜像校核的工况；留空取控制工况"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_numbering",
            "description":
                "节点编号与总刚存储方案对比（讲义 §3-9 八、§3-10、§4-6）。"
                "给出当前编号的节点号差与半带宽、RCM 重编号后的改善，"
                "以及满阵 / 等带宽 / 一维变带宽 / 稀疏四种存储量的对比。"
                "verify=true 时再用讲义的变带宽 LDLᵀ 解一遍，与稀疏 LU 的结果对表。"
                "用户问「带宽」「编号」「存储」「课上讲的一维存储」时用这个。"
                "**不会改动模型里的节点号**——重编号只发生在求解内部。",
            "parameters": {
                "type": "object",
                "properties": {
                    "verify": {"type": "boolean",
                               "description": "是否用变带宽 LDLᵀ 再解一遍对表，默认 true"},
                },
                "additionalProperties": False,
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


# 实验性工具：代码可用、有自动测试，但**不在正式主链路的承诺范围内**。
#
# 分级依据不是「做得好不好」，而是「有没有闭环证据」：
# - Abaqus 两个工具依赖本机许可证，CI 上跑不到，只能靠本机冒烟；
# - 节点实体子模型单元层有教科书解对照，但 Agent 评测集没有一道题碰它，
#   模型在对话里会不会正确调用、会不会把奇异峰值当 Kt 报出去，没有数据。
#
# 描述前缀「【实验性】」是给大模型看的，这份集合是给测试看的——
# `tests/test_tool_contracts.py` 核对两边一致，README 和能力矩阵也要标出来。
# 某个工具有了评测题、跑出稳定成绩，再把它从这里拿掉。
EXPERIMENTAL_TOOLS: frozenset[str] = frozenset({
    "solve_with_abaqus",
    "compare_solvers",
    "analyze_joint_solid",
})
