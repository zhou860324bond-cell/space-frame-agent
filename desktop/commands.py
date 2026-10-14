"""动作表。

功能区、菜单、快捷键**共用同一批 QAction**——在这里声明一次。分开定义的话，
迟早会出现"菜单里是灰的、功能区里是亮的"这种自相矛盾的界面。

每条记录说清楚四件事：叫什么、什么图标、点了调谁、鼠标停上去解释什么。
`handler` 是主窗口上的方法名，晚绑定（点的时候才 getattr），
所以这张表不依赖主窗口的构造顺序。

---

"""

from __future__ import annotations

from typing import NamedTuple


class Command(NamedTuple):
    name: str
    label: str
    icon: str
    tip: str
    handler: str                    # 主窗口上的方法名，晚绑定
    shortcut: str = ""
    checkable: bool = False


COMMANDS: tuple[Command, ...] = (
    # --- 项目 ---
    Command("new", "新建", "new", "清空当前模型，从头开始", "new_model", "Ctrl+N"),
    Command("open", "打开", "open", "从 JSON 文件读入模型", "open_model", "Ctrl+O"),
    Command("save", "保存", "save", "把当前模型存成 JSON", "save_model", "Ctrl+S"),
    Command("units", "单位制", "units",
            "在 N-m-Pa 与 N-mm-MPa 之间转换整份模型的数值",
            "edit_units", ""),
    Command("chat", "Agent 对话", "chat", "显示或隐藏右侧的对话面板",
            "toggle_chat", "Ctrl+G", True),
    Command("props", "属性", "table",
            "显示选中节点或杆件的属性并就地修改", "toggle_props", "Ctrl+P", True),

    # --- 编辑 ---
    #
    # 撤销之所以要紧，是因为 Agent 一轮就能改动整个模型。没有退路，
    # 用户每次回车前都要犹豫；有了退路，才敢放手让它试。
    # 快照本来就存在建模过程里，这里只是把它接出来。
    Command("undo", "撤销", "undo", "退回上一步建模操作（结果会一并清除）",
            "undo", "Ctrl+Z"),
    Command("redo", "重做", "redo", "重做被撤销的一步", "redo", "Ctrl+Y"),
    Command("timeline", "建模过程", "timeline",
            "逐步回看建模过程，点任意一步跳回那时的模型", "show_timeline",
            "Ctrl+H", True),
    Command("learning_trace", "导出学习轨迹", "report",
            "导出工具实参、结果、模型变化和校验状态，供 Agent 回放与训练筛选",
            "export_learning_trace"),

    # --- 建模 ---
    Command("modeling_workflow", "刚架建模全流程", "timeline",
            "按九个CAE阶段查看当前定义、遗漏项和真实操作入口", "open_modeling_workflow"),
    Command("assembly_check", "装配检查", "geometry",
            "检查重合节点、重复杆件和端点引用，报告问题位置", "check_assembly"),
    Command("model_sets", "命名集合", "table",
            "为节点或杆件组建立可保存、可复用的名字", "open_model_sets"),
    Command("batch_assignment", "批量属性指派", "assign_section",
            "按编号、方向或集合挑出多根杆件，建一个截面指派对象（与截面指派同一种对象，"
            "在截面指派管理器里管理）", "assign_member_group"),
    Command("beam_orientation", "梁截面方向", "iso",
            "Abaqus 式 Assign Beam Orientation：先在视口选杆件区域（Shift 加选、Ctrl 减选、拖框，"
            "或使用集合），再给截面高度方向 n1；方向是一个对象，区域是集合时随集合更新",
            "assign_beam_orientation"),
    Command("member_connections", "杆端连接", "hinge",
            "逐个局部转动方向设置刚接、释放或有依据的弹簧刚度", "set_member_connections"),
    Command("model_check", "完整性校验", "geometry",
            "提交前执行确定性完整性校验，并显示原因和问题编号", "check_manual_model"),
    Command("support_diagnosis", "机构诊断", "support",
            "用约束后的刚度阵定位零刚度运动，报告节点及自由度", "run_diagnose"),
    Command("seed_part", "全局布种", "mesh",
            "Abaqus 式 Seed Part：给全部杆件设默认段数（每根杆件剖成几个分析杆段）；"
            "频率、屈曲、P-Δ 本来就会每杆至少 4 段", "seed_part"),
    Command("seed_edges", "杆件布种", "mesh",
            "Abaqus 式 Seed Edges：先在视口选杆件（Shift 加选、Ctrl 减选、拖框，或使用集合），"
            "再给段数，覆盖全局布种", "seed_edges"),
    Command("clear_seeds", "清除布种", "remove",
            "清掉全部布种，回到每杆 1 段（分析时按类型自动加密）", "clear_seeds"),
    Command("mesh_preview", "预览梁网格", "mesh",
            "查看自动剖分后的分析杆段、节点及物理杆件编号映射", "show_analysis_mesh"),
    Command("job_manager", "作业管理器", "solve",
            "Abaqus 式 Job Manager：创建作业，数据检查、写 inp、后台提交、监控消息、"
            "查看结果、终止；结果与提交时的模型版本对应", "open_job_manager"),
    Command("solve_steps", "提交分析步", "solve",
            "在后台按声明顺序求解分析步，保留最后一步的有效结果", "solve_declared_steps"),
    # 参数化建模使用直接操作的对话框。
    Command("sketch", "草图建模", "sketch",
            "2D 画一榀框架草图，然后沿进深拉伸成 3D 空间刚架（Abaqus 式）",
            "open_sketch", "Ctrl+K"),
    Command("sketch_ai", "草图识别", "sketch_ai",
            "上传手绘草图，核对并编辑 AI 识别覆盖层，再标定尺度后加载",
            "open_sketch_ai", "Ctrl+I"),
    Command("frame", "规则框架", "frame",
            "弹出参数对话框，直接生成规则空间刚架（不经过 AI）",
            "open_parametric", "Ctrl+F"),
    Command("partition_member", "分割杆件", "member",
            "Abaqus 式 Partition Edge：选一根杆件，按比例加一个顶点，之后可把别的杆件连到这里；"
            "物理杆件仍是一根，荷载、释放、截面指派不用改", "partition_member_ui"),
    Command("datum_point", "基准点", "create_node",
            "Abaqus 式 Datum Point：输入坐标，或选两个节点取中点；只用于定位与建节点吸附，不参与计算",
            "datum_point_ui"),
    Command("measure", "测量", "fit",
            "Abaqus 式 Query：选两个节点测距离与分量，或选一两根杆件测长度、夹角", "measure_ui"),
    Command("tables", "模型表格", "table",
            "七张表直接编辑节点、杆件、约束与荷载，支持 CSV 导入导出",
            "open_tables", "Ctrl+T"),
    Command("json", "模型 JSON", "report",
            "直接编辑模型的原始数据，写回前会做结构校验", "open_json"),
    Command("portal", "门式刚架", "portal",
            "弹出参数对话框，直接生成坡屋面门式刚架（不经过 AI）",
            "open_parametric"),
    Command("section", "截面管理器", "section", "弹出对话框编辑材料与截面定义", "edit_materials_sections"),
    Command("material", "创建材料", "material",
            "Abaqus 式：定义弹性模量、泊松比、密度", "create_material"),
    Command("beam_section", "创建截面", "section",
            "Abaqus 式：选择梁截面类型（I/管/矩形等），输入尺寸，自动计算 A/I/J",
            "create_section"),
    Command("assign_section", "属性指派", "assign_section",
            "Abaqus 式截面指派：先在视口选杆件区域（Shift 加选、Ctrl 减选、拖框，或使用集合），"
            "再选截面与材料；指派是一个对象，区域是集合时随集合更新", "assign_section"),
    Command("assignment_manager", "截面指派管理器", "assign_section",
            "列出全部截面指派对象：创建、编辑、编辑区域、复制、改名、删除、抑制、恢复",
            "open_assignment_manager"),
    Command("hinge", "创建铰接", "hinge",
            "选中杆件后在 i端/j端/两端创建转动释放", "create_hinge"),
    Command("brace", "加支撑", "member",
            "在选中杆件所在节间加 X 型支撑。钢框架最有效的抗侧手段", "add_brace"),

    # --- 边界条件 ---
    Command("load", "载荷管理器", "load",
            "列出全部荷载对象：创建、编辑、编辑区域、复制、改名、删除、抑制、恢复", "open_load_manager"),
    Command("combo", "荷载组合", "combo", "弹出对话框编辑荷载组合（按分项系数组合）", "edit_load_combos"),
    Command("gravity", "重力", "gravity",
            "Abaqus 式 Gravity 载荷：创建载荷，类型预选重力，区域默认全部杆件；"
            "按材料密度与截面面积计算，结果是一个可编辑、可抑制的荷载对象", "apply_gravity"),
    Command("create_constraint", "创建约束", "hinge",
            "Abaqus 式 Create Constraint：名称、类型（MPC BEAM / TIE / PIN、运动耦合）→ 选主节点 → "
            "选从节点区域（Shift 加选、Ctrl 减选、拖框，或使用集合）→ 运动耦合勾选自由度", "create_constraint_ui"),
    Command("constraint_manager", "约束管理器", "hinge",
            "列出全部约束：创建、编辑、编辑区域、复制、改名、删除、抑制、恢复", "open_constraint_manager"),
    Command("create_bc", "创建边界条件", "support",
            "Abaqus 式：名称、分析步、类型 → 在视口选节点区域（Shift 加选、Ctrl 减选、拖框，"
            "或使用集合）→ 勾选 U1~UR3 或填弹性支承刚度", "create_bc"),
    Command("create_load", "创建载荷", "load",
            "Abaqus 式：名称、工况、类型（集中力/力矩/线荷载/重力）→ 在视口选区域"
            "（节点或杆件；点选、Shift 加选、Ctrl 减选、拖框，或使用集合）→ 填全局分量",
            "create_load"),

    # --- 分析 ---
    Command("pick_node", "选择节点", "pick_node",
            "拾取过滤器：只选中节点。先定选什么类型再点，"
            "否则三维视图里分不清点到的是节点还是杆件",
            "set_pick_node", "N", True),
    Command("pick_member", "选择杆件", "pick_member",
            "拾取过滤器：只选中杆件", "set_pick_member", "M", True),
    Command("model_node", "建节点", "create_node",
            "人工建模：在视口中点击任意位置创建节点",
            "set_model_node", "Ctrl+Shift+N", True),
    Command("model_member", "建杆件", "create_member",
            "人工建模：先点击一个节点，再点击另一个节点，在两者之间创建杆件",
            "set_model_member", "Ctrl+M", True),
    Command("delete", "删除选中", "remove",
            "删除当前选中的节点或杆件（Delete 键）", "delete_selected", "Delete"),
    Command("render_profiles", "截面渲染", "section",
            "按真实截面形状（工字、箱形、圆管、角钢等）显示杆件，看得出截面朝向与强弱轴；"
            "没有截面几何的杆件仍画成圆管", "toggle_render_profiles", "", True),
    Command("labels", "编号标注", "labels",
            "在视口中显示节点与杆件编号。定位报错和结果时必需",
            "toggle_labels", "L", True),
    # 荷载数值标注单独给一个开关。模型一大，几十个数字糊成一片，
    # 遮住的正是它们要说明的那根杆件——这时候能关掉比标得全更重要。
    Command("load_labels", "荷载数值", "load",
            "在荷载箭头旁标出数值。模型大时可以关掉，只看箭头方向和分布",
            "toggle_load_labels", "Shift+L", True),
    Command("lang", "中 / EN", "units",
            "在中文与英文界面之间切换（功能区、菜单、状态栏；对话框仍为中文）",
            "toggle_language", "", True),

    # 这个按钮从前叫「分析步」，但它做的是选求解过程和增量数——跟真正的
    # Step 对象没有关系。名字改诚实，真正的分析步管理器另立一个。
    Command("solve_options", "求解设置", "timeline",
            "选择线性静力、P-Delta 或双线性轴向材料非线性及增量参数",
            "edit_analysis_step"),
    Command("step_manager", "分析步", "timeline",
            "分析步管理器：新建、删除、按顺序求解。荷载与边界条件在步之间传播，"
            "上一步有而这一步没提的会自动沿用",
            "open_step_manager"),
    Command("code_combo", "规范组合", "combo",
            "按 GB 50068 / GB 50009 生成荷载组合，每个可变荷载轮流当控制荷载。"
            "漏一个组合完全看不出来",
            "open_code_combinations"),
    Command("live_pattern", "活载布置", "load",
            "活载最不利布置：只算满布会把跨中弯矩算小 43%",
            "open_live_patterns"),
    Command("area_load", "面荷载", "load",
            "把 kN/m² 按单向板或双向板导成梁上的线荷载",
            "open_area_load"),
    Command("bc_manager", "边界条件管理器", "support",
            "整张表列出全部边界条件并可删除。改错了不报错，摆出来才核对得了",
            "open_bc_manager"),
    Command("amplitude", "幅值曲线", "timeline",
            "幅值曲线管理器：荷载在分析步内怎么随伪时间变化。"
            "重力配 STEP、侧力配 RAMP 就是推覆加载",
            "open_amplitude_manager"),
    Command("solve", "求解", "solve", "按当前分析步求解，并执行收敛与平衡检查",
            "solve", "F5"),
    Command("stop_task", "停止任务", "remove",
            "请求在安全检查点停止当前后台任务；矩阵分解完成前请等待",
            "stop_task"),
    Command("modal", "模态", "modal", "自振频率与振型（需要材料密度）",
            "run_modal", "Ctrl+4", True),
    Command("buckling", "屈曲", "buckling", "线弹性屈曲因子与失稳模态",
            "run_buckling"),
    Command("diagnose", "模型检查", "support",
            "AI 辅助：检查模型完整性（支座/荷载/孤立节点/重复杆件），给出修正建议", "diagnose_model"),
    Command("analysis_mesh", "分析网格", "mesh",
            "只读查看物理杆件经自动剖分后，求解器实际使用的节点和杆段",
            "show_analysis_mesh", "", True),
    Command("solid_joint", "节点实体", "contour",
            "选中已求解的圆管节点，用自研 C3D10 局部实体求解器计算应力；"
            "Gmsh只负责网格，Abaqus保留为独立对标后端",
            "run_solid_joint"),

    # --- 结果 ---
    Command("model", "模型", "geometry", "显示几何、支座与荷载符号",
            "show_model", "Ctrl+1", True),
    Command("deformed", "变形", "deformed", "变形图（叠未变形轮廓）",
            "show_deformed", "Ctrl+2", True),
    Command("contour", "梁内力云图", "contour", "梁中心线内力结果；不是实体截面应力云图。合量用顺序色标，有符号局部分量关于零对称",
            "show_contour", "Ctrl+3", True),
    Command("solid_contour", "实体云图", "contour",
            "在主视口查看节点实体应力与位移，可旋转、缩放并切换网格档位",
            "show_solid_contour", "Ctrl+7", True),
    Command("open_solid_result", "打开实体结果", "open",
            "打开已保存的节点实体 JSON、VTU 或 NPZ 结果，在主视口查看",
            "open_solid_result"),
    Command("force_diagram", "内力图", "diagram",
            "三维内力图：每根杆旁画出内力沿杆的分布，高度即数值，"
            "弯矩画在受拉侧（SAP2000 / 盈建科式）",
            "show_force_diagram", "Ctrl+5", True),
    Command("utilization", "应力比", "strength",
            "应力比图：每根杆件按强度 / 折算应力 / 稳定三者最大值分档着色，"
            "超过 1.0 标红；判不了的画灰色。首次打开时先执行强度验算",
            "show_utilization", "Ctrl+6", True),
    Command("diagram", "内力分量", "diagram",
            "选择云图与内力图显示哪个内力分量", "pick_component"),
    Command("display_only", "仅显示所选", "geometry",
            "Abaqus 式显示组：在视口选杆件（Shift 加选、Ctrl 减选、拖框，或使用集合）后只显示它们；"
            "隐藏的杆件也不参与拾取", "display_only"),
    Command("display_hide", "隐藏所选", "remove",
            "在视口选杆件后把它们隐藏，看被遮住的部分", "display_hide"),
    Command("display_all", "全部显示", "fit", "取消显示组，显示全部杆件", "display_all"),
    Command("animate", "动画", "modal",
            "变形图与振型按比例系数来回播放；再点一次停止", "toggle_animation", "", True),
    Command("path_plot", "路径曲线", "diagram",
            "选一串首尾相连的杆件，画内力沿路径的分布（Abaqus 的 XY Data from path）",
            "path_plot"),
    Command("clear_results", "清除结果", "remove",
            "清除当前计算结果，保留模型、材料、约束与荷载", "clear_results"),
    Command("curve", "单杆内力图", "diagram",
            "沿杆长的内力曲线：峰值位置、过零点、各组合包络",
            "show_diagram", "Ctrl+D", True),
    Command("envelope", "包络", "envelope",
            "各分量沿杆件的极值包络与控制组合", "run_envelope"),
    Command("deflection", "最大挠度", "deformed",
            "最不利杆件的挠度、位置与挠跨比", "run_deflection"),
    Command("report", "报告", "report", "导出 Markdown + Word 计算书",
            "write_report"),

    # --- 校核 ---
    # 这三项对应课程讲义「程序灵活应用」里的三条，是结构程序该有的东西，
    # 不是附加功能：算完了不校核，等于只交了一半。
    Command("strength", "强度验算", "strength",
            "逐杆应力比 σ/[σ]（拉压许用分开）与受压杆的欧拉临界力 Pcr，"
            "需要材料给出许用应力、截面给出极端纤维距离",
            "run_strength_check"),
    Command("symmetry", "对称性", "symmetry",
            "检测结构与荷载的对称性；并用对称位置的位移互为镜像自校核结果",
            "run_symmetry_check"),
    Command("bandwidth", "编号与存储", "bandwidth",
            "节点编号的带宽、重编号后的改善，以及满阵/等带宽/一维变带宽/稀疏"
            "四种总刚存储量对比",
            "run_numbering_check"),

    # --- 视图 ---
    # 视角用数字键：**照抄 Abaqus / SolidWorks 的约定，不自己发明**。
    # 用户的肌肉记忆是跨软件的，自创一套只会让人按错
    Command("front", "前视", "front", "沿 -Y 方向观察，XZ 平面", "view_front", "1"),
    Command("side", "侧视", "side", "沿 +X 方向观察，YZ 平面", "view_side", "2"),
    Command("top", "顶视", "top", "自上而下观察，XY 平面", "view_top", "3"),
    Command("iso", "等轴测", "iso",
            "等轴测视角，三个方向同时可见，用于总览空间结构",
            "view_iso", "4"),
    Command("fit", "适应窗口", "fit", "缩放至模型充满视口", "view_fit", "F"),
    Command("camera", "截图", "camera", "把当前视口连同窗口一起存成图片",
            "export_shot"),

    # --- 视口背景与网格地面 ---
    Command("bg_settings", "视口背景", "bg",
            "打开背景设置弹窗：16 种预设（含 Abaqus 渐变）、四种渐变方向、自定义颜色、网格地面",
            "open_bg_settings"),
    Command("grid_floor", "网格地面", "grid_floor",
            "在 z=0 平面显示/隐藏参考网格地面", "toggle_grid_floor", "", True),
)


BY_NAME: dict[str, Command] = {c.name: c for c in COMMANDS}
