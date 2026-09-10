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
    # 参数化建模使用直接操作的对话框。
    Command("sketch", "草图建模", "frame",
            "2D 画一榀框架草图，然后沿进深拉伸成 3D 空间刚架（Abaqus 式）",
            "open_sketch", "Ctrl+K"),
    Command("sketch_ai", "草图识别", "frame",
            "上传手绘草图，核对并编辑 AI 识别覆盖层，再标定尺度后加载",
            "open_sketch_ai", "Ctrl+I"),
    Command("frame", "规则框架", "frame",
            "弹出参数对话框，直接生成规则空间刚架（不经过 AI）",
            "open_parametric", "Ctrl+F"),
    Command("tables", "模型表格", "table",
            "七张表直接编辑节点、杆件、约束与荷载，支持 CSV 导入导出",
            "open_tables", "Ctrl+T"),
    Command("json", "模型 JSON", "report",
            "直接编辑模型的原始数据，写回前会做结构校验", "open_json"),
    Command("portal", "门式刚架", "portal",
            "弹出参数对话框，直接生成坡屋面门式刚架（不经过 AI）",
            "open_parametric"),
    Command("section", "截面管理器", "section", "弹出对话框编辑材料与截面定义", "edit_materials_sections"),
    Command("material", "创建材料", "section",
            "Abaqus 式：定义弹性模量、泊松比、密度", "create_material"),
    Command("beam_section", "创建截面", "section",
            "Abaqus 式：选择梁截面类型（I/管/矩形等），输入尺寸，自动计算 A/I/J",
            "create_section"),
    Command("assign_section", "属性指派", "section",
            "选中杆件后一次指派材料与截面（Abaqus Property 模块核心操作）", "assign_section"),
    Command("hinge", "创建铰接", "support",
            "选中杆件后在 i端/j端/两端创建转动释放", "create_hinge"),
    Command("brace", "加支撑", "member",
            "在选中杆件所在节间加 X 型支撑。钢框架最有效的抗侧手段", "add_brace"),

    # --- 边界条件 ---
    Command("load", "载荷管理器", "load", "打开边界条件面板，管理工况与荷载", "show_bc_panel"),
    Command("combo", "荷载组合", "combo", "弹出对话框编辑荷载组合（按分项系数组合）", "edit_load_combos"),
    Command("gravity", "重力", "gravity", "按密度和截面面积自动计算自重荷载", "apply_gravity"),
    Command("support", "支座管理器", "support", "打开边界条件面板，设置支座约束", "show_bc_panel"),
    Command("create_bc", "创建边界条件", "support",
            "Abaqus 式：选中节点，勾选 U1/U2/U3/UR1/UR2/UR3 自由度", "create_bc"),
    Command("create_load", "创建载荷", "load",
            "Abaqus 式：集中力/线载荷/重力，选中对象后输入分量", "create_load"),

    # --- 分析 ---
    Command("pick_node", "选择节点", "pick_node",
            "拾取过滤器：只选中节点。先定选什么类型再点，"
            "否则三维视图里分不清点到的是节点还是杆件",
            "set_pick_node", "N", True),
    Command("pick_member", "选择杆件", "pick_member",
            "拾取过滤器：只选中杆件", "set_pick_member", "M", True),
    Command("model_node", "建节点", "pick_node",
            "人工建模：在视口中点击任意位置创建节点",
            "set_model_node", "Ctrl+Shift+N", True),
    Command("model_member", "建杆件", "pick_member",
            "人工建模：先点击一个节点，再点击另一个节点，在两者之间创建杆件",
            "set_model_member", "Ctrl+M", True),
    Command("delete", "删除选中", "remove",
            "删除当前选中的节点或杆件（Delete 键）", "delete_selected", "Delete"),
    Command("labels", "编号标注", "labels",
            "在视口中显示节点与杆件编号。定位报错和结果时必需",
            "toggle_labels", "L", True),

    Command("analysis_step", "分析步", "timeline",
            "选择线性静力、P-Delta 或双线性轴向材料非线性及增量参数", "edit_analysis_step"),
    Command("solve", "求解", "solve", "按当前分析步求解，并执行收敛与平衡检查",
            "solve", "F5"),
    Command("modal", "模态", "modal", "自振频率与振型（需要材料密度）",
            "run_modal", "Ctrl+4", True),
    Command("buckling", "屈曲", "buckling", "线弹性屈曲因子与失稳模态",
            "run_buckling"),
    Command("diagnose", "模型检查", "support",
            "AI 辅助：检查模型完整性（支座/荷载/孤立节点/重复杆件），给出修正建议", "diagnose_model"),
    Command("analysis_mesh", "分析网格", "member",
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
    Command("diagram", "内力分量", "diagram",
            "选择梁内力云图显示哪个局部分量", "pick_component"),
    Command("clear_results", "清除结果", "remove",
            "清除当前计算结果，保留模型、材料、约束与荷载", "clear_results"),
    Command("curve", "内力图", "diagram",
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
