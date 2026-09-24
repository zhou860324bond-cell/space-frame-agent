"""界面中英文切换。

**为什么不用 Qt Linguist 那一套。** `.ts`/`.qm` 是给"发布多语言版本"准备的：
要跑 pylupdate 抽词、人工翻译、lrelease 编译，再在运行时装 QTranslator。
这个项目的界面文字全在 `commands.py` 一张表和几处面板标题里，
上那套工具链的维护成本远大于它省下的事。

这里的做法是**一张对照表 + 一次重刷**：切语言时把界面上认得出来的文字
换掉。代价说清楚：

* 表里没有的词保持中文，不会变成空白或乱码——**宁可露出一半中文，
  也不能让按钮变成没有字**；
* 对话框正文、报错信息仍是中文。翻译它们是另一件事，不在这次范围内，
  界面上（命令提示里）也照实写了这一点。

`tests/test_i18n.py` 守两条：表里不许有空翻译，`commands.py` 里每条命令的
标签都必须在表里有对应——加了新按钮却忘了加翻译，测试会红。
"""

from __future__ import annotations

LANGS = ("zh", "en")

_LANG = "zh"

#: 中文 → 英文。键必须与界面上出现的中文**一字不差**。
ZH_EN: dict[str, str] = {
    # --- 功能区页签 ---
    "项目": "File", "建模": "Model", "属性": "Property", "载荷": "Load",
    "分析": "Step", "结果": "Results", "视图": "View", "主菜单": "Menu",
    # --- 功能区分组 ---
    "文件": "File", "编辑": "Edit", "面板": "Panels",
    "几何": "Geometry", "数据": "Data", "选择": "Select",
    "材料与截面": "Material & Section", "指派": "Assign",
    "边界条件": "Boundary", "荷载": "Loads", "工况": "Load Cases",
    "求解": "Solve", "特征值": "Eigen", "检查": "Check", "子模型": "Submodel",
    "显示": "Display", "评估": "Evaluate", "校核": "Checks", "输出": "Output",
    "视角": "Viewpoint", "视口": "Viewport", "工具": "Tools",
    # --- 常驻快捷栏 ---
    "工作平面": "Work plane", "捕捉": "Snap", "关闭": "Off",
    "坐标建点": "Point by XYZ", "放大": "Scale", "自动": "Auto",
    "≡  菜单": "≡  Menu",
    # --- 动作标签 ---
    "新建": "New", "打开": "Open", "保存": "Save", "单位制": "Units",
    "Agent 对话": "Agent Chat", "撤销": "Undo", "重做": "Redo",
    "建模过程": "History", "导出学习轨迹": "Trace",
    "草图建模": "Sketch", "草图识别": "Sketch AI", "规则框架": "Frame",
    "模型表格": "Tables", "模型 JSON": "JSON",
    "门式刚架": "Portal", "截面管理器": "Sections",
    "创建材料": "Material", "创建截面": "Section",
    "属性指派": "Assign", "创建铰接": "Release",
    "加支撑": "Bracing", "载荷管理器": "Loads",
    "荷载组合": "Combos", "重力": "Gravity",
    "支座管理器": "Supports", "创建边界条件": "Create BC",
    "创建载荷": "Create Load", "选择节点": "Node",
    "选择杆件": "Member", "建节点": "+Node", "建杆件": "+Member",
    "删除选中": "Delete", "编号标注": "Labels", "荷载数值": "Load Values",
    "中 / EN": "EN / 中", "分析步": "Step", "模态": "Modal",
    "求解设置": "Procedure", "幅值曲线": "Amplitude",
    "规范组合": "Code Combos", "活载布置": "Live Pattern",
    "面荷载": "Area Load", "边界条件管理器": "BC Manager",
    "规范与导荷": "Code & Tributary",
    "屈曲": "Buckling", "模型检查": "Check", "分析网格": "Mesh",
    "节点实体": "Joint", "模型": "Model", "变形": "Deformed",
    "梁内力云图": "Contour", "内力分量": "Component",
    "清除结果": "Clear", "内力图": "Diagram", "单杆内力图": "Member Chart",
    "包络": "Envelope", "最大挠度": "Deflection", "报告": "Report",
    "强度验算": "Strength", "对称性": "Symmetry",
    "编号与存储": "Numbering", "前视": "Front", "侧视": "Side",
    "顶视": "Top", "等轴测": "Iso", "适应窗口": "Fit",
    "截图": "Shot", "视口背景": "Background", "网格地面": "Grid",
    # --- 流程条 ---
    "1  建模": "1  Model", "2  定义": "2  Define",
    "3  校验": "3  Validate", "4  求解": "4  Solve", "5  结果": "5  Results",
    # --- 结果面板 ---
    "云图量程": "Range", "分级": "Bands", "色系": "Palette", "立体": "Shaded",
    "符号": "Sign", "叠加变形": "Overlay Deformed", "标注极值": "Mark Extrema",
    "定位当前分量极值": "Locate Extreme", "截面正应力": "Section Stress",
    "尚无分析结果": "No results yet",
    "95% 裁剪": "Clip 95%", "满量程": "Full range",
    "全部": "All", "仅正值": "Positive only", "仅负值": "Negative only",
    "彩虹（Abaqus 式）": "Rainbow (Abaqus)",
    "蓝—灰—红（发散）": "Blue-Grey-Red (diverging)",
    "单蓝（顺序）": "Blue (sequential)",
    # --- 流程条状态与提示 ---
    "尚未创建几何模型": "No geometry yet",
    "开始建模": "Start modelling", "修正模型": "Fix model",
    "开始求解": "Run analysis", "查看结果": "View results",
    "模型已通过确定性校验，可以提交分析":
        "Model passed deterministic checks; ready to submit",
    "求解完成，可查看变形与内力，并做强度、对称性校核，最后出报告":
        "Solved — view deformation and forces, run strength and symmetry "
        "checks, then export the report",
    "模型仍有校验问题；请查看下方问题并修正":
        "The model still has validation issues; see the list below",
    "当前 0 个节点；至少创建 2 个节点": "0 nodes so far; create at least 2",
    "节点已创建；下一步连接至少 1 根杆件":
        "Nodes created; next connect at least 1 member",
    "杆系几何已建立；下一步定义材料":
        "Geometry ready; next define a material",
    "材料已定义；下一步定义并指派截面":
        "Material defined; next define and assign a section",
    "属性已就绪；下一步施加支座约束":
        "Properties ready; next add supports",
    "约束已设置；下一步创建荷载工况":
        "Supports set; next create a load case",
    # --- 功能区分组（ribbon.py 里实际用到的组名）---
    "创建": "Create", "查看": "View", "查询": "Query", "历史": "History",
    "材料": "Material", "截面": "Section", "单位": "Units",
    "局部实体": "Local Solid", "参考显示": "References",
    "视角管理": "Cameras", "语言": "Language", "Agent 学习": "Agent Learning",
    # --- 停靠面板标题 ---
    "问题": "Issues", "对话": "Chat", "内力图表": "Diagrams",
    "模型树": "Model Tree", "AI 助手": "AI Assistant",
    "分析流程": "Workflow",
    # --- 分级下拉。逐条列出来而不是套模板：套模板就得在界面上拼字符串，
    #     而"N 级"和"N bands"的词序未必总是一致。
    "6 级": "6 bands", "8 级": "8 bands", "10 级": "10 bands",
    "12 级": "12 bands", "16 级": "16 bands", "20 级": "20 bands",
    "24 级": "24 bands",
}

#: 反向表，切回中文用。
EN_ZH: dict[str, str] = {v: k for k, v in ZH_EN.items()}


def language() -> str:
    return _LANG


def set_language(lang: str) -> bool:
    """切到 ``zh`` 或 ``en``。返回是否真的变了。"""
    global _LANG
    if lang not in LANGS or lang == _LANG:
        return False
    _LANG = lang
    return True


def tr(text: str) -> str:
    """把一条中文界面文字翻成当前语言。表里没有就原样返回。"""
    if _LANG == "zh":
        return text
    return ZH_EN.get(text, text)


def untr(text: str) -> str:
    """把界面上现有的文字还原成中文键，供重刷时反查。"""
    return EN_ZH.get(text, text)


def _keyed(widget, index, current: str) -> str:
    """取某个控件（或它第 index 项）的中文原文。

    第一次见到就把中文存进动态属性。之所以不每次反查 EN_ZH：反查对
    "表里没有、保持中文"的那些词是无解的，存一次原文就永远不会走丢。
    """
    prop = f"_i18n_zh_{index}"
    stored = widget.property(prop)
    if stored is None:
        stored = current
        widget.setProperty(prop, stored)
    return stored


def retranslate(roots, actions=()) -> None:
    """把给定的几块界面刷成当前语言。

    **刻意只收窄到这几块**（功能区、流程条、结果面板、停靠面板标题），
    不是整窗遍历。整窗遍历会把所有对话框、向导、提示正文一起卷进来，
    而那些没翻——结果是一屏中英混排，比全中文更难用。
    宁可"翻的这几块全翻到位"，也不要"到处翻一半"。

    表格里的数据、模型里的名字、报错正文一概不动：那些是内容不是界面，
    翻了反而让人对不上自己输入的东西。
    """
    from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget,
                                   QGroupBox, QLabel, QPushButton, QTabWidget,
                                   QToolButton)

    for act in actions:
        act.setText(tr(_keyed(act, "text", act.text())))
        tip = tr(_keyed(act, "tip", act.toolTip()))
        act.setToolTip(tip)
        act.setStatusTip(tip)

    for root in roots:
        if root is None:
            continue
        if isinstance(root, QDockWidget):
            root.setWindowTitle(
                tr(_keyed(root, "title", root.windowTitle())))
            continue
        for widget in root.findChildren(QTabWidget) + _self(root, QTabWidget):
            for i in range(widget.count()):
                widget.setTabText(
                    i, tr(_keyed(widget, f"tab{i}", widget.tabText(i))))
        for widget in root.findChildren(QComboBox):
            if widget.property("i18nSelfManaged"):
                continue
            for i in range(widget.count()):
                widget.setItemText(
                    i, tr(_keyed(widget, f"item{i}", widget.itemText(i))))
        for cls, get, set_ in ((QLabel, "text", "setText"),
                               (QCheckBox, "text", "setText"),
                               (QPushButton, "text", "setText"),
                               (QGroupBox, "title", "setTitle"),
                               (QToolButton, "text", "setText")):
            for widget in root.findChildren(cls):
                # 自己会翻的控件在这里让开。流程条的步骤按钮文字是
                # "步骤名 + 算出来的后缀（✓ / !3）"，整串缓存下来会把上一次
                # 的语言和上一次的状态一起冻住——它在 _set_state 里现翻。
                if widget.property("i18nSelfManaged"):
                    continue
                # 绑了 QAction 的工具按钮文字来自动作本身，上面已经翻过；
                # 在这里再 setText 会把按钮和动作解绑，动作再变文字就同步不了。
                if isinstance(widget, QToolButton) and widget.defaultAction():
                    continue
                current = getattr(widget, get)()
                if not current:
                    continue
                getattr(widget, set_)(tr(_keyed(widget, get, current)))


def _self(widget, cls):
    """findChildren 不含自己，而功能区本身就是个 QTabWidget。"""
    return [widget] if isinstance(widget, cls) else []
