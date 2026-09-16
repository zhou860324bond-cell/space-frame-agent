# 空间刚架智能计算 Agent

自然语言 → 结构模型 → 求解 → 自校验 → 出图，全流程可验证。

当前回归基线为 **1656 项通过、1 项按环境跳过**，Agent 注册 **49 个确定性工具**。
核心原则只有一条：
**大模型只产结构，不产数值**——报告里每个数字都能溯源到某次工具调用。

> **文档与代码同步是有测试守着的。** 改了菜单名、加减了工具、加了测试之后，
> `docs/课程报告.md`、`docs/大学生教程.md`、`docs/使用手册.md`
> 和这句回归基线都要跟着改，
> 改完再跑一次 `python tools/build_docx.py` 重新导出 `docs/word/` 下的 Word 版；
> 少做哪一步，`tests/test_docs_current.py` 都会红。
> 这条闸就是为了不让文档靠自觉维护。

## 三本文档，按你的底子挑

| 文档 | 写给谁 | 讲什么 |
|---|---|---|
| [`docs/使用手册.md`](docs/使用手册.md) | 完全零基础 | 点哪里、会看到什么、下一步做什么 |
| [`docs/大学生教程.md`](docs/大学生教程.md) | 学过结构力学 | 功能全景、每个功能在算什么、能力边界、判读要点 |
| [`docs/课程报告.md`](docs/课程报告.md) | 评阅与答辩 | 设计取舍、四层验证、数据出处 |

三本都有 Word 版在 `docs/word/`。

## 快速开始

双击 `run.bat`：建 `.venv`、装依赖、跑全部测试、跑多工况示例、跑一遍 Agent 离线演示。

装依赖有两种口径：`requirements.txt` 给的是**下限**（"至少要这么新"），
`requirements-lock.txt` 记的是**全量回归全绿的那一套确切版本**。
"我这边跑不过"时先按后者装一遍，能立刻分清是代码的问题还是版本的问题。
按用途分装见 `pyproject.toml` 的 extras：`desktop` / `web` / `solid` / `llm`。

回归由 `.github/workflows/tests.yml` 在干净机器上自动跑——**"我这边跑过了"
不是证据**。

手动跑（cmd。**注意 `set` 后面不加引号，PowerShell 语法在 cmd 里不认**）：

```
cd /d C:\Users\你的用户名\Desktop\agent开发
.venv\Scripts\activate.bat
set PYTHONPATH=src;evalset;abaqus_bench
pytest
python examples\run_json.py examples\portal_frame_cases.json
python examples\agent_demo.py
```

## 五个入口脚本

| 脚本 | 作用 | 要不要联网 |
|---|---|---|
| `build_exe.bat` | 打包成独立 Windows 应用（`dist\FrameLab\`，约 700 MB） | 不用 |
| `package_zip.bat` | 把打包体压成 `FrameLab.zip`；**包里有密钥就拒绝压** | 不用 |
| `run.bat` | 建环境、跑测试与离线演示 | 不用 |
| `run_desktop.bat` | **桌面端**（PySide6 + PyVista，CAE 式界面） | 首次装依赖要 |
| `run_gui.bat` | **网页端**（Streamlit，功能与桌面端等价） | 不用 |
| `run_live.bat` | 拿真实大模型跑 5 道探针题 | 要，密钥放 `deepseek.key` |
| `run_eval.bat` | 跑 14 题评测集，出成绩单 | 要 |

`deepseek.key` 里只写一行 API 密钥，已被 `.gitignore` 忽略。
**别把密钥敲进命令行**——每贴一次终端日志就泄露一次。

## 目录

```
src/frame3d.py        求解内核：3D 梁单元、坐标变换、杆端释放、多工况求解、
                      内力回算、静力平衡校核、奇异诊断
src/solid3d.py        自研 C3D10 实体内核：形函数、积分、稀疏组装与应力恢复
src/native_joint.py   Gmsh圆管节点网格、六分量切割面载荷与 native-solid 结果输出
src/model_io.py       版本化 Domain IR：JSON Schema、v0→v1 迁移与校验
src/model_compiler.py 物理构件→分析单元编译、集中荷载自动剖分与双向映射
src/result_db.py      统一结果层：Step / Frame / FieldOutput 与结果元数据
src/generator.py      参数化生成器：参数 -> 节点与杆件拓扑
src/agent.py          Agent 层：45 个工具、会话状态、对话循环、模型后端
src/plot3d.py         三维绘图：变形图、轴力图

tests/                回归基线见文首；主体离线且不需要密钥
evalset/              14 题评测集 + 评分器 + 成绩单生成
abaqus_bench/         与 Abaqus 对标：5 算例 × B33/B31
examples/             示例模型、端到端脚本、离线演示、探针题
```

## 设计原则

**大模型只产结构，不产数值。** 它可以决定"三跨两层、柱底固接"，位移内力反力
必须来自工具返回。系统提示里明令禁止它自己算。

**拓扑由代码展开。** 多层多跨让模型逐个列节点坐标必然出错，所以只给
`generate_frame(spans, storeys, bays, ...)` 这样的参数入口。不规则结构才走
`set_model` 逃生舱。

**算前守门。** `set_model` 与 `solve_model` 都先跑两级校验，不通过就不算，
把中文错误清单回给模型让它自己改——这就是自修复闭环的一轮。

**图给人看，数给模型看。** `plot_results` 只返回文件路径和数值摘要，不返回图像。
让多模态模型看云图再下结论，等于把可验证的数值换成像素猜测。

## Agent 工具

45 个工具的返回合同由 `tests/test_tool_contracts.py` 统一约束。下面只列正式演示
主链路；其余工具覆盖编辑、集合、扫参、包络、模态、屈曲和 Abaqus 对比。

每次模型调用都会收到由当前 `Session` 确定性推导的工作流状态：`empty`、
`draft`、`ready` 或 `solved`。状态中的校验错误和推荐工具会在每批工具执行后立即
刷新，避免 Agent 靠聊天记忆猜测“现在做到哪一步”。

| 工具 | 作用 |
| --- | --- |
| `define_materials_and_sections` | 定义材料与截面，建模前先调一次 |
| `generate_frame` | 按参数展开规则刚架，回传梁编号 |
| `set_model` | 提交完整 JSON，即时校验（不规则结构的逃生舱） |
| `set_load_cases` | 一次定义全部工况与组合 |
| `validate_model` | 两级校验，返回中文错误清单 |
| `preview_change` | 在模型副本上预演写操作，返回实体级差异与前后哈希 |
| `apply_preview` | 只应用经后续用户消息确认且尚未失效的预演 |
| `preview_analysis_mesh` | 只读预览自动剖分、分析节点/单元与物理构件映射 |
| `analyze_joint_solid` | 选定圆管节点做 C3D10 局部实体分析；默认自研求解，Abaqus作为可选对标后端 |
| `solve_model` | 先校验再求解，失败时直接附上奇异诊断 |
| `query_results` | 最大位移、支座反力、杆端力（带坐标与符号约定） |
| `plot_results` | 变形图 / 轴力图 |
| `diagnose_supports` | 定位缺失约束的节点与方向 |

## 当前能力

| 类别 | 支持内容 |
| --- | --- |
| 单元 | 空间 Timoshenko / Euler-Bernoulli 梁（6 自由度/节点，12×12） |
| 材料 | 各向同性线弹性；双线性轴向塑性增量求解 |
| 约束 | 六个方向逐个开关；柱底固接或铰接 |
| 释放/刚域 | 任意杆端局部自由度静力凝聚；全局向量刚域偏移 |
| 荷载 | 节点力/力矩、全局坐标均布杆间荷载 |
| 编译 | 物理构件到分析单元双向映射；杆间集中荷载、显式内节点自动剖分 |
| 结果 | 线性静力写入统一 Result DB，查询、图和报告读取同一结果合同 |
| 工况 | 线性多工况/组合；二阶弹性 P-Δ；非线性增量与收敛记录 |
| 生成 | 规则多层多跨多开间空间刚架 |
| 自校验 | 模型合法性、静力平衡、奇异模态定位、位移量级 |
| 出图 | 梁中心线变形、振型/失稳模态、内力曲线与云图 |

**仍有限制**：材料非线性当前只含轴向双线性塑性，弯曲塑性需要纤维截面；
P-Δ 是小应变二阶弹性，不是通用有限转动 NLGEOM；温度与接触尚未实现。
自动剖分当前由杆间集中荷载或已有显式内节点触发；任意相交点和属性突变不会自动切分。
带刚域偏移的杆间集中荷载会明确拒绝，避免静默丢失偏心力矩。完整边界见
[`docs/BETA_0.1_CAPABILITY_MATRIX.md`](docs/BETA_0.1_CAPABILITY_MATRIX.md)。

## 验证的四层

**第一层 梁：解析解。** 已冻结 **28 个**独立解析金标准算例：线性静力 12 项
（悬臂挠度 `PL³/3EI`、扭转角 `TL/GJ`、简支跨中 `5wL⁴/384EI`、Timoshenko
剪切项、集中/三角荷载、杆端释放）、自振特性 5 项、屈曲、二阶弹性与轴向塑性。
运行 `run_gold.bat`；真值、公式和阈值见
[`docs/BETA_0.1_GOLD_CASES.md`](docs/BETA_0.1_GOLD_CASES.md)。

真值一律来自经典梁理论、静力平衡和闭合解特征值，**不来自另一份本项目实现**
——拿自己的旧结果当基准，只能测出"没变过"，测不出"算得对"。

**第二层 实体单元：对照教科书解。** 单元本身早就验过（形函数、常应变分片
检验、六个刚体模态、制造解、线性应力场精确外推），但**装配起来之后**
——网格生成 → 荷载等效 → 求解 → 应力恢复 → 峰值提取这一串——原来一次都没对过
外部参照。补的算例是**中心带圆孔的受拉板**，Kt 有闭式解且孔边不是奇异点：

| 孔边网格 | 自由度 | Kt（节点平均） | 相对 Heywood 式 2.512 |
|---:|---:|---:|---:|
| 2.5 mm | 29,916 | 2.614 | +4.1% |
| 1.8 mm | 42,972 | 2.609 | +3.9% |
| 1.2 mm | 88,350 | 2.620 | +4.3% |

数值稳、峰值落在孔边 90°、偏差有**自己量出来的**解释（t/d=0.5 不算薄板，
沿板厚 Kt 从自由表面 2.52 升到中面 2.62，而 Heywood 是薄板平面应力的结论）。
这个算例和相贯线那个配成一对：后者是应力奇异点，判据报 `diverging`；
前者峰值有限，报 `converging`。**两个方向都试过，收敛判据才算验过。**
详见 [`docs/带孔板验证.md`](docs/带孔板验证.md)，工具
`tools/verify_plate_with_hole.py`。

**第三层 与 Abaqus 对标。** 见 `abaqus_bench/README.md`。主对标用 **B33**
（截面未给 Ay/Az 时与本程序的 Euler-Bernoulli 分支同理论）；给 Ay/Az 的
Timoshenko 模型应改用 B31 对标。

> **复现这一层前先读两条环境约束**，见
> [`docs/SOLID_JOINT_SCOPE.md`](docs/SOLID_JOINT_SCOPE.md) 末两节。两条都
> **不报错、只静默失败**，实测踩过：
>
> - **路径不能含非 ASCII 字符。** 同一个输入文件，ASCII 路径下跑通，含中文的
>   路径下 `pre.exe` 以系统错误码 529697949 中止，`.dat` 里一条 `***ERROR`
>   都没有。本仓库默认位置就带中文，对标作业因此曾长期静默失效。现已改为在
>   ASCII 临时目录里跑、产物复制回来。
> - **AMD 上必须带 `MKL_DEBUG_CPU_TYPE=5`。** Abaqus 6.14 捆的 Intel MKL 11.x
>   在 Zen 架构上会挑到走不通的分派路径，`standard.exe` 以 1073741795 中止，
>   同样不留任何错误信息。代码里由 `abaqus_backend.solver_environment()` 统一
>   处理，只在 AMD 上生效。

**第四层 Agent 评测集。** 14 题六类，判定一律读会话状态不读回复文本。
评分器本身有 14 个离线测试守着——一张没人查过的成绩单说明不了任何事。

## 可复现运行（实验胶囊）

每次求解后可以存一个**胶囊**（JSON 文件），记录完整输入模型、结果摘要、
环境信息和输入哈希。历史可查、可对比、可回归——解决了"上次跑的结果
和这次不一样，到底是模型改了还是环境变了"的问题。

```bash
# 求解并存档
python -m capsule save examples/portal_frame.json --label "门式刚架试算"

# 列出历史
python -m capsule list

# 查看详情
python -m capsule show 20260904_202228

# 对比两次运行
python -m capsule diff 20260904_202228 20260904_202245
```

代码在 `src/capsule.py`，MCP 工具 `save_capsule` / `list_capsules` /
`get_capsule` / `diff_capsules` 也能操作。

## 静默失败检测

工程仿真最危险的不是报错，而是**作业正常结束、数值看起来合理、但模型
本身有问题**。求解后自动扫描 8 类静默失败：

| 检测项 | 严重度 | 抓什么 |
|---|---|---|
| 支座约束不足 | critical | 约束自由度 < 6，或某个方向完全没约束 |
| 位移过大 | critical | 最大位移 > 结构尺寸 10%（机构运动或单位错） |
| 有载荷但反力为零 | critical | 约束完全失效，载荷没传到基础 |
| 有载荷但内力为零 | critical | 结构是机构，载荷没传递到杆件 |
| 刚度矩阵接近奇异 | critical | 条件数 > 1e12，存在近零能位移模式 |
| 反力与外载荷不平衡 | critical | 整体合力残差 > 1% |
| 载荷量级异常 | warning | 估算应力 > 0.1E 或 < 1e-3 Pa（单位错） |
| 截面主轴方向风险 | warning | Iy > Iz（截面特性写反或 ref_vector 错） |

```python
from silent_failures import detect_silent_failures, format_findings
findings = detect_silent_failures(model, sol)
print(format_findings(findings))
```

详细原理、真实案例和修复建议见 `docs/SILENT_FAILURES.md`。
MCP 工具 `detect_silent_failures_tool` 也能调用。

## MCP Server

把求解器封装成 **MCP（Model Context Protocol）Server**，让 Codex / Cursor /
Claude Desktop 等 MCP 客户端直接调用——不需要 Abaqus 许可证，纯 Python 求解。

```bash
# stdio（默认，MCP 客户端直接 spawn）
python -m mcp_server

# SSE（HTTP 流，适合远程访问）
python -m mcp_server --transport sse --host 0.0.0.0 --port 8765
```

MCP 客户端配置（Claude Desktop / Cursor）：

```json
{
  "mcpServers": {
    "space-frame": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "C:\\path\\to\\agent开发"
    }
  }
}
```

10 个工具：`solve_frame`（一站式求解+检测+存档）、`validate_frame`、
`query_result`、`diagnose_supports`、`modal_analysis`、`buckling_analysis`、
`detect_silent_failures_tool`、`list_capsules_tool`、`get_capsule_tool`、
`diff_capsules_tool`。代码在 `src/mcp_server.py`。

依赖：`pip install mcp`（已加入环境）。

## 多模型路由 + API 失败自动降级

单一 LLM 提供商不可用时（429 限流、500 宕机、密钥过期），整个 Agent 就瘫了。
`ModelRouter` 按优先级依次尝试多个模型，主模型失败自动降级到备用模型，
保证"只要有一个模型能用，Agent 就能跑"。对调用方透明——实现了和
`DeepSeekProvider` 一样的 `complete(messages, tools)` 接口，可直接传给 `run_turn()`。

```python
from model_router import ModelRouter

# 从环境变量 MODEL_ROUTER_CONFIG 加载多模型配置
router = ModelRouter.from_env()

# 或手动配置
from model_router import ModelConfig
router = ModelRouter([
    ModelConfig(name="deepseek-v4-flash", base_url="https://api.deepseek.com",
                api_key_env="DEEPSEEK_API_KEY", priority=1),
    ModelConfig(name="gpt-4o-mini", base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY", priority=2),
])

result = run_turn("建个三跨两层框架", router)
print(router.last_call_summary())  # 用了哪个模型、是否降级、耗时
```

每次调用记录到 `call_history`（模型名、成功/失败、耗时、错误原因、是否降级），
可从界面展示给用户。代码在 `src/model_router.py`，测试在 `tests/test_model_router.py`。

## 多参数多目标截面优化

从"单参数截面扫描"升级为"多参数多目标优化"：同时变化多个截面参数
（工字形的高度、翼缘宽度、腹板厚度、翼缘厚度），同时优化多个目标
（最小重量、最小位移、最小材料成本），并满足约束（应力、位移、最小面积）。

支持两种算法：
- **网格搜索**：在每个参数范围内均匀取点，穷举所有组合。结果确定、可复现，
  适合 2-3 个参数。
- **随机搜索**：在参数范围内随机采样指定次数。参数多时比网格搜索高效，
  适合 4+ 个参数。

多目标时返回 **Pareto 前沿**（没有任何一个目标能在不牺牲其他目标的情况下再改进）。

```python
from section_optimizer import SectionOptimizer

opt = SectionOptimizer(
    model=my_model,
    section_name="BEAM",
    section_type="工字形 / H 型钢",
    variables={
        "height": (0.3, 0.8, 5),        # (min, max, num_points)
        "flange_width": (0.15, 0.3, 4),
    },
    objectives=["weight", "max_displacement"],
    constraints={"max_displacement_limit": 0.02},  # 20mm
)
result = opt.grid_search()
print(result.summary())
print("Pareto 前沿点数:", len(result.pareto))
```

代码在 `src/section_optimizer.py`，测试在 `tests/test_section_optimizer.py`。

## 手绘草图 → 模型（多模态输入）

结构工程师的工作流通常是"先在纸上画草图，再输入软件"。多模态输入让用户上传
手绘草图，LLM 识别有图片证据的节点、杆件、支座、荷载和尺寸，先生成
`RecognitionDraft`，经人工核对后才进入模型编辑链路。

遵循项目核心原则：**LLM 只产结构，不产数值**——识别"哪有节点、哪有杆、
哪是支座、荷载在哪"，位移/内力/反力仍由确定性求解器计算。

支持 OpenAI、Anthropic、DeepSeek 的视觉模型。识别草稿经过拓扑、引用和有限数校验，
不通过时把错误回给 LLM 修正（自修复闭环），最多重试 N 次。系统不会默认为图片补材料、
截面、支座、荷载或米制尺寸；未知尺度必须用一根识别杆件的真实长度标定。

首次使用前安装可选的视觉模型 SDK：

```bash
.venv\Scripts\python.exe -m pip install -r requirements-multimodal.txt
```

桌面端从「建模 → 草图识别」进入：选择图片、提供商与模型；密钥可在面板中临时填写，
也可分别设置 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY`。识别后原图会
叠加节点、杆件和支座/荷载框；低置信度实体用警示色。用户可在原图拖动节点、从已有
节点补画杆件或删除误识别杆件，再逐项处理识别疑问、标定未知尺度并确认加载。删除仍被
荷载引用的杆件会被拒绝。加载的是允许缺材料和边界条件的几何草稿，仍需在正式求解前
完成模型检查。默认收起的“支座与节点荷载”区域支持支座新增/改型/删除，以及新建工况、
按名称新增/更新/删除六分量节点荷载；相应疑问会随明确操作自动标为已处理。提供商、模型、
密钥和原始 JSON 也默认收起，避免遮挡主流程。杆件分布荷载当前只显示并保护引用，确认
加载后使用正式边界条件面板编辑。

```python
from sketch_parser import SketchParser

parser = SketchParser.from_env("openai")  # 从 OPENAI_API_KEY 读密钥
result = parser.parse_with_retry("sketch.jpg", max_retries=3)
if result.success:
    draft = result.draft
    if draft.scale_status == "unknown":
        draft.calibrate(member_id=1, actual_length_m=6.0)
    frame_draft = draft.to_frame_draft()  # 再交给 Session.apply_draft 人工确认加载
else:
    print("识别失败:", result.errors)
```

确认层在 `src/recognition_draft.py`，解析器在 `src/sketch_parser.py`；测试见
`tests/test_recognition_draft.py`、`tests/test_sketch_parser.py` 和 `tests/test_sketch_panel.py`。
下一阶段的状态机、尺度/交点消歧、原子提交和评测任务卡见
[`docs/MULTIMODAL_V2_WORKFLOW.md`](docs/MULTIMODAL_V2_WORKFLOW.md)。
MM2-00 已提供 `src/multimodal_contract.py`、固定的 `multimodal_eval/manifest.json`
及 8 张离线种子图；运行
`.venv\Scripts\python.exe -m pytest tests\test_multimodal_contract.py -q --basetemp=.pytest-mm2`
可验证哈希绑定和确定性匹配。

## 四个已经处理掉的坑

**局部坐标系参考向量。** 水平杆件局部 y 取全局 +Z，竖直杆件参考向量退化改用全局 +X。
柱子算错八成出在这里。

**内力回算加固端力。** `f = k·u - p`。漏掉的话支座弯矩对、跨中弯矩全错。

**轴力符号。** 以受拉为正的是 j 端分量 `f[6]`，不是 `f[0]`，两者符号相反。
这个错误纯看数字时藏了很久，画成轴力图当场露馅——柱子被画成了受拉。

**奇异不一定报错。** `splu` 遇到奇异刚度阵时，若荷载恰好不激发那个刚体方向，
会安静地返回一个看似合理、实则错误的位移场，连平衡校核都能通过。所以求解后
额外查主元比值。**这正是"算前守门必须独立于求解是否报错"的理由。**

## 约定

单位制 **N-m-Pa**，全程不做换算，由调用方保证一致。
节点 6 自由度，顺序 `(ux, uy, uz, rx, ry, rz)`；杆端力 12 分量按 i 端、j 端各 6 个排列。
轴力以受拉为正。


## 建模三层

原来只有两个生成器（正交网格、坡屋面门式），中间到"逐个列节点坐标"之间
什么都没有。**模板是加法，算子是乘法：**

**① 一榀** `generate_bent(profile, columns, ...)` —— 屋面写成一条折线，
柱从折线上落下来。一个生成器覆盖六类：

```
平屋面   profile = [(0, 7.5), (24, 7.5)]
双坡     profile = [(0, 7.5), (12, 8.7), (24, 7.5)]   ← 中间那点是屋脊
单坡     profile = [(0, 6.0), (24, 9.0)]
悬挑     折线两端伸到柱子外面
错层     base_levels 逐柱给柱脚标高
多层     levels 给楼面标高
```

**② 拉伸** `extrude_bents(bays=[4.5, 6, 6, 4.5])` —— 不等开间是常态
（山墙往往小一些），自动生成连系梁。单榀时补的面外约束会一并去掉：
拉伸后不再是平面刚架，留着会把结构在 Y 向钉死，横向刚度偏刚且不报错。

**③ 算子** —— 加支撑（X / 人字 / 单斜）、抽柱开洞、抬高一片区域、换截面。

正确性怎么保证的：**同一个门式刚架，用一榀生成器和用原门式生成器，
最大位移与最大挠度必须完全一致**（27.21292 mm / 27.07787 mm）。
自己跟自己比说明不了任何问题，两条独立路径对上才算数。

## 命名集合

编号规则由生成器决定，**谁也不知道 27 号在哪**。没有名字，
"给顶层所有梁加 5 kN/m" 这句话就没有落点。

```
define_set("顶层梁", orientation="horizontal", z_range=[7.0, 7.4])
remove_members(ids="底层柱")      # 之后直接写名字
```

集合是纯粹的命名层：求解器看不到它，工具在调用时就展开成编号了。
删杆件时集合会跟着修剪，被删空的整个去掉——**空集合被引用时会静默
什么都不做，比报错难查得多**。

## 桌面端界面

顶部是功能区，按分析阶段分页：**项目 / 建模 / 属性 / 载荷 / 分析 / 结果**。
页签顺序就是 Abaqus/CAE 式的流程顺序：创建几何后定义材料和截面，再选择对象施加
边界条件与载荷，最后分析和查看结果。每个页签切换时，状态栏都会提示下一步操作。

视角、选择过滤器、手动画杆、结果显示模式和撤销/重做属于跨阶段高频操作，只保留在
常驻快捷栏，不再在各 Ribbon 页重复。视图截图等低频入口保留在菜单；功能与快捷键没有
因界面瘦身而删除。

「文件 → 最近打开」保留最近使用的模型；「结果 → 清除结果」只丢弃计算结果，
不修改模型、材料、约束或载荷，适合重新计算前清理当前显示状态。

「分析 → 分析网格」只读显示求解器实际使用的节点和单元：物理构件保留为参考轮廓，
自动剖分节点单独标出，底部表格给出 Analysis Element 到 Physical Member 的映射。
打开或关闭预览都不会修改模型、求解结果或撤销历史。

常驻建模栏支持选择 XY/XZ/YZ 工作平面、输入平面偏移、设置网格捕捉间距，或直接用
“坐标建点”输入带当前单位的 x/y/z。输入已有节点坐标时只选中并复用，不会重复建点。

只摆我们真有的东西。不做"网格""接触""非线性"这些页签，摆了就是骗人。

图标全部用 QPainter 画出来，仓库里没有一个图片资源：矢量的、高 DPI 下不糊、
换配色不用重新导图。图案画的是这个软件真正在做的事——屈曲画压弯的柱，
包络画两条曲线夹出来的带，视角按钮把你正对着的那个面涂亮。

功能区只显示已实现的直接操作；尚未提供界面流程的功能不会以示例话术或
预填输入框的形式出现。需要扩展功能时，可直接在 AI 助手中描述实际工程参数。

## 桌面端的 Agent 对话面板

桌面端右侧是对话面板，和工具栏共用**同一个 Session**——对话建的模型，
工具栏和视口立刻能看到，不会各改各的。

面板把**工具调用摊开显示**：

```
你：单跨 24 米门式刚架，柱脚铰接，屋面恒载 8 kN/m
    ⚙ define_materials_and_sections
    ⚙ generate_portal_frame   spans=[24.0] eave_height=7.5
    ⚙ set_load_cases          cases=[3 项] combos=[3 项]
    ⚙ solve_model
Agent：已建好并求解，静力平衡校核通过。
```

这是刻意的。整个项目最容易被质疑的一句是"这个数是不是大模型编的"，
把调用链摊开，答案就是自明的：**模型只决定调什么，数从求解器出来。**

没有密钥也能用：面板会切到离线演示模式，按关键词认意图，
但**工具和求解都是真的**——只有"把中文映射成工具调用"那一步被替掉了。
认不出来的话它会直说认不出来，不会假装听懂。

求解和大模型调用都在后台线程里跑，所以窗口不会卡成"未响应"。

## 桌面端打不开怎么办

先跑自检，它会逐项告诉你哪一步挂了、下一步做什么：

```
run_desktop.bat
```

启动失败时它自己就会跑自检并停住等你看。也可以手动跑：

```
.venv\Scripts\python.exe -m desktop.app --doctor
```

几个已知的坑，都已经在启动脚本里堵上了：

* **控制台乱码 / `UnicodeEncodeError`**——Windows 控制台默认 GBK，
  代码里的 `✓` `⚠` 这类字符编不出来，会把一次**已经算完**的运行打死在
  打印环节。启动脚本统一 `chcp 65001` + `PYTHONUTF8=1`，
  入口脚本再调一次 `console.use_utf8()` 兜底。
* **Qt 绑定选错**——`pyvistaqt` 经 `qtpy` 选绑定，默认可能挑中 PyQt5，
  而窗口代码用的是 PySide6，两套 Qt 在一个进程里必然出事。
  启动脚本和 `desktop/app.py` 都把 `QT_API=pyside6` 钉死了。
* **没有可用的 OpenGL**——远程桌面、虚拟机、没装显卡驱动的机器上，
  依赖全齐也画不出东西。自检会明确报这一条。这种情况用网页版
  `run_gui.bat`，功能是一样全的。
