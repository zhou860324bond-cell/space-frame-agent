# Beta 0.1 能力矩阵

> 更新：2026-09-08。此文件是 Beta 0.1 的能力边界；早期 Alpha 路线图中的数量和缺口描述仅作历史记录。

状态定义：**支持**表示已进入正式主链路并有自动测试；**实验性**表示代码可用但未作为本轮承诺；**延期**表示本轮明确不实现。

| 能力 | 状态 | 当前边界 | 验证证据 |
|---|---|---|---|
| Domain IR v1 | 支持 | 顶层 `schema_version=1`；无版本 Alpha 模型迁移为 v1；未知版本拒绝 | `tests/test_domain_ir.py` |
| 稳定物理构件 ID | 支持 | 编译不改用户物理构件 ID | `tests/test_model_compiler.py` |
| PhysicalMember→AnalysisElement | 支持 | 权威双向映射；一根物理构件可对应多个分析单元 | `tests/test_model_compiler.py` |
| 集中荷载自动剖分 | 支持 | 多工况切分点取并集；近端点不切分；同位置荷载合并 | `tests/test_model_compiler.py` |
| 显式内节点自动剖分 | 支持 | 已有节点精确落在构件内部时切分；集中荷载落在该节点时直接复用 | `tests/test_model_compiler.py` |
| 均布/梯形荷载随剖分转换 | 支持 | 均布复制到各段；梯形荷载按段端位置插值 | `tests/test_model_compiler.py` |
| 杆端释放映射 | 支持 | 仅保留在物理构件最外端，内部生成端不继承释放 | `tests/test_model_compiler.py` |
| 分析网格预览 | 支持（GUI/Agent） | GUI 在“分析→分析网格”叠加物理构件、分析单元与切分节点；`preview_analysis_mesh` 提供同一份只读映射；均不覆盖已有结果 | `tests/test_model_compiler.py`, `tests/test_tool_contracts.py`, `tests/test_desktop_scene.py`, `tests/test_desktop_window.py` |
| 精确人工建模 | 支持（GUI） | 支持工作平面及偏移、网格捕捉、已有节点吸附和带单位的 x/y/z 坐标建点；重复坐标为只读复用，不清除结果或污染撤销历史 | `tests/test_desktop_manual_model.py`, `tests/test_agent.py` |
| Agent 工作流状态 v1 | 支持 | 每轮由 Session 确定性推导 `empty/draft/ready/solved`；工具执行后即时刷新校验错误与推荐工具，不依赖模型自述 | `tests/test_workflow.py`, `tests/test_conversation.py` |
| Agent 变更预演与确认 v1 | 支持 | 写操作可在隔离 Session 预演并返回实体级 diff/前后哈希；Agent 删除已有节点或杆件时由代码强制预演，只有后续用户消息明确确认且模型未变化才可应用 | `tests/test_change_preview.py`, `tests/test_conversation.py` |
| 线性静力 Result DB | 支持 | `Analysis→Step→Frame→FieldOutput`；U/UR/RF/RM/N/Vy/Vz/T/My/Mz | `tests/test_result_db.py` |
| 结果元数据 | 支持 | 单位、位置、坐标系、平均规则、模型哈希、求解状态 | `tests/test_result_db.py` |
| 物理构件结果聚合 | 支持 | 首末物理端结果不平均；跨分析段连续恢复内力与挠度 | `tests/test_model_compiler.py` |
| 查询、绘图、报告同源 | 支持 | 三者由 Result DB 的只读结果视图和同一映射消费 | `tests/test_result_db.py`, `tests/test_model_compiler.py` |
| 28 个金标准算例 | 支持 | 独立经典梁理论、静力平衡与闭合解特征值真值；覆盖线性静力 12、自振 5、屈曲 9、二阶弹性与轴向塑性 2 | `run_gold.bat`, `docs/BETA_0.1_GOLD_CASES.md` |
| P-Δ 二阶弹性 | 支持（求解） | 收敛到精确解 `δ = H/(Pk)·(tan kL − kL)`，8 单元相对误差 6.8e-7；**小应变二阶弹性，不是有限转动 NLGEOM** | 金标准 27 |
| 轴向双线性塑性 | 支持（求解） | 对双线性应变闭合解相对 `2e-7`；**仅轴向，弯曲仍保持弹性**，弯曲屈服需纤维截面 | 金标准 28 |
| P-Δ / 轴向双线性结果入库 | 实验性 | 求解本身已有金标准；写入 Result DB 仍经通用适配器，未冻结 | 非线性既有回归测试 |
| 模态结果入 Result DB | 延期 | 模态仍使用原有结果对象 | 后续版本 |
| 任意空间交点/属性突变切分 | 延期 | 当前不做自动识别 | 后续版本 |
| 集中荷载 + 刚域偏移自动切分 | 明确不支持 | 编译时报错，要求用户显式拆分，避免偏心力矩丢失 | `tests/test_model_compiler.py` |
| 完整几何—材料联合非线性 | 延期 | 当前 P-Δ 与轴向双线性是两条独立能力 | 后续版本 |
| 多模态图片确认链路 | 实验性 | 图片→RecognitionDraft v2→逐项核对→尺度标定→结构化 diff→原子提交；不静默补工程属性、不自动求解 | `tests/test_multimodal_workflow.py`, `tests/test_issue_panel.py`, `tests/test_draft_commit.py` |
| 图片实体交互编辑 | 实验性 | V2 原图覆盖显示；节点拖动同步图片证据并重算尺寸/交点；已有节点间可补/删杆件；编辑立即使旧预演失效 | `tests/test_issue_panel.py`, `tests/test_sketch_panel.py` |
| 图片支座/节点荷载编辑 | 实验性 | V2 支座可新增/改型/删除；工况和命名节点荷载可新增/更新/删除；六分量及单位进入追溯证据 | `tests/test_issue_panel.py`, `tests/test_sketch_panel.py` |
| 图片草稿合并到已有模型 | 实验性 | add-only 稳定重排 ID；重合节点必须逐项明确复用；支座、工况和命名荷载引用随节点/杆件映射；重复杆件和名称冲突阻断 | `tests/test_draft_commit.py`, `tests/test_issue_panel.py` |
| 图片杆件荷载编辑 | 延期 | 当前显示识别框并保护引用；确认加载后由正式边界条件面板编辑 | 后续版本 |
| 图片透视校正/三维反演 | 延期 | 当前按二维图像轴向比例编辑，保留 y 坐标；不从单张图片猜测完整三维几何 | 后续版本 |
| 多模态离线发布门禁 | 支持 | 30 张原创 CC0 固定语料；manifest 唯一绑定图片、真值、响应、模型、提示词与 schema 指纹；fixture 回放分数不等同在线模型准确率 | `multimodal_eval/manifest.json`, `multimodal_eval/release_report.json`, `tests/test_multimodal_eval.py` |
| 多模态真实图片验收 | 工具链就绪，数据待提供 | 私有样本默认不入 Git；授权、已复核真值、provider 响应齐全后才能哈希冻结和评分；当前 0 张，不宣称真实准确率 | `multimodal_eval/real_world_eval.py`, `multimodal_eval/real_world_cases/README.md` |

## 正式演示链路

```text
Domain IR 物理构件
→ compile_model
→ preview_analysis_mesh
→ 线性静力 solve_model
→ Result DB
→ query_results / query_diagram / plot_results / write_report
```

Beta 0.1 可用于受能力边界约束的梁柱刚架研究与演示，不能替代规范校核、工程审图或商业有限元软件认证。
