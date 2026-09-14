# v0.2.0-rc1 候选版验收与演示

> 日期：2026-09-14。文件名为兼容既有链接暂时保留；实验性功能不作为验收前提。

当前回归清单共收集 1728 项。最近一次完整基线为 1727 项通过、1 项按环境跳过；
v0.2.0-rc1 四组增量并集共收集 720 项，718 项通过、2 项按环境跳过。发布标签创建前
仍需在干净环境执行一次全量回归，用来确认这些分组在同一进程中运行时没有状态耦合。

## 最终验收

| 项目 | 状态 | 证据 |
|---|---|---|
| 一条命令启动和测试 | 通过 | `run.bat`；桌面端 `run_desktop.bat` |
| 旧项目迁移 | 通过 | `tests/test_domain_ir.py` |
| 一根物理构件对应多个分析单元 | 通过 | `tests/test_model_compiler.py` |
| 集中荷载触发自动剖分 | 通过 | `examples/beta01_demo.py`、金标准 11 |
| 线性静力进入 Result DB | 通过 | `tests/test_result_db.py` |
| 查询、图和报告同源 | 通过 | `tests/test_result_db.py`, `tests/test_model_compiler.py` |
| 28 个金标准 | 通过 | `run_gold.bat`（原 12 项线性静力，2026-09-09 扩展到自振、屈曲、P-Δ 与轴向塑性）|
| 错误模型明确失败 | 通过 | 校验、奇异诊断和静默失败测试 |
| 版本、README 与能力矩阵一致 | 通过 | 发布名 v0.2.0-rc1，Python 包版本 `0.2.0rc1`；`pyproject.toml`, `README.md`, `docs/BETA_0.1_CAPABILITY_MATRIX.md` |
| 候选变更唯一归类 | 通过 | 72 个变化文件按五个责任域唯一归类；`docs/CANDIDATE_0.2.0_CHANGESET.md` |
| 分组增量回归入口 | 通过 | `run_incremental.bat`；Agent、前端、求解器、文档四组 |
| 当前候选增量回归 | 通过 | Agent 157；前端 154；求解器 246（含28项金标准）；文档与打包 163；并集718通过、2跳过 |
| 冻结 Abaqus 数据复核 | 通过 | 当前自研解重新对比既有 B33/B31 `.dat`；本轮未启动新的 Abaqus 作业 |
| 多模态离线发布门禁 | 通过 | 30 张 manifest 固定语料；全部门槛通过；`multimodal_eval/release_report.json` |
| 多模态 V2 桌面闭环 | 通过 | 尺度→问题处理→人工编辑→diff→原子加载→单步撤销；`tests/test_issue_panel.py` |
| 真实图片验收工具链 | 通过（数据待提供） | 私有数据隔离、授权/真值/响应完整性检查、哈希冻结与批量评分；当前真实集 0 张，不报告准确率 |
| 桌面工作台视觉与效率改版 | 通过 | 中文字体可靠加载、两层上下文工具带、浅色 CAD 控制台与深色视口、空项目快捷入口、非遮挡式 AI 助手 |
| 专项收敛工作区 | 通过 | B31、实体热点/峰值、非线性增量统一表；显示采样与分析网格严格分开；`tests/test_convergence.py` |
| 节点实体任务配置与 Kt 同门禁 | 通过 | 提交前展示网格档位；Abaqus/native 均须所有候选杆臂三档热点稳定才发布 Kt；`tests/test_solid_joint.py` |
| 节点实体双后端对标 | 通过 | 同节点同工况留存；按实际热点尺寸匹配，正式比较各自收敛解；名义应力不同则拒绝；`tests/test_convergence.py` |
| 一键双后端任务 | 通过 | native→Abaqus 分阶段进度；安全边界取消；模型、节点输入与网格三重指纹跨重启续跑；主结果文件完整性门禁；`tests/test_solid_task.py`, `tests/test_solid_cache.py` |
| 实体结果同页溯源 | 通过 | 来源、生成时间、模型/节点双指纹、提交网格、VTU/ODB 完整性集中展示；旧版和损坏摘要安全降级；`tests/test_result_inspector.py` |
| 实体计算历史与版本对比 | 通过 | 每次运行独立目录、latest 原子更新；同页选择两版比较输入、网格、热点、Kt 和文件状态；输入变化时拒绝求解器归因；`tests/test_solid_cache.py`, `tests/test_result_inspector.py` |
| 实体历史空间管理 | 通过 | 同页统计每次运行的目录占用和文件数；“分析→实体结果中心”跨节点/工况扫描，首次失败且无 latest 的残留也可见；中断残留按最后写入时间保护 24 小时，二次确认后只移入 Windows 回收站；各后端 latest、当前 A/B、平铺旧结果、越界和不可核验路径由底层拒绝；`tests/test_solid_cache.py`, `tests/test_result_inspector.py` |
| 三家在线 provider smoke | 按环境跳过 | OpenAI、Anthropic、DeepSeek 均为 `SKIPPED_NO_CREDENTIALS`；不影响离线门禁 |
| Git 候选提交 | 通过 | `codex/release-0.2.0-rc1` 分支；作为全量发布检查前的可回退点 |
| Git 发布标签 | 待全量回归 | 只有候选全量通过后才创建 `v0.2.0-rc1` 标签 |

最新留存的真实模型单轮评测为 13/14（2026-09-03）；本轮工作流状态机和确认闸门已有
离线确定性回归，但尚未重新消耗外部模型额度复跑。因此不能把旧的 13/14 写成已经提升，
发布演示以离线可复现主链路和金标准为准。

## 五分钟主演示

1. 运行 `.venv\Scripts\python.exe examples\beta01_demo.py`。
2. 展示输入中只有物理构件 `101`，跨中集中荷载位置为 `a=1.5 m`。
3. 展示分析网格预览：2 个物理节点、1 根物理构件，编译为 3 个分析节点、2 个分析单元。
4. 展示 `member_mapping = {101: [101, 102]}`，说明用户对象 ID 没有被编译改写。
5. 展示 Result DB 模型哈希、`Mz` 查询值及生成的弯矩图，说明查询与图读取同一结果源。

演示前可运行：

```bat
run_gold.bat
run_desktop.bat --check
```

## 故障演示

运行 `.venv\Scripts\python.exe examples\agent_demo.py`。脚本会故意漏掉一个支座方向，
展示求解失败、刚体模态定位、修复约束、重新求解和平衡校核；调用顺序由脚本固定，
数值仍由确定性工具实时计算。

## 回滚说明

- 模型内操作：桌面端 `Ctrl+Z`，或在建模历史中跳回指定步骤。
- 可复现实验：求解后保存胶囊，用 `python -m capsule diff <A> <B>` 比较输入哈希与结果。
- 代码级回滚：候选增量回归通过后创建独立 Git 提交；发布标签必须等待候选全量通过。
