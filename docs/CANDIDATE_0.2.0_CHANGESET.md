# v0.2.0-rc1 候选变更清单

> 冻结日期：2026-09-14。基线提交：`8beba37`。候选分支：
> `codex/release-0.2.0-rc1`。以下按主要责任域唯一归类；跨域文件只列一次。

开始稳定化时共有 69 个修改或新增文件。稳定化本身新增本清单、增量回归脚本和
Windows 入口，因此候选工作区共 72 个变化文件。

## 求解器与对标（21）

- `abaqus_bench/README.md`
- `abaqus_bench/benchmark_B31.md`
- `abaqus_bench/benchmark_B33.md`
- `abaqus_bench/compare.py`
- `abaqus_bench/models.py`
- `abaqus_bench/b31_refinement.py`
- `abaqus_bench/benchmark_B31_matched.md`
- `abaqus_bench/benchmark_B31_refinement.md`
- `src/abaqus_backend.py`
- `src/inp_writer.py`
- `src/native_joint.py`
- `src/solid_joint.py`
- `src/convergence.py`
- `src/solid_cache.py`
- `desktop/convergence_dialog.py`
- `desktop/solid_task.py`
- `tests/test_benchmark.py`
- `tests/test_solid_joint.py`
- `tests/test_convergence.py`
- `tests/test_solid_cache.py`
- `tests/test_solid_task.py`

## Agent（9）

- `desktop/chat_panel.py`
- `desktop/agent_guidance.py`
- `src/agent.py`
- `src/conversation.py`
- `src/model_router.py`
- `tests/test_agent.py`
- `tests/test_agent_guidance.py`
- `tests/test_conversation.py`
- `tests/test_desktop_chat.py`

## 前端交互（13）

- `desktop/bc_panel.py`
- `desktop/commands.py`
- `desktop/doctor.py`
- `desktop/i18n.py`
- `desktop/main_window.py`
- `desktop/panels.py`
- `desktop/result_inspector.py`
- `desktop/result_rows.py`
- `desktop/ribbon.py`
- `tests/test_console_and_doctor.py`
- `tests/test_desktop_manual_model.py`
- `tests/test_desktop_window.py`
- `tests/test_result_inspector.py`

## 云图与三维显示（6）

- `desktop/scene.py`
- `desktop/theme.py`
- `desktop/viewport.py`
- `tools/render_viewport.py`
- `tests/test_desktop_scene.py`
- `tests/test_engineering_visual_acceptance.py`

## 文档与打包（23）

- `.github/workflows/tests.yml`
- `.gitignore`
- `CHANGELOG.md`
- `README.md`
- `docs/BETA_0.1_CAPABILITY_MATRIX.md`
- `docs/BETA_0.1_RELEASE_CHECKLIST.md`
- `docs/SOLID_JOINT_SCOPE.md`
- `docs/word/使用手册.docx`
- `docs/word/大学生教程.docx`
- `docs/word/来源指纹.json`
- `docs/word/课程报告.docx`
- `docs/使用手册.md`
- `docs/大学生教程.md`
- `docs/报告骨架.md`
- `docs/课程报告.md`
- `pyproject.toml`
- `pytest.ini`
- `run.bat`
- `tests/test_packaging.py`
- `tools/build_docx.py`
- `docs/CANDIDATE_0.2.0_CHANGESET.md`
- `tools/run_incremental_tests.py`
- `run_incremental.bat`

## 回归映射

| 变化域 | 命令 | 当前结果 |
|---|---|---|
| Agent | `run_incremental.bat agent` | 157 项通过 |
| 前端交互、云图与三维显示 | `run_incremental.bat frontend` | 154 项通过 |
| 求解器与对标 | `run_incremental.bat solver` | 246 项通过，含28项金标准；当前解已重新对比冻结的 B33/B31 `.dat`，未启动新 Abaqus 作业 |
| 文档与打包 | `run_incremental.bat docs` | 161 项通过、2 项按环境跳过 |
| 当前候选全部增量 | `run_incremental.bat --candidate` | 718 项通过、2 项按环境跳过；不等于全量回归 |

发布标签只能在候选工作区完成一次全量回归后创建。增量测试通过只说明本轮直接受影响
的合同未回退，不能替代全仓状态耦合检查。
