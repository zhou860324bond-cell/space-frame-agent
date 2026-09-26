# 给编码助手的项目约定

Codex 直接读这份文件，Claude Code 通过 `CLAUDE.md` 引用它——两边共用一份，只改这里。

## 开工与收工

- **开工先读 `docs/PROGRESS.md`**：当前在做什么、下一步、已知问题都在那里。
- **收工前更新 `docs/PROGRESS.md`**：改了什么、没做完什么、下一步是什么，写上日期。
- 进度只通过 git 交换：开工前 `git pull`，收工时提交并推送。两个工具不要同时改同一个分支。
- 合进 master 走 PR，CI（ubuntu 3.11 / 3.12、windows 3.12）全绿再合；合并前先问用户。

## 运行

```
.venv\Scripts\activate.bat
set PYTHONUTF8=1
python -m pytest            :: 全量约 7 分钟
python -m ruff check .      :: 必须干净
```

- 无头环境要 `QT_QPA_PLATFORM=offscreen`；`tests/test_render_smoke.py` 需要真显卡，CI 上自动跳过。
- `tests/conftest.py` 把自动保存目录、设置文件、capsule 目录都重定向到临时目录，测试不会碰用户数据。

## 文档闸（`tests/test_docs_current.py`）

加减测试、工具、文件之后，下面几处要一起改，少一处就红：

1. `README.md` 首屏回归基线「**N 项通过、1 项按环境跳过**」。
2. `docs/课程报告.md` 里回归总数的**每一处**，以及 2.1 节代码规模表（`src` / `tests` / `desktop` / `tools` 的文件数精确、行数 5% 以内）。
3. 手册里「」引用的界面名字必须在 `desktop/` 代码里真实存在。
4. 改完 Markdown 运行 `python tools/build_docx.py` 重新导出 `docs/word/`。
5. `CHANGELOG.md` 最上面加一节，写明日期。

## 界面文字（`tests/test_ui_text.py`）

- 统一写「杆件」，不写「构件」。
- 不用口语和内部黑话：跑一遍、搞定、崩了、守门、兜底等。
- 提示要说清原因并给出下一步。
- 每个命令名都要在 `desktop/i18n.py` 的 `ZH_EN` 里有英文，且英文不重名。

## 代码

- 核心原则：**大模型只产结构，不产数值**——报告里的每个数都要能追到某次工具调用。
- 注释与提交信息用中文，提交信息按 `feat(范围): …` / `fix(…)` / `perf(…)` / `docs: …` 的格式。
- 新增修复要带回归测试，测试的文档字符串写清楚它防的是哪个真实问题。
- Windows 是目标平台：注意路径分隔符、换行符（CRLF）和中文编码。
