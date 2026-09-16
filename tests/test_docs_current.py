"""文档漂移闸：报告和手册里的话与代码对不上时，测试要红。

这条闸是用户当面提的要求——"后续这些每改一次都要做出改动"。
靠自觉是守不住的：改一个菜单名、加一个工具、多写一批测试，
《课程报告》和《使用手册》立刻就过期，而没有任何东西会提醒你。
所以把"文档要跟着代码走"变成会失败的断言。

守四类东西，都是**真会漂**的：

1. **手册里印的界面名字**（「」引起来的）必须真的出现在 `desktop/` 的字符串
   字面量里。改了菜单名不改手册，这里红。
2. **两份文档引用的文件路径**必须存在。删了脚本不改文档，这里红。
3. **报告里的关键数字**——工具数、金标准条数、测试总数——必须与代码对得上。
4. **金标准清单里点名的 pytest 用例**必须真的存在，编号必须连续无缺。

只有第 3 类里的"测试总数"需要跑全量才能核对，跑子集时自动跳过；
其余三类任何时候都在守。
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANUAL = ROOT / "docs" / "使用手册.md"
TUTORIAL = ROOT / "docs" / "大学生教程.md"
REPORT = ROOT / "docs" / "课程报告.md"
GOLD = ROOT / "docs" / "BETA_0.1_GOLD_CASES.md"
README = ROOT / "README.md"

# 用户自己新建的密钥文件，仓库里本来就不该有
OPTIONAL_PATHS = {"deepseek.key"}

_PATH_RE = re.compile(r"[\w./\\一-鿿-]+\.(?:py|md|bat|txt|toml|json|png|yml)")


def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _squash(text: str) -> str:
    """把连续空白压成一个空格。

    工作流条上印的是"3  校验"（两个空格是为了排版好看），手册里写成
    "3 校验"。这种差别不该让测试红，所以两边都压一遍再比。
    """
    return re.sub(r"\s+", " ", text).strip()


def _ui_literals() -> str:
    """`desktop/` 里所有字符串字面量拼成一大块，供"界面上有没有这句话"用。"""
    chunks = []
    for path in sorted((ROOT / "desktop").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                chunks.append(_squash(node.value))
    return "\n".join(chunks)


def _cited_paths(text: str) -> list[str]:
    """取反引号里那些看着像仓库路径的串。"""
    out = []
    for raw in re.findall(r"`([^`\n]+)`", text):
        s = raw.strip().replace("\\", "/")
        if _PATH_RE.fullmatch(s) and s not in OPTIONAL_PATHS:
            out.append(s)
    return sorted(set(out))


UI_TEXT = _ui_literals()


# ---------------------------------------------------------------- 手册

def test_manual_exists():
    assert MANUAL.exists(), "《使用手册》不在了；它是交付物之一，不能删"


@pytest.mark.parametrize("name", sorted({
    part.strip()
    for doc in (MANUAL, TUTORIAL)
    for whole in re.findall(r"「([^」]+)」", _text(doc))
    for part in whole.split("→")
    if part.strip()
}))
def test_manual_ui_names_still_exist_in_the_code(name):
    """手册里「」引的每个界面名字，代码里必须真有这句话。

    手册是写给零基础的人照着点的。点名一个已经改过名的按钮，
    读者只会卡在那一步，还以为是自己看错了。
    """
    assert _squash(name) in UI_TEXT, (
        f"《使用手册》里写着「{name}」，但 desktop/ 的字符串字面量里找不到这句话。"
        "要么界面改了名而手册没跟上，要么手册写错了——两种都得改。")


@pytest.mark.parametrize("cited", sorted(set(
    _cited_paths(_text(MANUAL)) + _cited_paths(_text(TUTORIAL)))))
def test_manual_file_paths_still_exist(cited):
    assert (ROOT / cited).exists(), \
        f"手册或教程让读者去找 {cited}，但这个文件不在仓库里"


def test_the_tutorial_exists():
    assert TUTORIAL.exists(), "《大学生教程》不在了；它是交付物之一，不能删"


def test_manual_screenshots_are_all_present():
    """手册里插的图必须都在，少一张读者就断在那一步。"""
    missing = [s for s in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", _text(MANUAL))
               if not (MANUAL.parent / s).exists()]
    assert not missing, f"《使用手册》引了这些图但文件不在：{missing}"


# ---------------------------------------------------------------- 报告

def test_report_exists():
    assert REPORT.exists(), "《课程报告》不在了；它是交付物之一，不能删"


@pytest.mark.parametrize("cited", _cited_paths(_text(REPORT)))
def test_report_file_paths_still_exist(cited):
    assert (ROOT / cited).exists(), \
        f"《课程报告》把 {cited} 写成出处，但这个文件不在仓库里——出处不能是空的"


def test_report_tool_count_matches_the_registry():
    """报告里的「N 个确定性工具」必须等于 agent.TOOLS 的实际长度。"""
    from agent import TOOLS

    claimed = {int(n) for n in re.findall(r"(\d+)\s*个确定性工具", _text(REPORT))}
    assert claimed, "《课程报告》里找不到「N 个确定性工具」这句话了"
    assert claimed == {len(TOOLS)}, (
        f"《课程报告》说有 {sorted(claimed)} 个确定性工具，实际注册 {len(TOOLS)} 个。"
        "加减工具之后要同步改报告。")


def test_report_gold_case_count_matches_the_frozen_list():
    numbers = _gold_case_numbers()
    claimed = {int(n) for n in re.findall(r"(\d+)\s*(?:项|个)解析金标准", _text(REPORT))}
    assert claimed, "《课程报告》里找不到「N 项解析金标准」这句话了"
    assert claimed == {len(numbers)}, (
        f"《课程报告》说有 {sorted(claimed)} 项解析金标准，"
        f"`docs/BETA_0.1_GOLD_CASES.md` 里实际冻结 {len(numbers)} 项。")


def _require_countable_full_run(collected: int) -> None:
    """总条数这类断言的公共前置：跑的得是全量，平台还得对得上。

    **收集条数是跟平台走的。** Linux 上比 Windows 少十几条——有些用例依赖
    只在 Windows 上成立的东西。文档里记的是 Windows 满配那一套数字（这个
    项目的用户和作者都在 Windows，Abaqus 也只在 Windows 上），所以一个数字
    不可能同时对得上两个平台。

    这一点是 CI 头一次加上 windows-latest 那格之后才暴露的：在那之前只跑
    Linux，数字就写成了本机（Windows）的，两边永远对不上。

    于是让这条闸只在 Windows 上生效。**没有变松**：改测试的人在 Windows 上，
    CI 的 windows-latest 那格也会执行它，该拦的照样拦得住。
    """
    if collected < 800:
        pytest.skip("只有跑全量回归时才数得出总条数")
    if sys.platform != "win32":
        pytest.skip("收集条数随平台不同；文档记的是 Windows 满配的基线，"
                    "由 CI 的 windows-latest 那一格来守")


def test_report_test_total_matches_the_full_run(request):
    """报告里**每一处**回归条数，都必须等于这次真正收集到的条数。

    用 `request.session.items`：pytest 先收集完再开跑，所以这里拿到的
    就是全量条数。跑子集时数不出来，直接跳过而不是瞎断言。

    刻意查全文而不是只查附录 C——这条闸第一次跑起来就抓到正文 2.1 节
    和 5.x 节各还留着一处旧数字，只盯一处是盯不住的。

    **注意这里有一圈自指**：本文件里那几条闸是按文档内容参数化的
    （文档多引一个路径、手册多提一个界面名，用例就多一条），所以改完文档
    再跑一次，总条数还会再变一点。跑两轮就收敛，不是死循环——
    知道这回事就不会以为是测试写错了。
    """
    collected = len(request.session.items)
    _require_countable_full_run(collected)
    flat = _text(REPORT).replace(",", "").replace("，", "，")
    claimed = {int(n) for n in re.findall(r"回归\s*\*{0,2}\s*(\d{3,})\s*项", flat)}
    assert claimed, "《课程报告》里找不到「回归 N 项」这句话了"
    assert claimed == {collected}, (
        f"《课程报告》里写着回归 {sorted(claimed)} 项，这次实际收集 {collected} 项。"
        f"加了测试就要把报告里**每一处**数字都改成 {collected}。")


@pytest.mark.parametrize("folder", ["src", "tests", "desktop", "tools"])
def test_report_code_scale_table_is_not_stale(folder):
    """2.1 节那张代码规模表：文件数必须精确对上，行数允许 5% 漂移。

    文件数卡死是因为它只在**加减模块**时才变，那种改动本来就该顺手更新报告。
    行数给 5% 的余量则是刻意的：改三行代码就逼人去改一次报告，
    这道闸会被当成噪音关掉，那就一点用都没有了。
    """
    row = re.search(rf"^\|\s*`{folder}`\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|",
                    _text(REPORT), re.M)
    assert row, f"《课程报告》的代码规模表里没有 `{folder}` 这一行了"
    said_files, said_lines = (int(g.replace(",", "")) for g in row.groups())

    files = sorted((ROOT / folder).glob("*.py"))
    lines = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files)

    assert said_files == len(files), (
        f"报告说 `{folder}` 有 {said_files} 个文件，实际 {len(files)} 个")
    assert abs(said_lines - lines) <= max(20, lines * 0.05), (
        f"报告说 `{folder}` 有 {said_lines:,} 行，实际 {lines:,} 行，"
        f"差得超过 5% 了——把表里的数字改成 {lines:,}")


# ------------------------------------------------------- 金标准清单

def _gold_case_numbers() -> set[int]:
    """把清单里的编号取出来，`18~21` 这种区间要展开。"""
    numbers: set[int] = set()
    for label in re.findall(r"^\|\s*([\d~ -]+?)\s*\|\s*`", _text(GOLD), re.M):
        label = label.replace("-", "~")
        if "~" in label:
            lo, hi = (int(x) for x in label.split("~"))
            numbers.update(range(lo, hi + 1))
        else:
            numbers.add(int(label))
    return numbers


def test_gold_case_numbers_are_continuous():
    """编号必须从 1 排到 N 不缺号——缺号就说明删过条目却没重排。"""
    numbers = _gold_case_numbers()
    assert numbers, "金标准清单里一条编号都没解析出来，表格格式八成改了"
    assert numbers == set(range(1, max(numbers) + 1)), (
        f"金标准编号不连续，缺：{sorted(set(range(1, max(numbers) + 1)) - numbers)}")


def test_gold_case_headline_matches_the_table():
    """清单开头那句「冻结 N 个」必须等于表格里真实的条数。"""
    numbers = _gold_case_numbers()
    claimed = {int(n) for n in re.findall(r"冻结\s*\*?\*?(\d+)\s*个", _text(GOLD))}
    assert claimed == {len(numbers)}, (
        f"`BETA_0.1_GOLD_CASES.md` 开头说冻结 {sorted(claimed)} 个，"
        f"表格里实际列了 {len(numbers)} 个")


@pytest.mark.parametrize("case", sorted({
    m.split("[")[0]
    for m in re.findall(r"^\|\s*[\d~ -]+?\s*\|\s*`([^`]+)`", _text(GOLD), re.M)
    if m.startswith("test_")
}))
def test_gold_cases_point_at_tests_that_exist(case):
    """金标准点名的 pytest 用例必须真的还在。

    改名或删掉一个金标准用例而清单没跟着改，等于报告里那"28 项"
    有一项是空头支票。
    """
    hit = any(re.search(rf"\bdef {re.escape(case)}\b", p.read_text(encoding="utf-8"))
              for p in sorted((ROOT / "tests").glob("test_*.py")))
    assert hit, f"金标准清单点名 `{case}`，但 tests/ 里没有这个用例了"


# ---------------------------------------------------------------- README

def test_readme_tool_count_matches_the_registry():
    from agent import TOOLS

    claimed = {int(n) for n in re.findall(r"(\d+)\s*个确定性工具", _text(README))}
    assert claimed == {len(TOOLS)}, (
        f"README 说注册 {sorted(claimed)} 个确定性工具，实际 {len(TOOLS)} 个")


def test_readme_regression_baseline_matches_the_full_run(request):
    """README 首屏那句回归基线，通过数 + 跳过数必须等于实际收集条数。

    README 是别人打开仓库第一眼看的东西。那里的数字过期，
    整个仓库的可信度就先打了折。
    """
    collected = len(request.session.items)
    _require_countable_full_run(collected)
    hit = re.search(r"(\d+)\s*项通过、(\d+)\s*项按环境跳过", _text(README))
    assert hit, "README 里找不到「N 项通过、M 项按环境跳过」这句话了"
    stated = int(hit.group(1)) + int(hit.group(2))
    assert stated == collected, (
        f"README 写的是 {hit.group(1)} 通过 + {hit.group(2)} 跳过 = {stated} 项，"
        f"这次实际收集 {collected} 项——跑完 pytest 要把这句话改过来。")


# ------------------------------------------------------------ Word 版

def test_word_exports_are_not_behind_the_markdown():
    """`docs/word/` 里的 Word 版必须是从当前 Markdown 导出的。

    交作业交的是 Word。正文改完忘了重新导出，交上去的就是旧版——
    这种错最难自己发现，因为 Markdown 看着是对的。

    时间戳过不了 git，所以比的是内容哈希：`tools/build_docx.py`
    每次导出会把两份 Markdown 的 sha256 写进 `docs/word/来源指纹.json`。
    对不上就说明该重新跑一次导出了。

    哈希前先把 CRLF 归一成 LF，口径必须和 `build_docx.fingerprint()` 一致。
    不归一的话，Windows 上按 Git 默认配置（`core.autocrlf=true`）clone
    下来的这份文件是 CRLF，字节对不上指纹，这条断言会拿"换行符不同"
    冒充"文档改了没重新导出"——两种红长得一样，没法查。
    """
    import json

    stamp_path = ROOT / "docs" / "word" / "来源指纹.json"
    assert stamp_path.exists(), (
        "找不到 docs/word/来源指纹.json——先跑一次 `python tools/build_docx.py`")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))

    stale = []
    for name, recorded in stamp.items():
        source = ROOT / "docs" / name
        assert source.exists(), f"指纹里记着 {name}，但这份文档不在了"
        now = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if now != recorded:
            stale.append(name)
        assert (ROOT / "docs" / "word" / (source.stem + ".docx")).exists(), \
            f"{name} 的 Word 版不见了"

    assert not stale, (
        f"这几份文档改过了，但 Word 版还是旧的：{stale}。"
        "跑一次 `python tools/build_docx.py` 重新导出。")
