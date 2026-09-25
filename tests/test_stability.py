"""多轮稳定性汇总的验证。

汇总逻辑要能把"每轮都过""每轮都挂""时好时坏"三种情况分清楚——
最后一种正是要在报告里点名的，混进前两种就等于把波动藏起来了。
注入假的 runner，不打任何 API。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evalset"))

from run_repeat import code_fingerprint, render, repeat   # noqa: E402
from score import Verdict                                  # noqa: E402

CASES = [
    {"id": "AAA", "category": "稳过", "checks": {}},
    {"id": "BBB", "category": "稳挂", "checks": {}},
    {"id": "CCC", "category": "时好时坏", "checks": {}},
]


def verdict(case, passed, rounds=5, failed_check="某项检查"):
    v = Verdict(case_id=case["id"], category=case["category"], rounds=rounds)
    v.checks = {"检查": True} if passed else {failed_check: False}
    v.usage = {"prompt_tokens": 100, "completion_tokens": 10, "cache_hit_tokens": 80}
    return v


def make_runner(pattern):
    """pattern[题号] 是一串布尔，第 k 轮取第 k 个。"""
    counter = {c["id"]: 0 for c in CASES}

    def runner(case):
        k = counter[case["id"]]
        counter[case["id"]] += 1
        return verdict(case, pattern[case["id"]][k], rounds=4 + k)
    return runner


def run_three():
    pattern = {"AAA": [True, True, True],
               "BBB": [False, False, False],
               "CCC": [True, False, True]}
    return repeat(CASES, make_runner(pattern), times=3, progress=False)


def test_repeat_collects_one_verdict_per_case_per_run():
    results = run_three()
    assert set(results) == {"AAA", "BBB", "CCC"}
    assert all(len(v) == 3 for v in results.values())


def test_report_marks_only_the_unstable_case():
    text = render(CASES, run_three(), 3, 12.0, {"源码摘要": "abc123"})
    assert "| AAA | 稳过 | 3/3 |" in text
    assert "| BBB | 稳挂 | 0/3 |" in text
    assert "| CCC | 时好时坏 | **2/3** |" in text, "只有时好时坏的题该被加粗点名"


def test_report_calls_out_instability_in_the_conclusion():
    text = render(CASES, run_three(), 3, 12.0, {"源码摘要": "abc123"})
    assert "不能只贴某一轮的成绩单" in text
    assert "CCC" in text.split("## 结论")[1]
    assert "AAA" not in text.split("## 结论")[1]


def test_report_says_so_when_every_run_agrees():
    pattern = {"AAA": [True] * 3, "BBB": [False] * 3, "CCC": [True] * 3}
    results = repeat(CASES, make_runner(pattern), times=3, progress=False)
    text = render(CASES, results, 3, 12.0, {"源码摘要": "abc123"})
    assert "结果完全一致" in text
    assert "**" not in text.split("## 逐题稳定性")[1].split("## 用量")[0]


def test_report_shows_the_pass_range_when_runs_differ():
    text = render(CASES, run_three(), 3, 12.0, {"源码摘要": "abc123"})
    assert "每轮通过数：[2, 1, 2]" in text
    assert "范围 1/3 ~ 2/3" in text


def test_rounds_are_summarised_min_mean_max():
    text = render(CASES, run_three(), 3, 12.0, {"源码摘要": "abc123"})
    assert "4 / 5.0 / 6" in text


def test_usage_is_summed_across_all_runs():
    text = render(CASES, run_three(), 3, 12.0, {"源码摘要": "abc123"})
    assert "输入：900" in text          # 3 题 × 3 轮 × 100
    assert "输出：90" in text


def test_fingerprint_covers_the_source_modules():
    fp = code_fingerprint()
    assert len(fp["源码摘要"]) == 12
    found, total = fp["计入文件数"].split("/")
    assert int(found) == int(total), f"有源文件没被指纹覆盖：{fp.get('未找到')}"


def test_fingerprint_changes_when_a_source_file_changes(tmp_path, monkeypatch):
    import run_repeat
    before = code_fingerprint()["源码摘要"]
    target = run_repeat.ROOT / "evalset" / "cases.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# fingerprint probe\n")
        assert code_fingerprint()["源码摘要"] != before
    finally:
        target.write_bytes(original)
    assert code_fingerprint()["源码摘要"] == before


def test_fingerprint_covers_modules_whose_text_reaches_the_model():
    """silent_failures.py 的提示原样进工具回包——改它等于改提示词，指纹必须变。

    旧的手写名单漏了它：2026-09-23 修 zero_internal_force 提示前后，指纹一模一样。
    """
    import run_repeat
    before = code_fingerprint()["源码摘要"]
    target = run_repeat.ROOT / "src" / "silent_failures.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# fingerprint probe\n")
        assert code_fingerprint()["源码摘要"] != before
    finally:
        target.write_bytes(original)
    assert code_fingerprint()["源码摘要"] == before
