"""把《课程报告》和《使用手册》导出成 Word。

为什么要有这个脚本，而不是手敲一条 pandoc 命令：**文档是会改的**。
用户的要求是"每改一次都要做出改动"，那导出就必须是一条能重复跑的命令，
不能是某次在某个终端里拼出来、下次谁也复现不了的参数串。

用法（在仓库根目录）：

```bat
python tools\\build_docx.py
```

产物落在 `docs/word/`。样式来自 `docs/word模板.docx`：
正文宋体五号、标题黑体、1.5 倍行距、正文首行缩进两字、表格加了边框和表头底纹。
模板是二进制，直接改模板文件即可，不用动这个脚本。

依赖 pandoc。优先使用系统命令，也支持 `space-frame-agent[docs]` 安装的便携版；
两者都没有时脚本会直说，而不是丢一个 FileNotFoundError。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from zipfile import BadZipFile, ZipFile

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
TEMPLATE = DOCS / "word模板.docx"
OUT_DIR = DOCS / "word"
# 导出时把每份 Markdown 的指纹记下来。测试拿它判断 Word 版是不是落后于正文——
# 文件时间戳过不了 git，内容哈希才靠得住。
STAMP = OUT_DIR / "来源指纹.json"


def fingerprint(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# 每份文档的标题页信息。副标题那行在 Markdown 里是正文第二行，
# 导出时抽出来当副标题，免得 Word 里标题和副标题挤成一段。
JOBS = [
    {
        "source": "课程报告.md",
        # title/subtitle 排在 Word 的标题页上；heading/lead 是正文里要删掉的那两行
        "title": "基于 AI 的空间刚架智能计算 Agent",
        "subtitle": "《计算结构力学》免试大作业 ｜ 大连理工大学 ｜ 指导教师：闫军",
        "heading": "基于 AI 的空间刚架智能计算 Agent",
        "lead": "《计算结构力学》免试大作业 ｜ 大连理工大学 ｜ 指导教师：闫军",
        "toc_depth": 2,
    },
    {
        "source": "大学生教程.md",
        "title": "空间刚架智能计算 Agent 上手教程",
        "subtitle": "大学生版 ｜ 功能全景与判读要点",
        "heading": "空间刚架智能计算 Agent · 上手教程（大学生版）",
        "lead": None,
        "toc_depth": 2,
    },
    {
        "source": "使用手册.md",
        "title": "空间刚架智能计算 Agent 使用手册",
        "subtitle": "从零开始的操作指南",
        "heading": "使用手册",
        "lead": None,   # 手册正文第二行是要留着的，别删
        "toc_depth": 2,
    },
]


# Word 的分页符。目录和正文之间要断开，否则正文从目录下半页起排，很难看。
PAGE_BREAK = ("`<w:r><w:br w:type=\"page\"/></w:r>`{=openxml}")


def strip_front_matter(text: str, heading: str, lead: str | None) -> str:
    """去掉正文开头的一级标题和副标题行。

    这两行改由 metadata 传给 pandoc，用 Word 的「标题」「副标题」样式排。
    留在正文里会重复印一遍，还会多占一条目录。
    副标题按**去掉全部 Markdown 星号**后的文字比对：原文那行是
    `**《计算结构力学》免试大作业** ｜ …`，只剥两端的星号是配不上的。
    不比对就删，则容易误删正文。
    """
    lines = text.splitlines()
    out, dropped_title, dropped_subtitle = [], False, False
    for line in lines:
        bare = line.replace("*", "").strip()
        if not dropped_title and line.startswith("# ") and line[2:].strip() == heading.strip():
            dropped_title = True
            continue
        if dropped_title and lead and not dropped_subtitle and bare == lead.strip():
            dropped_subtitle = True
            continue
        out.append(line)
    if not dropped_title:
        print(f"  ⚠ 正文开头没找到标题「{heading}」，标题页可能会重复", file=sys.stderr)
    return PAGE_BREAK + "\n\n" + "\n".join(out).lstrip("\n")


def find_pandoc() -> str:
    """找到系统 Pandoc，或 pypandoc_binary 随包携带的可执行文件。"""
    executable = shutil.which("pandoc")
    if executable:
        return executable
    try:
        import pypandoc

        executable = pypandoc.get_pandoc_path()
    except (ImportError, OSError):
        executable = ""
    if executable:
        candidate = pathlib.Path(executable)
        if candidate.is_file():
            return str(candidate)
        # pypandoc_binary 在 Windows 上返回不带 .exe 的资源路径。
        windows_candidate = candidate.with_suffix(".exe")
        if windows_candidate.is_file():
            return str(windows_candidate)
    raise SystemExit(
        "没找到 pandoc。可执行 pip install -e \".[docs]\"，"
        "或按 https://pandoc.org/installing.html 安装。")


def build(job: dict, pandoc: str, *, keep_temp: bool = False) -> pathlib.Path:
    source = DOCS / job["source"]
    if not source.exists():
        raise SystemExit(f"找不到 {source}——文档被删或改名了，先确认再导出")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / (source.stem + ".docx")
    pending = OUT_DIR / (source.stem + ".building.docx")
    pending.unlink(missing_ok=True)

    # pandoc 按**输入文件所在目录**解析图片相对路径，所以临时文件要放在 docs/ 下
    temp = DOCS / f".{source.stem}.docx.md"
    temp.write_text(strip_front_matter(source.read_text(encoding="utf-8"),
                                       job["heading"], job["lead"]),
                    encoding="utf-8")

    cmd = [
        pandoc, temp.name, "-o", str(pending),
        "-f", "gfm+raw_attribute",
        f"--reference-doc={TEMPLATE}",
        "--toc", f"--toc-depth={job['toc_depth']}",
        "-M", "toc-title=目录",
        "-M", f"title={job['title']}",
        "-M", f"subtitle={job['subtitle']}",
    ]
    try:
        proc = subprocess.run(cmd, cwd=DOCS, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("没找到 pandoc。装一个再来：https://pandoc.org/installing.html")
    finally:
        if not keep_temp:
            temp.unlink(missing_ok=True)

    if proc.returncode != 0:
        pending.unlink(missing_ok=True)
        raise SystemExit(f"pandoc 失败（退出码 {proc.returncode}）：\n{proc.stderr}")
    try:
        with ZipFile(pending) as archive:
            damaged = archive.testzip()
            names = set(archive.namelist())
    except (BadZipFile, OSError) as exc:
        pending.unlink(missing_ok=True)
        raise SystemExit(f"Word 临时文件校验失败：{exc}") from exc
    if damaged or "word/document.xml" not in names:
        pending.unlink(missing_ok=True)
        raise SystemExit(
            f"Word 临时文件结构不完整：{damaged or '缺少 word/document.xml'}")
    try:
        pending.replace(target)
    except PermissionError as exc:
        raise SystemExit(
            f"{target.name} 正被 Word 或预览程序占用，旧文件保持不变。\n"
            f"已生成并校验新版：{pending}\n关闭占用程序后重新运行本脚本即可替换。") from exc
    for line in proc.stderr.splitlines():
        if "Could not fetch resource" in line:
            print(f"  ⚠ {line.strip()}", file=sys.stderr)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="导出课程报告与使用手册的 Word 版")
    parser.add_argument("--keep-temp", action="store_true",
                        help="保留中间 Markdown，排查排版问题时用")
    args = parser.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"找不到样式模板 {TEMPLATE}")
    pandoc = find_pandoc()

    stamp = {}
    for job in JOBS:
        target = build(job, pandoc, keep_temp=args.keep_temp)
        size = target.stat().st_size
        stamp[job["source"]] = fingerprint(DOCS / job["source"])
        print(f"已导出 {target.relative_to(ROOT)}（{size / 1024:.0f} KB）")
    pending_stamp = STAMP.with_name(".来源指纹.building.json")
    pending_stamp.write_text(
        json.dumps(stamp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending_stamp.replace(STAMP)
    print("\nWord 里打开后按 Ctrl+A、F9 更新一次目录，页码才会填上。")


if __name__ == "__main__":
    main()
