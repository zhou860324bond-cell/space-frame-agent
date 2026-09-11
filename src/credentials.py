"""API 密钥的读取。

**这个模块的全部职责就是把密钥交出去，并且保证它不会出现在任何别的地方。**

密钥泄露最常见的路径不是被人攻破，而是自己抄给了别人：贴一段终端日志、
截一张图、把报错发到群里。所以这里立三条规矩：

1. 只从环境变量和本地 `deepseek.key`（已被 .gitignore 忽略）读；
2. **绝不打印、绝不写日志、绝不进异常消息**；
3. 需要在界面上"显示"密钥状态时，用 `fingerprint()`——它给的是一个
   不可还原的短指纹，能回答"是不是换了一把钥匙"，但换不回原文。

`mask()` 那种留后四位的做法这里**不用**：密钥不像卡号，后四位对用户没有
识别价值，却白白泄露了信息。
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ENV_VAR = "DEEPSEEK_API_KEY"
KEY_FILE = "deepseek.key"


def _repo_root() -> Path:
    """密钥文件该去哪儿找。

    源码运行时是仓库根；**打包成 exe 之后是 exe 所在的那个目录**，
    不是 `_MEIPASS`——那个临时解压目录每次启动都会重建，
    用户把 `deepseek.key` 放进去也留不住，而且他根本找不到那个目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # src/credentials.py → 仓库根目录
    return Path(__file__).resolve().parent.parent


def load_api_key(root: Path | None = None) -> str | None:
    """环境变量优先，其次本地密钥文件。找不到返回 None，不抛异常——
    "没有密钥"是一种正常状态（离线演示照样跑），不是错误。"""
    from_env = os.environ.get(ENV_VAR, "").strip()
    if from_env:
        return from_env
    path = (root or _repo_root()) / KEY_FILE
    if path.is_file():
        # utf-8-sig：Windows 记事本存文件会带 BOM，不吃掉的话密钥前面多个字符，
        # 报出来是 401，很难想到是记事本干的
        text = path.read_text(encoding="utf-8-sig").strip()
        if text:
            return text.splitlines()[0].strip()
    return None


def source(root: Path | None = None) -> str:
    """密钥从哪来。给界面显示用，**不含密钥内容**。"""
    if os.environ.get(ENV_VAR, "").strip():
        return f"环境变量 {ENV_VAR}"
    if ((root or _repo_root()) / KEY_FILE).is_file():
        return KEY_FILE
    return "未配置"


def fingerprint(key: str | None) -> str:
    """密钥的短指纹。用于回答"是不是换了一把钥匙"，且不可还原。"""
    if not key:
        return "无"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
