"""控制台输出编码。

Windows 控制台默认是 GBK。代码里的 `✓`、`⚠`、`↔` 这些字符 GBK 编不出来，
`print` 到一半就抛 UnicodeEncodeError——**一次已经算完的分析，
被一个装饰性字符打死在打印环节**。这类崩溃最没道理，也最容易骗人：
看起来像求解器出了问题，其实结果早就算对了。

所以两道保险：

1. 尽量把 stdout/stderr 切成 UTF-8（配合 .bat 里的 chcp 65001，
   中文也就不再是乱码）；
2. 切不动就退而求其次，把 errors 设成 replace——**宁可显示一个问号，
   也不能让打印把程序打死**。

每个入口脚本开头调一次 `use_utf8()` 即可。库代码不要调，
改全局 IO 状态是入口的事。
"""

from __future__ import annotations

import sys


def use_utf8() -> str:
    """把标准输出切到 UTF-8。返回实际生效的编码，方便自检时打印出来。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:                   # 被重定向成了别的对象
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:                         # noqa: BLE001
            # 切不成就只放宽错误处理。能少崩一次是一次
            try:
                reconfigure(errors="replace")
            except Exception:                     # noqa: BLE001
                pass
    return getattr(sys.stdout, "encoding", "?") or "?"
