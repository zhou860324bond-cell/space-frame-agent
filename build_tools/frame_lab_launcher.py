"""打包版的入口。

为什么不直接拿 `desktop/app.py` 当入口：它靠 `Path(__file__).parent.parent`
往 `sys.path` 里塞仓库根和 `src`。冻结之后那两个目录不存在，
这行要么塞进一个错的路径，要么塞进 `_MEIPASS` 下一个碰巧同名的目录——
两种都难查。打包版换一个显式入口，路径由 PyInstaller 负责。

`QT_API` 仍然要在任何 Qt import 之前钉死，理由见 `desktop/app.py` 的说明；
运行时钩子 `runtime_hook.py` 已经设过一次，这里再兜一次底，
因为钩子的执行时机在不同 PyInstaller 版本上变过。
"""

import os
import sys

os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("PYTHONUTF8", "1")


def main() -> int:
    from desktop.app import main as run
    return run() or 0


if __name__ == "__main__":
    sys.exit(main())
