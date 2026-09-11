"""PyInstaller 运行时钩子：在任何 import 之前定死几个环境变量。

三条，每条都是踩出来的：

1. **`QT_API=pyside6`**。`pyvistaqt` 经 `qtpy` 选绑定，`qtpy` 默认按
   PyQt5 → PySide2 → PyQt6 → PySide6 探测。打包体里只有 PySide6，
   但用户机器上的环境变量会漏进来，一旦它指向别的绑定就是闪退，
   而且报错看不出根源。
2. **`PYTHONUTF8=1`**。中文 Windows 默认 GBK，读写模型 JSON 会炸。
3. **`MPLBACKEND`** 交给 matplotlib 自己选之前先排除 tkinter：
   打包体里没有 tk，matplotlib 探到 TkAgg 会在第一次画图时才崩。
"""

import os

os.environ["QT_API"] = "pyside6"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "QtAgg")
# 用户机器上残留的这些会覆盖掉打包体自带的 Qt 插件路径，一定要清掉
for leaked in ("QT_PLUGIN_PATH", "QML2_IMPORT_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
    os.environ.pop(leaked, None)
