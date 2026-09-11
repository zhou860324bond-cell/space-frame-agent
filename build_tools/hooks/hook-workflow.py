"""覆盖 pyinstaller-hooks-contrib 自带的 `hook-workflow.py`。

**这是重名撞车，不是缺依赖。**

本项目有 `src/workflow.py`（建模流程状态机）。PyPI 上另有一个毫不相干的
包也叫 `workflow`，pyinstaller-hooks-contrib 为它写了一个 stdhook，
内容是 `datas = copy_metadata('workflow')`。

PyInstaller 按**模块名**匹配钩子，不看这个模块是谁的。于是分析到
`import workflow` 时它去找 PyPI `workflow` 的包元数据，机器上当然没有：

    importlib.metadata.PackageNotFoundError: No package metadata was found for workflow
    PyInstaller.exceptions.ImportErrorWhenRunningHook: ... hook-workflow.py

构建直接中止，而报错里一个字都没提"你有个同名模块"。

`hookspath` 里的钩子优先级高于自带钩子，所以放一个空的同名钩子即可让开。
留空是对的：我们这个 `workflow` 是本地源码，没有包元数据，也不需要收集任何
附加文件——它的代码由常规依赖分析收走。

改名 `src/workflow.py` 也能解决，但那会动到一堆 import 和测试，
为了迁就一个打包工具的重名匹配去改源码，代价和风险都更大。
"""

datas = []
binaries = []
hiddenimports = []
