"""桌面端（PySide6 + PyVista）。

分层与 Streamlit 版同一个原则：**会出错的东西不放在界面层**。

    scene.py        纯 PyVista 建网格，无 Qt，可无头逐条验
    viewport.py     Qt 视口，薄；只把网格塞进渲染器
    model_tree.py   模型树，只显示不改模型
    main_window.py  搭骨架、转发动作，不算也不画
    theme.py        样式表与配色（视口配色复用 viz_theme）

计算一律经 `agent.Session`——和 Streamlit 端、命令行、评测集是同一条链路。
"""
