"""密钥读取的边界。

这个模块的测试只关心**从哪儿读**和**会不会漏出去**，不关心密钥内容。
"""

from __future__ import annotations

def test_a_frozen_build_looks_for_the_key_next_to_the_exe(monkeypatch, tmp_path):
    """打包成 exe 之后，密钥要在 exe 旁边找。

    不能用 `_MEIPASS`：那个临时解压目录每次启动重建，用户放进去的文件
    下次就没了，而且他根本找不到那个目录。这条错了的表现是
    "明明把 key 放好了，程序还说未配置"——极难自查。
    """
    import credentials

    exe = tmp_path / "FrameLab.exe"
    exe.write_bytes(b"")
    (tmp_path / credentials.KEY_FILE).write_text("sk-from-next-to-the-exe",
                                                 encoding="utf-8")
    monkeypatch.delenv(credentials.ENV_VAR, raising=False)
    monkeypatch.setattr(credentials.sys, "frozen", True, raising=False)
    monkeypatch.setattr(credentials.sys, "executable", str(exe))

    assert credentials.load_api_key() == "sk-from-next-to-the-exe"
    assert credentials.source() == credentials.KEY_FILE
