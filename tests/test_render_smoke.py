"""真渲染冒烟测试（见 tools/render_smoke.py）。

其余桌面端测试都在无头环境里跑，视口绘制整段跳过；这几轮前端的问题几乎
全出在真渲染路径上（色标参数写错抛异常、抽屉越界、色标被盖住）。这条在
子进程里开一个真窗口，把每种显示模式真画一遍。

验证过它抓得住：把应力比色标那个 ``categories`` 参数错误放回去，它报
「应力比：绘制无异常」「应力比：有色标」两项失败、退出码 1。

需要能开窗口、有 OpenGL：CI（无显卡）上默认跳过；想在 CI 上强制跑，
设 ``FRAMELAB_RENDER_SMOKE=1``。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import opengl_available

ROOT = Path(__file__).resolve().parent.parent

pytestmark = [
    pytest.mark.skipif(not opengl_available(), reason="无可用 OpenGL"),
    pytest.mark.skipif(bool(os.environ.get("CI"))
                       and not os.environ.get("FRAMELAB_RENDER_SMOKE"),
                       reason="CI 没有真显卡；设 FRAMELAB_RENDER_SMOKE=1 强制运行"),
]


def test_every_display_mode_really_renders(tmp_path):
    pytest.importorskip("pyvistaqt")
    env = {key: value for key, value in os.environ.items()
           if key != "QT_QPA_PLATFORM"}
    env.update(PYTHONUTF8="1",
               FRAMELAB_AUTOSAVE_DIR=str(tmp_path / "autosave"),
               FRAMELAB_SETTINGS_FILE=str(tmp_path / "settings.ini"))
    report = tmp_path / "render_smoke.json"
    got = subprocess.run([sys.executable, str(ROOT / "tools" / "render_smoke.py"),
                          str(report)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=300)
    failed = [line for line in got.stdout.splitlines() if line.startswith("[!!]")]
    assert got.returncode == 0, ("真渲染冒烟未通过：\n" + "\n".join(failed)
                                 + "\n" + got.stderr[-1500:])
