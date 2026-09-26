"""模型自动保存。

**为什么要有。** 用 AI 对话建的模型从来不经过「保存」：用户说一句话，模型
就在内存里长出来了。一个 27 节点 42 杆的管结构就是这么建的——要不是求解
时顺手存了实验胶囊，关掉窗口它就没了（实际就是从胶囊里翻回来的）。

做法：

* 模型每变一次（界面每次 refresh 之后）就**推迟 1.5 s** 写一份，连续操作
  只写最后一次，不拖慢交互；
* 与上次「保存/打开」的文件内容一样就不写，并删掉旧的自动存档——已经
  落盘的东西不需要再备份，也免得下次启动问一个多余的问题；
* 写文件先写临时文件再替换，写到一半断电不会留下半个 JSON；
* 目录在用户数据目录下，不在项目文件夹里；测试把它指到临时目录
  （``FRAMELAB_AUTOSAVE_DIR``），否则一次全量回归会往用户目录里写上千次。

下次启动时 ``pending()`` 给出上次没保存的模型，由主窗口问要不要恢复。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PySide6.QtCore import QObject, QStandardPaths, QTimer

FILE_NAME = "unsaved_model.json"
DELAY_MS = 1500


def default_directory() -> Path:
    override = os.environ.get("FRAMELAB_AUTOSAVE_DIR")
    if override:
        return Path(override)
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base or Path.home() / ".framelab") / "autosave"


class AutoSave(QObject):
    """挂在主窗口上：``schedule()`` 在模型可能变了之后调，其余自己管。"""

    def __init__(self, window, directory: Path | None = None,
                 delay_ms: int = DELAY_MS):
        super().__init__(window)
        self._window = window
        self.directory = Path(directory) if directory else default_directory()
        self.path = self.directory / FILE_NAME
        self._saved_key: bytes | None = None      # 上次保存/打开时的模型指纹
        self._written_key: bytes | None = None    # 自动存档里现在是哪一版
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(delay_ms))
        self._timer.timeout.connect(self.flush)

    # --- 触发 ---

    def schedule(self) -> None:
        self._timer.start()

    def mark_saved(self) -> None:
        """用户刚保存或打开了一个文件：当前内容已经落盘，不再需要备份。"""
        self._saved_key = self._key()
        self._timer.stop()
        self.discard()

    def flush(self) -> None:
        """现在就写（或删）。关窗口时也调一次，不等计时器。"""
        self._timer.stop()
        model = self._window.session.model
        key = self._key()
        if not model.get("nodes") or key == self._saved_key:
            self.discard()
            return
        if key == self._written_key and self.path.exists():
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "model": model}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False),
                             encoding="utf-8")
        os.replace(temporary, self.path)
        self._written_key = key

    def discard(self) -> None:
        self._written_key = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    # --- 恢复 ---

    def pending(self) -> dict | None:
        """上次没保存的模型（``{"saved_at", "model"}``）；没有或读不了返回 None。"""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        model = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(model, dict) or not model.get("nodes"):
            return None
        return payload

    def _key(self) -> bytes:
        return self._window.session._model_key()
