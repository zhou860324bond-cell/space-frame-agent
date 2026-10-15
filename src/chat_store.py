"""对话记录落盘：每段对话一个 JSON 文件，重启后可以在会话列表里找回。

保存两样东西：
- ``entries``：界面上看到的消息、工具卡片、计划与改动记录，用来原样重绘；
- ``exchanges``：发给模型的完整回合（``Conversation.export_history``），
  恢复后继续追问时模型仍记得前文。

目录与模型连接配置放在一起（``FRAMELAB_SETTINGS_FILE`` 所在目录或
``%LOCALAPPDATA%/FrameLab``），测试通过环境变量隔离，不会碰用户数据。
写入先落临时文件再替换，断电或崩溃不会留下半截 JSON。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = "framelab-chat/v1"
_ID = re.compile(r"^[0-9a-f]{32}$")


def default_directory() -> Path:
    override = os.environ.get("FRAMELAB_SETTINGS_FILE")
    base = (Path(override).parent if override else
            Path(os.environ.get("LOCALAPPDATA", Path.home() / ".config")) / "FrameLab")
    return base / "chats"


def new_record(title: str = "新对话") -> dict[str, Any]:
    now = time.time()
    return {"schema": SCHEMA, "id": uuid.uuid4().hex, "title": title,
            "created": now, "updated": now, "entries": [], "exchanges": []}


class ChatStore:
    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory) if directory else default_directory()

    def _path(self, chat_id: str) -> Path:
        if not _ID.match(str(chat_id)):
            raise ValueError("对话编号无效")
        return self.directory / f"{chat_id}.json"

    def list(self) -> list[dict[str, Any]]:
        """按最近修改排序的会话摘要；损坏的文件跳过，不让一个坏文件拖垮列表。"""
        if not self.directory.is_dir():
            return []
        out = []
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("schema") != SCHEMA or not _ID.match(str(data.get("id", ""))):
                continue
            preview = next((e.get("text", "") for e in data.get("entries", [])
                            if e.get("type") == "user"), "")
            out.append({"id": data["id"], "title": data.get("title") or "新对话",
                        "updated": float(data.get("updated", 0)),
                        "messages": sum(1 for e in data.get("entries", []) if e.get("type") == "user"),
                        "preview": preview[:60]})
        return sorted(out, key=lambda item: item["updated"], reverse=True)

    def load(self, chat_id: str) -> dict[str, Any]:
        data = json.loads(self._path(chat_id).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError("对话文件版本不兼容")
        data.setdefault("entries", [])
        data.setdefault("exchanges", [])
        return data

    def save(self, record: dict[str, Any], secrets: list[str] | None = None) -> None:
        """写入一段对话。``secrets`` 中的密钥出现在任何文字里都会被替换掉。"""
        record = dict(record, schema=SCHEMA, updated=time.time())
        text = json.dumps(record, ensure_ascii=False, indent=1)
        if secrets:
            from model_profiles import redact
            text = redact(text, secrets)
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(record["id"])
        fd, temp = tempfile.mkstemp(prefix=".chat-", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
            os.replace(temp, target)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise

    def rename(self, chat_id: str, title: str) -> None:
        title = " ".join(str(title).split())[:40]
        if not title:
            raise ValueError("对话名称不能为空")
        data = self.load(chat_id)
        data["title"] = title
        self.save(data)

    def delete(self, chat_id: str) -> None:
        self._path(chat_id).unlink(missing_ok=True)

    def search(self, keyword: str) -> list[dict[str, Any]]:
        """标题或任意消息文字包含关键词的会话（不区分大小写）。"""
        keyword = keyword.strip().lower()
        if not keyword:
            return self.list()
        hits = []
        for item in self.list():
            if keyword in item["title"].lower():
                hits.append(item)
                continue
            try:
                entries = self.load(item["id"])["entries"]
            except (OSError, ValueError):
                continue
            if any(keyword in str(e.get("text", "")).lower() for e in entries):
                hits.append(item)
        return hits
