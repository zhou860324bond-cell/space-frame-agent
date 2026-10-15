"""应用模型配置；密钥只在内存或 Windows 当前用户的 DPAPI 密文中保存。"""
from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def redact(text: str, keys=()) -> str:
    """异常、日志与界面统一隐藏已知密钥及常见鉴权字段。"""
    for key in sorted((k for k in keys if k), key=len, reverse=True):
        text = text.replace(key, "[密钥已隐藏]")
    return re.sub(r"(?i)(bearer\s+|api[_-]?key[\s:=]+)([^\s,;\"']+)",
                  r"\1[密钥已隐藏]", text)


def _dpapi(data: bytes, decrypt: bool = False) -> bytes:
    if os.name != "nt":
        raise ValueError("此平台不支持 Windows 安全存储，请使用仅本次运行的密钥")

    class Blob(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buf = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    blob_ptr = ctypes.POINTER(Blob)
    crypt.CryptProtectData.argtypes = [blob_ptr, ctypes.c_wchar_p, blob_ptr,
                                       ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_ulong, blob_ptr]
    crypt.CryptUnprotectData.argtypes = [blob_ptr, ctypes.c_void_p, blob_ptr,
                                         ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_ulong, blob_ptr]
    if decrypt:
        ok = crypt.CryptUnprotectData(ctypes.byref(source), None, None, None,
                                      None, 1, ctypes.byref(target))
    else:
        ok = crypt.CryptProtectData(ctypes.byref(source), "FrameLab API", None,
                                    None, None, 1, ctypes.byref(target))
    if not ok:
        raise OSError("Windows 安全存储失败，请选择仅本次运行或检查用户账户")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(target.data)


@dataclass
class ModelProfile:
    id: str
    label: str
    model: str
    base_url: str
    protocol: str = "openai-chat"
    api_key: str = field(default="", repr=False)
    temperature: float | None = None
    max_tokens: int = 4096
    persist_key: bool = False

    def validate(self, require_key=True):
        if not all((self.id.strip(), self.label.strip(), self.model.strip())):
            raise ValueError("配置名称与模型标识不能为空，请填写提供商的实际模型标识")
        url = urlsplit(self.base_url)
        if (url.scheme != "https" or not url.netloc or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("API 地址须为 HTTPS 基础地址，不能包含账户、密钥或查询参数")
        if self.protocol not in ("openai-chat", "anthropic"):
            raise ValueError("接口协议不支持，请选择 Chat Completions 或 Anthropic Messages")
        if (require_key and not self.api_key.strip()) or any(c.isspace() for c in self.api_key):
            raise ValueError("API 密钥不能为空或包含空白，请重新粘贴密钥")
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("采样温度须在 0 到 1 之间，或使用提供商默认值")
        if not 256 <= self.max_tokens <= 32768:
            raise ValueError("输出上限须在 256 到 32768 之间")

    def identity(self):
        return (self.protocol, self.base_url.rstrip("/"), self.model,
                hashlib.sha256(self.api_key.encode()).hexdigest(),
                self.temperature, self.max_tokens)


class ProfileStore:
    def __init__(self, path: Path | None = None):
        override = os.environ.get("FRAMELAB_SETTINGS_FILE")
        folder = (Path(override).parent if override else
                  Path(os.environ.get("LOCALAPPDATA", Path.home() / ".config")) / "FrameLab")
        self.path = path or folder / "agent_models.json"

    def load(self) -> tuple[list[ModelProfile], str]:
        if not self.path.exists():
            return [], ""
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            if doc.get("version") != 1:
                raise ValueError
            profiles = []
            for entry in doc["profiles"]:
                entry = dict(entry)
                sealed = entry.pop("sealed_key", "")
                # 文件里的明文密钥永远不加载，避免旧文件成为不安全的保存方式。
                entry.pop("api_key", None)
                key = _dpapi(base64.b64decode(sealed, validate=True), True).decode() if sealed else ""
                profiles.append(ModelProfile(**entry, api_key=key))
                profiles[-1].validate(require_key=False)
            if len({p.id for p in profiles}) != len(profiles):
                raise ValueError
            return profiles, doc.get("selected", "")
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            raise ValueError("模型配置无法读取或密钥无法解密，请在模型连接设置中重新配置") from None

    def save(self, profiles: list[ModelProfile], selected: str):
        if len({p.id for p in profiles}) != len(profiles):
            raise ValueError("连接标识重复，请重新创建连接")
        entries = []
        for profile in profiles:
            profile.validate(require_key=False)
            entry = asdict(profile)
            entry.pop("api_key")
            entry["sealed_key"] = (base64.b64encode(_dpapi(profile.api_key.encode())).decode()
                                   if profile.persist_key else "")
            entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".agent-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "selected": selected, "profiles": entries},
                          stream, ensure_ascii=False, indent=2)
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
