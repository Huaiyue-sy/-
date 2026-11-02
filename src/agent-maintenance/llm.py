"""设备运维诊断智能体 —— 大模型接入层。

优先调用 DeepSeek API（环境变量 DEEPSEEK_API_KEY；也支持读取项目根目录
被 .gitignore 忽略的 .env 或 config/deepseek.env，兼容 DEEPSEEK_BASE_URL
覆盖）；无 key 或调用失败时自动降级为内置规则模板生成诊断内容，保证离线可演示。
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _load_local_env() -> None:
    """从本地密钥文件加载 DEEPSEEK_* 配置；已有环境变量优先。"""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    root = Path(__file__).resolve().parents[2]
    for path in (root / ".env", root / "config" / "deepseek.env"):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("DEEPSEEK_"):
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ.setdefault(key, value)


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat",
                 base_url: str | None = None, timeout: int = 60):
        _load_local_env()
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL")
                         or "https://api.deepseek.com").rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, temperature: float = 0.3,
             max_tokens: int = 1024) -> str | None:
        """返回模型回复文本；失败返回 None（由调用方走离线降级）。"""
        if not self.available:
            return None
        import requests

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 headers=headers, data=json.dumps(payload),
                                 timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM 调用失败，降级为离线规则模式] {type(e).__name__}: {e}")
            return None


def llm_backend_name(client: DeepSeekClient) -> str:
    return "DeepSeek API" if client.available else "离线规则引擎（未设置 DEEPSEEK_API_KEY）"
