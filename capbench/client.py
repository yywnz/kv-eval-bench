"""OpenAI 兼容端点客户端。只依赖标准库。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class ClientError(RuntimeError):
    pass


@dataclass
class Model:
    name: str
    base_url: str = "http://127.0.0.1:30000"
    api_key: str = "EMPTY"
    revision: str = "default"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    # 原样并入请求体，用于 chat_template_kwargs 等端点特有参数
    extra_body: dict = field(default_factory=dict)
    # True 时忽略题目自带的 max_tokens，一律用 self.max_tokens
    lock_max_tokens: bool = False
    # 有效上下文长度（tokens），仅用于超窗预检
    maxlen: int = 32768
    # 超窗预检的字符/token 换算比
    chars_per_token: float = 3.0

    def payload(self, messages: list[dict], max_tokens: int,
                temperature: float | None = None) -> dict:
        body = {
            "model": self.name,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "logprobs": True,
            "top_logprobs": 5,
            "stream": False,
        }
        body.update(self.extra_body or {})
        return body


class Client:
    """非流式 chat completions，同时取回正文、思考内容与首步 logprobs。"""

    def __init__(self, model: Model, timeout: float = 900.0):
        self.m = model
        self.base = model.base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.m.api_key:
            req.add_header("Authorization", f"Bearer {self.m.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:400]
            raise ClientError(f"HTTP {e.code}: {detail}") from e
        except Exception as e:  # 连接失败 / 超时
            raise ClientError(f"连接失败 {self.base}{path}: {e}") from e

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
                r.read(64)
            return True
        except Exception:
            return False

    def complete(self, messages: list[dict], max_tokens: int,
                 temperature: float | None = None) -> dict:
        body = self.m.payload(messages, max_tokens, temperature)
        t0 = time.perf_counter()
        out = self._post("/v1/chat/completions", body)
        dt = (time.perf_counter() - t0) * 1000.0
        ch = (out.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        return {
            "text": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "logprobs": ch.get("logprobs"),
            "finish_reason": ch.get("finish_reason"),
            "usage": out.get("usage") or {},
            "latency_ms": dt,
        }
