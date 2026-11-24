"""模型配置。

`config/models.json` 是模型注册表，对应 LongBench 的 `model2path.json` +
`model2maxlen.json`。字段：

    base_url        OpenAI 兼容端点（到端口，不带 /v1）
    maxlen          该模型的有效上下文长度（tokens），用于「超窗预检」
    max_tokens      生成上限（覆盖题目自带值）
    temperature     默认 0
    extra_body      原样并入请求体，如 {"chat_template_kwargs": {"thinking": true}}
    chars_per_token 超窗预检用的换算比，默认 3.0（英文按字符数估 token）

命令行参数优先级高于配置文件；给 `--base-url` 就能用未注册的模型名临时跑。
"""

from __future__ import annotations

import json
from pathlib import Path

from .client import Model

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "models.json"


def load(path: Path | None = None) -> dict:
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return {"models": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def resolve(name: str, cfg: dict | None = None, *, base_url: str | None = None,
            revision: str | None = None, max_tokens: int | None = None,
            thinking: str | None = None, extra_body: str | None = None,
            allow_adhoc: bool = True) -> Model:
    cfg = cfg if cfg is not None else load()
    entry = (cfg.get("models") or {}).get(name)
    if entry is None:
        if not (allow_adhoc and base_url):
            raise KeyError(
                f"config/models.json 里没有模型 {name}。"
                f"二选一：① 在 models 下加一条；② 给 --base-url http://host:port 临时跑。")
        entry = {"base_url": base_url, "maxlen": 32768}

    m = Model(
        name=name,
        base_url=base_url or entry.get("base_url") or "http://127.0.0.1:30000",
        revision=revision or entry.get("revision") or "default",
        temperature=float(entry.get("temperature", 0.0)),
        max_tokens=int(max_tokens or entry.get("max_tokens") or 4096),
        extra_body=dict(entry.get("extra_body") or {}),
        maxlen=int(entry.get("maxlen") or 32768),
        chars_per_token=float(entry.get("chars_per_token") or 3.0),
    )
    # 命令行给了 max_tokens 就锁住，忽略题目自带值
    if max_tokens:
        m.lock_max_tokens = True
    # 配置文件里显式写了 max_tokens 同样锁住。否则题目自带的 1024 会覆盖它，
    # 思考模型的整批输出在思考阶段就被切断。
    elif entry.get("max_tokens"):
        m.lock_max_tokens = True

    # 模型级解码覆盖：不写就沿用任务官方值（见 tasks.py 的 decode）
    if "temperature" in entry:
        m.decode["temperature"] = float(entry["temperature"])
    if "top_p" in entry:
        m.decode["top_p"] = float(entry["top_p"])

    if thinking:
        ctk = dict(m.extra_body.get("chat_template_kwargs") or {})
        ctk["thinking"] = thinking == "on"
        m.extra_body["chat_template_kwargs"] = ctk
        if not max_tokens:
            m.max_tokens = max(m.max_tokens, 8192)
            m.lock_max_tokens = True
    if extra_body:
        m.extra_body.update(json.loads(extra_body))
    return m
