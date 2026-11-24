"""生成与汇总。

`generate` 跑一批题并判分，`summarize` 出汇总。`pred.py` 和 CLI 都调这两个。

解码参数优先级：**模型配置 > 任务官方值 > 全局默认**。
任务官方值来自各自官方实现（见 `tasks.py` 的 `decode`），例如 LongBench v2
官方只给 128 输出 token——对思考模型不够，这种情况必须在报告里写明。
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .client import Client, Model
from .scoring import score
from .stats import mean, se_pp
from .tasks import TASKS


def load_rows(data_dir: Path, task: str) -> list[dict]:
    p = Path(data_dir) / f"{task}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"缺少数据文件 {p}，先跑 `python -m capbench build --tasks {task}`")
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_exemplars(splits_dir: Path) -> dict:
    p = Path(splits_dir) / "exemplars_mmlu_pro.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def filter_length(rows: list[dict], spec: str | None) -> list[dict]:
    """按 meta.length（short/medium/long）过滤，只对长上下文数据生效。"""
    if not spec:
        return rows
    want = {s.strip().lower() for s in spec.split(",") if s.strip()}
    kept = [r for r in rows if str((r.get("meta") or {}).get("length", "")).lower() in want]
    return kept or rows


def over_context(row: dict, model: Model, gen_tokens: int) -> bool:
    """超窗预检：按字符数估 token，算上生成长度后仍超 maxlen 就跳过。

    不做截断——截断会让「超窗」变成「看起来答完了」，分数里看不出丢了什么。
    跳过的题单独统计（`over_ctx`），不进准确率分母，报告里必须一并给出。
    """
    ratio = max(model.chars_per_token, 0.5)
    est = (len(row.get("context") or "") + len(row.get("question") or "")) / ratio
    return est + gen_tokens > model.maxlen


def effective_decode(model: Model, task: str) -> dict:
    """模型配置里的解码参数优先，其次任务官方值。"""
    official = dict(TASKS[task].decode or {})
    temp = model.decode.get("temperature", official.get("temperature", model.temperature))
    top_p = model.decode.get("top_p", official.get("top_p", model.top_p))
    mt = model.max_tokens if model.lock_max_tokens else \
        (official.get("max_tokens") or model.max_tokens)
    return {"temperature": float(temp), "top_p": float(top_p), "max_tokens": int(mt)}


def generate(model: Model, task: str, rows: list[dict],
             render: Callable[[dict], tuple[str, str]],
             concurrency: int = 16, limit: int | None = None,
             on_progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """逐题生成并判分，返回记录列表（不落盘）。单题失败不中断整批。"""
    if limit:
        rows = rows[:limit]
    client = Client(model)
    dec = effective_decode(model, task)
    lock = threading.Lock()
    out: list[dict] = []
    total = len(rows)

    def work(pair):
        _, row = pair
        tpl_name = ""
        try:
            prompt, tpl_name = render(row)
            rec = {"task": task, "item_id": row["id"], "model": model.name,
                   "revision": model.revision, "template": tpl_name,
                   "answer": row.get("answer"), "meta": row.get("meta") or {},
                   "decode": dec}
            if over_context(row, model, dec["max_tokens"]):
                rec.update({"skipped": "over_context", "correct": None, "scored": False,
                            "output_text": "", "latency_ms": 0.0, "finish_reason": "skipped"})
            else:
                resp = client.complete([{"role": "user", "content": prompt}],
                                       dec["max_tokens"], dec["temperature"], dec["top_p"])
                s = score(row, resp)
                usage = resp.get("usage") or {}
                rec.update({
                    "prompt_chars": len(prompt),
                    "correct": None if s.get("correct") is None else bool(s["correct"]),
                    "scored": s.get("correct") is not None,
                    "chosen": s.get("chosen"), "parse": s.get("parse"),
                    "official": s.get("official"),
                    "margin": s.get("margin"),
                    "output_text": (resp.get("text") or "")[:4000],
                    "reasoning_text": (resp.get("reasoning") or "")[:2000],
                    "finish_reason": resp.get("finish_reason"),
                    "usage": usage,
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                    "latency_ms": round(resp.get("latency_ms", 0.0), 1),
                })
        except Exception as e:
            rec = {"task": task, "item_id": row.get("id"), "model": model.name,
                   "revision": model.revision, "template": tpl_name,
                   "error": f"{type(e).__name__}: {e}", "latency_ms": 0.0}
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with lock:
            out.append(rec)
            if on_progress:
                on_progress(len(out), total)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        list(ex.map(work, list(enumerate(rows))))
    return sorted(out, key=lambda r: str(r.get("item_id")))


def summarize(task: str, model: Model, recs: list[dict]) -> dict:
    ok = [r for r in recs if "error" not in r and not r.get("skipped")]
    scored = [r for r in ok if r.get("scored")]
    correct = sum(1 for r in scored if r["correct"])
    margins = [r["margin"] for r in scored if r.get("margin") is not None]
    acc = correct / len(scored) if scored else None
    official_hits = sum(1 for r in scored if r.get("official"))
    dec = recs[0].get("decode") if recs else {}

    return {
        "task": task, "model": model.name, "revision": model.revision,
        "n": len(recs),
        "n_error": sum(1 for r in recs if "error" in r),
        "over_ctx": sum(1 for r in recs if r.get("skipped") == "over_context"),
        "n_scored": len(scored),
        "accuracy": round(acc * 100, 2) if acc is not None else None,
        "se_pp": round(se_pp(acc, len(scored)), 2) if acc is not None else None,
        "margin_mean": round(mean(margins), 4) if margins else None,
        "n_truncated": sum(1 for r in ok if r.get("finish_reason") == "length"),
        "n_unparsed": sum(1 for r in scored if r.get("chosen") in ("?", None)),
        "official_parse_rate": (round(official_hits / len(scored) * 100, 2)
                                if scored else None),
        "avg_latency_ms": round(mean([r["latency_ms"] for r in ok]), 1) if ok else None,
        "cache_hit_rate": (round(sum(1 for r in ok if r.get("cached_tokens")) / len(ok) * 100, 2)
                           if ok else None),
        "template": (scored or ok or [{}])[0].get("template"),
        "decode": dec,
        "maxlen": model.maxlen,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def group_by(recs: list[dict], keys: tuple[str, ...]) -> dict[str, dict]:
    """按 meta 里的字段分组算准确率（用于按学科 / 长度档看）。"""
    buckets: dict[str, list[dict]] = {}
    for r in recs:
        if "error" in r or r.get("skipped") or not r.get("scored"):
            continue
        meta = r.get("meta") or {}
        label = " / ".join(str(meta.get(k, "-")) for k in keys)
        buckets.setdefault(label, []).append(r)
    out = {}
    for label, rs in sorted(buckets.items()):
        c = sum(1 for r in rs if r["correct"])
        out[label] = {"n": len(rs), "accuracy": round(c / len(rs) * 100, 2)}
    return out
