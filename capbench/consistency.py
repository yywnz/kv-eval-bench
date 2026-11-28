"""一致性对照：缓存路径、PD 传输、以及任何"无损搬运"类改动。

这类改动的正确性判据不是「准确率掉没掉」，而是：

    **同一批 prompt，两次输出必须逐 token 相同。**

理由：缓存/传输不改数值计算。若输出变了，只有两种可能——命中了不同的
计算路径（bug），或者 KV 在被压缩/搬运时丢了精度（那就不是无损改动）。
准确率抓不住这个：198 题掉 2 分还在噪声里，但"特定前缀的请求整批错乱"
在准确率上可能只表现为掉 0.5 分。

三种用法：

    # ① 冷热一致（prefix cache / radix / HiCache）
    python -m capbench consistency --model X --mode cold-warm --n 40

    # ② 两个部署对比（非 PD vs PD 分离）
    python -m capbench consistency --model X --mode endpoints \\
        --base-url http://h1:30000 --compare-url http://h2:30000

    # ③ 只采样 prompt、不比较（检查请求本身是否稳定）
    python -m capbench consistency --model X --mode paint --n 40

判据：
    identical_rate == 100%  → 通过（无损搬运）
    否则 → 有分歧，看 first_div 的分布：集中在某个位置说明生成的早期就分叉，
           分散则是数值噪声累积
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from .client import Model
from .prompts import render
from .runner import load_exemplars, load_rows, effective_decode
from .tasks import TASKS


def _tokens_of(resp: dict) -> list[str]:
    """从 logprobs 还原 token 序列；拿不到就退回字符级切分。"""
    lp = resp.get("logprobs") or {}
    content = lp.get("content") if isinstance(lp, dict) else None
    if isinstance(content, list) and content:
        toks = [str(x.get("token", "")) for x in content]
        if any(toks):
            return toks
    text = resp.get("text") or ""
    return list(text)


def _compare(a: list[str], b: list[str]) -> dict:
    """逐 token 比，返回一致前缀长度与首个分歧点。"""
    n = min(len(a), len(b))
    first = None
    for i in range(n):
        if a[i] != b[i]:
            first = i
            break
    same = first is None and len(a) == len(b)
    return {"identical": same, "first_div": first,
            "len_a": len(a), "len_b": len(b),
            "prefix_ok": n if first is None else first}


def _mkprompts(model: Model, task: str, n: int, data_dir: Path,
               prompt_dir: Path, splits: Path, variant: str) -> list[dict]:
    """取 n 条真实题目，按该任务官方模板拼 prompt。"""
    rows = load_rows(data_dir, task)
    rows = rows[:n]
    ex = load_exemplars(splits)
    shape = TASKS[task].shape
    out = []
    for r in rows:
        text, tpl = render(r, shape, variant, prompt_dir, ex)
        out.append({"id": r["id"], "prompt": text, "template": tpl})
    return out


def run(model: Model, prompts: list[dict]) -> list[dict]:
    """跑一遍。两次调用之间不做额外预热——第一次跑就会把前缀灌进缓存。"""
    from .client import Client
    client = Client(model)

    out = []
    for p in prompts:
        rec = {"id": p["id"]}
        try:
            resp = client.complete([{"role": "user", "content": p["prompt"]}],
                                   model.max_tokens, 0.0, 1.0)
            rec.update({"tokens": _tokens_of(resp),
                        "cached": (resp.get("usage") or {})
                        .get("prompt_tokens_details", {}).get("cached_tokens", 0),
                        "finish": resp.get("finish_reason")})
        except Exception as e:
            rec.update({"error": f"{type(e).__name__}: {e}", "tokens": []})
        out.append(rec)
    return out


def summarize(a: list[dict], b: list[dict], label_a: str, label_b: str) -> dict:
    by_b = {x["id"]: x for x in b}
    per, divs = [], []
    for x in a:
        y = by_b.get(x["id"])
        if y is None:
            continue
        c = _compare(x.get("tokens") or [], y.get("tokens") or [])
        c["id"] = x["id"]
        c["cached_b"] = y.get("cached", 0)
        per.append(c)
        if c["first_div"] is not None:
            divs.append(c["first_div"])

    n = len(per) or 1
    ok = sum(1 for c in per if c["identical"])
    return {
        "label_a": label_a, "label_b": label_b,
        "n": len(per),
        "identical": ok,
        "identical_rate": round(ok / n * 100, 2),
        "avg_prefix_ratio": round(
            sum(c["prefix_ok"] / max(1, min(c["len_a"], c["len_b"] or 1)) for c in per) / n * 100, 2),
        "first_div_median": sorted(divs)[len(divs) // 2] if divs else None,
        "first_div_min": min(divs) if divs else None,
        "first_div_max": max(divs) if divs else None,
        "b_cache_hit_rate": round(
            sum(1 for c in per if c["cached_b"]) / n * 100, 2),
        "per_item": per,
    }


def print_report(s: dict) -> None:
    print(f"\n{s['label_a']}  ←→  {s['label_b']}")
    print(f"  题数            {s['n']}")
    print(f"  逐 token 一致   {s['identical']}/{s['n']}  "
          f"({s['identical_rate']}%)")
    print(f"  一致前缀占比    {s['avg_prefix_ratio']}%")
    if s["first_div_min"] is not None:
        print(f"  首个分歧位置    min={s['first_div_min']} "
              f"中位={s['first_div_median']} max={s['first_div_max']}")
    print(f"  第二侧缓存命中  {s['b_cache_hit_rate']}% 的请求")
    if s["b_cache_hit_rate"] == 0:
        print("  ! 第二遍没有任何请求命中缓存：前缀缓存未开启、容量不足，")
        print("    或者缓存已被驱逐。此时本次只验证了「两次输出一致」，")
        print("    并没有验证缓存路径本身——要测缓存，先确认 cached_tokens>0。")
    print()
    if s["identical_rate"] == 100.0:
        print("  通过：两组输出逐 token 相同。")
    else:
        bad = [c for c in s["per_item"] if not c["identical"]]
        print(f"  **不通过**：{len(bad)} 题输出不同。无损搬运的判据是 100%，"
              "不是『差不多』。")
        print("  首个分歧位置的分布能提示方向：集中在很小的下标 → 生成早期就"
              "走了不同分支（模板/位置编码差异）；分散 → 数值累积。")
        for c in bad[:5]:
            print(f"    - {c['id']}: first_div={c['first_div']} "
                  f"len_a={c['len_a']} len_b={c['len_b']}")



def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")