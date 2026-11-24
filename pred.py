#!/usr/bin/env python
"""生成预测，写入 pred/{model}.jsonl。

对应 LongBench 的 pred.py：只负责「按模板拼 prompt → 请求端点 → 落盘」，
统计和出表交给 result.py。这样换统计口径不用重跑模型。

  python pred.py --model deepseek-v4.1                       # 全部任务
  python pred.py --model deepseek-v4.1 --tasks mmlu_pro,aime
  python pred.py --model my-model --base-url http://127.0.0.1:30000 --limit 20
  python pred.py --model deepseek-v4.1 --prompt-variant cot
  python pred.py --model deepseek-v4.1 --tasks longbench_v2 --length short,medium
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from capbench import config as cfgmod                     # noqa: E402
from capbench.prompts import render                        # noqa: E402
from capbench.runner import filter_length, generate, load_rows, summarize  # noqa: E402
from capbench.tasks import ORDER, TASKS                    # noqa: E402


def parse_tasks(spec: str | None) -> list[str]:
    if not spec or spec == "all":
        return list(ORDER)
    out: list[str] = []
    for t in (x.strip() for x in spec.split(",")):
        if t and t not in out:          # 去重，保持顺序；重复会让同一题被算两次
            out.append(t)
    bad = [t for t in out if t not in TASKS]
    if bad:
        sys.exit(f"未知任务 {bad}，可选：{', '.join(ORDER)}")
    return out


def merge(path: Path, recs: list[dict], tasks: list[str], revision: str) -> int:
    """读旧文件，丢掉本次重跑的（task, revision）记录，再追加。"""
    kept = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("task") in tasks and r.get("revision") == revision:
                    continue
                kept.append(r)
    with path.open("w", encoding="utf-8") as f:
        for r in kept + recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(kept)


def main():
    ap = argparse.ArgumentParser(description="CapBench pred：生成预测")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--config", default=str(ROOT / "config" / "models.json"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--prompts", default=str(ROOT / "prompts"))
    ap.add_argument("--out", default=str(ROOT / "pred"))
    ap.add_argument("--prompt-variant", default="0shot", help="0shot / cot / no_context")
    ap.add_argument("--limit", type=int, default=None, help="每任务只跑前 N 题")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--length", default=None, help="只跑指定长度档，如 short,medium")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--revision", default=None, help="权重版本标签，用于区分多次结果")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--thinking", choices=["on", "off"], default=None)
    ap.add_argument("--extra-body", default=None)
    args = ap.parse_args()

    model = cfgmod.resolve(
        args.model, cfgmod.load(Path(args.config)),
        base_url=args.base_url, revision=args.revision, max_tokens=args.max_tokens,
        thinking=args.thinking, extra_body=args.extra_body)

    tasks = parse_tasks(args.tasks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{model.name}.jsonl"

    print(f"model    : {model.name}  rev={model.revision}")
    print(f"endpoint : {model.base_url}  max_tokens={model.max_tokens}  maxlen={model.maxlen}")
    print(f"template : {args.prompt_variant}    output: {pred_path}")
    print()

    all_recs: list[dict] = []
    for task in tasks:
        rows = filter_length(load_rows(Path(args.data), task), args.length)
        shape = TASKS[task].shape
        t0 = time.time()

        def render_fn(row, shape=shape):
            return render(row, shape, args.prompt_variant, Path(args.prompts))

        def progress(done, total, t0=t0):
            if done % max(1, total // 10) == 0 or done == total:
                print(f"\r  {done}/{total}  {time.time() - t0:.0f}s", end="", flush=True)

        print(f"[{task}] n={len(rows) if not args.limit else min(len(rows), args.limit)}")
        recs = generate(model, task, rows, render_fn, concurrency=args.concurrency,
                        limit=args.limit, on_progress=progress)
        print()
        all_recs.extend(recs)
        s = summarize(task, model, recs)
        acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']}%"
        print(f"  acc={acc:<8} n_scored={s['n_scored']}/{s['n']}  err={s['n_error']}  "
              f"超窗={s['over_ctx']}  截断={s['n_truncated']}  未解析={s['n_unparsed']}")

    prev = merge(pred_path, all_recs, tasks, model.revision)
    print(f"\n已写入 {len(all_recs)} 条（保留其他任务/版本 {prev} 条）-> {pred_path}")
    print(f"下一步：python result.py --model {model.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
