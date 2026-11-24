#!/usr/bin/env python
"""读 pred/ 出分，写 results/{model}.json 与 result.txt。

对应 LongBench 的 result.py：只做聚合，不碰端点。改了统计口径重跑这一条即可。

  python result.py --model deepseek-v4.1
  python result.py                         # 汇总 pred/ 下所有模型
  python result.py --model deepseek-v4.1 --detail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from capbench.client import Model                  # noqa: E402
from capbench.runner import group_by, summarize    # noqa: E402
from capbench.tasks import ORDER                   # noqa: E402

# 每个任务额外按什么维度拆（写进 results/{model}.json 的 groups）
GROUP_KEYS = {
    "mmlu_pro": ("category",),
    "longbench_v2": ("length", "domain"),
    "aime": ("year",),
    "livecodebench": ("platform", "difficulty"),
    "gpqa_diamond": ("domain",),
}


def load_pred(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyze(path: Path, detail: bool) -> dict:
    recs = load_pred(path)
    model_name = path.stem
    rev = (recs[0].get("revision") if recs else None) or "default"
    model = Model(name=model_name, revision=rev)

    tasks: dict[str, dict] = {}
    for task in ORDER:
        sub = [r for r in recs if r.get("task") == task]
        if not sub:
            continue
        s = summarize(task, model, sub)
        keys = GROUP_KEYS.get(task)
        if keys:
            s["groups"] = group_by(sub, keys)
        tasks[task] = s
    return {"model": model_name, "revision": rev,
            "n_records": len(recs), "tasks": tasks}


def write_table(out: dict, path: Path, header_done: bool) -> None:
    lines = []
    if not header_done:
        lines.append("Model\tTask\tn\tn_scored\tacc\tse\tover_ctx\terr\tunparsed\ttemplate")
    for task, s in out["tasks"].items():
        acc = "-" if s["accuracy"] is None else f"{s['accuracy']:.1f}"
        se = "-" if s["se_pp"] is None else f"{s['se_pp']:.2f}"
        lines.append(f"{out['model']}\t{task}\t{s['n']}\t{s['n_scored']}\t{acc}\t{se}\t"
                     f"{s['over_ctx']}\t{s['n_error']}\t{s['n_unparsed']}\t{s['template'] or '-'}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="CapBench result：聚合出分")
    ap.add_argument("--model", default=None, help="不填则汇总 pred/ 下全部")
    ap.add_argument("--pred", default=str(ROOT / "pred"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--table", default=str(ROOT / "result.txt"))
    ap.add_argument("--detail", action="store_true", help="打印分组明细")
    args = ap.parse_args()

    pred_dir = Path(args.pred)
    files = sorted(pred_dir.glob("*.jsonl"))
    if args.model:
        files = [f for f in files if f.stem == args.model]
        if not files:
            sys.exit(f"{pred_dir} 下没有 {args.model}.jsonl，先跑 pred.py")
    if not files:
        sys.exit(f"{pred_dir} 下没有任何预测，先跑 pred.py")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = Path(args.table)
    if table.exists():
        table.unlink()

    first = True
    for f in files:
        out = analyze(f, args.detail)
        (out_dir / f"{out['model']}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        write_table(out, table, header_done=not first)
        first = False

        print(f"\n== {out['model']}  rev={out['revision']}  {out['n_records']} 条记录")
        print(f"{'task':<16} {'n':>5} {'scored':>7} {'acc':>7} {'se':>6} "
              f"{'over':>5} {'err':>4} {'unparsed':>8}")
        print("-" * 70)
        for task, s in out["tasks"].items():
            acc = "-" if s["accuracy"] is None else f"{s['accuracy']:.2f}"
            se = "-" if s["se_pp"] is None else f"{s['se_pp']:.2f}"
            print(f"{task:<16} {s['n']:>5} {s['n_scored']:>7} {acc:>7} {se:>6} "
                  f"{s['over_ctx']:>5} {s['n_error']:>4} {s['n_unparsed']:>8}")

        if args.detail:
            for task, s in out["tasks"].items():
                if not s.get("groups"):
                    continue
                print(f"\n  [{task}]")
                for label, g in s["groups"].items():
                    print(f"    {label:<40} n={g['n']:>4}  acc={g['accuracy']:>6.2f}")

    print(f"\n表格 -> {table}\n明细 -> {out_dir}/<model>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
