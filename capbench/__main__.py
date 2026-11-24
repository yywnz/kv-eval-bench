"""命令行入口（数据构建 + 回归对比）。

日常评测走 `pred.py` → `result.py`（两段式，换统计口径不用重跑模型）；
这里的 `run/baseline/compare` 用于「改动前后配对比较」，会带时间戳落盘。

  python -m capbench list
  python -m capbench build --tasks all
  python -m capbench check --model deepseek-v4.1
  python -m capbench baseline --model deepseek-v4.1 --tasks all --revision base
  python -m capbench run      --model deepseek-v4.1 --tasks all --revision fp8
  python -m capbench compare  --model deepseek-v4.1 --revision fp8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import __version__
from . import config as cfgmod
from .client import Client, Model
from .prompts import render
from .runner import filter_length, generate, load_rows, summarize
from .stats import paired
from .tasks import ORDER, TASKS, build_one

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 2026

EXTERNAL_CMDS = {
    "ifeval": "lm_eval --model local-completions --model_args "
              "base_url={base}/v1,model={model},tokenized_requests=False "
              "--tasks ifeval --num_fewshot 0 --batch_size 32",
    "livecodebench": "lm_eval --model local-completions --model_args "
                     "base_url={base}/v1,model={model},tokenized_requests=False "
                     "--tasks livecodebench --num_fewshot 0 --batch_size 8",
}


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _model(args) -> Model:
    return cfgmod.resolve(
        args.model, cfgmod.load(Path(getattr(args, "config", None) or cfgmod.DEFAULT_PATH)),
        base_url=args.base_url, revision=args.revision,
        max_tokens=getattr(args, "max_tokens", None),
        thinking=getattr(args, "thinking", None),
        extra_body=getattr(args, "extra_body", None))


def _tasks(spec: str | None) -> list[str]:
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


def _prompt_fn(task: str, variant: str, prompt_dir: Path):
    shape = TASKS[task].shape
    return lambda row: render(row, shape, variant, prompt_dir)


# ---------------------------------------------------------------- list

def cmd_list(args):
    print(f"CapBench v{__version__}\n")
    print(f"{'task':<16} {'维度':<10} {'shape':<9} {'题数':>5} {'判分':<9} 指标")
    print("-" * 92)
    for name in ORDER:
        t = TASKS[name]
        sp = Path(args.splits) / f"{name}.json"
        n = json.loads(sp.read_text(encoding="utf-8")).get("n", "") if sp.exists() else ""
        print(f"{name:<16} {t.dimension:<10} {t.shape:<9} {n:>5} {t.judge:<9} {t.metric}")
    print("\n模板在 prompts/，模型在 config/models.json，说明在 docs/tasks/<task>.md")
    return 0


# ---------------------------------------------------------------- build

def cmd_build(args):
    data, splits = Path(args.data), Path(args.splits)
    data.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)

    manifest = {"suite": "capbench", "version": __version__, "schema": 2,
                "seed": args.seed, "built_at": time.strftime("%Y-%m-%d"), "tasks": {}}
    mpath = splits / "manifest.json"
    if mpath.exists() and not args.fresh:
        manifest["tasks"] = json.loads(mpath.read_text(encoding="utf-8")).get("tasks", {})

    for name in _tasks(args.tasks):
        print(f"\n=== {name}")
        t0 = time.time()
        kwargs = {}
        if args.limit:
            kwargs["limit"] = args.limit
        if name == "longbench_v2":
            kwargs["per_cell"] = args.per_cell
        try:
            rows, meta = build_one(name, args.seed, **kwargs)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {e}")
            continue
        if not rows:
            print("  空结果，跳过")
            continue
        p = data / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sp = {"task": name, "seed": args.seed, "n": len(rows), "sha256_16": _sha16(p),
              "ids": [r["id"] for r in rows]}
        (splits / f"{name}.json").write_text(json.dumps(sp, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
        meta.update({"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2),
                     "sha256_16": sp["sha256_16"], "build_seconds": round(time.time() - t0, 1)})
        manifest["tasks"][name] = meta
        print(f"  {len(rows):>4} 题  {meta['size_mb']:>7.2f} MB  {sp['sha256_16']}  "
              f"{time.time() - t0:.0f}s")

    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tot = sum(v["n"] for v in manifest["tasks"].values())
    size = sum(v["size_mb"] for v in manifest["tasks"].values())
    print(f"\n合计 {tot} 题 / {size:.1f} MB  -> {mpath}")
    return 0


# ---------------------------------------------------------------- check

def cmd_check(args):
    m = _model(args)
    print(f"capbench v{__version__}    model={m.name}    revision={m.revision}")
    print(f"endpoint={m.base_url}    max_tokens={m.max_tokens}    maxlen={m.maxlen}")
    print(f"extra_body={m.extra_body or '-'}    prompt_variant={args.prompt_variant}\n")
    print(f"{'task':<16} {'shape':<9} {'数据':<22} {'模板':<20} 状态")
    print("-" * 92)
    missing = []
    for name in _tasks(args.tasks):
        t = TASKS[name]
        p = Path(args.data) / f"{name}.jsonl"
        try:
            tpl = render({"question": "", "options": [], "context": ""},
                         t.shape, args.prompt_variant, Path(args.prompts))[1]
        except Exception as e:
            tpl = f"!{type(e).__name__}"
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8"))
            status = f"OK ({n})" + ("  [judge: external]" if t.judge == "external" else "")
        else:
            status = "MISSING" if t.judge == "local" else "EXTERNAL"
            if t.judge == "local":
                missing.append(name)
        print(f"{name:<16} {t.shape:<9} {p.name:<22} {tpl:<20} {status}")
    if not args.no_health:
        print(f"\nendpoint health: {'OK' if Client(m).health() else 'UNREACHABLE'}")
    if missing:
        print(f"\n缺少数据：{', '.join(missing)}  "
              f"生成：python -m capbench build --tasks {','.join(missing)}")
    ext = [t for t in _tasks(args.tasks) if TASKS[t].judge == "external"]
    if ext:
        print("\n本地不判分的任务（用官方 verifier）：")
        for e in ext:
            print(f"  {e:<15} " + EXTERNAL_CMDS.get(e, "").format(
                base=m.base_url.rstrip("/"), model=m.name))
    return 0


# ---------------------------------------------------------------- run / baseline

def cmd_run(args):
    m = _model(args)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summaries = []
    for name in _tasks(args.tasks):
        rows = filter_length(load_rows(Path(args.data), name), args.length)
        out = Path(args.out) / m.name / m.revision / name / stamp
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] n={len(rows) if not args.limit else min(len(rows), args.limit)} -> {out}")
        recs = generate(m, name, rows,
                        _prompt_fn(name, args.prompt_variant, Path(args.prompts)),
                        concurrency=args.concurrency, limit=args.limit)
        with (out / "raw.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize(name, m, recs)
        (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        summaries.append(s)
        acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']}%"
        print(f"  acc={acc:<8} n_scored={s['n_scored']}/{s['n']}  err={s['n_error']}  "
              f"超窗={s['over_ctx']}  截断={s['n_truncated']}  未解析={s['n_unparsed']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summaries, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    return 0


def cmd_baseline(args):
    m = _model(args)
    bdir = Path(args.baseline_dir) / m.name / m.revision
    bdir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in _tasks(args.tasks):
        rows = filter_length(load_rows(Path(args.data), name), args.length)
        recs = generate(m, name, rows,
                        _prompt_fn(name, args.prompt_variant, Path(args.prompts)),
                        concurrency=args.concurrency, limit=args.limit)
        with (bdir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize(name, m, recs)
        summaries.append(s)
        acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']}%"
        print(f"[{name}] baseline acc={acc:<8} n={s['n']}  -> {bdir / (name + '.jsonl')}")
    (bdir / "meta.json").write_text(json.dumps({
        "model": m.name, "revision": m.revision, "endpoint": m.base_url,
        "max_tokens": m.max_tokens, "maxlen": m.maxlen, "extra_body": m.extra_body,
        "prompt_variant": args.prompt_variant, "tasks": summaries,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- compare

def _latest(out_root: Path, model: str, rev: str, task: str) -> Path | None:
    d = out_root / model / rev / task
    if not d.exists():
        return None
    subs = sorted(p for p in d.iterdir() if p.is_dir())
    return subs[-1] if subs else None


def _load(p: Path) -> list[dict]:
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_compare(args):
    model = args.model
    rev = args.revision or "default"
    # 基线可以和本次用不同的 revision（例如 base vs fp8、cold vs warm）
    brev = args.baseline_revision or rev
    bdir = Path(args.baseline_dir) / model / brev
    print(f"基线 {bdir}")
    print(f"本次 {Path(args.out) / model / rev}\n")
    print(f"{'task':<16} {'base':>7} {'new':>7} {'Δpp':>7} {'bad':>4} {'good':>5} "
          f"{'p':>7} {'margin%':>8}")
    print("-" * 78)
    rows_out = []
    for name in _tasks(args.tasks):
        bp = bdir / f"{name}.jsonl"
        if not bp.exists():
            print(f"{name:<16} {'-':>7} {'-':>7}  无基线")
            continue
        run_dir = Path(args.run) if args.run else _latest(Path(args.out), model, rev, name)
        if not run_dir or not (run_dir / "raw.jsonl").exists():
            print(f"{name:<16} {'-':>7} {'-':>7}  无本次结果")
            continue
        c = paired(_load(bp), _load(run_dir / "raw.jsonl"))
        if not c.get("n_paired"):
            print(f"{name:<16} {'-':>7} {'-':>7}  无可配对题目")
            continue
        ms = "-" if c["margin_shrink_pct"] is None else f"{c['margin_shrink_pct']:+.1f}"
        print(f"{name:<16} {c['acc_base']:>7.2f} {c['acc_new']:>7.2f} {c['delta_pp']:>+7.2f} "
              f"{c['flip_bad']:>4} {c['flip_good']:>5} {c['mcnemar_p']:>7.4f} {ms:>8}")
        rows_out.append({"task": name, **c})
    if rows_out:
        print("\nΔpp 为百分点；bad=基线对本次错；good=反向；p 为 McNemar 双侧；"
              "margin% 为正表示正确项领先幅度收窄")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(rows_out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return 0


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="capbench", description="CapBench —— 六维能力基准")
    ap.add_argument("--version", action="version", version=f"capbench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model=True):
        p.add_argument("--tasks", default=None, help="逗号分隔，默认 all")
        p.add_argument("--data", default=str(ROOT / "data"))
        p.add_argument("--splits", default=str(ROOT / "splits"))
        p.add_argument("--prompts", default=str(ROOT / "prompts"))
        p.add_argument("--prompt-variant", default="0shot")
        if model:
            p.add_argument("--model", required=True)
            p.add_argument("--config", default=str(cfgmod.DEFAULT_PATH))
            p.add_argument("--base-url", default=None)
            p.add_argument("--revision", default=None)
            p.add_argument("--max-tokens", type=int, default=None)
            p.add_argument("--thinking", choices=["on", "off"], default=None)
            p.add_argument("--extra-body", default=None)
            p.add_argument("--length", default=None, help="只跑指定长度档 short,medium")

    c = sub.add_parser("list", help="列出任务与划分")
    common(c, model=False)
    c.set_defaults(fn=cmd_list)

    b = sub.add_parser("build", help="下载并按固定策略抽样")
    common(b, model=False)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--per-cell", type=int, default=14)
    b.add_argument("--fresh", action="store_true")
    b.set_defaults(fn=cmd_build)

    k = sub.add_parser("check", help="检查数据、模板与端点")
    common(k)
    k.add_argument("--no-health", action="store_true")
    k.set_defaults(fn=cmd_check)

    r = sub.add_parser("run", help="跑一轮（带时间戳，供 compare 用）")
    common(r)
    r.add_argument("--out", default=str(ROOT / "results"))
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--json-out", default=None)
    r.set_defaults(fn=cmd_run)

    bl = sub.add_parser("baseline", help="存基线快照")
    common(bl)
    bl.add_argument("--baseline-dir", default=str(ROOT / "baselines"))
    bl.add_argument("--limit", type=int, default=None)
    bl.add_argument("--concurrency", type=int, default=16)
    bl.set_defaults(fn=cmd_baseline)

    cp = sub.add_parser("compare", help="与基线配对比较")
    common(cp)
    cp.add_argument("--out", default=str(ROOT / "results"))
    cp.add_argument("--baseline-dir", default=str(ROOT / "baselines"))
    cp.add_argument("--run", default=None)
    cp.add_argument("--baseline-revision", default=None,
                    help="基线所在的 revision，默认与 --revision 相同")
    cp.add_argument("--json-out", default=None)
    cp.set_defaults(fn=cmd_compare)

    args = ap.parse_args(argv)
    try:
        return args.fn(args) or 0
    except FileNotFoundError as e:
        sys.exit(f"错误：{e}")
    except KeyError as e:
        sys.exit(f"错误：{e.args[0] if e.args else e}")
    except KeyboardInterrupt:
        sys.exit("已中断")


if __name__ == "__main__":
    sys.exit(main())
