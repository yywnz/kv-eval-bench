"""命令行入口。

  python -m capbench list                    列出四个任务与官方口径
  python -m capbench build --tasks all       下载并按固定策略抽样
  python -m capbench check  --model X        检查数据、模板与端点
  python -m capbench eval   --model X        跑一遍 → pred/{model}.jsonl
  python -m capbench report --model X        汇总 → result.txt + results/{model}.json

日常用 `pred.py` / `result.py`（两个薄包装，参数同名）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from . import config as cfgmod
from .client import Client, Model
from .prompts import render
from .runner import (filter_length, generate, group_by, load_exemplars,
                     load_rows, summarize)
from .tasks import ORDER, TASKS, build_mmlu_exemplars, build_one

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 2026

# 各任务报告时的分组维度（对应数据卡里的 meta 字段）
GROUP_KEYS = {
    "ifeval": (),
    "mmlu_pro": ("category",),
    "gpqa_diamond": ("domain",),
    "longbench_v2": ("difficulty", "length"),
}

EXTERNAL_CMDS = {
    "ifeval": "lm_eval --model local-completions --model_args "
              "base_url={base}/v1,model={model},tokenized_requests=False "
              "--tasks ifeval --num_fewshot 0 --batch_size 32",
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
        if t and t not in out:
            out.append(t)
    bad = [t for t in out if t not in TASKS]
    if bad:
        sys.exit(f"未知任务 {bad}，可选：{', '.join(ORDER)}")
    return out


def _prompt_fn(task: str, variant: str, prompt_dir: Path, exemplars: dict):
    shape = TASKS[task].shape
    return lambda row: render(row, shape, variant, prompt_dir, exemplars)


# ---------------------------------------------------------------- list

def cmd_list(args):
    print(f"CapBench v{__version__}\n")
    print(f"{'task':<16} {'维度':<10} {'题数':>5} {'判分':<9} 官方指标")
    print("-" * 96)
    total = 0
    for name in ORDER:
        t = TASKS[name]
        sp = Path(args.splits) / f"{name}.json"
        n = json.loads(sp.read_text(encoding="utf-8")).get("n", "") if sp.exists() else ""
        total += n if isinstance(n, int) else 0
        print(f"{name:<16} {t.dimension:<10} {n:>5} {t.judge:<9} {t.metric}")
    print("-" * 96)
    print(f"合计 {total} 题。模板在 prompts/，模型在 config/models.json，"
          f"口径在 docs/tasks/<task>.md")
    return 0


# ---------------------------------------------------------------- build

def cmd_build(args):
    data, splits = Path(args.data), Path(args.splits)
    data.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)

    manifest = {"suite": "capbench", "version": __version__, "schema": 3,
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

    if "mmlu_pro" in manifest["tasks"]:
        ex = build_mmlu_exemplars(args.seed)
        ep = splits / "exemplars_mmlu_pro.json"
        ep.write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest["mmlu_exemplars"] = {"file": ep.name, "categories": len(ex),
                                      "per_category": args.exemplars,
                                      "sha256_16": _sha16(ep)}
        print(f"\n5-shot 示例 {len(ex)} 类 × {args.exemplars} 条 -> {ep}")

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
    print(f"{'task':<16} {'shape':<9} {'数据':<22} {'模板':<22} 状态")
    print("-" * 96)
    ex = load_exemplars(Path(args.splits))
    missing = []
    for name in _tasks(args.tasks):
        t = TASKS[name]
        p = Path(args.data) / f"{name}.jsonl"
        try:
            tpl = render({"question": "", "options": [], "context": "",
                          "meta": {"category": "physics"}},
                         t.shape, args.prompt_variant, Path(args.prompts), ex)[1]
        except Exception as e:
            tpl = f"!{type(e).__name__}"
        if p.exists():
            n = sum(1 for _ in p.open(encoding="utf-8"))
            status = f"OK ({n})" + ("  [judge: external]" if t.judge == "external" else "")
        else:
            status = "MISSING" if t.judge == "local" else "EXTERNAL"
            if t.judge == "local":
                missing.append(name)
        print(f"{name:<16} {t.shape:<9} {p.name:<22} {tpl:<22} {status}")
    if not args.no_health:
        ok = Client(m).health()
        print(f"\nendpoint health: {'OK' if ok else 'UNREACHABLE'}")
    if missing:
        print(f"\n缺少数据：{', '.join(missing)}  "
              f"生成：python -m capbench build --tasks {','.join(missing)}")
    ext = [t for t in _tasks(args.tasks) if TASKS[t].judge == "external"]
    if ext:
        print("\n本仓库不判分的任务（用官方 verifier）：")
        for e in ext:
            print(f"  {e:<15} " + EXTERNAL_CMDS.get(e, "").format(
                base=m.base_url.rstrip("/"), model=m.name))
    return 0


# ---------------------------------------------------------------- eval

def cmd_eval(args):
    m = _model(args)
    out_dir = Path(args.out)
    ex = load_exemplars(Path(args.splits))
    st = (out_dir / f"{m.name}.jsonl")
    st.parent.mkdir(parents=True, exist_ok=True)
    n_new = 0

    for name in _tasks(args.tasks):
        if TASKS[name].judge == "external":
            print(f"[{name}] 判分需外部 verifier，跳过（check 里可打印命令）")
            continue
        rows = filter_length(load_rows(Path(args.data), name), args.length)
        print(f"[{name}] n={len(rows) if not args.limit else min(len(rows), args.limit)}")
        recs = generate(m, name, rows,
                        _prompt_fn(name, args.prompt_variant, Path(args.prompts), ex),
                        concurrency=args.concurrency, limit=args.limit,
                        on_progress=lambda d, t: print(f"  {d}/{t}", end="\r", flush=True))
        print()
        s = summarize(name, m, recs)
        acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']}%"
        print(f"  acc={acc:<8} n_scored={s['n_scored']}/{s['n']}  err={s['n_error']}  "
              f"超窗={s['over_ctx']}  截断={s['n_truncated']}  未解析={s['n_unparsed']}  "
              f"官方抽取={s['official_parse_rate']}%")
        if s["n"] and s["over_ctx"] / s["n"] > 0.2:
            print(f"  ⚠ {s['over_ctx']}/{s['n']} 题超出 maxlen={s['maxlen']}，未参与计分；"
                  f"分数只代表跑通的那部分，报数时必须一并给出超窗数")
        _upsert(st, recs)
        n_new += len(recs)

    print(f"\n已写入 {n_new} 条 -> {st}")
    print(f"下一步：python result.py --model {m.name}")
    return 0


def _upsert(path: Path, recs: list[dict]) -> None:
    """按 task+revision 整体替换，其他任务/版本保留。"""
    keep: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if (r.get("task"), r.get("revision")) not in {
                        (x["task"], x["revision"]) for x in recs}:
                    keep.append(r)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in keep + recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.move(tmp, path)


# ---------------------------------------------------------------- report

def _load(p: Path) -> list[dict]:
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_report(args):
    src = Path(args.pred)
    path = src / f"{args.model}.jsonl" if src.is_dir() else src
    if not path.exists():
        sys.exit(f"找不到 {path}，先跑 eval/pred.py")
    recs = _load(path)
    by_task: dict[str, list[dict]] = {}
    for r in recs:
        by_task.setdefault(r["task"], []).append(r)

    # report 只读本地文件，模型没注册也要能出分
    try:
        m = _model(args)
    except KeyError:
        m = Model(name=args.model, revision=args.revision or "default")
    lines = ["task\tn\tscored\tacc\tse\tover\terr\tunparsed\tofficial%\tmax_tokens"]
    print(f"{'task':<16}{'n':>5}{'scored':>8}{'acc':>9}{'se':>7}{'over':>6}"
          f"{'err':>5}{'unparsed':>10}{'官方%':>8}")
    print("-" * 84)
    summaries = []
    for name in _tasks(args.tasks):
        rs = by_task.get(name)
        if not rs:
            print(f"{name:<16}{'-':>5}  无数据")
            continue
        s = summarize(name, m, rs)
        summaries.append(s)
        acc = "-" if s["accuracy"] is None else f"{s['accuracy']}"
        print(f"{name:<16}{s['n']:>5}{s['n_scored']:>8}{acc:>9}{s['se_pp'] or '-':>7}"
              f"{s['over_ctx']:>6}{s['n_error']:>5}{s['n_unparsed']:>10}"
              f"{s['official_parse_rate'] or '-':>8}")
        if s["accuracy"] is not None:
            lines.append(f"{name}\t{s['n']}\t{s['n_scored']}\t{s['accuracy']}\t"
                         f"{s['se_pp']}\t{s['over_ctx']}\t{s['n_error']}\t"
                         f"{s['n_unparsed']}\t{s['official_parse_rate']}\t"
                         f"{(s['decode'] or {}).get('max_tokens')}")

    if args.detail:
        for s in summaries:
            keys = GROUP_KEYS.get(s["task"], ())
            if not keys:
                continue
            g = group_by(by_task[s["task"]], keys)
            if len(g) <= 1:
                continue
            print(f"\n[{s['task']}] 按 {'/'.join(keys)} 分组")
            for label, v in g.items():
                print(f"  {label:<40}{v['n']:>5}  {v['accuracy']:>6.2f}%")

    Path(args.table).write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json = Path(args.out) / f"{args.model}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "model": m.name, "revision": m.revision, "endpoint": m.base_url,
        "maxlen": m.maxlen, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n表格 -> {args.table}\n明细 -> {out_json}")
    return 0


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="capbench", description="CapBench —— 四任务能力评测集")
    ap.add_argument("--version", action="version", version=f"capbench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model=True):
        p.add_argument("--tasks", default=None, help="逗号分隔，默认 all")
        p.add_argument("--data", default=str(ROOT / "data"))
        p.add_argument("--splits", default=str(ROOT / "splits"))
        p.add_argument("--prompts", default=str(ROOT / "prompts"))
        p.add_argument("--prompt-variant", default="5shot",
                       help="mmlu: 5shot/0shot；longctx: 0shot/cot/no_context")
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
    b.add_argument("--limit", type=int, default=None, help="mmlu 题量上限")
    b.add_argument("--per-cell", type=int, default=14, help="longbench 每格上限")
    b.add_argument("--exemplars", type=int, default=5, help="mmlu 每类 few-shot 条数")
    b.add_argument("--fresh", action="store_true")
    b.set_defaults(fn=cmd_build)

    k = sub.add_parser("check", help="检查数据、模板与端点")
    common(k)
    k.add_argument("--no-health", action="store_true")
    k.set_defaults(fn=cmd_check)

    e = sub.add_parser("eval", help="跑一遍并写 pred/")
    common(e)
    e.add_argument("--out", default=str(ROOT / "pred"))
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--concurrency", type=int, default=16)
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("report", help="汇总 pred/ 出表格")
    common(r)
    r.add_argument("--pred", default=str(ROOT / "pred"))
    r.add_argument("--out", default=str(ROOT / "results"))
    r.add_argument("--table", default=str(ROOT / "result.txt"))
    r.add_argument("--detail", action="store_true")
    r.set_defaults(fn=cmd_report)

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
