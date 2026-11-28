"""命令行入口。

  python -m capbench list                    列出三个任务与官方口径
  python -m capbench build --tasks all       下载并按固定策略抽样
  python -m capbench check  --model X        检查数据、模板与端点
  python -m capbench eval   --model X        跑一遍 → pred/{model}.jsonl
  python -m capbench ifeval --model X        官方校验器跑 IFEval → results/ifeval-{model}.json
  python -m capbench report --model X        汇总 → result.txt + results/{model}.json
  python -m capbench compare --model X --baseline-revision base --revision fp8
                                             两个 revision 逐题配对 → Δpp / 翻转 / p

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
from . import consistency as consmod
from . import ifeval as ifevalmod
from .client import Client, Model
from .prompts import render
from .runner import (filter_length, generate, group_by, load_exemplars,
                     load_rows, summarize)
from .stats import paired
from .tasks import ORDER, TASKS, build_mmlu_exemplars, build_one

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 2026

# 各任务报告时的分组维度（对应数据卡里的 meta 字段）
GROUP_KEYS = {
    "ifeval": (),
    "mmlu_pro": ("category",),
    "longbench_v2": ("difficulty", "length"),
}

EXTERNAL_CMDS = {
    "ifeval": "python -m capbench ifeval --model {model}"
              "   （官方校验器由 lm_eval 提供，本命令封装了它的调用与结果合并）",
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
        if name == "ruler":
            kwargs["data_dir"] = data
            kwargs["repeat"] = args.ruler_repeat
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


# ---------------------------------------------------------------- ifeval

def cmd_ifeval(args):
    """跑官方校验器（lm_eval 子进程），结果写 results/ifeval-{model}.json。"""
    m = _model(args)
    results_dir = Path(args.out)
    mt = args.max_gen_toks or (m.max_tokens if m.lock_max_tokens else None)

    if args.from_results:
        run_dir = Path(args.from_results)
        if not run_dir.exists():
            sys.exit(f"找不到 {run_dir}")
        cli = ifevalmod.find_cli() or "lm-eval"
        print(f"复用已有输出：{run_dir}")
    else:
        cli = ifevalmod.find_cli()
        if not cli:
            sys.exit("没找到 lm_eval 命令。\n"
                     "  装：pip install lm-eval   （连带 torch，只为了跑 IFEval）\n"
                     "  已有输出可直接复用：--from-results <目录>")
        run_dir = results_dir / m.name / f"{time.strftime('%Y%m%d-%H%M%S')}-{m.revision}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"运行目录：{run_dir}")
        cmd = ifevalmod.build_cmd(cli, m, run_dir, concurrency=args.concurrency,
                                  batch_size=args.batch_size, limit=args.limit,
                                  max_gen_toks=mt)
        print("$ " + " ".join(cmd), flush=True)
        if args.dry_run:
            return 0
        try:
            ifevalmod.run(cmd)
        except ifevalmod.IfevalError as e:
            sys.exit(str(e))

    try:
        rec = ifevalmod.summarize(m, run_dir, cli, limit=args.limit)
    except ifevalmod.IfevalError as e:
        sys.exit(str(e))

    # 输出上限只信这一轮的记录。复用旧结果时尤其不能拿当前配置里的 max_tokens
    # 去填——旧结果可能只给了 1280。
    ifevalmod.apply_budget(rec["lm_eval"], from_results=bool(args.from_results),
                           explicit=args.max_gen_toks, passed=mt)

    p = ifevalmod.sidecar_path(ROOT / "results", m.name, m.revision)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    ifevalmod.print_run(rec)
    print(f"\n-> {p}")
    print(f"下一步：python result.py --model {m.name}")
    return 0


# ---------------------------------------------------------------- consistency

def cmd_consistency(args):
    """逐 token 对照两个部署/两种缓存状态，判据是 100% 相同。"""
    m = _model(args)
    task = args.consistency_task
    prompts = consmod._mkprompts(m, task, args.n, Path(args.data),
                                 Path(args.prompts), Path(args.splits),
                                 args.prompt_variant)
    print(f"{m.name}  任务={task}  题数={len(prompts)}  "
          f"max_tokens={m.max_tokens}  mode={args.mode}")

    if args.mode == "paint":
        mode_name = "单侧采样（不比较）"
        a = consmod.run(m, prompts)
        s = consmod.summarize(a, a, "run", "run")
        s["identical_rate"] = None
        print()
        print("只采样，不做一致性判定。")
    elif args.mode == "cold-warm":
        mode_name = "冷 → 热（同端点）"
        a = consmod.run(m, prompts)   # 冷：缓存为空
        b = consmod.run(m, prompts)   # 热：上一轮已把前缀灌进缓存
        s = consmod.summarize(a, b, "冷跑", "热跑")
        consmod.print_report(s)
    elif args.mode == "endpoints":
        if not args.compare_url:
            sys.exit("--mode endpoints 需要 --compare-url")
        mode_name = "端点 A → 端点 B"
        a = consmod.run(m, prompts)
        m2 = cfgmod.resolve(m.name, cfgmod.load(Path(args.config)),
                            base_url=args.compare_url, revision=args.revision,
                            max_tokens=args.max_tokens, thinking=args.thinking,
                            extra_body=args.extra_body)
        b = consmod.run(m2, prompts)
        s = consmod.summarize(a, b, args.base_url or m.base_url, m2.base_url)
        consmod.print_report(s)
    else:
        sys.exit(f"未知 mode {args.mode}")

    out = Path(args.out) if args.out else (
        ROOT / "results" / f"consistency-{m.name}-{args.mode}.json")
    payload = {"model": m.name, "task": task, "mode": args.mode,
               "mode_note": mode_name, "n": args.n,
               "max_tokens": m.max_tokens, "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "summary": {k: v for k, v in s.items() if k != "per_item"},
               "per_item": s.get("per_item")}
    consmod.write_json(out, payload)
    print()
    print(f"-> {out}")
    return 0 if s.get("identical_rate") in (None, 100.0) else 1


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

    # report 只读本地文件，模型没注册也要能出分
    try:
        m = _model(args)
    except KeyError:
        m = Model(name=args.model, revision=args.revision or "default")

    # 同一个 pred 文件里可能并存多个 revision（base / fp8 / int4）：
    # 不筛就会把几次运行混成一个分数，必须按 revision 取一份。
    revs = sorted({str(r.get("revision") or "default") for r in recs})
    rev = args.revision or m.revision
    if len(revs) > 1:
        if rev not in revs:
            sys.exit(f"{path} 里有多个 revision {revs}，用 --revision 指定其中一个")
        print(f"注意：{path} 含 {len(revs)} 个 revision {revs}，本次只统计 {rev}\n")
    elif revs:
        rev = revs[0]
    recs = [r for r in recs if str(r.get("revision") or "default") == rev]
    by_task: dict[str, list[dict]] = {}
    for r in recs:
        by_task.setdefault(r["task"], []).append(r)

    # IFEval 的判分在外部跑，结果按 model@revision 单独存一份 sidecar
    ext = ifevalmod.load_for(Path(args.out), args.model, rev)

    lines = ["task\tn\tscored\tacc\tse\tover\terr\tunparsed\tofficial%\tmaxlen\tmax_tokens"]
    print(f"{'task':<16}{'n':>5}{'scored':>8}{'acc':>9}{'se':>7}{'over':>6}"
          f"{'err':>5}{'unparsed':>10}{'官方%':>8}{'maxlen':>9}{'maxtok':>8}")
    print("-" * 93)
    summaries = []

    def emit(s):
        summaries.append(s)
        acc = "-" if s["accuracy"] is None else f"{s['accuracy']}"
        print(f"{s['task']:<16}{s['n']:>5}{s['n_scored']:>8}{acc:>9}{s['se_pp'] or '-':>7}"
              f"{s['over_ctx']:>6}{s['n_error']:>5}{s['n_unparsed']:>10}"
              f"{s['official_parse_rate'] or '-':>8}{s['maxlen']:>9}"
              f"{(s['decode'] or {}).get('max_tokens', '-'):>8}")
        if s["accuracy"] is not None:
            lines.append(f"{s['task']}\t{s['n']}\t{s['n_scored']}\t{s['accuracy']}\t"
                         f"{s['se_pp']}\t{s['over_ctx']}\t{s['n_error']}\t"
                         f"{s['n_unparsed']}\t"
                         f"{s['official_parse_rate'] if s['official_parse_rate'] is not None else '-'}\t"
                         f"{s['maxlen']}\t{(s['decode'] or {}).get('max_tokens')}")

    for name in _tasks(args.tasks):
        rs = by_task.get(name)
        if rs:
            emit(summarize(name, m, rs))
            continue
        if name == "ifeval" and ext:
            emit(ifevalmod.as_summary(ext, m.maxlen))
            continue
        print(f"{name:<16}{'-':>5}  无数据")

    if ext:
        mt = (ext.get("metrics") or {})
        lib = ext.get("lm_eval") or {}
        print(f"\nIFEval 由官方校验器判分（lm-eval {lib.get('version')}，"
              f"task v{lib.get('task_version')}）：prompt-strict {mt.get('prompt_strict')}% / "
              f"loose {mt.get('prompt_loose')}%；instruction-level strict "
              f"{mt.get('inst_strict')}% / loose {mt.get('inst_loose')}%")
        print("  表中 acc 取 prompt-level strict；IFEval 无超窗与解析失败概念，"
              "over / unparsed 恒为 0。")

    # 超窗题不进分母：比例一高，分数只代表跑通的那一段，必须当面说清
    thin = [s for s in summaries if s["n"] and s["over_ctx"] / s["n"] > 0.2]
    for s in thin:
        print(f"\n⚠ {s['task']}: {s['over_ctx']}/{s['n']} 题超出 maxlen={s['maxlen']}，"
              f"未参与计分；有效评分 {s['n_scored']} 题。"
              f"报数必须同时给出超窗数与 maxlen。")

    rr = [r for r in by_task.get("ruler", []) if "error" not in r
          and not r.get("skipped") and r.get("scored")]
    if rr:
        print()
        print("[ruler] 长度 × 位置 准确率（%）—— 看趋势，不要解读单格")
        cells: dict = {}
        for r in rr:
            md = r.get("meta") or {}
            key = (str(md.get("variant")), int(md.get("target_len") or 0))
            cells.setdefault(key, {}).setdefault(int(md.get("pos_pct") or 0),
                                                 []).append(bool(r["correct"]))
        depths = sorted({d for v in cells.values() for d in v})
        print(f"  {'variant':<10}{'len':>6}" + "".join(f"{f'{d}%':>8}" for d in depths)
              + f"{'合计':>9}")
        by_len: dict = {}
        for (var, L), per in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            row, allx = [], []
            for d in depths:
                xs = per.get(d) or []
                allx += xs
                row.append(f"{sum(xs) / len(xs) * 100:>7.0f}" if xs else f"{'-':>8}")
            by_len.setdefault(L, []).extend(allx)
            print(f"  {var:<10}{L // 1024:>5}k" + "".join(row)
                  + f"{sum(allx) / len(allx) * 100:>8.0f}")
        print("  按长度聚合：" + "   ".join(
            f"{L // 1024}k {sum(v) / len(v) * 100:.0f}%"
            for L, v in sorted(by_len.items())))
        mins = min(len(v) for per in cells.values() for v in per.values())
        print(f"  衰减判据：相邻长度档的掉幅是否显著大于基线；"
              f"单格仅 {mins} 题，别拿 0%/100% 当结论。")

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

    table_path = Path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json = Path(args.out) / f"{args.model}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "model": m.name, "revision": m.revision, "endpoint": m.base_url,
        "maxlen": m.maxlen, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n表格 -> {args.table}\n明细 -> {out_json}")
    return 0


# ---------------------------------------------------------------- compare

def _side(path: Path, rev: str | None, flag: str) -> tuple[list[dict], str]:
    if not path.exists():
        sys.exit(f"找不到 {path}，先跑 pred.py")
    recs = _load(path)
    revs = sorted({str(r.get("revision") or "default") for r in recs})
    if not rev:
        if len(revs) != 1:
            sys.exit(f"{path} 里有多个 revision {revs}，用 --{flag}revision 指定")
        rev = revs[0]
    got = [r for r in recs if str(r.get("revision") or "default") == rev]
    if not got:
        sys.exit(f"{path} 里没有 revision={rev} 的记录（现有 {revs}）")
    return got, rev


def cmd_compare(args):
    """两个 revision（或两个模型）逐题配对比较。

    基线默认取同一个 pred 文件里另一个 revision：base 与 fp8 都写在
    pred/{model}.jsonl，各任务各自按 item_id 配对。
    """
    pred = Path(args.pred)
    new_path = pred / f"{args.model}.jsonl" if pred.is_dir() else pred
    base_model = args.baseline_model or args.model
    base_path = (Path(args.baseline_pred) if args.baseline_pred
                 else (pred / f"{base_model}.jsonl" if pred.is_dir() else pred))

    new, new_rev = _side(new_path, args.revision, "revision")
    base, base_rev = _side(base_path, args.baseline_revision, "baseline-")
    if (base_model, base_rev) == (args.model, new_rev) and base_path == new_path:
        sys.exit("基线和本次是同一份记录，换一个 --baseline-revision 或 --baseline-model")

    print(f"基线 {base_model}@{base_rev}  ←→  本次 {args.model}@{new_rev}\n")
    print(f"{'task':<16}{'paired':>7}{'base':>8}{'new':>8}{'Δpp':>8}"
          f"{'翻转':>10}{'p':>9}")
    print("-" * 66)
    rows = []
    for name in _tasks(args.tasks):
        b = [r for r in base if r["task"] == name]
        n = [r for r in new if r["task"] == name]
        if not b or not n:
            print(f"{name:<16}{'-':>7}  一侧无数据")
            continue
        s = paired(b, n)
        if not s.get("n_paired"):
            print(f"{name:<16}{0:>7}  没有可配对的题（两边都判分的交集为空）")
            continue
        flip = f"{s['flip_good']}+/{s['flip_bad']}-"
        print(f"{name:<16}{s['n_paired']:>7}{s['acc_base']:>8.2f}{s['acc_new']:>8.2f}"
              f"{s['delta_pp']:>+8.2f}{flip:>10}{s['mcnemar_p']:>9.3f}")
        rows.append({"task": name, **s})

    if rows:
        print("\nΔpp 为正表示本次更好；翻转为「对→错 / 错→对」的题数，"
              "p 是 McNemar 精确检验（双侧）。")
        out = Path(args.out) if args.out else (
            ROOT / "results" / f"compare-{args.model}@{new_rev}-vs-{base_model}@{base_rev}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "model": args.model, "revision": new_rev,
            "baseline_model": base_model, "baseline_revision": base_rev,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tasks": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细 -> {out}")
    else:
        print("\n没有任何任务可比。")
    return 0


# ---------------------------------------------------------------- main

def _utf8_stdout() -> None:
    """Windows 控制台默认 GBK，遇到 ⚠ / ✓ 会抛 UnicodeEncodeError 把整批中断。

    改成 utf-8 + errors=replace：能显示就显示，不能显示就替换成 ?，
    绝不让一个提示符弄死一次跑了半小时的评测。
    """
    for s_ in (sys.stdout, sys.stderr):
        try:
            s_.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv=None):
    _utf8_stdout()
    ap = argparse.ArgumentParser(prog="capbench", description="CapBench —— 三任务能力评测集")
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
    b.add_argument("--ruler-repeat", type=int, default=1,
                   help="ruler 每格重复次数；出曲线建议 4")
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

    f = sub.add_parser("ifeval", help="用官方校验器跑 IFEval（外部 lm_eval）")
    common(f)
    f.add_argument("--out", default=str(ROOT / "results" / "lm_eval"),
                   help="lm_eval 输出目录")
    f.add_argument("--concurrency", type=int, default=8)
    f.add_argument("--batch-size", type=int, default=8)
    f.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟）")
    f.add_argument("--max-gen-toks", type=int, default=None,
                   help="覆盖官方 1280；思考模型需要调大")
    f.add_argument("--from-results", default=None,
                   help="不运行，直接解析已有的 lm_eval 输出目录")
    f.add_argument("--dry-run", action="store_true", help="只打印命令")
    f.set_defaults(fn=cmd_ifeval)

    cc = sub.add_parser("consistency", help="逐 token 对照缓存路径 / PD 传输")
    common(cc)
    cc.add_argument("--mode", choices=["cold-warm", "endpoints", "paint"],
                    default="cold-warm")
    cc.add_argument("--compare-url", default=None, help="endpoints 模式的第二个端点")
    cc.add_argument("--n", type=int, default=40)
    cc.add_argument("--out", default=None)
    # 复用已有参数名会与 common 冲突，这里单独命名
    cc.add_argument("--consistency-task", dest="consistency_task",
                    default="mmlu_pro",
                    choices=["mmlu_pro", "longbench_v2", "ruler"])
    cc.set_defaults(fn=cmd_consistency)

    r = sub.add_parser("report", help="汇总 pred/ 出表格")
    common(r)
    r.add_argument("--pred", default=str(ROOT / "pred"))
    r.add_argument("--out", default=str(ROOT / "results"))
    r.add_argument("--table", default=str(ROOT / "result.txt"))
    r.add_argument("--detail", action="store_true")
    r.set_defaults(fn=cmd_report)

    p = sub.add_parser("compare", help="两个 revision / 两个模型逐题配对比较")
    common(p)
    p.add_argument("--pred", default=str(ROOT / "pred"))
    p.add_argument("--baseline-revision", default=None,
                   help="基线 revision。与 --revision 之外的另一份记录，如 base")
    p.add_argument("--baseline-model", default=None, help="基线模型名，默认同 --model")
    p.add_argument("--baseline-pred", default=None, help="基线 pred 文件或目录")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_compare)

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
