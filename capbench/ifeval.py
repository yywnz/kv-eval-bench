"""IFEval 判分：调用官方校验器，本仓库不重写约束检查。

24 类约束的检查代码来自 `lm-eval` 的 ifeval 任务（上游
`google-research/instruction_following_eval`，Apache-2.0）。这里只做三件事：

1. 把 `config/models.json` 里的端点拼成 `lm_eval` 的命令；
2. 跑子进程，把输出收进 `results/lm_eval/{model}/`；
3. 把四个指标整理成与其余三项同形的汇总，写 `results/ifeval-{model}.json`，
   供 `result.py` 合并进同一张表。

    python -m capbench ifeval --model glm-flash
    python -m capbench ifeval --model glm-flash --dry-run        # 只打印命令
    python -m capbench ifeval --model glm-flash --from-results results/lm_eval/glm-flash
    python result.py --model glm-flash                          # 自动带上 ifeval

为什么不用 `python -m lm_eval`：判断器要单独装（连带 torch），可能不在当前解释器里，
而命令行入口是独立进程，能用就行。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

TASK = "ifeval"
CLI_NAMES = ("lm-eval", "lm_eval")

# lm_eval 的指标名 → 本仓库的叫法
METRICS = {
    "prompt_level_strict_acc": "prompt_strict",
    "prompt_level_loose_acc": "prompt_loose",
    "inst_level_strict_acc": "inst_strict",
    "inst_level_loose_acc": "inst_loose",
}


class IfevalError(RuntimeError):
    """跑不动官方校验器（没装 / 参数不对 / 端点不可达）。"""


# ---------------------------------------------------------------- 环境探测

def find_cli() -> str | None:
    for name in CLI_NAMES:
        p = shutil.which(name)
        if p:
            return p
    return None


def _sibling_pythons(exe: Path) -> list[Path]:
    """CLI 所在环境里的解释器（venv 的 Scripts/ 或 bin/）。"""
    cands = [exe.parent / "python.exe", exe.parent / "python",
             exe.parent / "python3",
             exe.parent.parent / "bin" / "python",
             exe.parent.parent / "bin" / "python3"]
    return [c for c in cands if c.exists()]


def _site_roots(exe: Path) -> list[Path]:
    """CLI 所在环境的 site-packages 候选。"""
    venv = exe.parent.parent
    roots = [venv / "Lib" / "site-packages", venv / "lib" / "site-packages"]
    roots += sorted((venv / "lib").glob("python3.*/site-packages")) if (venv / "lib").exists() else []
    return [r for r in roots if r.exists()]


def cli_version(cli: str) -> str:
    """取**被调用那个 CLI 所在环境**的 lm-eval 版本。

    不能读当前解释器的包元数据：`lm_eval` 命令可能属于另一个环境（本仓库正是
    这种情况——CapBench 跑在 A 环境，`lm_eval` 装在 B 环境），那样记下的版本号
    是错的，而版本号正是判断「分数能否并列」的依据。取不到就写 `unknown`，
    宁可空着也不要填错的。
    """
    exe = Path(cli).resolve()

    # ① 用 CLI 自己的解释器问它自己
    for py in _sibling_pythons(exe):
        try:
            out = subprocess.run(
                [str(py), "-c",
                 "import importlib.metadata as m; print(m.version('lm-eval'))"],
                capture_output=True, text=True, timeout=30)
            got = (out.stdout or "").strip()
            if out.returncode == 0 and got:
                return got
        except Exception:
            continue

    # ② 退一步：读 CLI 所在环境的 dist-info 目录名
    for root in _site_roots(exe):
        for d in root.glob("lm_eval-*.dist-info"):
            ver = d.name.split("-")[1].replace(".dist", "")
            if ver:
                return ver

    # ③ 不猜
    return "unknown"


def task_yaml(cli: str) -> Path | None:
    """从 `lm_eval ls tasks` 的输出里定位 ifeval.yaml，用于记录口径版本。"""
    try:
        out = subprocess.run([cli, "ls", "tasks"], capture_output=True, text=True,
                             timeout=180)
    except Exception:
        return None
    for line in out.stdout.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0] == TASK:
            p = Path(cells[1])
            if p.exists():
                return p
    return None


def yaml_meta(path: Path | None) -> dict:
    if not path:
        return {"file": None, "version": "unknown", "sha256_16": None}
    raw = path.read_bytes()
    text = raw.decode("utf-8", "ignore")
    ver = "unknown"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("version:"):
            ver = s.split(":", 1)[1].strip()
    return {"file": str(path), "version": ver,
            "sha256_16": hashlib.sha256(raw).hexdigest()[:16]}


# ---------------------------------------------------------------- 命令与运行

def build_cmd(cli: str, model, out_dir: Path, concurrency: int = 8,
              batch_size: int = 8, limit: int | None = None,
              max_gen_toks: int | None = None) -> list[str]:
    """拼 lm_eval 调用。

    `tokenizer_backend=none` 是必须的：默认会去 HF 拉同名的 tokenizer，
    端点上的模型名（`glm-flash` 这类）不在 HF 上，直接报
    `OSError: Can't load the configuration of ...`。IFEval 只需要按题面生成，
    不需要本地分词器。
    """
    base = model.base_url.rstrip("/")
    margs = ",".join([
        f"base_url={base}/v1",
        f"model={model.name}",
        "tokenized_requests=False",
        "tokenizer_backend=none",
        f"num_concurrent={concurrency}",
    ])
    cmd = [cli, "--model", "local-completions", "--model_args", margs,
           "--tasks", TASK, "--num_fewshot", "0",
           "--batch_size", str(batch_size),
           "--output_path", str(out_dir), "--log_samples"]
    if max_gen_toks:
        # 官方 yaml 是 1280；思考模型要在思考阶段之后还能写出答案
        cmd += ["--gen_kwargs", f"max_gen_toks={int(max_gen_toks)},temperature=0"]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def run(cmd: list[str], timeout: float | None = None) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        blob = (proc.stderr or proc.stdout or "").strip()
        tail = blob.splitlines()[-12:]
        msg = f"lm_eval 失败（退出码 {proc.returncode}）：\n" + "\n".join(tail)
        if "Proxy" in blob or "proxy" in blob:
            msg += ("\n\n看起来请求走了代理。内网端点需要绕开：\n"
                    "  set NO_PROXY=127.0.0.1,localhost,<端点主机>   （Windows）\n"
                    "  export NO_PROXY=127.0.0.1,localhost,<端点主机> （bash）")
        raise IfevalError(msg)


def latest_results(out_dir: Path) -> Path:
    got = sorted(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not got:
        raise IfevalError(f"{out_dir} 里没有 results_*.json，lm_eval 可能没跑完")
    return got[-1]


def count_samples(run_dir: Path) -> int:
    """只在**本次运行目录**里数样本，避免捡到上一次的旧文件。"""
    total = 0
    for p in Path(run_dir).rglob("samples_*.jsonl"):
        with p.open(encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


# ---------------------------------------------------------------- 结果整理

def _kv(v) -> dict:
    """把 lm_eval 的 `k=v,k=v` 字符串或 dict 统一成 dict。"""
    if isinstance(v, dict):
        return v
    out: dict = {}
    if isinstance(v, str):
        for part in v.split(","):
            if "=" in part:
                k, val = part.split("=", 1)
                out[k.strip()] = val.strip()
    return out


def gen_budget(data: dict) -> tuple[int | None, str]:
    """这次运行实际给了多少输出 token，以及这个数字的来源。

    优先级：
      1. `config.gen_kwargs` —— 命令行 `--gen_kwargs` 的运行时值，
         这是我们真正传下去的那个数；
      2. `configs.ifeval.generation_kwargs.max_gen_toks` —— 任务 yaml 的静态值，
         只有在没传 `--gen_kwargs` 时才等于实际值；
      3. 都没有 → (None, "unknown")。
    绝不拿「当前 config/models.json 里的 max_tokens」来填——复用的可能是
    上一轮 1280 的旧结果。
    """
    cfg = data.get("config") or {}
    kv = _kv(cfg.get("gen_kwargs"))
    if "max_gen_toks" in kv:
        try:
            return int(float(kv["max_gen_toks"])), "run"
        except (TypeError, ValueError):
            pass
    tc = ((data.get("configs") or {}).get(TASK) or {})
    gk = tc.get("generation_kwargs")
    if isinstance(gk, dict) and isinstance(gk.get("max_gen_toks"), (int, float)):
        return int(gk["max_gen_toks"]), "yaml"
    return None, "unknown"


def parse_results(path: Path) -> dict:
    """读 lm_eval 的 results.json。

    返回 `{metrics, n, task_version, lm_eval_version, max_gen_toks, budget_source}`：
    - `metrics` 换算成百分点，**四项必须齐全**，缺一项就拒绝合并——
      带着缺指标的报表比没有报表更糟；
    - `n` 取同一份 json 里的 `n-samples`，不是去目录里数文件；
    - 版本与输出上限取这份 json 自己的记录，不看当前配置。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    block = results.get(TASK) or (next(iter(results.values()), {}) if results else {})
    metrics = {}
    for key, val in block.items():
        base = str(key).split(",")[0]
        if base in METRICS and isinstance(val, (int, float)):
            metrics[METRICS[base]] = round(float(val) * 100.0, 2)

    missing = [v for v in METRICS.values() if v not in metrics]
    if missing:
        raise IfevalError(
            f"{path} 缺指标 {missing}（拿到 {sorted(metrics)}）。"
            "四项 prompt/instruction × strict/loose 必须齐全，不齐不合并。")

    n = None
    ns = (data.get("n-samples") or {})
    ns_block = ns.get(TASK) or (next(iter(ns.values()), {}) if ns else {})
    for key in ("effective", "original"):
        val = ns_block.get(key)
        if isinstance(val, int) and val > 0:
            n = val
            break

    ver = None
    versions = data.get("versions")
    if isinstance(versions, dict):
        ver = versions.get(TASK)
    budget, src = gen_budget(data)
    return {"metrics": metrics, "n": n, "task_version": ver,
            "lm_eval_version": data.get("lm_eval_version"),
            "max_gen_toks": budget, "budget_source": src}


def summarize(model, run_dir: Path, cli: str, limit: int | None = None) -> dict:
    """结果、题数、版本、输出预算全部取自同一轮的产物。"""
    res = latest_results(run_dir)
    parsed = parse_results(res)
    y = yaml_meta(task_yaml(cli))
    # n-samples 缺失时才退回去数样本；此时目录是本次独立的，不会串轮
    n = parsed["n"] or count_samples(run_dir) or (limit or 0)

    ver_file, ver_yaml = parsed["task_version"], y["version"]
    task_ver = ver_file if ver_file not in (None, "") else ver_yaml
    mismatch = bool(ver_file and ver_yaml and ver_yaml != "unknown"
                    and str(ver_file) != str(ver_yaml))

    return {
        "task": TASK,
        "model": model.name,
        "revision": model.revision,
        "n": n,
        "metrics": parsed["metrics"],
        "lm_eval": {
            "cli": cli,
            # 这一轮用的是哪个版本：以结果文件自己的记录为准，文件没记才去问 CLI 环境
            "version": parsed["lm_eval_version"] or cli_version(cli),
            "task_version": task_ver,
            "yaml_version": ver_yaml,
            "version_mismatch": mismatch,
            "yaml_sha256_16": y["sha256_16"],
            "results_file": str(res),
            # 输出上限只信这一次运行的记录；读不到就是 unknown
            "max_gen_toks": parsed["max_gen_toks"],
            "max_gen_toks_source": parsed["budget_source"],
        },
        "run_dir": str(run_dir),
        "endpoint": model.base_url,
        "maxlen": model.maxlen,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def sidecar_path(results_dir: Path, model_name: str, revision: str = "default") -> Path:
    """按 model@revision 分开存：base 与 fp8 的 IFEval 结果不能互相覆盖。"""
    rev = str(revision or "default").replace("/", "_").replace("\\", "_")
    return Path(results_dir) / f"ifeval-{model_name}@{rev}.json"


def load_sidecar(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def load_for(results_dir: Path, model_name: str, revision: str = "default") -> dict | None:
    """取当前 revision 的 sidecar；兼容早期不带 @revision 的文件名。"""
    want = str(revision or "default")
    rec = load_sidecar(sidecar_path(results_dir, model_name, want))
    if rec is None:
        rec = load_sidecar(Path(results_dir) / f"ifeval-{model_name}.json")
    if rec and str(rec.get("revision") or "default") != want:
        return None
    return rec


# ---------------------------------------------------------------- 报告用

def as_summary(rec: dict, maxlen: int | None = None) -> dict:
    """把 sidecar 转成 `runner.summarize` 的形状，好和其余三项并排打印。"""
    metrics = rec.get("metrics") or {}
    acc = metrics.get("prompt_strict")
    n = int(rec.get("n") or 0)
    se = None
    if acc is not None and n:
        p = acc / 100.0
        se = round((max(p * (1 - p), 0.0) / n) ** 0.5 * 100, 2)
    return {
        "task": TASK, "model": rec.get("model"), "revision": rec.get("revision"),
        "n": n, "n_scored": n, "n_error": 0, "over_ctx": 0, "n_unparsed": 0,
        "accuracy": acc, "se_pp": se, "official_parse_rate": None,
        "maxlen": rec.get("maxlen") or maxlen,
        "decode": {"max_tokens": (rec.get("lm_eval") or {}).get("max_gen_toks")},
        "external": True,
        "metrics": metrics,
        "lm_lib": rec.get("lm_eval"),
        "ts": rec.get("ran_at"),
    }


def apply_budget(lib: dict, *, from_results: bool,
                 explicit: int | None = None, passed: int | None = None) -> None:
    """决定 `lm_eval.max_gen_toks` 记什么。

    - 复用旧结果（`from_results`）：改写 `lib` 前先看有没有显式指定；
      否则只保留旧结果自己的记录，读不到就是 unknown。**绝不拿当前配置的
      max_tokens 顶替**——旧结果可能只给了 1280。
    - 本次新跑：结果文件没记下来、但我们确实传了 `--gen_kwargs`，才用传下去的值。
    """
    if from_results:
        if explicit:
            lib["max_gen_toks"], lib["max_gen_toks_source"] = explicit, "args"
        elif lib.get("max_gen_toks") is None:
            lib["max_gen_toks_source"] = "unknown"
    elif lib.get("max_gen_toks") is None and passed:
        lib["max_gen_toks"], lib["max_gen_toks_source"] = passed, "cli"


def print_run(rec: dict) -> None:
    m = rec["metrics"]
    lib = rec.get("lm_eval") or {}
    print(f"\nIFEval  n={rec['n']}  (lm-eval {lib.get('version')} / "
          f"task v{lib.get('task_version')})")
    print(f"  prompt-level strict  {m.get('prompt_strict')}%")
    print(f"  prompt-level loose   {m.get('prompt_loose')}%")
    print(f"  inst-level   strict  {m.get('inst_strict')}%")
    print(f"  inst-level   loose   {m.get('inst_loose')}%")

    budget, src = lib.get("max_gen_toks"), lib.get("max_gen_toks_source")
    label = {"run": "本次命令行", "yaml": "任务 yaml 默认",
             "args": "手动指定", "unknown": "取不到"}.get(src, src)
    print(f"  输出上限 {budget if budget else 'unknown'}（{label}）")
    if not budget:
        print("  ⚠ 这一轮没用 --gen_kwargs，也没留下记录：报表里会标 unknown。"
              "重跑时加 --max-gen-toks，或不要用 --from-results。")
    if lib.get("version_mismatch"):
        print(f"  ⚠ 结果里的 task 版本 ({lib.get('task_version')}) 与当前安装的 "
              f"ifeval.yaml ({lib.get('yaml_version')}) 不一致，口径可能已经变了。")
