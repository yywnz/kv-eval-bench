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


def cli_version(cli: str) -> str:
    """先试当前解释器的包元数据，再顺着命令所在目录找 dist-info。"""
    try:
        from importlib.metadata import version
        return version("lm-eval")
    except Exception:
        pass
    exe = Path(cli).resolve()
    roots = [exe.parent.parent / "Lib" / "site-packages",
             exe.parent.parent / "lib" / "python3.13" / "site-packages",
             exe.parent.parent / "lib" / "site-packages"]
    for root in roots:
        for d in root.glob("lm_eval-*.dist-info"):
            ver = d.name.split("-")[1].replace(".dist", "")
            if ver:
                return ver
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


def count_samples(out_dir: Path) -> int:
    for p in out_dir.rglob("samples_*.jsonl"):
        with p.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    return 0


# ---------------------------------------------------------------- 结果整理

def parse_results(path: Path) -> dict:
    """把 lm_eval 的 results.json 收成 {prompt_strict: 62.4, ...}（单位 pp）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    block = results.get(TASK) or (next(iter(results.values()), {}) if results else {})
    got = {}
    for key, val in block.items():
        base = str(key).split(",")[0]
        if base in METRICS and isinstance(val, (int, float)):
            got[METRICS[base]] = round(float(val) * 100.0, 2)
    if not got:
        raise IfevalError(f"{path} 里没找到 ifeval 的指标键，实际有：{list(block)[:6]}")
    return got


def summarize(model, out_dir: Path, cli: str, limit: int | None = None) -> dict:
    res = latest_results(out_dir)
    metrics = parse_results(res)
    y = yaml_meta(task_yaml(cli))
    n = count_samples(out_dir) or (limit or 541)
    return {
        "task": TASK,
        "model": model.name,
        "revision": model.revision,
        "n": n,
        "metrics": metrics,
        "lm_eval": {"cli": cli, "version": cli_version(cli),
                    "task_version": y["version"], "yaml_sha256_16": y["sha256_16"],
                    "results_file": str(res)},
        "endpoint": model.base_url,
        "maxlen": model.maxlen,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def sidecar_path(results_dir: Path, model_name: str) -> Path:
    return Path(results_dir) / f"ifeval-{model_name}.json"


def load_sidecar(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


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


def print_run(rec: dict) -> None:
    m = rec["metrics"]
    lib = rec.get("lm_eval") or {}
    print(f"\nIFEval  n={rec['n']}  (lm-eval {lib.get('version')} / "
          f"task v{lib.get('task_version')})")
    print(f"  prompt-level strict  {m.get('prompt_strict')}%")
    print(f"  prompt-level loose   {m.get('prompt_loose')}%")
    print(f"  inst-level   strict  {m.get('inst_strict')}%")
    print(f"  inst-level   loose   {m.get('inst_loose')}%")
