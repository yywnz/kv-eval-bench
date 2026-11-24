"""IFEval 封装的回归测试（不依赖 lm_eval 本体）。

只测本仓库自己写的那部分：命令怎么拼、结果怎么读、版本从哪来、结果怎么分轮。
真正的约束校验在 lm_eval 里，这里不碰。

    python tests/test_ifeval.py
"""

import json
import os
import sys
import tempfile
import time as _t
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capbench.client import Model
from capbench.ifeval import (IfevalError, as_summary, build_cmd, cli_version,
                             parse_results, sidecar_path, summarize, yaml_meta)

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name:<52} got={got!r}")
    if not ok:
        FAILED.append((name, got, want))


M = Model(name="glm-flash", base_url="http://10.0.0.1:18000/", maxlen=262144)

# ---------------------------------------------------------------- 1. 命令
cmd = build_cmd("lm-eval", M, Path("/tmp/out"), concurrency=4, batch_size=8)
args = " ".join(cmd)
check("带 tokenizer_backend=none", "tokenizer_backend=none" in args, True)
check("base_url 只有一个 /v1", "18000/v1,model=" in args, True)
check("不出现 /v1/v1", "/v1/v1" in args, False)
check("任务名", cmd[cmd.index("--tasks") + 1], "ifeval")
check("零样本", cmd[cmd.index("--num_fewshot") + 1], "0")
check("保留样本", "--log_samples" in cmd, True)
check("不传 gen_kwargs 时不出现", "--gen_kwargs" in cmd, False)

cmd2 = build_cmd("lm-eval", M, Path("/tmp/out"), max_gen_toks=8192, limit=10)
a2 = " ".join(cmd2)
check("思考模型覆盖输出上限", "max_gen_toks=8192" in a2, True)
check("冒烟 limit", "--limit 10" in a2, True)

# ---------------------------------------------------------------- 2. 结果解析
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "results_x.json"
    p.write_text(json.dumps({
        "results": {"ifeval": {
            "alias": "ifeval",
            "prompt_level_strict_acc,none": 0.6234,
            "prompt_level_loose_acc,none": 0.6512,
            "inst_level_strict_acc,none": 0.7123,
            "inst_level_loose_acc,none": 0.735,
        }},
        "n-samples": {"ifeval": {"original": 541, "effective": 541}},
        "versions": {"ifeval": 4.0},
    }), encoding="utf-8")
    got = parse_results(p)
    check("四个指标都取到", sorted(got["metrics"]),
          ["inst_loose", "inst_strict", "prompt_loose", "prompt_strict"])
    check("换算成百分点", got["metrics"]["prompt_strict"], 62.34)
    check("alias 不进指标", "alias" in got["metrics"], False)
    check("题数取 n-samples", got["n"], 541)
    check("任务版本取 versions", got["task_version"], 4.0)

    p2 = Path(d) / "results_y.json"
    p2.write_text(json.dumps({"results": {"ifeval": {
        "prompt_level_strict_acc,none": 0.5}}}), encoding="utf-8")
    check("缺 n-samples 时 n 为空", parse_results(p2)["n"], None)

    bad = Path(d) / "results_bad.json"
    bad.write_text(json.dumps({"results": {"other": {"acc": 1.0}}}), encoding="utf-8")
    try:
        parse_results(bad)
        check("缺指标要报错", "没抛异常", "IfevalError")
    except IfevalError:
        check("缺指标要报错", "IfevalError", "IfevalError")

# ---------------------------------------------------------------- 3. 版本来源
with tempfile.TemporaryDirectory() as d:
    env = Path(d) / "fakeenv"
    (env / "Scripts").mkdir(parents=True)
    (env / "Lib" / "site-packages" / "lm_eval-9.9.9.dist-info").mkdir(parents=True)
    fake_cli = env / "Scripts" / "lm-eval"
    fake_cli.write_text("", encoding="utf-8")
    check("版本取自 CLI 所在环境", cli_version(str(fake_cli)), "9.9.9")

    lonely = Path(d) / "nowhere" / "lm-eval"
    check("取不到时是 unknown", cli_version(str(lonely)), "unknown")

# ---------------------------------------------------------------- 4. 多轮不串
with tempfile.TemporaryDirectory() as d:
    run = Path(d) / "run"
    (run / "sub").mkdir(parents=True)
    (run / "sub" / "results_old.json").write_text(json.dumps({
        "results": {"ifeval": {"prompt_level_strict_acc,none": 0.60}},
        "n-samples": {"ifeval": {"original": 541, "effective": 541}}}), encoding="utf-8")
    (run / "sub" / "results_new.json").write_text(json.dumps({
        "results": {"ifeval": {"prompt_level_strict_acc,none": 0.70}},
        "n-samples": {"ifeval": {"original": 20, "effective": 20}}}), encoding="utf-8")
    # 旧样本文件行数故意多，且路径排序在前
    (run / "a_samples_old.jsonl").write_text(
        "\n".join("{}" for _ in range(541)), encoding="utf-8")
    (run / "sub" / "samples_new.jsonl").write_text(
        "\n".join("{}" for _ in range(20)), encoding="utf-8")

    old = _t.time() - 100
    os.utime(run / "sub" / "results_old.json", (old, old))

    rec = summarize(M, run, str(Path(d) / "no-such-cli"))
    check("题数与指标同源（取最新那轮）", rec["n"], 20)
    check("指标也是那一轮", rec["metrics"]["prompt_strict"], 70.0)
    check("版本取不到就 unknown", rec["lm_eval"]["version"], "unknown")

# ---------------------------------------------------------------- 5. sidecar 分 revision
check("文件名带 revision", sidecar_path("R", "glm", "fp8").name, "ifeval-glm@fp8.json")
check("默认 revision 也写出来", sidecar_path("R", "glm").name, "ifeval-glm@default.json")

# ---------------------------------------------------------------- 6. 汇总形状
rec = {"task": "ifeval", "model": "glm-flash", "revision": "default",
       "n": 541, "maxlen": 262144,
       "metrics": {"prompt_strict": 62.34, "prompt_loose": 65.12,
                   "inst_strict": 71.23, "inst_loose": 73.5},
       "lm_eval": {"version": "0.4.13", "task_version": "4.0",
                   "max_gen_toks": 16384}}
s = as_summary(rec)
check("acc 取 prompt-level strict", s["accuracy"], 62.34)
check("scored 等于 n", s["n_scored"], 541)
check("external 标记", s["external"], True)
check("标注为外部判分", s["official_parse_rate"], None)
check("maxlen 透传", s["maxlen"], 262144)
check("SE 是二项标准误", abs(s["se_pp"] - 2.08) < 0.02, True)
check("decode 记录输出上限", s["decode"]["max_tokens"], 16384)

# ---------------------------------------------------------------- 7. yaml 版本
with tempfile.TemporaryDirectory() as d:
    y = Path(d) / "ifeval.yaml"
    y.write_text("task: ifeval\nnum_fewshot: 0\nmetadata:\n  version: 4.0\n",
                 encoding="utf-8")
    meta = yaml_meta(y)
    check("解析 task 版本", meta["version"], "4.0")
    check("记录 yaml 哈希长度", len(meta["sha256_16"]), 16)
check("没有 yaml 时不炸", yaml_meta(None)["version"], "unknown")

print()
if FAILED:
    print(f"{len(FAILED)} 项失败")
    for n, g, w in FAILED:
        print(f"  - {n}: got={g!r} want={w!r}")
    sys.exit(1)
print("全部通过")
