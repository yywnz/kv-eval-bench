"""任务定义：数据来源、抽样策略、官方协议参数。

四条任务，行格式统一（**只存结构化字段，不拼 prompt**）：

    {"id", "task", "kind", "question", "options"?, "context"?, "answer", "meta"}

prompt 在生成时按 `prompts/*.txt` 现场装配，换模板不用重建数据。

每条任务带 `decode`，字段取自官方实现（温度、top_p、输出上限），
不写在这里的用 `Model` 的默认值。官方值与本仓库实际取值不一致时，
数据卡里必须写明，`report.py` 也会把它打出来。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LETTERS = "ABCDEFGHIJ"


def qid(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.sha256(text.strip().encode('utf-8')).hexdigest()[:10]}"


def _hf(repo: str, split: str, streaming: bool = True):
    from datasets import load_dataset
    return load_dataset(repo, split=split, streaming=streaming)


# ---------------------------------------------------------------- IFEval

def build_ifeval(seed: int, **_) -> tuple[list[dict], dict]:
    rows = []
    for r in _hf("google/IFEval", "train"):
        rows.append({
            "id": qid("ifeval", r["prompt"] + str(r["key"])),
            "task": "ifeval", "kind": "instruct",
            "question": r["prompt"],
            "answer": "",
            "meta": {"key": r["key"],
                     "instruction_id_list": r["instruction_id_list"],
                     "kwargs": r["kwargs"],
                     "n_constraints": len(r["instruction_id_list"])},
        })
    rows.sort(key=lambda x: x["id"])
    kinds = sorted({i for r in rows for i in r["meta"]["instruction_id_list"]})
    return rows, {"policy": "全量，未抽样",
                  "n_instruction_types": len(kinds),
                  "instruction_types": kinds}


# ---------------------------------------------------------------- MMLU-Pro

def build_mmlu_pro(seed: int, limit: int = 500, **_) -> tuple[list[dict], dict]:
    """按 category 分层抽 limit 题；同时导出官方 5-shot 示例（validation 集）。"""
    pool: dict[str, list[dict]] = defaultdict(list)
    for r in _hf("TIGER-Lab/MMLU-Pro", "test"):
        pool[str(r.get("category", "na"))].append(r)

    cats = sorted(pool)
    base, rem = divmod(limit, len(cats))
    quota = {c: base + (1 if i < rem else 0) for i, c in enumerate(cats)}

    rng = random.Random(seed)
    rows = []
    for cat in cats:
        items = sorted(pool[cat], key=lambda x: str(x.get("question_id")))
        for r in rng.sample(items, min(quota[cat], len(items))):
            opts = [str(o) for o in r["options"] if str(o).strip() != "N/A"]
            rows.append({
                "id": qid("mmlu_pro", str(r["question_id"]) + r["question"]),
                "task": "mmlu_pro", "kind": "mc",
                "question": r["question"],
                "options": opts,
                "answer": LETTERS[int(r["answer_index"])],
                "meta": {"category": cat, "n_options": len(opts),
                         "src_index": r.get("question_id")},
            })
    rows.sort(key=lambda x: x["id"])
    dist = {c: sum(1 for r in rows if r["meta"]["category"] == c) for c in cats}
    return rows, {"policy": f"按 category 分层均分，共 {limit} 题", "per_category": dist}


def build_mmlu_exemplars(seed: int, per_cat: int = 5, **_) -> dict:
    """官方 5-shot 示例：取自 validation 集，每类前 per_cat 条（官方按类取全部 5 条）。"""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in _hf("TIGER-Lab/MMLU-Pro", "validation", streaming=False):
        cat = str(r.get("category", "na"))
        if len(out[cat]) >= per_cat:
            continue
        opts = [str(o) for o in r["options"] if str(o).strip() != "N/A"]
        out[cat].append({
            "question": r["question"],
            "options": opts,
            "answer": LETTERS[int(r["answer_index"])],
            "cot": (r.get("cot_content") or "").strip(),
        })
    return {k: out[k] for k in sorted(out)}


# ---------------------------------------------------------------- GPQA

def build_gpqa(seed: int, **_) -> tuple[list[dict], dict]:
    """优先取官方 gated 数据（含四选项）；取不到则回落到公开的无选项镜像。

    两种来源的 `meta.official_options` 不同，报告里必须标明——无选项时的
    难度高于官方设置，分数不可与榜单横比。
    """
    from .scoring import extract_boxed
    rows: list[dict] = []
    source = ""
    try:
        for r in _hf("Idavidrein/gpqa", "diamond"):
            opts = [r.get(k) for k in ("Correct Answer", "Incorrect Answer 1",
                                       "Incorrect Answer 2", "Incorrect Answer 3")]
            rows.append({
                "id": qid("gpqa", r["Question"]),
                "task": "gpqa_diamond", "kind": "open", "match": "boxed",
                "question": r["Question"],
                "options": [str(o) for o in opts],
                "answer": str(r["Correct Answer"]).strip(),
                "meta": {"domain": r.get("Domain") or r.get("Subdomain"),
                         "official_options": True},
            })
        source = "Idavidrein/gpqa（官方 gated）"
    except Exception:
        for r in _hf("hendrydong/GPQA_diamond", "test"):
            ans = extract_boxed(r.get("solution") or "")
            if not ans:
                tail = (r.get("solution") or "").strip().splitlines()
                ans = tail[-1].strip() if tail else ""
            rows.append({
                "id": qid("gpqa", r["problem"]),
                "task": "gpqa_diamond", "kind": "open", "match": "boxed",
                "question": r["problem"],
                "options": [],
                "answer": ans,
                "meta": {"domain": r.get("domain"), "official_options": False},
            })
        source = "hendrydong/GPQA_diamond（公开镜像，无四选项）"

    rows.sort(key=lambda x: x["id"])
    has_opts = all(r["meta"]["official_options"] for r in rows)
    return rows, {"policy": "全量（Diamond 子集）", "source_used": source,
                  "official_options": has_opts,
                  "note": ("与官方设置一致（四选项 + boxed 字符串比对）" if has_opts else
                           "无四选项，模型只能凭题面作答，难度高于官方设置；"
                           "分数不可与公开榜单横比")}


# ---------------------------------------------------------------- LongBench v2

def build_longbench_v2(seed: int, per_cell: int = 14, limit: int = 200,
                       **_) -> tuple[list[dict], dict]:
    """按 domain × length 分层，每格用蓄水池随机抽 per_cell 题（单遍流式）。"""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: dict[tuple[str, str], int] = defaultdict(int)
    total = 0

    for r in _hf("zai-org/LongBench-v2", "train"):
        opts = [r.get("choice_A"), r.get("choice_B"), r.get("choice_C"), r.get("choice_D")]
        if any(o is None for o in opts):
            continue
        key = (str(r.get("domain", "na")), str(r.get("length", "na")))
        seen[key] += 1
        total += 1
        ctx = r.get("context") or ""
        item = {
            "id": qid("lbv2", str(r.get("_id")) + r.get("question", "")),
            "task": "longbench_v2", "kind": "mc",
            "context": ctx,
            "question": r["question"],
            "options": [str(o) for o in opts],
            "answer": r["answer"],
            "meta": {"domain": key[0], "sub_domain": r.get("sub_domain"),
                     "length": key[1], "difficulty": r.get("difficulty"),
                     "ctx_chars": len(ctx)},
        }
        bucket = cells[key]
        if len(bucket) < per_cell:
            bucket.append(item)
        else:
            # 标准蓄水池：j 同时决定「是否替换」和「替换哪个槽位」。
            # 一次抽样必须只用一次 randrange，多抽一个数会改变选中集合，
            # 导致与历史版本的 id 不一致。
            rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
            j = rng.randrange(seen[key])
            if j < per_cell:
                bucket[j] = item

    rows = [x for k in sorted(cells) for x in cells[k]]
    rng = random.Random(seed)
    if len(rows) > limit:
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), limit))]
    rows.sort(key=lambda x: x["id"])

    dist = defaultdict(int)
    for r in rows:
        dist[r["meta"]["length"]] += 1
    return rows, {"policy": f"按 domain × length 分层，每格蓄水池随机 {per_cell}，总量上限 {limit}",
                  "by_length": dict(sorted(dist.items())),
                  "by_cell": {f"{k[0]} | {k[1]}": len(v) for k, v in sorted(cells.items())},
                  "source_total": total}


# ---------------------------------------------------------------- 注册表

@dataclass
class Task:
    name: str
    dimension: str
    display: str
    source: str
    source_url: str
    license: str
    kind: str
    shape: str                  # 决定用 prompts/ 下哪组模板
    metric: str                 # 官方指标名
    judge: str                  # local / external
    decode: dict                # 官方解码参数
    note: str
    build: Callable[..., tuple[list[dict], dict]] = field(repr=False, default=None)
    doc: str = ""


TASKS: dict[str, Task] = {}


def _reg(**kw) -> None:
    t = Task(**kw)
    TASKS[t.name] = t


_reg(name="ifeval", dimension="指令遵循", display="IFEval",
     source="google/IFEval", source_url="https://huggingface.co/datasets/google/IFEval",
     license="Apache-2.0", kind="instruct", shape="instruct",
     metric="prompt-level strict accuracy / instruction-level accuracy",
     judge="external",
     decode={"temperature": 0.0, "max_tokens": 1280},
     note="24 类可程序验证约束；判分用官方 verifier（lm_eval）",
     build=build_ifeval, doc="docs/tasks/ifeval.md")

_reg(name="mmlu_pro", dimension="知识", display="MMLU-Pro",
     source="TIGER-Lab/MMLU-Pro", source_url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
     license="MIT", kind="mc", shape="mmlu",
     metric="accuracy（5-shot CoT，抽取 \"The answer is (X)\"）",
     judge="local",
     decode={"temperature": 0.1, "top_p": 0.95, "max_tokens": 8192},
     note="官方为 5-shot CoT，示例取自 validation 集，每类 5 条",
     build=build_mmlu_pro, doc="docs/tasks/mmlu-pro.md")

_reg(name="gpqa_diamond", dimension="科学推理", display="GPQA Diamond",
     source="Idavidrein/gpqa（官方 gated）",
     source_url="https://huggingface.co/datasets/Idavidrein/gpqa",
     license="CC BY 4.0", kind="open", shape="gpqa",
     metric="accuracy（\\boxed{} 抽取后归一比对）",
     judge="local",
     decode={"temperature": 0.5, "max_tokens": 8192},
     note="论文口径：逐步推理，答案放 \\boxed{}；官方数据 gated，无权限时回落无选项镜像",
     build=build_gpqa, doc="docs/tasks/gpqa-diamond.md")

_reg(name="longbench_v2", dimension="长上下文", display="LongBench v2",
     source="zai-org/LongBench-v2", source_url="https://huggingface.co/datasets/zai-org/LongBench-v2",
     license="MIT", kind="mc", shape="longctx",
     metric="accuracy（0-shot，抽取 \"The correct answer is (X)\"）",
     judge="local",
     decode={"temperature": 0.1, "max_tokens": 128},
     note="官方 0-shot 输出上限 128 token；思考模型需调大，报告须注明",
     build=build_longbench_v2, doc="docs/tasks/longbench-v2.md")


def build_one(name: str, seed: int, **params) -> tuple[list[dict], dict]:
    task = TASKS[name]
    rows, meta = task.build(seed, **params)
    meta = {"task": name, "dimension": task.dimension, "display": task.display,
            "shape": task.shape, "source": task.source, "license": task.license,
            "n": len(rows), "metric": task.metric, "judge": task.judge,
            "official_decode": task.decode, **meta}
    return rows, meta


# 固定顺序，CLI 与 pred.py 共用
ORDER = ["ifeval", "mmlu_pro", "gpqa_diamond", "longbench_v2"]
