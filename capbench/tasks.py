"""任务定义：数据来源、抽样策略、题型。

每条任务一个 Task，`build()` 产出统一格式的行（**只存结构化字段，不拼 prompt**）：

    {"id", "task", "kind", "question", "options"?, "context"?, "answer", "max_tokens", "meta"}

prompt 在 `pred.py` 里按 `prompts/*.txt` 模板现场拼，换模板不用重建数据。

抽样策略全部固定种子（默认 2026），并且**先收集再随机**——按流式顺序取前 N 条
会在类别内引入系统性偏差（源文件常按难度或年份排序）。
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
            "max_tokens": 1024,
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
            opts = [str(o) for o in r["options"]]
            rows.append({
                "id": qid("mmlu_pro", str(r["question_id"]) + r["question"]),
                "task": "mmlu_pro", "kind": "mc",
                "question": r["question"],
                "options": opts,
                "answer": LETTERS[int(r["answer_index"])],
                "max_tokens": 1024,
                "meta": {"category": cat, "n_options": len(opts),
                         "src_index": r.get("question_id")},
            })
    rows.sort(key=lambda x: x["id"])
    dist = {c: sum(1 for r in rows if r["meta"]["category"] == c) for c in cats}
    return rows, {"policy": f"按 category 分层均分，共 {limit} 题", "per_category": dist}


# ---------------------------------------------------------------- GPQA

def build_gpqa(seed: int, **_) -> tuple[list[dict], dict]:
    from .scoring import extract_boxed
    rows, fallback = [], 0
    for r in _hf("hendrydong/GPQA_diamond", "test"):
        ans = extract_boxed(r.get("solution") or "")
        if not ans:
            fallback += 1
            tail = (r.get("solution") or "").strip().splitlines()
            ans = tail[-1].strip() if tail else ""
        rows.append({
            "id": qid("gpqa", r["problem"]),
            "task": "gpqa_diamond", "kind": "open", "match": "boxed",
            "question": r["problem"],
            "answer": ans,
            "max_tokens": 8192,
            "meta": {"domain": r.get("domain"), "answer_from_boxed": ans != ""},
        })
    rows.sort(key=lambda x: x["id"])
    return rows, {"policy": "全量，未抽样（Diamond 子集）",
                  "format": "自由作答 + \\boxed{} 抽取（官方论文口径，非四选一）",
                  "answer_fallback_count": fallback}


# ---------------------------------------------------------------- AIME

def build_aime(seed: int, **_) -> tuple[list[dict], dict]:
    rows, years = [], {}
    for r in _hf("Maxwell-Jia/AIME_2024", "train", streaming=False):
        rows.append({
            "id": qid("aime", "2024:" + str(r["ID"]) + r["Problem"]),
            "task": "aime", "kind": "numeric",
            "question": r["Problem"],
            "answer": str(r["Answer"]).strip(),
            "max_tokens": 8192,
            "meta": {"year": 2024, "problem_id": r["ID"]},
        })
    years[2024] = len(rows)
    n2024 = len(rows)
    for r in _hf("MathArena/aime_2025", "train", streaming=False):
        rows.append({
            "id": qid("aime", "2025:" + str(r["problem_idx"]) + r["problem"]),
            "task": "aime", "kind": "numeric",
            "question": r["problem"],
            "answer": str(r["answer"]).strip(),
            "max_tokens": 8192,
            "meta": {"year": 2025, "problem_id": r.get("problem_idx"),
                     "type": r.get("problem_type")},
        })
    years[2025] = len(rows) - n2024
    rows.sort(key=lambda x: x["id"])
    return rows, {"policy": "全量（2024 + 2025 两套各 30 题）",
                  "by_year": years,
                  "note": "样本量小，单题≈1.7pp，建议 avg@k 多次采样后取平均"}


# ---------------------------------------------------------------- LiveCodeBench

def build_livecodebench(seed: int, limit: int = 150, **_) -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id="livecodebench/code_generation_lite",
                                filename="test6.jsonl", repo_type="dataset"))
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    items.sort(key=lambda x: str(x.get("contest_date", "")), reverse=True)
    picked = items[:limit]
    rows = []
    for r in picked:
        rows.append({
            "id": qid("lcb", str(r.get("question_id")) + (r.get("question_content") or "")[:200]),
            "task": "livecodebench", "kind": "code",
            "question": r.get("question_content") or "",
            "answer": "",
            "max_tokens": 4096,
            "meta": {"question_id": r.get("question_id"),
                     "title": r.get("question_title"),
                     "platform": r.get("platform"),
                     "difficulty": r.get("difficulty"),
                     "contest_date": r.get("contest_date"),
                     "starter_code": r.get("starter_code"),
                     "public_test_cases": r.get("public_test_cases"),
                     "private_test_cases": r.get("private_test_cases")},
        })
    dates = sorted(str(r["meta"]["contest_date"])[:10] for r in rows)
    return rows, {"policy": f"按 contest_date 倒序取最近 {limit} 题",
                  "release": "test6 / release_v6",
                  "date_range": [dates[0], dates[-1]] if dates else [],
                  "total_in_release": len(items)}


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
            "max_tokens": 1024,
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
    shape: str                  # 决定用 prompts/ 下哪个模板
    metric: str
    judge: str                  # local / external
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
     metric="prompt-level strict accuracy", judge="external",
     note="25 类可程序验证的约束；判分用官方 verifier",
     build=build_ifeval, doc="docs/tasks/ifeval.md")

_reg(name="mmlu_pro", dimension="知识", display="MMLU-Pro",
     source="TIGER-Lab/MMLU-Pro", source_url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
     license="MIT", kind="mc", shape="mc",
     metric="accuracy（首 token 选项）", judge="local",
     note="10 选项为主，非固定四选一",
     build=build_mmlu_pro, doc="docs/tasks/mmlu-pro.md")

_reg(name="gpqa_diamond", dimension="科学推理", display="GPQA Diamond",
     source="hendrydong/GPQA_diamond（Idavidrein/gpqa 的社区镜像）",
     source_url="https://huggingface.co/datasets/hendrydong/GPQA_diamond",
     license="CC BY 4.0", kind="open", shape="open",
     metric="accuracy（答案归一后匹配）", judge="local",
     note="自由作答 + \\boxed{} 抽取，与官方论文口径一致",
     build=build_gpqa, doc="docs/tasks/gpqa-diamond.md")

_reg(name="aime", dimension="数学", display="AIME 2024+2025",
     source="Maxwell-Jia/AIME_2024 + MathArena/aime_2025",
     source_url="https://huggingface.co/datasets/MathArena/aime_2025",
     license="CC BY 4.0 / MIT", kind="numeric", shape="numeric",
     metric="accuracy（avg@k 推荐）", judge="local",
     note="答案 0–999 整数；60 题，单题≈1.7pp",
     build=build_aime, doc="docs/tasks/aime.md")

_reg(name="livecodebench", dimension="代码", display="LiveCodeBench v6",
     source="livecodebench/code_generation_lite（test6）",
     source_url="https://huggingface.co/datasets/livecodebench/code_generation_lite",
     license="MIT", kind="code", shape="code",
     metric="pass@1（执行测试用例）", judge="external",
     note="按月份发布新题，抗污染；判分需沙盒执行",
     build=build_livecodebench, doc="docs/tasks/livecodebench.md")

_reg(name="longbench_v2", dimension="长上下文", display="LongBench v2",
     source="zai-org/LongBench-v2", source_url="https://huggingface.co/datasets/zai-org/LongBench-v2",
     license="MIT", kind="mc", shape="longctx",
     metric="accuracy（四选一）", judge="local",
     note="6 类任务 × 3 档长度分层；单题最长约 2M words",
     build=build_longbench_v2, doc="docs/tasks/longbench-v2.md")


def build_one(name: str, seed: int, **params) -> tuple[list[dict], dict]:
    task = TASKS[name]
    rows, meta = task.build(seed, **params)
    meta = {"task": name, "dimension": task.dimension, "display": task.display,
            "shape": task.shape, "source": task.source, "license": task.license,
            "n": len(rows), "metric": task.metric, "judge": task.judge, **meta}
    return rows, meta


# 固定顺序，CLI 与 pred.py 共用
ORDER = ["ifeval", "mmlu_pro", "gpqa_diamond", "aime", "livecodebench", "longbench_v2"]
