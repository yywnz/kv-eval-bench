"""判分：每项按各自官方实现抽取答案。

| 任务 | 官方抽取规则 | 来源 |
|---|---|---|
| MMLU-Pro | `The answer is (X)` | TIGER-Lab/MMLU-Pro `evaluate_from_apiX.py` |
| LongBench v2 | `The correct answer is (X)` | THUDM/LongBench `pred.py::extract_answer` |
| IFEval | 官方约束校验器 | 外部（lm_eval），本文件不实现 |

抽取失败时退回通用规则（最后一个独立字母），`parse` 字段标明走了哪条路，
`official` 字段标明是否命中官方规则——报告里要单独统计「官方抽取命中率」，
它偏低说明模型没遵守输出格式，分数与公开榜单不可比。
"""

from __future__ import annotations

import re
from typing import Any

# MMLU-Pro 官方：the answer is (X)
_MMLU_RE = re.compile(r"answer\s+is\s*\(?\s*([A-J])\s*\)?", re.I)
# LongBench v2 官方：The correct answer is (X)
_LB_PAREN = re.compile(r"The correct answer is \(([A-D])\)")
_LB_BARE = re.compile(r"The correct answer is ([A-D])")
# 通用兜底
_MARK_RE = re.compile(
    r"(?:final\s+answer|answer|正确答案|答案|选项|choice|option)"
    r"\s*(?:is|are|为|是|should\s+be)?\s*[:：]?\s*[*_`$\s]*\(?\s*([A-Z])(?![A-Za-z])",
    re.I)
_LONE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])")


def letters_for(n: int) -> list[str]:
    """按选项个数生成字母表；至少 4 个，最多 26 个。"""
    return [chr(ord("A") + i) for i in range(max(4, min(n, 26)))]


def _clean(s: Any) -> str:
    return str(s or "").replace("\u00a0", " ").replace("\u2212", "-")


def extract_choice(text: str, choices: str = "ABCDEFGHIJ") -> tuple[str | None, str]:
    """通用选项抽取：短回答 → 显式标记（取最后一次）→ 末行 → 全文最后一个独立字母。"""
    t = _clean(text).strip()
    if not t:
        return None, "empty"
    ok = set(choices)

    core = t.strip("*_`$# \t\r\n").strip("()[]{}.").strip()
    if 0 < len(core) <= 2:
        m = _LONE_RE.search(core)
        if m and m.group(1) in ok:
            return m.group(1), "short"
    hits = [m.group(1) for m in _MARK_RE.finditer(t) if m.group(1) in ok]
    if hits:
        return hits[-1], "marker"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines and len(lines[-1]) <= 40:
        cand = [m.group(1) for m in _LONE_RE.finditer(lines[-1]) if m.group(1) in ok]
        if cand:
            return cand[-1], "last_line"
    cand = [m.group(1) for m in _LONE_RE.finditer(t) if m.group(1) in ok]
    if cand:
        return cand[-1], "fallback"
    return None, "none"


def _first_step_logprobs(lp: Any) -> dict[str, float]:
    if not lp:
        return {}
    out: dict[str, float] = {}
    content = lp.get("content") if isinstance(lp, dict) else None
    if isinstance(content, list) and content:
        for item in content[0].get("top_logprobs") or []:
            out[str(item.get("token", "")).strip()] = float(item.get("logprob", -99.0))
    return out


def score_mc(resp: dict, task: str, choices: str) -> dict:
    """多选：先跑官方抽取规则，再退回通用规则，最后才用 logprob。"""
    text = resp.get("text", "")
    got, how, official = None, "none", False

    if task == "mmlu_pro":
        m = _MMLU_RE.search(text)
        if m and m.group(1) in choices:
            got, how, official = m.group(1), "official", True
    elif task == "longbench_v2":
        m = _LB_PAREN.search(text) or _LB_BARE.search(text)
        if m and m.group(1) in choices:
            got, how, official = m.group(1), "official", True

    if not got:
        got, how = extract_choice(text, choices)

    lp_map = {k: v for k, v in _first_step_logprobs(resp.get("logprobs")).items()
              if len(k) == 1 and k in set(choices)}
    if not got and lp_map:
        return {"chosen": max(lp_map, key=lp_map.get), "lp": lp_map,
                "parse": "logprob", "official": False}
    return {"chosen": got or "?", "lp": lp_map, "parse": how, "official": official}


def margin_of(correct: str, lp: dict[str, float]) -> float | None:
    """logp(正确) − max(logp(错误))。负值代表正确项被干扰项压过。"""
    if not lp or correct not in lp:
        return None
    wrongs = [v for k, v in lp.items() if k != correct]
    return round(lp[correct] - max(wrongs), 4) if wrongs else None


def extract_number(text: str) -> str | None:
    """取正文里**最后一个**数字串，去掉千分位。

    结论在末尾，所以取最后一个；思考模型会把中间量也写出来，
    取第一个几乎必错。
    """
    nums = re.findall(r"\d[\d,]*", _clean(text))
    if not nums:
        return None
    return nums[-1].replace(",", "")


def score_numeric(resp: dict, answer: str) -> dict:
    cand = extract_number(resp.get("text") or "")
    gold = re.sub(r"\D", "", str(answer))
    if cand is None:
        return {"correct": False, "chosen": None, "parse": "none", "official": False}
    return {"correct": cand == gold, "chosen": cand, "parse": "number",
            "official": True}


def score(row: dict, resp: dict) -> dict:
    """按任务分派，返回 {correct, chosen, parse, official, margin?}。"""
    task = row.get("task", "")
    kind = row.get("kind", "mc")

    if kind == "instruct":
        # 约束是否满足只有官方 verifier 能判，这里只存原文
        return {"correct": None, "chosen": (resp.get("text") or "")[:200],
                "parse": "external", "official": False}

    if kind == "numeric":
        return score_numeric(resp, row.get("answer", ""))

    if kind == "mc":
        opts = row.get("options") or []
        letters = "".join(letters_for(len(opts)))
        s = score_mc(resp, task, letters)
        s["correct"] = s["chosen"] == row.get("answer")
        s["margin"] = margin_of(row.get("answer", ""), s["lp"])
        return s

    raise ValueError(f"未知题型 {kind}（task={task}）")
