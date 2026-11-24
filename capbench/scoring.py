"""判分。

三道题型的判分口径分别写在 docs/tasks/*.md 里，这里是实现。
多选题优先用首步 logprob 定选项（答案只差一个 token 时最灵敏），
拿不到 logprob 时退回文本解析——思考模型的第一个 token 是思考内容，
必然走文本分支，所以 `parse` 字段会标明用了哪条路径。
"""

from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# 显式答案标记：answer is C / 答案是 D / 选项：I / final answer: B
_MARK_RE = re.compile(
    r"(?:final\s+answer|answer|正确答案|答案|选项|choice|option)"
    r"\s*(?:is|are|为|是|should\s+be)?\s*[:：]?\s*[*_`$\s]*\(?\s*([A-J])(?![A-Za-z])",
    re.I)
# 独立字母，前后不能紧邻字母或数字
_LONE_RE = re.compile(r"(?<![A-Za-z0-9])([A-J])(?![A-Za-z0-9])")
_ANS_NUM_RE = re.compile(
    r"(?:final\s+answer|answer|答案为|答案|结果)\s*(?:is|are|为|是|[:：])?\s*[*_`$\s]*"
    r"(-?\d[\d,]*(?:\.\d+)?)",
    re.I)


def _clean(s: Any) -> str:
    return str(s or "").replace("\u00a0", " ").replace("\u2212", "-")


def extract_boxed(s: str) -> str | None:
    """取 \\boxed{...} 的内容，支持一层嵌套。"""
    i = s.find("\\boxed{")
    if i < 0:
        return None
    j, depth, buf = i + 7, 1, []
    while j < len(s) and depth:
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        j += 1
    return "".join(buf).strip() or None


def extract_choice(text: str, choices: str = "ABCDEFGHIJ") -> tuple[str | None, str]:
    """从自由文本里抠选项字母，返回 (字母, 命中方式)。

    顺序：短回答 → 显式标记（取最后一次）→ 末行 → 全文最后一个独立字母。
    结论通常在末尾，所以后两级取「最后一个」而不是第一个。
    """
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


def _norm_math(s: Any) -> str:
    """归一 LaTeX / 符号写法，减少「答案对但写法不同」的假阴性。"""
    s = _clean(s)
    for a in ("$", "\\(", "\\)", "\\[", "\\]", "{", "}", "\\!", "\\,", "\\;", "\\ "):
        s = s.replace(a, "")
    s = (s.replace("\\boxed", "").replace("\\text", "").replace("\\mathrm", "")
          .replace("\\times", "x").replace("\\cdot", "x")
          .replace("×", "x").replace("·", "x").replace("*", "x"))
    s = re.sub(r"\s+", "", s).lower()
    return re.sub(r"(?<![\dx])1e([-+]?\d+)", r"10^\1", s)   # 1e-4 -> 10^-4


def score_mc(resp: dict, choices: list[str]) -> dict:
    lp_map = {k: v for k, v in _first_step_logprobs(resp.get("logprobs")).items()
              if len(k) == 1 and k in choices}
    if lp_map:
        return {"chosen": max(lp_map, key=lp_map.get), "lp": lp_map, "parse": "logprob"}
    got, how = extract_choice(resp.get("text", ""), "".join(choices))
    return {"chosen": got or "?", "lp": {}, "parse": how}


def margin_of(correct: str, chosen: str, lp: dict[str, float]) -> float | None:
    """logp(正确) − max(logp(错误))。负值代表正确项被干扰项压过。"""
    if not lp or correct not in lp or chosen not in lp:
        return None
    wrongs = [v for k, v in lp.items() if k != correct]
    return round(lp[correct] - max(wrongs), 4) if wrongs else None


def score_numeric(resp: dict, answer: str) -> dict:
    t = _clean(resp.get("text", ""))
    m = _ANS_NUM_RE.search(t)
    got = m.group(1).replace(",", "") if m else (_NUM_RE.findall(t.replace(",", "")) or [None])[-1]
    exp = (_NUM_RE.findall(_clean(answer).replace(",", "")) or [None])[-1]
    ok = False
    if got is not None and exp is not None:
        try:
            ok = abs(float(got) - float(exp)) < 1e-6
        except ValueError:
            ok = got == exp
    return {"chosen": got, "correct": ok, "parse": "number"}


def score_text(resp: dict, answer: str, mode: str = "boxed") -> dict:
    """自由作答。mode=boxed 时优先取输出里的 \\boxed{}，否则用全文。"""
    raw = resp.get("text") or ""
    if not str(answer).strip():
        return {"chosen": raw[:200], "correct": None, "parse": "none"}
    cand = extract_boxed(raw) if mode == "boxed" else None
    src = cand if cand else raw
    na, nt = _norm_math(answer), _norm_math(src)
    ok = bool(na) and (na == nt or na in nt)
    return {"chosen": (cand or raw.strip())[:200], "correct": ok,
            "parse": "boxed" if cand else "text"}


def score(row: dict, resp: dict) -> dict:
    """按题目 kind 分派，返回 {correct, chosen, parse, margin?}。"""
    kind = row.get("kind", "mc")
    if kind == "mc":
        s = score_mc(resp, row.get("choices") or list("ABCD"))
        s["correct"] = s["chosen"] == row.get("answer")
        s["margin"] = margin_of(row.get("answer", ""), s["chosen"], s["lp"])
        return s
    if kind == "numeric":
        return score_numeric(resp, row.get("answer", ""))
    return score_text(resp, row.get("answer", ""), row.get("match", "boxed"))
