"""判分。

三道题型的口径分别写在 docs/tasks/*.md 里，这里是实现。

三条原则，都是被实跑错误逼出来的：

1. **多选题以正文为准，logprob 只用来算 margin。** 思考模型的第一个 token 是思考
   内容，拿它定选项会随机命中；正文里模型自己写的字母才是它的答案。
2. **选项字母按 `options` 长度生成。** 十选项题（A–J）不能只认 A–D。
3. **自由作答要宽松归一。** GPQA 有 64% 的标准答案带单位或文字
   （`6.24x10^-7 M`、`1 and 2`、`A = NaHSO3, B = HCl`），严格等号会把
   「模型对了但少写单位」判成错。归一后依次尝试：完全相等 → 数值集合 →
   百分比互认 → 双向包含。
"""

from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# 显式答案标记：answer is C / 答案是 D / 选项：I / final answer: B
_MARK_RE = re.compile(
    r"(?:final\s+answer|answer|正确答案|答案|选项|choice|option)"
    r"\s*(?:is|are|为|是|should\s+be)?\s*[:：]?\s*[*_`$\s]*\(?\s*([A-Z])(?![A-Za-z])",
    re.I)
# 独立字母，前后不能紧邻字母或数字
_LONE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9])")
_ANS_NUM_RE = re.compile(
    r"(?:final\s+answer|answer|答案为|答案|结果)\s*(?:is|are|为|是|[:：])?\s*[*_`$\s]*"
    r"(-?\d[\d,]*(?:\.\d+)?)",
    re.I)
# 科学计数：6.24x10^-7 / 6.24 \times 10^{-7} / 10^-4
_SCI_MANT = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?")
_SCI_BARE = re.compile(r"(?<![\d.])10\s*\^?\s*\{?\s*(-?\d+)\s*\}?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def letters_for(n: int) -> list[str]:
    """按选项个数生成字母表；至少 4 个，最多 26 个。"""
    return [chr(ord("A") + i) for i in range(max(4, min(n, 26)))]


def _clean(s: Any) -> str:
    return str(s or "").replace("\u00a0", " ").replace("\u2212", "-")


def extract_boxed(s: str) -> str | None:
    """取**最后一个** \\boxed{...} 的内容，支持任意深度嵌套。

    取最后一个而不是第一个：模型常在中间步骤也写 boxed（例如先求中间量），
    第一个往往不是最终答案。
    """
    s = _clean(s)
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", s)]
    if not starts:
        return None
    for i in reversed(starts):
        j = s.find("{", i)
        depth, buf = 1, []
        j += 1
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
        got = "".join(buf).strip()
        if got:
            return got
    return None


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
    for a in ("$", "\\(", "\\)", "\\[", "\\]", "{", "}", "\\!", "\\,", "\\;",
              "\\ ", "\\quad", "\\left", "\\right"):
        s = s.replace(a, "")
    s = (s.replace("\\boxed", "").replace("\\text", " ").replace("\\mathrm", "")
          .replace("\\times", "x").replace("\\cdot", "x")
          .replace("×", "x").replace("·", "x").replace("*", "x")
          .replace("\\%", "%"))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r"(?<![\dx])1e([-+]?\d+)", r"10^\1", s)   # 1e-4 -> 10^-4


def _numbers(s: str) -> list[float]:
    """抽出所有数值，科学计数先折算成实数。"""
    s = _SCI_MANT.sub(lambda m: f" {float(m.group(1)) * 10 ** int(m.group(2)):.10g} ", s)
    s = _SCI_BARE.sub(lambda m: f" {10 ** int(m.group(1)):.10g} ", s)
    s = _THOUSANDS.sub("", s).replace(",", " ")
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, 0.005 * max(abs(a), abs(b)))


def _same_numbers(ga: list[float], ca: list[float]) -> bool:
    if not ga or len(ga) != len(ca):
        return False
    return all(_close(x, y) for x, y in zip(sorted(ga), sorted(ca)))


def answers_equal(gold: Any, cand: Any) -> bool:
    """自由作答的答案比对。宁可放过「写法不同」，也不放过「数值不同」。"""
    na, nc = _norm_math(gold), _norm_math(cand)
    if not na or not nc:
        return False
    if na == nc:
        return True

    ga, ca = _numbers(na), _numbers(nc)
    if _same_numbers(ga, ca):
        return True
    # 百分比与小数互认：44% vs 0.439
    if ga and ca:
        if "%" in na and "%" not in nc and _close(ga[0] / 100, ca[0]):
            return True
        if "%" in nc and "%" not in na and _close(ga[0], ca[0] / 100):
            return True
    # 双向包含：模型少写单位（10^-4 vs 10^-4 eV）或多写解释
    short, long = sorted((na, nc), key=len)
    if len(short) >= 3 and short in long:
        return True
    return False


def score_mc(resp: dict, choices: list[str]) -> dict:
    """多选：正文优先定选项，logprob 只用于 margin。"""
    letters = "".join(choices)
    got, how = extract_choice(resp.get("text", ""), letters)
    lp_map = {k: v for k, v in _first_step_logprobs(resp.get("logprobs")).items()
              if len(k) == 1 and k in set(letters)}
    if got:
        return {"chosen": got, "lp": lp_map, "parse": how}
    if lp_map:                       # 正文里抠不出来才退回 logprob
        return {"chosen": max(lp_map, key=lp_map.get), "lp": lp_map, "parse": "logprob"}
    return {"chosen": "?", "lp": lp_map, "parse": how}


def margin_of(correct: str, chosen: str, lp: dict[str, float]) -> float | None:
    """logp(正确) − max(logp(错误))。负值代表正确项被干扰项压过。"""
    if not lp or correct not in lp:
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
            ok = _close(float(got), float(exp))
        except ValueError:
            ok = got == exp
    return {"chosen": got, "correct": ok, "parse": "number"}


def score_text(resp: dict, answer: str, mode: str = "boxed") -> dict:
    """自由作答。mode=boxed 时优先取最后一个 \\boxed{}，否则用全文。"""
    raw = resp.get("text") or ""
    if not str(answer).strip():
        return {"chosen": raw[:200], "correct": None, "parse": "none"}
    cand = extract_boxed(raw) if mode == "boxed" else None
    src = cand if cand else raw
    ok = answers_equal(answer, src)
    return {"chosen": (cand or raw.strip())[:200], "correct": ok,
            "parse": "boxed" if cand else "text"}


def score(row: dict, resp: dict) -> dict:
    """按题目 kind 分派，返回 {correct, chosen, parse, margin?}。"""
    kind = row.get("kind", "mc")
    if kind == "mc":
        opts = row.get("options") or row.get("choices") or []
        s = score_mc(resp, letters_for(len(opts)))
        s["correct"] = s["chosen"] == row.get("answer")
        s["margin"] = margin_of(row.get("answer", ""), s["chosen"], s["lp"])
        return s
    if kind == "numeric":
        return score_numeric(resp, row.get("answer", ""))
    return score_text(resp, row.get("answer", ""), row.get("match", "boxed"))
