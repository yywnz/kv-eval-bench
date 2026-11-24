"""配对统计。"""

from __future__ import annotations

import math


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def se_pp(p: float, n: int) -> float:
    """比例的标准误，单位百分点。"""
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) * 100.0


def mcnemar(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）。b = 基线对/本次错，c = 反向。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired(base: list[dict], new: list[dict]) -> dict:
    """按 item_id 配对，返回准确率、翻转数与显著性。

    只有两边都判过分的题才参与配对，避免把「一边报错」算成翻转。
    """
    b = {r["item_id"]: r for r in base if r.get("scored")}
    n = {r["item_id"]: r for r in new if r.get("scored")}
    ids = sorted(set(b) & set(n))
    if not ids:
        return {"n_paired": 0}
    acc_b = sum(1 for i in ids if b[i]["correct"]) / len(ids) * 100
    acc_n = sum(1 for i in ids if n[i]["correct"]) / len(ids) * 100
    bad = sum(1 for i in ids if b[i]["correct"] and not n[i]["correct"])
    good = sum(1 for i in ids if not b[i]["correct"] and n[i]["correct"])

    mb = [b[i]["margin"] for i in ids if b[i].get("margin") is not None]
    mn = [n[i]["margin"] for i in ids if n[i].get("margin") is not None]
    shrink = None
    if mb and mn and mean(mb) != 0:
        shrink = round((mean(mb) - mean(mn)) / abs(mean(mb)) * 100, 1)

    return {
        "n_paired": len(ids),
        "acc_base": round(acc_b, 2),
        "acc_new": round(acc_n, 2),
        "delta_pp": round(acc_n - acc_b, 2),
        "flip_bad": bad,
        "flip_good": good,
        "mcnemar_p": round(mcnemar(bad, good), 4),
        "margin_shrink_pct": shrink,
        "se_pp": round(se_pp(acc_n / 100, len(ids)), 2),
    }
