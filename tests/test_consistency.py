"""一致性对照的逻辑回归（不联网）。

判据只有一条：**逐 token 相同才算通过**。这里锁住比对与汇总的实现。

    python tests/test_consistency.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capbench.consistency import _compare, summarize

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name:<52} got={got!r}")
    if not ok:
        FAILED.append((name, got, want))


# ---------------------------------------------------------------- 逐 token 比对
c = _compare(list("abcdef"), list("abcdef"))
check("完全相同", (c["identical"], c["first_div"]), (True, None))
check("一致前缀长度", c["prefix_ok"], 6)

c = _compare(list("abcdef"), list("abcXef"))
check("中间分歧", (c["identical"], c["first_div"]), (False, 3))
check("分歧前的前缀是对的", c["prefix_ok"], 3)

c = _compare(list("abc"), list("abcdef"))
check("短的是长的前缀 → 不算相同", c["identical"], False)
check("这时没有分歧点（只是长度不同）", c["first_div"], None)

c = _compare([], [])
check("空对空算相同", c["identical"], True)

# 真实场景：首 token 就分叉（模板/位置编码差异）
c = _compare(["The", " answer"], ["An", " answer"])
check("首 token 就分叉", c["first_div"], 0)

# ---------------------------------------------------------------- 汇总
a = [{"id": "1", "tokens": list("abcd")}, {"id": "2", "tokens": list("xy")}]
b = [{"id": "1", "tokens": list("abcd"), "cached": 100},
     {"id": "2", "tokens": list("xZ"), "cached": 0}]
s = summarize(a, b, "冷", "热")
check("题数", s["n"], 2)
check("一致题数", s["identical"], 1)
check("一致率", s["identical_rate"], 50.0)
check("缓存命中率", s["b_cache_hit_rate"], 50.0)
check("分歧位置中位数", s["first_div_median"], 1)

# 全一致 → 通过
s2 = summarize(a, [{"id": x["id"], "tokens": x["tokens"], "cached": 5} for x in a],
               "A", "B")
check("全一致 = 100%", s2["identical_rate"], 100.0)

# 缺一侧的题不参与统计（避免把"请求失败"算成一堆翻转）
s3 = summarize(a + [{"id": "3", "tokens": ["q"]}], b, "A", "B")
check("只统计两侧都有记录的题", s3["n"], 2)

print()
if FAILED:
    print(f"{len(FAILED)} 项失败")
    for n, g, w in FAILED:
        print(f"  - {n}: got={g!r} want={w!r}")
    sys.exit(1)
print("全部通过")