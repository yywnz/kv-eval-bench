"""抽样回归测试。

对应 `capbench/tasks.py::reservoir_by_cell`。这套划分一旦发出去就是「同一批题」
的凭据，抽样逻辑改动前后必须全绿：

    python tests/test_sampling.py

核心那条：**每个 item 都新建同种子的 RNG** 会让 `randrange` 退化成
`floor(u·n)`，只覆盖数据流前 per_cell/u 道题——2026-11 的 v2.0 就踩了这个坑。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capbench.tasks import reservoir_by_cell

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name:<52} got={got!r}")
    if not ok:
        FAILED.append((name, got, want))


def run(n, per_cell=14, seed=2026, cells=1):
    """cells 个分层格，每格 n 道题，题号从 1 开始。"""
    stream = ((("dom", f"len{i}"), idx) for i in range(cells) for idx in range(1, n + 1))
    got, seen = reservoir_by_cell(stream, per_cell, seed)
    return got, seen


# 1. 确定性：同 seed 两次抽出来必须一样
a, _ = run(300)
b, _ = run(300)
check("同 seed 结果一致", sorted(a[("dom", "len0")]) == sorted(b[("dom", "len0")]), True)

c, _ = run(300, seed=7)
check("换 seed 结果不同", sorted(a[("dom", "len0")]) != sorted(c[("dom", "len0")]), True)

# 2. 覆盖尾部：修好之后必须能抽到数据流靠后的题
picked = a[("dom", "len0")]
check("抽满 per_cell", len(picked), 14)
check("能覆盖到后半段（下标 > 150）", any(i > 150 for i in picked), True)
check("下限不越界", min(picked) >= 1 and max(picked) <= 300, True)

# 3. 边界：格子不足 per_cell 时全取
small, seen_small = run(5, per_cell=14)
check("格子不足时全取", len(small[("dom", "len0")]), 5)
check("seen 计数正确", seen_small[("dom", "len0")], 5)

# 4. 均匀性：多次试验后，前后两半入选比例应接近
hits = {i: 0 for i in range(1, 301)}
trials = 400
for t in range(trials):
    got, _ = run(300, seed=t)
    for i in got[("dom", "len0")]:
        hits[i] += 1
front = sum(v for i, v in hits.items() if i <= 150)
back = sum(v for i, v in hits.items() if i > 150)
total = sum(hits.values()) or 1
check("前半段入选占比在 45–55%", 0.45 <= front / total <= 0.55, True)
check("后半段入选占比在 45–55%", 0.45 <= back / total <= 0.55, True)
check("所有题都可能入选", all(v > 0 for v in hits.values()), True)

# 5. 各格独立：多加一道题不能把别格的结果带偏
got1, _ = run(120, cells=2)
stream = [(("dom", "len0"), i) for i in range(1, 121)] + \
         [(("dom", "len1"), i) for i in range(1, 71)]
mixed, _ = reservoir_by_cell(iter(stream), 14, 2026)
check("格与格互不干扰",
      sorted(mixed[("dom", "len0")]), sorted(got1[("dom", "len0")]))

print()
if FAILED:
    print(f"{len(FAILED)} 项失败")
    for n, g, w in FAILED:
        print(f"  - {n}: got={g!r} want={w!r}")
    sys.exit(1)
print("全部通过")
