"""RULER 题目的结构性回归。

这些不变量一旦破了，测出来的是"别的题"，而分数看上去仍然正常：

    python tests/test_ruler.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capbench.prompts import render
from capbench.ruler import DEPTHS, LENGTHS_TOK, VARIANTS
from capbench.scoring import score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ruler.jsonl"

if not DATA.exists():
    print("跳过：还没有 data/ruler.jsonl，先跑 build --tasks ruler")
    sys.exit(0)

ROWS = [json.loads(l) for l in DATA.open(encoding="utf-8")]

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name:<52} got={got!r}")
    if not ok:
        FAILED.append((name, got, want))


# ---------------------------------------------------------------- 覆盖
check("变体齐全", sorted({r["meta"]["variant"] for r in ROWS}), sorted(VARIANTS))
check("长度档齐全", sorted({r["meta"]["target_len"] for r in ROWS}), list(LENGTHS_TOK))
check("位置档齐全", sorted({r["meta"]["pos_pct"] for r in ROWS}),
      [int(d * 100) for d in DEPTHS])
check("id 唯一", len({r["id"] for r in ROWS}), len(ROWS))

# ---------------------------------------------------------------- 针的位置
worst = 0.0
for r in ROWS:
    ctx, ans = r["context"], r["answer"]
    needle = f"is: {ans}."
    i = ctx.find(needle)
    want = r["meta"]["pos_pct"] / 100
    if i < 0:
        worst = 999
        break
    # 允许插入时往后找空格的偏移：不超过 1% 的长度
    worst = max(worst, abs(i / len(ctx) - want))
check("每道题的针都在文档里且位置误差 < 2%", worst < 0.02, True)

check("multikey 是 4 针", sorted({r["meta"]["needles"] for r in ROWS
                                 if r["meta"]["variant"] == "multikey"}), [4])
check("single 是 1 针", sorted({r["meta"]["needles"] for r in ROWS
                               if r["meta"]["variant"] == "single"}), [1])

# 提问的 key 必须就是文档里那根针的 key
bad_q = 0
for r in ROWS:
    key = r["question"].split("for ")[1].split(" mentioned")[0]
    if f"for {key} is: {r['answer']}" not in r["context"]:
        bad_q += 1
check("问题里的 key 与针一致", bad_q, 0)

# ---------------------------------------------------------------- 判分
r0 = ROWS[0]
check("答对", score({"task": "ruler", "kind": "numeric", "answer": r0["answer"]},
                    {"text": f"reasoning... so it is {r0['answer']}."})["correct"], True)
check("取最后一个数字（不被中间量带偏）",
      score({"task": "ruler", "kind": "numeric", "answer": "7654321"},
            {"text": "first 1234567, finally 7654321"})["correct"], True)
check("千分位也算对",
      score({"task": "ruler", "kind": "numeric", "answer": "1234567"},
            {"text": "1,234,567"})["correct"], True)
check("答错", score({"task": "ruler", "kind": "numeric", "answer": "1234567"},
                    {"text": "9999999"})["correct"], False)
check("没有数字", score({"task": "ruler", "kind": "numeric", "answer": "1"},
                        {"text": "无法确定"})["parse"], "none")

# ---------------------------------------------------------------- 模板
p, name = render({"context": "DOCUMENT", "question": "Q?"}, "ruler", "0shot")
check("模板名", name, "ruler_0shot.txt")
check("含 <text> 块", "<text>" in p and "DOCUMENT" in p, True)
check("要求只答数字", p.rstrip().endswith("Answer with the number only."), True)

print()
if FAILED:
    print(f"{len(FAILED)} 项失败")
    for n, g, w in FAILED:
        print(f"  - {n}: got={g!r} want={w!r}")
    sys.exit(1)
print("全部通过")