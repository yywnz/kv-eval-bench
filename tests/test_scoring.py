"""判分回归测试。

每个用例都对应一次实跑踩到的错，改动 scoring.py 前后必须全绿：

    python tests/test_scoring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capbench.scoring import (answers_equal, extract_boxed, letters_for,
                              score, score_text)

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name:<56} got={got!r}")
    if not ok:
        FAILED.append((name, got, want))


def mc(text, answer, n_opts=10, logprobs=None):
    return score({"kind": "mc", "options": [""] * n_opts, "answer": answer},
                 {"text": text, "logprobs": logprobs})


# ---------------------------------------------------------------- 字母表
check("10 选项生成 A-J", letters_for(10)[-1], "J")
check("4 选项生成 A-D", letters_for(4), list("ABCD"))
check("0 选项仍给 4 个", letters_for(0), list("ABCD"))

# ---------------------------------------------------------------- 多选
check("正文只写 E（10 选项题）", mc("E", "E")["chosen"], "E")
check("正文只写 J", mc("J", "J")["chosen"], "J")
check("Answer: E", mc("Let me think.\n\nAnswer: E", "E")["chosen"], "E")
check("答案 G 不被截成 A", mc("G", "G")["correct"], True)

# 正文优先于 logprob：模型正文写 E，首 token logprob 偏向 H
lp = {"content": [{"top_logprobs": [{"token": "H", "logprob": -0.1},
                                    {"token": "E", "logprob": -2.0}]}]}
r = mc("E", "E", logprobs=lp)
check("正文优先，logprob 不夺权", r["chosen"], "E")
check("logprob 仍用于 margin", r["margin"] is not None, True)

# 正文抠不出来才退回 logprob
r = mc("无法确定", "E", logprobs={"content": [{"top_logprobs": [
    {"token": "E", "logprob": -0.3}]}]})
check("正文无效时退回 logprob", r["parse"], "logprob")

# ---------------------------------------------------------------- boxed
check("取最后一个 boxed", extract_boxed(r"先算 $x=\boxed{3}$，故 $\boxed{42}$"), "42")
check("嵌套 \\text", extract_boxed(r"$\boxed{1,\ \text{and }2}$"), r"1,\ \text{and }2")
check("无 boxed 返回 None", extract_boxed("答案是 42"), None)

# ---------------------------------------------------------------- 自由作答
check("集合写法 1,2,4", answers_equal("1,2,4", r"1,\ 2,\ \text{and }4"), True)
check("百分比 44% vs 0.439", answers_equal("44%", "0.439"), True)
check("百分比 44% vs 0.44", answers_equal("44%", "0.44"), True)
check("单位缺省 10^-4 eV", answers_equal("10^-4 eV", "10^{-4}"), True)
check("科学计数 6.24x10^-7 M", answers_equal("6.24x10^-7 M", "6.24e-7"), True)
check("自由文本包含", answers_equal("cytoplasmic localization of the red signal",
                              "Answer: cytoplasmic localization of the red signal."), True)
check("数值不同必须判错", answers_equal("42", "43"), False)
check("短串不做包含（1 vs 10）", answers_equal("1", "10"), False)
check("空答案不误判", answers_equal("", "42"), False)

# score_text：模型多写了解释仍能判对
r = score_text({"text": r"推导如下……最终 $\boxed{0.439}$"}, "44%")
check("score_text 百分比", r["correct"], True)

print()
if FAILED:
    print(f"{len(FAILED)} 项失败")
    for n, g, w in FAILED:
        print(f"  - {n}: got={g!r} want={w!r}")
    sys.exit(1)
print("全部通过")
