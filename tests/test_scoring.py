"""判分回归测试。

每条用例都对应官方实现里的一个具体行为，或一次实跑踩到的错。
改动 `capbench/scoring.py` 前后必须全绿：

    python tests/test_scoring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capbench.prompts import fmt_mmlu_options, render
from capbench.scoring import letters_for, score

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'}  {name:<52} got={got!r}")
    if not ok:
        FAILED.append((name, got, want))


def mc(task, text, answer, n=10, logprobs=None):
    return score({"task": task, "kind": "mc", "options": [""] * n, "answer": answer},
                 {"text": text, "logprobs": logprobs})


# ---------------------------------------------------------------- 字母表
check("10 选项 -> A-J", letters_for(10)[-1], "J")
check("4 选项 -> A-D", letters_for(4), list("ABCD"))
check("0 选项仍给 4 个", letters_for(0), list("ABCD"))

# ------------------------------------------------ MMLU-Pro 官方抽取
check("官方格式 The answer is (E)", mc("mmlu_pro", "…\nThe answer is (E).", "E")["chosen"], "E")
check("官方格式命中标记", mc("mmlu_pro", "The answer is (E)", "E")["official"], True)
check("带括号不带句点", mc("mmlu_pro", "the answer is (J)", "J")["chosen"], "J")
check("只写字母也认（兜底）", mc("mmlu_pro", "E", "E")["chosen"], "E")
check("兜底不算官方命中", mc("mmlu_pro", "E", "E")["official"], False)
check("Answer: E 也能抽", mc("mmlu_pro", "Answer: E", "E")["chosen"], "E")

# ------------------------------------------------ LongBench v2 官方抽取
check("官方格式 The correct answer is (C)",
      mc("longbench_v2", "…\nThe correct answer is (C)", "C")["chosen"], "C")
check("官方命中标记",
      mc("longbench_v2", "The correct answer is (C)", "C")["official"], True)
check("无括号也认", mc("longbench_v2", "The correct answer is B", "B")["chosen"], "B")

# ------------------------------------------------ logprob 不夺权
lp = {"content": [{"top_logprobs": [{"token": "H", "logprob": -0.1},
                                    {"token": "E", "logprob": -2.0}]}]}
r = mc("mmlu_pro", "The answer is (E)", "E", logprobs=lp)
check("正文优先于 logprob", r["chosen"], "E")
check("logprob 仍算 margin", r["margin"] is not None, True)
r = mc("mmlu_pro", "无法判断", "E", logprobs={"content": [{"top_logprobs": [
    {"token": "E", "logprob": -0.3}]}]})
check("正文无效才用 logprob", r["parse"], "logprob")

# ------------------------------------------------ 模板
check("MMLU 选项单行逗号", fmt_mmlu_options(["x", "y"]), "A. x, B. y")

p, name = render({"question": "Q", "options": ["x", "y"]}, "mmlu", "0shot",
                 exemplars={"physics": []})
check("mmlu 0shot 模板名", name, "mmlu_pro_0shot")
check("mmlu 0shot 不含示例", "The answer is" in p, True)

ex = {"physics": [{"question": "e", "options": ["a"], "answer": "A",
                   "cot": "reason"}]}
p, name = render({"question": "Q", "options": ["x"], "meta": {"category": "physics"}},
                 "mmlu", "5shot", exemplars=ex)
check("mmlu 5shot 模板名", name, "mmlu_pro_5shot")
check("mmlu 5shot 含示例答案", "The answer is (A)." in p, True)

p, _ = render({"context": "DOC", "question": "Q",
               "options": ["a", "b", "c", "d"]}, "longctx", "0shot")
check("LongBench 官方结尾", p.endswith('"The correct answer is (insert answer here)".'), True)
check("LongBench 选项带括号", "(A) a" in p, True)

print()
if FAILED:
    print(f"{len(FAILED)} 项失败")
    for n, g, w in FAILED:
        print(f"  - {n}: got={g!r} want={w!r}")
    sys.exit(1)
print("全部通过")
