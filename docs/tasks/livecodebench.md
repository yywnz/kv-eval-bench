# livecodebench · 代码

| | |
|---|---|
| 任务 ID | `livecodebench` |
| 维度 | 代码 |
| 题数 | 150（源自 release_v6，按时间取最近；覆盖 2025-01 ~ 2025-04） |
| 来源 | [`livecodebench/code_generation_lite`](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| 许可 | MIT |
| 指标 | pass@1（执行官方测试用例） |
| 判分 | **外部**，需沙盒执行 |
| 单题上限 | 4096 tokens |

## 测什么

竞赛编程。题面来自 LeetCode / AtCoder / Codeforces 的**真实比赛**，每题带官方测试用例（含未公开的 private cases）。判分是实际编译运行、比对输出——不是相似度，也不是 LLM 打分。

代码是量化损伤的第二报警器（第一是多选）：括号、缩进、`return` 这类低概率 token 一旦选错，编译直接失败。

## 为什么能抗污染

按月份持续发布新题（`release_v1` … `release_v6`），每题带 `contest_date`。默认取**最近 150 题**。本 release 覆盖 2025-01 至 2025-04 的比赛（`test6.jsonl` 共 175 题），对训练数据截止在 2025 年中的模型仍有污染可能；换更新的 release 需改 `capbench/tasks.py` 里的 `test6.jsonl`。

报分数时应同时声明 `release` 版本与 `contest_date` 区间，否则不同实现之间不可比——同一个模型在 v4 和 v6 上的分数能差 10pp 以上。

## 数据格式

```json
{"id": "lcb-a1b2c3d4e5", "task": "livecodebench", "kind": "code",
 "question": "...", "answer": "", "max_tokens": 4096,
 "meta": {"question_id": "abc123", "title": "...", "platform": "atcoder",
          "difficulty": "hard", "contest_date": "2025-09-14T00:00:00",
          "starter_code": "...", "public_test_cases": "[...]", "private_test_cases": "..."}}
```

`answer` 为空串是**故意的**：判定需要执行测试用例，本地不判分。
模板在 `prompts/code_0shot.txt`。

## Prompt 模板

`prompts/code_0shot.txt`：

```
Solve the following competitive programming problem in Python.

{question}
```

原始题面已含输入输出格式，未追加代码模板（`starter_code` 保留在 `meta`）。

## 抽样策略

按 `contest_date` 倒序取最近 150 题，不做类别平衡。

> 这里**刻意不做分层**：按平台/难度分层会引入旧题（更早的比赛），削弱抗污染性。时间优先是这项任务的核心设计。

## 判分口径

```bash
lm_eval --model local-completions \
  --model_args base_url=http://127.0.0.1:30000/v1,model=<name>,tokenized_requests=False \
  --tasks livecodebench --num_fewshot 0 --batch_size 8
```

要严格用本仓库这 150 题（而不是 lm_eval 自带的题库），需要自己起沙盒：解析 `private_test_cases`、执行、比对输出。本仓库提供用例但不提供执行环境。

## 已知限制

- 只测竞赛编程，**测不到工程能力**：读真实仓库、改多文件、跑通既有测试，全不在覆盖范围。
- Python 单语言；模型若只擅长其他语言会低估。
- 沙盒执行有安全与资源开销，慢（一大半时间花在编译运行）。
- pass@1 vs pass@k 差别很大：实际使用可以反复重试，pass@1 偏悲观。

## 引用

```bibtex
@misc{jain2024livecodebench,
  title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  author={Jain, Naman and others},
  year={2024}, eprint={2403.07974}, archivePrefix={arXiv}
}
```
