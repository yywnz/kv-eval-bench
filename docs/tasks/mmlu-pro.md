# mmlu_pro · 知识

| | |
|---|---|
| 任务 ID | `mmlu_pro` |
| 维度 | 知识 |
| 题数 | 500（源自 12,032） |
| 来源 | [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| 许可 | MIT |
| 指标 | accuracy（首 token 选项 / 文本解析） |
| 判分 | 本地 |
| 单题上限 | 1024 tokens |

## 测什么

学科知识记忆与辨析。14 个学科：biology、business、chemistry、computer science、economics、engineering、health、history、law、math、other、philosophy、physics、psychology。

只考知识，不考推理深度——错题多是「没记住」而不是「想错了」。用它代替已经饱和的 MMLU：MMLU 在 2026 年的前沿模型上普遍 88%+，分差被压到噪声级别；MMLU-Pro 换成 10 选项为主、去掉简单题，仍有区分度。

## 数据格式

```json
{"id": "mmlu_pro-9f8e7d6c5b", "task": "mmlu_pro", "kind": "mc",
 "question": "...", "options": ["...", "...", "..."], "answer": "I", "max_tokens": 1024,
 "meta": {"category": "physics", "n_options": 10, "src_index": 1215}}
```

选项数不固定：413 题为 10 选项，其余为 3–9 选项。`options` 是文本列表，`answer` 是字母，
渲染时按 A/B/C… 自动编号。数据里**不含拼好的 prompt**——模板在 `prompts/mc_0shot.txt`。

## Prompt 模板

`prompts/mc_0shot.txt`：

```
Answer the following multiple-choice question. Reply with the single letter of the correct option.

Question: {question}

{options}

Answer:
```

另有 `mc_cot.txt`（要求先推理，最后一行给 `Answer: <letter>`）。换模板不用重建数据。

## 抽样策略

按 `category` 分层**均分** 500 题：14 类中 10 类各 36 题、4 类各 35 题（`divmod(500, 14)`）。类内用种子 2026 **随机抽**，不是取前 N 条。

> 这一点是重做抽样时改的：按流式顺序取前 N 会在类内引入系统性偏差（源文件往往按难度或来源排序），随机抽样让每次重建的结果一致且无偏。

## 判分口径

1. **优先 logprob**：读第一个生成 token 的 `top_logprobs`，在 `choices` 里取概率最大者。答案只差一个 token 时这是最灵敏的读法。
2. **退回文本解析**：拿不到 logprob（或模型带思考、首 token 是思考内容）时，从输出里抠字母，四级策略见 `capbench/scoring.py::extract_choice`，`raw.jsonl` 的 `parse` 字段标明用了哪一级。
3. **margin**：`logp(正确) − max(logp(错误))`。答案没变但 margin 系统性收窄，通常早于准确率下降出现——这是量化检测里最好用的早期信号。思考模型下该字段为 null。

## 已知限制

- 公开题库，存在训练集污染风险（MMLU-Pro 2024 年发布）。
- 选项顺序未重排：源数据正确答案分布本身较均匀，未做额外打乱，但若你要与其他实现对齐分数，需确认对方的选项处理方式。
- 10 选项使随机基线约 10%，比四选一的 25% 低，同样题量下统计效率更高。

## 引用

```bibtex
@misc{wang2024mmlupro,
  title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author={Wang, Yubo and others},
  year={2024}, eprint={2406.01574}, archivePrefix={arXiv}
}
```
