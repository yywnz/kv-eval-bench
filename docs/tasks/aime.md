# aime · 数学

| | |
|---|---|
| 任务 ID | `aime` |
| 维度 | 数学 |
| 题数 | 60（2024 年 30 + 2025 年 30） |
| 来源 | [`Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) + [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025) |
| 许可 | 各自仓库许可（CC BY 4.0 / MIT） |
| 指标 | accuracy（推荐 avg@k） |
| 判分 | 本地 |
| 单题上限 | 8192 tokens |

## 测什么

竞赛数学的完整推导链。答案恒为 0–999 的整数，判分是纯数值比对——**没有抽取歧义，也没有部分分**，一步算错就是 0。它对长链推理的中间误差极其敏感，是量化损伤最直接暴露的地方。

## 为什么用两年而不是一年

2025 年单套只有 30 题，单题就是 3.3 个百分点，一次对错就能让分数跳一个档。加上 2024 年凑到 60 题，标准误从 9.1pp 降到 6.4pp（p≈0.5 时）。**仍然不够做单次结论**，所以：

- 想要稳定数字，用 avg@k：同一批题采样 k 次（temperature > 0），取平均正确率。
- 想要「能不能做出来」，用 pass@k：k 次里有一次对就算对。

两者对同一模型的排名可能完全不同，报分数时必须写明是哪个。

## 数据格式

```json
{"id": "aime-8b7a6c5d4e", "task": "aime", "kind": "numeric",
 "question": "...", "answer": "70", "max_tokens": 8192,
 "meta": {"year": 2025, "problem_id": 12, "type": ["Number Theory"]}}
```

`meta.year` 可按年份拆开看，2024 与 2025 的难度分布不同。
模板在 `prompts/numeric_0shot.txt`（另有 `numeric_cot.txt`）。

## Prompt 模板

`prompts/numeric_0shot.txt`：

```
Solve the following competition math problem. The answer is an integer from 0 to 999.

{question}

End with: Answer: <integer>
```

另有 `numeric_cot.txt`（先写推理，最后一行给整数）。

## 抽样策略

全量，不做抽样。两个源各 30 题。

## 判分口径

1. 优先取「答案 / answer」标记后的数字（`capbench/scoring.py::_ANS_NUM_RE`）。
2. 取不到则取输出里**最后一个**数字。
3. 数值比较，容差 1e-6。

取标记而不是「最后一个数字」是为了避开思考模式下的推演过程——模型常在思考里写一堆中间结果。

## 已知限制

- 60 题太少，单次运行的标准误 6pp 量级，不能用来比较相近的模型。
- 2025 年 AIME I/II 两套各 15 题；2024 年同理，两源合计 60。
- 公开题，2026 年的模型大概率已见过（污染）。要看真实水平得用当年新题。
- 判分只认最终整数，不认推导质量；过程全错但蒙对也算对。

## 引用

```bibtex
@misc{aime2025,
  title={AIME 2024 / 2025 Problems},
  howpublished={\url{https://huggingface.co/datasets/MathArena/aime_2025}},
  year={2025}
}
```
