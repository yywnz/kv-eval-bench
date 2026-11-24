# gpqa_diamond · 科学推理

| | |
|---|---|
| 任务 ID | `gpqa_diamond` |
| 维度 | 科学推理 |
| 题数 | 198（全量） |
| 来源 | [`hendrydong/GPQA_diamond`](https://huggingface.co/datasets/hendrydong/GPQA_diamond)（`Idavidrein/gpqa` 的社区镜像） |
| 许可 | CC BY 4.0 |
| 指标 | accuracy（答案归一后匹配） |
| 判分 | 本地 |
| 单题上限 | 8192 tokens |

## 测什么

博士级物理、化学、生物的多步推导。题目由领域专家出题并交叉审核，人类专家在不查资料时约 65%、限时约 50%。它是**目前对数值扰动最敏感**的一项：答案常是带有效数字的表达式（`10^-4 eV`、`3.7 M`），量化造成的低概率 token 偏差会直接反映在数字上。

## 关于「自由作答」而不是四选一

官方 `Idavidrein/gpqa` 需要申请权限，本仓库用的是社区镜像，只提供题干与解答，没有四个选项。

这**不是降级**：GPQA 原论文的评测协议就是自由作答 + 从输出中抽取答案，四选一是后续社区为了省事做的简化。自由作答的难度更高（要自己给出表达式，而不是认出来），也更接近实际使用。

判分从解答的 `\boxed{}` 里取标准答案（198 题全部成功提取，0 回退），模型输出同样要求放进 `\boxed{}`。

## 数据格式

```json
{"id": "gpqa-4c3b2a1f0e", "task": "gpqa_diamond", "kind": "open", "match": "boxed",
 "question": "...", "answer": "10^-4 eV", "max_tokens": 8192,
 "meta": {"domain": "Physics", "answer_from_boxed": true}}
```

模板在 `prompts/open_0shot.txt`，要求把最终答案放进 `\boxed{}`。

## Prompt 模板

`prompts/open_0shot.txt`：

```
Answer the following graduate-level science question. Put the final answer in \boxed{}.

{question}
```

## 抽样策略

全量 198 题（Diamond 子集），不抽样。

## 判分口径

1. 从输出里取 `\boxed{...}` 内容；取不到则用全文。
2. 归一化：去 `$`、`\(`、`\)`、花括号、`\,` 等排版符；`\times`/`×`/`*` 统一为 `x`；`1e-4` 写成 `10^-4`；去空格、转小写。
3. 归一后「相等」或「答案被输出包含」即判对。

归一化规则在 `capbench/scoring.py::_norm_math`，改它会改变历史分数，动手前先想清楚。

## 已知限制

- **数值等价判断是字符串级的**，`0.1` 与 `1/10`、`2√3` 与 `3.46` 不会互相匹配。分数因此偏低，但**纵向对比仍然有效**。
- 自由作答需要模型遵守 `\boxed{}` 约定；不遵守时退化为全文匹配，容易漏判。`raw.jsonl` 的 `parse` 字段会标明 `boxed` 还是 `text`。
- 官方四选一版本需自行申请 `Idavidrein/gpqa` 后替换本任务的数据构建函数。
- 198 题，标准误约 3.5pp（p≈0.5 时），单次结果不足以分辨 3pp 以内的差异，配合配对比较使用。

## 引用

```bibtex
@misc{rein2023gpqa,
  title={GPQA: A Graduate-Level Google-Proof Q\&A Benchmark},
  author={Rein, David and others},
  year={2023}, eprint={2311.12022}, archivePrefix={arXiv}
}
```
