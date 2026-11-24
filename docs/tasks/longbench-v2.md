# longbench_v2 · 长上下文

| | |
|---|---|
| 任务 ID | `longbench_v2` |
| 维度 | 长上下文 |
| 题数 | 200（源自 503） |
| 来源 | [`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2) |
| 许可 | MIT |
| 指标 | accuracy（四选一） |
| 判分 | 本地 |
| 单题上限 | 1024 tokens |

## 测什么

长文里的**深理解**，不是检索。题目由近 100 名高学历标注者出，上下文 8k–2M words，覆盖六类任务：

| domain | 含义 |
|---|---|
| Single-Document QA | 单文档问答 |
| Multi-Document QA | 多文档聚合 |
| Long In-context Learning | 长文里的示例学习 |
| Long-dialogue History Understanding | 长对话历史 |
| Code Repository Understanding | 代码仓库理解 |
| Long Structured Data Understanding | 长表格 / 结构化数据 |

**最难的设计在出题端**：先用三个模型自动初筛，三个模型都能答对的题直接淘汰。所以每道题至少有一个模型答错。

## 数据格式

```json
{"id": "lbv2-0f1e2d3c4b", "task": "longbench_v2", "kind": "mc",
 "context": "...", "question": "...", "options": ["...", "...", "...", "..."],
 "answer": "C", "max_tokens": 1024,
 "meta": {"domain": "Single-Document QA", "sub_domain": "...", "length": "medium",
          "difficulty": "hard", "ctx_chars": 412873}}
```

`context` 是独立字段，不是拼进 prompt 的字符串——模板在 `prompts/longctx_0shot.txt`，
渲染时才组装。`meta.ctx_chars` 用于超窗预检。

## Prompt 模板

`prompts/longctx_0shot.txt`：把文档、问题、选项依次拼好。

另有 `longctx_no_context.txt`——**只给问题和选项、不给文档**，用于对照：

```bash
python pred.py --model <name> --tasks longbench_v2 --prompt-variant no_context
```

它的分数就是「不读文档」的基线；与正常分数的差值才是文档带来的真实增益。
两者接近就说明这项任务能被常识蒙过去。

## 抽样策略

按 `domain × length` 分层（6 × 3 = 18 格，其中 `Long-dialogue History | long` 在源数据中为空，实际 17 格），每格**蓄水池随机抽 14 题**，总量上限 200。

> 重做抽样时改的两点：
> 1. 原来是「每格取前 12 条」，受源文件顺序影响；现在是蓄水池抽样，格内均匀且可复现。
> 2. 每格从 12 提到 14，总量 150 → 200，让中长档的题量够做配对比较。

`--per-cell` 可调。**不建议按长度档再砍**，除非你的上下文窗口装不下——砍了要重新算分母，报分数时说明。

## 判分口径

同 `mmlu_pro`：优先首 token logprob，拿不到走文本解析（`parse` 字段标明）。四选一随机基线 25%。

## 已知限制

- **全是四选一**，可以靠排除法和选项风格蒙分，不等于「读懂了」。要测有效窗口请配 RULER 类的变长检索任务。
- 静态题库 + 公开数据，2024 年 12 月发布，**污染风险实际很高**。纵向对比有效，跨模型绝对排名要谨慎。
- 单题最长约 2M words（约 300 万 token），**任何模型的窗口都装不下 Long 档的一部分**。超窗会直接报错，而报错题不进准确率分母——必须看 `n_error`，否则会拿到一个「只测了半套题」的分数。
- 503 题本身样本不大；抽 200 后，四选一下的标准误约 3.2pp（p≈0.5 时）。

## 与「以前从 LongBench v2 抽题」的关系

早期做法是从这 503 题里随手抽一批做回归。那份抽样的问题是：格内不均匀、没记录 id、换一次就换了题，历史结果无法配对。

本任务把抽到的 200 条 **id 全部写进 `splits/longbench_v2.json`**，重建时逐题比对即可确认是同一批题——这是做纵向对比的前提。

## 引用

```bibtex
@misc{bai2024longbenchv2,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks},
  author={Bai, Yushi and others},
  year={2024}, eprint={2412.15204}, archivePrefix={arXiv}
}
```
