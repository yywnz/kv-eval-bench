# MMLU-Pro

**维度**：知识 · **题数**：500（从 12,032 题中分层抽取）· **判分**：本地

## 来源

| 项 | 值 |
|---|---|
| 数据集 | [`TIGER-Lab/MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| 原始划分 | `test`（12,032 题）/ `validation`（70 题，供 5-shot 示例） |
| 许可 | MIT |
| 论文 | Wang et al., 2024, [arXiv:2406.01574](https://arxiv.org/abs/2406.01574) |

## 官方评测口径

官方仓库（TIGER-Lab/MMLU-Pro）的 `evaluate_from_apiX.py` 采用 **5-shot CoT**：

1. 对每道题，取该学科在 `validation` 集里的 **5 条**示例（含 `cot_content`）。
2. 头部一句：`The following are multiple choice questions (with answers) about {subject}.
   Think step by step and then output the answer in the format of "The answer is (X)" at the end.`
3. 每条示例：`Question: … / Options: A. …, B. … / Answer: {cot} The answer is (X).`
4. 待测题：`Question: … / Options: … / Answer: Let's think step by step.`
5. 解码：`temperature=0.1`、`top_p=0.95`、`max_tokens` 32768（推理模型计入思考内容）。
6. 抽取：`The answer is (X)`；与标准答案字母完全一致即判对。指标为全量 accuracy。

**选项排版**：单行、逗号分隔（`A. 10, B. 40, C. 6`），与官方示例一致。
**选项过滤**：值为 `N/A` 的选项丢弃，与官方 `preprocess` 一致。

## 本仓库的抽样

- 14 个学科**分层随机**，共 500 题（每类 35–36 题），固定种子 2026。
- 5-shot 示例按学科取 `validation` 前 5 条，落在 `splits/exemplars_mmlu_pro.json`，
  与题目划分一起版本化——换机器重建后示例不变。

## 数据格式

```json
{"id": "mmlu_pro-9f8e7d6c5b", "task": "mmlu_pro", "kind": "mc",
 "question": "...",
 "options": ["...", "..."],
 "answer": "G",
 "meta": {"category": "physics", "n_options": 10, "src_index": 1215}}
```

`options` 是文本列表，`answer` 是字母（A–J）。选项数不固定：多数为 10 项，少数 3–9 项。
数据里**不含拼好的 prompt**，模板在 `prompts/mmlu_pro_header.txt`、
`mmlu_pro_example.txt`、`mmlu_pro_question.txt`。

## 与官方的偏差

| 项 | 官方 | 本仓库 |
|---|---|---|
| 题量 | 12,032 | 500（分层子集） |
| 5-shot 示例 | validation 集该学科全部 5 条 | 相同 |
| 解码 | t=0.1, top_p=0.95, 32768 | 同左；模型配置写了 `max_tokens` 则以配置为准 |
| 抽取 | `The answer is (X)` | 相同，失败时退回通用规则（`parse` 字段标明） |

题量差异意味着**分数不能与官方榜单逐个比**，但同一模型不同配置之间可比。

## 判分细节

- 官方抽取命中率记在 `summary.json` 的 `official_parse_rate`。它低于 95% 说明模型没按
  格式输出，此时分数与公开榜单不可比。
- 选项字母表按 `len(options)` 生成，十选项题答 E–J 正常计分。
