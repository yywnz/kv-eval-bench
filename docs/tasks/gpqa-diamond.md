# GPQA Diamond

**维度**：科学推理 · **题数**：198（全量）· **判分**：本地

## 来源

| 项 | 值 |
|---|---|
| 数据集 | [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa)（`diamond` 子集，**gated**） |
| 备用源 | [`hendrydong/GPQA_diamond`](https://huggingface.co/datasets/hendrydong/GPQA_diamond)（公开镜像，无四选项） |
| 许可 | CC BY 4.0 |
| 论文 | Rein et al., 2023, [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) |

## 关于「官方标准」

**GPQA 没有官方判分代码。** 官方只发布了数据集（gated）和论文里的评测设置：

- 题目为四选一，要求模型逐步推理，**把最终答案写在 `\boxed{}` 里**；
- 判分是「`\boxed{}` 内容与标准答案字符串比对」，不是选项字母；
- 论文报的是 Diamond 子集准确率。

所以本项无法做到「与某个官方脚本逐行对齐」。本仓库的实现是：

1. 题面保持论文写法：`What is the correct answer to this question: {question}` +
   选项块（若有）+ `Reason step by step, and put your final answer within \boxed{}.`
2. 解码：`temperature=0.5`（论文设置）。
3. 抽取：取**最后一个** `\boxed{...}`，支持任意深度嵌套。
4. 比对：归一后依次尝试——完全相等 → 数值集合（含科学计数折算，容差 0.5%）→
   百分比与小数互认 → **双向**包含（短串需 ≥3 字符）。

第 4 步的宽松化是必要的：198 题里 **127 题（64%）的标准答案带单位或文字**
（`6.24x10^-7 M`、`1 and 2`、`A = NaHSO3, B = HCl`）。只判「标准答案被输出包含」时，
模型少写单位会整批判错——实测最多造成 49pp 的低估。

## 两种数据来源的差异

`build` 先尝试官方 gated 数据，失败则回落到公开镜像：

| 来源 | 选项 | 难度 | `meta.official_options` |
|---|---|---|---|
| `Idavidrein/gpqa` | 有（四选一） | 与官方一致 | `true` |
| `hendrydong/GPQA_diamond` | **无** | 高于官方（要靠题面自己推出答案） | `false` |

无选项时分数会显著偏低，**不可与公开榜单横比**。`result.py` 会读这个字段并在
`results/{model}.json` 里标出；写报告时必须一并说明。

## 数据格式

```json
{"id": "gpqa-4c3b2a1f0e", "task": "gpqa_diamond", "kind": "open", "match": "boxed",
 "question": "...", "options": ["...", "...", "...", "..."],
 "answer": "10^-4 eV",
 "meta": {"domain": "Physics", "official_options": true}}
```

## 已知限制

- **数值等价仍是字符串+数值级的**：`0.1` 与 `1/10`、`2√3` 与 `3.46` 不会互相匹配。
  这类漏判对同模型不同配置的对比无影响，但会让绝对分偏低 5pp 量级。
- 64% 的答案是自由文本，语义等价无法靠规则穷尽。要绝对分请用官方四选一数据并照论文口径人工复核。
- 198 题的标准误约 3.5pp；逐题配对后能看住 2–3pp 的差异。
