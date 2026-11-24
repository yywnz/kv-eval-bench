# IFEval

**维度**：指令遵循 · **题数**：541（全量）· **判分**：外部（官方 verifier）

## 来源

| 项 | 值 |
|---|---|
| 数据集 | [`google/IFEval`](https://huggingface.co/datasets/google/IFEval) |
| 官方实现 | [`google-research/instruction_following_eval`](https://github.com/google-research/google-research/tree/master/instruction_following_eval) |
| 许可 | Apache-2.0 |
| 论文 | Zhou et al., 2023, [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) |

## 官方评测口径

IFEval 是四项里唯一有**官方可执行校验器**的——判分不是比对字符串，而是按
`instruction_id_list` 里每个约束跑一段检查代码（24 类：字数、大小写、禁用词、
JSON 格式、标题格式、语言等）。

官方指标是两个维度叉乘：

| 指标 | 含义 |
|---|---|
| prompt-level strict | 一道题的**全部**约束都满足才算对 |
| prompt-level loose | 同上，但允许小范围格式差异 |
| instruction-level strict | 按**单条约束**计：满足的约束数 / 总约束数 |
| instruction-level loose | 同上，宽松版 |

本仓库**不重新实现这些校验器**——24 类约束的边界条件很多，自己写只会引入 bug。
判分交给 `lm_eval` 的 `ifeval` 任务，它内置官方实现。

## 怎么跑

```bash
lm_eval --model local-completions \
  --model_args base_url=http://127.0.0.1:30000/v1,model=<model>,tokenized_requests=False \
  --tasks ifeval --num_fewshot 0 --batch_size 32
```

`python -m capbench check` 会把这条命令连同当前端点直接打印出来。

`python pred.py` 遇到本任务会**跳过**（`judge: external`），因为本地没有判分器，
跑了也出不了分。若只需要原始回答，加 `--tasks ifeval` 单独跑并保留 `pred/` 里的输出，
再手工送进官方校验器。

## 数据格式

```json
{"id": "ifeval-1a2b3c4d5e", "task": "ifeval", "kind": "instruct",
 "question": "...", "answer": "",
 "meta": {"key": 1000,
          "instruction_id_list": ["length_constraints:number_words", "punctuation:no_comma"],
          "kwargs": [{...}],
          "n_constraints": 2}}
```

`answer` 为空串是**故意的**：约束是否满足只能由校验器判定，没有「标准答案字符串」。

## 与官方的偏差

| 项 | 官方 | 本仓库 |
|---|---|---|
| 题量 | 541 | 541（全量） |
| 判分 | 官方校验器 | 同一实现，通过 `lm_eval` 调用 |
| prompt | 原文直接送入 | 相同（`prompts/instruct_0shot.txt` 只含 `{question}`） |

**唯一需要对齐的是 chat template**：IFEval 的约束里有「全大写」「不加逗号」这类
对表面形式敏感的要求，system prompt 或模板里多一句话都可能让模型改变行为。
报告时须写明是否加了 system prompt。
