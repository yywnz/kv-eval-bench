# ifeval · 指令遵循

| | |
|---|---|
| 任务 ID | `ifeval` |
| 维度 | 指令遵循 |
| 题数 | 541（全量） |
| 来源 | [`google/IFEval`](https://huggingface.co/datasets/google/IFEval) |
| 许可 | Apache-2.0 |
| 指标 | prompt-level strict accuracy（全部约束同时满足才算对） |
| 判分 | **外部**，用官方 verifier / lm_eval |
| 单题上限 | 1024 tokens |

## 测什么

「听不听话」，不是「答得对不对」。题面里带若干条**可程序验证**的约束，例如：

- 输出至少 300 词，且不许出现逗号
- 全文用大写
- 用 JSON 格式返回，含 3 个指定字段
- 提到某个关键词恰好 2 次
- 结尾必须是给定句子

共 25 类约束（`instruction_id_list`），每题的约束条数不等。这项测的是模型能否在长输出中**全程保持**格式约束——量化、换 backend、换 chat template 之后，最容易在这里先崩。

## 数据格式

```json
{"id": "ifeval-1a2b3c4d5e", "task": "ifeval", "kind": "instruct",
 "question": "...", "answer": "",
 "max_tokens": 1024,
 "meta": {"key": 1000, "instruction_id_list": ["length_constraints:number_words", "punctuation:no_comma"],
          "kwargs": [{...}], "n_constraints": 2}}
```

`answer` 为空串是**故意的**：约束校验需要按 25 类规则逐条执行，交给官方实现比自行重写可靠。
数据里**不含 prompt**——模板在 `prompts/instruct_0shot.txt`，生成时才拼接。

## Prompt 模板

`prompts/instruct_0shot.txt`，内容就是 `{question}`——原始 prompt 原样送入，不加 system、不加 few-shot。

## 抽样策略

全量 541 题，不做抽样——这项的价值在于覆盖全部约束类型，抽样会破坏覆盖面。

## 判分口径

```bash
lm_eval --model local-completions \
  --model_args base_url=http://127.0.0.1:30000/v1,model=<name>,tokenized_requests=False \
  --tasks ifeval --num_fewshot 0 --batch_size 32
```

注意 lm_eval 用的是它自己下载的题库，与本仓库的 `data/ifeval.jsonl` 题面一致但**不共享文件**。本仓库这份用于离线查看、人工抽检、以及核对 id。

## 已知限制

- 单轮、单题，测不到多轮指令保持。
- 约束是「可程序验证」的，因此偏机械；语义类要求（有用、准确）不在覆盖范围。
- 模型若输出 markdown 包裹，部分约束（如精确字数）可能被 verifier 按严格口径判错。

## 引用

```bibtex
@misc{zhou2023ifeval,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and others},
  year={2023}, eprint={2311.07911}, archivePrefix={arXiv}
}
```
