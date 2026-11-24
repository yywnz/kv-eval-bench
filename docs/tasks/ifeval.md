# IFEval

**维度**：指令遵循 · **题数**：541（全量）· **判分**：官方校验器（外部执行，本仓库封装调用）

## 来源

| 项 | 值 |
|---|---|
| 数据集 | [`google/IFEval`](https://huggingface.co/datasets/google/IFEval) |
| 官方实现 | [`google-research/instruction_following_eval`](https://github.com/google-research/google-research/tree/master/instruction_following_eval) |
| 许可 | Apache-2.0 |
| 论文 | Zhou et al., 2023, [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) |

## 官方评测口径

IFEval 是三项里唯一有**官方可执行校验器**的——判分不是比对字符串，而是按
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

一条命令，内部起 `lm_eval` 子进程并把结果并进同一张表：

```bash
python -m capbench ifeval --model glm-flash          # → results/ifeval-glm-flash@default.json
python result.py --model glm-flash                   # ifeval 行自动出现
```

```bash
python -m capbench ifeval --model glm-flash --dry-run            # 只看拼出的命令
python -m capbench ifeval --model glm-flash --max-gen-toks 16384 # 思考模型必须调大
python -m capbench ifeval --model glm-flash --limit 20           # 冒烟
python -m capbench ifeval --model glm-flash --from-results DIR   # 复用已有输出，不重跑
```

需要先装校验器本体（仓库不 vendor 它）：

```bash
pip install lm-eval        # 连带 torch，数 GB；只用 API 模式也一样
```

### 两个坑，封装里已经处理

1. **`tokenizer_backend=none` 必须给。** 不给的话 `lm_eval` 会拿模型名去 HF 找
   tokenizer，端点上的名字（`glm-flash` 这种）不在 HF 上，直接
   `OSError: Can't load the configuration of ...`。手敲命令时最容易漏这一条。
2. **输出上限。** 官方 yaml 是 `max_gen_toks: 1280`，思考模型的推理段会把它耗尽。
   配置里写了 `max_tokens` 就用配置的值，也可以用 `--max-gen-toks` 显式给。

### 口径凭据

侧车文件名是 `results/ifeval-{model}@{revision}.json`——**revision 进文件名**，
base 与 fp8 各存一份，不会互相覆盖；`result.py` 按当前 revision 取对应那份。

里面记了这些：

| 字段 | 作用 |
|---|---|
| `lm_eval.version` | 校验器版本；它更新会改口径 |
| `lm_eval.task_version` | 任务 yaml 里的 `metadata.version`（现在是 4.0） |
| `lm_eval.yaml_sha256_16` | yaml 内容的哈希，比版本号更难糊弄 |
| `lm_eval.max_gen_toks` | 本次实际给的输出上限；读不到就是 `unknown` |
| `lm_eval.max_gen_toks_source` | 这个数字哪来的：`run`（本轮命令行）/ `yaml`（任务默认）/ `args`（手动指定）/ `unknown` |
| `run_dir` | 这一轮 lm_eval 的输出目录 |
| `n` | 题数，取自同一份 `results.json` 的 `n-samples` |

几个刻意的做法，都是为了防止「记录了对不上号」：

- **版本从被调用的那个 CLI 所在环境取**（用它自己的解释器问 `importlib.metadata`，
  退一步读它的 `dist-info`）。当前解释器里也装了 lm-eval 时不读它——两个环境版本
  可以不同。取不到就写 `unknown`，宁可空着也不填错的。
- **每轮一个独立运行目录** `results/lm_eval/{model}/{时间戳}-{revision}/`，
  题数和指标都从这一轮的产物里取，不跨轮拼。
- 早期不带 `@revision` 的旧侧车仍能读，但会校验里面的 `revision` 是否一致。
- **输出上限不从当前配置猜**。`--from-results` 复用旧结果时，只认旧结果里
  `config.gen_kwargs` 的记录（没有就退回任务 yaml 的静态值）；两者都没有就写
  `unknown` 并在终端告警——`config/models.json` 里现在的 `max_tokens=16384`
  跟上一轮实际给的 1280 没有关系。
- **四项指标必须齐全**（prompt/instruction × strict/loose），缺任何一项直接
  拒绝合并、报错退出。带着缺指标的报表比没有报表更容易误导。

这三项任一变化，ifeval 的分数就不该和以前并列。

**这里仍有一处没法自证**：`lm_eval` 自己从 `google/IFEval` 下载数据，我们
`data/ifeval.jsonl` 的 `splits/ifeval.json` 对它没有约束力。上游改题我们检测不到，
只能靠 `n=541` 和数据集版本间接判断。另外 `langdetect` 默认不固定随机种子
（官方建议 `DetectorFactory.seed = 0`），「用英语回答」「全大写英文」这两类检查
存在极小概率的抖动——这一点在外部实现里，本仓库改不了，只能记在这里。

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
| 题量 | 541 | 541（全量，但数据由 lm_eval 自行下载） |
| 判分 | 官方校验器 | 同一实现，经 `lm-eval` 子进程调用，结果自动并入报表 |
| prompt | 原文直接送入 | 相同（`prompts/instruct_0shot.txt` 只含 `{question}`） |
| 生成 | `temperature=0`、`max_gen_toks=1280` | 相同；思考模型用配置或 `--max-gen-toks` 调大后须注明 |

**唯一需要对齐的是 chat template**：IFEval 的约束里有「全大写」「不加逗号」这类
对表面形式敏感的要求，system prompt 或模板里多一句话都可能让模型改变行为。
报告时须写明是否加了 system prompt。
