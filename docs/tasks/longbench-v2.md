# LongBench v2

**维度**：长上下文 · **题数**：200（从 503 题中分层抽取）· **判分**：本地

## 来源

| 项 | 值 |
|---|---|
| 数据集 | [`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2)（官方 [`THUDM/LongBench-v2`](https://huggingface.co/datasets/THUDM/LongBench-v2)） |
| 代码 | [`THUDM/LongBench`](https://github.com/THUDM/LongBench) 仓库根目录（v1 已移到 `LongBench/` 子目录） |
| 许可 | MIT |
| 论文 | Bai et al., 2024, [arXiv:2412.15204](https://arxiv.org/abs/2412.15204) |

## 官方评测口径

官方 `pred.py` + `result.py`：

1. **0-shot**，单条 user 消息，模板原文见 `prompts/longctx_0shot.txt`
   （`<text>` 包住文档，`Choices:` 下每行一个 `(A) …`）。
2. 结尾固定一句：`Format your response as follows: "The correct answer is (insert answer here)".`
3. 解码：`temperature=0.1`、`max_new_tokens=128`。
4. 抽取（`pred.py::extract_answer`）：先匹配 `The correct answer is \(([A-D])\)`，
   再退回 `The correct answer is ([A-D])`；抽不到记 `None`。
5. 判分：`pred == answer`；报表为 Overall + Easy/Hard + Short/Medium/Long 六列。
   `result.py` 里有个 `compensated` 开关会把 `None` 记 0.25，**默认关闭**，本仓库同样不用。

CoT 变体（`--prompt-variant cot`）对应官方 `0shot_cot.txt`，官方此时会再发一次请求
（`0shot_cot_ans.txt`）把推理结果转成答案；本仓库只发一次，直接对输出做抽取。

## 本仓库的抽样

- 按 `domain × length` 分层，每格蓄水池随机 14 题，总量上限 200（固定种子 2026）。
- `length` 三档：short / medium / long；`difficulty` 两档：easy / hard。
- 抽样用单遍流式 + 蓄水池，不把 465 MB 的原始文件全量载入内存。

## 数据格式

```json
{"id": "lbv2-0f1e2d3c4b", "task": "longbench_v2", "kind": "mc",
 "context": "...", "question": "...", "options": ["...", "...", "...", "..."],
 "answer": "C",
 "meta": {"domain": "Single-Document QA", "sub_domain": "...", "length": "medium",
          "difficulty": "hard", "ctx_chars": 412873}}
```

`context` 是独立字段，不是拼好的 prompt。`meta.ctx_chars` 用于超窗预检与长度过滤。

## 与官方的偏差

| 项 | 官方 | 本仓库 |
|---|---|---|
| 题量 | 503 | 200（分层子集） |
| 超窗处理 | **头尾各半截断**到 `maxlen` | **预检跳过**，记 `over_ctx`，不进分母 |
| 输出上限 | 128 | 128；思考模型需用 `--max-tokens` 调大 |
| CoT | 二次请求转答案 | 单次请求直接抽取 |
| 报表 | Overall + 难度 + 长度 | 相同分组，另加 `over_ctx` 与 `official%` 两列 |

**超窗这一条是实质性差异。** 官方截断会让超窗题「看起来答完了」，分数里看不出丢了什么；
本仓库跳过并单独统计，代价是分母变小、与其他实现不可直接比。因此报告分数时
**必须同时给出 `over_ctx` 数量和 `maxlen`**，否则无法判断分母是哪些题。

## 判分细节

- 抽取只认官方那句 `The correct answer is (X)`。模型若只写 `(C)` 或 `C`，
  会退回通用规则并把 `official` 记为 false——`official_parse_rate` 低于 95% 时，
  分数与公开榜单不可比。
- 四选一随机基线 25%。200 题的标准误约 3.1pp，**差 2–3pp 以内不要当结论**。
