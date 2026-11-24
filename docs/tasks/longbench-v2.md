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

- 按 `domain × length` 分层（17 个格子），每格用**标准蓄水池**均匀随机抽 14 题，
  不足 14 题整格取走；总量上限 200（固定种子 2026）。
- 每个格子**只建一个 RNG 并持续取数**，因此格内每道题的入选概率相同；同一
  `(seed, domain, length)` 在任何机器上抽出的都是同一批题。
- `length` 三档：short / medium / long；`difficulty` 两档：easy / hard。
- 单遍流式 + 蓄水池，不把原始文件全量载入内存。

> **v2.1 重建过这套划分（`sha256_16` 从 `d6748c42…` 变为 `5d4df6a4…`，200 题里换了 88 题）。**
> v2.0 把 RNG 建在了循环里——每个 item 都 `random.Random(同一个 seed)`，
> 第一次 `randrange` 退化成 `floor(u·n)`，只覆盖每格前 `per_cell/u` 道题。
> 按格子量：旧划分选中的题在**格子内**的相对位置中位数只有 23%，53% 落在格子前四分之一、
> 仅 9% 落在后四分之一；修复后中位数 55%，四个分段为 27% / 20% / 24% / 30%，才算格内均匀。
> **v2.0 的分数不能和 v2.1 放进同一张表。**

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

**三条差异合起来，决定了这个分数只在仓库内部可比。** 题量是 200 题的分层子集、
超窗是跳过而非截断、CoT 变体少了官方的第二次转换请求——任何一条都足以让数字
偏离官方口径。所以 `longbench_v2` 的分数只用于**同仓库、同划分下不同配置之间**的比较，
不要和官方榜单或其他实现的结果并列在一张表里。

## 判分细节

- 抽取只认官方那句 `The correct answer is (X)`。模型若只写 `(C)` 或 `C`，
  会退回通用规则并把 `official` 记为 false——`official_parse_rate` 低于 95% 时，
  分数与公开榜单不可比。
- 四选一随机基线 25%。200 题的标准误约 3.1pp，**差 2–3pp 以内不要当结论**。
