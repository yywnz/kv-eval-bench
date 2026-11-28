# RULER 式 NIAH（自建）

**维度**：长上下文检索 · **题数**：24（可调）· **判分**：本地

## 来源

| 项 | 值 |
|---|---|
| 方法 | RULER（NVIDIA），`niah_single` / `niah_multikey` 两个子任务的简化版 |
| 参考 | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) · [arXiv:2404.06654](https://arxiv.org/abs/2404.06654) |
| 干草堆 | 本仓库 `data/longbench_v2.jsonl` 里的真实长文档 |
| 许可 | 生成逻辑 MIT；干草堆随 LongBench v2（MIT） |

## 为什么自建

官方 RULER 要下载 books3 等语料、依赖一串包，且含 13 个子任务。我们只需要
**长度 × 位置**两维可控的检索题，用来观察 KV 压缩后的衰减。

**这一项 LongBench v2 替代不了**：LongBench 是固定 200 题，每题的上下文长度是死的。
想看出「32k 还行、128k 崩了」，必须自己按长度造题。KV 量化/驱逐的典型故障正是
这种形态——总分只掉 1pp，但长端已经塌了。

## 题目矩阵

| 维度 | 取值 |
|---|---|
| 长度（token，按 3 字符 ≈ 1 token） | 8k / 32k / 64k / 128k |
| 位置（针在文档中的深度） | 10% / 50% / 90% |
| 变体 | `single`（1 针）· `multikey`（4 针中问 1 个，考干扰项辨别） |
| 重复 | 默认 1（`--ruler-repeat`，出曲线建议 4） |

默认 4 × 3 × 2 × 1 = **24 题，约 1.4M 输入 token**。

## 数据格式

```json
{"id": "ruler-3f2a1b0c9d", "task": "ruler", "kind": "numeric",
 "context": "...长文档...",
 "question": "What is the special magic number for ivory feather zinc flint mentioned in the provided text?",
 "answer": "1279907",
 "meta": {"variant": "single", "target_len": 8192, "depth": 0.9, "pos_pct": 90,
          "needles": 1, "ctx_chars": 24666, "length": "short",
          "filler_source": "longbench_v2", "approx_tokens": 8222}}
```

针的句式固定为 `One of the special magic numbers for {key} is: {value}.`，
`key` 是 4 个随机词，`value` 是 7 位数字。`multikey` 的另外 3 根针是关键干扰项——
模型必须定位到**问的那一个 key**，而不是随便抄一个数字。

## 判分

`kind="numeric"`：取正文里**最后一个**数字串，去掉千分位后与标准答案比对。

- 取最后一个：结论在末尾；思考模型会把中间量也写出来，取第一个几乎必错。
- `1,234,567` 与 `1234567` 视为相同。

## 读数

报告里会打印：

```
[ruler] 长度 × 位置 准确率（%）—— 看趋势，不要解读单格
  variant      len     10%     50%     90%      合计
  multikey      8k     100     100     100       100
  multikey     32k     100      50     100        83
  ...
  按长度聚合：8k 100%   32k 83%   64k 67%   128k 33%
```

**只看按长度聚合那一行**——每格只有 1 题（repeat=1），单格的 0%/100% 没有统计意义。
判断 KV 改动的依据是：**长端掉幅是否显著大于基线同一位置**，而不是绝对分。

## 已知限制

- **每格 n 极小**。默认配置下每个（变体 × 长度 × 位置）格子只有 1 题。
  要看形状够用，要下结论请 `--ruler-repeat 4`（96 题 / 5.6M token）。
- **干草堆依赖 LongBench v2 已构建**。`build --tasks ruler` 前先有
  `data/longbench_v2.jsonl`，否则报错并提示。
- **针的位置误差 ≤ 0.3%**（`tests/test_ruler.py` 锁住 ≤2%）。插入点避开长串无空格文本，
  否则可能漂出几百字符。
- **不是官方 RULER 分数**。子任务、语料、答案抽取都与官方不同，
  **不可与 RULER 榜单横比**；它只用于同仓库内部的纵向对比。