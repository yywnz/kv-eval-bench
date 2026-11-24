# 结果

`result.py` 会把每次运行的汇总写到 `results/{model}.json`，表格写到仓库根的 `result.txt`。
本目录只保留一个手工维护的榜单：

`leaderboard.csv` 列含义：

| 列 | 说明 |
|---|---|
| `model` / `size` / `type` | 模型名、参数量、dense / moe / vlm |
| `ifeval` | prompt-level strict accuracy（%） |
| `mmlu_pro` | accuracy（%），5-shot CoT |
| `longbench_v2` | accuracy（%），0-shot |
| `config` | 关键配置摘要：量化、并行、缓存、maxlen |
| `date` | 运行日期 |

**填表前必须满足：**

1. 三个任务用同一 `maxlen` 与 `max_tokens`，或在这一列里写清楚差异；
2. `longbench_v2` 的 `over_ctx` 数量一并记在 `config` 里（超窗题不进分母，分母不同的分数不可比）；
3. 每项的 `official_parse_rate` 高于 95%，否则说明模型没按格式输出，分数不可比；
4. `splits/manifest.json` 的 `version` 相同——**v2.1 重建过 `longbench_v2` 划分
   （`d6748c42…` → `5d4df6a4…`，换掉 88 题）**，跨划分的数字不能并列。

不满足这四条的数字仍可留作内部对比，但不要进榜单。
