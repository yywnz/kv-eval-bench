# results

评测结果与榜单。

```
results/
├─ leaderboard.csv           汇总表（进版本库）
└─ {model}/{revision}/{task}/
   └─ {时间戳}/
      ├─ raw.jsonl           逐题记录
      └─ summary.json        该任务汇总
```

## 字段

`summary.json` 主要字段：

| 字段 | 含义 |
|---|---|
| `n` / `n_error` | 总题数 / 请求失败题数 |
| `n_scored` | 参与判分的题数（`ifeval`、`livecodebench` 本地为 0） |
| `accuracy` | 正确率（%），`n_scored=0` 时为 null |
| `se_pp` | 标准误（百分点） |
| `margin_mean` | 正确项领先幅度；思考模型下为 null |
| `n_truncated` | 被 max_tokens 截断的题数，**不为 0 时分数不可用** |
| `n_unparsed` | 没能解析出答案的题数 |
| `cache_hit_rate` | 服务端报的缓存命中比例 |

`raw.jsonl` 关键字段：`item_id` / `correct` / `scored` / `chosen` / `parse` / `answer` / `margin` / `output_text` / `reasoning_text` / `finish_reason` / `usage` / `latency_ms`。

**排查异常先看 `finish_reason` 与 `parse`**：前者出现 `length` 说明被截断，后者大量出现 `fallback`/`none` 说明输出格式异常。

## 填榜单

```bash
python - <<'PY'
import json, pathlib, csv
rows = []
for f in pathlib.Path("results").glob("*/*/*/*/summary.json"):
    s = json.loads(f.read_text(encoding="utf-8"))
    rows.append(s)
print(f"{len(rows)} 条结果")
for r in rows:
    print(f"{r['model']:<20} {r['task']:<16} {r['accuracy']}")
PY
```

把数值填进 `leaderboard.csv`，并在 `config` 列写清量化方式、并行度、thinking 开关、框架版本。
