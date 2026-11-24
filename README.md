# CapBench

**四任务能力评测集，判分对齐各数据集官方实现**

固定划分、固定模板、固定判分。回答的问题是：*这套推理配置在通用能力上处于什么水平，改动前后差在哪一项。*

| 任务 | 维度 | 题数 | 官方指标 | 判分 |
|---|---|---:|---|---|
| `ifeval` | 指令遵循 | 541 | prompt-level / instruction-level strict accuracy | 外部（官方校验器） |
| `mmlu_pro` | 知识 | 500 | accuracy（5-shot CoT） | 本地 |
| `gpqa_diamond` | 科学推理 | 198 | accuracy（`\boxed{}` 比对） | 本地 |
| `longbench_v2` | 长上下文 | 200 | accuracy（0-shot） | 本地 |

合计 1,439 题。

---

## 关于「官方标准」

四项的官方支持程度不一样，直接说清楚：

| 任务 | 有官方判分代码吗 | 官方口径 | 本仓库怎么对齐 |
|---|---|---|---|
| IFEval | **有**（Python 校验器，24 类约束） | prompt-level / instruction-level，strict / loose | 不重写，`python -m capbench ifeval` 调 `lm_eval` 的官方实现并自动合并指标 |
| MMLU-Pro | **有**（`evaluate_from_apiX.py`） | 5-shot CoT，抽取 `The answer is (X)` | 逐条复刻：模板、示例、解码参数、抽取正则 |
| LongBench v2 | **有**（`pred.py` + `result.py`） | 0-shot，抽取 `The correct answer is (X)` | 逐条复刻：模板原文、t=0.1、抽取正则 |
| GPQA Diamond | **没有** | 论文：四选一、逐步推理、答案放 `\boxed{}`、字符串比对 | 按论文实现；无官方脚本可对齐，属近似 |

**GPQA 是唯一没有官方判分器的。** 官方只发布数据（gated）与论文设置，没有代码。
本仓库按论文口径实现并明确标注为近似——它的绝对分不与公开榜单可比，
但同一模型不同配置之间可比。

每项的细节、与官方的逐项偏差见 [`docs/tasks/`](docs/tasks/)。

---

## 快速开始

### 1. 取代码与数据

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 国内网络
pip install datasets

python -m capbench list                      # 看四项题数
python -m capbench build --tasks all         # 下载并抽样（约 500 MB）
```

`build` 会写出 `data/*.jsonl`（题）与 `splits/*.json`（固定划分 + sha256），
后者是「同一批题」的凭据，**进版本库**；`data/` 不进。

### 2. 注册模型

`config/models.json`，字段全部可选，不写就沿用任务官方值：

```json
"my-model": {
  "base_url": "http://127.0.0.1:30000",
  "maxlen": 131072,
  "max_tokens": 16384,
  "extra_body": {"chat_template_kwargs": {"thinking": true}}
}
```

也可以不注册，用 `--base-url` 临时跑。

### 3. 生成预测 → 出分

```bash
python -m capbench check  --model my-model          # 数据/模板/端点自检
python pred.py   --model my-model --tasks all       # → pred/my-model.jsonl
python -m capbench ifeval --model my-model          # IFEval 走官方校验器 → results/ifeval-my-model@default.json
python result.py --model my-model --detail          # → result.txt + results/my-model.json
```

两段式对应 LongBench 的 `pred.py` / `result.py`：换统计口径只重跑 `result.py`，不重跑模型。

`pred.py` 只跑三项本地判分的任务（`ifeval` 标为 `judge: external`，会跳过）；
IFEval 单列一条命令，因为它要外部校验器（`pip install lm-eval`，会连带 torch）。
`result.py` 会把两边合成同一张表。

---

## 结果

```
task              n  scored     acc     se  over  err unparsed  官方%   maxlen  maxtok
---------------------------------------------------------------------------------------
mmlu_pro        500     500   62.40   2.17     0    0        0   97.6   262144   16384
gpqa_diamond    198     186   38.89   3.58    12    0        3   94.1   262144   16384
longbench_v2    200     158   64.56   3.81    42    0        1   99.4   262144   16384
```

（示例数值取自 2026-11 的 16 卡 PD 分离实跑；`longbench_v2` 那一行是 v2.1 之前的旧划分，
新划分要重跑。列结构才是这一段要说明的东西。）

| 列 | 含义 |
|---|---|
| `scored` | **有效评分题数**，也就是准确率的分母。`n − scored = over + err` |
| `over` | 超窗跳过（不进分母）。报分数必须同时给这一列和 `maxlen`，否则分母不可知 |
| `maxlen` | 该次运行使用的上下文上限，超窗判定和它直接绑定 |
| `官方%` | 命中官方抽取规则的比例。低于 95% 说明模型没按格式输出，分数与公开榜单不可比 |
| `unparsed` | 完全抽不出答案的题数（按错计） |
| `maxtok` | 实际使用的输出上限。低于官方值时须在报告里注明 |

超窗比例超过 20% 时 `result.py` 会单独打一行告警，提示该分只代表跑通的那一段。

### 跑基线 vs 改后

```bash
python pred.py   --model glm-flash --revision base --tasks all
python pred.py   --model glm-flash --revision fp8  --tasks all   # 同一文件，多版本并存
python -m capbench compare --model glm-flash --baseline-revision base --revision fp8
```

`compare` 按 `item_id` 逐题配对，输出每任务 Δpp、翻转数（对→错 / 错→对）与 McNemar 精确检验 p。
基线默认从**同一个 pred 文件的另一个 revision** 取；跨模型比较用 `--baseline-model`
或 `--baseline-pred`。`report` 也按 revision 过滤，不会把两次运行混成一个分数。

榜单模板在 [`results/leaderboard.csv`](results/leaderboard.csv)。

---

## 评测协议（跨实现比对前必须对齐）

1. **few-shot**：MMLU-Pro 默认 5-shot（官方）；其余 0-shot。`--prompt-variant 0shot` 会改变这一项。
2. **max_tokens**：思考模型必须调大。LongBench 官方只给 128 token，思考模型会在推理阶段被切断。
3. **thinking 开关**：同一个权重开/关是两套分数，必须显式声明。
4. **chat template**：是否加 system prompt 会影响 IFEval（它有「全大写」「不许逗号」这类约束）。
5. **超窗策略**：本仓库跳过并统计；官方 LongBench 是头尾截断。两者分母不同。
6. **判分归属**：`ifeval` 用官方校验器（经 `lm-eval` 子进程，版本与 yaml 哈希记在结果里），其余用本仓库实现。
7. **revision**：`--revision base` / `--revision fp8` 写进同一个 `pred/{model}.jsonl`，
   `report` 与 `compare` 都按 revision 取数；不声明 revision 就报「默认」那一份。
8. **划分版本**：`splits/*.json` 里的 `sha256_16` 是「同一批题」的凭据。
   v2.1 重建过 `longbench_v2` 划分（`d6748c42…` → `5d4df6a4…`），
   **跨划分的分数不可放进同一张表**。

`results/{model}.json` 会记录 `endpoint / maxlen / 每任务的 decode / 模板名`。

---

## 目录结构

```
capbench/
├─ pred.py                   生成预测 → pred/{model}.jsonl
├─ result.py                 聚合出分 → result.txt + results/{model}.json
├─ prompts/                  模板（官方原文，换模板不用改代码）
│   ├─ mmlu_pro_{header,example,question}.txt   官方 5-shot 三段式
│   ├─ longctx_{0shot,cot,no_context}.txt       官方 0shot / CoT / 无上下文对照
│   ├─ gpqa_0shot.txt                           论文口径（boxed）
│   └─ instruct_0shot.txt                       IFEval 原样送入
├─ config/models.json        模型注册表
├─ capbench/
│   ├─ tasks.py              四任务的数据源、抽样、官方解码参数
│   ├─ prompts.py            模板装配（选项排版按各自官方习惯）
│   ├─ scoring.py            官方抽取规则 + 归一比对
│   ├─ runner.py             生成循环与汇总
│   ├─ stats.py              标准误、配对检验
│   ├─ client.py             OpenAI 兼容客户端
│   ├─ ifeval.py             IFEval 封装：调 lm_eval、解析结果、并进报表
│   └─ __main__.py           CLI：list / build / check / eval / ifeval / report / compare
├─ splits/                   固定划分（进版本库）：id 列表 + sha256 + 5-shot 示例
├─ docs/tasks/               四张数据卡（官方口径、抽样、偏差）
├─ tests/test_scoring.py     判分回归测试
├─ tests/test_sampling.py    抽样回归测试（均匀性 / 确定性）
├─ tests/test_ifeval.py      IFEval 封装测试（命令拼装 / 结果解析）
├─ data/                     （不进版本库）构建产物
└─ results/                  聚合结果 + leaderboard.csv
```

---

## 数据来源与许可

| 任务 | 来源 | 许可 |
|---|---|---|
| IFEval | `google/IFEval` | Apache-2.0 |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | MIT |
| GPQA Diamond | `Idavidrein/gpqa`（gated）/ `hendrydong/GPQA_diamond`（镜像） | CC BY 4.0 |
| LongBench v2 | `zai-org/LongBench-v2` | MIT |

本仓库只做抽样、模板与判分实现，**数据版权归各上游**，使用前请遵守各自许可。
各任务的引用见 `CITATION.cff` 与对应数据卡。

---

## 设计取舍

**为什么抽样而不是全量。** 全量 MMLU-Pro 12,032 题在百 token/s 级服务上要跑数小时，
日常回归跑不动。500 题够看住 2–3pp 的差异，也不至于让人放弃跑。

**为什么不重写 IFEval 判分。** 24 类约束的边界条件（字数怎么数、标题算什么格式）
官方实现里有大量细节，重写只会引入 bug 且不可比。宁可多装一个 `lm_eval`。

**为什么 GPQA 用宽松归一而不是严格比对。** 198 题里 64% 的答案带单位或文字
（`6.24x10^-7 M`、`1 and 2`）。严格比对会把「答对但少写单位」判错，实测最多低估 49pp。
宽松的代价是极少数情况下会放过写法不同但实质不同的答案，这个代价比系统性低估小。

**为什么不静默截断超窗题。** LongBench 官方是头尾各半截断，保证跑得完；本仓库改为
预检跳过并单独统计 `over_ctx`。截断会让超窗题「看起来答完了」，分数里看不出丢了什么。

**为什么抽样宁可重来一遍。** v2.0 的分层蓄水池把 RNG 建在循环里，每个 item 都用同一
seed 新建，`randrange` 退化成 `floor(u·n)`，只覆盖每格前 `per_cell/u` 道题。按格子量：
旧划分选中的题在格子内的相对位置中位数 23%，53% 落在格子前四分之一；修复后中位数 55%，
四分段 27% / 20% / 24% / 30%。这属于「声称随机、实际取头部」——看分数看不出来，只能翻
代码看出来。v2.1 改为每格一个持久 RNG 并重建划分（换了 88 题），旧分数作废。
抽样逻辑现在有回归测试（`tests/test_sampling.py`）。

**为什么不设通过阈值。** 那是使用方的决策。本仓库只保证「同一批题、同一口径」。

---

## 不支持范围

| 缺口 | 说明 |
|---|---|
| 数学 / 代码 | 未收录。需要时接 AIME、LiveCodeBench（判分需外部执行环境） |
| 多模态 | 纯文本，VLM 只能测文本通路 |
| 结构化输出 / JSON | 无 JSON schema 专项，量化后格式崩塌测不到 |
| Agent 多轮工具调用 | 工具选择、参数正确性、状态维护未覆盖 |
| 中文母语评测 | 无 C-Eval / CMMLU 类专项 |
| 性能 | 吞吐 / 延迟 / 显存不在本仓库范围 |

---

## 版本

| 版本 | 变更 |
|---|---|
| 2.2.2 | IFEval 输出上限只信本轮记录（`--from-results` 不再拿当前配置的 `max_tokens` 顶替）；四项指标缺一项即拒绝合并 |
| 2.2.1 | 修 IFEval 封装三处隐患：版本改从被调用 CLI 的环境取（取不到记 `unknown`）；每轮独立运行目录、题数与指标同源；侧车文件名加 `@revision`，base/fp8 不再互相覆盖 |
| 2.2.0 | 新增 `ifeval` 子命令：封装 `lm_eval` 调用（含 `tokenizer_backend=none`、输出上限、结果合并），IFEval 指标自动进报表并记录校验器版本与 yaml 哈希 |
| 2.1.0 | 修复 LongBench 分层抽样的 RNG 重用（**划分重建，200 题换 88 题**）；`top_logprobs` 5→20；`report` 增 `maxlen` / `maxtok` 列并按 revision 过滤；新增 `compare`；抽样测试独立成 `tests/test_sampling.py` |
| 2.0.0 | 四项任务定型，判分对齐各数据集官方实现（去掉 AIME / LiveCodeBench） |

---

## 引用

```bibtex
@misc{capbench2026,
  title={CapBench: A Four-Task Capability Benchmark with Official Scoring Protocols},
  year={2026},
  note={\url{https://github.com/<org>/capbench}}
}
```

各任务的原始论文见对应数据卡。
