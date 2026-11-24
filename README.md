# CapBench

**面向推理侧优化的六维能力基准**

固定划分、固定 prompt、固定判分。回答的问题是：*这次量化 / 并行 / 缓存 / 调度改动，让模型变差了吗，差在哪一项。*

- **6 个任务 / 1,649 题**：指令遵循、知识、科学推理、数学、代码、长上下文
- **划分可复现**：`splits/` 存每题 id，重建后逐题比对即可确认是同一批题
- **两段式**：`pred.py` 只生成，`result.py` 只统计。改统计口径不用重跑模型
- **模板与配置外置**：`prompts/*.txt` 换 prompt 不用改代码，`config/models.json` 注册端点与窗口
- **跑评测零依赖**：只用 Python 标准库（`build` 除外）

---

## 任务一览

| 任务 | 维度 | 题数 | 指标 | 判分 | 来源 |
|---|---|---:|---|---|---|
| [`ifeval`](docs/tasks/ifeval.md) | 指令遵循 | 541 | prompt-level strict accuracy | 外部（官方 verifier） | google/IFEval |
| [`mmlu_pro`](docs/tasks/mmlu-pro.md) | 知识 | 500 | accuracy | 本地 | TIGER-Lab/MMLU-Pro |
| [`gpqa_diamond`](docs/tasks/gpqa-diamond.md) | 科学推理 | 198 | accuracy | 本地 | GPQA Diamond |
| [`aime`](docs/tasks/aime.md) | 数学 | 60 | accuracy（avg@k） | 本地 | AIME 2024 + 2025 |
| [`livecodebench`](docs/tasks/livecodebench.md) | 代码 | 150 | pass@1（执行） | 外部（沙盒） | LiveCodeBench v6 |
| [`longbench_v2`](docs/tasks/longbench-v2.md) | 长上下文 | 200 | accuracy | 本地 | LongBench v2 |

---

## 快速开始

### 1. 取代码与数据

```bash
git clone <repo-url> capbench && cd capbench
pip install -r requirements.txt          # 仅 build 需要；跑评测只要标准库
export HF_ENDPOINT=https://hf-mirror.com # 国内网络必需

python -m capbench build --tasks all     # 约 600 MB 下载，产出 data/ 与 splits/
```

`data/` 不进版本库，`splits/` 进版本库（几十 KB）——后者是复现「同一批题」的依据。
拿不到网络时用发布页的数据快照：`tar -xzf capbench-data-v1.0.tar.gz`。

### 2. 注册模型（一次性）

`config/models.json`：

```json
{
  "models": {
    "deepseek-v4.1": {
      "base_url": "http://127.0.0.1:30000",
      "maxlen": 131072,
      "max_tokens": 16384,
      "extra_body": {"chat_template_kwargs": {"thinking": true}}
    }
  }
}
```

`maxlen` 用于**超窗预检**：预估放不下就不发送，记为 `over_ctx`，不进准确率分母。
不做静默截断——截断会让「超窗」变成「看起来答完了」。临时跑一个未注册的模型不用改这个文件：

```bash
python pred.py --model whatever --base-url http://127.0.0.1:30001
```

### 3. 生成预测

```bash
python pred.py --model deepseek-v4.1                    # 全部任务
python pred.py --model deepseek-v4.1 --tasks mmlu_pro,aime
python pred.py --model deepseek-v4.1 --limit 10          # 冒烟
python pred.py --model deepseek-v4.1 --prompt-variant cot
python pred.py --model deepseek-v4.1 --tasks longbench_v2 --length short,medium
```

写到 `pred/{model}.jsonl`，一题一行，含模型原文、思考内容、抽取方式、耗时。
**重跑某个任务只会替换该任务的记录**，同文件里其他任务的结果保留。

### 4. 出分

```bash
python result.py --model deepseek-v4.1
python result.py --model deepseek-v4.1 --detail     # 按学科 / 长度档 / 年份拆开
```

```
task                  n  scored     acc     se  over  err  unparsed
mmlu_pro            500     500   48.60   2.24     0    0         0
gpqa_diamond        198     198   31.31   3.30     0    0         0
longbench_v2        200     188   62.23   3.55    12    0         0
```

同时写 `results/{model}.json`（含分组明细）和 `result.txt`（制表符表，可直接粘进表格）。

### 5. 改动前后对比

```bash
python -m capbench baseline --model deepseek-v4.1 --tasks all --revision base
python -m capbench run      --model deepseek-v4.1 --tasks all --revision fp8
python -m capbench compare  --model deepseek-v4.1 --tasks all \
  --revision fp8 --baseline-revision base
```

基线和本次可以用不同的 `--revision`，`compare` 用 `--baseline-revision` 指前者。

```
task                base     new     Δpp  bad  good       p  margin%
gpqa_diamond       30.00   28.50   -1.50    4     1  0.3750     +2.1
```

`bad` 是「基线对、本次错」，`good` 是反向翻转，`p` 为 McNemar 双侧检验。
**工具只出数字，不判通过与否。**

同一个机制也能测**缓存冷热一致性**（开了 prefix cache / HiCache 时建议每次改动都跑）：

```bash
python -m capbench baseline --model X --tasks longbench_v2 --revision cold
python -m capbench run      --model X --tasks longbench_v2 --revision warm   # 前缀命中
python -m capbench compare  --model X --tasks longbench_v2 \
  --revision warm --baseline-revision cold
```

两次跑的是同一批题、同一模板，第二次几乎全部前缀命中。**`bad` 与 `good` 应当为 0**；
不为 0 说明缓存路径改变了输出，属于实现缺陷而不是模型退化。

### 两项外部判分

```bash
lm_eval --model local-completions \
  --model_args base_url=http://127.0.0.1:30000/v1,model=deepseek-v4.1,tokenized_requests=False \
  --tasks ifeval --num_fewshot 0 --batch_size 32
```

`python -m capbench check` 会把这两条命令直接打印出来。

---

## 耗时

耗时几乎全由**输出 token** 决定（thinking 模型每题几千 token），输入侧可忽略——
六项输入合计约 4.3M token，prefill 只占几分钟。

| 任务 | 题数 | 单题输出 | 合计 | 占比 |
|---|---:|---:|---:|---:|
| gpqa_diamond | 198 | 3,500 | 0.69M | 34% |
| mmlu_pro | 500 | 1,200 | 0.60M | 29% |
| aime | 60 | 6,000 | 0.36M | 17% |
| ifeval | 541 | 350 | 0.19M | 9% |
| livecodebench | 150 | 1,200 | 0.18M | 9% |
| longbench_v2 | 78 | 600 | 0.05M | 2% |
| **合计** | **1,527** | | **2.07M** | |

时间 ≈ `2.07M / 聚合输出吞吐`：

| 聚合 tok/s | 六项全量 |
|---:|---:|
| 100 | 5.7 小时 |
| 300 | 1.9 小时 |
| 800 | 43 分钟 |
| 1500 | 23 分钟 |

实测参考：16 卡、开启设备前缀缓存、C32 并发（聚合 548 tok/s）下六项约 **63 分钟**；
同卡数无缓存基线（132 tok/s）约 **4.4 小时**。

**注意区分两个数**：单请求 decode 速度（TPOT 的倒数）和聚合吞吐。前者 100 tok/s、
并发 16 时后者约 1,000 tok/s，差一个数量级。判定方法：

```bash
time python pred.py --model X --tasks mmlu_pro --limit 10 --concurrency 1
time python pred.py --model X --tasks mmlu_pro --limit 10 --concurrency 16
```

第二次耗时约为第一次的 1/8，说明 100 是单请求速度、还能靠并发放大；
两次接近，说明已经饱和，100 就是聚合上限。

`aime` 只有 60 题却占 17%，因为它每题思考极长——想压时间优先砍它，而不是砍题数最多的 `ifeval`。

---

## Prompt 模板

模板在 `prompts/`，文件名是 `{shape}_{variant}.txt`：

| shape | 用在哪 | 占位符 |
|---|---|---|
| `instruct` | ifeval | `{question}`（原样送，不包壳） |
| `mc` | mmlu_pro | `{question}` `{options}` |
| `longctx` | longbench_v2 | `{context}` `{question}` `{options}` |
| `numeric` | aime | `{question}` |
| `open` | gpqa_diamond | `{question}` |
| `code` | livecodebench | `{question}` |

变体：`0shot`（默认）、`cot`（要求先推理再给最终行）、`no_context`（只给 longctx，用于对照
「模型到底读没读文档」）。找不到精确变体时回落到 `{shape}_0shot.txt`。

改模板不需要改代码，也不用重建数据——`data/` 里只存结构化字段（question / options / context）。

```bash
python pred.py --model deepseek-v4.1 --tasks longbench_v2 --prompt-variant no_context
```

`no_context` 的分数就是「不看文档的基线」；与正常分数的差值才是文档带来的真实增益。

---

## 结果与榜单

榜单用 `results/leaderboard.csv` 维护：

```csv
model,size,type,ifeval,mmlu_pro,gpqa_diamond,aime,livecodebench,longbench_v2,config,date
```

`config` 列写清量化方式、并行度、thinking 开关、prompt 变体、框架版本。
**不同 prompt 模板、不同 max_tokens、不同 thinking 开关的分数不能放在一张表里比**——
同一个权重开/关 thinking 能差 10pp 以上。

---

## 评测协议（跨实现比对前必须对齐）

1. **prompt 变体**：默认 `0shot`。用 `cot` 必须在报告里写明。
2. **max_tokens**：思考模型 16384 以上。被截断的题全判错，`n_truncated` 不一定显眼。
   `config/models.json` 里写了 `max_tokens` 即锁定；题目自带的 1024 类提示值不会覆盖它。
   实测未锁定时的截断率：mmlu 18%、gpqa 69%、aime 50%。
3. **thinking**：开或关都行，必须显式声明——同一个权重是两套分数。
4. **温度**：默认 0（greedy）。要采样必须写明。
5. **聚合方式**：AIME 这类小样本，`avg@1` 与 `avg@8` 差很远。
6. **判分归属**：`ifeval`/`livecodebench` 用官方 verifier，其余用本仓库判分器。

`baseline` 会把 `endpoint / max_tokens / maxlen / extra_body / prompt_variant` 写进
`baselines/{model}/{rev}/meta.json`。

### PD 分离部署

接口不变，但瓶颈会从 decode 转到 prefill，跑法要调三处：

- **并发要扫，不能拍。** `--concurrency` 太低 decode 跑不满，太高则 prefill 排队、
  KV 传输拥塞，吞吐不增而 TTFT 恶化。16 卡 1P1D 建议从 16/32/64 各跑 20 题取拐点。
- **长文单独跑。** `longbench_v2` 单题输入可达 50k+ token，混在短题里会抢占 P 节点，
  让其余任务的 TTFT 失真；反过来 decode 节点在处理长题时是闲的。分两次跑，
  两次的数字都更稳。
- **任务顺序固定。** 混跑时前一项的负载会影响后一项的排队，`pred.py` 按
  `ifeval → mmlu_pro → gpqa_diamond → aime → livecodebench → longbench_v2`
  固定顺序执行，不要用 `--tasks` 随便重排后跟历史结果比。

---

## 目录结构

```
capbench/
├─ pred.py                   生成预测 → pred/{model}.jsonl
├─ result.py                 聚合出分 → results/{model}.json + result.txt
├─ prompts/                  prompt 模板（换模板不用改代码）
├─ config/
│   └─ models.json           模型注册表：端点 / maxlen / max_tokens / extra_body
├─ capbench/                 评测实现（仅标准库）
│   ├─ tasks.py              六任务的数据来源、抽样策略、题型
│   ├─ prompts.py            模板装配
│   ├─ config.py             模型配置解析
│   ├─ scoring.py            判分（选项抽取 / 数值 / boxed 归一）
│   ├─ runner.py             生成循环与汇总
│   ├─ stats.py              配对统计（McNemar）
│   ├─ client.py             OpenAI 兼容客户端
│   └─ __main__.py           CLI：list / build / check / run / baseline / compare
├─ tests/test_scoring.py     判分回归测试（python tests/test_scoring.py）
├─ splits/                   固定划分（进版本库）：每题 id + sha256
├─ docs/tasks/               六张数据卡
├─ data/                     （不进版本库）构建产物
├─ pred/                     （不进版本库）逐题预测
└─ results/                  聚合结果 + leaderboard.csv
```

`pred.py` / `result.py` 对应 LongBench 的同名脚本：前者只生成，后者只统计。
`python -m capbench run|baseline|compare` 是额外的回归工作流，会带时间戳落盘并支持配对检验。

---

## 数据来源与许可

代码 MIT。数据保留上游许可：

| 任务 | 上游 | 许可 |
|---|---|---|
| ifeval | google/IFEval | Apache-2.0 |
| mmlu_pro | TIGER-Lab/MMLU-Pro | MIT |
| gpqa_diamond | hendrydong/GPQA_diamond（官方库需申请） | CC BY 4.0 |
| aime | Maxwell-Jia/AIME_2024 + MathArena/aime_2025 | CC BY 4.0 / MIT |
| livecodebench | livecodebench/code_generation_lite | MIT |
| longbench_v2 | zai-org/LongBench-v2 | MIT |

试题版权归原作者。商业用途请逐项核对上游许可。

---

## 设计取舍

**为什么是这六项。** 对应推理侧优化最常崩的六条路径：格式约束、知识表征、数值敏感的长链推导、
符号级精度、长序列注意力。少一项就有一类退化漏检。

**为什么保留已饱和的 GPQA / AIME。** 它们对 logit 扰动最敏感，作为报警器比作为排名工具更有价值。
绝对分没意义，但掉了 3pp 一定出了问题。

**为什么抽样改成先收集再随机。** 按流式顺序取前 N 条会在类别内引入系统性偏差（源文件常按难度
或年份排序），换一次环境就可能换一批题。现在固定种子 + 随机抽样 + id 落盘，重建可逐题比对。

**为什么不静默截断。** LongBench 的做法是头尾各半截断，保证跑得完；本仓库改为预检跳过并单独
统计 `over_ctx`。截断会让超窗题变成「看起来答完了」，而分数里看不出丢了什么。

**为什么不设通过阈值。** 那是使用方的决策。本仓库只保证「同一批题、同一口径」。

---

## 不支持的范围

| 缺口 | 说明 |
|---|---|
| 结构化输出 / JSON | 量化后格式崩塌测不到，建议自建 50 条 schema 题 |
| Agent 多轮工具调用 | 工具选择、参数正确性、状态维护均未覆盖 |
| 中文母语评测 | 无 C-Eval / CMMLU 类专项 |
| 多模态 | 纯文本，VLM 只能测文本通路 |
| 主观质量 | 冗长、语气、过度拒答测不到 |
| 性能 | 吞吐 / 延迟 / 显存不在本仓库范围 |

---

## 引用

```bibtex
@misc{capbench2026,
  title={CapBench: A Six-Dimension Capability Benchmark for Inference-Side Validation},
  year={2026},
  note={\url{https://github.com/<org>/capbench}}
}
```

各任务的原始论文见对应数据卡。
