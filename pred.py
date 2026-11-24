#!/usr/bin/env python
"""生成预测，写入 pred/{model}.jsonl。

对应 LongBench 的 pred.py：只负责「按模板拼 prompt → 请求端点 → 落盘」，
统计和出表交给 result.py，这样换统计口径不用重跑模型。

  python pred.py --model deepseek-v4.1                       # 全部任务
  python pred.py --model deepseek-v4.1 --tasks mmlu_pro,longbench_v2
  python pred.py --model my-model --base-url http://127.0.0.1:30000 --limit 20
  python pred.py --model deepseek-v4.1 --prompt-variant 0shot
  python pred.py --model deepseek-v4.1 --tasks longbench_v2 --length short,medium
"""

import sys

from capbench.__main__ import main

if __name__ == "__main__":
    sys.exit(main(["eval"] + sys.argv[1:]))
