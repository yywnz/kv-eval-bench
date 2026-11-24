#!/usr/bin/env python
"""读 pred/ 出分，写 results/{model}.json 与 result.txt。

对应 LongBench 的 result.py：只做聚合，不碰端点。改了统计口径重跑这一条即可。

  python result.py --model deepseek-v4.1
  python result.py --model deepseek-v4.1 --detail    # 按学科/难度/长度分组
"""

import sys

from capbench.__main__ import main

if __name__ == "__main__":
    sys.exit(main(["report"] + sys.argv[1:]))
