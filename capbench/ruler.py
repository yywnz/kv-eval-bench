"""RULER 式长上下文检索（自建 needle-in-a-haystack）。

为什么自建而不直接跑官方 RULER：官方那套要下载 books3 等语料、依赖一串包，
并且有 13 个子任务；我们只需要「**长度 × 位置**」两维可控的检索题，
用来观察 KV 压缩后的衰减，不需要官方那套覆盖面。

干草堆取自 `data/longbench_v2.jsonl` 里的真实长文档（本仓库已有），
所以题目长度可控、内容自然、**不引入新的下载依赖**。

为什么这一项 LongBench v2 替代不了：LongBench 是固定 200 题、每题的上下文
长度是死的。想看「32k 还行、128k 崩了」这种曲线，必须自己按长度造题。
KV 量化/驱逐的典型故障正是这种形态——总分只掉 1pp，但长端已经塌了。

题目矩阵（默认）：

    长度 8k / 32k / 64k / 128k（token，按 3 字符 ≈ 1 token 折算）
    × 位置 10% / 50% / 90%（针在文档中的深度）
    × 变体 single（1 针）/ multikey（4 针中问 1 个）
    × 重复 1 次（换针、换干草堆偏移）

合计 24 题，约 1.4M 输入 token。每个格子 1 题，**只能看长度聚合（每档
6 题）的趋势，不要解读单格 0%/100%**——要出像样的曲线，把
`--ruler-repeat` 调到 4（96 题 / 5.6M token），或至少 2。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

LENGTHS_TOK = (8192, 32768, 65536, 131072)
DEPTHS = (0.1, 0.5, 0.9)
VARIANTS = ("single", "multikey")
CHARS_PER_TOKEN = 3.0

_WORDS = """
amber anchor autumn basalt beacon birch bramble canyon cedar cinder cobalt copper coral
crimson crystal cypress delta ember falcon feather flint forest garnet glacier granite
harbor hazel heron indigo ivory jasper juniper kestrel lagoon lantern larch lichen linen
lunar maple marble meadow mercury mistral moss myrtle nickel oak obsidian onyx opal
orchid osprey pebble pepper pine plume quartz quill raven reed resin ripple river
saffron sage sable sandstone shell silver slate sorrel spruce stellar summit thistle
thyme timber topaz tundra umber velvet violet walnut willow wren yarrow zephyr zinc
""".split()


def _bucket(tokens: int) -> str:
    if tokens < 32768:
        return "short"
    if tokens < 131072:
        return "medium"
    return "long"


def _qid(text: str) -> str:
    return "ruler-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def _load_filler(data_dir: Path) -> list[str]:
    """读长文任务已构建好的文档当干草堆。"""
    p = Path(data_dir) / "longbench_v2.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"缺少 {p}。RULER 的干草堆复用 LongBench v2 的文档，"
            "先跑 `python -m capbench build --tasks longbench_v2`")
    docs = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ctx = json.loads(line).get("context") or ""
                if len(ctx) > 20000:
                    docs.append(ctx)
    if not docs:
        raise ValueError("longbench_v2.jsonl 里没有足够长的文档可当干草堆")
    docs.sort(key=len, reverse=True)
    return docs


def _haystack(docs: list[str], need: int, rng: random.Random) -> str:
    """从随机位置的文档开始拼接，直到够长，再在空白处截断。"""
    out: list[str] = []
    total = 0
    i = rng.randrange(len(docs))
    while total < need + 2000:
        d = docs[i % len(docs)]
        out.append(d)
        total += len(d) + 2
        i += 1
    blob = "\n\n".join(out)
    cut = need
    if cut < len(blob):
        sp = blob.find(" ", cut)
        cut = sp + 1 if sp > 0 else cut
    return blob[:cut]


def _insert(text: str, depth: float, sentence: str) -> str:
    """在 depth 附近找一个空格插入，保证位置误差足够小。

    干草堆里有代码块/无空格长串，向后找空格可能漂出几百字符，
    所以只在 ±200 字符窗口内找最近的空格，找不到就直接插在 depth 处。
    """
    pos = int(len(text) * depth)
    win = 200
    left = text.rfind(" ", max(0, pos - win), pos)
    right = text.find(" ", pos, min(len(text), pos + win))
    if left > 0 and (right < 0 or pos - left <= right - pos):
        pos = left + 1
    elif right > 0:
        pos = right + 1
    return text[:pos] + sentence + "\n" + text[pos:]


def build_ruler(seed: int, data_dir: Path | None = None, repeat: int = 1,
                lengths: tuple[int, ...] = LENGTHS_TOK,
                depths: tuple[float, ...] = DEPTHS,
                **_):
    if data_dir is None:
        raise ValueError("build_ruler 需要 data_dir（干草堆来源）")
    docs = _load_filler(Path(data_dir))
    rng = random.Random(f"{seed}:ruler")
    rows: list[dict] = []

    for variant in VARIANTS:
        for L in lengths:
            need = int(L * CHARS_PER_TOKEN)
            for d in depths:
                for rep in range(repeat):
                    keys = [" ".join(rng.choice(_WORDS) for _ in range(4))
                            for _ in range(4)]
                    vals = [str(rng.randint(1_000_000, 9_999_999))
                            for _ in range(4)]
                    hay = _haystack(docs, need, rng)

                    if variant == "single":
                        target = 0
                        hay = _insert(hay, d, f"One of the special magic numbers "
                                             f"for {keys[0]} is: {vals[0]}.")
                        n_needles = 1
                    else:
                        # 目标针放在 d，另外 3 个干扰针散在别处，避免与 d 太近
                        target = rng.randrange(4)
                        others = iter([x for x in (0.1, 0.3, 0.5, 0.7, 0.9)
                                       if abs(x - d) > 0.08][:3])
                        spots = {i: (d if i == target else next(others))
                                 for i in range(4)}
                        # 逐个插入时按下标倒序，避免前面的插入挪动后面的位置
                        for idx in sorted(spots, reverse=True):
                            hay = _insert(hay, spots[idx],
                                          f"One of the special magic numbers "
                                          f"for {keys[idx]} is: {vals[idx]}.")
                        n_needles = 4

                    q = (f"What is the special magic number for {keys[target]} "
                         f"mentioned in the provided text?")
                    rows.append({
                        "id": _qid(f"{variant}|{L}|{d}|{rep}|{keys[target]}|{vals[target]}"),
                        "task": "ruler", "kind": "numeric",
                        "context": hay,
                        "question": q,
                        "answer": vals[target],
                        "meta": {"variant": variant, "target_len": L,
                                 "depth": round(d, 2), "pos_pct": int(d * 100),
                                 "needles": n_needles, "ctx_chars": len(hay),
                                 "length": _bucket(L),
                                 "filler_source": "longbench_v2",
                                 "approx_tokens": int(len(hay) / CHARS_PER_TOKEN)},
                    })

    rows.sort(key=lambda x: x["id"])
    meta = {"policy": (f"长度 {len(lengths)} 档 × 位置 {len(depths)} 档 × "
                       f"{len(VARIANTS)} 变体 × 重复 {repeat}"),
            "lengths_tok": list(lengths), "depths": list(depths),
            "variants": list(VARIANTS), "repeat": repeat,
            "filler_source": "longbench_v2 的真实长文档",
            "chars_per_token": CHARS_PER_TOKEN,
            "note": ("每格 1 题（repeat=1），只看长度聚合趋势；"
                     "单格 0%/100% 不具统计意义")}
    return rows, meta