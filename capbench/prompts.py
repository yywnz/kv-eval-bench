"""prompt 模板装配。

模板放在 `prompts/`，文件名是 `{shape}_{variant}.txt`：

    shape    instruct / mc / longctx / numeric / open / code
    variant  0shot / cot / no_context

占位符用最朴素的字符串替换（不用 str.format，避免模板里的花括号被当成转义）：

    {question}  {options}  {context}
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "prompts"
LETTERS = "ABCDEFGHIJ"

VARIANTS = ("0shot", "cot", "no_context")


def template_path(shape: str, variant: str, prompt_dir: Path | None = None) -> Path:
    """按「精确变体 → 0shot → passthrough」顺序找模板。"""
    d = Path(prompt_dir or DEFAULT_DIR)
    for name in (f"{shape}_{variant}.txt", f"{shape}_0shot.txt", "passthrough.txt"):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{d} 下找不到 {shape} 的模板（试过 {shape}_{variant}.txt / {shape}_0shot.txt）")


def options_block(options: list[str] | None) -> str:
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options or []))


def render(row: dict, shape: str, variant: str = "0shot",
           prompt_dir: Path | None = None) -> tuple[str, str]:
    """返回 (拼好的 prompt, 用的模板文件名)。"""
    p = template_path(shape, variant, prompt_dir)
    tpl = p.read_text(encoding="utf-8")
    text = (tpl.replace("{question}", row.get("question") or "")
               .replace("{options}", options_block(row.get("options")))
               .replace("{context}", row.get("context") or ""))
    return text.strip(), p.name
