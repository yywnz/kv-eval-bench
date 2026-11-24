"""模板装配。

每个任务对应 `prompts/` 下一组文件，内容取自官方实现，占位符保持官方命名习惯。
模板用 `.replace()` 替换而不是 `str.format()`——题目正文里常有花括号（LaTeX、
代码），`format` 会当成占位符直接报错。

选项排版按官方各自的习惯：
  MMLU-Pro    `A. 10, B. 40, ...`   单行、逗号分隔（官方 5-shot 示例的写法）
  LongBench   `(A) ...`              每行一个（官方 0shot.txt 的写法）
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "prompts"
LETTERS = "ABCDEFGHIJ"


def _read(name: str, directory: Path | None = None) -> str:
    p = Path(directory or DEFAULT_DIR) / name
    if not p.exists():
        raise FileNotFoundError(f"缺少模板 {p}")
    return p.read_text(encoding="utf-8")


def _sub(tpl: str, **kw) -> str:
    for k, v in kw.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def fmt_mmlu_options(options: Iterable[str]) -> str:
    """官方写法：`A. x, B. y` —— 单行、逗号分隔。"""
    return ", ".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))


def template_name(shape: str, variant: str) -> str:
    if shape == "mmlu":
        return "mmlu_pro_5shot" if variant not in ("0shot", "zs") else "mmlu_pro_0shot"
    if shape == "longctx":
        return f"longctx_{variant}.txt"
    return {
        "instruct": "instruct_0shot.txt",
        "passthrough": "passthrough.txt",
    }.get(shape, f"{shape}_{variant}.txt")


def render(row: dict, shape: str, variant: str = "0shot",
           prompt_dir: Path | None = None,
           exemplars: dict | None = None) -> tuple[str, str]:
    """返回 (prompt, 模板标识)。找不到指定变体时回落到 0shot。"""
    directory = Path(prompt_dir or DEFAULT_DIR)
    question = row.get("question") or ""
    options = list(row.get("options") or [])

    if shape == "instruct":
        name = "instruct_0shot.txt"
        return _sub(_read(name, directory), question=question).strip(), name

    if shape == "mmlu":
        n_shot = 0 if variant in ("0shot", "zs") else int(str(variant).replace("shot", ""))
        header = _read("mmlu_pro_header.txt", directory)
        q_tpl = _read("mmlu_pro_question.txt", directory)
        cat = str((row.get("meta") or {}).get("category") or "all")
        parts = [_sub(header, subject=cat)]
        if n_shot and exemplars:
            ex_tpl = _read("mmlu_pro_example.txt", directory)
            for ex in (exemplars.get(cat) or [])[:n_shot]:
                cot = (ex.get("cot") or "").strip()
                ans = f"({ex['answer']})"
                answer = cot if ans in cot else f"{cot}\nThe answer is {ans}."
                parts.append(_sub(ex_tpl, question=ex["question"],
                                  options=fmt_mmlu_options(ex["options"]),
                                  answer=answer.strip()))
        parts.append(_sub(q_tpl, question=question,
                          options=fmt_mmlu_options(options)))
        return "\n\n".join(parts).strip() + "\n", f"mmlu_pro_{n_shot}shot"

    if shape == "longctx":
        name = f"longctx_{variant}.txt"
        if not (directory / name).exists():
            name = "longctx_0shot.txt"
        tpl = _read(name, directory)
        return _sub(tpl,
                    context=row.get("context") or "",
                    question=question,
                    choice_A=options[0] if len(options) > 0 else "",
                    choice_B=options[1] if len(options) > 1 else "",
                    choice_C=options[2] if len(options) > 2 else "",
                    choice_D=options[3] if len(options) > 3 else "").strip(), name

    if shape == "passthrough":
        name = "passthrough.txt"
        return _sub(_read(name, directory), question=question).strip(), name

    raise ValueError(f"未知 shape: {shape}")
