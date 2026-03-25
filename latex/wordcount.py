from __future__ import annotations

import re
import sys
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
COMMAND_WITH_ARG_RE = re.compile(r"\\[A-Za-z@]+[*]?(?:\[[^\]]*\])?\{([^{}]*)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")
INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)|\$(.*?)\$", re.DOTALL)
DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]|\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", re.DOTALL)


def strip_comments(text: str) -> str:
    return "\n".join(COMMENT_RE.sub("", line) for line in text.splitlines())


def strip_tc_ignored_blocks(text: str) -> str:
    lines: list[str] = []
    ignoring = False
    for line in text.splitlines():
        if line.strip() == "%TC:ignore":
            ignoring = True
            continue
        if line.strip() == "%TC:endignore":
            ignoring = False
            continue
        if not ignoring:
            lines.append(line)
    return "\n".join(lines)


def simplify_tex(text: str) -> str:
    text = strip_comments(text)
    text = strip_tc_ignored_blocks(text)
    text = DISPLAY_MATH_RE.sub(" ", text)
    text = INLINE_MATH_RE.sub(" ", text)

    previous = None
    while previous != text:
        previous = text
        text = COMMAND_WITH_ARG_RE.sub(r" \1 ", text)

    text = re.sub(r"\\[A-Za-z@]+[*]?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}_^&~]", " ", text)
    return text


def count_words(tex_path: Path) -> int:
    text = tex_path.read_text(encoding="utf-8")
    plain_text = simplify_tex(text)
    return len(WORD_RE.findall(plain_text))


def main() -> int:
    if len(sys.argv) != 3:
        return 1

    tex_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    count = count_words(tex_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{count} ", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
