"""文本规范化（对齐 HowToCook 书写习惯）。"""

from __future__ import annotations

import re

_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")


def strip_html_comments(text: str) -> str:
    return _HTML_COMMENT_RE.sub("", text)


def normalize_text(text: str) -> str:
    """去除首尾空白，合并连续空行，统一换行符。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
