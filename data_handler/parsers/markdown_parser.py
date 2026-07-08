"""Markdown 菜谱解析器（HowToCook 模板结构）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from data_handler.parsers.base import DocumentParser
from data_handler.schemas import (
    SECTION_APPENDIX,
    SECTION_BODY,
    SECTION_CALCULATION,
    SECTION_INGREDIENTS,
    SECTION_INTRO,
    SECTION_STEPS,
    ParsedDocument,
    TextBlock,
)
from data_handler.utils.text_normalize import normalize_text, strip_html_comments

_H1_RE = re.compile(r"^#\s+(.+)$")
_H2_RE = re.compile(r"^##\s+(.+)$")
_LIST_ITEM_RE = re.compile(r"^[-*+]\s+(.+)$")
_DIFFICULTY_RE = re.compile(r"^预估烹饪难度[：:]\s*(.+)$")
_RECIPE_TITLE_RE = re.compile(r"^(.+?)的做法\s*$")


def _section_from_heading(heading: str) -> str:
    heading = heading.strip()
    if "必备原料" in heading or heading.startswith("原料"):
        return SECTION_INGREDIENTS
    if heading == "计算" or heading.startswith("计算"):
        return SECTION_CALCULATION
    if heading == "操作" or heading.startswith("操作"):
        return SECTION_STEPS
    if "附加" in heading:
        return SECTION_APPENDIX
    return SECTION_BODY


def _recipe_name_from_h1(h1_text: str) -> Optional[str]:
    h1_text = h1_text.strip()
    m = _RECIPE_TITLE_RE.match(h1_text)
    if m:
        return m.group(1).strip()
    if h1_text.endswith("的做法"):
        return h1_text[: -len("的做法")].strip()
    return h1_text or None


class MarkdownParser(DocumentParser):
    doc_type = "markdown"
    supported_suffixes = (".md", ".markdown")

    def parse(self, path: Path) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw = strip_html_comments(raw)
        lines = raw.split("\n")

        blocks: List[TextBlock] = []
        current_section = SECTION_INTRO
        title: Optional[str] = None
        recipe_name: Optional[str] = None
        difficulty: Optional[str] = None
        intro_lines: List[str] = []
        block_index = 0

        def flush_intro() -> None:
            nonlocal block_index
            text = normalize_text("\n".join(intro_lines))
            intro_lines.clear()
            if text:
                blocks.append(
                    self.make_block(text, SECTION_INTRO, block_index, path)
                )
                block_index += 1

        def add_block(text: str, section: str, image_refs: Optional[List[str]] = None) -> None:
            nonlocal block_index
            text = normalize_text(text)
            if not text and not image_refs:
                return
            blocks.append(
                self.make_block(
                    text,
                    section,
                    block_index,
                    path,
                    image_refs=image_refs or [],
                )
            )
            block_index += 1

        seen_h2 = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            h1 = _H1_RE.match(stripped)
            if h1:
                flush_intro()
                title = h1.group(1).strip()
                recipe_name = _recipe_name_from_h1(title)
                current_section = SECTION_INTRO
                continue

            h2 = _H2_RE.match(stripped)
            if h2:
                flush_intro()
                seen_h2 = True
                current_section = _section_from_heading(h2.group(1))
                continue

            diff = _DIFFICULTY_RE.match(stripped)
            if diff:
                difficulty = diff.group(1).strip()
                if not seen_h2:
                    intro_lines.append(stripped)
                continue

            md_images = self.extract_markdown_images(stripped)
            if md_images and stripped.startswith("!"):
                flush_intro()
                alts, refs = zip(*md_images) if md_images else ([], [])
                caption = " / ".join(a for a in alts if a) or "(image)"
                add_block(caption, current_section, image_refs=list(refs))
                continue

            list_m = _LIST_ITEM_RE.match(stripped)
            if list_m:
                flush_intro()
                add_block(list_m.group(1), current_section)
                continue

            if not seen_h2:
                intro_lines.append(stripped)
            else:
                flush_intro()
                add_block(stripped, current_section)

        flush_intro()

        metadata = {
            "difficulty": difficulty,
            "category_path": _infer_category(path),
        }

        return self.build_document(
            path,
            blocks,
            title=title,
            recipe_name=recipe_name or path.parent.name,
            metadata=metadata,
        )


def _infer_category(path: Path) -> Optional[str]:
    """从 data/dishes/<category>/... 推断菜系分类目录名。"""
    parts = path.resolve().parts
    try:
        idx = parts.index("dishes")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return None
