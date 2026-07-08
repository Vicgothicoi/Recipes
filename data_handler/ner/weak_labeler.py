"""从 ParsedDocument 生成弱监督 NER 标注（用于引导训练）。"""

from __future__ import annotations

import re
from typing import List, Optional

from data_handler.ner.dataset import NerSample
from data_handler.ner.labels import SECTION_ENTITY_HINT
from data_handler.schemas import (
    SECTION_CALCULATION,
    SECTION_INGREDIENTS,
    SECTION_INTRO,
    SECTION_STEPS,
    ParsedDocument,
)

_AMOUNT_RE = re.compile(
    r"^(.+?)\s+(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*([a-zA-Z\u4e00-\u9fff]+)?\s*$"
)


def _strip_optional_suffix(text: str) -> str:
    text = re.sub(r"[（(].*?[）)]", "", text).strip()
    return text


def weak_label_block(
    text: str,
    section: str,
    *,
    recipe_name: Optional[str] = None,
) -> List[dict]:
    """单块文本的字符级弱标注。"""
    entities: List[dict] = []
    hint = SECTION_ENTITY_HINT.get(section, "O")
    if hint == "O" or not text.strip():
        return entities

    if section == SECTION_INTRO and recipe_name and recipe_name in text:
        start = text.index(recipe_name)
        entities.append(
            {
                "start": start,
                "end": start + len(recipe_name),
                "label": "Recipe",
            }
        )

    if section in (SECTION_INGREDIENTS, SECTION_CALCULATION):
        name = _strip_optional_suffix(text)
        m = _AMOUNT_RE.match(text)
        if m:
            name = m.group(1).strip()
        if name:
            start = text.find(name)
            if start >= 0:
                entities.append(
                    {
                        "start": start,
                        "end": start + len(name),
                        "label": "Ingredient",
                    }
                )
        return entities

    if section == SECTION_STEPS:
        entities.append({"start": 0, "end": len(text), "label": "CookingStep"})
        return entities

    return entities


def weak_label_document(document: ParsedDocument) -> List[NerSample]:
    """将 ParsedDocument 各块转为 NerSample 列表。"""
    samples: List[NerSample] = []
    for block in document.blocks:
        if block.metadata.get("type") == "image_caption":
            continue
        text = block.text.strip()
        if not text or text == "(image)":
            continue
        ents = weak_label_block(
            text,
            block.section,
            recipe_name=document.recipe_name,
        )
        samples.append(
            NerSample(
                text=text,
                entities=ents,
                metadata={
                    "source_path": document.source_path,
                    "section": block.section,
                    "block_index": block.block_index,
                    "weak": True,
                },
            )
        )
    return samples
