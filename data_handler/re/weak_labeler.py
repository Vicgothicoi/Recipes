"""从 ParsedDocument + 实体列表生成弱监督关系标注。"""

from __future__ import annotations

import re
from typing import List, Optional

from data_handler.re.dataset import EntityMention, ReSample
from data_handler.re.labels import (
    RELATION_CONTAINS_STEP,
    RELATION_REQUIRES,
)
from data_handler.schemas import EntitySpan, ParsedDocument, SECTION_CALCULATION, SECTION_INGREDIENTS, SECTION_STEPS

_AMOUNT_RE = re.compile(
    r"^(.+?)\s+(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*([a-zA-Z\u4e00-\u9fff]+)?\s*$"
)


def _parse_amount_unit(text: str) -> dict:
    m = _AMOUNT_RE.match(text.strip())
    if not m:
        return {}
    props = {}
    if m.group(2):
        props["amount"] = m.group(2)
    if m.group(3):
        props["unit"] = m.group(3)
    return props


def _recipe_mention(document: ParsedDocument) -> Optional[EntityMention]:
    name = document.recipe_name or document.title
    if not name:
        return None
    return EntityMention(text=name, label="Recipe")


def weak_label_from_entities(
    document: ParsedDocument,
    entities: List[EntitySpan],
) -> List[ReSample]:
    """根据实体 span 与文档结构生成 REQUIRES / CONTAINS_STEP 正样本。"""
    recipe = _recipe_mention(document)
    if not recipe:
        return []

    samples: List[ReSample] = []
    meta_base = {"source_path": document.source_path, "weak": True}

    ingredients = [e for e in entities if e.label == "Ingredient"]
    steps = [e for e in entities if e.label == "CookingStep"]

    for ent in ingredients:
        context = ent.text
        for block in document.blocks:
            if ent.block_index is not None and block.block_index == ent.block_index:
                context = block.text
                break
        props = _parse_amount_unit(context)
        samples.append(
            ReSample(
                text=context,
                head=recipe,
                tail=EntityMention(
                    text=ent.text,
                    label=ent.label,
                    start=ent.start,
                    end=ent.end,
                ),
                relation=RELATION_REQUIRES,
                metadata={**meta_base, "properties": props, "section": ent.section},
            )
        )

    step_order = 0
    for ent in steps:
        step_order += 1
        context = ent.text
        samples.append(
            ReSample(
                text=context,
                head=recipe,
                tail=EntityMention(
                    text=ent.text,
                    label=ent.label,
                    start=ent.start,
                    end=ent.end,
                ),
                relation=RELATION_CONTAINS_STEP,
                metadata={
                    **meta_base,
                    "properties": {"step_order": float(step_order)},
                    "section": ent.section,
                },
            )
        )

    return samples


def weak_label_document(
    document: ParsedDocument,
    entities: Optional[List[EntitySpan]] = None,
) -> List[ReSample]:
    """
    若未提供 entities，则按章节规则生成简易实体再建关系。
    """
    if entities is not None:
        return weak_label_from_entities(document, entities)

    from data_handler.ner.weak_labeler import weak_label_document as ner_weak

    entities_spans: List[EntitySpan] = []
    for sample in ner_weak(document):
        for ent in sample.entities:
            entities_spans.append(
                EntitySpan(
                    start=ent["start"],
                    end=ent["end"],
                    label=ent["label"],
                    text=sample.text[ent["start"] : ent["end"]],
                    section=sample.metadata.get("section"),
                    source_path=document.source_path,
                    block_index=sample.metadata.get("block_index"),
                )
            )

    if not entities_spans and document.recipe_name:
        for block in document.blocks:
            if block.section in (SECTION_INGREDIENTS, SECTION_CALCULATION):
                text = block.text.strip()
                if text:
                    entities_spans.append(
                        EntitySpan(
                            start=0,
                            end=len(text),
                            label="Ingredient",
                            text=text,
                            section=block.section,
                            source_path=document.source_path,
                            block_index=block.block_index,
                        )
                    )
            elif block.section == SECTION_STEPS:
                text = block.text.strip()
                if text:
                    entities_spans.append(
                        EntitySpan(
                            start=0,
                            end=len(text),
                            label="CookingStep",
                            text=text,
                            section=block.section,
                            source_path=document.source_path,
                            block_index=block.block_index,
                        )
                    )

    return weak_label_from_entities(document, entities_spans)
