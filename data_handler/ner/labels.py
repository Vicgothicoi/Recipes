"""NER 实体类型与 BIO 标签表（对齐 data/cypher/nodes.csv 的 labels）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# 从菜谱文本中抽取的实体类型（不含层次结构节点 Root / RecipeCategory 等）
ENTITY_TYPES: Tuple[str, ...] = (
    "Recipe",
    "Ingredient",
    "CookingStep",
    "CookingMethod",
    "CookingTool",
)

O_LABEL = "O"
BIO_PREFIXES = ("B", "I")

# 章节 → 默认实体类型提示（弱标注 / 推理后处理可用）
SECTION_ENTITY_HINT: Dict[str, str] = {
    "intro": "Recipe",
    "ingredients": "Ingredient",
    "calculation": "Ingredient",
    "steps": "CookingStep",
    "appendix": "O",
    "body": "O",
}


def build_bio_labels(entity_types: Tuple[str, ...] = ENTITY_TYPES) -> List[str]:
    labels = [O_LABEL]
    for et in entity_types:
        labels.append(f"B-{et}")
        labels.append(f"I-{et}")
    return labels


BIO_LABELS: List[str] = build_bio_labels()
LABEL2ID: Dict[str, int] = {lb: i for i, lb in enumerate(BIO_LABELS)}
ID2LABEL: Dict[int, str] = {i: lb for lb, i in LABEL2ID.items()}
NUM_LABELS = len(BIO_LABELS)


def entity_label_to_bio(label: str, is_begin: bool) -> str:
    if label == O_LABEL or not label:
        return O_LABEL
    prefix = "B" if is_begin else "I"
    return f"{prefix}-{label}"


def bio_to_entity_type(bio_label: str) -> Optional[str]:
    if bio_label == O_LABEL or "-" not in bio_label:
        return None
    return bio_label.split("-", 1)[1]


def entities_to_bio(
    text: str,
    entities: List[Dict],
    *,
    label_key: str = "label",
    start_key: str = "start",
    end_key: str = "end",
) -> List[str]:
    """
    将字符级实体标注转为与 text 等长的 BIO 标签序列。
    entities: [{"start": int, "end": int, "label": "Ingredient"}, ...]
    """
    tags = [O_LABEL] * len(text)
    sorted_ents = sorted(entities, key=lambda e: (e[start_key], e[end_key]))
    for ent in sorted_ents:
        start, end = int(ent[start_key]), int(ent[end_key])
        label = ent[label_key]
        if start < 0 or end > len(text) or start >= end:
            continue
        tags[start] = entity_label_to_bio(label, is_begin=True)
        for i in range(start + 1, end):
            tags[i] = entity_label_to_bio(label, is_begin=False)
    return tags


def bio_to_spans(text: str, bio_tags: List[str]) -> List[Tuple[int, int, str]]:
    """BIO 序列 → (start, end, entity_type) 列表。"""
    spans: List[Tuple[int, int, str]] = []
    i = 0
    n = len(bio_tags)
    while i < n:
        tag = bio_tags[i]
        if tag == O_LABEL:
            i += 1
            continue
        if tag.startswith("B-"):
            etype = bio_to_entity_type(tag)
            if not etype:
                i += 1
                continue
            start = i
            i += 1
            while i < n and bio_tags[i] == f"I-{etype}":
                i += 1
            spans.append((start, i, etype))
        else:
            i += 1
    return spans
