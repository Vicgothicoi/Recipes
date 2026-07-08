"""标签解码：subword → 字符级 BIO。"""

from __future__ import annotations

from typing import List, Optional

from data_handler.ner.labels import ID2LABEL, O_LABEL, bio_to_entity_type


def subword_ids_to_char_bio(
    text: str,
    tag_ids: List[int],
    offset_mapping: List[tuple[int, int]],
) -> List[str]:
    """将 subword 预测标签还原为字符级 BIO。"""
    char_bio = [O_LABEL] * len(text)
    prev_char_start: Optional[int] = None

    for tag_id, (start, end) in zip(tag_ids, offset_mapping):
        if start == end == 0:
            prev_char_start = None
            continue
        if start >= len(text):
            continue
        if start == prev_char_start:
            continue
        label = ID2LABEL.get(tag_id, O_LABEL)
        if label == O_LABEL:
            prev_char_start = start
            continue
        char_bio[start] = label
        prev_char_start = start

    return _fill_inside_entities(char_bio)


def _fill_inside_entities(char_bio: List[str]) -> List[str]:
    """在 B- 与下一 B-/O 之间填充 I-。"""
    n = len(char_bio)
    i = 0
    while i < n:
        tag = char_bio[i]
        if not tag.startswith("B-"):
            i += 1
            continue
        etype = bio_to_entity_type(tag)
        if not etype:
            i += 1
            continue
        j = i + 1
        while j < n and char_bio[j] == O_LABEL:
            char_bio[j] = f"I-{etype}"
            j += 1
        i = j
    return char_bio
