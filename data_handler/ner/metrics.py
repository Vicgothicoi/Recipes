"""实体级精确率 / 召回率 / F1。"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

Span = Tuple[int, int, str]


def spans_from_entities(entities: List[Dict]) -> Set[Span]:
    out: Set[Span] = set()
    for e in entities:
        out.add((int(e["start"]), int(e["end"]), e["label"]))
    return out


def spans_from_bio(text: str, bio_tags: List[str]) -> Set[Span]:
    from data_handler.ner.labels import bio_to_spans

    return set(bio_to_spans(text, bio_tags))


def compute_span_f1(
    true_spans: Set[Span],
    pred_spans: Set[Span],
) -> Dict[str, float]:
    if not pred_spans and not true_spans:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_spans:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not true_spans:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(true_spans & pred_spans)
    precision = tp / len(pred_spans)
    recall = tp / len(true_spans)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
