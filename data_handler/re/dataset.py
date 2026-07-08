"""关系分类数据集。"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from data_handler.re.labels import LABEL2ID, RELATION_NONE

logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"text": self.text, "label": self.label}
        if self.start is not None:
            d["start"] = self.start
        if self.end is not None:
            d["end"] = self.end
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityMention:
        return cls(
            text=data["text"],
            label=data["label"],
            start=data.get("start"),
            end=data.get("end"),
        )


@dataclass
class ReSample:
    """关系分类单样本。"""

    text: str
    head: EntityMention
    tail: EntityMention
    relation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
            "relation": self.relation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReSample:
        return cls(
            text=data["text"],
            head=EntityMention.from_dict(data["head"]),
            tail=EntityMention.from_dict(data["tail"]),
            relation=data["relation"],
            metadata=dict(data.get("metadata", {})),
        )


def format_re_input(head: EntityMention, tail: EntityMention, context: str) -> str:
    """构造 BERT 输入（无需改词表）。"""
    return (
        f"头实体:{head.text}({head.label}) "
        f"尾实体:{tail.text}({tail.label}) "
        f"句子:{context}"
    )


def load_jsonl(path: Path | str) -> List[ReSample]:
    path = Path(path)
    samples: List[ReSample] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(ReSample.from_dict(json.loads(line)))
            except json.JSONDecodeError as e:
                logger.warning("跳过无效 JSONL 行 %s:%d: %s", path, lineno, e)
    return samples


def save_jsonl(path: Path | str, samples: Sequence[ReSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def train_val_split(
    samples: Sequence[ReSample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[ReSample], List[ReSample]]:
    items = list(samples)
    random.Random(seed).shuffle(items)
    if len(items) < 2:
        return items, []
    val_size = max(1, int(len(items) * val_ratio))
    return items[val_size:], items[:val_size]


def add_negative_samples(
    positives: List[ReSample],
    *,
    ratio: float = 1.0,
    seed: int = 42,
) -> List[ReSample]:
    """为同一 context 内随机实体对生成 NONE 负样本。"""
    rng = random.Random(seed)
    by_source: Dict[str, List[ReSample]] = {}
    for s in positives:
        key = s.metadata.get("source_path", s.text[:32])
        by_source.setdefault(key, []).append(s)

    negatives: List[ReSample] = []
    positive_pairs = {(s.head.text, s.tail.text) for s in positives}

    for _, group in by_source.items():
        heads = list({s.head.text: s.head for s in group}.values())
        tails = list({s.tail.text: s.tail for s in group}.values())
        context = group[0].text
        n_neg = max(1, int(len(group) * ratio))
        tries = 0
        while len(negatives) < n_neg and tries < n_neg * 20:
            tries += 1
            h = rng.choice(heads)
            t = rng.choice(tails)
            if (h.text, t.text) in positive_pairs:
                continue
            negatives.append(
                ReSample(
                    text=context,
                    head=h,
                    tail=t,
                    relation=RELATION_NONE,
                    metadata={**group[0].metadata, "negative": True},
                )
            )
    return positives + negatives


class ReDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ReSample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        encoded_text = format_re_input(sample.head, sample.tail, sample.text)
        encoding = self.tokenizer(
            encoded_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL2ID[sample.relation], dtype=torch.long),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item


def re_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = ["input_ids", "attention_mask", "labels"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])
    return out


def create_dataloader(
    samples: Sequence[ReSample],
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        ReDataset(samples, tokenizer, max_length=max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=re_collate_fn,
    )
