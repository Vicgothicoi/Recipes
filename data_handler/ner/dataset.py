"""NER 数据集：JSONL 标注加载与 BERT 对齐。"""

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

from data_handler.ner.labels import LABEL2ID, O_LABEL, entities_to_bio

logger = logging.getLogger(__name__)


@dataclass
class NerSample:
    """单条训练/推理样本。"""

    text: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NerSample:
        return cls(
            text=data["text"],
            entities=list(data.get("entities", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entities": self.entities,
            "metadata": self.metadata,
        }


def load_jsonl(path: Path | str) -> List[NerSample]:
    path = Path(path)
    samples: List[NerSample] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(NerSample.from_dict(json.loads(line)))
            except json.JSONDecodeError as e:
                logger.warning("跳过无效 JSONL 行 %s:%d: %s", path, lineno, e)
    return samples


def save_jsonl(path: Path | str, samples: Sequence[NerSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def train_val_split(
    samples: Sequence[NerSample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[NerSample], List[NerSample]]:
    items = list(samples)
    random.Random(seed).shuffle(items)
    if len(items) < 2:
        return items, []
    val_size = max(1, int(len(items) * val_ratio))
    return items[val_size:], items[:val_size]


def align_labels_to_subwords(
    text: str,
    char_bio: List[str],
    offset_mapping: List[Tuple[int, int]],
) -> List[int]:
    """
    将字符级 BIO 对齐到 tokenizer 的 subword：
    每个字符仅第一个 subword 保留标签，其余为 -100（ignore）。
    """
    label_ids: List[int] = []
    prev_char_start: Optional[int] = None
    for start, end in offset_mapping:
        if start == end == 0:
            label_ids.append(-100)
            prev_char_start = None
            continue
        if start >= len(char_bio):
            label_ids.append(-100)
            continue
        if start == prev_char_start:
            label_ids.append(-100)
        else:
            tag = char_bio[start]
            label_ids.append(LABEL2ID.get(tag, LABEL2ID[O_LABEL]))
            prev_char_start = start
    return label_ids


class NerDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[NerSample],
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
        text = sample.text
        char_bio = entities_to_bio(text, sample.entities)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"][0].tolist()
        label_ids = align_labels_to_subwords(text, char_bio, offsets)

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "text": text,
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item


def ner_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = ["input_ids", "attention_mask", "labels"]
    out: Dict[str, Any] = {
        k: torch.stack([b[k] for b in batch]) for k in keys if k in batch[0]
    }
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])
    out["texts"] = [b["text"] for b in batch]
    return out


def create_dataloader(
    samples: Sequence[NerSample],
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    ds = NerDataset(samples, tokenizer, max_length=max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ner_collate_fn,
    )
