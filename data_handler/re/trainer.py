"""BERT 关系分类训练。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_handler.config import DataHandlerConfig, DEFAULT_CONFIG
from data_handler.parsers import parse_directory
from data_handler.re.bert_re_model import BertRelationClassifier
from data_handler.re.dataset import (
    ReSample,
    add_negative_samples,
    create_dataloader,
    load_jsonl,
    save_jsonl,
    train_val_split,
)
from data_handler.re.labels import ID2LABEL, LABEL2ID, NUM_RELATIONS, RELATION_NONE
from data_handler.re.weak_labeler import weak_label_document

logger = logging.getLogger(__name__)


class ReTrainer:
    def __init__(
        self,
        config: Optional[DataHandlerConfig] = None,
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.model_name = model_name or self.config.re_bert_model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BertRelationClassifier(
            self.model_name, num_labels=NUM_RELATIONS
        ).to(self.device)
        self.checkpoint_dir = Path(self.config.re_checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def build_weak_dataset(
        self,
        dishes_dir: Optional[Path] = None,
        *,
        limit: Optional[int] = None,
        negative_ratio: float = 1.0,
    ) -> List[ReSample]:
        dishes_dir = Path(dishes_dir or self.config.dishes_dir)
        docs = parse_directory(dishes_dir, recursive=True)
        if limit:
            docs = docs[:limit]
        positives: List[ReSample] = []
        for doc in docs:
            positives.extend(weak_label_document(doc))
        samples = add_negative_samples(positives, ratio=negative_ratio)
        logger.info(
            "弱标注关系样本: 正样本 %d, 总计 %d（%d 篇文档）",
            len(positives),
            len(samples),
            len(docs),
        )
        return samples

    def train(
        self,
        train_samples: List[ReSample],
        val_samples: Optional[List[ReSample]] = None,
        *,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        warmup_ratio: float = 0.1,
        save_name: str = "best",
    ) -> Dict[str, Any]:
        epochs = epochs or self.config.re_epochs
        batch_size = batch_size or self.config.re_batch_size
        learning_rate = learning_rate or self.config.re_learning_rate

        train_loader = create_dataloader(
            train_samples,
            self.tokenizer,
            batch_size=batch_size,
            max_length=self.config.re_max_length,
            shuffle=True,
        )
        val_loader = None
        if val_samples:
            val_loader = create_dataloader(
                val_samples,
                self.tokenizer,
                batch_size=batch_size,
                max_length=self.config.re_max_length,
                shuffle=False,
            )

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        best_f1 = -1.0
        history: Dict[str, Any] = {"epochs": []}

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, scheduler
            )
            epoch_log: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            if val_loader and val_samples:
                metrics = self.evaluate(val_loader, val_samples)
                epoch_log["val"] = metrics
                logger.info(
                    "Epoch %d/%d  loss=%.4f  acc=%.4f  val_f1=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    train_acc,
                    metrics["f1"],
                )
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    self.save_checkpoint(save_name)
            else:
                logger.info(
                    "Epoch %d/%d  loss=%.4f  acc=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    train_acc,
                )
                self.save_checkpoint(save_name)
            history["epochs"].append(epoch_log)

        return history

    def _train_epoch(self, loader, optimizer, scheduler) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            logits, loss = self.model(
                input_ids,
                attention_mask,
                labels=labels,
                token_type_ids=token_type_ids,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / max(len(loader), 1), correct / max(total, 1)

    @torch.no_grad()
    def evaluate(self, loader, val_samples: List[ReSample]) -> Dict[str, float]:
        self.model.eval()
        y_true: List[str] = []
        y_pred: List[str] = []
        idx = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            logits, _ = self.model(
                input_ids, attention_mask, token_type_ids=token_type_ids
            )
            preds = logits.argmax(dim=-1).cpu().tolist()
            for pred_id in preds:
                if idx < len(val_samples):
                    y_true.append(val_samples[idx].relation)
                    y_pred.append(ID2LABEL[pred_id])
                    idx += 1

        return _classification_metrics(y_true, y_pred)

    def save_checkpoint(self, name: str = "best") -> Path:
        ckpt_dir = self.checkpoint_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        meta = {
            "model_name": self.model_name,
            "label2id": LABEL2ID,
            "id2label": {int(k): v for k, v in ID2LABEL.items()},
            "num_labels": NUM_RELATIONS,
            "max_length": self.config.re_max_length,
            "none_label": RELATION_NONE,
        }
        (ckpt_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.tokenizer.save_pretrained(ckpt_dir / "tokenizer")
        logger.info("已保存 RE checkpoint: %s", ckpt_dir)
        return ckpt_dir

    def train_from_jsonl(
        self,
        path: Path | str,
        *,
        val_ratio: float = 0.1,
        negative_ratio: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        samples = load_jsonl(path)
        if negative_ratio > 0:
            pos = [s for s in samples if s.relation != RELATION_NONE]
            samples = add_negative_samples(pos, ratio=negative_ratio)
        train, val = train_val_split(samples, val_ratio=val_ratio)
        return self.train(train, val, **kwargs)

    def train_from_weak_labels(
        self,
        *,
        dishes_dir: Optional[Path] = None,
        limit: Optional[int] = None,
        output_jsonl: Optional[Path] = None,
        negative_ratio: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        samples = self.build_weak_dataset(
            dishes_dir, limit=limit, negative_ratio=negative_ratio
        )
        if output_jsonl:
            save_jsonl(output_jsonl, samples)
        train, val = train_val_split(samples, val_ratio=0.1)
        return self.train(train, val, **kwargs)


def _classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lb in labels:
        if lb == RELATION_NONE:
            continue
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lb and p == lb)
        prec = tp / max(sum(1 for p in y_pred if p == lb), 1)
        rec = tp / max(sum(1 for t in y_true if t == lb), 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    return {"accuracy": acc, "f1": macro_f1, "macro_f1_non_none": macro_f1}
