"""BERT+CRF NER 训练。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_handler.config import DataHandlerConfig, DEFAULT_CONFIG
from data_handler.ner.bert_crf_model import BertCrfModel
from data_handler.ner.dataset import (
    NerSample,
    create_dataloader,
    load_jsonl,
    save_jsonl,
    train_val_split,
)
from data_handler.ner.labels import ID2LABEL, LABEL2ID, NUM_LABELS, bio_to_spans
from data_handler.ner.metrics import compute_span_f1, spans_from_entities
from data_handler.ner.decoding import subword_ids_to_char_bio
from data_handler.ner.weak_labeler import weak_label_document
from data_handler.parsers import parse_directory
logger = logging.getLogger(__name__)


class NerTrainer:
    def __init__(
        self,
        config: Optional[DataHandlerConfig] = None,
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.model_name = model_name or self.config.ner_bert_model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BertCrfModel(self.model_name, num_labels=NUM_LABELS).to(
            self.device
        )
        self.checkpoint_dir = Path(self.config.ner_checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def build_weak_dataset(
        self,
        dishes_dir: Optional[Path] = None,
        *,
        limit: Optional[int] = None,
    ) -> List[NerSample]:
        dishes_dir = Path(dishes_dir or self.config.dishes_dir)
        docs = parse_directory(dishes_dir, recursive=True)
        if limit:
            docs = docs[:limit]
        samples: List[NerSample] = []
        for doc in docs:
            samples.extend(weak_label_document(doc))
        logger.info("弱标注样本数: %d（来自 %d 篇文档）", len(samples), len(docs))
        return samples

    def train(
        self,
        train_samples: List[NerSample],
        val_samples: Optional[List[NerSample]] = None,
        *,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        warmup_ratio: float = 0.1,
        save_name: str = "best",
    ) -> Dict[str, Any]:
        epochs = epochs or self.config.ner_epochs
        batch_size = batch_size or self.config.ner_batch_size
        learning_rate = learning_rate or self.config.ner_learning_rate

        train_loader = create_dataloader(
            train_samples,
            self.tokenizer,
            batch_size=batch_size,
            max_length=self.config.ner_max_length,
            shuffle=True,
        )
        val_loader = None
        if val_samples:
            val_loader = create_dataloader(
                val_samples,
                self.tokenizer,
                batch_size=batch_size,
                max_length=self.config.ner_max_length,
                shuffle=False,
            )

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_f1 = -1.0
        history: Dict[str, Any] = {"epochs": []}

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            epoch_log: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
            }
            if val_loader:
                metrics = self.evaluate(val_loader, val_samples or [])
                epoch_log["val"] = metrics
                logger.info(
                    "Epoch %d/%d  loss=%.4f  val_f1=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    metrics["f1"],
                )
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    self.save_checkpoint(save_name)
            else:
                logger.info("Epoch %d/%d  loss=%.4f", epoch, epochs, train_loss)
                self.save_checkpoint(save_name)
            history["epochs"].append(epoch_log)

        return history

    def _train_epoch(self, loader, optimizer, scheduler) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            _, loss = self.model(
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
            steps += 1
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        val_samples: List[NerSample],
    ) -> Dict[str, float]:
        self.model.eval()
        all_true = []
        all_pred = []
        sample_idx = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            tag_paths = self.model.predict(
                input_ids, attention_mask, token_type_ids=token_type_ids
            )
            for i, tags in enumerate(tag_paths):
                text = batch["texts"][i]
                encoding = self.tokenizer(
                    text,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    max_length=self.config.ner_max_length,
                    truncation=True,
                )
                offsets = encoding["offset_mapping"][0].tolist()
                char_bio = subword_ids_to_char_bio(text, tags, offsets)
                pred_spans = set(bio_to_spans(text, char_bio))
                if sample_idx < len(val_samples):
                    true_spans = spans_from_entities(val_samples[sample_idx].entities)
                    all_true.append(true_spans)
                    all_pred.append(pred_spans)
                    sample_idx += 1

        if not all_true:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        tp = sum(len(t & p) for t, p in zip(all_true, all_pred))
        pred_total = sum(len(p) for p in all_pred)
        true_total = sum(len(t) for t in all_true)
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / true_total if true_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    def save_checkpoint(self, name: str = "best") -> Path:
        ckpt_dir = self.checkpoint_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        meta = {
            "model_name": self.model_name,
            "label2id": LABEL2ID,
            "id2label": {int(k): v for k, v in ID2LABEL.items()},
            "num_labels": NUM_LABELS,
            "max_length": self.config.ner_max_length,
        }
        (ckpt_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.tokenizer.save_pretrained(ckpt_dir / "tokenizer")
        logger.info("已保存 checkpoint: %s", ckpt_dir)
        return ckpt_dir

    def load_checkpoint(self, name: str = "best") -> None:
        ckpt_dir = self.checkpoint_dir / name
        state = torch.load(ckpt_dir / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        meta_path = ckpt_dir / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.model_name = meta.get("model_name", self.model_name)
        logger.info("已加载 checkpoint: %s", ckpt_dir)

    def train_from_jsonl(
        self,
        path: Path | str,
        *,
        val_ratio: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        samples = load_jsonl(path)
        train, val = train_val_split(samples, val_ratio=val_ratio)
        return self.train(train, val, **kwargs)

    def train_from_weak_labels(
        self,
        *,
        dishes_dir: Optional[Path] = None,
        limit: Optional[int] = None,
        output_jsonl: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        samples = self.build_weak_dataset(dishes_dir, limit=limit)
        if output_jsonl:
            save_jsonl(output_jsonl, samples)
        train, val = train_val_split(samples, val_ratio=0.1)
        return self.train(train, val, **kwargs)
