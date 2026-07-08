"""BERT+CRF NER 推理。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from data_handler.config import DataHandlerConfig, DEFAULT_CONFIG
from data_handler.ner.bert_crf_model import BertCrfModel
from data_handler.ner.decoding import subword_ids_to_char_bio
from data_handler.ner.labels import ID2LABEL, LABEL2ID, NUM_LABELS, bio_to_spans
from data_handler.parsers import parse_file
from data_handler.schemas import EntitySpan, ParsedDocument

logger = logging.getLogger(__name__)


class NerPredictor:
    def __init__(
        self,
        checkpoint: str | Path = "best",
        config: Optional[DataHandlerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.checkpoint_dir = Path(self.config.ner_checkpoint_dir) / str(checkpoint)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        meta_path = self.checkpoint_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"未找到 NER checkpoint: {self.checkpoint_dir}，请先训练或指定正确路径。"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model_name = meta["model_name"]
        self.max_length = meta.get("max_length", self.config.ner_max_length)
        self.label2id = meta.get("label2id", LABEL2ID)
        self.id2label = {int(k): v for k, v in meta.get("id2label", ID2LABEL).items()}

        tok_dir = self.checkpoint_dir / "tokenizer"
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tok_dir)
            if tok_dir.is_dir()
            else AutoTokenizer.from_pretrained(self.model_name)
        )
        self.model = BertCrfModel(self.model_name, num_labels=meta.get("num_labels", NUM_LABELS))
        try:
            state = torch.load(
                self.checkpoint_dir / "model.pt",
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(
                self.checkpoint_dir / "model.pt",
                map_location=self.device,
            )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        text: str,
        *,
        section: Optional[str] = None,
        source_path: Optional[str] = None,
        block_index: Optional[int] = None,
    ) -> List[EntitySpan]:
        if not text.strip():
            return []

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        tag_paths = self.model.predict(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )
        offsets = encoding["offset_mapping"][0].tolist()
        seq_len = int(attention_mask.sum().item())
        tags = tag_paths[0][:seq_len]
        char_bio = subword_ids_to_char_bio(text, tags, offsets)

        spans = []
        for start, end, label in bio_to_spans(text, char_bio):
            spans.append(
                EntitySpan(
                    start=start,
                    end=end,
                    label=label,
                    text=text[start:end],
                    section=section,
                    source_path=source_path,
                    block_index=block_index,
                )
            )
        return spans

    def predict_document(self, document: ParsedDocument) -> List[EntitySpan]:
        all_spans: List[EntitySpan] = []
        for block in document.blocks:
            if block.metadata.get("type") == "image_caption":
                continue
            text = block.text.strip()
            if not text or text == "(image)":
                continue
            spans = self.predict(
                text,
                section=block.section,
                source_path=document.source_path,
                block_index=block.block_index,
            )
            all_spans.extend(spans)
        document.metadata["entities"] = [s.to_dict() for s in all_spans]
        return all_spans

    def predict_file(self, path: Path | str) -> List[EntitySpan]:
        doc = parse_file(path)
        return self.predict_document(doc)
