"""BERT 关系分类推理。"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from data_handler.config import DataHandlerConfig, DEFAULT_CONFIG
from data_handler.re.bert_re_model import BertRelationClassifier
from data_handler.re.dataset import EntityMention, ReSample, format_re_input
from data_handler.re.labels import (
    ID2LABEL,
    LABEL2ID,
    NUM_RELATIONS,
    RELATION_NONE,
    candidate_relations,
    is_allowed_pair,
    relation_name,
)
from data_handler.schemas import EntitySpan, ParsedDocument, RelationTriple

logger = logging.getLogger(__name__)


class RePredictor:
    def __init__(
        self,
        checkpoint: str | Path = "best",
        config: Optional[DataHandlerConfig] = None,
        device: Optional[str] = None,
        *,
        min_score: float = 0.5,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.checkpoint_dir = Path(self.config.re_checkpoint_dir) / str(checkpoint)
        self.min_score = min_score
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        meta_path = self.checkpoint_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"未找到 RE checkpoint: {self.checkpoint_dir}，请先训练。"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model_name = meta["model_name"]
        self.max_length = meta.get("max_length", self.config.re_max_length)
        self.id2label = {int(k): v for k, v in meta.get("id2label", ID2LABEL).items()}
        self.none_label = meta.get("none_label", RELATION_NONE)

        tok_dir = self.checkpoint_dir / "tokenizer"
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tok_dir)
            if tok_dir.is_dir()
            else AutoTokenizer.from_pretrained(self.model_name)
        )
        self.model = BertRelationClassifier(
            self.model_name, num_labels=meta.get("num_labels", NUM_RELATIONS)
        )
        try:
            state = torch.load(
                self.checkpoint_dir / "model.pt",
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(
                self.checkpoint_dir / "model.pt", map_location=self.device
            )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_pair(
        self,
        head: EntityMention,
        tail: EntityMention,
        context: str,
    ) -> Tuple[str, float]:
        """返回 (relation_type, confidence)。"""
        encoded = format_re_input(head, tail, context)
        encoding = self.tokenizer(
            encoded,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            logits, _ = self.model(
                input_ids, attention_mask, token_type_ids=token_type_ids
            )
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = int(probs.argmax().item())
            score = float(probs[pred_id].item())
        return self.id2label[pred_id], score

    def predict_sample(self, sample: ReSample) -> RelationTriple:
        rel_type, score = self.predict_pair(sample.head, sample.tail, sample.text)
        props = dict(sample.metadata.get("properties", {}))
        return RelationTriple(
            head_text=sample.head.text,
            head_label=sample.head.label,
            tail_text=sample.tail.text,
            tail_label=sample.tail.label,
            relation_type=rel_type,
            relation_name=relation_name(rel_type),
            score=score,
            properties=props,
            source_path=sample.metadata.get("source_path"),
        )

    def predict_entities(
        self,
        document: ParsedDocument,
        entities: List[EntitySpan],
        *,
        context: Optional[str] = None,
    ) -> List[RelationTriple]:
        """对文档内实体做两两（或定向）关系预测，过滤 NONE 与非法类型对。"""
        if len(entities) < 2 and not any(e.label == "Recipe" for e in entities):
            recipe = document.recipe_name
            if recipe:
                entities = list(entities)
                entities.insert(
                    0,
                    EntitySpan(
                        start=0,
                        end=len(recipe),
                        label="Recipe",
                        text=recipe,
                        source_path=document.source_path,
                    ),
                )

        ctx = context or document.full_text
        triples: List[RelationTriple] = []

        recipes = [e for e in entities if e.label == "Recipe"]
        others = [e for e in entities if e.label != "Recipe"]

        pairs: List[Tuple[EntitySpan, EntitySpan]] = []
        if recipes:
            head_ent = recipes[0]
            for tail_ent in others:
                pairs.append((head_ent, tail_ent))
        else:
            pairs = list(combinations(entities, 2))

        for head_ent, tail_ent in pairs:
            candidates = candidate_relations(head_ent.label, tail_ent.label)
            if not candidates and not is_allowed_pair(
                head_ent.label, tail_ent.label, RELATION_NONE
            ):
                continue

            block_ctx = ctx
            if tail_ent.block_index is not None:
                for b in document.blocks:
                    if b.block_index == tail_ent.block_index:
                        block_ctx = b.text
                        break

            head = EntityMention(
                text=head_ent.text,
                label=head_ent.label,
                start=head_ent.start,
                end=head_ent.end,
            )
            tail = EntityMention(
                text=tail_ent.text,
                label=tail_ent.label,
                start=tail_ent.start,
                end=tail_ent.end,
            )
            rel_type, score = self.predict_pair(head, tail, block_ctx)

            if rel_type == self.none_label:
                continue
            if candidates and rel_type not in candidates:
                continue
            if score < self.min_score:
                continue

            props = {}
            if rel_type == "801000001":
                from data_handler.re.weak_labeler import _parse_amount_unit

                props = _parse_amount_unit(block_ctx)
            elif rel_type == "801000003":
                step_blocks = [
                    e
                    for e in entities
                    if e.label == "CookingStep" and e.block_index is not None
                ]
                order = 1
                for i, e in enumerate(sorted(step_blocks, key=lambda x: x.block_index or 0)):
                    if e.text == tail_ent.text:
                        order = i + 1
                        break
                props = {"step_order": float(order)}

            triples.append(
                RelationTriple(
                    head_text=head_ent.text,
                    head_label=head_ent.label,
                    tail_text=tail_ent.text,
                    tail_label=tail_ent.label,
                    relation_type=rel_type,
                    relation_name=relation_name(rel_type),
                    score=score,
                    properties=props,
                    source_path=document.source_path,
                )
            )

        return triples

    def predict_document(
        self,
        document: ParsedDocument,
        entities: Optional[List[EntitySpan]] = None,
    ) -> List[RelationTriple]:
        if entities is None:
            entities = []
            for ent in document.metadata.get("entities", []):
                entities.append(EntitySpan.from_dict(ent))
        triples = self.predict_entities(document, entities)
        document.metadata["relations"] = [t.to_dict() for t in triples]
        return triples
