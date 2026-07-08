"""BERT + [CLS] 关系分类。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from data_handler.re.labels import NUM_RELATIONS


class BertRelationClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_RELATIONS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained(model_name)
        self.bert: PreTrainedModel = AutoModel.from_pretrained(model_name, config=config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.bert(**kwargs)
        pooled = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return logits, loss
