"""BERT + 线性链 CRF 序列标注模型。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from data_handler.ner.labels import NUM_LABELS


class LinearChainCRF(nn.Module):
    """线性链 CRF（batch_first）。"""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """负对数似然损失。"""
        log_den = self._compute_log_partition(emissions, mask)
        log_num = self._compute_score(emissions, tags, mask)
        nll = log_den - log_num
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        return nll

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        return self._viterbi_decode(emissions, mask)

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # emissions: (B, L, C)
        batch_size, seq_len, _ = emissions.shape
        mask = mask.bool()
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)

        for i in range(1, seq_len):
            emit = emissions[:, i].gather(1, tags[:, i : i + 1]).squeeze(1)
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            step = (emit + trans) * mask[:, i]
            score = score + step

        last_tag_indices = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        mask = mask.bool()
        log_prob = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            emit = emissions[:, i].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            next_log = log_prob.unsqueeze(2) + trans + emit
            log_prob = torch.logsumexp(next_log, dim=1)
            log_prob = torch.where(mask[:, i].unsqueeze(1), log_prob, log_prob)

        log_prob = log_prob + self.end_transitions
        return torch.logsumexp(log_prob, dim=1)

    def _viterbi_decode(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> List[List[int]]:
        batch_size, seq_len, num_tags = emissions.shape
        mask = mask.bool()
        history: List[torch.Tensor] = []

        score = self.start_transitions + emissions[:, 0]
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)

        score = score + self.end_transitions
        seq_ends = mask.long().sum(dim=1) - 1
        best_tags_list: List[List[int]] = []

        for b in range(batch_size):
            end = int(seq_ends[b].item())
            _, best_last = score[b].max(dim=0)
            best_tags = [int(best_last.item())]
            for hist in reversed(history[:end]):
                best_last = hist[b][best_tags[-1]]
                best_tags.append(int(best_last.item()))
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list


class BertCrfModel(nn.Module):
    """BERT encoder + dropout + linear + CRF。"""

    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
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
        self.crf = LinearChainCRF(num_labels)

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
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        mask = attention_mask.bool()
        if labels is not None:
            crf_labels = labels.clone()
            crf_mask = mask.clone()
            crf_mask[labels == -100] = False
            crf_labels[labels == -100] = 0
            loss = self.crf(emissions, crf_labels, crf_mask, reduction="mean")
            return emissions, loss
        return emissions, None

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        self.eval()
        with torch.no_grad():
            emissions, _ = self.forward(
                input_ids, attention_mask, labels=None, token_type_ids=token_type_ids
            )
            mask = attention_mask.bool()
            return self.crf.decode(emissions, mask)
