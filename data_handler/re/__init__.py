"""BERT 关系分类（实体对 + 上下文）。"""

from data_handler.re.dataset import ReSample, load_jsonl, save_jsonl, train_val_split
from data_handler.re.labels import RELATION_REQUIRES, RELATION_TYPES
from data_handler.re.predictor import RePredictor
from data_handler.re.trainer import ReTrainer
from data_handler.re.weak_labeler import weak_label_document

__all__ = [
    "RELATION_REQUIRES",
    "RELATION_TYPES",
    "ReSample",
    "load_jsonl",
    "save_jsonl",
    "train_val_split",
    "ReTrainer",
    "RePredictor",
    "weak_label_document",
]
