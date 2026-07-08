"""data_handler 配置（路径与解析选项）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class DataHandlerConfig:
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    dishes_dir: Path | None = None
    intermediate_dir: Path | None = None

    txt_encoding: str = "utf-8"
    txt_encoding_fallback: str = "gb18030"

    # 视觉模型（OpenAI 兼容 API，默认复用 OPENAI_API_KEY / OPENAI_BASE_URL）
    vision_model: str | None = None
    vision_max_tokens: int = 512
    vision_temperature: float = 0.2
    caption_cache_dir: Path | None = None

    # NER（BERT + CRF）
    ner_bert_model: str = "hfl/chinese-bert-wwm-ext"
    ner_checkpoint_dir: Path | None = None
    ner_annotations_dir: Path | None = None
    ner_max_length: int = 256
    ner_batch_size: int = 16
    ner_learning_rate: float = 5e-5
    ner_epochs: int = 10
    ner_dropout: float = 0.1

    # RE（BERT 关系分类）
    re_bert_model: str = "hfl/chinese-bert-wwm-ext"
    re_checkpoint_dir: Path | None = None
    re_annotations_dir: Path | None = None
    re_max_length: int = 256
    re_batch_size: int = 16
    re_learning_rate: float = 5e-5
    re_epochs: int = 10
    re_dropout: float = 0.1
    re_min_score: float = 0.5

    # 图谱 CSV 输出目录
    cypher_out_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.dishes_dir is None:
            self.dishes_dir = self.project_root / "data" / "dishes"
        if self.intermediate_dir is None:
            self.intermediate_dir = self.project_root / "data" / "intermediate"
        self.dishes_dir = Path(self.dishes_dir)
        self.intermediate_dir = Path(self.intermediate_dir)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        if self.vision_model is None:
            self.vision_model = os.getenv("VISION_MODEL") or os.getenv("LLM_MODEL")
        if self.caption_cache_dir is None:
            self.caption_cache_dir = self.intermediate_dir / "image_captions"

        if self.ner_checkpoint_dir is None:
            self.ner_checkpoint_dir = self.project_root / "checkpoints" / "ner"
        self.ner_checkpoint_dir = Path(self.ner_checkpoint_dir)

        if self.ner_annotations_dir is None:
            self.ner_annotations_dir = self.project_root / "data" / "annotations" / "ner"
        self.ner_annotations_dir = Path(self.ner_annotations_dir)

        env_ner_model = os.getenv("NER_BERT_MODEL")
        if env_ner_model:
            self.ner_bert_model = env_ner_model

        if self.re_checkpoint_dir is None:
            self.re_checkpoint_dir = self.project_root / "checkpoints" / "re"
        self.re_checkpoint_dir = Path(self.re_checkpoint_dir)

        if self.re_annotations_dir is None:
            self.re_annotations_dir = self.project_root / "data" / "annotations" / "re"
        self.re_annotations_dir = Path(self.re_annotations_dir)

        env_re_model = os.getenv("RE_BERT_MODEL")
        if env_re_model:
            self.re_bert_model = env_re_model

        if self.cypher_out_dir is None:
            self.cypher_out_dir = self.project_root / "data" / "cypher"
        self.cypher_out_dir = Path(self.cypher_out_dir)

    @classmethod
    def from_env(cls) -> DataHandlerConfig:
        root = Path(os.getenv("PROJECT_ROOT", str(_PROJECT_ROOT)))
        dishes = os.getenv("DISHES_DIR")
        intermediate = os.getenv("DATA_INTERMEDIATE_DIR")
        vision_model = os.getenv("VISION_MODEL") or os.getenv("LLM_MODEL")
        vision_max = int(os.getenv("VISION_MAX_TOKENS", "512"))
        vision_temp = float(os.getenv("VISION_TEMPERATURE", "0.2"))
        cache_dir = os.getenv("VISION_CACHE_DIR")
        ner_ckpt = os.getenv("NER_CHECKPOINT_DIR")
        ner_ann = os.getenv("NER_ANNOTATIONS_DIR")
        return cls(
            project_root=root,
            dishes_dir=Path(dishes) if dishes else None,
            intermediate_dir=Path(intermediate) if intermediate else None,
            vision_model=vision_model,
            vision_max_tokens=vision_max,
            vision_temperature=vision_temp,
            caption_cache_dir=Path(cache_dir) if cache_dir else None,
            ner_bert_model=os.getenv("NER_BERT_MODEL", "hfl/chinese-bert-wwm-ext"),
            ner_checkpoint_dir=Path(ner_ckpt) if ner_ckpt else None,
            ner_annotations_dir=Path(ner_ann) if ner_ann else None,
            ner_max_length=int(os.getenv("NER_MAX_LENGTH", "256")),
            ner_batch_size=int(os.getenv("NER_BATCH_SIZE", "16")),
            ner_learning_rate=float(os.getenv("NER_LEARNING_RATE", "5e-5")),
            ner_epochs=int(os.getenv("NER_EPOCHS", "10")),
            re_bert_model=os.getenv("RE_BERT_MODEL", "hfl/chinese-bert-wwm-ext"),
            re_checkpoint_dir=Path(os.getenv("RE_CHECKPOINT_DIR"))
            if os.getenv("RE_CHECKPOINT_DIR")
            else None,
            re_annotations_dir=Path(os.getenv("RE_ANNOTATIONS_DIR"))
            if os.getenv("RE_ANNOTATIONS_DIR")
            else None,
            re_max_length=int(os.getenv("RE_MAX_LENGTH", "256")),
            re_batch_size=int(os.getenv("RE_BATCH_SIZE", "16")),
            re_learning_rate=float(os.getenv("RE_LEARNING_RATE", "5e-5")),
            re_epochs=int(os.getenv("RE_EPOCHS", "10")),
            re_min_score=float(os.getenv("RE_MIN_SCORE", "0.5")),
            cypher_out_dir=Path(os.getenv("CYPHER_OUT_DIR"))
            if os.getenv("CYPHER_OUT_DIR")
            else None,
        )


DEFAULT_CONFIG = DataHandlerConfig()
