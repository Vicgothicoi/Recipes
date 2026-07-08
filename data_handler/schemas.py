"""data_handler 共用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# 与 HowToCook 模板章节对应的 section 名称
SECTION_INTRO = "intro"
SECTION_INGREDIENTS = "ingredients"
SECTION_CALCULATION = "calculation"
SECTION_STEPS = "steps"
SECTION_APPENDIX = "appendix"
SECTION_BODY = "body"


@dataclass
class EntitySpan:
    """识别出的实体片段（字符级起止下标）。"""

    start: int
    end: int
    label: str
    text: str
    score: float = 1.0
    section: Optional[str] = None
    source_path: Optional[str] = None
    block_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text": self.text,
            "score": self.score,
            "section": self.section,
            "source_path": self.source_path,
            "block_index": self.block_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntitySpan:
        return cls(
            start=int(data["start"]),
            end=int(data["end"]),
            label=data["label"],
            text=data["text"],
            score=float(data.get("score", 1.0)),
            section=data.get("section"),
            source_path=data.get("source_path"),
            block_index=data.get("block_index"),
        )


@dataclass
class RelationTriple:
    """识别出的关系三元组。"""

    head_text: str
    head_label: str
    tail_text: str
    tail_label: str
    relation_type: str
    relation_name: str
    score: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_text": self.head_text,
            "head_label": self.head_label,
            "tail_text": self.tail_text,
            "tail_label": self.tail_label,
            "relation_type": self.relation_type,
            "relation_name": self.relation_name,
            "score": self.score,
            "properties": dict(self.properties),
            "source_path": self.source_path,
        }


@dataclass
class TextBlock:
    """文档中的一个文本块（段落、列表项或页面片段）。"""

    text: str
    section: str = SECTION_BODY
    block_index: int = 0
    image_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "section": self.section,
            "block_index": self.block_index,
            "image_refs": list(self.image_refs),
            "metadata": dict(self.metadata),
        }


@dataclass
class ParsedDocument:
    """解析后的统一文档表示。"""

    source_path: str
    doc_type: str
    blocks: List[TextBlock] = field(default_factory=list)
    title: Optional[str] = None
    recipe_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "doc_type": self.doc_type,
            "title": self.title,
            "recipe_name": self.recipe_name,
            "metadata": dict(self.metadata),
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParsedDocument:
        blocks = [
            TextBlock(
                text=b["text"],
                section=b.get("section", SECTION_BODY),
                block_index=b.get("block_index", 0),
                image_refs=list(b.get("image_refs", [])),
                metadata=dict(b.get("metadata", {})),
            )
            for b in data.get("blocks", [])
        ]
        return cls(
            source_path=data["source_path"],
            doc_type=data["doc_type"],
            blocks=blocks,
            title=data.get("title"),
            recipe_name=data.get("recipe_name"),
            metadata=dict(data.get("metadata", {})),
        )


def resolve_path(base: Path, ref: str) -> str:
    """将相对路径解析为绝对路径；http(s) 保持不变。"""
    ref = ref.strip()
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    return str((base.parent / ref).resolve())
