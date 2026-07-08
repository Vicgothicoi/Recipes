"""文档解析器抽象基类。"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

from data_handler.schemas import (
    ParsedDocument,
    TextBlock,
    resolve_path,
)

logger = logging.getLogger(__name__)

_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HTML_IMG_RE = re.compile(
    r'<img[^>]+src=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


class DocumentParser(ABC):
    """将单文件解析为 ParsedDocument。"""

    doc_type: ClassVar[str] = "unknown"
    supported_suffixes: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        raise NotImplementedError

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_suffixes

    @staticmethod
    def extract_markdown_images(text: str) -> List[Tuple[str, str]]:
        """返回 (alt, ref) 列表。"""
        return _MD_IMAGE_RE.findall(text)

    @staticmethod
    def extract_html_images(html: str) -> List[str]:
        return _HTML_IMG_RE.findall(html)

    def resolve_image_refs(self, base: Path, refs: List[str]) -> List[str]:
        resolved: List[str] = []
        for ref in refs:
            try:
                resolved.append(resolve_path(base, ref))
            except OSError as e:
                logger.warning("无法解析图片路径 %s: %s", ref, e)
                resolved.append(ref)
        return resolved

    def make_block(
        self,
        text: str,
        section: str,
        block_index: int,
        base: Path,
        image_refs: Optional[List[str]] = None,
        **metadata,
    ) -> TextBlock:
        refs = image_refs or []
        md_refs = [r for _, r in self.extract_markdown_images(text)]
        all_refs = list(dict.fromkeys(refs + md_refs))
        return TextBlock(
            text=text.strip(),
            section=section,
            block_index=block_index,
            image_refs=self.resolve_image_refs(base, all_refs),
            metadata=metadata,
        )

    def build_document(
        self,
        path: Path,
        blocks: List[TextBlock],
        *,
        title: Optional[str] = None,
        recipe_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ParsedDocument:
        path = path.resolve()
        return ParsedDocument(
            source_path=str(path),
            doc_type=self.doc_type,
            blocks=blocks,
            title=title,
            recipe_name=recipe_name,
            metadata=metadata or {},
        )
