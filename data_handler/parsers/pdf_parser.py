"""PDF 解析器（基于 pypdf）。"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from data_handler.parsers.base import DocumentParser
from data_handler.schemas import SECTION_BODY, TextBlock
from data_handler.utils.text_normalize import normalize_text


class PdfParser(DocumentParser):
    doc_type = "pdf"
    supported_suffixes = (".pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        reader = PdfReader(str(path))
        blocks: List[TextBlock] = []
        meta = reader.metadata or {}
        title = meta.get("/Title") or meta.get("Title") or path.stem

        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = normalize_text(raw)
            if not text:
                continue
            blocks.append(
                self.make_block(
                    text,
                    SECTION_BODY,
                    len(blocks),
                    path,
                    page=page_num,
                )
            )

        return self.build_document(
            path,
            blocks,
            title=str(title) if title else path.stem,
            metadata={
                "page_count": len(reader.pages),
                "pdf_metadata": {
                    k: str(v) for k, v in meta.items() if v is not None
                },
            },
        )
