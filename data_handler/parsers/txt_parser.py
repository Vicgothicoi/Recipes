"""纯文本 (.txt) 解析器。"""

from __future__ import annotations

from pathlib import Path
from typing import List

from charset_normalizer import from_bytes

from data_handler.parsers.base import DocumentParser
from data_handler.schemas import SECTION_BODY, TextBlock
from data_handler.utils.text_normalize import normalize_text


class TxtParser(DocumentParser):
    doc_type = "txt"
    supported_suffixes = (".txt",)

    def _read_bytes(self, path: Path) -> tuple[str, str]:
        raw = path.read_bytes()
        if not raw:
            return "", "utf-8"
        result = from_bytes(raw).best()
        if result is None:
            return raw.decode("utf-8", errors="replace"), "utf-8"
        return str(result), result.encoding or "utf-8"

    def parse(self, path: Path) -> ParsedDocument:
        text, encoding = self._read_bytes(path)
        text = normalize_text(text)
        blocks: List[TextBlock] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs and text:
            paragraphs = [text]
        for i, para in enumerate(paragraphs):
            blocks.append(
                self.make_block(para, SECTION_BODY, i, path)
            )
        return self.build_document(
            path,
            blocks,
            title=path.stem,
            metadata={"encoding": encoding, "paragraph_count": len(blocks)},
        )
