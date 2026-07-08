"""HTML 解析器。"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

from data_handler.parsers.base import DocumentParser
from data_handler.schemas import SECTION_BODY, TextBlock
from data_handler.utils.text_normalize import normalize_text


class _SimpleHTMLExtractor(HTMLParser):
    """轻量 HTML 抽取：标题、段落、列表项、图片。"""

    def __init__(self) -> None:
        super().__init__()
        self.title: Optional[str] = None
        self._in_title = False
        self._in_script = False
        self._in_style = False
        self._buffer: List[str] = []
        self._blocks: List[tuple[str, List[str]]] = []
        self._pending_images: List[str] = []
        self._list_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        attr = dict(attrs)
        if tag in ("script", "style"):
            self._in_script = tag == "script"
            self._in_style = tag == "style"
            return
        if tag == "title":
            self._in_title = True
        elif tag == "img" and "src" in attr:
            self._pending_images.append(attr["src"])
        elif tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self._flush()
            if tag == "li":
                self._list_depth += 1
        elif tag == "br":
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("script", "style"):
            self._in_script = False
            self._in_style = False
            return
        if tag == "title":
            self._in_title = False
        elif tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self._flush()
            if tag == "li" and self._list_depth > 0:
                self._list_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_script or self._in_style:
            return
        if self._in_title:
            self.title = (self.title or "") + data
            return
        self._buffer.append(data)

    def _flush(self) -> None:
        text = normalize_text("".join(self._buffer))
        self._buffer.clear()
        if text:
            imgs = list(self._pending_images)
            self._pending_images.clear()
            self._blocks.append((text, imgs))
        elif self._pending_images:
            self._blocks.append(("", list(self._pending_images)))
            self._pending_images.clear()

    def close(self) -> None:
        self._flush()
        super().close()


class HtmlParser(DocumentParser):
    doc_type = "html"
    supported_suffixes = (".html", ".htm")

    def parse(self, path: Path) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw = re.sub(r"<!--[\s\S]*?-->", "", raw)

        extractor = _SimpleHTMLExtractor()
        extractor.feed(raw)
        extractor.close()

        blocks: List[TextBlock] = []
        for i, (text, imgs) in enumerate(extractor._blocks):
            if not text and not imgs:
                continue
            blocks.append(
                self.make_block(
                    text or "(image)",
                    SECTION_BODY,
                    i,
                    path,
                    image_refs=imgs,
                )
            )

        if not blocks:
            plain = normalize_text(re.sub(r"<[^>]+>", " ", raw))
            if plain:
                blocks.append(self.make_block(plain, SECTION_BODY, 0, path))

        html_imgs = self.extract_html_images(raw)
        if html_imgs and blocks:
            extra = self.resolve_image_refs(path, html_imgs)
            for b in blocks:
                for ref in extra:
                    if ref not in b.image_refs:
                        b.image_refs.append(ref)

        return self.build_document(
            path,
            blocks,
            title=extractor.title.strip() if extractor.title else path.stem,
            metadata={"block_count": len(blocks)},
        )
