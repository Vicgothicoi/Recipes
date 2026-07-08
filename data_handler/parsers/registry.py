"""解析器注册与批量调度。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Type

from data_handler.parsers.base import DocumentParser
from data_handler.parsers.html_parser import HtmlParser
from data_handler.parsers.markdown_parser import MarkdownParser
from data_handler.parsers.txt_parser import TxtParser
from data_handler.schemas import ParsedDocument

logger = logging.getLogger(__name__)

_PARSER_CLASSES: tuple[Type[DocumentParser], ...] = (
    MarkdownParser,
    HtmlParser,
    TxtParser,
)

try:
    from data_handler.parsers.pdf_parser import PdfParser

    _PARSER_CLASSES = (MarkdownParser, PdfParser, HtmlParser, TxtParser)
except ImportError:
    logger.debug("未安装 pypdf，PDF 解析不可用")

_SUFFIX_TO_PARSER: Dict[str, DocumentParser] = {}
for cls in _PARSER_CLASSES:
    instance = cls()
    for suffix in cls.supported_suffixes:
        _SUFFIX_TO_PARSER[suffix.lower()] = instance


def get_parser(path: Path) -> Optional[DocumentParser]:
    return _SUFFIX_TO_PARSER.get(path.suffix.lower())


def parse_file(path: Path | str) -> ParsedDocument:
    path = Path(path)
    parser = get_parser(path)
    if parser is None:
        raise ValueError(
            f"不支持的文件类型: {path.suffix!r}（路径: {path}）。"
            f"支持: {sorted(_SUFFIX_TO_PARSER)}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    logger.debug("解析文件: %s (%s)", path, parser.doc_type)
    return parser.parse(path)


def parse_directory(
    directory: Path | str,
    *,
    recursive: bool = True,
    suffixes: Optional[Iterable[str]] = None,
) -> List[ParsedDocument]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    allowed = {s.lower() for s in (suffixes or _SUFFIX_TO_PARSER.keys())}
    pattern = "**/*" if recursive else "*"
    results: List[ParsedDocument] = []
    errors: List[tuple[Path, Exception]] = []

    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        try:
            results.append(parse_file(path))
        except Exception as e:
            errors.append((path, e))
            logger.warning("解析失败 %s: %s", path, e)

    if errors:
        logger.info(
            "目录解析完成: 成功 %d, 失败 %d",
            len(results),
            len(errors),
        )
    return results


def supported_suffixes() -> List[str]:
    return sorted(_SUFFIX_TO_PARSER.keys())
