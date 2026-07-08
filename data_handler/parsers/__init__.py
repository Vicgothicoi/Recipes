"""多格式文档解析器。"""

from data_handler.parsers.base import DocumentParser
from data_handler.parsers.html_parser import HtmlParser
from data_handler.parsers.markdown_parser import MarkdownParser
from data_handler.parsers.registry import (
    get_parser,
    parse_directory,
    parse_file,
    supported_suffixes,
)
from data_handler.parsers.txt_parser import TxtParser

try:
    from data_handler.parsers.pdf_parser import PdfParser
except ImportError:
    PdfParser = None  # type: ignore[misc, assignment]

__all__ = [
    "DocumentParser",
    "MarkdownParser",
    "PdfParser",
    "HtmlParser",
    "TxtParser",
    "get_parser",
    "parse_file",
    "parse_directory",
    "supported_suffixes",
]
