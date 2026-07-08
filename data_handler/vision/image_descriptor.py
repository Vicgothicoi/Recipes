"""批量图像描述，并写回 ParsedDocument。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from data_handler.config import DataHandlerConfig, DEFAULT_CONFIG
from data_handler.schemas import ParsedDocument, TextBlock
from data_handler.vision.client import VisionClient
from data_handler.vision.image_locator import (
    collect_images,
    group_by_block,
    unique_image_paths,
)

logger = logging.getLogger(__name__)

_CAPTION_META_KEY = "image_captions"
_DOC_CAPTIONS_KEY = "image_captions"


class ImageDescriptor:
    """为文档中的图片生成描述，支持磁盘缓存。"""

    def __init__(
        self,
        client: Optional[VisionClient] = None,
        config: Optional[DataHandlerConfig] = None,
        *,
        use_cache: bool = True,
        skip_remote: bool = False,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.client = client or VisionClient(
            model=self.config.vision_model,
            max_tokens=self.config.vision_max_tokens,
            temperature=self.config.vision_temperature,
        )
        self.use_cache = use_cache
        self.skip_remote = skip_remote
        self.cache_dir = self.config.caption_cache_dir or (
            self.config.intermediate_dir / "image_captions"
        )
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def describe_document(
        self,
        document: ParsedDocument,
        *,
        recipe_context: Optional[str] = None,
    ) -> ParsedDocument:
        """
        为文档内所有可访问的本地图片生成描述，写入各块的 metadata.image_captions。
        同时在 document.metadata.image_captions 中保存全局映射。
        """
        located = collect_images(document)
        paths = unique_image_paths(
            located,
            local_only=self.skip_remote,
            must_exist=True,
        )
        if self.skip_remote:
            paths = [p for p in paths if not p.startswith(("http://", "https://"))]

        if not paths:
            logger.debug("文档无待描述图片: %s", document.source_path)
            return document

        context = recipe_context or self._build_context(document)
        captions = self._describe_paths(paths, context=context)

        block_groups = group_by_block(located)
        for block in document.blocks:
            refs = block_groups.get(block.block_index, [])
            if not refs:
                continue
            block_caps = {
                ref: captions[ref]
                for ref in refs
                if ref in captions
            }
            if block_caps:
                block.metadata[_CAPTION_META_KEY] = block_caps

        doc_caps = dict(document.metadata.get(_DOC_CAPTIONS_KEY, {}))
        doc_caps.update(captions)
        document.metadata[_DOC_CAPTIONS_KEY] = doc_caps

        self._append_caption_blocks(document, captions, block_groups)
        return document

    def describe_paths(
        self,
        paths: List[str],
        *,
        context: Optional[str] = None,
    ) -> Dict[str, str]:
        """仅描述给定路径列表，返回 path -> caption。"""
        return self._describe_paths(paths, context=context or "")

    def _describe_paths(
        self,
        paths: List[str],
        *,
        context: str,
    ) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for path in paths:
            cached = self._load_cache(path) if self.use_cache else None
            if cached is not None:
                result[path] = cached
                logger.debug("缓存命中: %s", path)
                continue
            try:
                caption = self.client.describe_image(path, context=context)
                result[path] = caption
                if self.use_cache:
                    self._save_cache(path, caption)
            except Exception as e:
                logger.warning("图像描述失败 %s: %s", path, e)
        return result

    def _append_caption_blocks(
        self,
        document: ParsedDocument,
        captions: Dict[str, str],
        block_groups: Dict[int, List[str]],
    ) -> None:
        """为含图块追加纯描述块，便于后续 NER 使用。"""
        max_index = max((b.block_index for b in document.blocks), default=-1)
        next_index = max_index + 1

        for block in document.blocks:
            refs = block_groups.get(block.block_index, [])
            if not refs:
                continue
            parts = []
            for ref in refs:
                cap = captions.get(ref)
                if cap:
                    name = Path(ref).name if not ref.startswith("http") else ref
                    parts.append(f"[图片 {name}] {cap}")
            if not parts:
                continue
            document.blocks.append(
                TextBlock(
                    text="\n".join(parts),
                    section=block.section,
                    block_index=next_index,
                    image_refs=list(refs),
                    metadata={
                        "derived_from_block": block.block_index,
                        "type": "image_caption",
                        _CAPTION_META_KEY: {
                            r: captions[r] for r in refs if r in captions
                        },
                    },
                )
            )
            next_index += 1

    @staticmethod
    def _build_context(document: ParsedDocument) -> str:
        parts = []
        if document.recipe_name:
            parts.append(f"菜名：{document.recipe_name}")
        if document.title:
            parts.append(f"标题：{document.title}")
        intro = [b.text for b in document.blocks if b.section == "intro" and b.text]
        if intro:
            parts.append(intro[0][:300])
        return "\n".join(parts)

    def _cache_key(self, path: str) -> str:
        p = Path(path)
        if p.is_file():
            stat = p.stat()
            raw = f"{p.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        else:
            raw = path
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _cache_path(self, image_path: str) -> Path:
        return self.cache_dir / f"{self._cache_key(image_path)}.json"

    def _load_cache(self, image_path: str) -> Optional[str]:
        cache_file = self._cache_path(image_path)
        if not cache_file.is_file():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if data.get("image_path") != image_path:
                return None
            return data.get("caption")
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, image_path: str, caption: str) -> None:
        cache_file = self._cache_path(image_path)
        payload = {
            "image_path": image_path,
            "caption": caption,
            "model": self.client.model,
        }
        cache_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
