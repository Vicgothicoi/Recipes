"""从 ParsedDocument 中收集待描述的图像。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

from data_handler.schemas import ParsedDocument


@dataclass(frozen=True)
class LocatedImage:
    """文档内一处图像引用。"""

    path: str
    block_index: int
    source_path: str

    @property
    def is_remote(self) -> bool:
        return self.path.startswith(("http://", "https://"))


def collect_images(document: ParsedDocument) -> List[LocatedImage]:
    """收集文档中所有块上的 image_refs（保留块索引）。"""
    located: List[LocatedImage] = []
    for block in document.blocks:
        for ref in block.image_refs:
            ref = ref.strip()
            if not ref:
                continue
            located.append(
                LocatedImage(
                    path=ref,
                    block_index=block.block_index,
                    source_path=document.source_path,
                )
            )
    return located


def unique_image_paths(
    images: Iterable[LocatedImage],
    *,
    local_only: bool = False,
    must_exist: bool = True,
) -> List[str]:
    """去重后的图像路径列表（保持首次出现顺序）。"""
    seen: Set[str] = set()
    result: List[str] = []
    for img in images:
        path = img.path
        if local_only and img.is_remote:
            continue
        if must_exist and not img.is_remote and not Path(path).is_file():
            continue
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def group_by_block(images: Iterable[LocatedImage]) -> Dict[int, List[str]]:
    """按 block_index 分组图像路径。"""
    groups: Dict[int, List[str]] = {}
    for img in images:
        groups.setdefault(img.block_index, [])
        if img.path not in groups[img.block_index]:
            groups[img.block_index].append(img.path)
    return groups
