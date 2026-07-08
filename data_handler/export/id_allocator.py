"""nodeId / relationshipId 分配（对齐 data/cypher 现有编码）。"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

# 菜谱实体节点 ID 起始段（现有数据约 201000001～201005725）
RECIPE_ENTITY_BASE = 201_000_001
# 层次节点上界（Root / RecipeCategory 等），新实体 ID 必须大于该值
HIERARCHY_ID_MAX = 200_000_000

_REL_ID_PATTERN = re.compile(r"R_(\d+)")


class IdAllocator:
    """为单篇菜谱及其子实体顺序分配 nodeId，为关系分配 relationshipId。"""

    def __init__(
        self,
        *,
        start_node_id: Optional[int] = None,
        start_rel_index: Optional[int] = None,
    ) -> None:
        self._next_node_id = start_node_id or RECIPE_ENTITY_BASE
        self._rel_index = start_rel_index or 1

    @classmethod
    def from_existing_csv(
        cls,
        nodes_path: Path | str,
        relationships_path: Path | str,
    ) -> IdAllocator:
        """从已有 CSV 续号，避免覆盖。"""
        max_node = RECIPE_ENTITY_BASE - 1
        nodes_path = Path(nodes_path)
        if nodes_path.is_file():
            with nodes_path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        nid = int(row["nodeId"])
                        if nid > HIERARCHY_ID_MAX:
                            max_node = max(max_node, nid)
                    except (KeyError, ValueError):
                        continue

        max_rel = 0
        rel_path = Path(relationships_path)
        if rel_path.is_file():
            with rel_path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("relationshipId", "")
                    m = _REL_ID_PATTERN.match(rid)
                    if m:
                        max_rel = max(max_rel, int(m.group(1)))

        return cls(start_node_id=max_node + 1, start_rel_index=max_rel + 1)

    def allocate_recipe_id(self) -> int:
        """菜谱根节点 ID（块内第一个）。"""
        rid = self._next_node_id
        self._next_node_id += 1
        return rid

    def allocate_entity_id(self) -> int:
        """食材 / 步骤等子节点 ID。"""
        eid = self._next_node_id
        self._next_node_id += 1
        return eid

    def allocate_relationship_id(self) -> str:
        rid = f"R_{self._rel_index:06d}"
        self._rel_index += 1
        return rid

    @property
    def next_node_id(self) -> int:
        return self._next_node_id
