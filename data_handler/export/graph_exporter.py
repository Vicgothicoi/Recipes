"""将 ParsedDocument + 实体 + 关系导出为 nodes.csv / relationships.csv。"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from data_handler.export.id_allocator import IdAllocator
from data_handler.re.labels import (
    RELATION_CONTAINS_STEP,
    RELATION_DIFFICULTY_LEVEL,
    RELATION_REQUIRES,
)
from data_handler.schemas import EntitySpan, ParsedDocument, RelationTriple

logger = logging.getLogger(__name__)

NODE_COLUMNS = [
    "nodeId",
    "labels",
    "name",
    "preferredTerm",
    "fsn",
    "conceptType",
    "synonyms",
    "category",
    "difficulty",
    "cuisineType",
    "prepTime",
    "cookTime",
    "servings",
    "tags",
    "filePath",
    "amount",
    "unit",
    "isMain",
    "description",
    "stepNumber",
    "methods",
    "tools",
    "timeEstimate",
]

REL_COLUMNS = [
    "startNodeId",
    "endNodeId",
    "relationshipType",
    "relationshipId",
    "amount",
    "unit",
    "step_order",
]

# dishes 子目录 → 图谱 RecipeCategory 中文名
FOLDER_TO_CATEGORY: Dict[str, str] = {
    "aquatic": "水产",
    "breakfast": "早餐",
    "condiment": "调料",
    "dessert": "甜品",
    "drink": "饮料",
    "meat_dish": "荤菜",
    "soup": "汤类",
    "staple": "主食",
    "vegetable_dish": "素菜",
    "semi-finished": "主食",
}

# 难度星级 → 现有 DifficultyLevel 节点 nodeId
STAR_TO_DIFFICULTY_NODE: Dict[int, str] = {
    1: "610000000",
    2: "620000000",
    3: "630000000",
    4: "640000000",
    5: "650000000",
}


@dataclass
class GraphExportResult:
    nodes: List[Dict[str, str]] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def _empty_node_row() -> Dict[str, str]:
    return {col: "" for col in NODE_COLUMNS}


def _parse_star_difficulty(difficulty: Optional[str]) -> Optional[float]:
    if not difficulty:
        return None
    stars = difficulty.count("★") + difficulty.count("☆")
    if stars == 0:
        return None
    return float(min(stars, 5))


def _relative_file_path(source_path: str, project_root: Path) -> str:
    try:
        rel = Path(source_path).resolve().relative_to(project_root.resolve())
        return str(rel).replace("/", "\\")
    except ValueError:
        return source_path


class GraphExporter:
    def __init__(
        self,
        config=None,
        *,
        allocator: Optional[IdAllocator] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        from data_handler.config import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG
        self.project_root = project_root or self.config.project_root
        self.allocator = allocator or IdAllocator.from_existing_csv(
            self.config.cypher_out_dir / "nodes.csv",
            self.config.cypher_out_dir / "relationships.csv",
        )

    def export_document(
        self,
        document: ParsedDocument,
        entities: List[EntitySpan],
        relations: List[RelationTriple],
        *,
        skip_if_exists: bool = True,
        existing_file_paths: Optional[Set[str]] = None,
    ) -> GraphExportResult:
        result = GraphExportResult()
        file_path = _relative_file_path(document.source_path, self.project_root)

        if skip_if_exists and existing_file_paths and file_path in existing_file_paths:
            result.skipped.append(file_path)
            return result

        recipe_name = document.recipe_name or document.title or Path(document.source_path).stem
        recipe_id = self.allocator.allocate_recipe_id()
        category_en = document.metadata.get("category_path", "")
        category_cn = FOLDER_TO_CATEGORY.get(category_en, "")
        difficulty = _parse_star_difficulty(document.metadata.get("difficulty"))

        recipe_row = _empty_node_row()
        recipe_row.update(
            {
                "nodeId": str(recipe_id),
                "labels": "Recipe",
                "name": recipe_name,
                "preferredTerm": recipe_name,
                "conceptType": "Recipe",
                "category": category_cn,
                "difficulty": str(difficulty) if difficulty is not None else "",
                "filePath": file_path,
            }
        )
        result.nodes.append(recipe_row)

        entity_key_to_id: Dict[Tuple[str, str], str] = {}
        entity_key_to_id[("Recipe", recipe_name)] = str(recipe_id)

        for ent in entities:
            if ent.label == "Recipe":
                continue
            key = (ent.label, ent.text.strip())
            if not key[1] or key in entity_key_to_id:
                continue
            nid = str(self.allocator.allocate_entity_id())
            entity_key_to_id[key] = nid
            row = _empty_node_row()
            row.update(
                {
                    "nodeId": nid,
                    "labels": ent.label,
                    "name": ent.text.strip(),
                    "preferredTerm": ent.text.strip(),
                    "conceptType": ent.label,
                }
            )
            if ent.label == "CookingStep":
                row["description"] = ent.text.strip()
            result.nodes.append(row)

        for rel in relations:
            head_key = (rel.head_label, rel.head_text.strip())
            tail_key = (rel.tail_label, rel.tail_text.strip())
            if rel.head_label == "Recipe" and rel.head_text.strip() == recipe_name:
                head_id = str(recipe_id)
            else:
                head_id = entity_key_to_id.get(head_key)
            tail_id = entity_key_to_id.get(tail_key)
            if not head_id or not tail_id:
                continue

            rel_row = {col: "" for col in REL_COLUMNS}
            rel_row.update(
                {
                    "startNodeId": head_id,
                    "endNodeId": tail_id,
                    "relationshipType": rel.relation_type,
                    "relationshipId": self.allocator.allocate_relationship_id(),
                }
            )
            props = rel.properties or {}
            if rel.relation_type == RELATION_REQUIRES:
                if props.get("amount"):
                    rel_row["amount"] = str(props["amount"])
                if props.get("unit"):
                    rel_row["unit"] = str(props["unit"])
                for node in result.nodes:
                    if node["nodeId"] == tail_id and node["labels"] == "Ingredient":
                        if props.get("amount"):
                            node["amount"] = str(props["amount"])
                        if props.get("unit"):
                            node["unit"] = str(props["unit"])
            elif rel.relation_type == RELATION_CONTAINS_STEP:
                order = props.get("step_order")
                if order is not None:
                    rel_row["step_order"] = str(float(order))
                for node in result.nodes:
                    if node["nodeId"] == tail_id and node["labels"] == "CookingStep":
                        if order is not None:
                            node["stepNumber"] = str(int(float(order)))

            result.relationships.append(rel_row)

        star_count = int(difficulty) if difficulty else 0
        if star_count in STAR_TO_DIFFICULTY_NODE:
            result.relationships.append(
                {
                    "startNodeId": str(recipe_id),
                    "endNodeId": STAR_TO_DIFFICULTY_NODE[star_count],
                    "relationshipType": RELATION_DIFFICULTY_LEVEL,
                    "relationshipId": self.allocator.allocate_relationship_id(),
                    "amount": "",
                    "unit": "",
                    "step_order": "",
                }
            )

        return result

    def load_existing_recipe_paths(self) -> Set[str]:
        existing: Set[str] = set()
        nodes_path = self.config.cypher_out_dir / "nodes.csv"
        if not nodes_path.is_file():
            return existing
        with nodes_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("labels") == "Recipe" and row.get("filePath"):
                    existing.add(row["filePath"])
        return existing

    def export_batch(
        self,
        items: List[Tuple[ParsedDocument, List[EntitySpan], List[RelationTriple]]],
        *,
        skip_existing: bool = True,
    ) -> GraphExportResult:
        combined = GraphExportResult()
        existing_paths = self.load_existing_recipe_paths() if skip_existing else set()

        for doc, entities, relations in items:
            part = self.export_document(
                doc,
                entities,
                relations,
                skip_if_exists=skip_existing,
                existing_file_paths=existing_paths,
            )
            if part.skipped:
                combined.skipped.extend(part.skipped)
                continue
            combined.nodes.extend(part.nodes)
            combined.relationships.extend(part.relationships)
            fp = _relative_file_path(doc.source_path, self.project_root)
            existing_paths.add(fp)

        return combined

    def write_csv(
        self,
        result: GraphExportResult,
        *,
        output_dir: Optional[Path] = None,
        merge_existing: bool = True,
    ) -> Tuple[Path, Path]:
        out = Path(output_dir or self.config.cypher_out_dir)
        out.mkdir(parents=True, exist_ok=True)
        nodes_path = out / "nodes.csv"
        rels_path = out / "relationships.csv"

        old_nodes: List[Dict[str, str]] = []
        old_rels: List[Dict[str, str]] = []
        if merge_existing and nodes_path.is_file():
            with nodes_path.open(encoding="utf-8", newline="") as f:
                old_nodes = list(csv.DictReader(f))
        if merge_existing and rels_path.is_file():
            with rels_path.open(encoding="utf-8", newline="") as f:
                old_rels = list(csv.DictReader(f))

        new_node_ids = {r["nodeId"] for r in result.nodes}
        new_rel_ids = {r["relationshipId"] for r in result.relationships}
        nodes_to_write = [r for r in old_nodes if r.get("nodeId") not in new_node_ids]
        nodes_to_write.extend(result.nodes)
        rels_to_write = [r for r in old_rels if r.get("relationshipId") not in new_rel_ids]
        rels_to_write.extend(result.relationships)

        with nodes_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=NODE_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(nodes_to_write)

        with rels_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REL_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rels_to_write)

        logger.info(
            "已写入 %s (%d 节点), %s (%d 关系), 跳过 %d 篇",
            nodes_path,
            len(nodes_to_write),
            rels_path,
            len(rels_to_write),
            len(result.skipped),
        )
        return nodes_path, rels_path
