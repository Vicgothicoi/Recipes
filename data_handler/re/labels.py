"""关系类型（对齐 data/cypher/relationships.csv 与 neo4j_import.cypher）。"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# CSV 中的 relationshipType 编码 → Neo4j 关系名
RELATION_REQUIRES = "801000001"
RELATION_CONTAINS_STEP = "801000003"
RELATION_BELONGS_TO = "801000004"
RELATION_DIFFICULTY_LEVEL = "801000005"
RELATION_NONE = "NONE"

RELATION_NAMES: Dict[str, str] = {
    RELATION_NONE: "NO_RELATION",
    RELATION_REQUIRES: "REQUIRES",
    RELATION_CONTAINS_STEP: "CONTAINS_STEP",
    RELATION_BELONGS_TO: "BELONGS_TO",
    RELATION_DIFFICULTY_LEVEL: "DIFFICULTY_LEVEL",
}

# 训练用关系类型（含负类 NONE）
RELATION_TYPES: Tuple[str, ...] = (
    RELATION_NONE,
    RELATION_REQUIRES,
    RELATION_CONTAINS_STEP,
    RELATION_BELONGS_TO,
    RELATION_DIFFICULTY_LEVEL,
)

LABEL2ID: Dict[str, int] = {r: i for i, r in enumerate(RELATION_TYPES)}
ID2LABEL: Dict[int, str] = {i: r for r, i in LABEL2ID.items()}
NUM_RELATIONS = len(RELATION_TYPES)

# (head_label, tail_label) → 允许的关系类型（不含 NONE 时用于候选过滤）
ALLOWED_HEAD_TAIL: Dict[Tuple[str, str], Set[str]] = {
    ("Recipe", "Ingredient"): {RELATION_REQUIRES, RELATION_NONE},
    ("Recipe", "CookingStep"): {RELATION_CONTAINS_STEP, RELATION_NONE},
    ("Recipe", "RecipeCategory"): {RELATION_BELONGS_TO, RELATION_NONE},
    ("Recipe", "DifficultyLevel"): {RELATION_DIFFICULTY_LEVEL, RELATION_NONE},
}


def relation_name(relation_type: str) -> str:
    return RELATION_NAMES.get(relation_type, relation_type)


def is_allowed_pair(head_label: str, tail_label: str, relation_type: str) -> bool:
    allowed = ALLOWED_HEAD_TAIL.get((head_label, tail_label))
    if allowed is None:
        return relation_type == RELATION_NONE
    return relation_type in allowed


def candidate_relations(head_label: str, tail_label: str) -> List[str]:
    """推理时该实体对可能的关系（不含 NONE）。"""
    allowed = ALLOWED_HEAD_TAIL.get((head_label, tail_label), set())
    return [r for r in allowed if r != RELATION_NONE]
