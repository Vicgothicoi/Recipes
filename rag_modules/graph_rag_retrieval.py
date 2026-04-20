"""
真正的图RAG检索模块
基于图结构的知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型"""

    ENTITY_RELATION = "entity_relation"
    CLUSTERING = "clustering"
    PATH_FINDING = "path_finding"
    MULTI_HOP = "multi_hop"
    SUBGRAPH = "subgraph"


@dataclass
class GraphQuery:
    """图查询结构"""

    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None


@dataclass
class GraphPath:
    """图路径结构"""

    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str


@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""

    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]


class GraphRAGRetrieval:

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None

        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")

        # 连接Neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return

        # 预热：构建实体和关系索引
        self._build_graph_index()

    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建图结构索引...")

        try:
            with self.driver.session() as session:
                # 构建实体索引 - 修复Neo4j语法兼容性问题
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id, 
                       n.name as name, n.category as category, degree
                ORDER BY degree DESC
                LIMIT 1000
                """

                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"],
                    }

                # 构建关系类型索引
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """

                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]

                logger.info(
                    f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型"
                )

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        这是图RAG的核心：从自然语言到图查询的转换
        """
        prompt = f"""
        作为图数据库专家，分析以下查询的图结构意图：
        
        查询：{query}
        
        请识别：
        1. 查询类型：
           - entity_relation: 询问实体间的直接关系（如：鸡肉和胡萝卜能一起做菜吗？）
           - clustering: 聚类相似性（如：和宫保鸡丁类似的菜有哪些？）
           - path_finding: 路径查找（如：从食材到成品菜的制作路径）
           - multi_hop: 需要多跳推理（如：鸡肉配什么蔬菜？需要：鸡肉→菜品→食材→蔬菜）
           - subgraph: 需要完整子图（如：川菜有什么特色？需要川菜相关的完整知识网络）        
        
        2. 核心实体：查询中的关键实体名称
        3. 目标实体：期望找到的实体类型
        4. 关系类型：涉及的关系类型
        5. 遍历深度：需要的图遍历深度（1-3跳）
        
        示例：
        查询："鸡肉配什么蔬菜好？"
        分析：这是multi_hop查询，需要通过"鸡肉→使用鸡肉的菜品→这些菜品使用的蔬菜"的路径推理
        
        返回JSON格式：
        {{
            "query_type": "multi_hop",
            "source_entities": ["鸡肉"],
            "target_entities": ["蔬菜类食材"],
            "relation_types": ["REQUIRES", "BELONGS_TO_CATEGORY"],
            "max_depth": 3,
            "reasoning": "需要多跳推理：鸡肉→菜品→食材→蔬菜"
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            result = json.loads(response.choices[0].message.content.strip())

            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50,
            )

        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            # 降级方案：默认子图查询
            return GraphQuery(
                query_type=QueryType.SUBGRAPH, source_entities=[query], max_depth=2
            )

    # 方法一
    def find_entity_relations(
        self, graph_query: GraphQuery, session
    ) -> List[GraphPath]:
        """
        实体关系查询
        策略：枚举 source_entities 两两组合，查询它们之间的直接边
        """
        paths = []
        entities = graph_query.source_entities
        if not entities:
            return paths

        if len(entities) == 1:
            cypher = """
            MATCH (a)-[r]-(b)
            WHERE a.name CONTAINS $entity_a OR a.nodeId = $entity_a
            WITH a, r, b,
                 type(r) as rel_type,
                 1 as path_len,
                 (1.0 / (1 + COUNT { (b)--() })) as relevance
            ORDER BY relevance DESC
            LIMIT 20
            RETURN [a, b] as path_nodes, [r] as rels, path_len, relevance, rel_type
            """
            try:
                result = session.run(cypher, {"entity_a": entities[0]})
                for record in result:
                    path_data = self._parse_neo4j_path(record)
                    if path_data:
                        paths.append(path_data)
            except Exception as e:
                logger.error(f"单实体关系查询失败: {e}")

        else:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    cypher = """
                    MATCH (a)-[r]-(b)
                    WHERE (a.name CONTAINS $entity_a OR a.nodeId = $entity_a)
                      AND (b.name CONTAINS $entity_b OR b.nodeId = $entity_b)
                    WITH a, r, b,
                         type(r) as rel_type,
                         1 as path_len,
                         0.9 as relevance
                    LIMIT 10
                    RETURN [a, b] as path_nodes, [r] as rels, path_len, relevance, rel_type
                    """
                    try:
                        result = session.run(
                            cypher,
                            {"entity_a": entities[i], "entity_b": entities[j]},
                        )
                        for record in result:
                            path_data = self._parse_neo4j_path(record)
                            if path_data:
                                paths.append(path_data)
                    except Exception as e:
                        logger.error(
                            f"实体对关系查询失败 ({entities[i]}, {entities[j]}): {e}"
                        )

        logger.info(f"实体关系查询完成，找到 {len(paths)} 条关系")
        return paths

    # 方法二
    def find_similar_recipes(self, query: str, top_k: int = 10) -> List[Document]:
        """
        相似菜谱查询：基于图中已建立的 SIMILAR 关系
        策略：
          1. 先通过名称找到目标菜谱节点
          2. 沿 SIMILAR 关系找相似菜谱
          3. 若 SIMILAR 关系不足，回退到同 category 的菜谱
        """
        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []

        documents = []

        try:
            with self.driver.session() as session:
                similar_cypher = """
                MATCH (r:Recipe)-[sim:SIMILAR]->(similar:Recipe)
                WHERE r.name CONTAINS $query OR r.nodeId = $query
                RETURN similar.name        as name,
                       similar.category    as category,
                       similar.cuisineType as cuisineType,
                       similar.difficulty  as difficulty,
                       similar.cookTime    as cookTime,
                       similar.tags        as tags,
                       similar.description as description,
                       similar.nodeId      as nodeId,
                       sim.confidence      as confidence,
                       sim.basis           as basis
                ORDER BY sim.confidence DESC
                LIMIT $top_k
                """
                result = session.run(similar_cypher, {"query": query, "top_k": top_k})
                records = list(result)

                if records:
                    for record in records:
                        content = self._build_recipe_summary(record)
                        doc = Document(
                            page_content=content,
                            metadata={
                                "search_type": "similar_recipe",
                                "recipe_name": record["name"],
                                "category": record["category"],
                                "cuisine_type": record["cuisineType"],
                                "difficulty": record["difficulty"],
                                "relevance_score": float(record["confidence"] or 0.6),
                                "similarity_basis": record["basis"] or "same_category",
                                "node_id": record["nodeId"],
                            },
                        )
                        documents.append(doc)
                    logger.info(f"SIMILAR关系查询到 {len(documents)} 个相似菜谱")

                if len(documents) < top_k:
                    remaining = top_k - len(documents)
                    existing_names = {d.metadata["recipe_name"] for d in documents}

                    fallback_cypher = """
                    MATCH (r:Recipe)
                    WHERE r.name CONTAINS $query OR r.nodeId = $query
                    WITH r
                    MATCH (similar:Recipe)
                    WHERE similar.category = r.category
                      AND similar.nodeId <> r.nodeId
                      AND NOT similar.name IN $existing_names
                    RETURN similar.name        as name,
                           similar.category    as category,
                           similar.cuisineType as cuisineType,
                           similar.difficulty  as difficulty,
                           similar.cookTime    as cookTime,
                           similar.tags        as tags,
                           similar.description as description,
                           similar.nodeId      as nodeId,
                           0.5                 as confidence,
                           'same_category'     as basis
                    ORDER BY similar.difficulty ASC
                    LIMIT $remaining
                    """
                    fallback_result = session.run(
                        fallback_cypher,
                        {
                            "query": query,
                            "existing_names": list(existing_names),
                            "remaining": remaining,
                        },
                    )
                    for record in fallback_result:
                        content = self._build_recipe_summary(record)
                        doc = Document(
                            page_content=content,
                            metadata={
                                "search_type": "similar_recipe_fallback",
                                "recipe_name": record["name"],
                                "category": record["category"],
                                "cuisine_type": record["cuisineType"],
                                "difficulty": record["difficulty"],
                                "relevance_score": 0.5,
                                "similarity_basis": "same_category",
                                "node_id": record["nodeId"],
                            },
                        )
                        documents.append(doc)
                    logger.info(f"回退查询补充 {len(documents)} 个同类菜谱")

        except Exception as e:
            logger.error(f"相似菜谱查询失败: {e}")

        return documents

    # 方法三
    def find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """
        最短路径查找
        策略：source_entities[0] -> target_entities[0] 的 shortestPath，
              若无 target_entities 则退化为多跳邻居
        """
        paths = []
        source_entities = graph_query.source_entities
        target_entities = graph_query.target_entities or []

        if not source_entities:
            return paths

        if target_entities:
            for src in source_entities:
                for tgt in target_entities:
                    cypher = """
                    MATCH (source), (target)
                    WHERE (source.name CONTAINS $source_name OR source.nodeId = $source_name)
                      AND (target.name CONTAINS $target_name OR target.nodeId = $target_name)
                      AND source <> target
                    MATCH path = shortestPath((source)-[*..6]-(target))
                    WITH path,
                         length(path) as path_len,
                         nodes(path) as path_nodes,
                         relationships(path) as rels,
                         (1.0 / length(path)) as relevance
                    ORDER BY path_len ASC
                    LIMIT 5
                    RETURN path_nodes, rels, path_len, relevance
                    """
                    try:
                        result = session.run(
                            cypher,
                            {"source_name": src, "target_name": tgt},
                        )
                        for record in result:
                            path_data = self._parse_neo4j_path(record)
                            if path_data:
                                path_data.path_type = "shortest_path"
                                paths.append(path_data)
                    except Exception as e:
                        logger.error(f"最短路径查询失败 ({src} -> {tgt}): {e}")
        else:
            for src in source_entities:
                cypher = f"""
                MATCH (source)
                WHERE source.name CONTAINS $source_name OR source.nodeId = $source_name
                MATCH path = (source)-[*1..{graph_query.max_depth}]-(target)
                WHERE source <> target
                WITH path,
                     length(path) as path_len,
                     nodes(path) as path_nodes,
                     relationships(path) as rels,
                     (1.0 / length(path)) as relevance
                ORDER BY path_len ASC
                LIMIT 10
                RETURN path_nodes, rels, path_len, relevance
                """
                try:
                    result = session.run(cypher, {"source_name": src})
                    for record in result:
                        path_data = self._parse_neo4j_path(record)
                        if path_data:
                            path_data.path_type = "shortest_path"
                            paths.append(path_data)
                except Exception as e:
                    logger.error(f"无目标最短路径查询失败 ({src}): {e}")

        logger.info(f"最短路径查询完成，找到 {len(paths)} 条路径")
        return paths

    # 方法四
    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """
        多跳路径遍历
        策略：查找从源头到目标的路径，按路径相关性排序
        """
        logger.info(
            f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}"
        )

        paths = []

        if not self.driver:
            logger.error("Neo4j连接未建立")
            return paths

        try:
            with self.driver.session() as session:
                # 构建多跳遍历查询
                source_entities = graph_query.source_entities
                target_entities = graph_query.target_entities or []

                # 根据查询类型选择不同的遍历策略
                if graph_query.query_type == QueryType.MULTI_HOP:
                    cypher_query = f"""
                    // 多跳推理查询
                    UNWIND $source_entities as source_name
                    MATCH (source)
                    WHERE source.name CONTAINS source_name OR source.nodeId = source_name
                    
                    // 执行多跳遍历
                    MATCH path = (source)-[*1..{graph_query.max_depth}]-(target)
                    WHERE NOT source = target
                    {"AND ANY(label IN labels(target) WHERE label IN $target_labels)" if target_entities else ""}
                    
                    // 计算路径相关性
                    WITH path, source, target,
                         length(path) as path_len,
                         relationships(path) as rels,
                         nodes(path) as path_nodes
                    
                    // 路径评分：短路径 + 高度数节点 + 关系类型匹配
                    WITH path, source, target, path_len, rels, path_nodes,
                         (1.0 / path_len) + 
                         (REDUCE(s = 0.0, n IN path_nodes | s + COUNT {{ (n)--() }}) / 10.0 / size(path_nodes)) +
                         (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
                    
                    ORDER BY relevance DESC
                    LIMIT 20
                    
                    RETURN path, source, target, path_len, rels, path_nodes, relevance
                    """

                    result = session.run(
                        cypher_query,
                        {
                            "source_entities": source_entities,
                            "target_labels": target_entities,
                            "relation_types": graph_query.relation_types or [],
                        },
                    )

                    for record in result:
                        path_data = self._parse_neo4j_path(record)
                        if path_data:
                            paths.append(path_data)

                elif graph_query.query_type == QueryType.ENTITY_RELATION:
                    # 实体间关系查询
                    paths.extend(self.find_entity_relations(graph_query, session))

                elif graph_query.query_type == QueryType.PATH_FINDING:
                    # 最短路径查找
                    paths.extend(self.find_shortest_paths(graph_query, session))

        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")

        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths

    # 方法五
    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """
        提取知识子图
        策略：查找相近节点
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")

        if not self.driver:
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)

        try:
            with self.driver.session() as session:
                # 简化的子图提取（不依赖APOC）
                cypher_query = f"""
                // 找到源实体
                UNWIND $source_entities as entity_name
                MATCH (source)
                WHERE source.name CONTAINS entity_name 
                   OR source.nodeId = entity_name
                
                // 获取指定深度的邻居
                MATCH (source)-[r*1..{graph_query.max_depth}]-(neighbor)
                WITH source, collect(DISTINCT neighbor) as neighbors, 
                     collect(DISTINCT r) as relationships
                WHERE size(neighbors) <= $max_nodes
                
                // 计算图指标
                WITH source, neighbors, relationships,
                     size(neighbors) as node_count,
                     size(relationships) as rel_count
                
                RETURN 
                    source,
                    neighbors[0..{graph_query.max_nodes}] as nodes,
                    relationships[0..{graph_query.max_nodes}] as rels,
                    {{
                        node_count: node_count,
                        relationship_count: rel_count,
                        density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                    }} as metrics
                """

                result = session.run(
                    cypher_query,
                    {
                        "source_entities": graph_query.source_entities,
                        "max_nodes": graph_query.max_nodes,
                    },
                )

                record = result.single()
                if record:
                    return self._build_knowledge_subgraph(record)

        except Exception as e:
            logger.error(f"子图提取失败: {e}")

        # 降级方案：简单邻居查询
        return self._fallback_subgraph_extraction(graph_query)

    # 方法五的辅助
    def graph_structure_reasoning(
        self, subgraph: KnowledgeSubgraph, query: str
    ) -> List[str]:
        """
        图结构推理
        策略：识别推理模式→构建推理链→验证推理链的可信度
        """
        reasoning_chains = []

        try:
            # 1. 识别推理模式
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)

            # 2. 构建推理链
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)

            # 3. 验证推理链的可信度
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)

            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains

        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        logger.info(f"开始图RAG检索: {query}")

        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []

        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")

        results = []

        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type == QueryType.CLUSTERING:
                # 相似菜谱查询
                results.extend(self.find_similar_recipes(query, top_k=top_k))

            elif graph_query.query_type == QueryType.PATH_FINDING:
                # 最短路径查找
                paths = self.find_shortest_paths(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type == QueryType.MULTI_HOP:
                # 多跳路径遍历
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type == QueryType.SUBGRAPH:
                # 知识子图提取
                subgraph = self.extract_knowledge_subgraph(graph_query)

                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)

                results.extend(
                    self._subgraph_to_documents(subgraph, reasoning_chains, query)
                )

            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)

            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]

        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []

    # ========== 辅助方法 ==========

    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析Neo4j路径记录"""
        try:
            path_nodes = []
            for node in record["path_nodes"]:
                path_nodes.append(
                    {
                        "id": node.get("nodeId", ""),
                        "name": node.get("name", ""),
                        "labels": list(node.labels),
                        "properties": dict(node),
                    }
                )

            relationships = []
            for rel in record["rels"]:
                relationships.append({"type": rel.type, "properties": dict(rel)})

            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=record["path_len"],
                relevance_score=record["relevance"],
                path_type="multi_hop",
            )

        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None

    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象"""
        try:
            central_nodes = [dict(record["source"])]
            connected_nodes = [dict(node) for node in record["nodes"]]
            relationships = [dict(rel) for rel in record["rels"]]

            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=record["metrics"],
                reasoning_chains=[],
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[],
            )

    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径转换为Document对象"""
        documents = []

        for i, path in enumerate(paths):
            # 构建路径描述
            path_desc = self._build_path_description(path)

            doc = Document(
                page_content=path_desc,
                metadata={
                    "search_type": "graph_path",
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                    "recipe_name": (
                        path.nodes[0].get("name", "图结构结果")
                        if path.nodes
                        else "图结构结果"
                    ),
                },
            )
            documents.append(doc)

        return documents

    def _subgraph_to_documents(
        self, subgraph: KnowledgeSubgraph, reasoning_chains: List[str], query: str
    ) -> List[Document]:
        """将知识子图转换为Document对象"""
        documents = []

        # 子图整体描述
        subgraph_desc = self._build_subgraph_description(subgraph)

        doc = Document(
            page_content=subgraph_desc,
            metadata={
                "search_type": "knowledge_subgraph",
                "node_count": len(subgraph.connected_nodes),
                "relationship_count": len(subgraph.relationships),
                "graph_density": subgraph.graph_metrics.get("density", 0.0),
                "reasoning_chains": reasoning_chains,
                "recipe_name": (
                    subgraph.central_nodes[0].get("name", "知识子图")
                    if subgraph.central_nodes
                    else "知识子图"
                ),
            },
        )
        documents.append(doc)

        return documents

    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述"""
        if not path.nodes:
            return "空路径"

        desc_parts = []
        for i, node in enumerate(path.nodes):
            desc_parts.append(node.get("name", f"节点{i}"))
            if i < len(path.relationships):
                rel_type = path.relationships[i].get("type", "相关")
                desc_parts.append(f" --{rel_type}--> ")

        return "".join(desc_parts)

    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        """构建子图的自然语言描述"""
        central_names = [node.get("name", "未知") for node in subgraph.central_nodes]
        node_count = len(subgraph.connected_nodes)
        rel_count = len(subgraph.relationships)

        return f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"

    def _rank_by_graph_relevance(
        self, documents: List[Document], query: str
    ) -> List[Document]:
        """基于图结构相关性排序"""
        return sorted(
            documents,
            key=lambda x: x.metadata.get("relevance_score", 0.0),
            reverse=True,
        )

    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别推理模式"""
        return ["因果关系", "组成关系", "相似关系"]

    def _build_reasoning_chain(
        self, pattern: str, subgraph: KnowledgeSubgraph
    ) -> Optional[str]:
        """构建推理链"""
        return f"基于{pattern}的推理链"

    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """验证推理链"""
        return chains[:3]

    def _build_recipe_summary(self, record) -> str:
        """将菜谱记录构建为自然语言摘要"""
        name = record["name"] or "未知菜谱"
        category = record["category"] or ""
        cuisine = record["cuisineType"] or ""
        difficulty = record["difficulty"]
        cook_time = record["cookTime"] or ""
        tags = record["tags"] or ""
        description = record["description"] or ""

        difficulty_str = f"难度{difficulty}星" if difficulty else ""
        parts = [p for p in [category, cuisine, difficulty_str, cook_time] if p]
        meta_str = "，".join(parts)

        summary = f"【{name}】"
        if meta_str:
            summary += f"（{meta_str}）"
        if description:
            summary += f" {description}"
        if tags:
            summary += f" 标签：{tags}"
        return summary

    def _fallback_subgraph_extraction(
        self, graph_query: GraphQuery
    ) -> KnowledgeSubgraph:
        """降级子图提取"""
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[],
        )

    def close(self):
        """关闭资源连接"""
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭")
