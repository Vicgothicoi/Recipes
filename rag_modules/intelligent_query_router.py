"""
智能查询路由器
根据查询特点自动选择最适合的检索策略：
- 传统混合检索：适合简单的信息查找
- 图RAG检索：适合复杂的关系推理和知识发现
"""

import json
import logging
import os
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """搜索策略枚举"""

    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"  # 图RAG检索
    COMBINED = "combined"  # 组合策略


@dataclass
class QueryAnalysis:
    """查询分析结果"""

    query_complexity: float  # 查询复杂度 (0-1)
    relationship_intensity: float  # 关系密集度 (0-1)
    graph_useful: bool  # LLM 判断图检索是否对本 query 有帮助
    recommended_strategy: SearchStrategy


class IntelligentQueryRouter:
    """
    智能查询路由器
    """

    def __init__(
        self,
        traditional_retrieval,
        graph_rag_retrieval,
        llm_client,
        config,
    ):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config

        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0,
        }

        # 懒加载：spaCy NLP 管道（依存句法分析）
        self._nlp = None
        # 懒加载：sentence-transformers 编码器（bge-small-zh-v1.5）
        self._sent_encoder = None
        # 懒加载：图节点实体词典
        self._entity_dict: Optional[set] = None

        # 高价值关系类型集合：懒加载，首次使用时从 .cypher 建表文件解析
        self._high_value_relations: Optional[frozenset] = None
        # .cypher 建表文件路径，用于解析跨文档关系
        self._cypher_path: str = getattr(
            config,
            "cypher_path",
            os.path.join(
                os.path.dirname(__file__), "..", "data", "cypher", "neo4j_import.cypher"
            ),
        )

        # 高关系密集度模板句：仅在图结构直查失败时使用
        self._intensity_templates: List[str] = [
            "这道菜需要哪些食材",
            "有没有和这道菜类似的做法",
            "这道菜属于哪个菜系",
            "用相同烹饪方式的菜还有哪些",
            "用同一种厨具可以做哪些菜",
            "哪些食材经常和这个一起出现在菜谱里",
        ]
        self._intensity_template_embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # 懒加载辅助方法：需要的时候才创建或加载资源                             #
    # ------------------------------------------------------------------ #

    def _get_high_value_relations(self) -> frozenset:
        """
        从 .cypher 建表文件解析高价值关系集合。

        识别标准：跨文档关系在建表语句中的共同特征——
          同一 MERGE 块里用两个不同变量 MATCH 独立节点，
          且 WHERE 子句包含两变量间的共享属性相等条件。

        例如：
          MATCH (n1), (n2) WHERE n1.category = n2.category ...
          MERGE (n1)-[:SIMILAR]->(n2)          ← 跨文档，高价值

          MATCH (source:Recipe ...) MATCH (target:Ingredient ...)
          MERGE (source)-[:REQUIRES]->(target) ← 单文档派生，非高价值
        """
        if self._high_value_relations is not None:
            return self._high_value_relations

        try:
            self._high_value_relations = self._parse_cross_doc_relations(
                self._cypher_path
            )
            logger.info(f"高价值关系集合构建完成: {sorted(self._high_value_relations)}")
        except Exception as e:
            logger.warning(f"高价值关系集合解析失败，降级为空集: {e}")
            self._high_value_relations = frozenset()

        return self._high_value_relations

    def _parse_cross_doc_relations(self, cypher_path: str) -> frozenset:
        """
        解析 .cypher 文件，提取跨文档关系类型。

        跨文档关系的识别模式（在同一逻辑块内同时满足）：
          1. MATCH 语句中出现两个不同的独立变量（非父子结构）
          2. WHERE 子句中存在 var1.prop = var2.prop 的共享属性条件
          3. MERGE 语句中包含 [:<REL_TYPE>] 的关系定义
        """
        import re

        with open(cypher_path, encoding="utf-8") as f:
            content = f.read()

        # 按空行或 RETURN/MATCH 重置点切分成逻辑块
        # 用双换行或独立 RETURN 语句作为块分隔符
        blocks = re.split(r"\n(?=(?:MATCH|RETURN|CREATE|LOAD)\b)", content)

        cross_doc_rels: set = set()

        # 匹配 WHERE 中的跨变量同名属性相等：var1.prop = var2.prop（属性名必须相同）
        # 用反向引用 \2 确保两侧属性名一致
        # 用负向前瞻排除算术表达式（如 s1.stepNumber + 1），确保是纯属性相等
        cross_attr_pattern = re.compile(
            r"WHERE\b.*?(\b\w+)\.(\w+)\s*=\s*(\b\w+)\.\2\b(?!\s*[+\-*/])",
            re.DOTALL | re.IGNORECASE,
        )
        # 匹配 MERGE 中的关系类型：-[:REL_TYPE
        merge_rel_pattern = re.compile(r"MERGE\s*\([^)]*\)-\[.*?:(\w+)", re.IGNORECASE)

        for block in blocks:
            # 必须有跨变量的共享属性条件
            cross_match = cross_attr_pattern.search(block)
            if not cross_match:
                continue
            var1, var2 = cross_match.group(1), cross_match.group(3)
            if var1 == var2:
                continue

            # 必须有 MERGE 关系定义
            merge_match = merge_rel_pattern.search(block)
            if not merge_match:
                continue

            cross_doc_rels.add(merge_match.group(1))

        return frozenset(cross_doc_rels)

    def _get_entity_dict(self) -> set:
        """从 Neo4j 加载所有节点名，构建实体词典（懒加载+缓存）。"""
        if self._entity_dict is not None:
            return self._entity_dict

        driver = getattr(self.graph_rag_retrieval, "driver", None)
        if driver is None:
            self._entity_dict = set()
            return self._entity_dict

        try:
            with driver.session() as session:
                result = session.run(
                    "MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name"
                )
                self._entity_dict = {r["name"] for r in result if r["name"]}
            logger.info(f"实体词典加载完成，共 {len(self._entity_dict)} 个节点名")
        except Exception as e:
            logger.warning(f"实体词典加载失败: {e}")
            self._entity_dict = set()

        return self._entity_dict

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load("zh_core_web_sm")
                logger.info("spaCy zh_core_web_sm 加载成功")
            except Exception as e:
                logger.warning(f"spaCy 加载失败，依存句法将降级为关键词匹配: {e}")
                self._nlp = False
        return self._nlp if self._nlp else None

    def _get_sent_encoder(self):
        if self._sent_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = getattr(
                    self.config, "embedding_model", "BAAI/bge-small-zh-v1.5"
                )
                self._sent_encoder = SentenceTransformer(model_name)
                logger.info(f"SentenceTransformer ({model_name}) 加载成功")
            except Exception as e:
                logger.warning(f"SentenceTransformer 加载失败，隐含关系检测将降级: {e}")
                self._sent_encoder = False
        return self._sent_encoder if self._sent_encoder else None

    def _get_template_embeddings(
        self, templates: List[str], cache_attr: str
    ) -> Optional[np.ndarray]:
        if getattr(self, cache_attr) is None:
            encoder = self._get_sent_encoder()
            if encoder is None:
                return None
            try:
                setattr(
                    self,
                    cache_attr,
                    encoder.encode(
                        templates, normalize_embeddings=True, show_progress_bar=False
                    ),
                )
            except Exception as e:
                logger.warning(f"模板句编码失败 ({cache_attr}): {e}")
                return None
        return getattr(self, cache_attr)

    def _cosine_max_sim(
        self, query: str, templates: List[str], cache_attr: str
    ) -> float:
        """计算 query 与模板句集合的 cosine 相似度最高分。降级返回 0.0。"""
        encoder = self._get_sent_encoder()
        template_embeddings = self._get_template_embeddings(templates, cache_attr)
        if encoder is None or template_embeddings is None:
            return 0.0
        try:
            query_emb = encoder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )
            similarities = (query_emb @ template_embeddings.T).flatten()
            best_idx = int(similarities.argmax())
            max_sim = float(similarities[best_idx])
            logger.debug(
                f"cosine_max_sim ({cache_attr}): {max_sim:.3f} "
                f"(最近模板: {templates[best_idx]})"
            )
            return max_sim
        except Exception as e:
            logger.warning(f"cosine 相似度计算失败: {e}")
            return 0.0

    # ------------------------------------------------------------------ #
    # 查询复杂度：查询长度 × 0.1 + 实体识别数量 × 0.4 + 语法结构 × 0.5       #
    # ------------------------------------------------------------------ #

    def _extract_entity_texts(self, query: str) -> List[str]:
        """在 query 中查找与图节点名匹配的实体（字符串包含匹配）。"""
        entity_dict = self._get_entity_dict()
        if not entity_dict:
            return []
        matched = [name for name in entity_dict if name in query]
        # 去除被更长匹配覆盖的子串（如"鲈鱼"被"清蒸鲈鱼"覆盖）
        matched.sort(key=len, reverse=True)
        filtered, covered = [], ""
        for name in matched:
            if name not in covered:
                filtered.append(name)
                covered += name
        return filtered

    def _score_entity_count(self, query: str) -> Tuple[float, int]:
        matched = self._extract_entity_texts(query)
        entity_count = len(matched)
        logger.debug(f"图节点字典匹配实体: {matched}")
        score = min(entity_count / 4.0, 1.0)
        return score, entity_count

    def _score_syntax_pattern(self, query: str) -> float:
        """
        依存句法分析，从句子树结构判断查询复杂度。
        三个结构信号（可叠加，上限1.0）：
          并列关系 (conj)  0.5  —— "红烧和清蒸哪种更健康" 中 conj(清蒸, 红烧)
          树深度 >= 4      0.4  —— 嵌套修饰/从句，句子结构复杂
          advmod/amod >= 2 0.3  —— 多个状语/定语修饰，多条件约束
        """
        nlp = self._get_nlp()
        if nlp is not None:
            doc = nlp(query)
            score = 0.0

            # 并列关系：对比型查询的核心特征
            conj_count = sum(1 for t in doc if t.dep_ == "conj")
            if conj_count >= 1:
                score += 0.5

            # 树深度：句子嵌套程度
            max_depth = max((len(list(t.ancestors)) for t in doc), default=0)
            if max_depth >= 4:
                score += 0.4

            # 修饰语数量：多条件约束的结构特征
            modifier_count = sum(1 for t in doc if t.dep_ in ("advmod", "amod"))
            if modifier_count >= 2:
                score += 0.3

            result = min(score, 1.0)
            logger.debug(
                f"依存句法 — conj:{conj_count} 树深:{max_depth} 修饰:{modifier_count} → {result:.3f}"
            )
            return result

        return min(kw_score, 0.5)

    def _compute_query_complexity(self, query: str) -> float:

        # 1. 长度得分
        length_score = min(len(query) / 60.0, 1.0)
        # logger.debug(f"{length_score}")

        # 2. 多实体得分
        entity_score, _ = self._score_entity_count(query)
        # logger.debug(f"{entity_score}")

        # 3. 语法结构信号（依存句法：对比/多条件/嵌套）
        syntax_score = self._score_syntax_pattern(query)
        # logger.debug(f"{syntax_score}")

        complexity = 0.1 * length_score + 0.4 * entity_score + 0.5 * syntax_score
        logger.debug(
            f"复杂度打分 — 长度:{length_score:.3f} 实体:{entity_score:.3f} "
            f"语法:{syntax_score:.3f} → 综合:{complexity:.3f}"
        )
        return round(complexity, 4)

    # -------------------------------------------------------------------- #
    # 关系密集度：图结构直查（Entity Linking → Neo4j 一度邻居关系统计）       #
    # -------------------------------------------------------------------- #

    def _query_entity_relations(self, query: str) -> Tuple[float, str]:
        entity_texts = self._extract_entity_texts(query)
        if not entity_texts:
            return -1.0, "no_entity"

        driver = getattr(self.graph_rag_retrieval, "driver", None)
        if driver is None:
            logger.debug("Neo4j driver 不可用，跳过图结构直查")
            return -1.0, "no_driver"

        rel_types: set = set()
        try:
            with driver.session() as session:
                for text in entity_texts:
                    # 查实体节点的所有出边关系类型（一度邻居）
                    result = session.run(
                        "MATCH (n)-[r]->() "
                        "WHERE n.name = $name OR n.preferredTerm = $name "
                        "RETURN DISTINCT type(r) AS rel_type",
                        name=text,
                    )
                    for record in result:
                        rel_types.add(record["rel_type"])
        except Exception as e:
            logger.warning(f"图结构直查 Neo4j 查询失败: {e}")
            return -1.0, "query_error"

        if not rel_types:
            logger.debug(f"实体 {entity_texts} 在图中无出边关系")
            return -1.0, "no_relations"

        high_value_relations = self._get_high_value_relations()
        if not high_value_relations:
            # 高价值集合为空（Neo4j 不可用或图为空），无法评分，触发语义兜底
            return -1.0, "no_high_value_relations"

        # 命中数 / 高价值关系的总数
        high_value_hits = rel_types & high_value_relations
        intensity = round(len(high_value_hits) / len(high_value_relations), 4)

        source = f"graph_direct:{','.join(sorted(high_value_hits)) or 'none'}"
        logger.debug(
            f"图结构直查 — 关系类型:{rel_types} "
            f"高价值命中:{len(high_value_hits)}/{len(rel_types)} → {intensity:.3f}"
        )
        return intensity, source

    def _score_intensity_semantic(self, query: str) -> float:
        return self._cosine_max_sim(
            query, self._intensity_templates, "_intensity_template_embeddings"
        )

    def _compute_relationship_intensity(self, query: str) -> Tuple[float, str]:
        # 第一层：图结构直查
        graph_score, graph_source = self._query_entity_relations(query)

        if graph_score >= 0:
            return graph_score, graph_source

        # 第二层：语义兜底（图结构直查失败时）
        semantic_score = self._score_intensity_semantic(query)
        logger.debug(
            f"关系密集度降级语义兜底 (原因:{graph_source}) → {semantic_score:.3f}"
        )
        return round(semantic_score, 4), f"semantic_fallback({graph_source})"

    # -------------------------------------------------------------------- #
    # LLM判断是否需要推理 + 路由选择逻辑                                      #
    # -------------------------------------------------------------------- #

    def _llm_judge_reasoning(self, query: str) -> bool:
        """
        用 LLM 判断图检索对本 query 是否有实质帮助。
        True  → query 需要跨文档关联，图结构能提供向量检索拿不到的信息。
        False → 单篇文档已足够，或推理靠 LLM 自身知识即可。
        失败时降级为 False。
        """
        prompt = (
            "判断图数据库检索（跨文档关系遍历，如找相似菜、共用食材、相同烹饪方法）"
            "是否对回答以下查询有实质帮助。\n\n"
            "判断标准：\n"
            "- true：query 需要跨菜谱的关联信息，单篇文档无法回答\n"
            "- false：单篇菜谱文档已足够回答，或靠常识/LLM 知识即可\n\n"
            f"查询：{query}\n\n"
            '只返回 JSON，不要任何其他内容：{"graph_useful": true} 或 {"graph_useful": false}'
        )
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
                timeout=10,
            )
            result = json.loads(response.choices[0].message.content.strip())
            return bool(result.get("graph_useful", False))
        except Exception as e:
            logger.warning(f"LLM 图检索判断失败，降级为 False: {e}")
            return False

    def analyze_query(self, query: str) -> QueryAnalysis:
        logger.info(f"分析查询特征: {query}")

        import concurrent.futures

        # 三个步骤互相独立，并行执行
        # 耗时：LLM(500-2000ms) >> Neo4j(10-100ms) > spaCy(5-50ms)
        # 并行后总延迟 ≈ max(三者) ≈ LLM 调用时间
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_complexity = executor.submit(self._compute_query_complexity, query)
            future_intensity = executor.submit(
                self._compute_relationship_intensity, query
            )
            future_graph_useful = executor.submit(self._llm_judge_reasoning, query)

            query_complexity = future_complexity.result()
            relationship_intensity, intensity_source = future_intensity.result()
            graph_useful = future_graph_useful.result()

        # ── 路由决策（完全确定性）─────────────────────────────
        recommended_strategy = self._select_strategy(
            query_complexity, relationship_intensity, graph_useful
        )

        analysis = QueryAnalysis(
            query_complexity=query_complexity,
            relationship_intensity=relationship_intensity,
            graph_useful=graph_useful,
            recommended_strategy=recommended_strategy,
        )

        logger.info(
            f"查询分析完成: {recommended_strategy.value} "
            f"(查询复杂度:{query_complexity:.2f} 关系密集度:{relationship_intensity:.2f} "
            f"图有用:{graph_useful} 来源:{intensity_source})"
        )
        return analysis

    def _select_strategy(
        self,
        query_complexity: float,
        relationship_intensity: float,
        graph_useful: bool,
    ) -> SearchStrategy:
        """
        决策表：
          graph_useful=True  + intensity > 0  → GRAPH_RAG
              LLM 认为需要图，图里也有跨文档边，直接走图
          graph_useful=True  + intensity = 0  → COMBINED
              LLM 认为需要图，但图里没有直接证据，传统检索兜底
          graph_useful=False + complexity >= 0.4 → COMBINED
              LLM 认为不需要图，但 query 复杂，两路都跑
          graph_useful=False + complexity < 0.4  → HYBRID_TRADITIONAL
              LLM 认为不需要图，query 也简单，直接传统检索
        """
        if graph_useful:
            if relationship_intensity > 0:
                return SearchStrategy.GRAPH_RAG
            else:
                return SearchStrategy.COMBINED
        else:
            if query_complexity >= 0.4:
                return SearchStrategy.COMBINED
            return SearchStrategy.HYBRID_TRADITIONAL

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """完全降级分析（LLM 完全不可用时使用，graph_useful 用启发式估算）"""
        query_complexity = self._compute_query_complexity(query)
        relationship_intensity, _ = self._compute_relationship_intensity(query)
        graph_useful = relationship_intensity > 0
        strategy = self._select_strategy(
            query_complexity, relationship_intensity, graph_useful
        )

        return QueryAnalysis(
            query_complexity=query_complexity,
            relationship_intensity=relationship_intensity,
            graph_useful=graph_useful,
            recommended_strategy=strategy,
        )

    def route_query(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[Document], QueryAnalysis]:
        """
        智能路由查询到最适合的检索引擎
        """
        logger.info(f"开始智能路由: {query}")

        # 1. 分析查询特征
        analysis = self.analyze_query(query)

        # 2. 更新统计
        self._update_route_stats(analysis.recommended_strategy)

        # 3. 根据策略执行检索
        documents = []

        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                logger.info("使用传统混合检索")
                documents = self.traditional_retrieval.hybrid_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                logger.info("🕸️ 使用图RAG检索")
                documents = self.graph_rag_retrieval.graph_rag_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                logger.info("🔄 使用组合检索策略")
                documents = self._combined_search(query, top_k)

            # 4. 结果后处理
            documents = self._post_process_results(documents, analysis)

            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis

        except Exception as e:
            logger.error(f"查询路由失败: {e}")
            # 降级到传统检索
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            return documents, analysis

    def _combined_search(
        self, query: str, top_k: int, rrf_k: int = 60
    ) -> List[Document]:
        """
        组合搜索策略：并行执行传统检索和图RAG检索，RRF合并
        """
        import concurrent.futures

        traditional_docs = []
        graph_docs = []

        def traditional_search():
            nonlocal traditional_docs
            try:
                traditional_docs = self.traditional_retrieval.hybrid_search(
                    query, top_k
                )
                logger.info(f"传统检索完成: {len(traditional_docs)} 个结果")
            except Exception as e:
                logger.error(f"传统检索失败: {e}")
                traditional_docs = []

        def graph_search():
            nonlocal graph_docs
            try:
                graph_docs = self.graph_rag_retrieval.graph_rag_search(query, top_k)
                logger.info(f"图RAG检索完成: {len(graph_docs)} 个结果")
            except Exception as e:
                logger.error(f"图RAG检索失败: {e}")
                graph_docs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_traditional = executor.submit(traditional_search)
            future_graph = executor.submit(graph_search)
            concurrent.futures.wait([future_traditional, future_graph], timeout=30)

        # RRF合并：score = sum(1 / (rrf_k + rank)) across retrievers
        rrf_scores: dict = {}
        doc_map: dict = {}

        for rank, doc in enumerate(graph_docs):
            doc_id = hash(doc.page_content[:100])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_map:
                doc.metadata["search_source"] = "graph_rag"
                doc_map[doc_id] = doc

        for rank, doc in enumerate(traditional_docs):
            doc_id = hash(doc.page_content[:100])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_map:
                doc.metadata["search_source"] = "traditional"
                doc_map[doc_id] = doc
            else:
                doc_map[doc_id].metadata["search_source"] = "both"

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        combined_docs = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.metadata["rrf_score"] = rrf_scores[doc_id]
            combined_docs.append(doc)

        logger.info(
            f"RRF合并：从总共{len(graph_docs) + len(traditional_docs)}个结果合并为{len(combined_docs)}个文档"
        )
        return combined_docs

    def _post_process_results(
        self, documents: List[Document], analysis: QueryAnalysis
    ) -> List[Document]:
        """
        结果后处理：根据查询分析优化结果
        """
        for doc in documents:
            # 添加路由信息到元数据
            doc.metadata.update(
                {
                    "route_strategy": analysis.recommended_strategy.value,
                    "query_complexity": analysis.query_complexity,
                }
            )

        return documents

    def _update_route_stats(self, strategy: SearchStrategy):
        """更新路由统计"""
        self.route_stats["total_queries"] += 1

        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1

    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats

        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total,
        }

    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程"""
        analysis = self.analyze_query(query)

        explanation = f"""
        查询路由分析报告

        查询：{query}

        特征分析：
        - 查询复杂度：{analysis.query_complexity:.2f} ({'简单' if analysis.query_complexity < 0.4 else '中等' if analysis.query_complexity < 0.8 else '复杂'})
        - 关系密集度：{analysis.relationship_intensity:.2f} ({'无高价值关系' if analysis.relationship_intensity < 0.2 else '有高价值关系' if analysis.relationship_intensity < 0.4 else '高价值关系丰富'})
        - 图检索有帮助：{'是' if analysis.graph_useful else '否'}

        推荐策略：{analysis.recommended_strategy.value}
        """

        return explanation

    # -------------------------------------------------------------------- #
    # 测试路由准确率，需要人工标签                                            #
    # -------------------------------------------------------------------- #

    def evaluate_routing(self, test_file: Optional[str] = None):

        import ast

        _STRATEGY_MAP = {
            0: SearchStrategy.HYBRID_TRADITIONAL,
            1: SearchStrategy.GRAPH_RAG,
            2: SearchStrategy.COMBINED,
        }

        if test_file is None:
            test_file = os.path.join(
                os.path.dirname(__file__), "..", "data", "query", "route_copy.txt"
            )

        labeled: List[tuple] = []
        try:
            with open(test_file, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if line == "END":
                        break
                    if not line:
                        continue
                    try:
                        row = ast.literal_eval(line)
                        query = str(row[0])
                        expected = _STRATEGY_MAP.get(int(row[2]))
                        if expected is None:
                            logger.warning(f"第 {lineno} 行策略值无效，跳过: {row[2]}")
                            continue
                        labeled.append((query, expected))
                    except Exception as e:
                        logger.warning(f"第 {lineno} 行解析失败，跳过: {e}")
        except FileNotFoundError:
            print(f"测试文件不存在: {test_file}")
            return 0.0

        if not labeled:
            print("测试集为空")
            return 0.0

        results = []
        for q, expected in labeled:
            analysis = self.analyze_query(q)
            actual = analysis.recommended_strategy
            ok = actual == expected
            results.append((q, expected.value, actual.value, ok))
            print(
                f"  {'✓' if ok else '✗'}  [{expected.value:20s}] <- 期望  [{actual.value:20s}] <- 实际  |  {q}\n"
                f"      查询复杂度:{analysis.query_complexity:.3f}  关系密集度:{analysis.relationship_intensity:.3f}  图有用:{'是' if analysis.graph_useful else '否'}\n"
            )

        correct = sum(r[3] for r in results)
        accuracy = correct / len(results)
        print(f"\n准确率: {correct}/{len(results)} = {accuracy:.1%}")
        return accuracy


if __name__ == "__main__":
    import os
    import sys

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # 加载 .env（项目根目录）
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=== evaluate_routing ===\n")

    # ── 初始化 config ──────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import GraphRAGConfig

    config = GraphRAGConfig()

    # ── 初始化 LLM client ─────────────────────────────────────────────
    from openai import OpenAI

    llm_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    # ── 初始化 graph_rag_retrieval 并连接 Neo4j ───────────────────────
    from rag_modules.graph_rag_retrieval import GraphRAGRetrieval

    graph_rag = GraphRAGRetrieval(config=config, llm_client=llm_client)
    try:
        from neo4j import GraphDatabase

        graph_rag.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        graph_rag.driver.verify_connectivity()
        print(f"Neo4j 连接成功: {config.neo4j_uri}\n")
    except Exception as e:
        print(f"[警告] Neo4j 连接失败，关系密集度将降级为语义兜底: {e}\n")
        graph_rag.driver = None

    # ── 构造 router（evaluate_routing 不调 traditional_retrieval）─────
    router = IntelligentQueryRouter(
        traditional_retrieval=None,
        graph_rag_retrieval=graph_rag,
        llm_client=llm_client,
        config=config,
    )

    router.evaluate_routing()
