"""
混合检索模块
基于双层检索范式：实体级 + 主题级检索
结合图结构检索和向量检索，使用RRF融合结果
"""

import os

# 必须在 sentence_transformers import 之前设置，阻止联网检查
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果数据结构"""

    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'entity' or 'topic'
    metadata: Dict[str, Any]


class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级）
    2. 向量检索
    3. RRF融合策略
    由于丢弃了图索引精确词索引，图数据库召回兜底几乎不可能生效，也一并删除了
    """

    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.bm25_retriever = None
        self.reranker: CrossEncoder = None

    def initialize(self, chunks: List[Document]):
        """初始化检索系统"""
        logger.info("初始化混合检索模块...")

        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )

        # 构建父文档索引：node_id -> 完整菜谱文档（用于 Parent Document Retrieval）
        self.parent_doc_map: Dict[str, Document] = {}
        for doc in self.data_module.documents or []:
            nid = doc.metadata.get("node_id")
            if nid:
                self.parent_doc_map[nid] = doc
        logger.info(f"父文档索引构建完成，共 {len(self.parent_doc_map)} 个菜谱")

        # 初始化BM25检索器
        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(chunks)}")

        # 初始化 reranker
        try:
            self.reranker = CrossEncoder(self.config.rerank_model)
            logger.info(f"Reranker初始化完成: {self.config.rerank_model}")
        except Exception as e:
            logger.warning(f"Reranker初始化失败，将跳过rerank: {e}")
            self.reranker = None

    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """提取查询关键词：实体级 + 主题级"""
        prompt = f"""
        作为烹饪知识助手，请分析以下查询并提取关键词，分为两个层次：

        查询：{query}

        提取规则：
        1. 实体级关键词：具体的食材、菜品名称、工具、品牌等有形实体
           - 例如：鸡胸肉、西兰花、红烧肉、平底锅、老干妈
           - 对于抽象查询，推测相关的具体食材/菜品

        2. 主题级关键词：抽象概念、烹饪主题、饮食风格、营养特点等
           - 例如：减肥、低热量、川菜、素食、下饭菜、快手菜
           - 排除动作词：推荐、介绍、制作、怎么做等

        示例：
        查询："推荐几个减肥菜" 
        {{
            "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
            "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
        }}

        查询："川菜有什么特色"
        {{
            "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
            "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
        }}

        请严格按照JSON格式返回，不要包含多余的文字：
        {{
            "entity_keywords": ["实体1", "实体2", ...],
            "topic_keywords": ["主题1", "主题2", ...]
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            result = json.loads(response.choices[0].message.content.strip())
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])

            logger.info(
                f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}"
            )
            return entity_keywords, topic_keywords

        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            keywords = query.split()
            return keywords[:3], keywords[3:6] if len(keywords) > 3 else keywords

    def _resolve_parent(self, doc: Document) -> Document:
        """
        chunk命中后回溯完整父文档。
        """
        parent_id = doc.metadata.get("parent_id") or doc.metadata.get("node_id")
        parent = self.parent_doc_map.get(parent_id) if parent_id else None
        if parent:
            # 保留chunk的检索元数据，内容替换为完整父文档
            merged_metadata = {
                **parent.metadata,
                "chunk_id": doc.metadata.get("chunk_id"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "retrieved_from_chunk": True,
            }
            return Document(page_content=parent.page_content, metadata=merged_metadata)
        return doc

    def _bm25_search(self, keywords: List[str], top_k: int) -> List[Document]:
        """用BM25对关键词列表做检索，合并去重后返回top_k文档（自动回溯父文档）"""
        if not self.bm25_retriever:
            return []

        seen_ids = set()
        docs = []
        for keyword in keywords:
            self.bm25_retriever.k = top_k
            results = self.bm25_retriever.invoke(keyword)
            for doc in results:
                doc = self._resolve_parent(doc)
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs.append(doc)
        return docs[:top_k]

    def entity_level_retrieval(
        self, entity_keywords: List[str], top_k: int = 5
    ) -> List[RetrievalResult]:
        """实体级检索：用BM25匹配具体实体关键词"""
        results = []

        bm25_docs = self._bm25_search(entity_keywords, top_k)
        for doc in bm25_docs:
            node_id = doc.metadata.get("node_id", hash(doc.page_content))
            results.append(
                RetrievalResult(
                    content=doc.page_content,
                    node_id=str(node_id),
                    node_type=doc.metadata.get("node_type", "Recipe"),
                    relevance_score=0.9,
                    retrieval_level="entity",
                    metadata={
                        **doc.metadata,
                        "source": "bm25",
                    },
                )
            )

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def topic_level_retrieval(
        self, topic_keywords: List[str], top_k: int = 5
    ) -> List[RetrievalResult]:
        """主题级检索：用BM25匹配主题关键词"""
        results = []

        bm25_docs = self._bm25_search(topic_keywords, top_k)
        for doc in bm25_docs:
            node_id = doc.metadata.get("node_id", hash(doc.page_content))
            results.append(
                RetrievalResult(
                    content=doc.page_content,
                    node_id=str(node_id),
                    node_type=doc.metadata.get("node_type", "Recipe"),
                    relevance_score=0.85,
                    retrieval_level="topic",
                    metadata={
                        **doc.metadata,
                        "source": "bm25",
                    },
                )
            )

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """双层检索：结合实体级和主题级检索"""
        logger.info(f"开始双层检索: {query}")

        entity_keywords, topic_keywords = self.extract_query_keywords(query)

        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)

        all_results = entity_results + topic_results

        seen_nodes = set()
        unique_results = []
        for result in sorted(
            all_results, key=lambda x: x.relevance_score, reverse=True
        ):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)

        documents = []
        for result in unique_results[:top_k]:
            recipe_name = result.metadata.get("recipe_name") or result.metadata.get(
                "name", "未知菜品"
            )
            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "recipe_name": recipe_name,
                    "search_type": "dual_level",
                    **result.metadata,
                },
            )
            documents.append(doc)

        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return documents

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """向量检索（命中chunk后回溯完整父文档）"""
        try:
            vector_docs = self.milvus_module.similarity_search(query, k=top_k * 2)

            seen_parent_ids = set()
            docs = []
            for result in vector_docs:
                metadata = result.get("metadata", {})
                vector_score = result.get("score", 0.0)
                recipe_name = metadata.get("recipe_name", "未知菜品")
                logger.debug(f"向量检索得分: {recipe_name} = {vector_score}")

                # 构建临时 chunk doc，用于回溯父文档
                chunk_doc = Document(
                    page_content=result.get("text", ""),
                    metadata=metadata,
                )
                parent_doc = self._resolve_parent(chunk_doc)
                parent_id = parent_doc.metadata.get(
                    "node_id", hash(parent_doc.page_content)
                )

                # 父文档去重：同一菜谱只保留得分最高的那次命中
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)

                docs.append(
                    Document(
                        page_content=parent_doc.page_content,
                        metadata={
                            **parent_doc.metadata,
                            "recipe_name": parent_doc.metadata.get(
                                "recipe_name", recipe_name
                            ),
                            "score": vector_score,
                            "search_type": "vector_enhanced",
                        },
                    )
                )

            return docs[:top_k]

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def _rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        """用 CrossEncoder 对候选文档精排，过滤低相关性结果"""
        if not self.reranker or not docs:
            return docs[:top_k]

        try:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)

            scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

            result = []
            for score, doc in scored:
                if score < self.config.rerank_score_threshold:
                    break
                doc.metadata["rerank_score"] = float(score)
                result.append(doc)

            # 过滤后为空时，至少保留得分最高的1个
            if not result and scored:
                top_score, top_doc = scored[0]
                top_doc.metadata["rerank_score"] = float(top_score)
                result = [top_doc]

            result = result[:top_k]
            logger.info(
                f"Rerank完成：{len(docs)} -> {len(result)} 个文档"
                f"（阈值={self.config.rerank_score_threshold}）"
            )
            return result
        except Exception as e:
            logger.error(f"Rerank失败，降级返回RRF结果: {e}")
            return docs[:top_k]

    def hybrid_search(
        self, query: str, top_k: int = 5, rrf_k: int = 60
    ) -> List[Document]:
        """混合检索：并行执行双层检索和向量检索，RRF合并"""
        import concurrent.futures

        logger.info(f"开始并行混合检索: {query}")

        dual_docs = []
        vector_docs = []

        def dual_search():
            nonlocal dual_docs
            try:
                dual_docs = self.dual_level_retrieval(query, top_k * 2)
                logger.info(f"双层检索完成: {len(dual_docs)} 个结果")
            except Exception as e:
                logger.error(f"双层检索失败: {e}")

        def vector_search():
            nonlocal vector_docs
            try:
                vector_docs = self.vector_search_enhanced(query, top_k * 2)
                logger.info(f"向量检索完成: {len(vector_docs)} 个结果")
            except Exception as e:
                logger.error(f"向量检索失败: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_dual = executor.submit(dual_search)
            future_vector = executor.submit(vector_search)
            concurrent.futures.wait([future_dual, future_vector], timeout=20)

        # RRF合并：score = sum(1 / (rrf_k + rank)) across retrievers
        rrf_scores: dict = {}
        doc_map: dict = {}

        for rank, doc in enumerate(dual_docs):
            doc_id = doc.metadata.get("node_id", hash(doc.page_content))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_map:
                doc.metadata["search_method"] = "dual_level"
                doc_map[doc_id] = doc

        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("node_id", hash(doc.page_content))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            if doc_id not in doc_map:
                doc.metadata["search_method"] = "vector_enhanced"
                doc_map[doc_id] = doc
            else:
                # 文档同时出现在两路，标记来源
                doc_map[doc_id].metadata["search_method"] = "both"

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        final_docs = []
        for doc_id in sorted_ids[: top_k * 2]:  # 扩大候选集，rerank后再截断
            doc = doc_map[doc_id]
            doc.metadata["final_score"] = rrf_scores[doc_id]
            final_docs.append(doc)

        logger.info(
            f"RRF合并：从总共{len(dual_docs) + len(vector_docs)}个结果合并为{len(final_docs)}个文档"
        )

        # RRF 之后做精排和过滤
        final_docs = self._rerank(query, final_docs, top_k)

        return final_docs

    def close(self):
        """关闭资源连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")


if __name__ == "__main__":
    import ast
    import json
    import os
    import sys
    import time

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # )

    from dotenv import load_dotenv

    load_dotenv()

    from config import DEFAULT_CONFIG
    from rag_modules.graph_data_preparation import GraphDataPreparationModule
    from rag_modules.milvus_index_construction import MilvusIndexConstructionModule
    from rag_modules.generation_integration import GenerationIntegrationModule

    # ── 初始化系统 ──────────────────────────────────────────────
    config = DEFAULT_CONFIG

    print("初始化数据准备模块...")
    data_module = GraphDataPreparationModule(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database,
    )

    print("初始化向量索引...")
    index_module = MilvusIndexConstructionModule(
        host=config.milvus_host,
        port=config.milvus_port,
        collection_name=config.milvus_collection_name,
        dimension=config.milvus_dimension,
        model_name=config.embedding_model,
        api_key=config.embedding_api_key,
        base_url=config.embedding_base_url,
        persist_directory=config.chroma_persist_dir,
    )

    print("初始化生成模块...")
    generation_module = GenerationIntegrationModule(
        model_name=config.llm_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # 加载数据
    data_module.load_graph_data()
    data_module.build_recipe_documents()
    chunks = data_module.chunk_documents(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # 加载向量索引
    if index_module.has_collection():
        index_module.load_collection()
    else:
        index_module.build_vector_index(chunks)

    # 初始化检索模块
    retrieval = HybridRetrievalModule(
        config=config,
        milvus_module=index_module,
        data_module=data_module,
        llm_client=generation_module.client,
    )
    retrieval.initialize(chunks)

    # ── 评测函数 ──────────────────────────────────────────────
    def evaluate_query(query: str, relevant_names: list[str], top_k: int = 5) -> dict:

        t0 = time.time()
        docs = retrieval.hybrid_search(query, top_k=top_k)
        latency = time.time() - t0

        # 取检索结果中的 recipe_name 字段
        retrieved_names = set()
        for doc in docs:
            name = doc.metadata.get("recipe_name") or doc.metadata.get("name", "")
            if name:
                retrieved_names.add(name)

        relevant_set = set(relevant_names)
        # print(retrieved_names)
        hits = retrieved_names & relevant_set

        precision = len(hits) / len(retrieved_names) if retrieved_names else 0.0
        recall = len(hits) / len(relevant_set) if relevant_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "query": query,
            "relevant": relevant_names,
            "retrieved": list(retrieved_names),
            "hits": list(hits),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "score": round(f1, 4),
            "latency_s": round(latency, 3),
        }

    # ── 读取评测集并运行 ──────────────────────────────────────
    eval_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "query",
        "relevant.txt",
    )

    eval_set = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "END":
                break
            if not line:
                continue
            row = json.loads(line)  # 每行是 JSON 数组
            query_text = row[0]
            relevant_docs = row[1:]
            eval_set.append((query_text, relevant_docs))

    top_k = 5

    print(f"\n{'='*60}")
    print(f"评测开始  top_k={top_k}  共 {len(eval_set)} 条查询")
    print(f"{'='*60}")

    all_scores = []
    for query_text, relevant_docs in eval_set:
        result = evaluate_query(query_text, relevant_docs, top_k=top_k)
        all_scores.append(result["score"])
        print(
            f"[{result['score']:.4f}]  P={result['precision']:.4f}  "
            f"R={result['recall']:.4f}  {result['latency_s']:.2f}s  "
            f"Q: {query_text[:30]}"
        )

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n{'='*60}")
    print(f"F1分数: {avg_score:.4f}")
    print(f"{'='*60}\n")
