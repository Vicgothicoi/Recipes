"""
向量索引构建模块

注意：类名保留为 MilvusIndexConstructionModule 以兼容既有调用，
      但底层实现已替换为本地 ChromaDB（PersistentClient），不再依赖 Milvus。
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MilvusIndexConstructionModule:
    """向量索引构建模块 - 负责向量化和 ChromaDB 索引构建"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 collection_name: str = "cooking_knowledge",
                 dimension: int = 1024,
                 model_name: str = "text-embedding-v4",
                 api_key: str = "",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 persist_directory: str = "./chroma_db"):
        """
        初始化向量索引构建模块（ChromaDB 本地持久化）

        Args:
            host: 兼容保留参数（ChromaDB 本地模式下未使用）
            port: 兼容保留参数（ChromaDB 本地模式下未使用）
            collection_name: 集合名称
            dimension: 向量维度
            model_name: 嵌入模型名称
            api_key: 百炼 API Key（为空时从 DASHSCOPE_API_KEY 环境变量读取）
            base_url: Embedding API 地址
            persist_directory: ChromaDB 本地持久化目录
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", "./chroma_db"
        )

        self.client = None
        self.collection = None
        self.embeddings = None
        self.collection_created = False

        self._setup_client()
        self._setup_embeddings()

    def _safe_truncate(self, text: str, max_length: int) -> str:
        """
        安全截取字符串，处理None值

        Args:
            text: 输入文本
            max_length: 最大长度

        Returns:
            截取后的字符串
        """
        if text is None:
            return ""
        return str(text)[:max_length]

    def _setup_client(self):
        """初始化 ChromaDB 本地持久化客户端"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"已连接到 ChromaDB（本地持久化目录: {self.persist_directory}）")

            collections = [c.name for c in self.client.list_collections()]
            logger.info(f"连接成功，当前集合: {collections}")

        except Exception as e:
            logger.error(f"连接 ChromaDB 失败: {e}")
            raise

    def _setup_embeddings(self):
        """初始化嵌入模型（阿里云百炼 OpenAI 兼容接口）"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")

        if not self.api_key:
            logger.warning("DASHSCOPE_API_KEY 未设置，Embedding API 调用将失败")

        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            dimensions=self.dimension,
            chunk_size=10,
            # 百炼兼容接口只接受字符串输入，需跳过 tiktoken 分词（否则会发送 token id 数组）
            check_embedding_ctx_length=False,
        )

        logger.info(
            f"嵌入模型初始化完成 (API: {self.base_url}, dimensions={self.dimension})"
        )

    def _build_metadata(self, chunk: Document, fallback_id: str) -> Dict[str, Any]:
        """从文档块构建 ChromaDB metadata（仅保留标量类型）"""
        return {
            "text": self._safe_truncate(chunk.page_content, 15000),
            "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
            "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
            "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
            "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
            "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
            "difficulty": int(chunk.metadata.get("difficulty", 0) or 0),
            "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
            "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", fallback_id), 150),
            "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100),
        }

    def create_collection(self, force_recreate: bool = False) -> bool:
        """
        创建 ChromaDB 集合

        Args:
            force_recreate: 是否强制重新创建集合

        Returns:
            是否创建成功
        """
        try:
            existing = [c.name for c in self.client.list_collections()]

            if self.collection_name in existing:
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection = self.client.get_collection(self.collection_name)
                    self.collection_created = True
                    return True

            # 使用余弦距离创建集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(f"成功创建集合: {self.collection_name}")
            self.collection_created = True
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def build_vector_index(self, chunks: List[Document]) -> bool:
        """
        构建向量索引

        Args:
            chunks: 文档块列表

        Returns:
            是否构建成功
        """
        logger.info(f"正在构建 ChromaDB 向量索引，文档数量: {len(chunks)}...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        try:
            # 1. 创建集合（强制重建，保证与新向量维度一致）
            if not self.create_collection(force_recreate=True):
                return False

            # 2. 生成向量
            logger.info("正在生成向量embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)

            # 3. 准备数据
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            for i, chunk in enumerate(chunks):
                chunk_id = self._safe_truncate(
                    chunk.metadata.get("chunk_id", f"chunk_{i}"), 150
                )
                # ChromaDB 要求 id 唯一，重复时追加序号
                unique_id = chunk_id or f"chunk_{i}"
                ids.append(f"{unique_id}_{i}")
                documents.append(self._safe_truncate(chunk.page_content, 15000))
                metadatas.append(self._build_metadata(chunk, f"chunk_{i}"))

            # 4. 批量写入
            logger.info("正在插入向量数据...")
            batch_size = 100
            total = len(ids)
            for i in range(0, total, batch_size):
                end = min(i + batch_size, total)
                self.collection.add(
                    ids=ids[i:end],
                    embeddings=vectors[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                )
                logger.info(f"已插入 {end}/{total} 条数据")

            logger.info(f"向量索引构建完成，包含 {total} 个向量")
            return True

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            return False

    def add_documents(self, new_chunks: List[Document]) -> bool:
        """
        向现有索引添加新文档

        Args:
            new_chunks: 新的文档块列表

        Returns:
            是否添加成功
        """
        if not self.collection_created or self.collection is None:
            raise ValueError("请先构建向量索引")

        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")

        try:
            texts = [chunk.page_content for chunk in new_chunks]
            vectors = self.embeddings.embed_documents(texts)

            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            timestamp = int(time.time())
            for i, chunk in enumerate(new_chunks):
                fallback = f"new_chunk_{i}_{timestamp}"
                chunk_id = self._safe_truncate(
                    chunk.metadata.get("chunk_id", fallback), 150
                )
                ids.append(f"{chunk_id or fallback}_{timestamp}_{i}")
                documents.append(self._safe_truncate(chunk.page_content, 15000))
                metadatas.append(self._build_metadata(chunk, fallback))

            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info("新文档添加完成")
            return True

        except Exception as e:
            logger.error(f"添加新文档失败: {e}")
            return False

    def _build_where(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将简单过滤条件转换为 ChromaDB where 语法"""
        if not filters:
            return None

        conditions: List[Dict[str, Any]] = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append({key: {"$in": value}})
            else:
                conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filters: 过滤条件

        Returns:
            搜索结果列表
        """
        if not self.collection_created or self.collection is None:
            raise ValueError("请先构建或加载向量索引")

        try:
            query_vector = self.embeddings.embed_query(query)
            where = self._build_where(filters) if filters else None

            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_vector],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where

            results = self.collection.query(**query_kwargs)

            formatted_results: List[Dict[str, Any]] = []
            ids = results.get("ids", [[]])
            if not ids or not ids[0]:
                return formatted_results

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, hit_id in enumerate(ids[0]):
                metadata = metas[i] or {}
                distance = distances[i] if distances else 0.0
                # cosine 距离转相似度：越大越相似，与下游语义保持一致
                score = 1.0 - float(distance)
                text = docs[i] if docs else metadata.get("text", "")

                formatted_results.append({
                    "id": hit_id,
                    "score": score,
                    "text": text,
                    "metadata": {
                        "node_id": metadata.get("node_id", ""),
                        "recipe_name": metadata.get("recipe_name", ""),
                        "node_type": metadata.get("node_type", ""),
                        "category": metadata.get("category", ""),
                        "cuisine_type": metadata.get("cuisine_type", ""),
                        "difficulty": metadata.get("difficulty", 0),
                        "doc_type": metadata.get("doc_type", ""),
                        "chunk_id": metadata.get("chunk_id", ""),
                        "parent_id": metadata.get("parent_id", ""),
                    },
                })

            return formatted_results

        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息字典
        """
        try:
            if not self.collection_created or self.collection is None:
                return {"error": "集合未创建"}

            row_count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "row_count": row_count,
                "stats": {"row_count": row_count},
            }

        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}

    def delete_collection(self) -> bool:
        """
        删除集合

        Returns:
            是否删除成功
        """
        try:
            existing = [c.name for c in self.client.list_collections()]
            if self.collection_name in existing:
                self.client.delete_collection(self.collection_name)
                logger.info(f"集合 {self.collection_name} 已删除")
                self.collection = None
                self.collection_created = False
                return True
            else:
                logger.info(f"集合 {self.collection_name} 不存在")
                return True

        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    def has_collection(self) -> bool:
        """
        检查集合是否存在

        Returns:
            集合是否存在
        """
        try:
            existing = [c.name for c in self.client.list_collections()]
            return self.collection_name in existing
        except Exception as e:
            logger.error(f"检查集合存在性失败: {e}")
            return False

    def load_collection(self) -> bool:
        """
        加载集合（ChromaDB 持久化后直接获取集合句柄即可）

        Returns:
            是否加载成功
        """
        try:
            existing = [c.name for c in self.client.list_collections()]
            if self.collection_name not in existing:
                logger.error(f"集合 {self.collection_name} 不存在")
                return False

            self.collection = self.client.get_collection(self.collection_name)
            self.collection_created = True
            logger.info(f"集合 {self.collection_name} 已加载")
            return True

        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False

    def close(self):
        """关闭连接（ChromaDB 本地客户端无需显式关闭）"""
        if hasattr(self, 'client') and self.client:
            logger.info("ChromaDB 连接已关闭")

    def __del__(self):
        """析构函数"""
        self.close()
