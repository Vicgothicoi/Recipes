"""
基于图数据库的RAG系统配置文件
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "all-in-rag")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # 向量数据库配置（ChromaDB 本地持久化）
    # 说明：host/port 为兼容旧接口保留，ChromaDB 本地模式下不使用
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    milvus_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME",
        os.getenv("MILVUS_COLLECTION_NAME", "cooking_knowledge"),
    )
    milvus_dimension: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # 模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    embedding_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    embedding_base_url: str = os.getenv(
        "EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm_model: str = os.getenv("LLM_MODEL", "moonshot-v1-8k")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    rerank_score_threshold: float = float(
        os.getenv("RERANK_SCORE_THRESHOLD", "0.5")
    )  # 低于此分数的文档丢弃，至少保留得分最高的1个

    # 检索配置（
    top_k: int = 5

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 图数据处理配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_graph_depth: int = 2  # 图遍历最大深度

    def __post_init__(self):
        """初始化后的处理"""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GraphRAGConfig":
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": self.neo4j_password,
            "neo4j_database": self.neo4j_database,
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "milvus_collection_name": self.milvus_collection_name,
            "milvus_dimension": self.milvus_dimension,
            "chroma_persist_dir": self.chroma_persist_dir,
            "embedding_model": self.embedding_model,
            "embedding_base_url": self.embedding_base_url,
            "llm_model": self.llm_model,
            "rerank_model": self.rerank_model,
            "rerank_score_threshold": self.rerank_score_threshold,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_graph_depth": self.max_graph_depth,
        }


# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig()
