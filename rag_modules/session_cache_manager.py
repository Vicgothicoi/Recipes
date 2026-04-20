"""
会话缓存管理模块
负责管理会话级语义缓存和上下文
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionCacheManager:
    """
    会话级缓存管理器

    功能：
    1. 会话级语义缓存 - 每个聊天窗口独立缓存
    2. 上下文管理 - 维护对话历史
    3. 语义相似度匹配 - 智能缓存命中
    """

    def __init__(self, embedding_model=None):
        """初始化缓存管理器"""
        self.embedding_model = embedding_model

        # 🚀 会话级语义缓存系统 - 针对每个聊天窗口独立缓存
        self.session_caches = (
            {}
        )  # 按session_id分组的缓存：{session_id: {query: response}}
        self.session_embeddings = (
            {}
        )  # 按session_id分组的向量：{session_id: {query: embedding}}
        self.session_contexts = {}  # 按session_id分组的上下文：{session_id: [messages]}

        # 会话上下文的 query embedding 缓存，用于相关性加权选取
        # 结构：{session_id: {query_str: np.ndarray}}
        self.context_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

        # 缓存配置
        self.cache_threshold = 0.75  # 语义相似度阈值
        self.max_session_cache_size = 50  # 每个会话最大缓存条目数
        self.max_context_length = 10  # 每个会话保留的最大上下文消息数
        self.context_top_k = 3  # get_context_for_query 最多选取的轮数
        self.recency_weight = 0.4  # 时间衰减权重（0~1），余下为相关性权重

    def _calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """计算两个向量的余弦相似度"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def check_semantic_cache(self, query: str, session_id: str = None) -> Optional[str]:
        """检查会话级语义缓存中是否有相似查询"""
        if not session_id or session_id not in self.session_caches:
            return None

        session_cache = self.session_caches[session_id]
        session_embeddings = self.session_embeddings[session_id]

        if not session_cache:
            return None

        try:
            # 计算查询向量
            query_embedding = self.embedding_model.embed_documents([query])[0]
        except Exception as e:
            logger.warning(f"查询向量计算失败: {e}")
            return None

        # 查找最相似的缓存查询
        best_similarity = 0
        best_response = None

        for cached_query, cached_data in session_cache.items():
            cached_embedding = session_embeddings.get(cached_query)
            if cached_embedding is not None:
                similarity = self._calculate_similarity(
                    query_embedding, cached_embedding
                )
                if similarity > best_similarity and similarity >= self.cache_threshold:
                    best_similarity = similarity
                    best_response = cached_data["response"]

        if best_response:
            logger.info(
                f"🎯 会话缓存命中! Session: {session_id}, 相似度: {best_similarity:.3f}"
            )
            return best_response

        return None

    def add_to_semantic_cache(self, query: str, response: str, session_id: str = None):
        """将查询-答案对添加到会话级语义缓存"""
        try:
            if not session_id:
                return

            # 初始化会话缓存
            if session_id not in self.session_caches:
                self.session_caches[session_id] = {}
                self.session_embeddings[session_id] = {}

            session_cache = self.session_caches[session_id]
            session_embeddings = self.session_embeddings[session_id]

            # 限制会话缓存大小
            if len(session_cache) >= self.max_session_cache_size:
                # 删除最旧的缓存项
                oldest_key = next(iter(session_cache))
                del session_cache[oldest_key]
                del session_embeddings[oldest_key]

            # 计算查询向量
            query_embedding = self.embedding_model.embed_documents([query])[0]

            # 添加到缓存
            session_cache[query] = {
                "response": response,
                "timestamp": datetime.now().isoformat(),
            }
            session_embeddings[query] = query_embedding

            logger.info(
                f"📝 已添加到会话缓存 {session_id}, 当前大小: {len(session_cache)}"
            )

        except Exception as e:
            logger.warning(f"添加到语义缓存失败: {e}")

    def add_to_context(self, session_id: str, query: str, response: str):
        """添加对话到上下文历史，同时缓存 query embedding 供相关性计算使用"""
        try:
            if not session_id:
                return

            # 初始化会话上下文
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = []
            if session_id not in self.context_embeddings:
                self.context_embeddings[session_id] = {}

            context = self.session_contexts[session_id]
            emb_store = self.context_embeddings[session_id]

            # 添加新的对话
            context.append(
                {
                    "query": query,
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # 顺带计算并缓存该轮 query 的 embedding，失败不影响主流程
            try:
                emb_store[query] = np.array(
                    self.embedding_model.embed_documents([query])[0]
                )
            except Exception as e:
                logger.warning(f"上下文 query embedding 缓存失败: {e}")

            # 限制上下文长度，同步清理对应 embedding
            if len(context) > self.max_context_length:
                removed = context.pop(0)
                emb_store.pop(removed["query"], None)

            logger.info(f"📝 已添加上下文到会话 {session_id}, 当前长度: {len(context)}")

        except Exception as e:
            logger.warning(f"添加上下文失败: {e}")

    def get_context_for_query(self, session_id: str, current_query: str) -> str:
        """
        获取增强的查询上下文。

        从历史对话中选取与当前 query 最相关的 top_k 轮，
        评分 = (1 - recency_weight) * 语义相似度 + recency_weight * 时间衰减得分。
        embedding 不可用时降级为取最近 top_k 轮。
        """
        try:
            if not session_id or session_id not in self.session_contexts:
                return current_query

            context = self.session_contexts[session_id]
            if not context:
                return current_query

            emb_store = self.context_embeddings.get(session_id, {})
            n = len(context)

            # ── 尝试计算当前 query 的 embedding ──────────────────────
            try:
                current_emb = np.array(
                    self.embedding_model.embed_documents([current_query])[0]
                )
                has_embedding = True
            except Exception as e:
                logger.warning(f"当前 query embedding 计算失败，降级为时间窗口: {e}")
                has_embedding = False

            # ── 对每轮历史对话打分 ────────────────────────────────────
            scored = []
            for i, item in enumerate(context):
                # 时间衰减得分：越近越高，线性归一化到 [0, 1]
                recency_score = (i + 1) / n  # i=0 最旧，i=n-1 最新

                if has_embedding:
                    cached_emb = emb_store.get(item["query"])
                    if cached_emb is not None:
                        sim = float(self._calculate_similarity(current_emb, cached_emb))
                    else:
                        # 该轮 embedding 缺失，用时间得分代替相关性
                        sim = recency_score

                    score = (
                        1 - self.recency_weight
                    ) * sim + self.recency_weight * recency_score
                else:
                    # 无 embedding，纯时间衰减
                    score = recency_score

                scored.append((score, i, item))

            # ── 取得分最高的 top_k 轮，按原始时间顺序排列 ────────────
            top = sorted(scored, key=lambda x: x[0], reverse=True)[: self.context_top_k]
            top = sorted(top, key=lambda x: x[1])  # 恢复时间顺序

            # ── 拼接上下文 ────────────────────────────────────────────
            context_parts = []
            for _, _, item in top:
                context_parts.append(f"用户问: {item['query']}")
                # response 保留前 200 字符，比原来的 100 更完整
                context_parts.append(f"AI答: {item['response'][:200]}")

            context_parts.append(f"当前问题: {current_query}")

            enhanced_query = "\n".join(context_parts)
            logger.info(
                f"🔗 会话 {session_id} 上下文增强完成，"
                f"从 {n} 轮中选取 {len(top)} 轮"
                f"{'（相关性加权）' if has_embedding else '（时间衰减降级）'}"
            )
            return enhanced_query

        except Exception as e:
            logger.warning(f"上下文获取失败: {e}")
            return current_query

    def get_session_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "total_sessions": len(self.session_caches),
            "total_cached_queries": sum(
                len(cache) for cache in self.session_caches.values()
            ),
            "total_contexts": sum(
                len(context) for context in self.session_contexts.values()
            ),
            "cache_threshold": self.cache_threshold,
            "max_session_cache_size": self.max_session_cache_size,
            "max_context_length": self.max_context_length,
            "context_top_k": self.context_top_k,
            "recency_weight": self.recency_weight,
        }

    def clear_session_cache(self, session_id: str):
        """清除指定会话的缓存"""
        if session_id in self.session_caches:
            del self.session_caches[session_id]
        if session_id in self.session_embeddings:
            del self.session_embeddings[session_id]
        if session_id in self.session_contexts:
            del self.session_contexts[session_id]
        if session_id in self.context_embeddings:
            del self.context_embeddings[session_id]
        logger.info(f"🗑️ 已清除会话 {session_id} 的缓存")

    def clear_all_caches(self):
        """清除所有缓存"""
        self.session_caches.clear()
        self.session_embeddings.clear()
        self.session_contexts.clear()
        self.context_embeddings.clear()
        logger.info("🗑️ 已清除所有会话缓存")
