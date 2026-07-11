# 基于图数据库的的菜品推荐与制作RAG系统

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [环境变量](#环境变量)
- [模块说明](#模块说明)
- [待办事项](#待办事项)

---

## 项目简介

"今天吃什么"是一个端到端的 RAG（检索增强生成）系统，以中文菜谱知识库为数据源，结合图数据库（Neo4j）与向量数据库（Milvus），为用户提供智能烹饪问答与菜谱推荐服务。

---

## 快速开始

### 前置条件

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose

### 1. 启动数据库

```bash
docker compose up -d
```

这将启动 Neo4j、Milvus（含 etcd 和 MinIO）并自动导入菜谱图数据。

等待所有容器健康检查通过（约 1 分钟）：

```bash
docker compose ps
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 LLM API Key：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 启动后端

```bash
python main.py
```

后端默认监听 `http://localhost:8000`。

### 5. 安装并启动前端

```bash
cd frontend
npm install
npm run dev
```

前端默认运行在 `http://localhost:3000`。

### 关闭顺序

关闭时按照与启动相反的顺序操作：先停前端，再停后端，最后停数据库容器。

```bash
docker compose down
```

---

## 系统架构

```
用户查询
   │
   ▼
智能查询路由器 (IntelligentQueryRouter)
   ├── 查询复杂度分析（长度 × 0.1 + 实体数 × 0.4 + 句法信号 × 0.5）
   ├── 关系密集度检测（图节点一度邻居高价值类型命中）
   └── BGE 分类头判断是否需要推理
         │
   ┌─────┴──────┐
   ▼            ▼
传统混合检索   图 RAG 检索
(BM25 + 向量)  (Neo4j 多跳遍历)
   │            │
   └─────┬──────┘
         ▼
    RRF 融合 + BGE Cross-Encoder 重排序
         │
         ▼
    LLM 生成回答（流式输出）
```

---

## 核心特性

### 检索路由
- 查询复杂度由公式计算，不再完全依赖 LLM 打分，降低延迟与 Token 消耗
- 关系密集度通过图结构直查（一度邻居高价值节点命中数）衡量，退化方案为与预定义模板句做余弦相似度
- 使用 BGE + 分类头替代 LLM 判断"是否需要推理"，显著提升可解释性
- 硬编码路由逻辑，避免 LLM 路由的不确定性
- 并行执行双引擎检索，充分利用性能优势

### 传统混合检索
- 双层检索范式：实体级 + 主题级
- BM25 关键词检索 + 向量语义检索
- RRF融合，解决 round-robin 无法增加重复文档权重的问题
- Parent Document Retrieval：通过子文档回溯父文档，提升上下文完整性
- BGE Cross-Encoder 重排序，过滤低相关度文档

### 图 RAG 检索
- 支持 5 种查询类型：实体关系查询、聚类查询、路径查找、多跳遍历、子图提取
- 基于 Neo4j 的知识图谱，包含菜谱、食材、烹饪方式、菜系等节点及其关系

### 会话管理
- 滑动摘要机制：用查询语义相似度 + 时间衰减打分，取最相关的 3 轮历史对话
- 检索阶段使用原始 query，生成阶段使用携带历史的增强 query

### 菜谱推荐
- 由 LLM 完成食材清单与烹饪步骤的准确匹配提取
- 难度字段直接从菜谱文本中读取

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 前端 | Next.js 14、React 18、TypeScript、Tailwind CSS、Framer Motion |
| 后端 | Python、Flask |
| 图数据库 | Neo4j 5.11（含 APOC、GDS 插件） |
| 向量数据库 | Milvus 2.3（etcd + MinIO） |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（512 维） |
| 重排序模型 | BAAI/bge-reranker-base |
| 语言模型 | 兼容 OpenAI 格式的任意供应商（默认 moonshotai/Kimi-K2-Instruct） |
| NLP | spaCy zh_core_web_sm（依存句法分析） |
| 检索框架 | LangChain、rank-bm25、sentence-transformers |
| 容器化 | Docker Compose |

---

## 项目结构

```
.
├── main.py                        # 主程序入口
├── config.py                      # 系统配置
├── requirements.txt               # Python 依赖
├── docker-compose.yml             # 数据库容器配置
├── .env.example                   # 环境变量示例
├── REVISED.md                     # 改动说明与 TODO
├── rag_modules/
│   ├── graph_data_preparation.py  # 数据准备：从 Neo4j 加载菜谱文档
│   ├── milvus_index_construction.py # 向量索引构建与管理
│   ├── hybrid_retrieval.py        # 传统混合检索（BM25 + 向量 + RRF + 重排序）
│   ├── graph_rag_retrieval.py     # 图 RAG 检索（多跳遍历、子图提取等）
│   ├── intelligent_query_router.py # 智能查询路由器
│   ├── BGE_classifier.py          # BGE + 分类头（判断是否需要推理）
│   ├── generation_integration.py  # LLM 生成集成（流式输出）
│   ├── session_cache_manager.py   # 会话缓存与短期记忆管理
│   ├── recipe_recommendation.py   # 菜谱推荐
│   └── web_service_handler.py     # Flask API 路由
├── data/
│   ├── cypher/                    # Neo4j 导入脚本及 CSV 数据
│   └── dishes/                    # 菜谱 Markdown 文件（按类别分类）
│       ├── aquatic/               # 水产类
│       ├── breakfast/             # 早餐类
│       ├── condiment/             # 调料类
│       ├── dessert/               # 甜点类
│       └── ...
└── frontend/                      # Next.js 前端
    └── src/
```

---


## 环境变量

完整配置项见 [.env.example](./.env.example)。

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `NEO4J_URI` | Neo4j 连接地址 | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j 用户名 | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j 密码 | `all-in-rag` |
| `MILVUS_HOST` | Milvus 主机 | `localhost` |
| `MILVUS_PORT` | Milvus 端口 | `19530` |
| `EMBEDDING_MODEL` | 嵌入模型名称 | `BAAI/bge-small-zh-v1.5` |
| `LLM_MODEL` | 语言模型名称 | `moonshotai/Kimi-K2-Instruct` |
| `OPENAI_API_KEY` | LLM API Key | — |
| `OPENAI_BASE_URL` | LLM API 地址 | `https://api.siliconflow.cn/v1` |

前端环境变量在 `frontend/.env.local` 中配置：

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 模块说明

### `IntelligentQueryRouter`
智能查询路由器，综合以下三个维度决定检索策略：
- **查询复杂度**：`长度 × 0.1 + 实体数 × 0.4 + 句法结构信号 × 0.5`
- **关系密集度**：提取实体后查询其一度邻居中高价值节点类型的命中数量
- **推理需求**：BGE + 分类头模型判断（约需 300 条标注数据训练）

路由结果为三选一：`HYBRID_TRADITIONAL` / `GRAPH_RAG` / `COMBINED`。

### `HybridRetrievalModule`
传统混合检索，流程为：
1. LLM 提取实体级 + 主题级关键词
2. BM25 关键词检索 + Milvus 向量检索
3. RRF 融合两路结果
4. 通过 `parent_id` 回溯父文档
5. BGE Cross-Encoder 重排序并过滤低分文档

### `GraphRAGRetrieval`
图 RAG 检索，支持 5 种查询类型：
- `ENTITY_RELATION`：实体关系查询
- `CLUSTERING`：相似菜谱聚类
- `PATH_FINDING`：最短路径查找
- `MULTI_HOP`：多跳路径遍历
- `SUBGRAPH`：子图提取

### `SessionCacheManager`
会话缓存管理，使用**滑动摘要**策略：对历史对话按语义相似度 + 时间衰减打分，取得分最高的 3 轮作为短期记忆。

