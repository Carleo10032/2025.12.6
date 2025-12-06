# Comesen_1 项目流程总览

本项目实现从学术 PDF 到知识图谱的完整 Graph RAG 工作流：解析文档→抽取三元组→导入 Neo4j→社区检测与摘要→向量化存储（Qdrant/Neo4j）→检索工具与 ReAct 智能体联动。

## 目录
- 项目概述
- 整体流程
- 快速开始
- 环境变量配置
- 数据处理管道
- 图谱构建与增强
- 检索与智能体
- 目录结构
- 常见问题与排错

## 项目概述
- 目标：通过解析 PDF 并抽取结构化知识三元组，构建可检索的知识图谱，并结合向量数据库与 LLM 智能体实现高质量问答与推理。
- 技术栈：`Neo4j`、`Qdrant`、`LlamaIndex`、`OpenAI API`、`PyMuPDF`。
- 关键组件：
  - PDF 解析器：`str/pdf_to_json.py:259`
  - 三元组抽取：`str/extract_triples.py:339`
  - 图谱导入：`str/neo4j_ingest.py:118`
  - 社区检测与摘要/向量化：`str/graph_RAG.py:249`
  - 图检索工具：`str/search_tool.py:179`
  - ReAct 智能体：`str/react_workflow.py:114`

## 整体流程
1. 解析 PDF 为结构化 JSON（章节/段落/表格统计）
2. 使用 LLM 从 JSON 文本块中抽取高质量知识三元组（含质量控制与去重）
3. 将三元组导入 Neo4j，建立实体、关系、文档/块的连接以及层级分类
4. 使用 Neo4j GDS 进行社区检测，并通过 LLM 生成社区摘要
5. 对实体与关系进行向量化，写入 `Qdrant` 或回写到 `Neo4j` 节点/边属性
6. 基于向量检索 + 图结构检索，返回社区洞察与关联路径
7. 通过 ReAct 智能体整合工具调用与推理过程，提供交互式问答

## 快速开始
1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
2. 启动基础服务（Neo4j + Qdrant）
   ```bash
   docker-compose up -d
   ```
3. 配置环境变量（见下文 `.env`）
4. 运行各阶段命令（见数据处理管道、图谱构建与增强、检索与智能体）

## 环境变量配置
在项目根目录创建 `.env`，示例：
```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=www163com
NEO4J_DB=neo4j

# 向量/LLM（云雾 API 兼容 OpenAI 格式）
YUNWU_BASE_URL=https://yunwu.ai/v1
YUNWU_API_KEY=your_api_key
YUNWU_MODEL=gpt-4o-mini
YUNWU_EMBED_MODEL=text-embedding-3-large

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY= #可空
QDRANT_COLLECTION=kg_items
```
- 这些环境变量在多处被读取：如 `str/search_tool.py:21-47`、`str/graph_RAG.py:29-46`、`str/extract_triples.py:256-273`、`str/neo4j_ingest.py:124-133`。

## 数据处理管道
- 解析 PDF 为结构化 JSON
  ```bash
  python str/pdf_to_json.py --input ./agent/datas --output ./output --format structured
  ```
  - 主入口：`str/pdf_to_json.py:259`
  - 输出示例结构：`pdf_output/*.json`（本仓库提供样例）

- 抽取知识三元组（批处理 + 质量控制）
  ```bash
  python str/extract_triples.py \
    --input ./output \
    --output ./triples \
    --model gpt-4o-mini \
    --prompt-version v3_academic \
    --sample 0 \
    --batch-size 20
  ```
  - 主入口：`str/extract_triples.py:339`
  - 核心工作流：`TripleExtractionWorkflow.execute` 在 `str/extract_triples.py:63`
  - 提示词模块：`str/prompts.py:1-36`
  - 质量控制：`str/triple_quality.py:1-31`

## 图谱构建与增强
- 将三元组导入 Neo4j
  ```bash
  python str/neo4j_ingest.py --input ./triples --env ./.env --clear
  ```
  - 主入口：`str/neo4j_ingest.py:118`
  - 约束设置：`ensure_constraints` 在 `str/neo4j_ingest.py:21-26`
  - 导入逻辑：`ingest_file` 在 `str/neo4j_ingest.py:67-115`
  - 层级分类：`classify_layer` 在 `str/neo4j_ingest.py:54-65`

- 社区检测 + 摘要 + 向量化
  ```bash
  # 运行 GDS 社区检测、可选生成摘要，并写入向量库
  python str/graph_RAG.py \
    --env ./.env \
    --algo louvain \
    --graph kg \
    --summarize \
    --vectorstore qdrant \
    --edge-limit 50
  ```
  - 主入口：`str/graph_RAG.py:249`
  - 构建 GDS 投影：`ensure_gds_graph` 在 `str/graph_RAG.py:49-67`
  - 社区检测：`run_community_detection` 在 `str/graph_RAG.py:69-109`
  - 摘要生成：`summarize_group` 在 `str/graph_RAG.py:126-136`
  - 向量化采集：`collect_items` 在 `str/graph_RAG.py:154-174`
  - 写入 Qdrant：`upsert_qdrant` 在 `str/graph_RAG.py:177-223`
  - 写回 Neo4j：`upsert_neo4j_embeddings` 在 `str/graph_RAG.py:226-246`

## 检索与智能体
- 交互式图检索 CLI（向量检索 + 图结构检索）
  ```bash
  python str/search_tool.py
  ```
  - 主入口：`str/search_tool.py:179-217`
  - 检索流程：`GraphRAGSearcher.search` 在 `str/search_tool.py:113-156`
  - Qdrant 查询：`search_qdrant` 在 `str/search_tool.py:51-72`
  - Neo4j 上下文：`search_neo4j_context` 在 `str/search_tool.py:74-112`

- ReAct 智能体（带推理追踪）
  ```bash
  python str/react_workflow.py
  ```
  - 主入口：`str/react_workflow.py:114-149`
  - 工具集成：加载 `search_tool` 于 `str/react_workflow.py:19-23`
  - 事件追踪：`ReActTraceHandler` 于 `str/react_workflow.py:29-57`
  - 也可运行示例测试：`python str/test_trace.py`（`str/test_trace.py:1-16`）

## 目录结构
```
Comesen_1/
├─ str/
│  ├─ pdf_to_json.py        # PDF→结构化JSON
│  ├─ extract_triples.py    # JSON→三元组(LLM+质量控制)
│  ├─ neo4j_ingest.py       # 三元组→Neo4j图谱
│  ├─ graph_RAG.py          # 社区检测/摘要/向量化
│  ├─ search_tool.py        # 图检索工具(向量+图)
│  ├─ react_workflow.py     # ReAct 智能体集成
│  ├─ prompts.py            # 抽取提示词
│  └─ triple_quality.py     # 质量控制
├─ pdf_output/              # 示例解析输出(JSON)
├─ data/
│  ├─ neo4j/                # Neo4j 挂载数据卷
│  └─ qdrant/               # Qdrant 存储
├─ requirements.txt
└─ docker-compose.yml
```

## 常见问题与排错
- 缺少 `YUNWU_API_KEY`
  - 报错来源：`str/search_tool.py:45`、`str/graph_RAG.py:32-46`、`str/extract_triples.py:256-273`
  - 解决：在 `.env` 中正确设置 `YUNWU_API_KEY` 和模型配置。
- 缺少 `NEO4J_PASSWORD`
  - 报错来源：`str/search_tool.py:26-29`、`str/neo4j_ingest.py:129-133`、`str/graph_RAG.py:262-266`
  - 解决：在 `.env` 设置密码，并确认 `docker-compose.yml:10` 的默认密码一致或调整。
- Neo4j GDS 插件
  - 已在 `docker-compose.yml:11-12` 启用 `graph-data-science` 与 `apoc` 插件；若本地 Neo4j 非容器部署需手动安装。
- Qdrant 可选
  - 若不启用向量检索：运行 `graph_RAG.py` 时指定 `--vectorstore none`，或仅写回 Neo4j 向量。
- 数据量与性能
  - 处理社区与向量化数量已在代码中做限制，例如 `str/graph_RAG.py:271-291`；可按需调整。