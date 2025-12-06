import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.openai import OpenAIEmbedding

# 加载环境变量
load_dotenv()

class GraphRAGSearcher:
    def __init__(self):
        self._init_neo4j()
        self._init_qdrant()
        self._init_embed()

    def _init_neo4j(self):
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        database = os.environ.get("NEO4J_DB", "neo4j")
        if not password:
            raise ValueError("Missing NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.db_name = database

    def _init_qdrant(self):
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        api_key = os.environ.get("QDRANT_API_KEY")
        self.collection = os.environ.get("QDRANT_COLLECTION", "kg_items")
        if api_key:
            self.qdrant = QdrantClient(url=url, api_key=api_key)
        else:
            self.qdrant = QdrantClient(url=url)

    def _init_embed(self):
        base_url = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")
        api_key = os.environ.get("YUNWU_API_KEY")
        model = os.environ.get("YUNWU_EMBED_MODEL", "text-embedding-3-large")
        if not api_key:
            raise ValueError("Missing YUNWU_API_KEY")
        self.embed_model = OpenAIEmbedding(model=model, api_base=base_url, api_key=api_key)

    def get_embedding(self, text: str) -> List[float]:
        return self.embed_model.get_text_embedding(text)

    def search_qdrant(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """在Qdrant中进行语义搜索，返回相关的实体ID"""
        try:
            results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit
            )
            # 过滤出 entity 类型的记录
            entities = []
            for point in results:
                payload = point.payload or {}
                if payload.get("kind") == "entity":
                    entities.append({
                        "id": payload.get("entity_id"),
                        "text": payload.get("text"),
                        "score": point.score
                    })
            return entities
        except Exception as e:
            print(f"Qdrant search failed: {e}")
            return []

    def search_neo4j_context(self, entity_ids: List[str]) -> Dict[str, Any]:
        """
        在Neo4j中查询:
        1. 实体所属的社区及其摘要
        2. 实体的直接邻居关系（图推理/路径）
        """
        if not entity_ids:
            return {"communities": [], "relations": []}

        with self.driver.session(database=self.db_name) as session:
            # 1. 查询社区摘要 (Community Retrieval)
            comm_query = """
            MATCH (e:Entity)<-[:HAS_MEMBER]-(c:Community)
            WHERE e.id IN $ids
            RETURN DISTINCT c.cid AS cid, c.summary AS summary, collect(e.name) AS members
            LIMIT 3
            """
            comm_res = session.run(comm_query, ids=entity_ids)
            communities = [
                {"id": r["cid"], "summary": r["summary"], "members": r["members"]}
                for r in comm_res
            ]

            # 2. 查询图路径/关系 (Graph Traversal)
            # 查询这些实体作为头节点或尾节点的关系
            rel_query = """
            MATCH (s:Entity)-[r]->(o:Entity)
            WHERE s.id IN $ids OR o.id IN $ids
            RETURN s.name AS source, type(r) AS rel, o.name AS target
            LIMIT 20
            """
            rel_res = session.run(rel_query, ids=entity_ids)
            relations = [
                f"{r['source']} --[{r['rel']}]--> {r['target']}"
                for r in rel_res
            ]

            return {"communities": communities, "relations": relations}

    def search(self, query: str) -> str:
        """
        执行完整的 Graph RAG 检索流程
        """
        print(f"🔍 正在检索: {query}")
        
        # 1. 向量化查询
        query_vec = self.get_embedding(query)
        
        # 2. 语义检索 (Vector Search)
        top_entities = self.search_qdrant(query_vec, limit=5)
        if not top_entities:
            return "❌ 未在知识库中找到相关实体。"
            
        entity_ids = [e["id"] for e in top_entities]
        
        # 3. 图结构与社区检索 (Graph & Community Search)
        context = self.search_neo4j_context(entity_ids)
        
        # 4. 格式化结果 (Plain Text)
        lines = []
        lines.append(f"检索报告: {query}")
        lines.append("=" * 50)
        
        lines.append(f"\n[核心实体]")
        for e in top_entities:
            score = e.get('score', 0)
            lines.append(f"- {e['text']} (相似度: {score:.4f})")
        
        if context["communities"]:
            lines.append(f"\n[社区洞察]")
            for c in context["communities"]:
                summary = c["summary"] or "暂无摘要"
                members = ", ".join(c["members"][:5])
                lines.append(f"社区 {c['id']}:")
                lines.append(f"  摘要: {summary}")
                lines.append(f"  核心成员: {members} 等...")
        
        if context["relations"]:
            lines.append(f"\n[关联路径]")
            for r in context["relations"]:
                lines.append(f"- {r}")
            
        return "\n".join(lines)

# 创建全局实例
_searcher = GraphRAGSearcher()

def graph_rag_tool_function(query: str) -> str:
    """
    使用 Graph RAG 技术查询知识图谱。
    当用户询问有关特定实体、技术概念、关系或社区摘要时使用此工具。
    该工具结合了向量相似度搜索和图数据库结构搜索。
    
    Args:
        query (str): 用户的自然语言查询字符串。
    """
    return _searcher.search(query)

# 构建 LlamaIndex Tool
search_tool = FunctionTool.from_defaults(
    fn=graph_rag_tool_function,
    name="GraphKnowledgeSearch",
    description="用于在知识图谱中检索实体、关系和社区摘要的智能工具。支持语义搜索和图关联推理。"
)

def main():
    """
    交互式查询主函数
    """
    print("=" * 60)
    print("🚀 Graph RAG 知识图谱检索系统")
    print("=" * 60)
    print("提示: 输入 'exit' 或 'quit' 退出程序\n")
    
    try:
        searcher = GraphRAGSearcher()
        print("✅ 系统初始化成功！\n")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    while True:
        try:
            query = input("\n💬 请输入你的问题: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit', '退出']:
                print("\n👋 再见！")
                break
            
            result = searcher.search(query)
            print("\n" + result + "\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 查询出错: {e}\n")

if __name__ == "__main__":
    main()


