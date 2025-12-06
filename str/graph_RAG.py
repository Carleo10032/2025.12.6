import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import click
from neo4j import GraphDatabase
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import ChatMessage, MessageRole

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct, VectorParams, Distance
except Exception:
    QdrantClient = None
    PointStruct = None
    VectorParams = None
    Distance = None

from neo4j_ingest import load_env as load_env_file
try:
    import networkx as nx
except Exception:
    nx = None


def build_llm(model: str = None, temperature: float = 0.1) -> LlamaOpenAI:
    base_url = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")
    api_key = os.environ.get("YUNWU_API_KEY")
    if not api_key:
        raise RuntimeError("未找到YUNWU_API_KEY")
    mdl = model or os.environ.get("YUNWU_MODEL", "gpt-4o-mini")
    llm = LlamaOpenAI(model=mdl, api_key=api_key, api_base=base_url, temperature=temperature)
    Settings.llm = llm
    return llm


def build_embed_model(model: str = None) -> OpenAIEmbedding:
    base_url = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")
    api_key = os.environ.get("YUNWU_API_KEY")
    if not api_key:
        raise RuntimeError("未找到YUNWU_API_KEY")
    mdl = model or os.environ.get("YUNWU_EMBED_MODEL", "text-embedding-3-large")
    return OpenAIEmbedding(model=mdl, api_base=base_url, api_key=api_key)


def ensure_gds_graph(session, graph_name: str) -> None:
    try:
        session.run("CALL gds.graph.drop($g) YIELD graphName", g=graph_name)
    except Exception:
        pass
    try:
        session.run(
            """
            CALL gds.graph.project.cypher(
              $g,
              'MATCH (n:Entity) RETURN id(n) AS id',
              'MATCH (n:Entity)-[r]->(m:Entity) RETURN id(n) AS source, id(m) AS target, type(r) AS type'
            ) YIELD graphName
            """,
            g=graph_name,
        )
    except Exception:
        pass


def run_community_detection(session, graph_name: str, algo: str) -> Dict[int, List[Dict[str, Any]]]:
    try:
        if algo == "label":
            res = session.run(
                """
                CALL gds.labelPropagation.stream($g)
                YIELD nodeId, communityId
                RETURN communityId AS cid, gds.util.asNode(nodeId).id AS id, gds.util.asNode(nodeId).name AS name
                """,
                g=graph_name,
            )
        else:
            res = session.run(
                """
                CALL gds.louvain.stream($g)
                YIELD nodeId, communityId
                RETURN communityId AS cid, gds.util.asNode(nodeId).id AS id, gds.util.asNode(nodeId).name AS name
                """,
                g=graph_name,
            )
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for r in res:
            cid = r["cid"]
            groups.setdefault(cid, []).append({"id": r["id"], "name": r["name"]})
        return groups
    except Exception:
        if nx is None:
            raise
        nodes = {}
        for r in session.run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name"):
            nodes[r["id"]] = r["name"]
        g = nx.Graph()
        for nid, name in nodes.items():
            g.add_node(nid, name=name)
        for r in session.run("MATCH (s:Entity)-[r]->(o:Entity) RETURN s.id AS sid, o.id AS oid"):
            g.add_edge(r["sid"], r["oid"])
        comms = list(nx.algorithms.community.label_propagation_communities(g))
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for i, c in enumerate(comms):
            groups[i] = [{"id": nid, "name": nodes.get(nid)} for nid in c]
        return groups


def fetch_edges_for_group(session, ids: List[str], limit: int = 50) -> List[Tuple[str, str, str]]:
    res = session.run(
        """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE s.id IN $ids AND o.id IN $ids
        RETURN s.name AS s, type(r) AS t, o.name AS o
        LIMIT $lim
        """,
        ids=ids,
        lim=limit,
    )
    return [(r["s"], r["t"], r["o"]) for r in res]


def summarize_group(llm: LlamaOpenAI, members: List[Dict[str, Any]], edges: List[Tuple[str, str, str]]) -> str:
    print(f"  > 正在生成摘要 (实体数: {len(members)}, 关系数: {len(edges)})...")
    names = ", ".join([m.get("name") or m.get("id") for m in members])
    rels = "; ".join([f"{s} {t} {o}" for s, t, o in edges])
    sys = "你是知识图谱社区摘要助手。只返回简洁中文摘要。"
    usr = f"社区包含实体: {names}\n关系示例: {rels}\n请生成该社区的核心主题摘要,不超过120字。"
    msg = llm.chat(messages=[
        ChatMessage(role=MessageRole.SYSTEM, content=sys),
        ChatMessage(role=MessageRole.USER, content=usr),
    ])
    return getattr(msg, "text", None) or getattr(getattr(msg, "message", None), "content", "") or str(msg)


def write_community(session, cid: int, summary: str, members: List[Dict[str, Any]]) -> None:
    print(f"  > 写入社区 {cid} 到数据库...")
    session.run("MERGE (c:Community {cid:$cid}) SET c.size=$sz, c.summary=$s", cid=cid, sz=len(members), s=summary)
    session.run(
        """
        UNWIND $members AS m
        MATCH (e:Entity {id:m.id})
        MERGE (c:Community {cid:$cid})
        MERGE (c)-[:HAS_MEMBER]->(e)
        """,
        members=members,
        cid=cid,
    )


def collect_items(session, limit: int = 1000) -> List[Dict[str, Any]]:
    print("正在收集图谱数据用于向量化...")
    items: List[Dict[str, Any]] = []
    for r in session.run(f"MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.layer AS layer LIMIT {limit}"):
        text = f"实体 {r['name']} 层级 {r['layer'] or ''}".strip()
        items.append({"key": f"entity:{r['id']}", "text": text, "kind": "entity", "entity_id": r["id"]})
    
    rel_limit = limit - len(items)
    if rel_limit > 0:
        for r in session.run(
            f"""
            MATCH (s:Entity)-[r]->(o:Entity)
            RETURN s.id AS sid, s.name AS sname, type(r) AS t, o.id AS oid, o.name AS oname, coalesce(r.predicate, type(r)) AS pred
            LIMIT {rel_limit}
            """
        ):
            text = f"关系 {r['sname']} {r['pred']} {r['oname']}"
            key = f"rel:{r['sid']}->{r['t']}->{r['oid']}"
            items.append({"key": key, "text": text, "kind": "rel", "source": r["sid"], "target": r["oid"], "type": r["t"]})
    print(f"共收集到 {len(items)} 条数据 (实体+关系)")
    return items


def upsert_qdrant(items: List[Dict[str, Any]], embed_model: OpenAIEmbedding) -> Dict[str, Any]:
    if QdrantClient is None:
        raise RuntimeError("缺少qdrant依赖")
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    collection = os.environ.get("QDRANT_COLLECTION", "kg_items")
    client = QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)
    
    total = len(items)
    batch_size = 10
    print(f"开始写入 Qdrant (总数: {total}, 批次大小: {batch_size})...")
    
    # Check collection
    try:
        client.get_collection(collection)
    except Exception:
        # Create with dummy vector to init, will be updated
        # Need one vector to know dim? No, execute first batch to get dim
        pass

    processed = 0
    dim = 0
    
    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        texts = [it["text"] for it in batch]
        vectors = embed_model.get_text_embedding_batch(texts)
        
        if not vectors:
            continue
            
        current_dim = len(vectors[0])
        if dim == 0:
            dim = current_dim
            try:
                client.get_collection(collection)
            except Exception:
                print(f"创建集合 {collection} (维度: {dim})")
                client.create_collection(collection_name=collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

        points = [PointStruct(id=processed + j, vector=vectors[j], payload=batch[j]) for j in range(len(batch))]
        client.upsert(collection_name=collection, points=points)
        processed += len(batch)
        print(f"  已处理 {processed}/{total}")

    return {"collection": collection, "count": processed, "dim": dim}



def upsert_neo4j_embeddings(session, items: List[Dict[str, Any]], embed_model: OpenAIEmbedding) -> Dict[str, Any]:
    texts = [it["text"] for it in items]
    vectors = embed_model.get_text_embedding_batch(texts)
    updated = 0
    for it, vec in zip(items, vectors):
        if it["kind"] == "entity":
            session.run("MATCH (e:Entity {id:$id}) SET e.embedding=$v", id=it["entity_id"], v=vec)
        else:
            session.run(
                """
                MATCH (s:Entity {id:$sid})-[r]->(o:Entity {id:$oid})
                WHERE type(r)=$t
                SET r.embedding=$v
                """,
                sid=it["source"],
                oid=it["target"],
                t=it["type"],
                v=vec,
            )
        updated += 1
    return {"updated": updated, "dim": (len(vectors[0]) if vectors else 0)}


@click.command()
@click.option("--env", "env_path", default=Path(".env"), type=click.Path(path_type=Path))
@click.option("--algo", default="louvain", type=click.Choice(["louvain", "label"]))
@click.option("--graph", "graph_name", default="kg")
@click.option("--summarize", is_flag=True)
@click.option("--vectorstore", default="none", type=click.Choice(["none", "qdrant", "neo4j"]))
@click.option("--edge-limit", default=50, type=int)
@click.option("--embed-model", default=None)
@click.option("--llm-model", default=None)
def main(env_path: Path, algo: str, graph_name: str, summarize: bool, vectorstore: str, edge_limit: int, embed_model: str, llm_model: str):
    load_env_file(env_path)
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DB")
    if not password:
        raise RuntimeError("缺少 NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        ensure_gds_graph(session, graph_name)
        groups = run_community_detection(session, graph_name, algo)
        
        # 限制处理的社区数量以提高速度
        MAX_COMMUNITIES = 3
        sorted_cids = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)[:MAX_COMMUNITIES]
        
        print(f"发现 {len(groups)} 个社区，将处理最大的 {len(sorted_cids)} 个...")

        manifest: Dict[str, Any] = {"communities": []}
        if summarize:
            llm = build_llm(model=llm_model)
            for i, cid in enumerate(sorted_cids):
                members = groups[cid]
                print(f"处理社区 {cid} ({i+1}/{len(sorted_cids)})...")
                ids = [m["id"] for m in members]
                edges = fetch_edges_for_group(session, ids, edge_limit)
                summary = summarize_group(llm, members, edges)
                write_community(session, cid, summary, members)
                manifest["communities"].append({"cid": cid, "size": len(members), "summary": summary})
        
        # 限制向量化的数量
        items = collect_items(session, limit=200) # 限制 200 个用于演示/测试
        if vectorstore != "none":
            embed = build_embed_model(model=embed_model)
            if vectorstore == "qdrant":
                info = upsert_qdrant(items, embed)
                manifest["vectorstore"] = {"target": "qdrant", **info}
            else:
                info = upsert_neo4j_embeddings(session, items, embed)
                manifest["vectorstore"] = {"target": "neo4j", **info}
        print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
