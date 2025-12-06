import os
import json
from pathlib import Path
from typing import Dict, Any, List

import click
from neo4j import GraphDatabase


def load_env(env_path: Path) -> None:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def ensure_constraints(session) -> None:
    # Entity 使用 id 作为唯一标识 (id = name.strip().lower())
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.file IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Block) REQUIRE (b.id, b.file) IS UNIQUE")


def clear_database(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def normalize_key(text: str) -> str:
    return text.strip().lower() if text else ""


def rel_type(pred: str) -> str:
    t = (pred or "RELATED_TO").upper()
    out = []
    for ch in t:
        if ch.isalnum():
            out.append(ch)
        elif ch in " -/":
            out.append("_")
    r = "".join(out).strip("_")
    return r or "RELATED_TO"


MATERIAL_KWS = {"材料", "material", "纳米", "碳", "锂", "铜", "钴", "镍", "氧化物", "聚合物", "graphene", "oxide"}
DEVICE_KWS = {"器件", "device", "电池", "battery", "电容", "capacitor", "memristor", "sensor"}
SYSTEM_KWS = {"系统", "system", "控制", "control", "network", "grid"}
APPLICATION_KWS = {"应用", "application", "场景", "scenario", "工业", "medical", "bio", "能源"}


def classify_layer(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in MATERIAL_KWS):
        return "material"
    if any(k in n for k in DEVICE_KWS):
        return "device"
    if any(k in n for k in SYSTEM_KWS):
        return "system"
    if any(k in n for k in APPLICATION_KWS):
        return "application"
    return "unknown"


def ingest_file(session, jf: Path, collapse_blocks: bool = False) -> Dict[str, Any]:
    data = json.loads(jf.read_text(encoding="utf-8"))
    triples: List[Dict[str, Any]] = data.get("triples") or []
    sf = data.get("source_file") or jf.name
    count = 0
    session.run("MERGE (d:Document {file:$sf})", sf=sf)
    tx = session.begin_transaction()
    for t in triples:
        s = t.get("subject")
        o = t.get("object")
        p = t.get("predicate")
        sid = t.get("source_text_id")
        if not s or not o:
            continue
        
        s = s.strip()
        o = o.strip()
        sid_key = normalize_key(s)
        oid_key = normalize_key(o)
        bid = "doc" if collapse_blocks else (sid or "doc")
        
        rt = rel_type(p)
        sl = classify_layer(s)
        ol = classify_layer(o)
        tx.run("MERGE (b:Block {id:$bid, file:$sf})", bid=bid, sf=sf)
        tx.run(
            "MERGE (d:Document {file:$sf}) MERGE (b:Block {id:$bid, file:$sf}) MERGE (d)-[:HAS_BLOCK]->(b)",
            sf=sf, bid=bid,
        )
        # 使用 id 合并，保留原始 name
        tx.run(
            "MERGE (s:Entity {id:$sid_key}) ON CREATE SET s.name=$s, s.layer=$sl SET s.layer = coalesce(s.layer,$sl)",
            sid_key=sid_key, s=s, sl=sl,
        )
        tx.run(
            "MERGE (o:Entity {id:$oid_key}) ON CREATE SET o.name=$o, o.layer=$ol SET o.layer = coalesce(o.layer,$ol)",
            oid_key=oid_key, o=o, ol=ol,
        )
        tx.run(
            f"MERGE (s:Entity {{id:$sid_key}}) MERGE (o:Entity {{id:$oid_key}}) MERGE (s)-[r:{rt}]->(o) ON CREATE SET r.predicate=$p, r.source_text_id=$sid, r.source_file=$sf",
            sid_key=sid_key, oid_key=oid_key, p=p, sid=sid, sf=sf,
        )
        tx.run(
            "MATCH (s:Entity {id:$sid_key}), (o:Entity {id:$oid_key}), (b:Block {id:$bid, file:$sf}) MERGE (s)-[:FROM]->(b) MERGE (o)-[:FROM]->(b)",
            sid_key=sid_key, oid_key=oid_key, bid=bid, sf=sf,
        )
        count += 1
    tx.commit()
    return {"file": jf.name, "imported": count}


@click.command()
@click.option("--input", "input_dir", required=True, type=click.Path(path_type=Path))
@click.option("--env", "env_path", default=Path(".env"), type=click.Path(path_type=Path))
@click.option("--clear", is_flag=True, help="Import前清空数据库")
@click.option("--collapse-blocks", is_flag=True, help="按文档合并Block为单一节点")
def main(input_dir: Path, env_path: Path, clear: bool, collapse_blocks: bool):
    load_env(env_path)
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    database = os.environ.get("NEO4J_DB")
    if not password:
        raise RuntimeError("缺少 NEO4J_PASSWORD,请在 .env 设置")
    files = sorted([p for p in input_dir.glob("*.triples.json")])
    if not files:
        raise RuntimeError("未找到 triples JSON")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        if clear:
            print("[*] 清空数据库...")
            clear_database(session)
        ensure_constraints(session)
        manifest = []
        for jf in files:
            manifest.append(ingest_file(session, jf, collapse_blocks=collapse_blocks))
    print(json.dumps({"items": manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()

