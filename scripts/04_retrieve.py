# scripts/04_retrieve.py
# -*- coding: utf-8 -*-
import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

MONGO_URI  = os.getenv("MONGO_URI")
DB_NAME    = os.getenv("MONGO_DB", "kieu_bot")
COL_NAME   = os.getenv("MONGO_COL", "chunks")
EMB_MODEL  = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")

@lru_cache(maxsize=1)
def _get_clients():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COL_NAME]
    embedder = SentenceTransformer(EMB_MODEL)  # load 1 lần
    return col, embedder

def retrieve_context(query: str, k: int = 5, filters=None, num_candidates: int = 100):
    """
    Vector search thuần: trả về top-k chunks theo cosine (via Atlas $vectorSearch).
    """
    col, embedder = _get_clients()
    qvec = embedder.encode("query: " + query, normalize_embeddings=True).tolist()
    stage = {
        "$vectorSearch": {
            "index": INDEX_NAME,
            "path": "vector",
            "queryVector": qvec,
            "numCandidates": num_candidates,   # ~100 là hợp lý
            "limit": k
        }
    }
    if filters:
        stage["$vectorSearch"]["filter"] = filters

    pipeline = [
        stage,
        {"$project": {"_id": 0, "text": 1, "meta": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    return list(col.aggregate(pipeline))

# --------- TÙY CHỌN: smart retrieve theo loại tài liệu ----------
def smart_retrieve(query: str, k: int = 4, num_candidates: int = 90):
    """
    Lấy kết quả từ nhiều 'type' rồi trộn theo điểm, ưu tiên loại phù hợp câu hỏi.
    Trả về đúng k kết quả cuối cùng.
    """
    q = query.lower()
    # Ưu tiên loại theo heuristic
    if any(w in q for w in ["định nghĩa", "là gì", "khái niệm", "thuật ngữ"]):
        order = [("term", 4), ("summary", 3), ("analysis", 4), ("bio", 2)]
    elif any(w in q for w in ["tóm tắt", "bao nhiêu", "bố cục", "thể thơ", "số câu"]):
        order = [("summary", 4), ("term", 3), ("analysis", 4), ("bio", 2)]
    elif any(w in q for w in ["nguyễn du", "nguyen du", "tiểu sử", "quê quán", "năm sinh", "bối cảnh"]):
        order = [("bio", 4), ("summary", 3), ("analysis", 4), ("term", 2)]
    else:
        order = [("analysis", 4), ("summary", 3), ("term", 2), ("bio", 2)]

    pool = []
    for t, kk in order:
        hits = retrieve_context(
            query, k=kk, num_candidates=num_candidates,
            filters={"meta.type": t}
        )
        pool.extend(hits)

    # dedupe theo (source, line_range) nếu có
    seen, uniq = set(), []
    for h in pool:
        sig = (h.get("meta", {}).get("source"), h.get("meta", {}).get("line_range"))
        if sig not in seen:
            seen.add(sig)
            uniq.append(h)

    uniq.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return uniq[:k]

