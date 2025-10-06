# scripts/02_embed_and_index_mongo.py
# -*- coding: utf-8 -*-
import os, json
from pathlib import Path
from typing import Iterator, List, Dict
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from sentence_transformers import SentenceTransformer

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB", "kieu_bot")
COL_NAME  = os.getenv("MONGO_COL", "chunks")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")  # <-- base
BATCH_SZ  = int(os.getenv("EMBED_BATCH_SIZE", "128"))
CHUNKS_DIR = Path("data/rag_chunks")

assert MONGO_URI, "Thiếu MONGO_URI trong .env"

client = MongoClient(MONGO_URI)
col = client[DB_NAME][COL_NAME]

col.create_index("meta.type")
col.create_index("meta.source")

embedder = SentenceTransformer(EMB_MODEL)

def embed_texts_passage(texts: List[str]) -> List[List[float]]:
    # E5-spec: prefix "passage: "
    texts2 = [("passage: " + t) for t in texts]
    return embedder.encode(
        texts2,
        normalize_embeddings=True,
        batch_size=BATCH_SZ,
        show_progress_bar=False
    ).tolist()

def iter_chunks() -> Iterator[Dict]:
    for p in sorted(CHUNKS_DIR.glob("*.txt")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw.startswith("###META###"):
            continue
        meta_line, _, body = raw.partition("\n")
        try:
            meta = json.loads(meta_line.replace("###META###","").strip())
        except Exception:
            meta = {}
        text = body.strip()
        if not text or len(text.split()) < 5:
            continue
        _id = meta.get("id") or p.stem
        yield {"_id": _id, "text": text, "meta": meta}

def batched(iterable, n=128):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    # In dim để so với Atlas index
    dim = len(embed_texts_passage(["probe"])[0])
    print(f"[INFO] EMBEDDING_MODEL={EMB_MODEL} | dim={dim} (e5-base=768) | batch={BATCH_SZ}")

    total = 0
    for batch in tqdm(batched(iter_chunks(), n=BATCH_SZ), desc="Embedding & upserting"):
        texts = [d["text"] for d in batch]
        vecs  = embed_texts_passage(texts)

        ops = []
        for d, v in zip(batch, vecs):
            d["vector"] = v
            ops.append(UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True))

        if ops:
            col.bulk_write(ops, ordered=False)
            total += len(ops)
    print(f"[DONE] Upsert {total} chunks → {DB_NAME}.{COL_NAME}")
    print("Nhớ đảm bảo Vector Index của Atlas có numDimensions = 768.")

if __name__ == "__main__":
    main()
