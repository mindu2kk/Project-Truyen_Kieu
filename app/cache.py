# app/cache.py
# -*- coding: utf-8 -*-
import hashlib
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
col = client[os.getenv("MONGO_DB", "kieu_bot")]["cache"]

# TTL index trên trường datetime (bắt buộc dùng BSON Date)
try:
    col.create_index([("created_at", ASCENDING)], expireAfterSeconds=7*24*3600)
except Exception:
    pass

def _key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def get_cached(query: str) -> str | None:
    doc = col.find_one({"_id": _key(query)})
    return doc["answer"] if doc else None

def set_cached(query: str, answer: str):
    col.replace_one(
        {"_id": _key(query)},
        {"_id": _key(query), "answer": answer, "created_at": datetime.now(timezone.utc)},
        upsert=True
    )
