# app/rerank.py
# -*- coding: utf-8 -*-
"""
Reranker linh hoạt theo .env:
  RERANKER=none (mặc định, nhanh nhất)
  RERANKER=jina  (nhẹ hơn BGE)
  RERANKER=cohere (API rất nhanh, cần COHERE_API_KEY)
  RERANKER=bge   (chất lượng cao nhưng nặng ~2.2GB)
"""
import os
from typing import List, Dict, Any

MODE = os.getenv("RERANKER", "none").strip().lower()

_tok = None
_mdl = None
_bge = None

def _ensure_jina():
    global _tok, _mdl
    if _tok is None or _mdl is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tok = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v2-base-multilingual")
        _mdl = AutoModelForSequenceClassification.from_pretrained("jinaai/jina-reranker-v2-base-multilingual")
    return _tok, _mdl

def _ensure_bge():
    global _bge
    if _bge is None:
        from FlagEmbedding import FlagReranker
        _bge = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _bge

def rerank(query: str, hits: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    if not hits:
        return hits
    if MODE == "none":
        return hits[:top_k]

    if MODE == "cohere":
        import cohere, os
        co = cohere.Client(os.getenv("COHERE_API_KEY", ""))
        resp = co.rerank(model="rerank-multilingual-v3.0", query=query,
                         documents=[h["text"] for h in hits], top_n=top_k)
        idxs = [r.index for r in resp.results]
        return [hits[i] for i in idxs]

    if MODE == "jina":
        import torch
        tok, mdl = _ensure_jina()
        with torch.inference_mode():
            batch = tok([query]*len(hits), [h["text"] for h in hits],
                        return_tensors="pt", padding=True, truncation=True)
            scores = mdl(**batch).logits.squeeze(-1).tolist()
        for h, s in zip(hits, scores):
            h["re_score"] = float(s)
        hits.sort(key=lambda x: x.get("re_score", 0.0), reverse=True)
        return hits[:top_k]

    if MODE == "bge":
        rr = _ensure_bge()
        scores = rr.compute_score([(query, h["text"]) for h in hits], normalize=True)
        for h, s in zip(hits, scores):
            h["re_score"] = float(s)
        hits.sort(key=lambda x: x.get("re_score", 0.0), reverse=True)
        return hits[:top_k]

    return hits[:top_k]
