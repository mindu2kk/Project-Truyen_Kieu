# -*- coding: utf-8 -*-
"""
FAQ lookup: nếu câu hỏi khớp các mẫu đơn giản -> trả thẳng (không gọi RAG).
"""
import json, re
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any, List

@lru_cache(maxsize=1)
def _load_facts() -> List[Dict[str, Any]]:
    p = Path(__file__).resolve().parents[1] / "data" / "faq" / "facts.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def lookup_faq(query: str) -> Optional[Dict[str, Any]]:
    q = _norm(query)
    for item in _load_facts():
        for pat in item["patterns"]:
            if _norm(pat) in q:
                return {"answer": item["answer"], "sources": item.get("sources", [])}
    return None
