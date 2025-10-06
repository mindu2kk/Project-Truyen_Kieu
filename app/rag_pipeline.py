# app/rag_pipeline.py
# -*- coding: utf-8 -*-
"""
RAG pipeline (single-shot synthesis mặc định).
- retrieve_context từ scripts/04_retrieve.py
- rerank theo env (none/jina/cohere/bge)
- single-shot synthesis (1 call) để nhanh
- Hỗ trợ history_text để giữ mạch hội thoại
"""

from typing import List, Dict, Any
import importlib.util, sys
from pathlib import Path
from rerank import rerank
from generation import generate_answer_gemini

# ==== dynamic import scripts/04_retrieve.py ====
ROOT = Path(__file__).resolve().parents[1]
retr_path = ROOT / "scripts" / "04_retrieve.py"
spec = importlib.util.spec_from_file_location("retrieve_mod", retr_path)
retrieve_mod = importlib.util.module_from_spec(spec)
sys.modules["retrieve_mod"] = retrieve_mod
assert spec and spec.loader
spec.loader.exec_module(retrieve_mod)  # type: ignore
retrieve_context = retrieve_mod.retrieve_context

# ---- Prompt template cho single-shot
SYNTH_SINGLE_TEMPLATE = """{hist}[NGỮ CẢNH]
{ctx}

[HƯỚNG DẪN]
- Trả lời câu hỏi: "{query}" bằng 3–6 gạch đầu dòng, mỗi gạch ≤ 2 câu.
- Chỉ dùng thông tin trong NGỮ CẢNH; nếu thiếu căn cứ, nói 'chưa đủ căn cứ'.
- Nếu NGỮ CẢNH có thơ (type=poem), hãy **trích 1–2 câu thơ nguyên văn** làm dẫn chứng (đặt trong ngoặc kép).
- Không lan man. Giới hạn ~120–200 từ.
"""

def _format_sources(ctx_list: List[Dict[str, Any]]) -> str:
    srcs = []
    for c in ctx_list:
        s = c.get("meta", {}).get("source")
        if s and s not in srcs:
            srcs.append(s)
    return "; ".join(srcs)

def _build_single_shot_prompt(query: str,
                              ctx_list: List[Dict[str, Any]],
                              history_text: str | None = None) -> str:
    merged_ctx = "\n\n---\n\n".join(c["text"] for c in ctx_list)
    hist_block = f"[LỊCH SỬ NGẮN]\n{history_text}\n\n" if history_text else ""
    return SYNTH_SINGLE_TEMPLATE.format(ctx=merged_ctx, query=query, hist=hist_block)

def build_prompt(query: str, ctx_list: List[Dict[str, Any]],
                 history_text: str | None = None) -> str:
    # vẫn giữ để tương thích nếu nơi khác gọi
    return _build_single_shot_prompt(query, ctx_list, history_text)

def answer_question(
    query: str,
    k: int = 4,
    filters: Dict[str, Any] | None = None,
    num_candidates: int = 100,
    synthesize: str | bool = "single",       # "single" | "mapreduce" | False
    gen_model: str = "gemini-2.0-flash",
    force_quote: bool = False,
    long_answer: bool = False,
    history_text: str | None = None,
) -> Dict[str, Any]:
    if filters is None:
        filters = {"meta.type": {"$in": ["analysis", "poem", "summary", "bio"]}}

    # 1) Retrieve rộng vừa đủ
    hits = retrieve_context(query, k=max(k, 10), filters=filters, num_candidates=num_candidates)
    if not hits:
        prompt = build_prompt(query, [], history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 2) Độ tin cậy nhanh
    avg_score = sum(h.get("score", 0.0) for h in hits) / max(1, len(hits))
    if avg_score < 0.2:
        prompt = build_prompt(query, [], history_text)
        return {"query": query, "prompt": prompt, "contexts": []}

    # 3) Rerank theo env rồi lấy top-k
    hits = rerank(query, hits, top_k=k)

    # 4) Synthesize
    prompt = build_prompt(query, hits, history_text)
    result = {"query": query, "prompt": prompt, "contexts": hits}

    if synthesize and synthesize != "mapreduce":
        ans = generate_answer_gemini(prompt, model=gen_model, long_answer=long_answer)
    elif synthesize == "mapreduce":
        from synthesis import map_reduce_answer
        ans = map_reduce_answer(query, hits, model=gen_model)
    else:
        ans = None

    if ans:
        srcs = _format_sources(hits)
        if srcs:
            ans = f"{ans}\n\n**Nguồn:** {srcs}"
        result["answer"] = ans

    return result
