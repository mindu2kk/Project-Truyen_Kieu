# app/orchestrator.py
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple, Iterable
from router import route_intent, parse_poem_request
from rag_pipeline import answer_question
from generation import generate_answer_gemini
from faq import lookup_faq
from cache import get_cached, set_cached

# poem tools
try:
    from poem_tools import poem_ready, get_opening, get_range
    _POEM_AVAILABLE = True
except Exception:
    _POEM_AVAILABLE = False

SMALL_TALK_SYS = "Bạn là trợ lý thân thiện. Trả lời rất ngắn (≤ 2 câu), lịch sự."
GENERIC_SYS    = "Bạn là trợ lý kiến thức tổng quát. Trả lời chính xác, ngắn gọn, dễ hiểu."

def _wrap_user_prompt(system: str, user: str, history_text: str | None = None) -> str:
    hist = f"[HISTORY]\n{history_text}\n\n" if history_text else ""
    return f"[SYSTEM]\n{system}\n\n{hist}[USER]\n{user}"

def _norm_key(q: str) -> str:
    return (q or "").strip().lower()

def _sources_from_ctx(ctx: List[dict]) -> str:
    seen = []
    for c in (ctx or []):
        src = (c.get("meta") or {}).get("source")
        if src and src not in seen:
            seen.append(src)
    return "; ".join(seen)

def _history_to_text(history: Iterable[Tuple[str, str]] | None,
                     max_turns: int = 6, max_chars: int = 1200) -> str | None:
    """
    history: list of (role, text), role in {"user","assistant"}.
    Giữ tối đa max_turns cặp lượt, cắt gọn theo max_chars.
    """
    if not history:
        return None
    # lấy đoạn cuối
    pairs = []
    # gom thành các turn (user->assistant)
    buf = []
    for role, text in history:
        if role not in ("user", "assistant"):
            continue
        buf.append((role, text.strip()))
    # chỉ lấy đoạn cuối cùng trong giới hạn
    buf = buf[-(max_turns*2):]

    lines = []
    for role, text in buf:
        tag = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"{tag}: {text}")
    joined = "\n".join(lines)
    if len(joined) > max_chars:
        joined = "…\n" + joined[-max_chars:]
    return joined

def answer_with_router(query: str,
                       k: int = 4,
                       gemini_model: str = "gemini-2.0-flash",
                       long_answer: bool = False,
                       history: List[Tuple[str, str]] | None = None) -> Dict[str, Any]:
    """
    history: danh sách [(role, text), ...] từ UI để giữ mạch hội thoại ngắn hạn.
    """
    qkey = _norm_key(query)
    hist_text = _history_to_text(history)

    # 0) Cache theo câu hỏi gốc (đơn giản)
    cached = get_cached(qkey)
    if cached:
        return {"intent": "cache", "answer": cached, "sources": []}

    # 1) Định tuyến sớm để không “nuốt” yêu cầu trích thơ
    intent = route_intent(query)

    # 1.a) Poem-mode
    if intent == "poem":
        if not _POEM_AVAILABLE or not poem_ready():
            msg = "Kho thơ chưa sẵn sàng. Hãy đặt toàn văn vào data/interim/poem/poem.txt rồi chạy chunk + embed."
            set_cached(qkey, msg)
            return {"intent": "poem", "answer": msg, "sources": []}

        spec = parse_poem_request(query)
        if spec and spec[0] == "opening":
            n = max(1, min(int(spec[1]), 500))
            lines = get_opening(n)
            txt = "\n".join(f"{i+1:>4}: {ln}" for i, ln in enumerate(lines))
            ans = f"**{n} câu đầu Truyện Kiều:**\n\n{txt}"
            set_cached(qkey, ans)
            return {"intent": "poem", "answer": ans, "sources": ["poem"]}

        if spec and spec[0] == "range":
            a, b = int(spec[1]), int(spec[2])
            if a > b: a, b = b, a
            lines = get_range(a, b)
            txt = "\n".join(f"{a+i:>4}: {ln}" for i, ln in enumerate(lines))
            ans = f"**Các câu {a}–{b} trong Truyện Kiều:**\n\n{txt}"
            set_cached(qkey, ans)
            return {"intent": "poem", "answer": ans, "sources": ["poem"]}

        # không parse được: hỏi lại ngắn
        prompt = _wrap_user_prompt(
            "Bạn giúp người dùng trích thơ theo chỉ dẫn. Nếu họ chưa nêu rõ số dòng/khoảng, hãy hỏi lại rất ngắn.",
            query, history_text=hist_text
        )
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=False)
        set_cached(qkey, ans)
        return {"intent": "poem", "answer": ans, "sources": []}

    # 2) FAQ nhanh (không ảnh hưởng poem)
    hit = lookup_faq(query)
    if hit:
        ans = hit["answer"]
        srcs = hit.get("sources", [])
        if srcs: ans += "\n\n**Nguồn:** " + "; ".join(srcs)
        set_cached(qkey, ans)
        return {"intent": "faq", "answer": ans, "sources": srcs}

    # 3) Small talk / Generic
    if intent == "chitchat":
        prompt = _wrap_user_prompt(SMALL_TALK_SYS, query, history_text=hist_text)
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=False)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": []}

    if intent == "generic":
        prompt = _wrap_user_prompt(GENERIC_SYS, query, history_text=hist_text)
        ans = generate_answer_gemini(prompt, model=gemini_model, long_answer=False)
        set_cached(qkey, ans)
        return {"intent": intent, "answer": ans, "sources": []}

    # 4) Domain → RAG (single-shot; ép chèn 1–2 câu thơ nếu ngữ cảnh có 'poem')
    pack = answer_question(query, k=k, synthesize="single",
                           gen_model=gemini_model, force_quote=True,
                           long_answer=long_answer, history_text=hist_text)
    if "answer" in pack:
        ans = pack["answer"]
        ctx = pack.get("contexts") or []
        src_join = _sources_from_ctx(ctx)
        if src_join:
            ans = f"{ans}\n\n**Nguồn:** {src_join}"
        set_cached(qkey, ans)
        return {"intent": "domain", "answer": ans, "sources": ctx}

    # 5) Fallback
    ans = generate_answer_gemini(
        _wrap_user_prompt(GENERIC_SYS, query, history_text=hist_text),
        model=gemini_model, long_answer=long_answer
    )
    set_cached(qkey, ans)
    return {"intent": "domain", "answer": ans, "sources": []}
