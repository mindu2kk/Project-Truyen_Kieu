# app/router.py
# -*- coding: utf-8 -*-
"""
Router: phân loại intent câu hỏi
- poem: yêu cầu trích NGUYÊN VĂN thơ theo số câu/khoảng (vd: "10 câu đầu", "câu 241-260", "trích 1-20", "từ câu 5 đến 18")
- chitchat: chào hỏi/ngắn/không nội dung học thuật
- generic: phép tính đơn giản, câu cực ngắn/siêu chung (không rõ thuộc miền Kiều)
- domain: mặc định -> hỏi kiến thức Truyện Kiều (RAG)
"""
from __future__ import annotations
import re

# ---------- Poem patterns ----------
# "10 câu đầu", "trích 10 câu đầu", "xin 12 câu mở đầu"
_OPENING_PAT = re.compile(
    r"\b(\d{1,3})\s*câu\s*(đầu|mở\s*đầu)\b", re.I | re.U
)

# "câu 241-260", "câu 1 – 20", "từ câu 5 đến câu 18", "câu 5 đến 18"
_RANGE_PATS = [
    re.compile(r"\bcâu\s*(\d{1,4})\s*[-–—]\s*(\d{1,4})\b", re.I | re.U),
    re.compile(r"\btừ\s*câu\s*(\d{1,4})\s*đến\s*câu\s*(\d{1,4})\b", re.I | re.U),
    re.compile(r"\bcâu\s*(\d{1,4})\s*đến\s*(\d{1,4})\b", re.I | re.U),
    # phòng TH người dùng gõ "lines 10-20"
    re.compile(r"\blines?\s*(\d{1,4})\s*[-–—]\s*(\d{1,4})\b", re.I | re.U),
]

# "trích 30 câu" (hiểu là 30 câu đầu nếu không chỉ rõ khoảng)
_TRICH_N_CAU_PAT = re.compile(r"\btr(í|i)ch\s*(\d{1,3})\s*câu\b", re.I | re.U)

# ---------- Chitchat / generic helpers ----------
_CHITCHAT_SET = {
    "hi", "hello", "xin chào", "chào", "chào bạn", "chào cậu", "yo", "hey", "alo"
}

_MATH_PAT = re.compile(r"^\s*\d+\s*([+\-*/x:])\s*\d+\s*(=\s*)?\??\s*$")

# Một số câu cực ngắn, không ràng buộc miền (có thể để generic)
_GENERIC_SHORT_PAT = re.compile(
    r"^(là gì|là ai|bao nhiêu|mấy giờ|ở đâu|tại sao|vì sao)\b", re.I | re.U
)

def _norm(s: str) -> str:
    return (s or "").strip()

# ---------- API chính ----------
def parse_poem_request(q: str):
    """
    Trả:
      ("opening", N)  -> N câu đầu
      ("range", A, B) -> từ câu A đến B (bao gồm)
      None            -> không phải yêu cầu trích thơ theo chỉ dẫn
    """
    qs = _norm(q).lower()

    # 10 câu đầu / 12 câu mở đầu...
    m = _OPENING_PAT.search(qs)
    if m:
        return ("opening", int(m.group(1)))

    # câu 241-260 / từ câu 5 đến câu 18 / lines 3-12 ...
    for pat in _RANGE_PATS:
        m = pat.search(qs)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return ("range", a, b)

    # trích 30 câu -> hiểu là opening nếu không chỉ rõ khoảng
    m = _TRICH_N_CAU_PAT.search(qs)
    if m:
        n = int(m.group(2))
        return ("opening", n)

    return None


def route_intent(q: str) -> str:
    qs = _norm(q).lower()
    if not qs:
        return "chitchat"

    # thơ ưu tiên bắt đầu tiên
    if parse_poem_request(q) is not None:
        return "poem"

    # chitchat: lời chào/ngắn
    if qs in _CHITCHAT_SET:
        return "chitchat"
    # câu rất ngắn, kết thúc bằng "không?" / "ko?" hay "à?"... coi như chitchat
    if len(qs) <= 20 and any(qs.endswith(suf) for suf in ("không?", "ko?", "à?", "hả?", "?")):
        return "chitchat"

    # bài toán cực cơ bản
    if _MATH_PAT.match(qs):
        return "generic"

    # câu cực ngắn dạng hỏi chung chung
    if len(qs) <= 30 and _GENERIC_SHORT_PAT.match(qs):
        return "generic"

    # mặc định: domain (Truyện Kiều / văn học)
    return "domain"
