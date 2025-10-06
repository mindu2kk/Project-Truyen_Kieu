# -*- coding: utf-8 -*-
"""
Map-Reduce synthesis: tóm tắt từng đoạn -> hợp nhất -> câu trả lời gọn, có chiều sâu.
"""
from typing import List, Dict, Any
from generation import generate_answer_gemini

TEMPLATE_SUMMARIZE = """[CTX]
{ctx}
[INSTRUCTION]
- Trích các ý trực tiếp trả lời câu hỏi: "{query}".
- Nêu 1–2 dẫn chứng (ngắn gọn).
- Viết 3–5 gạch đầu dòng, mỗi gạch ≤ 2 câu, ≤80 từ.
"""

TEMPLATE_MERGE = """[NOTES]
{notes}

[INSTRUCTION]
- Hợp nhất ý, bỏ trùng lặp; chỉ giữ nội dung liên quan trực tiếp câu hỏi.
- Nêu mối quan hệ/diễn giải thêm nếu có trong NOTES.
- Văn phong học thuật, ≤180 từ.
"""

def map_reduce_answer(query: str, ctx_list: List[Dict[str, Any]], model: str = "gemini-2.0-flash") -> str:
    if not ctx_list:
        return "Hiện chưa đủ căn cứ trong kho tri thức để trả lời chính xác."
    notes = []
    for c in ctx_list:
        notes.append(
            generate_answer_gemini(
                TEMPLATE_SUMMARIZE.format(ctx=c["text"], query=query),
                model=model
            )
        )
    merged = generate_answer_gemini(
        TEMPLATE_MERGE.format(notes="\n\n---\n\n".join(notes)),
        model=model
    )
    return merged.strip()
