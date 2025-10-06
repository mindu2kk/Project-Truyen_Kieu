# scripts/20_make_sft_from_templates.py
# -*- coding: utf-8 -*-
"""
Sinh SFT JSONL từ template + tự gán context theo meta.tags trong data/rag_chunks.
Output: data/sft_json/instruct.jsonl
"""

import json, re, random
from pathlib import Path

RAG_DIR   = Path("data/rag_chunks")
OUT_PATH  = Path("data/sft_json"); OUT_PATH.mkdir(parents=True, exist_ok=True)
OUT_FILE  = OUT_PATH / "instruct.jsonl"

DEFAULT_SYSTEM = "Bạn là học giả Truyện Kiều. Văn phong trang trọng, mạch lạc, có dẫn chứng thơ khi thích hợp."

# ===== 1) KHAI BÁO TEMPLATE =====
# Mỗi mục: instruction, required_tags (ít nhất 1 tag), optional 'output_seed' (mẫu gợi ý)
TEMPLATES = [
    {
        "instruction": "Phân tích cái nhìn nhân đạo trong hình tượng Thúy Kiều (200–240 từ).",
        "required_tags": ["char:thuy_kieu", "theme:nhan_dao"],
        "output_seed": "Nguyễn Du nhìn con người từ lòng thương; “Chữ tâm kia mới bằng ba chữ tài” là kết luận đạo lý..."
    },
    {
        "instruction": "So sánh Thúy Vân – Thúy Kiều về vẻ đẹp và số phận (180–220 từ).",
        "required_tags": ["section:chi_em_thuy_kieu","device:uoc_le","char:thuy_kieu","char:thuy_van"],
        "output_seed": "Thúy Vân phúc hậu, điều hòa; Thúy Kiều sắc sảo, dữ dội — dự báo số phận khác nhau…"
    },
    {
        "instruction": "Phân tích hình tượng Từ Hải ở phương diện chí khí anh hùng (200–240 từ).",
        "required_tags": ["char:tu_hai","section:chi_khi_anh_hung"],
        "output_seed": "Chân dung phi phàm “râu hùm, hàm én, mày ngài”; lý tưởng nghĩa khí; tôn trọng Kiều…"
    },
    {
        "instruction": "Giải thích thủ pháp tả cảnh ngụ tình trong cảnh ngày xuân (150–190 từ).",
        "required_tags": ["device:ta_canh_ngu_tinh","section:canh_ngay_xuan"],
    },
    {
        "instruction": "Bình luận ý nghĩa câu “Chữ tâm kia mới bằng ba chữ tài” (130–170 từ).",
        "required_tags": ["theme:chu_tam"],
    },
    {
        "instruction": "Tóm tắt đoạn Trao duyên ~150 từ.",
        "required_tags": ["section:trao_duyen","char:thuy_kieu","char:thuy_van","char:kim_trong"],
    },
    {
        "instruction": "Vì sao Hoạn Thư vừa đáng trách vừa đáng nể? (150–190 từ).",
        "required_tags": ["char:hoan_thu"],
    },
    {
        "instruction": "Trình bày ngắn: giá trị hiện thực và giá trị nhân đạo của Truyện Kiều (120–160 từ).",
        "required_tags": ["theme:nhan_dao"],
    },
    # Mẫu “chống bịa” — context trống bắt buộc nói “chưa đủ căn cứ”
    {
        "instruction": "Liệt kê đầy đủ năm sinh, năm mất và quê quán của 10 nhà phê bình hiện đại viết về Truyện Kiều.",
        "required_tags": [],  # deliberately empty → có thể tạo bản ghi context rỗng
        "output_seed": "[CHỐNG BỊA] Nếu thiếu nguồn, phải nói 'chưa đủ căn cứ' và gợi ý bổ sung."
    },
]

# ===== 2) ĐỌC CHUNK & LỌC THEO TAG =====
def load_chunks():
    items=[]
    for p in RAG_DIR.glob("*.txt"):
        raw = p.read_text(encoding="utf-8")
        if not raw.startswith("###META###"): 
            continue
        meta_line, _, body = raw.partition("\n")
        try:
            meta = json.loads(meta_line.replace("###META###","").strip())
        except Exception:
            meta = {}
        tags = meta.get("tags") or []
        text = body.strip()
        items.append({"path": str(p), "meta": meta, "tags": set(tags), "text": text})
    return items

def pick_context(chunks, required_tags, max_blocks=2, max_len=900):
    """Chọn 1–2 chunk có giao tags lớn nhất; ghép làm context."""
    if not required_tags:
        return ""  # cho phép context rỗng để sinh mẫu 'chống bịa'
    req = set(required_tags)
    scored = []
    for c in chunks:
        score = len(req & c["tags"])
        if score>0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return ""
    chosen = [c for _, c in scored[:max_blocks]]
    ctx = "\n\n---\n\n".join(c["text"] for c in chosen)
    # chặn quá dài
    if len(ctx) > max_len:
        ctx = ctx[:max_len] + "…"
    return ctx

# ===== 3) LẮP THÀNH DÒNG JSONL =====
def make_records(chunks):
    out=[]
    for tpl in TEMPLATES:
        ctx = pick_context(chunks, tpl.get("required_tags", []))
        rec = {
            "system": DEFAULT_SYSTEM,
            "instruction": tpl["instruction"],
            "context": ctx,            # có thể rỗng
            "output": tpl.get("output_seed","").strip(),  # có thể trống; bạn có thể điền tay dần
            "tags": tpl.get("required_tags", [])
        }
        out.append(rec)
    return out

def main():
    chunks = load_chunks()
    assert chunks, "Chưa có chunk trong data/rag_chunks. Hãy chạy 01_build_chunks.py trước."
    recs = make_records(chunks)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✔ Wrote {len(recs)} records → {OUT_FILE}")

if __name__=="__main__":
    main()
