# scripts/22_make_hard_negatives.py
# Tạo mẫu chống bịa: context rỗng hoặc sai mục tiêu → output phải "chưa đủ căn cứ..."
import json, random
from pathlib import Path

IN_FILE  = Path("data/sft_json/instruct_filled.jsonl")
OUT_FILE = Path("data/sft_json/instruct_augmented.jsonl")

NEG_PROMPTS = [
    "Hãy liệt kê chính xác năm xuất bản các ấn bản Truyện Kiều thời Nguyễn.",
    "Nêu địa chỉ lưu trữ nguyên bản Kim Vân Kiều truyện và số ký hiệu.",
    "So sánh cấu trúc 5 hồi của Hamlet với Truyện Kiều (chi tiết).",
    "Kể 10 học giả hiện đại viết về Kiều kèm năm sinh, năm mất, quê quán.",
]

REFUSAL = "Chưa đủ căn cứ trong kho ngữ liệu để trả lời chính xác. Vui lòng cung cấp nguồn/bằng chứng cụ thể."

def main():
    items = [json.loads(l) for l in open(IN_FILE,"r",encoding="utf-8") if l.strip()]
    # thêm 1 bản copy giữ nguyên
    out = items[:]
    # thêm hard negatives
    for q in NEG_PROMPTS:
        out.append({
            "system": "Bạn là học giả Truyện Kiều.",
            "instruction": q,
            "context": "",
            "output": REFUSAL,
            "tags": ["anti_hallu"]
        })
    with OUT_FILE.open("w","w",encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✔ Augmented → {OUT_FILE}")

if __name__=="__main__":
    main()
