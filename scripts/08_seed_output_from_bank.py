# scripts/21_seed_outputs_from_bank.py
# Ghép "output" chất lượng từ ngân hàng mẫu vào instruct.jsonl theo rule/regex.
import json, re
from pathlib import Path

IN_FILE  = Path("data/sft_json/instruct.jsonl")
OUT_FILE = Path("data/sft_json/instruct_filled.jsonl")

# Ngân hàng output mẫu (viết kỹ vài chục cái trước, rồi mở rộng)
BANK = [
  {
    "pattern": r"nhân đạo.*Thúy Kiều",
    "output": "Nguyễn Du nhìn con người từ lòng thương; với Kiều, ông đề cao 'chữ Tâm'..."
  },
  {
    "pattern": r"So sánh.*Thúy Vân.*Thúy Kiều",
    "output": "Thúy Vân phúc hậu, điều hòa — báo hiệu đời yên ả; Thúy Kiều sắc sảo, dữ dội — dự cảm truân chuyên..."
  },
  {
    "pattern": r"Từ Hải.*chí khí",
    "output": "Từ Hải là lý tưởng anh hùng: tầm vóc, đạo nghĩa, tôn trọng Kiều; 'gươm đàn nửa gánh, non sông một chèo'..."
  },
  {
    "pattern": r"tả cảnh ngụ tình.*ngày xuân",
    "output": "Bút pháp ngụ tình thấm vào phong cảnh 'cỏ non xanh...', 'cành lê trắng điểm'; cảnh – tình tương ứng..."
  },
  {
    "pattern": r"Chữ tâm.*ba chữ tài",
    "output": "Câu thơ kết tinh đạo lý: tâm đức vượt lên tài năng; ánh sáng 'tâm' hướng tài về điều thiện..."
  }
]

def pick_output(instruction: str) -> str|None:
    for b in BANK:
        if re.search(b["pattern"], instruction, flags=re.I):
            return b["output"]
    return None

def main():
    lines = [json.loads(l) for l in open(IN_FILE,"r",encoding="utf-8") if l.strip()]
    out=[]
    for r in lines:
        if not r.get("output"):
            o = pick_output(r.get("instruction",""))
            if o: r["output"] = o
        out.append(r)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✔ Wrote {len(out)} → {OUT_FILE}")

if __name__=="__main__":
    main()
