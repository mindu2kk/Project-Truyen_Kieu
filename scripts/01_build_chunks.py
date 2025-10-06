# scripts/01_build_chunks.py
# -*- coding: utf-8 -*-
"""
Make RAG chunks from TXT files and gắn meta.tags tự động.

Input (TXT only):
  data/interim/**.txt
  ├─ poem/       # thơ (Truyện Kiều) – mỗi câu 1 dòng càng tốt
  ├─ analysis/   # bình giảng, nghị luận
  ├─ summary/    # tóm tắt, dàn ý
  └─ bio/        # tiểu sử, bối cảnh

Output:
  data/rag_chunks/*.txt   (mỗi chunk 1 file, dòng đầu là ###META### {json})

Meta bổ sung:
  meta.tags: danh sách tag phẳng, ví dụ:
    ["char:thuy_kieu", "char:thuy_van", "device:uoc_le", "theme:tai_menh", "section:trao_duyen"]
"""

from pathlib import Path
import re, json, unicodedata, hashlib

# ==== Config đường dẫn ====
SRC = Path("data/interim")
DST = Path("data/rag_chunks"); DST.mkdir(parents=True, exist_ok=True)

# Kích thước chunk
PROSE_MAX_WORDS = 220          # văn xuôi/bình giảng: 150–250 từ
POEM_LINES_PER_BLOCK = (2, 4)  # khuyến nghị: 2–4 câu/khổ
POEM_OVERLAP_LINES = 1         # overlap 1–2 câu để giữ mạch

# map thư mục -> type
TYPE_BY_DIR = {
    "poem": "poem",
    "analysis": "analysis",
    "summary": "summary",
    "bio": "bio",
}

# ==== Từ khoá/regex để gắn TAGS ====
# Lưu ý: dùng re.I (không phân biệt hoa/thường); có dấu/không dấu đều khớp tốt trên phần lớn nguồn.
CHAR_PAT = {
    "char:thuy_kieu":  [r"\bthúy?\s*kiều\b", r"\bvu(o|ơ)ng\s*thúy?\s*kiều\b"],
    "char:thuy_van":   [r"\bthúy?\s*vân\b"],
    "char:kim_trong":  [r"\bkim\s*trọng\b"],
    "char:tu_hai":     [r"\bt(ừ|u)\s*h(ả|a)i\b"],
    "char:hoan_thu":   [r"\bhoạn?\s*th(ư|u)\b"],
    "char:ma_giam_sinh":[r"\bm(ã|a)\s*gi(á|a)m\s*sinh\b"],
    "char:so_khanh":   [r"\bs(ở|o)\s*khanh\b"],
    "char:tu_ba":      [r"\bt(ú|u)\s*b(à|a)\b"],
    "char:giac_duyen": [r"\bgi(á|a)c\s*duy(ê|e)n\b"],
    "char:dam_tien":   [r"\b(đ|d)ạm\s*ti(ê|e)n\b"],
    "char:vuong_quan": [r"\bv(ư|u)o(ng)?\s*quan\b"],
}

DEVICE_PAT = {
    "device:uoc_le":        [r"\bước\s*lệ\b", r"\b(ước\s*lệ|tượng\s*trưng)\b"],
    "device:dien_co":       [r"\bđi(ể|e)n\s*c(ố|o)\b", r"\bđi(ể|e)n\s*t(ích|ich)\b"],
    "device:an_du":         [r"\bẩn\s*d(ụ|u)\b"],
    "device:nhan_hoa":      [r"\bnhân\s*h(ó|o)a\b"],
    "device:ta_canh_ngu_tinh":[r"\bt(ả|a)\s*c(ả|a)nh\s*ng(ụ|u)\s*t(ì|i)nh\b"],
    "device:phung_du":      [r"\bphúng\s*d(ụ|u)\b"],
    "device:cuc_ta":        [r"\bc(ư|u)ờng\s*đi(ễ|e)m\b", r"\bph(ó|o)ng\s*đ(ạ|a)i\s*(ch(ử|u)|ph(á|a)p)\b"],
}

THEME_PAT = {
    "theme:tai_menh":   [r"\bt(à|a)i\s*m(ệ|e)nh\b", r"\bt(à|a)i\s*v(ậ|a)n\b"],
    "theme:chu_tam":    [r"\bch(ữ|u)\s*t(â|a)m\b"],
    "theme:nhan_dao":   [r"\bnh(â|a)n\s*đ(ạ|a)o\b"],
    "theme:tinh_yeu":   [r"\bt(ì|i)nh\s*y(ê|e)u\b", r"\bd(uy|uy)ên\b"],
    "theme:so_phan":    [r"\bs(ố|o)\s*ph(ậ|a)n\b", r"\bb(ạ|a)c\s*m(ệ|e)nh\b"],
    "theme:gia_bien":   [r"\bgia\s*bi(ế|e)n\b"],
}

SECTION_PAT = {
    "section:trao_duyen":        [r"\btrao\s*duy(ê|e)n\b"],
    "section:chi_khi_anh_hung":  [r"\bch(í|i)\s*kh(í|i)\s*anh\s*h(ù|u)ng\b"],
    "section:bao_an_bao_oan":    [r"\bb(á|a)o\s*(â|a)n\b.*b(á|a)o\s*(o|ô)an\b", r"\bth(ú|u)y\s*ki(ề|e)u\s*b(á|a)o\s*(â|a)n\b"],
    "section:gap_kim_trong":     [r"\bg(ặ|a)p\s*kim\s*tr(ọ|o)ng\b"],
    "section:khoc_duong_truong": [r"\b(đ|d)o(à|a)n\s*tr(ư|u)ờng\s*t(â|a)n\s*th(à|a)nh\b"],  # tên gốc Trung Hoa
}

# ==== Helpers ====
def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\u00a0", " ")
    # gọn khoảng trắng
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def read_txt(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def file_type(fp: Path) -> str:
    parts = list(fp.relative_to(SRC).parts)
    return TYPE_BY_DIR.get(parts[0], "analysis")

def make_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

# ==== Tag extractor ====
def _find_any(patterns, text):
    return any(re.search(p, text, flags=re.I) for p in patterns)

def extract_tags(text: str, ftype: str) -> list[str]:
    """
    QUA QUY TẮC:
      - char:* nếu xuất hiện tên nhân vật
      - device:* nếu có từ khóa thủ pháp
      - theme:* chủ đề lớn
      - section:* tên đoạn/nhan đề quen thuộc
      - type:* luôn thêm type (poem/analysis/summary/bio)
    """
    tags: set[str] = set()
    # dấu hiệu theo danh mục
    for tag, pats in CHAR_PAT.items():
        if _find_any(pats, text):
            tags.add(tag)
    for tag, pats in DEVICE_PAT.items():
        if _find_any(pats, text):
            tags.add(tag)
    for tag, pats in THEME_PAT.items():
        if _find_any(pats, text):
            tags.add(tag)
    for tag, pats in SECTION_PAT.items():
        if _find_any(pats, text):
            tags.add(tag)

    # “bonus” heuristic: nếu nhắc cả “hoa” và “liễu” trong cùng khối văn xuôi → uớc lệ
    if re.search(r"\bhoa\b", text, flags=re.I) and re.search(r"\bli(ễ|e)u\b", text, flags=re.I):
        tags.add("device:uoc_le")

    tags.add(f"type:{ftype}")
    return sorted(tags)

# ==== Chunkers ====
def split_poem(text: str) -> list[str]:
    """Chia theo 2–4 câu lục bát, có overlap để giữ mạch."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blocks = []
    i = 0
    minL, maxL = POEM_LINES_PER_BLOCK
    while i < len(lines):
        L = min(maxL, max(minL, maxL))  # mặc định dùng 4 nếu set (2,4)
        blk = "\n".join(lines[i:i+L])
        blocks.append(blk)
        i += (L - POEM_OVERLAP_LINES)
    return blocks

def split_prose(text: str, max_words=PROSE_MAX_WORDS) -> list[str]:
    """Chia theo đoạn 150–250 từ, hạn chế xén giữa câu."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks, cur = [], ""
    for p in paras:
        if len((cur + " " + p).split()) <= max_words:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                blocks.append(cur); cur = p
            else:
                # đoạn quá dài → cắt mềm theo câu
                sentences = re.split(r"(?<=[\.\?\!…:;])\s+", p)
                buf = ""
                for s in sentences:
                    if len((buf + " " + s).split()) <= max_words:
                        buf = (buf + " " + s).strip()
                    else:
                        if buf: blocks.append(buf); buf = s
                        else:   blocks.append(s)
                if buf: blocks.append(buf)
                cur = ""
    if cur: blocks.append(cur)
    return blocks

def write_chunks(blocks: list[str], meta_base: dict, base_name: str):
    for idx, blk in enumerate(blocks):
        meta = dict(meta_base)
        meta["chunk_index"] = idx
        meta["id"] = f"{meta_base['source_id']}_{idx:04d}_{make_id(blk)[:6]}"
        # gắn TAGS (dựa trên chính nội dung chunk)
        meta["tags"] = extract_tags(blk, meta_base["type"])
        hdr = "###META### " + json.dumps(meta, ensure_ascii=False)
        outp = DST / f"{base_name}_{idx:04d}.txt"
        outp.write_text(hdr + "\n" + blk.strip() + "\n", encoding="utf-8")

# ==== Main ====
def main():
    files = list(SRC.rglob("*.txt"))
    if not files:
        raise SystemExit("❗ Không tìm thấy .txt trong data/interim/. Hãy đặt file TXT vào đó rồi chạy lại.")

    total_files, total_chunks = 0, 0
    for fp in files:
        raw = read_txt(fp)
        text = normalize_text(raw)
        ftype = file_type(fp)

        rel = fp.relative_to(SRC)
        base_name = fp.stem
        meta_base = {
            "source": str(rel).replace("\\", "/"),
            "source_id": base_name,
            "type": ftype,
            "title": base_name,
        }

        if ftype == "poem":
            blocks = split_poem(text)
        else:
            blocks = split_prose(text)

        write_chunks(blocks, meta_base, base_name)
        total_files += 1
        total_chunks += len(blocks)

    print(f"✔ Done. {total_files} file -> {total_chunks} chunks.")
    print(f"   Output: {DST.resolve()}")

if __name__ == "__main__":
    main()
