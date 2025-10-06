# app/poem_tools.py
# -*- coding: utf-8 -*-
"""
Tiện ích trích NGUYÊN VĂN Truyện Kiều theo số dòng.
- Ưu tiên đọc: data/interim/poem/poem.txt (mỗi câu 1 dòng)
- Fallback: ghép từ data/rag_chunks/ type=poem (nếu có), độ chính xác kém hơn.

API:
- poem_ready() -> bool
- get_opening(n: int) -> list[str]
- get_range(a: int, b: int) -> list[str]
"""

from __future__ import annotations
import re
from pathlib import Path
from functools import lru_cache

# app/poem_tools.py => parents[1] = project root "kieu-bot"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POEM_TXT   = PROJECT_ROOT / "data" / "interim" / "poem" / "poem.txt"
CHUNK_DIR  = PROJECT_ROOT / "data" / "rag_chunks"

def _strip_leading_number(s: str) -> str:
    # Bỏ "1: ", "001. ", "1) " đầu dòng nếu có
    return re.sub(r"^\s*\d{1,4}\s*[:\.\)]\s*", "", s).strip()

def _clean_line(s: str) -> str:
    s = s.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _looks_like_verse(s: str) -> bool:
    if not s:
        return False
    return bool(re.search(r"[A-Za-zÀ-ỹ]", s))

def _read_poem_txt() -> list[str]:
    """
    Đọc file poem.txt. Yêu cầu: mỗi câu 1 dòng. Có thể có số thứ tự đầu dòng.
    Trả về danh sách các dòng (không rỗng, đã làm sạch).
    """
    if not POEM_TXT.exists():
        return []
    lines = []
    for raw in POEM_TXT.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = _clean_line(_strip_leading_number(raw))
        if ln and _looks_like_verse(ln):
            lines.append(ln)
    return lines

def _read_poem_from_chunks() -> list[str]:
    """
    Fallback: nếu chưa có poem.txt thì thử ráp từ các chunk type=poem
    (Không đảm bảo đủ/đúng 3254 câu và thứ tự hoàn hảo, chỉ dùng tạm.)
    """
    if not CHUNK_DIR.exists():
        return []
    verses = []
    for p in CHUNK_DIR.glob("*.txt"):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if not raw.startswith("###META###"):
            continue
        meta_line, _, body = raw.partition("\n")
        m = re.search(r'"type"\s*:\s*"([^"]+)"', meta_line)
        if not m or m.group(1) != "poem":
            continue
        for rawln in body.splitlines():
            ln = _clean_line(_strip_leading_number(rawln))
            if ln and _looks_like_verse(ln):
                verses.append(ln)
    return verses

@lru_cache(maxsize=1)
def _load_poem_lines() -> list[str]:
    lines = _read_poem_txt()
    if not lines:
        lines = _read_poem_from_chunks()
    cleaned = []
    for ln in lines:
        if ln.strip() in {"I", "II", "III", "IV"}:
            continue
        if not cleaned or cleaned[-1] != ln:
            cleaned.append(ln)
    return cleaned

def poem_ready() -> bool:
    """Có thơ để trích chưa? (>=100 dòng coi như sẵn sàng)"""
    return len(_load_poem_lines()) >= 100

def get_opening(n: int) -> list[str]:
    """Lấy N câu đầu (1-based)."""
    lines = _load_poem_lines()
    n = max(1, min(n, len(lines)))
    return lines[:n]

def get_range(a: int, b: int) -> list[str]:
    """Lấy các câu [a..b] (1-based, inclusive)."""
    lines = _load_poem_lines()
    if not lines:
        return []
    if a > b:
        a, b = b, a
    a = max(1, a)
    b = min(b, len(lines))
    return lines[a-1:b]
