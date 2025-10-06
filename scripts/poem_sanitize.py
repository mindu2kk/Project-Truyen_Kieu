# -*- coding: utf-8 -*-
"""
Sanitize numbers in-place for data/interim/poem/poem.txt
- Bỏ dòng chỉ có số (đánh số câu / số trang)
- Bỏ số đầu dòng (vd: "12. ", "34: ", "56- ")
- Loại mọi chữ số còn lại trong dòng
- Gọn khoảng trắng, nén nhiều dòng trống thành 1
Usage:
  python scripts/poem_sanitize.py
  # hoặc chỉ định file khác:
  python scripts/poem_sanitize.py --file "data/interim/poem/poem.txt"
"""
import argparse, re, unicodedata, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "interim" / "poem" / "poem.txt"

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def save_text(p: Path, s: str):
    p.write_text(s, encoding="utf-8")

def backup_file(p: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bkp = p.with_suffix(".raw.txt")
    # nếu đã có .raw.txt thì tạo file theo timestamp
    if bkp.exists():
        bkp = p.with_suffix(f".backup-{ts}.txt")
    bkp.write_text(p.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    return bkp

def sanitize_numbers(raw: str) -> str:
    t = unicodedata.normalize("NFC", raw).replace("\u00A0", " ")
    t = re.sub(r"\r\n?", "\n", t)

    out = []
    removed_line_nums = 0
    for ln in t.splitlines():
        s = ln.strip()

        # giữ nguyên dòng trống (để ngắt khổ), xử lý sau
        if s == "":
            out.append("")
            continue

        # 1) bỏ dòng chỉ có số
        if re.fullmatch(r"\d+", s):
            removed_line_nums += 1
            continue

        # 2) xóa số đầu dòng (có thể kèm ., -, :, ))
        s = re.sub(r"^\s*\d+[\.\-:)]*\s*", "", s)

        # 3) xóa mọi chữ số còn lại trong dòng
        s = re.sub(r"\d+", "", s)

        # 4) gọn khoảng trắng
        s = re.sub(r"\s{2,}", " ", s).strip()

        if s:
            out.append(s)
        else:
            # nếu sau khi xóa số mà rỗng thì coi như dòng trống
            out.append("")

    # 5) nén nhiều dòng trống liên tiếp thành 1
    cleaned_lines = []
    blank = False
    for s in out:
        if s == "":
            if not blank:
                cleaned_lines.append("")
                blank = True
        else:
            cleaned_lines.append(s)
            blank = False

    cleaned = "\n".join(cleaned_lines).strip() + "\n"
    return cleaned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_PATH), help="Đường dẫn poem.txt (mặc định: data/interim/poem/poem.txt)")
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"❗ Không thấy file: {p}")

    print(f"[INFO] Load: {p}")
    raw = load_text(p)

    bkp = backup_file(p)
    print(f"[INFO] Backup  -> {bkp.name}")

    cleaned = sanitize_numbers(raw)
    save_text(p, cleaned)
    print(f"[OK] Wrote    -> {p.name}")
    print(f"[DONE] Poem sanitized (numbers removed).")

if __name__ == "__main__":
    main()
