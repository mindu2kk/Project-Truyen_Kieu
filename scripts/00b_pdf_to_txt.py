# -*- coding: utf-8 -*-
"""
PDF -> TXT ROBUST (ưu tiên không vỡ font)
Thứ tự:
  1) Poppler pdftotext (-enc UTF-8 -layout)
  2) pypdfium2 (extract text)
  3) ocrmypdf (force-ocr) -> pdftotext
  4) pdf2image + Tesseract (trang nào lỗi mới OCR)

Usage:
  python scripts/10_pdf_to_txt_robust.py --pdf "D:/path/file.pdf" --out "data/interim/analysis"
  python scripts/10_pdf_to_txt_robust.py --pdf-dir "D:/pdfs" --out "data/interim/analysis"
Env/Tools (khuyên cài):
  - Poppler: pdftotext trong PATH (choco install poppler)
  - pypdfium2: pip install pypdfium2
  - OCR fallback: pip install ocrmypdf pdf2image pytesseract pillow; choco install tesseract ghostscript
    (cài gói ngôn ngữ vie cho Tesseract)
"""

import argparse, os, re, shutil, subprocess, tempfile, unicodedata
from pathlib import Path
from typing import Optional, List

# ---------- Utils ----------
def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t).replace("\u00A0", " ")
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def text_quality_bad(t: str) -> bool:
    """Heuristic: rỗng, quá ngắn, hoặc nhiều dấu � / ký tự rác -> coi như kém."""
    if not t or len(t.strip()) < 40:
        return True
    # tỉ lệ replacement char hoặc khuyết dấu
    bad = t.count("�")
    if bad / max(1, len(t)) > 0.01:  # >1% ký tự lỗi
        return True
    return False

def write_out(text: str, pdf_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / (pdf_path.stem + ".txt")
    outp.write_text(norm_text(text) + "\n", encoding="utf-8")
    return outp

# ---------- Layer 1: Poppler ----------
def have_pdftotext() -> bool:
    return shutil.which("pdftotext") is not None

def pdftotext_extract(pdf: Path) -> Optional[str]:
    if not have_pdftotext():
        return None
    with tempfile.TemporaryDirectory() as td:
        out_txt = Path(td) / "out.txt"
        cmd = ["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), str(out_txt)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            if out_txt.exists():
                return out_txt.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
    return None

# ---------- Layer 2: pypdfium2 ----------
def pypdfium2_extract(pdf: Path) -> Optional[str]:
    try:
        import pypdfium2 as pdfium  # pip install pypdfium2
    except Exception:
        return None
    try:
        doc = pdfium.PdfDocument(str(pdf))
        parts: List[str] = []
        for i in range(len(doc)):
            page = doc.get_page(i)
            txtpage = page.get_textpage()
            parts.append(txtpage.get_text_bounded().strip())
            txtpage.close()
            page.close()
        return "\n\n".join(p for p in parts if p.strip())
    except Exception:
        return None

# ---------- Layer 3: OCRmyPDF ----------
def have_ocrmypdf() -> bool:
    return shutil.which("ocrmypdf") is not None

def ocrmypdf_then_pdftotext(pdf: Path) -> Optional[str]:
    if not have_ocrmypdf() or not have_pdftotext():
        return None
    with tempfile.TemporaryDirectory() as td:
        ocr_pdf = Path(td) / "ocr.pdf"
        # --- tạo OCR layer (force) ---
        cmd1 = ["ocrmypdf", "--force-ocr", "--language", "vie+eng", "--deskew", "--rotate-pages", str(pdf), str(ocr_pdf)]
        try:
            subprocess.run(cmd1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        except Exception:
            return None
        # --- trích text từ PDF đã OCR ---
        cmd2 = ["pdftotext", "-enc", "UTF-8", "-layout", str(ocr_pdf), str(Path(td) / "out.txt")]
        try:
            subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            return (Path(td) / "out.txt").read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

# ---------- Layer 4: pdf2image + Tesseract ----------
def ocr_direct_each_page(pdf: Path, dpi=300, lang="vie+eng") -> Optional[str]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return None
    try:
        images = convert_from_path(str(pdf), dpi=dpi)
        parts = []
        for im in images:
            txt = pytesseract.image_to_string(im, lang=lang)
            parts.append(txt)
        return "\n\n".join(parts)
    except Exception:
        return None

# ---------- Orchestrator ----------
def convert_pdf(pdf: Path, out_dir: Path) -> Path:
    # 1) Poppler
    txt = pdftotext_extract(pdf)
    if txt and not text_quality_bad(txt):
        return write_out(txt, pdf, out_dir)

    # 2) pypdfium2
    txt = pypdfium2_extract(pdf)
    if txt and not text_quality_bad(txt):
        return write_out(txt, pdf, out_dir)

    # 3) OCRmyPDF -> pdftotext
    txt = ocrmypdf_then_pdftotext(pdf)
    if txt and not text_quality_bad(txt):
        return write_out(txt, pdf, out_dir)

    # 4) OCR trực tiếp
    txt = ocr_direct_each_page(pdf)
    if txt:
        return write_out(txt, pdf, out_dir)

    # Hết cách: vẫn lưu những gì có
    return write_out(txt or "", pdf, out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="Đường dẫn 1 PDF")
    ap.add_argument("--pdf-dir", help="Thư mục chứa nhiều PDF")
    ap.add_argument("--out", default="data/interim/analysis")
    args = ap.parse_args()

    targets: List[Path] = []
    if args.pdf:
        targets = [Path(args.pdf)]
    elif args.pdf_dir:
        targets = sorted(Path(args.pdf_dir).glob("*.pdf"))
    else:
        raise SystemExit("Hãy truyền --pdf hoặc --pdf-dir")

    if not targets:
        raise SystemExit("Không tìm thấy PDF nào.")

    out_dir = Path(args.out)
    for p in targets:
        try:
            outp = convert_pdf(p, out_dir)
            print(f"[OK] {p.name} -> {outp}")
        except Exception as e:
            print(f"[ERR] {p}: {e}")

if __name__ == "__main__":
    main()
