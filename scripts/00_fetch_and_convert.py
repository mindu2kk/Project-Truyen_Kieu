# scripts/00_fetch_and_convert.py
# -*- coding: utf-8 -*-
"""
Crawl & convert HTML -> Markdown/TXT cho kho Truyện Kiều (bản đơn giản, đổ vào data/interim/ana)

Cách dùng:
  python scripts/00_fetch_and_convert.py --url "https://vi-du-trang.com/binh-giang-kieu"
  python scripts/00_fetch_and_convert.py --url-file urls.txt
  python scripts/00_fetch_and_convert.py --local-html "D:/path/to/file.html"

Ghi chú:
- Luôn lưu vào: data/interim/ana/
- Tên file có gắn hash 8 ký tự theo URL để KHÔNG bị ghi đè.
- Xuất kèm .md và .txt (txt đã bỏ format link).
"""

import argparse
import re
import unicodedata
import hashlib
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, unquote

import requests
from bs4 import BeautifulSoup
import html2text

# ====== Thư mục đích cố định như "cách cũ" ======
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "interim" / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ====== Helpers ======
def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-zA-Z0-9\-_.\s]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    return (text.lower()[:120] or "source")

def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]

def url_to_basename(url: str) -> str:
    """
    Tạo base name có ý nghĩa từ URL (host + path + query rút gọn).
    Dù thế nào, tên file vẫn được gắn thêm hash để đảm bảo duy nhất.
    """
    p = urlparse(url)
    host = p.netloc.replace("www.", "")
    path = p.path.strip("/").replace("/", "-")
    if p.query:
        qs = "-".join([
            f"{slugify(k)}-{slugify(unquote(v))}"
            for k, v in parse_qsl(p.query, keep_blank_values=True)
        ])[:120]
    else:
        qs = ""
    parts = [slugify(host)]
    if path: parts.append(slugify(path))
    if qs:   parts.append(qs)
    base = "-".join([x for x in parts if x])
    return base or "source"

def clean_markdown(md: str) -> str:
    md = unicodedata.normalize("NFC", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    return md.strip()

def extract_main(html: str) -> tuple[str, str | None]:
    """Trả về (main_html, title). Ưu tiên vùng nội dung bài viết."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    main = soup.select_one("article, .post, .entry-content, .content, main, .main")
    if not main:
        main = soup.body or soup
    for sel in [
        "header", "footer", "nav", "aside",
        ".sidebar", ".ads", ".advert", ".banner", ".breadcrumbs",
        ".menu", ".navbar", ".comments", ".related-posts"
    ]:
        for t in main.select(sel):
            t.decompose()
    for t in main.find_all(["script", "style"]):
        t.decompose()
    return str(main), title

def html_to_markdown(html_fragment: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html_fragment)

def fetch_url(url: str, timeout=25) -> str:
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari")
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r.text

def save_markdown(md: str, url: str | None = None, title_hint: str | None = None) -> Path:
    """
    Lưu .md và .txt vào OUT_DIR. Tên file = <base>-<hash8>.
    - Với URL: hash theo URL.
    - Với local HTML: hash theo nội dung.
    """
    if url:
        base = url_to_basename(url)
        h = url_hash(url)
        name = f"{base}-{h}"
    else:
        h = hashlib.sha1(md.encode("utf-8")).hexdigest()[:8]
        name = f"{slugify(title_hint or 'source')}-{h}"

    header = f"<!-- source: {url} -->\n\n" if url else ""
    md_full = header + md

    md_path  = OUT_DIR / f"{name}.md"
    txt_path = OUT_DIR / f"{name}.txt"

    if not md_path.exists():
        md_path.write_text(md_full, encoding="utf-8")

    if not txt_path.exists():
        txt = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", md)  # bỏ link -> plain text
        txt_path.write_text(txt, encoding="utf-8")

    return md_path

def read_url_list(file_path: str) -> list[str]:
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


# ====== Main workers ======
def process_url(url: str) -> Path:
    print(f"[INFO] Fetching: {url}")
    html = fetch_url(url)
    try:
        main_html, page_title = extract_main(html)
        md = clean_markdown(html_to_markdown(main_html))
        if len(md) < 80:
            print("[WARN] Main content too short, fallback to full page conversion.")
            md = clean_markdown(html_to_markdown(html))
    except Exception as e:
        print(f"[WARN] extract/convert failed: {e}. Fallback to full page.")
        md = clean_markdown(html_to_markdown(html))
        page_title = None

    outp = save_markdown(md, url=url, title_hint=page_title)
    print(f"[OK] Saved -> {outp}")
    return outp

def process_local_html(path_str: str) -> Path:
    path = Path(path_str)
    html = path.read_text(encoding="utf-8", errors="ignore")
    main_html, page_title = extract_main(html)
    md = clean_markdown(html_to_markdown(main_html))
    return save_markdown(md, url=None, title_hint=page_title or path.stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Một URL đơn để tải & chuyển đổi")
    ap.add_argument("--url-file", help="Đường dẫn file .txt chứa danh sách URL (mỗi dòng 1 URL)")
    ap.add_argument("--local-html", help="Đường dẫn file HTML local để chuyển đổi")
    args = ap.parse_args()

    outputs = []

    if args.url:
        outputs.append(process_url(args.url))

    if args.url_file:
        for u in read_url_list(args.url_file):
            try:
                outputs.append(process_url(u))
            except Exception as e:
                print(f"[ERR] {u}: {e}")

    if args.local_html:
        outputs.append(process_local_html(args.local_html))

    if not outputs:
        print("Chưa có tham số. Dùng --url hoặc --url-file hoặc --local-html.")
    else:
        print("Đã lưu:")
        for p in outputs:
            print(" -", p)

if __name__ == "__main__":
    main()
