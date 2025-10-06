# -*- coding: utf-8 -*-
"""
Debug truy hồi (vector search) từ Mongo.
Cho phép:
- Chạy 1 truy vấn đơn lẻ (--q "câu hỏi")
- Hoặc đọc nhiều truy vấn từ file (--qfile data/eval/retrieval_queries.txt)
- In top-k kết quả với id, score, source (+ optional xem snippet)

Usage:
  python eval/debug_retrieval.py --q "Truyện Kiều có bao nhiêu câu?"
  python eval/debug_retrieval.py --qfile data/eval/retrieval_queries.txt
  python eval/debug_retrieval.py --q "Tả cảnh ngụ tình" --k 10 --candidates 200 --types analysis,summary --show
"""

import argparse
from pathlib import Path
import importlib.util
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RETRIEVE_PY = ROOT / "scripts" / "04_retrieve.py"

def load_retrieve():
    spec = importlib.util.spec_from_file_location("retrieve_mod", RETRIEVE_PY)
    if not spec or not spec.loader:
        raise SystemExit(f"Không tìm thấy {RETRIEVE_PY}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["retrieve_mod"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod.retrieve_context

def read_lines(p: Path):
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]

def print_hit(i, h, show=False, max_chars=320):
    mid = h.get("meta", {}).get("id") or "N/A"
    src = h.get("meta", {}).get("source") or "N/A"
    sc  = h.get("score", 0.0)
    print(f" {i:2d}. id={mid:<24}  score={sc:.3f}  src={src}")
    if show:
        txt = (h.get("text") or "").strip().replace("\r","")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + " ..."
        block = textwrap.indent(txt, prefix="     ")
        print(block)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", help="Một câu hỏi/query")
    ap.add_argument("--qfile", help="File chứa nhiều query (mỗi dòng 1 câu hỏi)")
    ap.add_argument("--k", type=int, default=10, help="Top-k hiển thị")
    ap.add_argument("--candidates", type=int, default=100, help="numCandidates cho $vectorSearch")
    ap.add_argument("--types", help="Lọc meta.type, ví dụ: analysis,summary,bio,poem")
    ap.add_argument("--show", action="store_true", help="In kèm snippet nội dung chunk")
    args = ap.parse_args()

    retrieve_context = load_retrieve()

    queries = []
    if args.q:
        queries = [args.q]
    elif args.qfile:
        queries = read_lines(Path(args.qfile))
    else:
        print("Nhập query (Enter trống để kết thúc):")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            queries.append(line)

    if not queries:
        print("❗ Không có query. Dùng --q hoặc --qfile")
        return

    # build filter theo type
    filters = None
    if args.types:
        allow = [x.strip() for x in args.types.split(",") if x.strip()]
        if allow:
            filters = {"meta.type": {"$in": allow}}

    for q in queries:
        print(f"\n=== QUERY: {q}")
        try:
            hits = retrieve_context(q, k=args.k, num_candidates=args.candidates, filters=filters)
        except Exception as e:
            print("  [ERR] truy hồi lỗi:", e)
            continue
        if not hits:
            print("  (Không có kết quả)")
            continue
        for i, h in enumerate(hits, 1):
            print_hit(i, h, show=args.show)

if __name__ == "__main__":
    main()
