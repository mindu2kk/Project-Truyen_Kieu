# app/ui_streamlit.py
# -*- coding: utf-8 -*-
import time
import streamlit as st

from rag_pipeline import answer_question
from generation import (

    generate_answer_gemini,   # <— import Gemini
)

st.set_page_config(page_title="Kiều Bot · RAG", page_icon="📚", layout="wide")
st.title("📚 Kiều Bot — RAG demo")

with st.sidebar:
    st.header("Thiết lập truy hồi")
    k = st.slider("Số đoạn ngữ cảnh (top-k)", 3, 8, 5)
    use_custom_filter = st.checkbox("Tùy chỉnh filter `meta.type`", value=False)
    allowed_types = ["analysis","poem","summary","bio"]
    custom_types = st.multiselect("Chọn type", allowed_types, default=allowed_types)

    st.header("Sinh câu trả lời")
    engine = st.selectbox(
        "Động cơ sinh",
        ["(Chỉ dựng prompt)", "Gemini (Google)"],
        index=1  # mặc định chọn Gemini
    )
    if engine == "Gemini (Google)":
        st.caption("Cần GOOGLE_API_KEY trong .env · Mặc định dùng gemini-1.5-flash")
    elif engine == "Local Transformers":
        st.caption("Tải model lần đầu (~1GB).")

query = st.text_input(
    "Nhập câu hỏi:",
    placeholder="Ví dụ: Ý nghĩa 'Hoa ghen thua thắm, liễu hờn kém xanh' là gì?"
)

b1, b2, _ = st.columns([1,1,2])
with b1:
    run = st.button("🔍 Tra cứu", type="primary", use_container_width=True)
with b2:
    clear = st.button("🧹 Xóa", use_container_width=True)

if clear:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_answer", None)
    st.rerun()

if run and query:
    t0 = time.time()
    filters = {"meta.type": {"$in": custom_types}} if use_custom_filter else None
    try:
        pack = answer_question(query, k=k, filters=filters)
    except Exception as e:
        st.error(f"Lỗi khi truy hồi: {e}")
        st.stop()
    st.session_state["last_result"] = pack
    st.success(f"Đã truy hồi & dựng prompt trong ~{(time.time()-t0)*1000:.0f} ms")

pack = st.session_state.get("last_result")
if pack:
    ctx = pack["contexts"]
    prompt = pack["prompt"]

    colL, colR = st.columns([1,1])
    with colL:
        st.subheader("🔎 Ngữ cảnh truy hồi")
        if not ctx:
            st.warning("Không có ngữ cảnh phù hợp.")
        else:
            for i, c in enumerate(ctx, 1):
                with st.expander(f"[{i}] score={c.get('score',0):.4f} · source={c.get('meta',{}).get('source')}", expanded=(i==1)):
                    st.markdown("**Meta**")
                    st.json(c.get("meta", {}), expanded=False)
                    st.markdown("**Đoạn văn**")
                    st.write(c["text"])

    with colR:
        st.subheader("🧩 Prompt đã dựng")
        st.code(prompt, language="text")
        st.download_button("⬇️ Tải prompt (.txt)", prompt, "kieu_rag_prompt.txt", "text/plain", use_container_width=True)

        st.subheader("🧠 Câu trả lời")
        answer_btn = st.button("✨ Sinh câu trả lời", use_container_width=True)
        if answer_btn:
            with st.spinner("Đang sinh câu trả lời..."):
                if engine == "(Chỉ dựng prompt)":
                    st.info("Bạn đang ở chế độ chỉ dựng prompt.")
                elif engine == "Gemini (Google)":
                    ans = generate_answer_gemini(prompt)
                    st.session_state["last_answer"] = ans

        if "last_answer" in st.session_state:
            st.markdown(st.session_state["last_answer"])

    st.subheader("📎 Danh sách nguồn")
    sources = []
    for c in ctx:
        src = c.get("meta", {}).get("source")
        if src and src not in sources:
            sources.append(src)
    st.write("; ".join(sources) if sources else "(Không có nguồn)")
