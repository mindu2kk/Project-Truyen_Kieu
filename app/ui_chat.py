# app/ui_chat.py
# -*- coding: utf-8 -*-
"""
Chat UI cho Kiều Bot (RAG, có định tuyến + nhớ ngữ cảnh ngắn hạn)
- Ẩn prompt/context, chỉ hiển thị hội thoại + nguồn.
- Định tuyến: FAQ -> trả thẳng; chitchat/generic -> Gemini; domain (Truyện Kiều) -> RAG+Gemini.
"""

import time
import streamlit as st
from orchestrator import answer_with_router  # điều phối trả lời theo intent

# ====== Cấu hình trang ======
st.set_page_config(page_title="Kiều Bot", page_icon="📚", layout="centered")
st.title("📚 Kiều Bot")

# ====== Sidebar: cấu hình ngắn gọn ======
with st.sidebar:
    st.header("Thiết lập")
    k = st.slider("Top-k ngữ cảnh", 3, 6, 4)
    model = st.selectbox("Gemini model", ["gemini-2.0-flash", "gemini-2.0-flash-lite"], index=0)
    long_ans = st.toggle("Trả lời dài hơn (nhập vai luận văn)", value=False)
    st.caption("Kho tri thức: dữ liệu bạn đã nạp vào Atlas")

# ====== Bộ nhớ hội thoại ======
if "chat" not in st.session_state:
    st.session_state.chat = []  # list[(role, text)]

# Hiển thị lịch sử hội thoại
for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

# ====== Ô nhập người dùng ======
user_msg = st.chat_input("Hỏi về Truyện Kiều…")

if user_msg:
    # hiển thị tin người dùng
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    # lấy history ngắn hạn (cuối 6-8 lượt)
    history = st.session_state.chat[-12:]

    # gọi router để trả lời đúng nhánh
    with st.chat_message("assistant"):
        t0 = time.time()
        ret = answer_with_router(
            user_msg,
            k=k,
            gemini_model=model,
            long_answer=long_ans,
            history=history,            # <— truyền history vào đây
        )
        st.markdown(ret["answer"])
        st.caption(f"⏱️ {(time.time() - t0) * 1000:.0f} ms")

    # lưu câu trả lời vào lịch sử
    st.session_state.chat.append(("assistant", ret["answer"]))
