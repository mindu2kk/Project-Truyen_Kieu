# app/generation.py
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
load_dotenv()  # nạp .env từ thư mục gốc repo

import google.generativeai as genai

def _configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY/GEMINI_API_KEY chưa được thiết lập. "
            "Hãy đặt vào .env hoặc biến môi trường hệ thống."
        )
    genai.configure(api_key=api_key)

def generate_answer_gemini(prompt: str, model: str = "gemini-2.0-flash",
                           max_output_tokens: int = 4096,
                           long_answer: bool = False) -> str:
    _configure_gemini()
    gm = genai.GenerativeModel(model)
    # nếu bạn muốn cho phép trả lời dài hơn UI mặc định:
    if long_answer:
        max_output_tokens = max(2048, max_output_tokens)

    res = gm.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_output_tokens, "temperature": 0.3}
    )
    # xử lý kết quả & lỗi nhẹ
    try:
        return res.text.strip()
    except Exception:
        return str(res)
