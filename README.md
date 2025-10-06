# Kieu-Bot
Chatbot RAG + (tùy chọn) LoRA cho Truyện Kiều.

## Chạy nhanh
1) Tạo venv, `pip install -r requirements.txt`, copy `.env.example` → `.env`
2) Đặt dữ liệu vào `data/raw/` rồi convert/làm sạch → `data/interim/`
3) `python scripts/01_build_chunks.py` → tạo `data/rag_chunks/`
4) `python scripts/02_embed_and_index_mongo.py`
5) Tạo Vector Search Index trong Atlas (paste `scripts/03_create_mongo_vector_index.js`)
6) `streamlit run app/ui_streamlit.py`
