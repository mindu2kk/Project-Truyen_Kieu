# ---- Base image
FROM python:3.12

# ---- System deps (nhỏ gọn, đủ để build wheels cơ bản)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# ---- Workdir
WORKDIR /app

# ---- Copy code
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY app /app/app
COPY scripts /app/scripts
COPY data /app/data
COPY .env /app/.env

# ---- Cấu hình cache model (đặt theo ENV, có thể override khi run)
ENV HF_HOME=/cache/hf_home \
    TRANSFORMERS_CACHE=/cache/transformers \
    RAG_CACHE_DIR=/cache/rag_cache

# tạo sẵn thư mục cache
RUN mkdir -p /cache/hf_home /cache/transformers /cache/rag_cache

# ---- Expose và run Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app/ui_chat.py", "--server.port=8501", "--server.address=0.0.0.0"]
