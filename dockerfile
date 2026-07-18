# Slim runtime — no PyTorch. FAISS index is committed under data/faiss_index/.
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY rag ./rag
COPY data ./data

EXPOSE 10000

CMD ["python", "app.py"]
