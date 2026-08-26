# Chatbot-RAG

Personal portfolio RAG assistant for [Sai Vishnu Vamsi Senagasetty](https://github.com/VishnuVamsi7).

**Architecture (free-tier friendly):**

1. **Offline (laptop):** curate `data/knowledge.json` → chunk → embed via **Hugging Face Inference API** → save FAISS under `data/faiss_index/`
2. **Online (Render):** embed the user question via HF API → FAISS top-k → **Groq** answer  
   No PyTorch / sentence-transformers on the server (avoids free-tier disk limits).

## Knowledge rules

- Source of truth is curated `data/knowledge.json` (from current portfolio — not legacy `myinfo.json`).
- Zibtek work is high-level only; Snapcite is excluded; PreachTogether / AskHeaven contributions are limited to what is written in the knowledge file.
- Rebuild the index after any knowledge edit.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY and HF_API_TOKEN
```

### Build FAISS index

```bash
python scripts/build_index.py
```

Writes:

- `data/faiss_index/index.faiss`
- `data/faiss_index/chunks.json`
- `data/chunks.jsonl` (debug)

### Run API

```bash
python app.py
# POST http://localhost:10000/receive  {"message":"What is TwinMind?"}
# GET  http://localhost:10000/health
```

### Render

- **Build:** `pip install -r requirements.txt`
- **Start:** leave blank to use `Procfile`, or set  
  `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app`
- Python is pinned in `runtime.txt` (`3.11.x`). Avoid Render’s default 3.13 with FAISS.

### Docker

```bash
docker build -t chatbot-rag .
docker run -p 10000:10000 --env-file .env chatbot-rag
```

## Env vars

| Var | Required | Purpose |
|-----|----------|---------|
| `GROQ_API_KEY` | yes | Chat generation |
| `HF_API_TOKEN` | yes | Query (+ build) embeddings |
| `GROQ_MODEL` | no | default `openai/gpt-oss-20b` |
| `EMBED_MODEL` | no | default `sentence-transformers/all-MiniLM-L6-v2` |
| `RAG_TOP_K` | no | default `4` |

## Portfolio integration

The Next.js portfolio chatbot posts to `/receive` on the Render deployment.
