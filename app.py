"""Chatbot-RAG Flask API: FAISS retrieve (HF query embeds) → Groq generate.

Free-tier friendly: no PyTorch / sentence-transformers on the server.
Index is built offline via scripts/build_index.py and shipped under data/faiss_index/.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

import dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq

from rag.retriever import get_retriever

dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.getenv("RAG_TOP_K", "4"))

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot-rag")

question_history: List[str] = []
_groq = Groq(api_key=GROQ_API_KEY)

SYSTEM_RULES = """You are a helpful assistant for Sai Vishnu Vamsi Senagasetty's portfolio.
Answer ONLY using the retrieved context below. If the context is insufficient, say you do not have that detail.
Do not invent employers, metrics, or project claims.
Never attribute Snapcite work to him.
Never reveal confidential client internals, credentials, or private APIs.
Prefer concise, recruiter-friendly answers grounded in the context."""


def _clean_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def get_verbosity_instruction(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ["explain", "describe", "elaborate", "how", "architecture"]):
        return "Provide a detailed answer in 6–10 lines."
    return "Answer concisely in 2–4 sentences."


def format_recent_questions(history: List[str], limit: int = 5) -> str:
    if not history:
        return "None yet."
    return "\n".join(f"- {q}" for q in history[-limit:])


def build_prompt(question: str, contexts: List[dict], recent: str) -> str:
    ctx_block = "\n\n".join(
        f"[{c.get('id', i)}] (score={c.get('score', 0):.3f})\n{c.get('text', '')}"
        for i, c in enumerate(contexts)
    )
    return f"""{SYSTEM_RULES}

Retrieved context:
{ctx_block}

Recent questions:
{recent}

{get_verbosity_instruction(question)}

Question: {question}
Answer:"""


@app.route("/health", methods=["GET"])
def health():
    try:
        r = get_retriever()
        n = r.index.ntotal if r.index is not None else 0
        return jsonify({"status": "ok", "faiss_vectors": n, "model": GROQ_MODEL})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "degraded", "error": str(exc)}), 503


@app.route("/receive", methods=["POST"])
def receive():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        logger.info("Received message: %s", message[:200])
        retriever = get_retriever()
        hits = retriever.search(message, top_k=TOP_K)
        recent = format_recent_questions(question_history)
        prompt = build_prompt(message, hits, recent)

        completion = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=2048,
            top_p=0.9,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        answer = _clean_response(raw)

        question_history.append(message)
        return jsonify(
            {
                "status": "ok",
                "message": message,
                "response": answer,
                "retrieved": [
                    {"id": h.get("id"), "score": h.get("score"), "section": (h.get("metadata") or {}).get("section")}
                    for h in hits
                ],
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in /receive", exc_info=True)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Eager-load index so cold start fails fast if missing
    try:
        get_retriever()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FAISS index not loaded at startup: %s", exc)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
