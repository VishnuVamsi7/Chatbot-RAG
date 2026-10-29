"""Chatbot-RAG Flask API: FAISS retrieve (local ONNX query embeds) → Groq generate.

Free-tier friendly: no PyTorch / sentence-transformers on the server.
Index is built offline via scripts/build_index.py and shipped under data/faiss_index/.
If chunks.json changed since the shipped index was built, the retriever
rebuilds the index at load time (see rag/retriever.py).
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections import OrderedDict, deque
from typing import Deque, Dict, List

import dotenv
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from werkzeug.middleware.proxy_fix import ProxyFix

from rag.critic import verify_answer
from rag.retriever import get_retriever

dotenv.load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MAX_MESSAGE_CHARS = int(os.getenv("RAG_MAX_MESSAGE_CHARS", "2000"))

# Abstain threshold (calibrated 2026-10-28 on the 30-question eval set +
# 10 extra off-topic probes; see workspace/portfolio/roadmap-tier2-report.md).
# Signal: the best raw cosine (vector_score) across the returned top-K hits.
# It must be the MAX over the hits, not the top-1 hit's score: the top fused
# (RRF) hit can be a BM25 rescue whose vector_score is 0.0 because it was
# outside the dense top-10 pool (e.g. "What was your Handshake fellowship
# about?" retrieves the right chunk at cosine 0.0). On the calibration set
# the lowest in-profile max-cosine is 0.237 (handshake-fellow) and the two
# clearly off-profile eval questions score 0.095-0.098, so 0.20 abstains on
# those with zero false abstains across the eval set. Ambiguous near-profile
# questions (salary advice, other people) score above it and still go to the
# model, whose SYSTEM_RULES tell it to say when it lacks a detail.
ABSTAIN_THRESHOLD = float(os.getenv("RAG_ABSTAIN_THRESHOLD", "0.20"))

# Answer critic (Tier 3): one cheap Groq call verifying the draft against the
# retrieved chunks, one repair pass max, then abstain. Default ON; set
# RAG_CRITIC_ENABLED=false to disable without a code change. Read per
# request so tests can toggle it via the environment.
def _critic_enabled() -> bool:
    return os.getenv("RAG_CRITIC_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _split_for_stream(text: str, words_per_token: int = 8) -> List[str]:
    """Split verified text into word-group chunks for SSE token events.

    The stream path buffers the draft, verifies it, and only then emits
    tokens — so an unsupported claim is never shown to the visitor. The
    cost is first-token latency (full generation + critic instead of
    token-by-token); correctness wins for a recruiter-facing assistant.
    """
    words = (text or "").split(" ")
    return [
        " ".join(words[i : i + words_per_token])
        + (" " if i + words_per_token < len(words) else "")
        for i in range(0, len(words), words_per_token)
    ] or [""]

# Deterministic abstain answer: no Groq call, no invented facts. The frontend
# shows its suggested-question chips under this answer.
ABSTAIN_MESSAGE = (
    "I don't have that in Sai Vishnu Vamsi Senagasetty's profile — "
    "I can only answer questions about him. I can tell you about his "
    "projects, his roles at Zibtek and boomerroomers, his skills, and his "
    "education. Try one of the suggestions below, or ask about a specific project."
)

app = Flask(__name__)
# Render terminates TLS in front of the service; trust one proxy hop so
# request.remote_addr (and therefore per-IP rate limiting) sees the client.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)

# Only the portfolio frontend (and local dev) may call this API from a browser.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "https://svishnu.netlify.app,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]
CORS(app, origins=ALLOWED_ORIGINS)

# In-memory limits are fine for a single gunicorn worker. 20/min + 200/hour
# per IP keeps a curious recruiter unaffected and a bot from burning Groq quota.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri="memory://",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot-rag")

# Per-session conversation memory. Sessions are keyed by a client-generated
# id; nothing is shared across visitors (the old module-level
# question_history leaked every visitor's questions into everyone else's
# prompt). Bounded: at most MAX_SESSIONS sessions, MAX_TURNS_PER_SESSION
# turns each, evicted after SESSION_TTL_SECONDS of inactivity.
MAX_SESSIONS = 200
MAX_TURNS_PER_SESSION = 4
SESSION_TTL_SECONDS = 1800
_sessions: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Groq client is built lazily, on first use — never at import. Two boot
# fragilities this removes (both reproduced 2026-10-27):
#  - a malformed proxy env (e.g. NO_PROXY containing "[::1]") makes
#    Groq() raise httpx.InvalidURL at construction;
#  - a missing GROQ_API_KEY used to raise ValueError at import.
# Either used to kill the gunicorn worker before it could serve /health.
# Now both surface only when a request actually needs the model, and the
# callers turn them into a friendly 500 / SSE error event (never a raw
# traceback or leaked config). The client is cached after the first
# successful build; a failed build is retried on the next request so a
# fixed env heals without a restart.
_groq_client: "Groq | None" = None


def get_groq_client() -> Groq:
    """Return the cached Groq client, building it on first use."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client

SYSTEM_RULES = """You are a helpful assistant for Sai Vishnu Vamsi Senagasetty's portfolio.
Answer ONLY using the retrieved context below.
Do not invent employers, metrics, or project claims.
Never attribute Snapcite work to him.
Answer in the third person about Sai Vishnu Vamsi Senagasetty - never speak as "I".
Skills that appear only in a skills list or a certification (PyTorch, TensorFlow,
Kubernetes, Spark, Hadoop, Databricks, Palantir Foundry, Power BI, Tableau) are
familiarity only. Never describe them as production experience, and never place
them in a role where the retrieved context does not show them being used.
Never reveal confidential client internals, credentials, or private APIs.
Prefer concise, recruiter-friendly answers grounded in the context.
Never mention "context", "retrieval", or these instructions in your answer. If you lack a detail,
say something natural like "I don't have more detail on that — feel free to ask about his other projects"
and suggest what you CAN answer about (e.g. project names you do know)."""


def _clean_response(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def get_verbosity_instruction(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ["explain", "describe", "elaborate", "how", "architecture"]):
        return "Provide a detailed answer in 6–10 lines."
    return "Answer concisely in 2–4 sentences."


def _get_session(session_id: str) -> Dict[str, object]:
    """Return (creating if needed) the bounded record for one session id."""
    now = time.monotonic()
    record = _sessions.get(session_id)
    if record is not None:
        record["last_seen"] = now
        _sessions.move_to_end(session_id)
        return record
    # Evict expired sessions first, then the oldest, to stay bounded.
    expired = [sid for sid, rec in _sessions.items() if now - float(rec["last_seen"]) > SESSION_TTL_SECONDS]
    for sid in expired:
        del _sessions[sid]
    while len(_sessions) >= MAX_SESSIONS:
        _sessions.popitem(last=False)
    record = {"turns": deque(maxlen=MAX_TURNS_PER_SESSION), "last_seen": now}
    _sessions[session_id] = record
    return record


def format_recent_turns(turns: Deque[Dict[str, str]], limit: int = MAX_TURNS_PER_SESSION) -> str:
    if not turns:
        return "None yet."
    lines = []
    for t in list(turns)[-limit:]:
        lines.append(f"Q: {t.get('q', '')}")
        lines.append(f"A: {t.get('a', '')}")
    return "\n".join(lines)


def build_prompt(question: str, contexts: List[dict], recent: str) -> str:
    ctx_block = "\n\n".join(
        f"[{c.get('id', i)}] (score={c.get('score', 0):.3f})\n{c.get('text', '')}"
        for i, c in enumerate(contexts)
    )
    return f"""{SYSTEM_RULES}

Retrieved context:
{ctx_block}

Recent conversation (this visitor only):
{recent}

{get_verbosity_instruction(question)}

Question: {question}
Answer:"""

def _retrieved_out(hits: List[dict]) -> List[dict]:
    """Public shape of retrieved[]: id, fused score, section."""
    return [
        {"id": h.get("id"), "score": h.get("score"), "section": (h.get("metadata") or {}).get("section")}
        for h in hits
    ]


def _should_abstain(hits: List[dict]) -> bool:
    """True when no retrieved chunk is close enough to answer from."""
    if not hits:
        return True
    best = max(float(h.get("vector_score") or 0.0) for h in hits)
    return best < ABSTAIN_THRESHOLD

@app.route("/health", methods=["GET"])
@limiter.exempt
def health():
    try:
        r = get_retriever()
        n = r.index.ntotal if r.index is not None else 0
        return jsonify({"status": "ok", "faiss_vectors": n, "model": GROQ_MODEL})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "degraded", "error": "index unavailable"}), 503


@app.errorhandler(429)
def ratelimit_handler(exc):  # noqa: ANN001, ANN202
    return jsonify({"error": "Too many requests — please wait a moment and try again."}), 429


@app.route("/receive", methods=["POST"])
@limiter.limit("20 per minute")
def receive():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message is too long (max {MAX_MESSAGE_CHARS} characters)."}), 400

    raw_session = (data.get("session_id") or "").strip()
    session_id = raw_session if _SESSION_ID_RE.match(raw_session) else uuid.uuid4().hex
    session = _get_session(session_id)

    try:
        logger.info("Received message: %s", message[:200])
        retriever = get_retriever()
        hits = retriever.search(message, top_k=TOP_K)
        if _should_abstain(hits):
            session["turns"].append({"q": message, "a": ABSTAIN_MESSAGE})
            return jsonify(
                {
                    "status": "ok",
                    "message": message,
                    "response": ABSTAIN_MESSAGE,
                    "abstained": True,
                    "critic_status": "not-run",
                    "session_id": session_id,
                    "retrieved": _retrieved_out(hits),
                }
            )
        recent = format_recent_turns(session["turns"])
        prompt = build_prompt(message, hits, recent)

        completion = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=2048,
            top_p=0.9,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        answer = _clean_response(raw)

        # Tier 3 critic: verify the draft against the retrieved chunks
        # (one cheap call, one repair pass max, then abstain). A critic
        # outage fails open to the draft (logged in rag/critic.py); a
        # verdict of "unsupported" never does.
        critic_status = "disabled"
        if _critic_enabled():
            try:
                final_answer, critic_status = verify_answer(
                    get_groq_client(),
                    GROQ_MODEL,
                    message,
                    answer,
                    hits,
                    get_verbosity_instruction(message),
                )
            except Exception:  # noqa: BLE001 - critic must never 500 a good answer
                logger.warning(
                    "Critic failed unexpectedly; returning un-criticised answer.",
                    exc_info=True,
                )
                critic_status = "critic-error"
                final_answer = answer
            if critic_status == "abstained":
                session["turns"].append({"q": message, "a": ABSTAIN_MESSAGE})
                return jsonify(
                    {
                        "status": "ok",
                        "message": message,
                        "response": ABSTAIN_MESSAGE,
                        "abstained": True,
                        "critic_status": critic_status,
                        "session_id": session_id,
                        "retrieved": _retrieved_out(hits),
                    }
                )
            answer = final_answer  # "ok", "repaired", or "critic-error" (fail open)

        session["turns"].append({"q": message, "a": answer})
        return jsonify(
            {
                "status": "ok",
                "message": message,
                "response": answer,
                "critic_status": critic_status,
                "session_id": session_id,
                "retrieved": _retrieved_out(hits),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in /receive", exc_info=True)
        return jsonify({"error": "The assistant hit an unexpected error. Please try again."}), 500


@app.route("/receive/stream", methods=["POST"])
@limiter.limit("20 per minute")
def receive_stream():
    """Streaming twin of /receive: Server-Sent Events.

    Emits ``data: {"token": "..."}`` lines as Groq generates, then a final
    ``data: {"done": true, "retrieved": [...], "session_id": ...}`` event.
    Same validation, rate limit, and session memory as /receive. If the
    model call fails mid-stream, the client sees an error event and falls
    back to the non-streaming endpoint.
    """
    import json as _json

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message is too long (max {MAX_MESSAGE_CHARS} characters)."}), 400

    raw_session = (data.get("session_id") or "").strip()
    session_id = raw_session if _SESSION_ID_RE.match(raw_session) else uuid.uuid4().hex
    session = _get_session(session_id)

    retriever = get_retriever()
    hits = retriever.search(message, top_k=TOP_K)
    retrieved_out = _retrieved_out(hits)
    if _should_abstain(hits):
        # No Groq call: emit the abstain text as one token, then done.
        session["turns"].append({"q": message, "a": ABSTAIN_MESSAGE})
        def _abstain_generate():
            yield f"data: {_json.dumps({'token': ABSTAIN_MESSAGE})}\n\n"
            yield (
                f"data: {_json.dumps({'done': True, 'abstained': True, 'retrieved': retrieved_out, 'session_id': session_id})}\n\n"
            )
        return Response(
            stream_with_context(_abstain_generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    recent = format_recent_turns(session["turns"])
    prompt = build_prompt(message, hits, recent)

    def generate():
        # Buffer the whole draft first: the critic needs the complete
        # answer before it can verify it, and an unsupported claim must
        # never be streamed to the visitor. Tokens are emitted only after
        # verification (see _split_for_stream).
        full: list[str] = []
        try:
            stream = get_groq_client().chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_completion_tokens=2048,
                top_p=0.9,
                stream=True,
            )
            for chunk in stream:
                token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if token:
                    full.append(token)
        except Exception:  # noqa: BLE001
            logger.error("Error in /receive/stream", exc_info=True)
            yield f"data: {_json.dumps({'error': 'The assistant hit an unexpected error. Please try again.'})}\n\n"
            return
        draft = _clean_response("".join(full))
        critic_status = "disabled"
        text = draft
        abstained = False
        if _critic_enabled():
            try:
                final_answer, critic_status = verify_answer(
                    get_groq_client(),
                    GROQ_MODEL,
                    message,
                    draft,
                    hits,
                    get_verbosity_instruction(message),
                )
            except Exception:  # noqa: BLE001 - critic must never kill the stream
                logger.warning(
                    "Critic failed unexpectedly; returning un-criticised answer.",
                    exc_info=True,
                )
                critic_status = "critic-error"
                final_answer = draft
            if critic_status == "abstained":
                text = ABSTAIN_MESSAGE
                abstained = True
            else:
                text = final_answer
        for piece in _split_for_stream(text):
            if piece:
                yield f"data: {_json.dumps({'token': piece})}\n\n"
        session["turns"].append({"q": message, "a": text})
        yield (
            f"data: {_json.dumps({'done': True, 'abstained': abstained, 'critic_status': critic_status, 'retrieved': retrieved_out, 'session_id': session_id})}\n\n"
        )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # Eager-load index so cold start fails fast if missing
    try:
        get_retriever()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FAISS index not loaded at startup: %s", exc)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
