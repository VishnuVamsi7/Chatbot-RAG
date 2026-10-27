"""Local ONNX embeddings — no Hugging Face Inference API, no torch.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim), run on-device via
ONNX Runtime. The ONNX export + tokenizer come from Xenova/all-MiniLM-L6-v2
(quantized model, 21.9 MB) and are downloaded from the Hugging Face Hub at
first use into a local cache dir (ONNX_MODEL_DIR, else ~/.cache/chatbot-rag).
No API key, no credits, no per-query network call.

Mean pooling (attention-mask weighted) + L2 normalization reproduce
sentence-transformers behavior, so vectors match the existing FAISS index,
which was built with the same model.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_REPO = "Xenova/all-MiniLM-L6-v2"
# (filename on the Hub, expected bytes) — size-checked after download.
ONNX_FILES = {
    "model_quantized.onnx": 22972370,
    "tokenizer.json": 711661,
    "tokenizer_config.json": 366,
}
MAX_SEQ_LEN = 256
EMBED_DIM = 384
_DOWNLOAD_ATTEMPTS = 3


def _model_dir() -> Path:
    override = os.getenv("ONNX_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "chatbot-rag" / "all-MiniLM-L6-v2"


def _download_one(name: str, expected_bytes: int, dest: Path) -> None:
    url = f"https://huggingface.co/{ONNX_REPO}/resolve/main/{name}"
    if name == "model_quantized.onnx":
        url = f"https://huggingface.co/{ONNX_REPO}/resolve/main/onnx/model_quantized.onnx"
    last_err: Exception | None = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            logger.info("Downloading %s (attempt %s/%s) …", name, attempt, _DOWNLOAD_ATTEMPTS)
            req = urllib.request.Request(url, headers={"User-Agent": "chatbot-rag-onnx/1.0"})
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
            size = tmp_path.stat().st_size
            if expected_bytes and size != expected_bytes:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{name}: downloaded {size} bytes, expected {expected_bytes}"
                )
            os.replace(tmp_path, dest)
            logger.info("Downloaded %s (%s bytes)", name, size)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Download of %s failed: %s", name, exc)
    raise RuntimeError(f"Failed to download {name} from {url}: {last_err}") from last_err


def _ensure_model_files() -> Path:
    model_dir = _model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in ONNX_FILES.items():
        dest = model_dir / name
        if dest.exists() and expected and dest.stat().st_size == expected:
            continue
        if dest.exists():
            logger.warning("%s size mismatch — re-downloading", name)
            dest.unlink(missing_ok=True)
        _download_one(name, expected, dest)
    return model_dir


class _LocalEmbedder:
    """Single ONNX Runtime session + tokenizer, loaded once."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = _ensure_model_files()
        self.model_dir = model_dir
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        # Don't let the arena hold on to freed blocks — free-tier RAM is 512 MB.
        sess_options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(model_dir / "model_quantized.onnx"),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX embedder loaded from %s", model_dir)

    def embed_one(self, text: str) -> List[float]:
        enc = self.tokenizer.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        type_ids = np.array([enc.type_ids], dtype=np.int64)
        outputs = self.session.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": type_ids,
            },
        )
        token_embeddings = outputs[0][0]  # [seq_len, 384]
        mask_f = mask[0].astype(np.float32)
        summed = (token_embeddings * mask_f[:, None]).sum(axis=0)
        counts = mask_f.sum()
        mean_pooled = summed / max(counts, 1e-9)
        norm = float(np.linalg.norm(mean_pooled))
        if norm > 0:
            mean_pooled = mean_pooled / norm
        return mean_pooled.astype(np.float32).tolist()


_embedder: _LocalEmbedder | None = None
_embedder_lock = threading.Lock()


def _get_embedder() -> _LocalEmbedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = _LocalEmbedder()
    return _embedder


def embedder_status() -> dict:
    """Small status dict for /health — does NOT force a model load."""
    return {
        "backend": "onnx-local",
        "model": DEFAULT_MODEL,
        "loaded": _embedder is not None,
        "model_dir": str(_model_dir()),
    }


def embed_texts(
    texts: Sequence[str],
    model: str | None = None,
    *,
    max_retries: int = 5,
    timeout: float = 60.0,
) -> List[List[float]]:
    """Embed texts locally with all-MiniLM-L6-v2 (ONNX).

    Same signature as the old HF-backed version so retriever.py and
    scripts/build_index.py need no changes. ``model`` must be None or
    all-MiniLM-L6-v2 — the FAISS index was built with that model, and any
    other model would produce incompatible vectors. ``max_retries`` /
    ``timeout`` are accepted for compatibility and unused (no network call).
    """
    if model is not None and model != DEFAULT_MODEL:
        raise ValueError(
            f"Local embedder only supports {DEFAULT_MODEL} "
            f"(the FAISS index was built with it); got {model!r}"
        )
    embedder = _get_embedder()
    vectors: List[List[float]] = []
    for i, text in enumerate(texts):
        logger.info("Embedding %s/%s …", i + 1, len(texts))
        vectors.append(embedder.embed_one(text))
    return vectors


def embed_query(text: str, model: str | None = None) -> List[float]:
    return embed_texts([text], model=model)[0]
