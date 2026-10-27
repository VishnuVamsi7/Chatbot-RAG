"""Hugging Face Inference API embeddings — no local torch/sentence-transformers."""

from __future__ import annotations

import logging
import os
import time
from typing import List, Sequence

import httpx
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Only the current HF Inference router is live. The legacy
# api-inference.huggingface.co hosts are retired (they no longer answer),
# so keeping them as fallbacks only masked the real error — a 401 from the
# router surfaced as a DNS failure on the dead hosts after ~40 s of retries.
HF_URL_TEMPLATES = (
    "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction",
    "https://router.huggingface.co/hf-inference/models/{model}",
)


def _hf_token() -> str | None:
    return os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def _flatten_embedding(payload) -> List[float]:
    """Normalize HF feature-extraction responses to a single float vector."""
    arr = np.array(payload, dtype=np.float32)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        vec = arr.mean(axis=0)
    elif arr.ndim == 3:
        vec = arr[0].mean(axis=0)
    else:
        raise ValueError(f"Unexpected embedding shape: {arr.shape}")
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32).tolist()


def _embed_one(
    client: httpx.Client,
    text: str,
    model: str,
    headers: dict,
    max_retries: int,
) -> List[float]:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        for url_tmpl in HF_URL_TEMPLATES:
            url = url_tmpl.format(model=model)
            try:
                resp = client.post(
                    url,
                    headers=headers,
                    json={"inputs": text, "options": {"wait_for_model": True}},
                )
                if resp.status_code in (401, 402, 403):
                    # Auth/billing will not fix itself by retrying — fail fast
                    # with the real cause instead of burning ~30 s of retries.
                    if resp.status_code == 402:
                        detail = (
                            "the Hugging Face account has no Inference Providers "
                            "credits (free accounts get none since Oct 2026). Buy "
                            "credits or upgrade to PRO at "
                            "huggingface.co/settings/billing."
                        )
                    else:
                        detail = (
                            "the HF_API_TOKEN on Render is missing or invalid. Set a "
                            "valid token (hf.co/settings/tokens, with Inference "
                            "Providers permission) as HF_API_TOKEN on the Render service."
                        )
                    raise RuntimeError(
                        f"Hugging Face Inference returned {resp.status_code} — {detail}"
                    )
                if resp.status_code == 404:
                    last_err = RuntimeError(f"404 for {url}")
                    continue
                if resp.status_code in (429, 503):
                    wait = min(2 ** attempt, 20)
                    logger.warning("HF %s — sleeping %ss", resp.status_code, wait)
                    time.sleep(wait)
                    last_err = RuntimeError(resp.text[:300])
                    continue
                resp.raise_for_status()
                return _flatten_embedding(resp.json())
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("embed failed via %s: %s", url, exc)
        time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Failed to embed text after retries: {last_err}") from last_err


def embed_texts(
    texts: Sequence[str],
    model: str | None = None,
    *,
    max_retries: int = 5,
    timeout: float = 60.0,
) -> List[List[float]]:
    """Embed texts with HF Inference API (same model for build + query)."""
    model = model or os.getenv("EMBED_MODEL", DEFAULT_MODEL)
    headers = {"Content-Type": "application/json"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        logger.warning(
            "No HF_API_TOKEN/HF_TOKEN set — HF Inference now requires a token, "
            "so unauthenticated calls fail with 401. "
            "Add HF_API_TOKEN to .env for reliable builds and Render deploys."
        )
    vectors: List[List[float]] = []
    with httpx.Client(timeout=timeout) as client:
        for i, text in enumerate(texts):
            logger.info("Embedding %s/%s …", i + 1, len(texts))
            vectors.append(_embed_one(client, text, model, headers, max_retries))
            # gentle pacing for free-tier HF
            if i + 1 < len(texts):
                time.sleep(0.35)
    return vectors


def embed_query(text: str, model: str | None = None) -> List[float]:
    return embed_texts([text], model=model)[0]
