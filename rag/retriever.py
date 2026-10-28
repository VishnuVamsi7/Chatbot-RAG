"""FAISS retriever backed by offline index + local ONNX query embeddings.

The shipped index (data/faiss_index/index.faiss) is a build artifact: if
chunks.json has more/fewer chunks than the index holds — e.g. after a
knowledge.json edit — the index is rebuilt from chunks.json at load time
with the same local ONNX embedder (no API, no torch). A rebuild failure
falls back to the stale shipped index instead of taking the service down.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from rag.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/faiss_index")


def _build_index_from_chunks(chunks: List[Dict[str, Any]]) -> faiss.Index:
    """Embed chunk texts with the local ONNX model and build an exact index.

    Same construction as scripts/build_index.py (IndexFlatIP over
    L2-normalized MiniLM vectors); kept here so the service can rebuild a
    stale shipped index on its own.
    """
    vectors = embed_texts([c["text"] for c in chunks])
    matrix = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


class FaissRetriever:
    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []

    def load(self) -> None:
        index_path = self.index_dir / "index.faiss"
        chunks_path = self.index_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"FAISS chunks missing under {self.index_dir}. "
                "Run: python scripts/build_index.py"
            )
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)

        shipped = None
        if index_path.exists():
            try:
                shipped = faiss.read_index(str(index_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Shipped FAISS index unreadable, rebuilding: %s", exc)
        if shipped is not None and shipped.ntotal == len(self.chunks):
            self.index = shipped
            logger.info(
                "Loaded FAISS index with %s vectors / %s chunks",
                self.index.ntotal,
                len(self.chunks),
            )
            return

        if shipped is not None:
            logger.warning(
                "Shipped FAISS index has %s vectors but chunks.json has %s chunks — rebuilding",
                shipped.ntotal,
                len(self.chunks),
            )
        else:
            logger.warning("FAISS index missing under %s — building from chunks.json", self.index_dir)
        try:
            self.index = _build_index_from_chunks(self.chunks)
            try:
                faiss.write_index(self.index, str(index_path))
            except OSError as exc:  # noqa: BLE001
                logger.warning("Could not persist rebuilt index (continuing in-memory): %s", exc)
        except Exception:  # noqa: BLE001
            if shipped is not None:
                logger.warning(
                    "Index rebuild failed, serving stale shipped index (%s vectors)", shipped.ntotal
                )
                self.index = shipped
                return
            raise
        logger.info(
            "Built FAISS index with %s vectors / %s chunks",
            self.index.ntotal,
            len(self.chunks),
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            self.load()
        assert self.index is not None
        vec = np.array([embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(vec, k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "id": chunk.get("id"),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                    "score": float(score),
                }
            )
        return results


_retriever: Optional[FaissRetriever] = None


def get_retriever() -> FaissRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FaissRetriever()
        _retriever.load()
    return _retriever
