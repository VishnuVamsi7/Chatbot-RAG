"""FAISS retriever backed by offline index + HF query embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from rag.embeddings import embed_query

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/faiss_index")


class FaissRetriever:
    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []

    def load(self) -> None:
        index_path = self.index_dir / "index.faiss"
        chunks_path = self.index_dir / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"FAISS index missing under {self.index_dir}. "
                "Run: python scripts/build_index.py"
            )
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        logger.info("Loaded FAISS index with %s vectors / %s chunks", self.index.ntotal, len(self.chunks))

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
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
