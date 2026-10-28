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

from rag.bm25 import BM25Index
from rag.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/faiss_index")

# Reciprocal Rank Fusion constant (standard k=60).
RRF_K = 60
# Score multiplier for chunks whose section matches the question's intent.
SECTION_BOOST = 1.25
# Stronger multiplier for the exact company a question names
# ("Tell me about your work at boomerroomers" must surface that role, not a
# resume-builder project that happens to share generic words).
COMPANY_BOOST = 1.5

# Keyword rules mapping a question to the sections most likely to answer it.
# Rules-based (0 LLM calls); the boost is a tie-breaker on top of RRF, not a
# filter — a strong cross-section hit can still win.
_INTENT_RULES = [
    (
        {"experience", "experience-index"},
        ["work", "worked", "job", "role", "company", "employer", "before",
         "previous", "career", "experience", "position", "join"],
    ),
    (
        {"project", "projects-index", "project-pipeline"},
        ["project", "built", "build", "strongest", "best", "portfolio"],
    ),
    (
        {"skills", "skills-category"},
        ["skill", "skills", "know", "proficient", "years", "tech", "stack",
         "tools", "languages"],
    ),
    (
        {"education"},
        ["degree", "education", "university", "college", "study", "studied",
         "school", "masters", "bachelor"],
    ),
    (
        {"contact"},
        ["email", "contact", "phone", "linkedin", "github", "reach"],
    ),
    (
        {"faq"},
        ["what is rag", "what is", "how does"],
    ),
]


def classify_intent(query: str) -> set:
    """Sections the question is most likely about (may be empty)."""
    q = query.lower()
    sections: set = set()
    for wanted, keywords in _INTENT_RULES:
        if any(kw in q for kw in keywords):
            sections |= wanted
    return sections



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
        self.bm25: Optional[BM25Index] = None
        self.companies: set = set()

    def _build_auxiliary(self) -> None:
        """BM25 index + company-name set, rebuilt whenever chunks change."""
        self.bm25 = BM25Index([c.get("text", "") for c in self.chunks])
        self.companies = {
            str(c["metadata"].get("company", "")).lower()
            for c in self.chunks
            if c.get("metadata", {}).get("company")
        }
        logger.info(
            "Built BM25 index over %s chunks (%s companies)",
            len(self.chunks),
            len(self.companies),
        )

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
            self._build_auxiliary()
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
        self._build_auxiliary()
        logger.info(
            "Built FAISS index with %s vectors / %s chunks",
            self.index.ntotal,
            len(self.chunks),
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search: dense vector ranking + BM25 keyword ranking, fused
        with Reciprocal Rank Fusion, then section/company boosts.

        Returns the same hit shape as before (id/text/metadata/score), where
        ``score`` is the fused RRF score and ``vector_score`` is the raw
        cosine similarity (kept for the abstain threshold, which is
        calibrated on cosine, not on the fused scale).
        """
        if self.index is None:
            self.load()
        assert self.index is not None
        vec = np.array([embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(vec)
        pool = min(top_k * 2, self.index.ntotal)
        scores, idxs = self.index.search(vec, pool)
        vector_rank = {int(idx): rank + 1 for rank, idx in enumerate(idxs[0]) if idx >= 0}
        vector_score = {int(idx): float(s) for idx, s in zip(idxs[0], scores[0]) if idx >= 0}

        bm25_rank: Dict[int, int] = {}
        if self.bm25 is not None:
            for rank, idx in enumerate(self.bm25.top(query, pool)):
                bm25_rank[idx] = rank + 1

        intent_sections = classify_intent(query)
        named_company = next(
            (c for c in self.companies if c and c in query.lower()), None
        )

        fused: Dict[int, float] = {}
        for idx in set(vector_rank) | set(bm25_rank):
            s = 0.0
            if idx in vector_rank:
                s += 1.0 / (RRF_K + vector_rank[idx])
            if idx in bm25_rank:
                s += 1.0 / (RRF_K + bm25_rank[idx])
            chunk = self.chunks[idx]
            section = (chunk.get("metadata") or {}).get("section")
            if named_company and str(
                (chunk.get("metadata") or {}).get("company", "")
            ).lower() == named_company:
                s *= COMPANY_BOOST
            elif section in intent_sections:
                s *= SECTION_BOOST
            fused[idx] = s

        results: List[Dict[str, Any]] = []
        for idx in sorted(fused, key=lambda i: (-fused[i], i))[:top_k]:
            chunk = self.chunks[idx]
            results.append(
                {
                    "id": chunk.get("id"),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                    "score": float(fused[idx]),
                    "vector_score": vector_score.get(idx, 0.0),
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
