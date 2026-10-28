"""Tiny BM25 (Okapi) over retrieval chunks — pure stdlib, no new dependency.

Used by the hybrid retriever: exact-term matching (company names, "FAISS",
"Zibtek") is where dense MiniLM vectors are weakest on a 91–109 chunk corpus,
so keyword scores are fused with vector scores via Reciprocal Rank Fusion.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small English stop list: without it, "what is your" matches everything
# and drowns the content terms.
STOP_WORDS = frozenset(
    """
    a an the and or of to in on for with at by from as is are was were be been
    being have has had do does did will would shall should can could may might
    i you he she it we they this that these those there here what which who
    whom whose when where why how all any both each few more most other some
    such no nor not only own same so than too very just about into through
    after over under again further then once your his her its our their me
    him us them my mine yours ours theirs am
    """.split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


class BM25Index:
    """BM25Okapi (k1=1.5, b=0.75) over a fixed list of documents."""

    def __init__(self, docs: Sequence[str]):
        self.docs = list(docs)
        self.doc_tokens = [tokenize(d) for d in self.docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.doc_freq: Dict[str, int] = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                self.doc_freq[t] += 1
        self.num_docs = len(self.docs)
        # Precompute IDF once.
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freq.items()
        }

    def scores(self, query: str) -> List[float]:
        q_terms = tokenize(query)
        if not q_terms or not self.docs:
            return [0.0] * len(self.docs)
        k1, b = 1.5, 0.75
        out = [0.0] * len(self.docs)
        for qi, term in enumerate(q_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for di, toks in enumerate(self.doc_tokens):
                tf = toks.count(term)
                if not tf:
                    continue
                denom = tf + k1 * (1 - b + b * self.doc_len[di] / max(1, self.avg_len))
                out[di] += idf * tf * (k1 + 1) / denom
        return out

    def top(self, query: str, k: int) -> List[int]:
        """Indices of the top-k docs by BM25 score, best first (score > 0 only)."""
        s = self.scores(query)
        ranked = sorted(
            (i for i, v in enumerate(s) if v > 0),
            key=lambda i: (-s[i], i),
        )
        return ranked[:k]
