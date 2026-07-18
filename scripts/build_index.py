#!/usr/bin/env python3
"""Build FAISS index offline (laptop). Uses HF Inference API — no local PyTorch required.

Usage (from repo root):
  set HF_API_TOKEN=...
  python scripts/build_index.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunking import build_chunks, load_knowledge  # noqa: E402
from rag.embeddings import embed_texts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_index")


def main() -> None:
    load_dotenv(ROOT / ".env")
    knowledge_path = ROOT / "data" / "knowledge.json"
    out_dir = ROOT / "data" / "faiss_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    knowledge = load_knowledge(knowledge_path)
    chunks = build_chunks(knowledge)
    texts = [c["text"] for c in chunks]
    logger.info("Built %s chunks from %s", len(chunks), knowledge_path)

    vectors = embed_texts(texts)
    matrix = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # debug dump
    with open(ROOT / "data" / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info(
        "Saved FAISS index (%s vectors, dim=%s) → %s",
        index.ntotal,
        matrix.shape[1],
        out_dir,
    )


if __name__ == "__main__":
    main()
