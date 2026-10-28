#!/usr/bin/env python3
"""Eval runner for the portfolio RAG assistant.

Runs every question in scripts/eval_set.json against a live backend and
scores three things per question:
  1. facts — every expected fact (any keyword variant) appears in the answer.
  2. support — at least one expected chunk id is in retrieved[].
  3. grounded numbers — every number in the answer also appears in the
     retrieved chunk texts (a number the chunks don't contain is, by
     construction, ungrounded).
Abstain questions pass only when the backend abstains (or the answer says
it doesn't have that information) and no fact is asserted.

This is a deterministic check, not an LLM judge: it catches wrong facts,
missing retrieval, and invented numbers — the failure modes that matter for
a recruiter-facing assistant. Run it after every knowledge.json change:

  python scripts/run_eval.py --base-url https://chatbot-rag-mun6.onrender.com

Rate limits are 20/min on /receive, so questions are spaced --delay seconds
apart (default 3.5s). Results print as a table and, with --out, as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET = ROOT / "scripts" / "eval_set.json"

ABSTAIN_PHRASES = [
    "don't have",
    "do not have",
    "not in his profile",
    "no information",
    "can't answer",
    "cannot answer",
    "isn't in",
    "not something i have",
    "i don't know",
]

_NUMBER_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _norm(text: str) -> str:
    # Normalize curly quotes/dashes so checks match model output either way.
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return re.sub(r"\s+", " ", text.lower()).strip()


def _numbers(text: str) -> list[str]:
    """Digit groups in an answer, normalized (%, commas, $ stripped)."""
    out = []
    for m in _NUMBER_RE.findall(text):
        digits = re.sub(r"[^\d.]", "", m)
        if digits:
            out.append(digits)
    return out


def _post(base_url: str, question: str) -> dict:
    body = json.dumps(
        {"message": question, "session_id": f"eval-{uuid.uuid4().hex[:12]}"}
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/receive",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            return {"_http_error": exc.code, **json.loads(exc.read())}
        except Exception:
            return {"_http_error": exc.code, "error": "HTTP error"}


def check_question(q: dict, result: dict) -> dict:
    answer = (result.get("response") or "").strip()
    retrieved = result.get("retrieved") or []
    retrieved_ids = [r.get("id") for r in retrieved]
    abstained = bool(result.get("abstained"))
    answer_n = _norm(answer)

    detail: dict = {"id": q["id"], "question": q["question"], "answer": answer[:400]}

    if q.get("abstain"):
        said_no = abstained or any(p in answer_n for p in ABSTAIN_PHRASES)
        detail["abstained"] = said_no
        detail["passed"] = bool(said_no)
        return detail

    # 1. Expected facts present.
    missing_facts = []
    for variants in q.get("expected", []):
        if not any(_norm(v) in answer_n for v in variants):
            missing_facts.append(" / ".join(variants))
    detail["missing_facts"] = missing_facts

    # 2. Forbidden phrases absent.
    found_forbidden = [f for f in q.get("forbidden", []) if _norm(f) in answer_n]
    detail["forbidden_found"] = found_forbidden

    # 3. At least one expected chunk retrieved.
    want = q.get("support_chunks", [])
    detail["support_hit"] = any(c in retrieved_ids for c in want) if want else None
    detail["retrieved_top"] = retrieved_ids[:3]

    # 4. Numbers in the answer must appear in the retrieved chunk texts.
    # (The runner only sees chunk ids, not texts — it re-reads chunks.jsonl.)
    chunk_texts = _chunk_texts()
    retrieved_digits = " ".join(
        re.sub(r"[^\d. ]", " ", _norm(chunk_texts.get(cid, "")))
        for cid in retrieved_ids
    )
    ungrounded = []
    for num in _numbers(answer):
        # Single digits match everywhere (years, counts) — not a claim.
        if len(num) == 1:
            continue
        if not re.search(rf"(?<!\d){re.escape(num)}(?!\d)", retrieved_digits):
            ungrounded.append(num)
    detail["ungrounded_numbers"] = sorted(set(ungrounded))

    detail["passed"] = (
        not missing_facts
        and not found_forbidden
        and (detail["support_hit"] is not False)
        and not detail["ungrounded_numbers"]
    )
    return detail


_CHUNKS: dict | None = None


def _chunk_texts() -> dict:
    global _CHUNKS
    if _CHUNKS is None:
        _CHUNKS = {}
        path = ROOT / "data" / "chunks.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    c = json.loads(line)
                    _CHUNKS[c["id"]] = c.get("text", "")
                except (json.JSONDecodeError, KeyError):
                    continue
    return _CHUNKS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="https://chatbot-rag-mun6.onrender.com")
    ap.add_argument("--set", default=str(DEFAULT_SET))
    ap.add_argument("--delay", type=float, default=3.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.set, encoding="utf-8") as f:
        eval_set = json.load(f)
    questions = eval_set["questions"]
    print(f"Eval set v{eval_set.get('version', '?')}: {len(questions)} questions")
    print(f"Backend: {args.base_url}\n")

    results = []
    for i, q in enumerate(questions):
        if i:
            time.sleep(args.delay)
        res = _post(args.base_url, q["question"])
        if "_http_error" in res:
            detail = {
                "id": q["id"],
                "question": q["question"],
                "passed": False,
                "error": f"HTTP {res['_http_error']}: {res.get('error', '')}",
            }
        else:
            detail = check_question(q, res)
        results.append(detail)
        mark = "PASS" if detail.get("passed") else "FAIL"
        print(f"[{mark}] {q['id']}: {q['question'][:70]}")
        if not detail.get("passed"):
            for key in ("missing_facts", "forbidden_found", "ungrounded_numbers", "error"):
                if detail.get(key):
                    print(f"       {key}: {detail[key]}")
            if detail.get("support_hit") is False:
                print(f"       support: none of {q.get('support_chunks')} in {detail.get('retrieved_top')}")
            if q.get("abstain") and not detail.get("abstained"):
                print(f"       did not abstain; answer: {detail.get('answer', '')[:160]}")

    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{passed}/{len(results)} passed")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"passed": passed, "total": len(results), "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"Results written to {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
