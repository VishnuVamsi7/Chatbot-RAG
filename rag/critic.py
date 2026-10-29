"""Answer critic (Tier 3): verify the drafted answer against the retrieved chunks.

Design (roadmap §7): Router (rules, 0 calls) -> Retriever -> Answerer ->
Critic (one cheap Groq call, JSON verdict) -> one repair pass max -> abstain.
No agent framework, no LangGraph — one function + one Groq call in the
existing /receive path.

Two layers, cheapest first:

1. Deterministic exposure check (free, no LLM call). Skills that are
   familiarity only (PyTorch, TensorFlow, Kubernetes, Spark, Hadoop,
   Databricks, Palantir Foundry, Power BI, Tableau) must never be presented
   as production experience or as a core/strongest skill. A draft sentence
   that puts one of those skills within ~60 characters of a production
   marker ("deployed", "in production", "production-grade", ...) is a
   production claim; it is supported only if some retrieved chunk makes the
   same tight claim (skill + marker within ~30 characters in one sentence).
   The 60-character window matters: faq-0 contains both "production" and
   "PyTorch" in one chunk ("specializes in production RAG/LLM systems ...
   TensorFlow/PyTorch") but ~140 characters apart, in different clauses —
   it does NOT support a PyTorch-production claim, and the window keeps it
   from counting as support. The ImageCraft chunk says "PyTorch
   training/inference pipeline" with no production marker at all, so the
   live PyTorch overclaim ("deployed PyTorch in production environments")
   fails this check deterministically. The window is 30 characters: tight
   enough that a correct sentence like "He builds production RAG systems
   with Python and PyTorch" (production-to-PyTorch ~40 chars) does not
   trigger, while "deployed PyTorch in production" (~10 chars) and
   "production experience with PyTorch" (~27 chars) do. A sentence that
   negates the claim ("does not have production experience with PyTorch")
   is not a claim at all and is skipped.

2. LLM critic (one Groq call, gpt-oss-20b, reasoning_effort="low",
   response_format JSON). It lists every factual claim in the draft and
   marks each supported / unsupported against the cited chunks, with rules
   for the two failure modes the eval exposed:
   - exposure-tier skills are familiarity only unless a chunk explicitly
     says otherwise; "strongest / core strength / expert" needs a chunk
     that ranks the skill — a skills list does not support a ranking claim;
   - a claim is supported only if a chunk states the same fact (same
     employer, project, metric, date). Anything no chunk states is
     unsupported. A draft that declines to answer ("I don't have more
     detail") makes no claims and passes.

If either layer flags an unsupported claim: ONE repair pass (one more Groq
call) rewrites the answer using only supported claims, dropping the flagged
claims entirely — and flipping to a plain "No" where the chunks support the
negation (e.g. "No, he does not have production experience with PyTorch").
The repaired draft is re-checked with the deterministic layer plus a
flagged-claim-absence check (the flagged claim text must be gone). If it
still fails, the answer abstains with the existing friendly message
(abstained: true). Worst case per answer: 1 critic call + 1 repair call.

Failure policy: if the critic *call itself* errors (timeout, API error,
unparseable JSON), fail OPEN to the un-criticised answer and log a warning
— a critic outage must not take the assistant down. A verdict of
"unsupported" never fails open.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Skills that appear only in a skills list / certification: familiarity only.
# Never production experience, never a core/strongest skill, unless a chunk
# explicitly says so. Must match SYSTEM_RULES in app.py.
EXPOSURE_SKILLS = (
    "pytorch",
    "tensorflow",
    "kubernetes",
    "spark",
    "hadoop",
    "databricks",
    "palantir",
    "power bi",
    "tableau",
)

# Words that, next to an exposure skill, assert production experience.
_PRODUCTION_WORDS = (
    "production",
    "deployed",
    "deploy",
    "deployment",
    "serving",
    "serves",
    "production-grade",
    "production experience",
)
_PRODUCTION_RE = re.compile(
    r"(production(?:-grade| experience)?|deployed|deploy(?:ment)?|serving|serves)",
    re.IGNORECASE,
)

# A production claim is a skill + production marker within this many
# characters of each other in one sentence. Tight enough that faq-0
# ("production RAG/LLM systems ... TensorFlow/PyTorch", ~140 chars apart)
# does not count as support for a PyTorch-production claim, and a correct
# sentence like "He builds production RAG systems with Python and PyTorch"
# (~40 chars) does not trigger — while "deployed PyTorch in production"
# (~10 chars) and "production experience with PyTorch" (~27 chars) do.
_CLAIM_WINDOW_CHARS = 30

# Negations that turn a production mention into a denial, not a claim.
_NEGATION_RE = re.compile(
    r"\b(not|no|never|doesn'?t|does not|don'?t|do not|isn'?t|is not|wasn'?t|was not|without|lack(?:s|ing)?|no production)\b",
    re.IGNORECASE,
)

# Critic call budget: short prompt, JSON-only, strict timeout. gpt-oss-20b
# is a reasoning model; reasoning_effort="low" keeps the call cheap.
CRITIC_TIMEOUT_SECONDS = 15.0
CRITIC_MAX_TOKENS = 1024
_CHUNK_TRUNC_CHARS = 1500

CRITIC_SYSTEM = """You are the answer critic for a portfolio Q&A assistant. \
You check a drafted answer against the retrieved source chunks. \
Return JSON only, no other text.

Rules:
- List every FACTUAL claim in the draft (employers, roles, projects, skills, \
metrics, dates, education). General connective phrasing is not a claim. A draft \
that declines to answer ("I don't have more detail") makes no claims: return \
an empty claims list with verdict "supported".
- A claim is "supported" only if a cited chunk states the same fact (same \
employer, same project, same metric, same date). Anything no chunk states is \
"unsupported". Give the supporting chunk id for supported claims, null otherwise.
- PyTorch, TensorFlow, Kubernetes, Spark, Hadoop, Databricks, Palantir Foundry, \
Power BI, and Tableau are familiarity only. A claim of production experience \
with any of them, or that any of them is a core/strongest skill, is \
"unsupported" unless a chunk explicitly says so.
- Superlatives ("strongest", "best", "core strength", "expert in") need a \
chunk that ranks the skill or calls it a top strength. A skills list does not \
support a ranking claim.
- Numbers and percentages must appear in a chunk (same value) to be supported.

Return exactly this shape:
{"claims": [{"claim": "<short claim>", "verdict": "supported|unsupported", "chunk_id": "<id or null>"}], "unsupported_claims": ["<claim text>", ...], "verdict": "supported|unsupported"}"""

REPAIR_SYSTEM = """You are repairing a drafted answer for a portfolio Q&A assistant. \
Rewrite it using ONLY facts stated in the retrieved source chunks below. \
Drop every claim listed as unsupported — do not restate it in softer form. \
If the chunks show an unsupported claim is false, say so plainly \
(e.g. "No, he does not have production experience with PyTorch"). \
Stay in the third person about Sai Vishnu Vamsi Senagasetty, keep the \
requested length, and never invent employers, metrics, or project claims. \
If the chunks cannot answer the question at all, output exactly: CANNOT_ANSWER"""

CANNOT_ANSWER = "CANNOT_ANSWER"


def _sentences(text: str) -> List[str]:
    """Naive sentence split; good enough for a proximity check."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _skill_production_spans(sentence: str) -> List[Tuple[str, int]]:
    """(skill, position) for each exposure skill that has a production
    marker within _CLAIM_WINDOW_CHARS of it in this sentence.

    A sentence that negates the claim ("does not have production
    experience with PyTorch") is not a claim and is skipped entirely.
    """
    if _NEGATION_RE.search(sentence):
        return []
    found = []
    lowered = sentence.lower()
    for skill in EXPOSURE_SKILLS:
        for m in re.finditer(r"\b" + re.escape(skill) + r"\b", lowered):
            start = max(0, m.start() - _CLAIM_WINDOW_CHARS)
            end = min(len(sentence), m.end() + _CLAIM_WINDOW_CHARS)
            if _PRODUCTION_RE.search(sentence[start:end]):
                found.append((skill, m.start()))
    return found


def deterministic_exposure_findings(
    draft: str, chunks: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Exposure-tier skills the draft presents as production experience
    with no chunk support. Returns [{skill, sentence}] per unsupported claim.

    Support requires the same tight claim in a chunk: the skill and a
    production marker within _CLAIM_WINDOW_CHARS in one chunk sentence.
    """
    chunk_sentences: List[str] = []
    for c in chunks:
        chunk_sentences.extend(_sentences(str(c.get("text", ""))))
    chunk_supports = {
        skill
        for sent in chunk_sentences
        for skill, _ in _skill_production_spans(sent)
    }
    findings: List[Dict[str, str]] = []
    seen = set()
    for sentence in _sentences(draft):
        for skill, _ in _skill_production_spans(sentence):
            if skill not in chunk_supports and (skill, sentence) not in seen:
                seen.add((skill, sentence))
                findings.append({"skill": skill, "sentence": sentence})
    return findings


def parse_critic_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the critic's JSON verdict, tolerating fences and stray text."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("verdict", None)
    claims = parsed.get("claims")
    if claims is None:
        parsed["claims"] = []
    if not isinstance(parsed["claims"], list):
        return None
    unsupported = parsed.get("unsupported_claims")
    if unsupported is None:
        unsupported = [
            str(c.get("claim", ""))
            for c in parsed["claims"]
            if isinstance(c, dict) and c.get("verdict") == "unsupported"
        ]
        parsed["unsupported_claims"] = unsupported
    return parsed


def _chunks_block(chunks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{c.get('id', i)}]\n{str(c.get('text', ''))[:_CHUNK_TRUNC_CHARS]}"
        for i, c in enumerate(chunks)
    )


def _critic_completion(client, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
    """One Groq chat completion; returns the stripped text content."""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_completion_tokens=kwargs.get("max_completion_tokens", CRITIC_MAX_TOKENS),
        reasoning_effort="low",
        timeout=CRITIC_TIMEOUT_SECONDS,
        **({"response_format": {"type": "json_object"}} if kwargs.get("json_mode") else {}),
    )
    return (completion.choices[0].message.content or "").strip()


def critique_answer(
    client, model: str, question: str, draft: str, chunks: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run the LLM critic. Returns (verdict, error); exactly one is None.

    verdict: {"verdict": "supported"|"unsupported", "claims": [...],
              "unsupported_claims": [...]}. error is a short string when the
    call itself failed or the JSON was unparseable.
    """
    prompt = (
        f"Question: {question}\n\nDraft answer:\n{draft}\n\n"
        f"Retrieved source chunks:\n{_chunks_block(chunks)}"
    )
    try:
        raw = _critic_completion(
            client,
            model,
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
        )
    except Exception as exc:  # noqa: BLE001 - critic outage must fail open (logged)
        logger.warning("Critic call failed: %s", exc)
        return None, f"critic call failed: {exc}"
    parsed = parse_critic_json(raw)
    if parsed is None:
        logger.warning("Critic returned unparseable JSON: %r", raw[:300])
        return None, "critic returned unparseable JSON"
    if parsed.get("verdict") not in ("supported", "unsupported"):
        # Derive the overall verdict from the claims when the model omits it.
        parsed["verdict"] = (
            "unsupported"
            if any(
                isinstance(c, dict) and c.get("verdict") == "unsupported"
                for c in parsed["claims"]
            )
            else "supported"
        )
    return parsed, None


def repair_answer(
    client,
    model: str,
    question: str,
    draft: str,
    unsupported_claims: List[str],
    chunks: List[Dict[str, Any]],
    verbosity_instruction: str = "Answer concisely in 2-4 sentences.",
) -> Tuple[Optional[str], Optional[str]]:
    """One repair pass: rewrite using only supported claims.
    Returns (repaired_text, error); repaired_text is CANNOT_ANSWER when the
    chunks cannot answer the question."""
    claims_block = "\n".join(f"- {c}" for c in unsupported_claims) or "- (none listed)"
    prompt = (
        f"Question: {question}\n\nDraft answer (do NOT reuse its unsupported claims):\n"
        f"{draft}\n\nClaims to drop entirely:\n{claims_block}\n\n"
        f"Retrieved source chunks:\n{_chunks_block(chunks)}\n\n{verbosity_instruction}"
    )
    try:
        raw = _critic_completion(
            client,
            model,
            [
                {"role": "system", "content": REPAIR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2048,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Repair call failed: %s", exc)
        return None, f"repair call failed: {exc}"
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    return text, None


def _flagged_claims_present(text: str, unsupported_claims: List[str]) -> List[str]:
    """Which flagged claims still appear in the text (fuzzy: first 40 chars
    of the claim, normalized). Used for the post-repair re-check."""
    normalized = re.sub(r"\s+", " ", (text or "").lower())
    still = []
    for claim in unsupported_claims:
        key = re.sub(r"\s+", " ", claim.lower()).strip()[:40]
        if len(key) >= 15 and key in normalized:
            still.append(claim)
    return still


def verify_answer(
    client,
    model: str,
    question: str,
    draft: str,
    chunks: List[Dict[str, Any]],
    verbosity_instruction: str = "Answer concisely in 2-4 sentences.",
) -> Tuple[Optional[str], str]:
    """Verify a drafted answer. Returns (final_answer, status).

    status: "ok" (verified as-is) | "repaired" (one repair pass, re-checked)
    | "abstained" (final_answer None: still unsupported after the repair)
    | "critic-error" (final_answer is the original draft: the critic call
    itself failed, logged, fail open — never on a verdict of "unsupported").
    """
    det_findings = deterministic_exposure_findings(draft, chunks)
    verdict, error = critique_answer(client, model, question, draft, chunks)
    if error is not None:
        # Critic outage: fail open to the draft, loudly logged. The
        # deterministic layer already ran; if IT found an unsupported
        # production claim, that verdict stands — never fail open on it.
        if det_findings:
            logger.warning(
                "Critic errored but deterministic check found unsupported "
                "exposure claims (%s); abstaining.",
                [f["skill"] for f in det_findings],
            )
            return None, "abstained"
        logger.warning("Critic errored (%s); returning un-criticised answer.", error)
        return draft, "critic-error"

    unsupported_claims: List[str] = list(verdict.get("unsupported_claims") or [])
    for f in det_findings:
        unsupported_claims.append(
            f"Production experience with {f['skill']}: {f['sentence']}"
        )
    if verdict.get("verdict") != "unsupported" and not det_findings:
        return draft, "ok"

    # One repair pass max.
    repaired, repair_error = repair_answer(
        client, model, question, draft, unsupported_claims, chunks, verbosity_instruction
    )
    if repair_error is not None or repaired is None:
        logger.warning("Repair failed (%s); abstaining.", repair_error)
        return None, "abstained"
    if repaired.strip() == CANNOT_ANSWER:
        return None, "abstained"

    # Re-check the repaired draft: deterministic layer + flagged claims gone.
    still_det = deterministic_exposure_findings(repaired, chunks)
    still_flagged = _flagged_claims_present(repaired, unsupported_claims)
    if still_det or still_flagged:
        logger.warning(
            "Repaired answer still fails checks (deterministic=%s, flagged=%s); abstaining.",
            [f["skill"] for f in still_det],
            still_flagged,
        )
        return None, "abstained"
    return repaired, "repaired"
