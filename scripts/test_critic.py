#!/usr/bin/env python3
"""Tests for the Tier 3 answer critic (rag/critic.py + app.py wiring).

Run from the repo root with the project venv (stdlib unittest only):
    .venv-critic/bin/python scripts/test_critic.py

The Groq client is stubbed (no real key exists in the sandbox), but
retrieval is REAL: the local FAISS index + ONNX embedder answer the eval
questions, so the critic sees the same chunks the live service sees.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "fake-key-for-tests")

import app as app_module  # noqa: E402
from rag import critic as critic_mod  # noqa: E402

# --- Live-eval answer texts (workspace/portfolio/eval-live-2026-10-28.json) ---
LIVE_PYTORCH_ANSWER = (
    "Yes. Sai Vishnu Vamsi Senagasetty has deployed PyTorch in production environments. "
    "He built a full-stack Text-to-Image pipeline (ImageCraft) that trains and serves a "
    "UNet-based Stable Diffusion model using PyTorch, and he integrated PyTorch CUDA wheels "
    "for distributed training in the CAOP Spark + MLflow experiments."
)
LIVE_STRONGEST_SKILLS_ANSWER = (
    "Sai Vishnu Vamsi Senagasetty's core strengths lie in full-stack development, with strong "
    "proficiency in Python, TypeScript, JavaScript, and SQL. He builds modern web and mobile "
    "interfaces using React, Next.js, Vite, and Flutter, and has experience deploying REST APIs "
    "with FastAPI, NestJS, and Express. In addition, he is well-versed in AI/ML tooling, "
    "including PyTorch, TensorFlow, and Transformers."
)
LIVE_ROLES_ANSWER = (
    "Before joining Zibtek, Sai Vishnu Vamsi Senagasetty was an AI Software Engineer at "
    "boomerroomers LLC from January 2026 to April 2026. In that role he focused on developing "
    "AI-driven features for the company's products."
)
REFUSAL = "I don't have more detail on that\u2014feel free to ask about his other projects."
REPAIRED_PYTORCH = (
    "No. Sai Vishnu Vamsi Senagasetty does not have production experience with PyTorch. "
    "He has used PyTorch for project work, including the ImageCraft text-to-image pipeline, "
    "where it is part of a training and inference pipeline. PyTorch is familiarity-level for "
    "him; his production systems are RAG/LLM systems built with FastAPI and FAISS."
)


REPAIRED_SKILLS = (
    "His core strengths are RAG and LLM systems, Python, and FastAPI. "
    "He builds production retrieval-augmented generation pipelines, and works with "
    "React and TypeScript on the frontend. PyTorch and TensorFlow are familiarity-level "
    "tools for him, not core strengths."
)


# --- Fake Groq client -------------------------------------------------------
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Delta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _StreamChunk:
    def __init__(self, content):
        self.choices = [_StreamChoice(content)]


class FakeCompletions:
    """Routes calls by prompt kind; behaviour is set per test."""

    def __init__(self, fake):
        self.fake = fake

    def create(self, *, model, messages, temperature=0.5, max_completion_tokens=None,
               top_p=None, stream=False, reasoning_effort=None, timeout=None,
               response_format=None, **kwargs):
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""
        if "answer critic" in system:
            self.fake.calls.append("critic")
            if self.fake.critic_behavior == "raise":
                raise RuntimeError("simulated critic outage")
            verdict = self.fake.critic_verdict(user)
            if stream:
                raise AssertionError("critic must not be called with stream=True")
            return _Completion(json.dumps(verdict))
        if "repairing a drafted answer" in system:
            self.fake.calls.append("repair")
            if self.fake.repair_behavior == "raise":
                raise RuntimeError("simulated repair outage")
            if self.fake.repair_behavior == "cannot":
                return _Completion(critic_mod.CANNOT_ANSWER)
            if "strongest skills" in user:
                return _Completion(REPAIRED_SKILLS)
            return _Completion(self.fake.repair_text)
        # Answerer call.
        self.fake.calls.append("answer")
        question = ""
        m = re.search(r"Question:\s*(.+)", user)
        if m:
            question = m.group(1).strip()
        text = self.fake.answer_text(question)
        if stream:
            words = text.split(" ")
            chunks = [" ".join(words[i:i + 6]) + " " for i in range(0, len(words), 6)]
            return iter(_StreamChunk(c) for c in chunks)
        return _Completion(text)


class FakeChat:
    def __init__(self, fake):
        self.completions = FakeCompletions(fake)


class FakeGroq:
    """Stub Groq client. Tests set answer_text/critic_verdict/repair_*."""

    def __init__(self):
        self.chat = FakeChat(self)
        self.calls = []
        self.critic_behavior = "normal"  # or "raise"
        self.repair_behavior = "corrected"  # "corrected" | "cannot" | "raise"
        self.repair_text = REPAIRED_PYTORCH

    def answer_text(self, question):
        q = question.lower()
        if "pytorch" in q:
            return LIVE_PYTORCH_ANSWER
        if "strongest skills" in q:
            return LIVE_STRONGEST_SKILLS_ANSWER
        if "before zibtek" in q:
            return LIVE_ROLES_ANSWER
        if "current role" in q:
            return REFUSAL
        return "Sai Vishnu Vamsi Senagasetty is an AI Developer at Zibtek."

    def critic_verdict(self, user_prompt):
        if "deployed PyTorch in production" in user_prompt:
            return {
                "claims": [
                    {"claim": "He has deployed PyTorch in production environments",
                     "verdict": "unsupported", "chunk_id": None},
                    {"claim": "He built the ImageCraft text-to-image pipeline",
                     "verdict": "supported", "chunk_id": "project-imagecraft"},
                ],
                "unsupported_claims": [
                    "He has deployed PyTorch in production environments"
                ],
                "verdict": "unsupported",
            }
        if "core strengths lie in full-stack development" in user_prompt:
            return {
                "claims": [
                    {"claim": "His core strengths include PyTorch and TensorFlow",
                     "verdict": "unsupported", "chunk_id": None},
                ],
                "unsupported_claims": [
                    "His core strengths include PyTorch and TensorFlow"
                ],
                "verdict": "unsupported",
            }
        return {"claims": [], "verdict": "supported", "unsupported_claims": []}


def real_chunks(*ids):
    """Chunk dicts (id + text) from the shipped chunks.jsonl."""
    out = []
    wanted = set(ids)
    with open(ROOT / "data" / "faiss_index" / "chunks.json") as fh:
        data = json.load(fh)
    chunks = data if isinstance(data, list) else data.get("chunks", [])
    for c in chunks:
        if c.get("id") in wanted:
            out.append({"id": c["id"], "text": c.get("text", "")})
    return out


class CriticUnitTests(unittest.TestCase):
    def test_parse_bare_json(self):
        raw = '{"claims": [], "verdict": "supported", "unsupported_claims": []}'
        self.assertEqual(critic_mod.parse_critic_json(raw)["verdict"], "supported")

    def test_parse_fenced_json(self):
        raw = '```json\n{"claims": [], "verdict": "unsupported", "unsupported_claims": ["x"]}\n```'
        parsed = critic_mod.parse_critic_json(raw)
        self.assertEqual(parsed["verdict"], "unsupported")
        self.assertEqual(parsed["unsupported_claims"], ["x"])

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(critic_mod.parse_critic_json("not json at all"))
        self.assertIsNone(critic_mod.parse_critic_json(""))

    def test_parse_derives_unsupported_claims(self):
        raw = json.dumps({"claims": [{"claim": "c1", "verdict": "unsupported", "chunk_id": None}],
                          "verdict": "unsupported"})
        parsed = critic_mod.parse_critic_json(raw)
        self.assertEqual(parsed["unsupported_claims"], ["c1"])

    def test_deterministic_flags_live_pytorch_overclaim(self):
        chunks = real_chunks("experience-zibtek-ai-part-3", "project-caop-spark",
                             "skills-cat-ml-dl", "project-imagecraft")
        self.assertEqual(len(chunks), 4)
        findings = critic_mod.deterministic_exposure_findings(LIVE_PYTORCH_ANSWER, chunks)
        self.assertTrue(any(f["skill"] == "pytorch" for f in findings),
                        f"expected a pytorch finding, got {findings}")

    def test_deterministic_no_false_positive_faq0(self):
        # faq-0 holds "production" and "PyTorch" in one chunk but ~140 chars
        # apart, in different clauses: it must NOT support a production claim,
        # and a correct sentence must not trigger either.
        chunks = real_chunks("faq-0")
        draft = "He specializes in production RAG/LLM systems, Python, and PyTorch."
        self.assertEqual(critic_mod.deterministic_exposure_findings(draft, chunks), [])

    def test_deterministic_skips_negation(self):
        # The repaired answer denies production experience: not a claim.
        chunks = real_chunks("project-imagecraft")
        self.assertEqual(
            critic_mod.deterministic_exposure_findings(REPAIRED_PYTORCH, chunks), [])

    def test_deterministic_correct_production_sentence_not_flagged(self):
        chunks = real_chunks("faq-0", "project-imagecraft")
        draft = "He builds production RAG systems with Python and PyTorch."
        self.assertEqual(critic_mod.deterministic_exposure_findings(draft, chunks), [])


class CriticEndpointTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeGroq()
        self._orig = app_module.get_groq_client
        app_module.get_groq_client = lambda: self.fake
        self.client = app_module.app.test_client()
        self._env = mock.patch.dict(os.environ, {"RAG_CRITIC_ENABLED": "true"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        app_module.get_groq_client = self._orig

    def _post(self, message):
        return self.client.post("/receive", json={
            "message": message, "session_id": "test-critic-1"})

    def test_pytorch_overclaim_is_repaired(self):
        resp = self._post("Do you have production experience with PyTorch?")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertFalse(body.get("abstained"), body)
        self.assertEqual(body.get("critic_status"), "repaired", body)
        self.assertNotIn("deployed PyTorch in production", body["response"])
        self.assertNotIn("has deployed PyTorch", body["response"])
        self.assertIn("does not have production experience with PyTorch",
                      body["response"])
        # Budget: 1 answer + 1 critic + 1 repair call.
        self.assertEqual(self.fake.calls, ["answer", "critic", "repair"],
                         self.fake.calls)

    def test_strongest_skills_repaired(self):
        resp = self._post("What are your strongest skills?")
        body = resp.get_json()
        self.assertEqual(body.get("critic_status"), "repaired", body)
        self.assertIn("RAG", body["response"])
        self.assertNotIn("core strengths lie in full-stack development",
                         body["response"])
        self.assertIn("familiarity-level", body["response"])

    def test_repair_failure_abstains(self):
        self.fake.repair_behavior = "cannot"
        resp = self._post("Do you have production experience with PyTorch?")
        body = resp.get_json()
        self.assertTrue(body.get("abstained"), body)
        self.assertEqual(body.get("critic_status"), "abstained", body)
        self.assertEqual(body["response"], app_module.ABSTAIN_MESSAGE)

    def test_good_answer_unchanged(self):
        resp = self._post("Where did you work before Zibtek?")
        body = resp.get_json()
        self.assertEqual(body.get("critic_status"), "ok", body)
        self.assertEqual(body["response"], LIVE_ROLES_ANSWER)
        self.assertEqual(self.fake.calls, ["answer", "critic"])

    def test_critic_outage_fails_open_logged(self):
        self.fake.critic_behavior = "raise"
        with self.assertLogs("rag.critic", level="WARNING"):
            resp = self._post("Where did you work before Zibtek?")
        body = resp.get_json()
        self.assertEqual(body.get("critic_status"), "critic-error", body)
        self.assertEqual(body["response"], LIVE_ROLES_ANSWER)

    def test_critic_disabled(self):
        with mock.patch.dict(os.environ, {"RAG_CRITIC_ENABLED": "false"}):
            resp = self._post("Do you have production experience with PyTorch?")
        body = resp.get_json()
        self.assertEqual(body.get("critic_status"), "disabled", body)
        self.assertEqual(body["response"], LIVE_PYTORCH_ANSWER)
        self.assertEqual(self.fake.calls, ["answer"])

    def test_refusal_passes_through_unchanged(self):
        # current-role: retrieval fails, the model refuses. The critic
        # checks claims; a refusal makes none, so it passes through.
        # (The critic does NOT fix retrieval failures — honest scope.)
        resp = self._post("What is your current role?")
        body = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(body["response"], REFUSAL)
        self.assertEqual(body.get("critic_status"), "ok", body)

    def test_abstain_questions_make_no_groq_call(self):
        for question in ("What is the weather in Paris?",
                         "What is the capital of France?"):
            before = len(self.fake.calls)
            resp = self._post(question)
            body = resp.get_json()
            self.assertTrue(body.get("abstained"), body)
            self.assertEqual(body.get("critic_status"), "not-run", body)
            self.assertEqual(len(self.fake.calls), before,
                             "abstain must not call Groq at all")

    def _collect_stream(self, message):
        resp = self.client.post("/receive/stream", json={
            "message": message, "session_id": "test-critic-stream"})
        self.assertEqual(resp.status_code, 200)
        tokens, done = [], None
        for line in resp.response:
            if isinstance(line, bytes):
                line = line.decode()
            for part in line.split("\n\n"):
                part = part.strip()
                if part.startswith("data: "):
                    payload = json.loads(part[6:])
                    if "token" in payload:
                        tokens.append(payload["token"])
                    elif payload.get("done"):
                        done = payload
        return "".join(tokens), done

    def test_stream_pytorch_repaired_never_shows_overclaim(self):
        text, done = self._collect_stream(
            "Do you have production experience with PyTorch?")
        self.assertNotIn("deployed PyTorch in production", text)
        self.assertIn("does not have production experience with PyTorch", text)
        self.assertEqual(done.get("critic_status"), "repaired", done)
        self.assertFalse(done.get("abstained"))

    def test_stream_repair_failure_abstains(self):
        self.fake.repair_behavior = "cannot"
        text, done = self._collect_stream(
            "Do you have production experience with PyTorch?")
        self.assertEqual(text, app_module.ABSTAIN_MESSAGE)
        self.assertTrue(done.get("abstained"))
        self.assertEqual(done.get("critic_status"), "abstained", done)


if __name__ == "__main__":
    unittest.main(verbosity=2)
