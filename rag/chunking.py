"""Structure-aware chunking from curated knowledge.json (schema v1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_knowledge(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _join(items: List[Any], sep: str = ", ") -> str:
    return sep.join(str(x) for x in items if x)


def _add(
    chunks: List[Dict[str, Any]],
    chunk_id: str,
    text: str,
    **metadata: Any,
) -> None:
    text = " ".join((text or "").split())
    if not text:
        return
    chunks.append(
        {
            "id": chunk_id,
            "text": text,
            "metadata": metadata or {"section": chunk_id},
        }
    )


def build_chunks(knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build retrieval chunks from flat (legacy) or nested (v1) knowledge schemas."""
    if "identity" in knowledge or "projects" in knowledge and isinstance(
        knowledge.get("projects"), dict
    ):
        return _build_v1(knowledge)
    return _build_legacy(knowledge)


def _build_v1(knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    identity = knowledge.get("identity") or {}
    name = identity.get("name") or knowledge.get("meta", {}).get(
        "subject", "Sai Vishnu Vamsi Senagasetty"
    )
    titles = _join(identity.get("titles") or [])
    location = identity.get("location", "")
    pitch = knowledge.get("pitch") or ""
    hero = knowledge.get("hero") or {}
    bio = knowledge.get("bio") or {}
    bio_paras = " ".join(bio.get("paragraphs") or [])

    _add(
        chunks,
        "summary",
        (
            f"Profile summary for {name}. Titles: {titles}. "
            f"Job title: {identity.get('jobTitle', '')}. Location: {location}. "
            f"{pitch} {hero.get('tagline', '')} {bio_paras}"
        ),
        section="summary",
    )

    open_to = identity.get("openToWork") or {}
    if open_to:
        _add(
            chunks,
            "open-to-work",
            (
                f"{name} is open to work: seeking {open_to.get('seeking', '')}. "
                f"Work modes: {_join(open_to.get('workModes') or [])}. "
                f"Location preference: {open_to.get('locationPreference', '')}."
            ),
            section="open-to-work",
        )

    contact_bits = []
    for label, key in [
        ("email", "email"),
        ("phone", "phone"),
        ("LinkedIn", "linkedin"),
        ("GitHub", "github"),
        ("website", "website"),
    ]:
        if identity.get(key):
            contact_bits.append(f"{label}: {identity[key]}")
    if contact_bits:
        _add(
            chunks,
            "contact",
            f"Contact for {name}: {'; '.join(contact_bits)}.",
            section="contact",
        )

    for i, edu in enumerate(knowledge.get("education") or []):
        _add(
            chunks,
            f"education-{i}",
            (
                f"Education: {edu.get('degree')} at {edu.get('school')} "
                f"({edu.get('period', '')})."
            ),
            section="education",
            school=edu.get("school"),
        )

    skills = knowledge.get("skills") or {}
    skill_all = skills.get("all") if isinstance(skills, dict) else skills
    if isinstance(skill_all, list) and skill_all:
        # Split long skill lists so MiniLM context stays useful
        step = 40
        for i in range(0, len(skill_all), step):
            part = skill_all[i : i + step]
            _add(
                chunks,
                f"skills-{i // step}",
                f"Technical skills for {name}: {_join(part)}.",
                section="skills",
            )

    for cat in (skills.get("categories") if isinstance(skills, dict) else None) or []:
        cat_skills = _join(cat.get("skills") or [])
        _add(
            chunks,
            f"skills-cat-{cat.get('id', cat.get('name', 'cat'))}",
            f"Skill category {cat.get('name')}: {cat_skills}.",
            section="skills-category",
            category=cat.get("name"),
        )

    experience = knowledge.get("experience") or {}
    summaries = (
        experience.get("summaries")
        if isinstance(experience, dict)
        else experience
        if isinstance(experience, list)
        else []
    ) or []
    detailed = (experience.get("detailed") if isinstance(experience, dict) else []) or []
    detailed_by_id = {d.get("id"): d for d in detailed if d.get("id")}

    if summaries:
        roles = "; ".join(
            f"{e.get('title')} at {e.get('company')} ({e.get('period') or e.get('date') or ''})"
            for e in summaries
        )
        _add(
            chunks,
            "experience-index",
            f"Complete work history for {name}: {roles}.",
            section="experience-index",
        )

    for exp in summaries:
        eid = str(exp.get("id") or exp.get("company") or "role").replace(" ", "-").lower()
        exp_skills = _join(exp.get("skills") or [])
        detail = detailed_by_id.get(exp.get("id")) or {}
        diff = " ".join(detail.get("diffStats") or [])
        _add(
            chunks,
            f"experience-{eid}",
            (
                f"Experience: {exp.get('title')} at {exp.get('company')} "
                f"({exp.get('period') or detail.get('date') or ''}). "
                f"Location: {detail.get('location', '')}. Type: {detail.get('type', '')}. "
                f"{exp.get('summary', '')} Skills: {exp_skills}. {diff}"
            ),
            section="experience",
            company=exp.get("company"),
            id=exp.get("id"),
        )

    projects_block = knowledge.get("projects") or {}
    projects = (
        projects_block.get("items")
        if isinstance(projects_block, dict)
        else projects_block
        if isinstance(projects_block, list)
        else []
    ) or []

    if projects:
        listing = "; ".join(
            f"{p.get('title')} ({p.get('subtitle') or p.get('shortTitle') or ''})"
            for p in projects
        )
        _add(
            chunks,
            "projects-index",
            (
                f"Complete list of {name}'s projects ({len(projects)} total): {listing}. "
                "Ask about any specific project for full details."
            ),
            section="projects-index",
        )

    for proj in projects:
        pid = proj.get("id") or "project"
        tech = _join(proj.get("techStack") or proj.get("tech") or [])
        tags = _join(proj.get("tags") or [])
        _add(
            chunks,
            f"project-{pid}",
            (
                f"Project: {proj.get('title')} — {proj.get('subtitle', '')}. "
                f"Goal: {proj.get('goal', '')} Outcome: {proj.get('outcome', '')} "
                f"Approach: {proj.get('approach', '')} "
                f"Why this approach: {proj.get('whyApproach', '')} "
                f"How built: {proj.get('howBuilt', '')} "
                f"Evals: {proj.get('evals', '')} "
                f"Tech: {tech}. Tags: {tags}."
            ),
            section="project",
            id=pid,
            title=proj.get("title"),
        )
        pipeline = proj.get("pipeline") or []
        components = proj.get("components") or []
        if pipeline or components:
            _add(
                chunks,
                f"project-{pid}-pipeline",
                (
                    f"Project {proj.get('title')} pipeline: {_join(pipeline, sep=' | ')}. "
                    f"Components: {_join(components, sep=' | ')}."
                ),
                section="project-pipeline",
                id=pid,
                title=proj.get("title"),
            )

    for i, item in enumerate(knowledge.get("faq") or []):
        _add(
            chunks,
            f"faq-{i}",
            f"FAQ: {item.get('question')} Answer: {item.get('answer')}",
            section="faq",
        )

    for i, pub in enumerate(knowledge.get("publications") or []):
        _add(
            chunks,
            f"publication-{i}",
            (
                f"Publication: {pub.get('title')} — {pub.get('venue', '')} "
                f"({pub.get('date') or pub.get('year') or ''}). URL: {pub.get('url', '')}."
            ),
            section="publication",
        )

    for i, cert in enumerate(knowledge.get("certifications") or []):
        _add(
            chunks,
            f"certification-{i}",
            (
                f"Certification: {cert.get('title')} from {cert.get('provider', '')} "
                f"({cert.get('year', '')}). Category: {cert.get('category', '')}."
            ),
            section="certification",
        )

    for i, badge in enumerate(knowledge.get("badges") or []):
        _add(
            chunks,
            f"badge-{i}",
            (
                f"Badge: {badge.get('name')} from {badge.get('provider', '')} "
                f"({badge.get('year', '')}). {badge.get('description', '')}"
            ),
            section="badge",
        )

    chatbot = knowledge.get("chatbot") or {}
    if chatbot:
        _add(
            chunks,
            "chatbot-architecture",
            (
                f"This portfolio Chatbot-RAG: {chatbot.get('architecture', '')} "
                f"Live endpoint: {chatbot.get('liveEndpoint', '')}. "
                f"GitHub: {chatbot.get('github', '')}."
            ),
            section="chatbot",
        )

    instructions = knowledge.get("instructionsForAssistant") or {}
    for key in ("persona", "style", "refusals", "contacts"):
        if instructions.get(key):
            _add(
                chunks,
                f"policy-{key}",
                f"Answering policy ({key}): {instructions[key]}",
                section="policy",
            )

    meta_notes = (knowledge.get("meta") or {}).get("notes") or []
    for i, note in enumerate(meta_notes):
        _add(
            chunks,
            f"meta-note-{i}",
            f"Knowledge rule: {note}",
            section="policy",
        )

    for i, rule in enumerate(knowledge.get("exclusions") or []):
        _add(
            chunks,
            f"exclusion-{i}",
            f"Answering policy: {rule}",
            section="policy",
        )

    return chunks


def _build_legacy(knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Previous flat schema used before knowledge.json v1."""
    chunks: List[Dict[str, Any]] = []
    name = knowledge.get("name", "Sai Vishnu Vamsi Senagasetty")
    titles = _join(knowledge.get("titles", []))
    location = knowledge.get("location", "")

    _add(
        chunks,
        "summary",
        (
            f"Profile summary for {name}. Titles: {titles}. Location: {location}. "
            f"{knowledge.get('summary', '')}"
        ),
        section="summary",
    )

    for i, edu in enumerate(knowledge.get("education", [])):
        _add(
            chunks,
            f"education-{i}",
            (
                f"Education: {edu.get('degree')} at {edu.get('school')} "
                f"({edu.get('period', '')})."
            ),
            section="education",
            school=edu.get("school"),
        )

    skills = knowledge.get("skills", [])
    if skills:
        _add(
            chunks,
            "skills",
            f"Technical skills for {name}: {_join(skills)}.",
            section="skills",
        )

    langs = knowledge.get("languages", [])
    if langs:
        _add(
            chunks,
            "languages",
            f"Spoken languages: {_join(langs)}.",
            section="languages",
        )

    projects = knowledge.get("projects", [])
    experiences = knowledge.get("experience", [])

    if experiences:
        roles = "; ".join(
            f"{e.get('title')} at {e.get('company')} ({e.get('period', '')})"
            for e in experiences
        )
        _add(
            chunks,
            "experience-index",
            f"Complete work history for {name}: {roles}.",
            section="experience-index",
        )

    for exp in experiences:
        eid = str(exp.get("id", exp.get("company", "role"))).replace(" ", "-").lower()
        _add(
            chunks,
            f"experience-{eid}",
            (
                f"Experience: {exp.get('title')} at {exp.get('company')} "
                f"({exp.get('period', '')}). {exp.get('summary', '')}"
            ),
            section="experience",
            company=exp.get("company"),
            id=exp.get("id"),
        )

    if projects:
        listing = "; ".join(
            f"{p.get('title')} ({p.get('subtitle', '')})" for p in projects
        )
        _add(
            chunks,
            "projects-index",
            (
                f"Complete list of {name}'s projects ({len(projects)} total): {listing}. "
                "Ask about any specific project for full details."
            ),
            section="projects-index",
        )

    for proj in projects:
        tech = _join(proj.get("tech", []))
        _add(
            chunks,
            f"project-{proj.get('id')}",
            (
                f"Project: {proj.get('title')} — {proj.get('subtitle', '')}. "
                f"{proj.get('summary', '')} Tech: {tech}."
            ),
            section="project",
            id=proj.get("id"),
            title=proj.get("title"),
        )

    for i, rule in enumerate(knowledge.get("exclusions", [])):
        _add(
            chunks,
            f"exclusion-{i}",
            f"Answering policy: {rule}",
            section="policy",
        )

    return chunks
