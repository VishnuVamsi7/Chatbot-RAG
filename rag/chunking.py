"""Structure-aware chunking from curated knowledge.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_knowledge(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_chunks(knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    name = knowledge.get("name", "Sai Vishnu Vamsi Senagasetty")
    titles = ", ".join(knowledge.get("titles", []))
    location = knowledge.get("location", "")

    chunks.append(
        {
            "id": "summary",
            "text": (
                f"Profile summary for {name}. Titles: {titles}. Location: {location}. "
                f"{knowledge.get('summary', '')}"
            ),
            "metadata": {"section": "summary"},
        }
    )

    for i, edu in enumerate(knowledge.get("education", [])):
        chunks.append(
            {
                "id": f"education-{i}",
                "text": (
                    f"Education: {edu.get('degree')} at {edu.get('school')} "
                    f"({edu.get('period', '')})."
                ),
                "metadata": {"section": "education", "school": edu.get("school")},
            }
        )

    skills = knowledge.get("skills", [])
    if skills:
        chunks.append(
            {
                "id": "skills",
                "text": f"Technical skills for {name}: {', '.join(skills)}.",
                "metadata": {"section": "skills"},
            }
        )

    langs = knowledge.get("languages", [])
    if langs:
        chunks.append(
            {
                "id": "languages",
                "text": f"Spoken languages: {', '.join(langs)}.",
                "metadata": {"section": "languages"},
            }
        )

    for exp in knowledge.get("experience", []):
        eid = exp.get("id", exp.get("company", "role")).replace(" ", "-").lower()
        chunks.append(
            {
                "id": f"experience-{eid}",
                "text": (
                    f"Experience: {exp.get('title')} at {exp.get('company')} "
                    f"({exp.get('period', '')}). {exp.get('summary', '')}"
                ),
                "metadata": {
                    "section": "experience",
                    "company": exp.get("company"),
                    "id": exp.get("id"),
                },
            }
        )

    for proj in knowledge.get("projects", []):
        tech = ", ".join(proj.get("tech", []))
        chunks.append(
            {
                "id": f"project-{proj.get('id')}",
                "text": (
                    f"Project: {proj.get('title')} — {proj.get('subtitle', '')}. "
                    f"{proj.get('summary', '')} Tech: {tech}."
                ),
                "metadata": {
                    "section": "project",
                    "id": proj.get("id"),
                    "title": proj.get("title"),
                },
            }
        )

    for i, rule in enumerate(knowledge.get("exclusions", [])):
        chunks.append(
            {
                "id": f"exclusion-{i}",
                "text": f"Answering policy: {rule}",
                "metadata": {"section": "policy"},
            }
        )

    return chunks
