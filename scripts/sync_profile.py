#!/usr/bin/env python3
"""Sync the RAG knowledge base from the portfolio frontend's profile data.

The site renders from app/data/profile.js (experience, education, skills,
projects); the assistant answers from data/knowledge.json. The two files
describe the same career by hand, and they drifted before (boomerroomers +
Handshake were missing from knowledge.json until Tier 1 fixed it by hand).
This script makes profile.js the single source of truth for the fields the
frontend owns, and merges them into knowledge.json:

  synced from profile.js: identity (name/titles/location), pitch, education,
    experience summaries, skills (all + categories + aliases),
    projects (title/subtitle/tags/techStack/goal/outcome/github, by id)
  preserved in knowledge.json: everything curated by hand — bio, hero,
    experience "detailed" entries (STAR diffStats), project approach /
    whyApproach / pipeline / components / howBuilt / evals, faq,
    publications, certifications, badges, chatbot blurb, policy notes.

New roles/projects that appear in profile.js get a minimal knowledge entry
plus a warning (their curated detail still needs a human). Nothing is ever
deleted silently: knowledge entries missing from profile.js are kept and
reported.

Usage (from the Chatbot-RAG repo root):
  python scripts/sync_profile.py [--frontend ../portfolio] [--check]
  python scripts/sync_profile.py --check    # exit 1 + diff summary if drifted

After a real sync, rebuild the index:
  python scripts/build_index.py
(or rely on the retriever's self-healing rebuild at startup.)

Requires: node on PATH (profile.js is an ES module; it is evaluated in a
sandbox, not executed as an app — no imports are resolved, the three data
files are concatenated after stripping their import/export lines).
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fields of a project item that profile.js owns (the mapped subset in
# profile.js). Every other project field stays as curated in knowledge.json.
PROJECT_SYNC_FIELDS = ["title", "subtitle", "tags", "techStack", "goal", "outcome", "github"]

NODE_LOADER = r"""
const fs = require("fs");
const vm = require("vm");
function loadDataFile(path) {
  let src = fs.readFileSync(path, "utf8");
  const lines = src.split("\n");
  const kept = [];
  for (const line of lines) {
    if (/^\s*import\s/.test(line)) continue;               // imports: provided by us
    if (/^\s*export\s+\{/.test(line)) continue;            // re-export lists
    kept.push(line.replace(/^(\s*)export\s+/, "$1"));      // export const/fn -> plain
  }
  return kept.join("\n");
}
// All three files are evaluated as ONE script, so the top-level consts
// (projects, allSkills, ...) are in scope for profile.js and for the tail
// expression. Nothing is executed as an app: no imports are resolved.
const dataDir = process.argv[2];
const src = ["skills.js", "projects.js", "profile.js"]
  .map((name) => loadDataFile(dataDir + "/" + name))
  .join("\n");
const tail = "\n;JSON.stringify({ profile, projects, skillCategories, allSkills,"
  + " exposure: [...EXPOSURE_SKILLS], core: [...CORE_SKILLS],"
  + " aliases: getSkillTaxonomy().aliases });";
process.stdout.write(vm.runInNewContext(src + tail, { console }));
"""


def load_frontend_data(frontend_dir: Path) -> dict:
    """Evaluate the frontend's three data files in a Node vm; return JSON."""
    data_dir = frontend_dir / "app" / "data"
    missing = [n for n in ("profile.js", "projects.js", "skills.js") if not (data_dir / n).exists()]
    if missing:
        raise SystemExit(
            f"frontend data files not found under {data_dir}: {', '.join(missing)}. "
            "Pass --frontend <path-to-portfolio-updated-clone>."
        )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(NODE_LOADER)
        loader = f.name
    try:
        proc = subprocess.run(
            ["node", loader, str(data_dir)],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        raise SystemExit("node is not on PATH — sync_profile.py needs node to read profile.js.")
    finally:
        Path(loader).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise SystemExit(f"node loader failed:\n{proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"node loader returned non-JSON output:\n{proc.stdout[:500]}")


def _norm_list(items):
    return [i for i in (items or []) if i is not None]


def sync(knowledge: dict, front: dict) -> tuple[dict, list[str], list[str]]:
    """Merge frontend-owned fields into a copy of knowledge.json.

    Returns (new_knowledge, changes, warnings). changes are human-readable
    one-liners; warnings are things a human still has to curate.
    """
    new = copy.deepcopy(knowledge)
    profile = front["profile"]
    changes: list[str] = []
    warnings: list[str] = []

    def _changed(label: str, old, new_val) -> None:
        if old != new_val:
            changes.append(f"{label}: {json.dumps(old, ensure_ascii=False)[:120]} -> "
                           f"{json.dumps(new_val, ensure_ascii=False)[:120]}")

    # -- identity (name/titles/location only; contact/links stay curated) --
    ident = new.setdefault("identity", {})
    for key in ("name", "titles", "location"):
        _changed(f"identity.{key}", ident.get(key), profile.get(key))
        ident[key] = profile.get(key)

    # -- pitch / education (wholesale; profile.js is the source) --
    _changed("pitch", new.get("pitch"), profile.get("pitch"))
    new["pitch"] = profile.get("pitch")
    _changed("education", new.get("education"), profile.get("education"))
    new["education"] = copy.deepcopy(profile.get("education") or [])

    # -- experience summaries (by id; "detailed" entries stay curated) --
    exp = new.setdefault("experience", {})
    old_summaries = {s.get("id"): s for s in exp.get("summaries") or []}
    new_summaries = []
    for s in profile.get("experience") or []:
        entry = {k: copy.deepcopy(s.get(k)) for k in ("id", "title", "company", "period", "skills", "summary")}
        new_summaries.append(entry)
        if s.get("id") not in old_summaries:
            changes.append(f"experience.summaries: + {s.get('id')} ({s.get('title')} @ {s.get('company')})")
        elif old_summaries[s.get("id")] != entry:
            changes.append(f"experience.summaries: ~ {s.get('id')} (title/company/period/skills/summary changed)")
    for sid in old_summaries:
        if sid not in {s.get("id") for s in new_summaries}:
            warnings.append(f"experience '{sid}' is in knowledge.json but not profile.js — kept as-is")
    exp["summaries"] = new_summaries
    detailed_ids = {d.get("id") for d in exp.get("detailed") or []}
    for s in new_summaries:
        if s.get("id") not in detailed_ids:
            warnings.append(
                f"experience '{s.get('id')}' has no 'detailed' entry in knowledge.json "
                "(STAR diffStats/location/type) — chunk will use the summary only"
            )

    # -- skills --
    skills = new.setdefault("skills", {})
    _changed("skills.all", skills.get("all"), front.get("allSkills"))
    skills["all"] = copy.deepcopy(front.get("allSkills") or [])
    # Categories: only the fields the chunker reads (id/name/skills). The
    # frontend's UI-only fields (e.g. icon) are not knowledge-base data.
    front_cats = [
        {k: copy.deepcopy(c.get(k)) for k in ("id", "name", "skills")}
        for c in (front.get("skillCategories") or [])
    ]
    _changed("skills.categories", skills.get("categories"), front_cats)
    skills["categories"] = front_cats
    _changed("skills.aliases", skills.get("aliases"), front.get("aliases"))
    skills["aliases"] = copy.deepcopy(front.get("aliases") or {})

    # -- projects (by id; only the profile-owned fields) --
    proj_block = new.setdefault("projects", {})
    items = proj_block.get("items") or []
    by_id = {p.get("id"): p for p in items}
    front_projects = {p.get("id"): p for p in (profile.get("projects") or [])}
    new_items = []
    for p in profile.get("projects") or []:
        pid = p.get("id")
        existing = by_id.get(pid)
        if existing is None:
            item = {k: copy.deepcopy(p.get(k)) for k in PROJECT_SYNC_FIELDS if p.get(k) is not None}
            item["id"] = pid
            new_items.append(item)
            changes.append(f"projects.items: + {pid} (new — approach/whyApproach/pipeline need curation)")
            warnings.append(f"project '{pid}' is new from profile.js — add approach/whyApproach/pipeline/components to knowledge.json")
            continue
        updated = dict(existing)
        for k in PROJECT_SYNC_FIELDS:
            if p.get(k) is not None and existing.get(k) != p.get(k):
                changes.append(f"projects.{pid}.{k}: updated from profile.js")
            if p.get(k) is not None:
                updated[k] = copy.deepcopy(p.get(k))
        new_items.append(updated)
    for pid in by_id:
        if pid not in front_projects:
            warnings.append(f"project '{pid}' is in knowledge.json but not profile.js — kept as-is")
            new_items.append(by_id[pid])
    proj_block["items"] = new_items

    # -- provenance stamp --
    meta = new.setdefault("meta", {})
    meta["syncedFrom"] = "app/data/profile.js (+ projects.js, skills.js)"
    meta["syncedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return new, changes, warnings


def _chunk_summary(knowledge: dict) -> tuple[int, set[str]]:
    from rag.chunking import build_chunks

    chunks = build_chunks(knowledge)
    return len(chunks), {c["id"] for c in chunks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frontend", default=None,
                    help="path to the portfolio-updated clone (default: ../portfolio next to this repo)")
    ap.add_argument("--knowledge", default=None,
                    help="path to knowledge.json (default: data/knowledge.json in this repo)")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 with a diff summary if the file would change")
    ap.add_argument("--out", default=None, help="write the synced file here instead of --knowledge")
    args = ap.parse_args()

    frontend = Path(args.frontend) if args.frontend else ROOT.parent / "portfolio"
    knowledge_path = Path(args.knowledge) if args.knowledge else ROOT / "data" / "knowledge.json"
    out_path = Path(args.out) if args.out else knowledge_path

    front = load_frontend_data(frontend)
    with open(knowledge_path, encoding="utf-8") as f:
        knowledge = json.load(f)

    new, changes, warnings = sync(knowledge, front)

    old_n, old_ids = _chunk_summary(knowledge)
    new_n, new_ids = _chunk_summary(new)

    print(f"Frontend: {frontend} (profile: {len(front['profile'].get('experience') or [])} roles, "
          f"{len(front['profile'].get('projects') or [])} projects, "
          f"{len(front.get('allSkills') or [])} skills)")
    print(f"Knowledge: {knowledge_path}")
    print(f"Chunks: {old_n} -> {new_n} "
          f"(+{sorted(new_ids - old_ids)[:5] if new_ids - old_ids else []}"
          f"{' -' + str(sorted(old_ids - new_ids)[:5]) if old_ids - new_ids else ''})")
    if changes:
        print(f"\nChanges ({len(changes)}):")
        for c in changes:
            print(f"  - {c}")
    else:
        print("\nNo changes: knowledge.json already matches profile.js.")
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")
    if not changes and not warnings:
        return 0

    if args.check:
        if changes:
            print("\n--check: knowledge.json is out of sync (would change). Run without --check to sync.")
            return 1
        return 0

    if out_path == knowledge_path and not changes:
        print("\nNothing to write (warnings only).")
        return 0
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nWrote {out_path}")
    print("Next: python scripts/build_index.py  (or rely on the startup self-heal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
