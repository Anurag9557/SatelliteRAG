"""
Build slides/Aditya_L1_Mission_Overview.pptx from the actual ingested
Aditya-L1 documents: satellites/aditya-l1/extracted/chunks.json and
satellites/aditya-l1/documents/SOURCES.json. Content is pulled from real
extracted text, not hardcoded, so re-running this after re-extraction
keeps the deck accurate.

Usage: python slides/build_adityal1_overview_pptx.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_utils import add_bullet_slide, add_text_slide, new_deck

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "satellites" / "aditya-l1" / "extracted" / "chunks.json"
SOURCES_PATH = REPO_ROOT / "satellites" / "aditya-l1" / "documents" / "SOURCES.json"
OUT_PATH = Path(__file__).resolve().parent / "Aditya_L1_Mission_Overview.pptx"

DEMO_QUERY = "What payloads does Aditya-L1 carry, and what does each one measure?"


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def clean_lines(text: str) -> list:
    """Split extracted PDF text into non-trivial lines, stripping bullet glyphs."""
    lines = []
    for raw in text.split("\n"):
        line = raw.strip().lstrip("•-*").strip()
        if len(line) < 3:
            continue
        lines.append(line)
    return lines


def find_chunk(chunks: list, must_contain: list, doc_filename: str = None,
               exclude_ids: set = frozenset()) -> dict:
    """First chunk whose text contains all of `must_contain` (case-insensitive)."""
    for c in chunks:
        if c["chunk_id"] in exclude_ids:
            continue
        if doc_filename and c.get("doc_filename") != doc_filename:
            continue
        text_lower = c["text"].lower()
        if all(kw.lower() in text_lower for kw in must_contain):
            return c
    return None


def run_demo_query() -> str:
    args = [sys.executable, "cli.py", "query", "aditya-l1", DEMO_QUERY, "-k", "3"]
    if not have_ollama():
        args.append("--no-answer")
    try:
        result = subprocess.run(
            args, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=400,
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return f"(query demo unavailable: {exc})"


def have_ollama() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def main():
    chunks = load_json(CHUNKS_PATH, [])
    sources = load_json(SOURCES_PATH, [])
    if not chunks:
        sys.exit(f"No chunks at {CHUNKS_PATH}. Run `python cli.py build aditya-l1` first.")

    prs = new_deck(
        "Aditya-L1 Mission Overview",
        "Built from public ISRO source documents — SatelliteRAG",
    )

    # Documents collected
    doc_bullets = []
    for s in sources:
        doc_bullets.append(f"{s.get('title', s['filename'])}  ({s['filename']})")
    add_bullet_slide(prs, "Documents Collected (public ISRO sources)", doc_bullets or ["No SOURCES.json found"])

    used_ids = set()

    # Mission overview, pulled from the real MOR chunk about science objectives.
    overview_chunk = find_chunk(chunks, ["Major Science"])
    if overview_chunk:
        used_ids.add(overview_chunk["chunk_id"])
        lines = clean_lines(overview_chunk["text"])
        # Drop repeated header/footer lines (page title + page number) at the top.
        lines = [l for l in lines if not re.fullmatch(r"\d+", l)]
        add_bullet_slide(prs, "Mission Overview", lines[:14])

    # Payloads: the OPS_MANUAL science-payloads chunk names all seven
    # instruments, distinct from the mission-overview chunk above.
    instr_chunk = find_chunk(chunks, ["VELC", "SUIT", "ASPEX"], exclude_ids=used_ids)
    if instr_chunk:
        used_ids.add(instr_chunk["chunk_id"])
        lines = clean_lines(instr_chunk["text"])
        add_bullet_slide(prs, "Payloads", lines[:14])

    # Mission status, pulled from the real AO chunk that names the launch
    # date and ISSDC's role in operations and data archival.
    status_chunk = find_chunk(chunks, ["successfully launched", "ISSDC"], exclude_ids=used_ids)
    if status_chunk:
        used_ids.add(status_chunk["chunk_id"])
        lines = clean_lines(status_chunk["text"])
        add_bullet_slide(prs, "Mission Status (ISRO / ISSDC)", lines[:14])

    # Real retrieval demo transcript.
    transcript = run_demo_query()
    add_text_slide(prs, f'Retrieval Demo: "{DEMO_QUERY}"', transcript[:1800] or "(no output captured)", font_size=11)

    # References, from the manifest the fetch script wrote.
    ref_lines = [f"{s.get('title', '')} — {s['source_url']}  (retrieved {s.get('retrieved_date','')})"
                 for s in sources]
    add_bullet_slide(prs, "References", ref_lines or ["No sources recorded"], font_size=13)

    prs.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
