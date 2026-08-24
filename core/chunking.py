"""
Per-document-type chunking.

Each document type gets its own splitting policy, because they are
structurally different:

    FCP         procedures are self-contained sequences. Splitting one in
                half produces a chunk that starts mid-sequence, so these
                get a large budget and split only at step boundaries.
    MOR         long prose with numbered headings. Small chunks, paragraph
                boundaries, generous overlap.
    OPS_MANUAL  heading-delimited prose.
    GUIDELINES  heading-delimited prose.
    toc_index   one FCP index row. Atomic.
    image_cap.  one caption. Atomic.
    command_log one telecommand record. Atomic.

Chunks are never mixed across document types: a chunk carries the
doc_type of the file it came from, and retrieval can filter on it.
"""

import re
from typing import Callable


# Types that must never be split, whatever their length.
ATOMIC_TYPES = {"toc_index", "image_caption", "command_log"}


def _clean_caption(text: str) -> str:
    """Strip caption-prefix artifacts the vision model sometimes emits."""
    for pattern in (
        r'^Image\s+Caption\s*:?\s*',
        r'^Caption\s*:?\s*',
        r'^Figure\s+\d+[.:]\s*',
        r'^Fig\.\s*\d+[.:]\s*',
    ):
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()


def split_on_boundaries(text: str, max_chars: int, overlap: int,
                        boundaries: list) -> list:
    """
    Generic overlapping splitter. Walks back from each cut point looking
    for the first boundary in the list, so chunks end at a sensible place
    instead of mid-word.
    """
    if len(text) <= max_chars:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end == len(text):
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        cut = end
        for sep in boundaries:
            pos = text.rfind(sep, start + overlap, end)
            if pos != -1 and pos > start:
                cut = pos + len(sep)
                break

        piece = text[start:cut].strip()
        if piece:
            chunks.append(piece)
        start = max(start + 1, cut - overlap)

    return [c for c in chunks if c.strip()]


# Splitting policies. Each takes (text, max_chars, overlap) -> list[str].

def split_fcp(text: str, max_chars: int, overlap: int) -> list:
    """
    Procedure-aware. Prefers to cut at an end-of-procedure marker or a
    notes block so a chunk never starts mid-step.
    """
    return split_on_boundaries(
        text, max_chars, overlap,
        ["END OF PROCEDURE\n", "\nNOTES:", "\n\n", "\n", ". ", " "],
    )


def split_prose(text: str, max_chars: int, overlap: int) -> list:
    """Paragraph-first splitting for MOR, ops manual and guidelines."""
    return split_on_boundaries(
        text, max_chars, overlap,
        ["\n\n", ".\n", ". ", "\n", " "],
    )


def split_atomic(text: str, max_chars: int, overlap: int) -> list:
    return [text]


STRATEGY_BY_DOC_TYPE = {
    "FCP":        split_fcp,
    "MOR":        split_prose,
    "OPS_MANUAL": split_prose,
    "GUIDELINES": split_prose,
    "TCH":        split_atomic,
}


def strategy_for(doc_type: str, content_type: str = "") -> Callable:
    """Content type wins: a caption inside the MOR is still atomic."""
    if content_type in ATOMIC_TYPES:
        return split_atomic
    return STRATEGY_BY_DOC_TYPE.get(doc_type, split_prose)


def deduplicate(items: list, id_field: str) -> tuple:
    """
    Drop items whose id has already been seen. ChromaDB rejects a batch
    containing duplicate ids, so this has to happen before ingestion.
    Returns (unique_items, n_removed).
    """
    seen, out, dropped = set(), [], 0
    for item in items:
        key = str(item.get(id_field, ""))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(item)
    return out, dropped


def apply(items: list, satellite, id_field: str = "chunk_id") -> tuple:
    """
    Run the right splitting policy over every item.

    Oversized items become several sub-chunks that keep a link back to the
    parent, so retrieval can reconstruct the full section when it needs to.

    Returns (chunks, stats).
    """
    out      = []
    n_split  = 0
    per_type = {}

    for item in items:
        doc_type     = item.get("source_doc", "")
        content_type = item.get("content_type", "")
        text         = item.get("text", "")

        if content_type == "image_caption":
            text = _clean_caption(text)

        params    = satellite.chunking_for(doc_type)
        max_chars = params.get("max_chars", 1200)
        overlap   = params.get("overlap", 200)
        splitter  = strategy_for(doc_type, content_type)

        parts = splitter(text, max_chars, overlap)

        if len(parts) == 1:
            row = dict(item)
            row["text"]            = parts[0]
            row["is_sub_chunk"]    = False
            row["parent_chunk_id"] = ""
            row["sub_index"]       = 0
            out.append(row)
        else:
            n_split += 1
            base_id  = str(item.get(id_field, ""))
            for i, part in enumerate(parts):
                row = dict(item)
                row["text"]            = part
                row[id_field]          = f"{base_id}_s{i}"
                row["is_sub_chunk"]    = True
                row["parent_chunk_id"] = base_id
                row["sub_index"]       = i
                out.append(row)

        per_type[doc_type] = per_type.get(doc_type, 0) + len(parts)

    sizes = [len(c.get("text", "")) for c in out] or [0]
    stats = {
        "in":        len(items),
        "out":       len(out),
        "split":     n_split,
        "avg_chars": sum(sizes) // len(sizes),
        "max_chars": max(sizes),
        "per_type":  per_type,
    }
    return out, stats
