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


# Block-aware splitting, for sections read by core/layout.py.
#
# The splitters above see a section as a string and look for a good place
# to cut near a character count. These see it as the document wrote it:
# paragraphs and table rows. A cut falls between two of those. A table
# that needs more than one chunk has its header row repeated on each, so
# no chunk is a block of values with nothing to say what they are.

SENTENCE_END = re.compile(r'(?<=[.!?;])\s+')

# A short paragraph ending in a colon introduces whatever follows it. If
# it would be the last thing in a chunk it goes to the next one instead.
LEAD_IN_MAX = 200


def _fit(text: str, max_chars: int, seps: list) -> list:
    """
    Cut text into pieces of at most max_chars, at the first separator in
    seps that allows it. Only used on a single sentence or table row that
    is longer than a whole chunk.
    """
    if len(text) <= max_chars:
        return [text]
    for sep in seps:
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) < 2:
            continue
        out, cur = [], ""
        for part in parts:
            if cur and len(cur) + len(sep) + len(part) > max_chars:
                out.append(cur)
                cur = part
            else:
                cur = f"{cur}{sep}{part}" if cur else part
        out.append(cur)
        # A piece can still be too long; try the next, finer separator.
        return [piece for o in out for piece in _fit(o, max_chars, seps)]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _units(blocks: list, max_chars: int) -> list:
    """
    The pieces a cut may fall between, in order: whole paragraphs and
    table rows. A paragraph longer than a chunk is the one thing that has
    to be cut inside, and it is cut between its sentences.

    glue is what joins a unit to the one before it when both land in the
    same chunk: a newline, or a space between sentences of one paragraph.
    """
    units = []
    for n, block in enumerate(blocks):
        if block["kind"] == "table":
            for row in block["rows"]:
                for piece in _fit(row, max_chars, [" | ", " "]):
                    units.append({"text": piece, "glue": "\n",
                                  "table": n, "para": None})
            continue

        text = block["text"].strip()
        if not text:
            continue
        if len(text) <= max_chars:
            units.append({"text": text, "glue": "\n",
                          "table": None, "para": None})
            continue
        for line in text.split("\n"):
            glue = "\n"
            for sentence in SENTENCE_END.split(line):
                for piece in _fit(sentence.strip(), max_chars, [" "]):
                    if piece:
                        units.append({"text": piece, "glue": glue,
                                      "table": None, "para": n})
                        glue = " "
    return units


def _join(entries: list) -> str:
    out = ""
    for glue, text in entries:
        out = out + glue + text if out else text
    return out


def split_blocks(blocks: list, max_chars: int, overlap: int) -> list:
    """Pack paragraphs and table rows into chunks of about max_chars."""
    pieces  = []
    current = []            # (unit, the (glue, text) entries it added)
    size    = 0
    started = set()         # tables already begun in an earlier chunk
    shown   = set()         # tables whose header is in the current chunk

    def entries_for(unit) -> list:
        """The unit, behind its table's title and header if this chunk
        has not shown them yet."""
        n = unit["table"]
        if n is None or n in shown:
            return [(unit["glue"], unit["text"])]
        block = blocks[n]
        title = block.get("title", "")
        if n in started and title:
            title += " (continued)"
        head = [("\n", x) for x in (title, block["header"]) if x]
        return head + [("\n", unit["text"])]

    def add(unit) -> None:
        nonlocal size
        entries = entries_for(unit)
        if unit["table"] is not None:
            shown.add(unit["table"])
            started.add(unit["table"])
        current.append((unit, entries))
        size += sum(len(text) + 1 for _, text in entries)

    def close() -> tuple:
        """
        End the current chunk. Returns (move, repeat): a lead-in line that
        was held back for the next chunk, or a sentence to say again there.
        """
        nonlocal current, size
        move = repeat = None
        last = current[-1][0]
        if (len(current) > 1 and last["table"] is None
                and len(last["text"]) <= LEAD_IN_MAX
                and last["text"].rstrip().endswith(":")):
            current.pop()
            move = last
        elif last["para"] is not None and len(last["text"]) <= overlap:
            repeat = last
        pieces.append(_join([e for _, entries in current for e in entries]))
        current, size = [], 0
        shown.clear()
        return move, repeat

    for unit in _units(blocks, max_chars):
        need = sum(len(text) + 1 for _, text in entries_for(unit))
        if current and size + need > max_chars:
            move, repeat = close()
            if move:
                add(move)
            elif repeat and repeat["para"] == unit["para"]:
                # The cut fell inside one long paragraph, so its last
                # sentence opens the next chunk as well. Between whole
                # paragraphs nothing is repeated.
                add(repeat)
        add(unit)
    if current:
        pieces.append(_join([e for _, entries in current for e in entries]))

    return [p for p in pieces if p.strip()]


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

        # A section that came with its blocks is cut between them. The
        # blocks have done their job once the section is split, and are
        # not copied into every chunk.
        blocks = item.get("blocks")
        if blocks and splitter is not split_atomic:
            parts = split_blocks(blocks, max_chars, overlap) or [text]
        else:
            parts = splitter(text, max_chars, overlap)
        if blocks:
            item = {k: v for k, v in item.items() if k != "blocks"}

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
