"""
Retrieval over one satellite or several.

Per satellite the search is hybrid: a dense semantic pass over ChromaDB
and a BM25 keyword pass, fused with Reciprocal Rank Fusion. Filters for
document type, subsystem and date are applied to both passes so BM25
cannot smuggle in results the semantic filter excluded.

Cross-satellite search runs the same thing per satellite and fuses the
ranked lists again. RRF operates on ranks rather than raw scores, so
satellites embedded with different models still merge correctly.
"""

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import embeddings, rerank as reranker
from .ingest import COLLECTION_COMMANDS, COLLECTION_KNOWLEDGE, tokenize

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise SystemExit("Missing dependency: pip install chromadb")


@dataclass
class Hit:
    chunk_id:  str
    text:      str
    score:     float
    satellite: str
    metadata:  dict = field(default_factory=dict)

    @property
    def doc_type(self) -> str:
        return self.metadata.get("doc_type", "")

    @property
    def content_type(self) -> str:
        return self.metadata.get("content_type", "")

    def cite(self, show_satellite: bool = False) -> str:
        m     = self.metadata
        parts = []
        if show_satellite:
            parts.append(m.get("satellite_name") or self.satellite)

        # Command records carry their own fields; a page number means
        # nothing here.
        if self.content_type == "command_log":
            parts.append("TCH")
            if m.get("cid"):
                parts.append(m["cid"])
            if m.get("mnemonic"):
                parts.append(m["mnemonic"])
            if m.get("date"):
                stamp = m["date"]
                if m.get("time"):
                    stamp += f" {m['time']}"
                parts.append(stamp)
            if m.get("status"):
                parts.append(m["status"])
            return " | ".join(parts)

        parts.append(self.doc_type or "?")
        if m.get("fcp_number"):
            parts.append(f"FCP-{m['fcp_number']}")
        if m.get("subsystem") and m["subsystem"] not in ("", "UNKNOWN"):
            parts.append(m["subsystem"])
        if m.get("page"):
            parts.append(f"p.{m['page']}")
        heading = m.get("section_heading", "")
        if heading:
            clipped = heading[:55] + ("..." if len(heading) > 55 else "")
            parts.append(f'"{clipped}"')
        return " | ".join(parts)


# Query understanding

DATE_RE = re.compile(
    r'\b(20\d{2})[-/\s](0[1-9]|1[0-2])[-/\s](0[1-9]|[12]\d|3[01])\b')

# Order matters. The first pattern to match wins, so specific component
# names come first and generic words last. "power" appears in almost
# every switch-on question ("gyro power on", "wheel power on", "HGA motor
# power on"), so a POWER entry matching a bare \bpower\b near the top
# swallows every one of them and routes the query to the wrong subsystem.
SUBSYSTEM_PATTERNS = [
    ("DTG",        r'\bDTG\b|\bgyro\b|\bCSA\b'),
    ("SENSOR",     r'\bstar\s*sensor\b|\bSS-?[12]\b'),
    ("WHEEL",      r'\bwheel\b|\bWDE\b|\bsuspension\b|\bmomentum\s*dump'),
    ("MECHANISM",  r'\bHGA\b|\bmechanism\b|\bSPDM\b|\bMCE\b|\bMEPU\b|\bdeploy'),
    ("PROPULSION", r'\bthruster\b|\bpropulsion\b|\blatch\s*valve\b'
                   r'|\boxidiser\b|\bpyro\b'),
    ("TTC_XBAND",  r'\bTTC\b|\bX.band\b|\btransponder\b|\bTWTA\b|\bXBS\b'
                   r'|\btone\s*ranging\b'),
    ("ODHS",       r'\bODHS\b|\bSSR\b|\bDFU\b|\bBDH\b|\bdata\s*handling\b'),
    ("PAYLOAD",    r'\bPAPA\b|\bASPEX\b|\bVELC\b|\bHELIOS\b|\bSOLEXS?\b'
                   r'|\bSUIT\b|\bmagnetometer\b|\bpayload\b'),
    ("OBC",        r'\bOBC\b|\bCPU\b|\bEEPROM\b|\bwatchdog\b'
                   r'|\btelecommand\s*link\b'),
    ("AOCS",       r'\bAOCS\b|\bsun\s*acquisition\b|\battitude\b'
                   r'|\brate\s*damp'),
    # Generic last. A bare "power" is too weak a signal to route on.
    ("POWER",      r'\bFPGA\b|\bTCR\b|\bDC.DC\b|\bbattery\b'
                   r'|\bpower\s*(system|bus|supply)\b'),
]

DIAGRAM_RE = re.compile(
    r'\b(diagram|figure|chart|plot|image|picture|schematic|flow\s*chart|'
    r'block\s*diagram|show\s+me|visual)\b', re.IGNORECASE)

# A question that asks how to do something wants the procedure itself.
# Without this, a manual section that merely mentions the subsystem
# competes on equal terms with the procedure that answers the question.
HOWTO_RE = re.compile(
    r'\b(fcp|procedure|how\s+to|how\s+do|how\s+is|steps?|switch\w*|'
    r'power\w*|turn\w*|enabl\w*|disabl\w*|reset|select\w*|activat\w*|'
    r'deactivat\w*|operat\w*|execut\w*|configur\w*|uplink\w*|command\w*)\b',
    re.IGNORECASE)


def wants_commands(query: str) -> bool:
    """
    Route to the telecommand log only on a real command-log signal.

    A bare "command" is far too weak: "programmable command sequencer",
    "configurable command block" and "telecommand link test" are all
    procedure questions, and sending them to the command store loses them
    entirely. The phrasing has to be about commands that were uplinked,
    a CID, or a date.
    """
    if DATE_RE.search(query):
        return True
    return bool(re.search(
        r'\bcommands?\s+(were\s+|that\s+were\s+)?(uplink|sent|issued|executed)'
        r'|\buplinked\s+(on|from|to)\b'
        r'|\btelecommand\s+history\b|\bcommand\s+history\b'
        r'|\bCID\s+[A-Z]{2,}\d+|\borbit\s+\d{3,}\b',
        query, re.IGNORECASE))


def find_date(query: str) -> Optional[str]:
    m = DATE_RE.search(query)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def find_subsystem(query: str) -> Optional[str]:
    for sub, pattern in SUBSYSTEM_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return sub
    return None


def wants_diagram(query: str) -> bool:
    return bool(DIAGRAM_RE.search(query))


def wants_procedure(query: str) -> bool:
    return bool(HOWTO_RE.search(query))


# Store access

_clients = {}


def _collections(satellite):
    if satellite.slug not in _clients:
        if not satellite.is_ingested():
            raise FileNotFoundError(
                f"{satellite.slug} has no vector store. "
                f"Run: python cli.py ingest {satellite.slug}"
            )
        client = chromadb.PersistentClient(
            path     = str(satellite.vectorstore_dir),
            settings = Settings(anonymized_telemetry=False),
        )
        _clients[satellite.slug] = (
            client.get_collection(COLLECTION_KNOWLEDGE),
            client.get_collection(COLLECTION_COMMANDS),
        )
    return _clients[satellite.slug]


def _where(doc_types=None, subsystem=None, date=None) -> Optional[dict]:
    """Build a ChromaDB filter. Several conditions need an explicit $and."""
    clauses = []
    if doc_types:
        clauses.append({"doc_type": {"$in": list(doc_types)}})
    if subsystem:
        clauses.append({"subsystem": {"$eq": subsystem}})
    if date:
        clauses.append({"date": {"$eq": date}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def dense_search(query, collection, satellite, k, where=None) -> list:
    embedder = embeddings.load(satellite.embedding_model)
    kwargs   = {
        "query_embeddings": [embedder.encode_query(query)],
        "n_results":        k,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    res = collection.query(**kwargs)
    if not res["ids"] or not res["ids"][0]:
        return []

    return [
        Hit(chunk_id=cid, text=doc, score=1 - dist,
            satellite=satellite.slug, metadata=meta)
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0])
    ]


def bm25_search(query, pkl_path: Path, satellite, k,
                doc_types=None, subsystem=None, date=None) -> list:
    if not pkl_path.exists():
        return []
    try:
        data = pickle.loads(pkl_path.read_bytes())
    except Exception:
        return []

    scores = data["bm25"].get_scores(tokenize(query))
    order  = sorted(range(len(scores)), key=lambda i: -scores[i])

    hits = []
    for idx in order:
        if scores[idx] <= 0 or len(hits) >= k:
            break
        item = data["items"][idx]

        # Apply the same filters the dense pass used. UNKNOWN subsystem is
        # kept because it means detection failed, not that it is wrong.
        # Apply exactly the filters the dense pass used. Letting UNKNOWN
        # through here would make the keyword pass more permissive than
        # the semantic one, so chunks the semantic filter excluded would
        # still reach the results through BM25.
        if doc_types and item.get("source_doc") not in doc_types:
            continue
        if subsystem and item.get("subsystem") != subsystem:
            continue
        if date and item.get("date") != date:
            continue

        hits.append(Hit(
            chunk_id  = data["ids"][idx],
            text      = item.get("text", ""),
            score     = float(scores[idx]),
            satellite = satellite.slug,
            metadata  = {**item, "satellite": satellite.slug,
                         "satellite_name": satellite.name,
                         "doc_type": item.get("source_doc", "")},
        ))
    return hits


def rrf(ranked_lists: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion. Rank-based, so mixed score scales are fine."""
    scores, store = {}, {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked):
            key = (hit.satellite, hit.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            store[key]  = hit
    ordered = sorted(scores, key=lambda key: -scores[key])
    out = []
    for key in ordered:
        hit = store[key]
        hit.score = scores[key]
        out.append(hit)
    return out


def prepend(primary: list, extra: list, n: int = 2) -> list:
    """
    Put the best n of extra at the front of primary, dropping duplicates.

    This is used instead of multiplying scores after fusion. RRF scores
    all land between 1/61 and 1/80, so a multiplier fights over
    hundredths and the intended item often fails to move at all. A
    targeted second search plus a prepend puts it where it belongs
    regardless of the fusion arithmetic.
    """
    seen, out = set(), []
    for h in list(extra[:n]) + list(primary):
        key = (h.satellite, h.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def boost_captions(hits: list, factor: float = 1.5) -> list:
    """Push image captions up when the question asks for a diagram."""
    for h in hits:
        if h.content_type == "image_caption":
            h.score *= factor
    return sorted(hits, key=lambda h: -h.score)


def boost_procedures(hits: list, factor: float = 1.6) -> list:
    """
    Lift the procedures when the question asks how to do something.
    Notes count too: a procedure keeps its notes page, and both answer
    the same question.
    """
    for h in hits:
        if h.content_type in ("procedure", "note"):
            h.score *= factor
    return sorted(hits, key=lambda h: -h.score)


def demote_index_rows(hits: list, factor: float = 0.75) -> list:
    """
    A table-of-contents row matches almost any procedure question,
    because it is little more than the title. It should sit below the
    procedure itself, but only just: an index row that both retrievers
    rank first is the strongest evidence available about which procedure
    the question is about, and a harsh penalty throws that away.
    """
    if all(h.content_type == "toc_index" for h in hits):
        return hits
    for h in hits:
        if h.content_type == "toc_index":
            h.score *= factor
    return sorted(hits, key=lambda h: -h.score)


def asks_which_procedure(query: str) -> bool:
    """
    'which FCP covers X' wants the index; 'how do I do X' wants the
    procedure. Only the first should see index rows ranked normally.
    """
    return bool(re.search(
        r'\b(which|what)\s+(fcp|procedure|number)\b|'
        r'\bfcp\s+(number|list|index)\b|\blist\s+(all\s+)?(fcp|procedures)\b',
        query, re.IGNORECASE))


def search_one(satellite, query: str, top_k=None, doc_types=None,
               subsystem=None, force_target=None, mode="hybrid") -> list:
    """
    Hybrid search within a single satellite.

    mode selects which passes run:
        hybrid  dense + BM25 fused by RRF, the shipping pipeline
        dense   semantic only, which is what isolates the embedding model
        bm25    keyword only, an embedder-independent floor

    Use dense when comparing embedders. Hybrid mixes in BM25, which does
    not depend on the embedder at all, so it masks the difference between
    one model and another.
    """
    cfg          = satellite.retrieval
    top_k        = top_k or cfg["top_k"]
    col_k, col_c = _collections(satellite)

    target = force_target or (
        "commands" if wants_commands(query) else "knowledge")

    if target == "commands":
        date = find_date(query)

        # "What commands on <date>" wants the whole day, not the five most
        # semantically similar records. With a date, read every matching
        # record straight from the store in chronological order.
        if date:
            got = col_c.get(where={"date": {"$eq": date}},
                            include=["documents", "metadatas"])
            metas = got.get("metadatas") or []
            ids   = got.get("ids") or []
            docs  = got.get("documents") or []
            if metas:
                order = sorted(range(len(metas)),
                               key=lambda i: metas[i].get("time", ""))
                return [Hit(
                    chunk_id  = ids[i] if i < len(ids) else "",
                    text      = docs[i] if i < len(docs) else "",
                    score     = 1.0,
                    satellite = satellite.slug,
                    metadata  = metas[i],
                ) for i in order]

        where = _where(date=date)
        dense = dense_search(query, col_c, satellite, cfg["n_dense"], where)
        keyw  = bm25_search(
            query, satellite.vectorstore_dir / "bm25_commands.pkl",
            satellite, cfg["n_bm25"], date=date)
        return rrf([dense, keyw], cfg["rrf_k"])[:top_k]

    # The subsystem is only used as a filter when the caller asks for one.
    #
    # Guessing it from the question and then filtering on the guess loses
    # answers outright. The documents file "auto thruster firing" under
    # AOCS, not PROPULSION; "reaction wheel logics" under AOCS, not WHEEL;
    # "gyro calibration" under AOCS, not DTG. Every one of those queries
    # guesses the wrong subsystem, and an exact-match filter then removes
    # the correct chunk before anything scores it, so reranking cannot
    # recover it. A wrong guess costs far more than no filter saves.
    sub   = subsystem
    where = _where(doc_types=doc_types, subsystem=sub)

    dense = [] if mode == "bm25" else dense_search(
        query, col_k, satellite, cfg["n_dense"], where)
    # A filter that excludes everything is worse than no filter at all.
    if not dense and where and mode != "bm25":
        dense = dense_search(query, col_k, satellite, cfg["n_dense"])
        sub   = None

    keyw = [] if mode == "dense" else bm25_search(
        query, satellite.vectorstore_dir / "bm25_knowledge.pkl",
        satellite, cfg["n_bm25"], doc_types=doc_types, subsystem=sub)

    merged = rrf([dense, keyw], cfg["rrf_k"])

    # Reranking reads each candidate against the question, so it orders
    # them far better than fusion arithmetic can. On the 100-question
    # bench it lifted Hit@1 from 0.65 to 0.82.
    #
    # It runs before the content-type rules rather than instead of them.
    # A cross-encoder rates an index row highly for a procedure question,
    # because the row is close to a title match, but a row carries no
    # steps and cannot answer a how-to. So rerank for relevance first,
    # then apply the rules that know what kind of passage was asked for.
    if cfg.get("rerank", True):
        pool   = merged[:cfg.get("rerank_pool", reranker.DEFAULT_POOL)]
        merged = reranker.rerank(query, pool, len(pool),
                                 model_name=cfg.get("rerank_model",
                                                    reranker.DEFAULT_MODEL))

    # A diagram request wants the figure itself at the front, so search the
    # captions directly rather than hoping fusion floats one up.
    if wants_diagram(query):
        caps = dense_search(query, col_k, satellite, 3,
                            {"content_type": {"$eq": "image_caption"}})
        merged = prepend(merged, caps, 2)

    # A procedure question wants the procedure document, and in these files
    # that usually means an index row rather than a body: the excerpt holds
    # 6 procedure bodies against 225 index rows, so the row is the only
    # chunk that can answer most of them. Search the whole procedure
    # document, not just the bodies.
    #
    # This is a recall step, not a ranking one. Procedure chunks are short,
    # so a longer manual section that merely mentions the subsystem beats
    # them on fusion even for a near-exact title match.
    if wants_procedure(query):
        procs = dense_search(query, col_k, satellite, 3,
                             {"source_doc": {"$eq": "FCP"}})
        merged = prepend(merged, procs, 2)

    return merged[:top_k]


def search(satellites: list, query: str, top_k: int = 5,
           doc_types=None, subsystem=None, mode: str = "hybrid") -> list:
    """
    Search one or many satellites. With several, each is searched
    independently and the ranked lists are fused, so no satellite can
    dominate purely because its store is larger.
    """
    if len(satellites) == 1:
        return search_one(satellites[0], query, top_k, doc_types, subsystem,
                          mode=mode)

    per_satellite = []
    for sat in satellites:
        try:
            hits = search_one(sat, query, top_k * 2, doc_types, subsystem,
                              mode=mode)
            if hits:
                per_satellite.append(hits)
        except FileNotFoundError:
            continue

    if not per_satellite:
        return []

    rrf_k = satellites[0].retrieval["rrf_k"]
    return rrf(per_satellite, rrf_k)[:top_k]


def format_hits(hits: list, show_satellite: bool = False,
                show_text: bool = False) -> str:
    if not hits:
        return "  no matches"

    lines, seen, n = [], set(), 1
    for h in hits:
        m = h.metadata

        # Command records share an empty heading and fcp_number, so keying
        # on those would collapse distinct commands into one line. Image
        # captions key on their file so a figure is never folded into a
        # text chunk from the same page: the diagram is its own source.
        if h.content_type == "command_log":
            key = (h.satellite, m.get("cid", ""), m.get("date", ""),
                   m.get("time", ""))
        elif h.content_type == "image_caption":
            key = (h.satellite, "IMG", m.get("image_file", ""))
        else:
            key = (h.satellite, m.get("fcp_number", ""),
                   m.get("section_heading", "")[:40])
        if key in seen:
            continue
        seen.add(key)

        lines.append(f"  [{n}] {h.cite(show_satellite)}")
        if h.content_type == "image_caption" and m.get("image_file"):
            lines.append(f"      image: {m['image_file']}")
        if show_text:
            snippet = h.text.strip().replace("\n", " ")[:200]
            lines.append(f"      {snippet}...")
        n += 1
    return "\n".join(lines)


def build_context(hits: list, show_satellite: bool = False,
                  char_budget: int = 2000) -> str:
    """Assemble retrieved chunks into a prompt context block."""
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[Source {i} | {h.cite(show_satellite)}]\n"
                      f"{h.text.strip()[:char_budget]}")
    return "\n\n".join(blocks)
