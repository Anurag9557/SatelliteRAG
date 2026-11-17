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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import embeddings, router, rerank as reranker
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


# Store access

_clients = {}
_command_values = {}
_procedures = {}


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


def _known_command_values(satellite, col_c) -> dict:
    """
    The CIDs and mnemonics this satellite's log actually holds, read once.
    Keyed upper-case, mapped to the value exactly as stored.
    """
    if satellite.slug not in _command_values:
        metas = col_c.get(include=["metadatas"]).get("metadatas") or []
        _command_values[satellite.slug] = {
            key: {str(m[key]).upper(): m[key] for m in metas if m.get(key)}
            for key in ("cid", "mnemonic")
        }
    return _command_values[satellite.slug]


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
                doc_types=None, subsystem=None, date=None,
                content_types=None) -> list:
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

        # Apply exactly the filters the dense pass used. Letting UNKNOWN
        # through here would make the keyword pass more permissive than
        # the semantic one, so chunks the semantic filter excluded would
        # still reach the results through BM25.
        if doc_types and item.get("source_doc") not in doc_types:
            continue
        if content_types and item.get("content_type") not in content_types:
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


# Procedures: ranked by title, answered with steps
#
# A procedure is in the store twice: as one line of the index ("FCP 5201 -
# OBC - title") and, when the document holds it, as its steps. The two
# were ranked like any other chunks, and that went wrong in one direction.
# A body is a long table of step text and command codes. It scored well
# for any question about its subsystem, so the bodies of two or three
# procedures sat in the top five for most OBC questions and pushed out the
# index row of the procedure that was actually asked for. The model then
# answered from the wrong procedure.
#
# The fix is to make a procedure one candidate instead of two or four. Its
# index row stands for it. When the reranker judges it, it reads the title
# and the steps and the better of the two counts, so a procedure whose
# title is worded unlike the question can still be found by its steps, and
# one whose steps were never retrieved can still be found by its title.
# Once the order is settled, the best-placed rows are swapped for the
# steps they point to.

# How many procedures bring their steps. Each can be several thousand
# characters, and the model reads every passage it is given.
STEPS_FOR = 2

BODY_TYPES = ("procedure", "note")


def _join_pieces(parts: list) -> str:
    """Put a split body back together, dropping the overlap between pieces."""
    out = parts[0]
    for piece in parts[1:]:
        shared = 0
        for n in range(min(len(out), len(piece), 400), 20, -1):
            if out.endswith(piece[:n]):
                shared = n
                break
        out += piece[shared:] if shared else "\n" + piece
    return out


def _procedure_map(satellite, col_k) -> dict:
    """
    {"rows": {number: index row}, "bodies": {number: whole procedure}},
    read from the store once. A body cut into pieces at ingest is joined
    again here, so its steps reach the model as one passage in order.
    """
    # Keyed by the store, not the satellite: the comparison scripts point
    # one satellite at several stores in a single run.
    key = str(satellite.vectorstore_dir)
    if key not in _procedures:
        got = col_k.get(where={"doc_type": {"$eq": "FCP"}},
                        include=["documents", "metadatas"])
        rows, pieces = {}, {}
        for cid, doc, meta in zip(got.get("ids") or [],
                                  got.get("documents") or [],
                                  got.get("metadatas") or []):
            number = meta.get("fcp_number", "")
            if not number:
                continue
            if meta.get("content_type") == "toc_index":
                rows.setdefault(number, Hit(cid, doc or "", 0.0,
                                            satellite.slug, meta))
            elif meta.get("content_type") in BODY_TYPES:
                pieces.setdefault(number, []).append((cid, doc or "", meta))

        bodies = {}
        for number, found in pieces.items():
            found.sort(key=lambda t: (t[2].get("page", 0),
                                      t[2].get("parent_chunk_id", ""),
                                      t[2].get("sub_index", 0)))
            cid, _, meta = found[0]
            meta = {**meta, "is_sub_chunk": False, "sub_index": 0}
            bodies[number] = Hit(meta.get("parent_chunk_id") or cid,
                                 _join_pieces([doc for _, doc, _ in found]),
                                 0.0, satellite.slug, meta)
        _procedures[key] = {"rows": rows, "bodies": bodies}
    return _procedures[key]


def _is_body(hit) -> bool:
    return hit.doc_type == "FCP" and hit.content_type in BODY_TYPES


def rows_for_bodies(hits: list, procedures: dict) -> list:
    """
    Stand each procedure body's index row in for it, once per procedure.
    A body with no index row is left as it is.
    """
    out, seen = [], set()
    for hit in hits:
        row = procedures["rows"].get(hit.metadata.get("fcp_number", ""))
        if _is_body(hit) and row is not None:
            hit = Hit(row.chunk_id, row.text, hit.score, row.satellite,
                      row.metadata)
        key = (hit.satellite, hit.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def bodies_of(hits: list, procedures: dict) -> list:
    """The steps behind each index row in hits, for the reranker to read."""
    out = []
    for hit in hits:
        body = procedures["bodies"].get(hit.metadata.get("fcp_number", ""))
        if hit.content_type == "toc_index" and body is not None:
            out.append(Hit(body.chunk_id, body.text, hit.score,
                           body.satellite, body.metadata))
    return out


def steps_for_rows(hits: list, procedures: dict, limit: int = STEPS_FOR) -> list:
    """
    Swap the best-placed index rows for the steps they point to, where the
    store has them. One for one, so nothing else moves.
    """
    out, done = [], 0
    for hit in hits:
        body = procedures["bodies"].get(hit.metadata.get("fcp_number", ""))
        if hit.content_type == "toc_index" and body is not None and done < limit:
            hit = Hit(body.chunk_id, body.text, hit.score, body.satellite,
                      body.metadata)
            done += 1
        out.append(hit)
    return out


def search_commands(satellite, query: str, plan, top_k: int,
                    mode: str = "hybrid") -> list:
    """
    Search the telecommand log.

    Exact values in the plan (date, CID, mnemonic) are read straight from
    the store, every matching record in time order: "what was sent on
    <date>" wants the whole day, not the five records most similar to the
    wording. Without them, or when they match nothing, the log is searched
    like any other store.
    """
    cfg      = satellite.retrieval
    _, col_c = _collections(satellite)
    known    = _known_command_values(satellite, col_c)

    clauses = []
    if plan.date:
        clauses.append({"date": {"$eq": plan.date}})
    # A CID or mnemonic the log does not hold would empty the result, so
    # only values the store actually contains become filters.
    for key in ("cid", "mnemonic"):
        stored = known[key].get((getattr(plan, key) or "").upper())
        if stored:
            clauses.append({key: {"$eq": stored}})

    if clauses:
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        got   = col_c.get(where=where, include=["documents", "metadatas"])
        ids   = got.get("ids") or []
        docs  = got.get("documents") or []
        metas = got.get("metadatas") or []
        if metas:
            order = sorted(range(len(metas)), key=lambda i: (
                metas[i].get("date", ""), metas[i].get("time", "")))
            return [Hit(chunk_id=ids[i], text=docs[i] or "", score=1.0,
                        satellite=satellite.slug, metadata=metas[i])
                    for i in order]

    # The log follows the same mode as the documents. It used to run both
    # passes whatever was asked for, so a "dense" evaluation still had
    # BM25 answering its command questions.
    where = _where(date=plan.date)
    dense = [] if mode == "bm25" else dense_search(
        query, col_c, satellite, cfg["n_dense"], where)
    keyw  = [] if mode == "dense" else bm25_search(
        query, satellite.vectorstore_dir / "bm25_commands.pkl",
        satellite, cfg["n_bm25"], date=plan.date)
    return rrf([dense, keyw], cfg["rrf_k"])[:top_k]


def search_one(satellite, query: str, top_k=None, doc_types=None,
               subsystem=None, plan=None, mode="hybrid") -> list:
    """
    Search within a single satellite, following the query plan.

    mode selects which passes run over the documents:
        hybrid  dense + BM25 fused by RRF, the shipping pipeline
        dense   semantic only, which is what isolates the embedding model
        bm25    keyword only, an embedder-independent floor

    Use dense when comparing embedders. Hybrid mixes in BM25, which does
    not depend on the embedder at all, so it masks the difference between
    one model and another.
    """
    cfg   = satellite.retrieval
    top_k = top_k or cfg["top_k"]
    plan  = plan or router.plan(query, cfg)

    commands = []
    if "commands" in plan.stores:
        commands = search_commands(satellite, query, plan, top_k, mode)
        # Nothing in the log for a log question: the documents may still
        # answer it, so fall through rather than return nothing.
        if plan.stores == ["commands"] and commands:
            return commands

    col_k, _ = _collections(satellite)
    bm25_pkl = satellite.vectorstore_dir / "bm25_knowledge.pkl"

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
        query, bm25_pkl, satellite, cfg["n_bm25"],
        doc_types=doc_types, subsystem=sub)

    fcp_only  = {"doc_type": {"$eq": "FCP"}}
    wants_fcp = not doc_types or "FCP" in doc_types

    # The soft router never decides whether a question wants a procedure.
    # The procedure document is searched for every question and fused as a
    # third list, so an FCP chunk that is genuinely relevant rises on its
    # own and nothing is forced to the top.
    lists = [dense, keyw]
    if plan.router == "soft" and mode == "hybrid" and wants_fcp:
        lists.append(dense_search(
            query, col_k, satellite, cfg["n_dense"], fcp_only))
    merged = rrf(lists, cfg["rrf_k"])

    # From here on a procedure is its index row; see rows_for_bodies.
    procedures = _procedure_map(satellite, col_k)
    merged     = rows_for_bodies(merged, procedures)

    # Reranking reads each candidate against the question, which fusion
    # arithmetic cannot. On the 100-question bench it sharpens the first
    # position and can cost a place in the top five, so it is a setting
    # rather than a given; results/ has the numbers.
    #
    # It runs before the content-type rules rather than instead of them.
    # A cross-encoder rates an index row highly for a procedure question,
    # because the row is close to a title match, but a row carries no
    # steps and cannot answer a how-to. So rerank for relevance first,
    # then apply the rules that know what kind of passage was asked for.
    if cfg.get("rerank", True):
        pool   = merged[:cfg.get("rerank_pool", reranker.DEFAULT_POOL)]
        # A procedure is read twice, as its title and as its steps, and
        # keeps the better place of the two.
        pool   = pool + bodies_of(pool, procedures)
        merged = reranker.rerank(query, pool, len(pool),
                                 model_name=cfg.get("rerank_model",
                                                    reranker.DEFAULT_MODEL))
        merged = rows_for_bodies(merged, procedures)

    # A diagram request wants the figure itself at the front. Captions are
    # short, and a dense search alone missed some of them, so both passes
    # run over captions only and the best two go first.
    if plan.intent == "figure":
        caption_only = {"content_type": {"$eq": "image_caption"}}
        caps = rrf([
            [] if mode == "bm25" else dense_search(
                query, col_k, satellite, 3, caption_only),
            [] if mode == "dense" else bm25_search(
                query, bm25_pkl, satellite, 3,
                content_types=("image_caption",)),
        ], cfg["rrf_k"])
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
    if (plan.intent in ("procedure", "which_procedure")
            and plan.router != "soft" and wants_fcp):
        procs = rows_for_bodies(
            dense_search(query, col_k, satellite, 3, fcp_only), procedures)
        merged = prepend(merged, procs, 2)

    if commands:
        merged = rrf([merged, commands], cfg["rrf_k"])
    merged = merged[:top_k]

    # "Which FCP covers X" is answered by the index line itself. Anything
    # else that reaches a procedure wants its steps.
    if plan.intent != "which_procedure":
        merged = steps_for_rows(merged, procedures)
    return merged


def search(satellites: list, query: str, top_k: int = 5,
           doc_types=None, subsystem=None, mode: str = "hybrid",
           plan=None) -> list:
    """
    Search one or many satellites. With several, each is searched
    independently and the ranked lists are fused, so no satellite can
    dominate purely because its store is larger.

    The question is routed once, with the first satellite's settings, and
    every satellite follows that plan.
    """
    if not satellites:
        return []
    plan = plan or router.plan(query, satellites[0].retrieval)

    if len(satellites) == 1:
        return search_one(satellites[0], query, top_k, doc_types, subsystem,
                          plan=plan, mode=mode)

    per_satellite = []
    for sat in satellites:
        try:
            hits = search_one(sat, query, top_k * 2, doc_types, subsystem,
                              plan=plan, mode=mode)
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
            # fcp_number and section_heading are both blank for page-chunked
            # fallback text (e.g. a prose PDF with no numbered headings), so
            # falling back to the always-unique chunk_id there stops every
            # such chunk from collapsing onto the same dedup key.
            fcp_number, heading = m.get("fcp_number", ""), m.get("section_heading", "")
            key = (h.satellite, fcp_number, heading[:40]) if (fcp_number or heading) \
                else (h.satellite, "chunk", h.chunk_id)
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


# A procedure is given whole. Cut at the ordinary budget it loses its last
# steps and its notes, and a procedure that stops part-way is worse than
# none: it reads as complete.
PROCEDURE_BUDGET = 6000


def build_context(hits: list, show_satellite: bool = False,
                  char_budget: int = 2000, total_budget: int = 16000) -> str:
    """
    Assemble retrieved chunks into a prompt context block.

    A command-log lookup can return a whole day of records, far more than
    the model's context window holds, so the block stops at total_budget
    characters and says how many sources were left out.
    """
    blocks, used = [], 0
    for i, h in enumerate(hits, 1):
        budget = max(char_budget, PROCEDURE_BUDGET) if _is_body(h) else char_budget
        block = (f"[Source {i} | {h.cite(show_satellite)}]\n"
                 f"{h.text.strip()[:budget]}")
        if blocks and used + len(block) > total_budget:
            blocks.append(f"[{len(hits) - i + 1} further sources not shown]")
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
