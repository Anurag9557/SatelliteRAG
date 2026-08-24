"""
Build one satellite's vector store.

Reads the extracted JSON in satellites/<slug>/extracted/, applies the
per-document-type chunking policy, embeds with the model named in that
satellite's config, and writes two ChromaDB collections plus two BM25
indexes into satellites/<slug>/vectorstore/.

Two collections, split by access pattern rather than document type:

    knowledge   procedures, guidelines, manual prose, image captions.
                Searched by meaning, top-k semantic.
    commands    the telecommand log. Looked up by exact filter (date,
                CID, subsystem) and returned as a set.

Document types stay separated inside 'knowledge' through the doc_type
field, which retrieval can filter on.
"""

import json
import pickle
import re
from pathlib import Path

from . import chunking, embeddings

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise SystemExit("Missing dependency: pip install chromadb")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


COLLECTION_KNOWLEDGE = "knowledge"
COLLECTION_COMMANDS  = "commands"


def tokenize(text: str) -> list:
    """
    Tokenizer for spacecraft documents.

    Compound identifiers are indexed whole and in parts, so a search for
    'OBC' reaches 'OBC-1', 'DC-DC' reaches 'DC-DC-11' and 'L1' reaches
    'L1/L2', while the full token stays available for exact matches.

    Without the parts, a procedure titled 'OBC-1 ON / OBC-2 ON' scores
    zero against a search for OBC, because 'obc-1' and 'obc' are
    different tokens.
    """
    lower = text.lower()
    whole = re.findall(r'[a-z0-9]+(?:[/\.\-][a-z0-9]+)+', lower)
    plain = re.findall(r'[a-z0-9_]+', lower)
    parts = []
    for tok in whole:
        parts.extend(re.split(r'[/\.\-]', tok))
    seen, out = set(), []
    for tok in whole + plain + parts:
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def embed_text(item: dict) -> str:
    """
    The string actually handed to the embedding model. Metadata is
    prefixed so a query like 'OBC switch on procedure' can reach FCP-5201
    and 'MRS recovery flow diagram' can reach the right caption.
    """
    parts = []
    ctype = item.get("content_type", "")

    if ctype == "image_caption":
        parts.append("Image caption")
        if item.get("source_doc"):
            parts.append(f"Source: {item['source_doc']}")
        if item.get("page"):
            parts.append(f"Page: {item['page']}")

    if item.get("fcp_number"):
        parts.append(f"FCP {item['fcp_number']}")
    sub = item.get("subsystem", "")
    if sub and sub not in ("", "UNKNOWN"):
        parts.append(f"Subsystem: {sub}")
    if item.get("mnemonic"):
        parts.append(f"Command: {item['mnemonic']}")
    if item.get("cid"):
        parts.append(f"CID: {item['cid']}")
    if item.get("section_heading"):
        parts.append(item["section_heading"])

    body = item.get("text", "").strip()
    if body:
        parts.append(body)
    return "\n".join(parts)


def metadata_for(item: dict, satellite) -> dict:
    """Flatten to ChromaDB-safe scalars. Lists and dicts are not allowed."""
    images = item.get("images", [])
    return {
        "satellite":       satellite.slug,
        "satellite_name":  satellite.name,
        "doc_type":        str(item.get("source_doc", "")),
        "doc_filename":    str(item.get("doc_filename", "")),
        "content_type":    str(item.get("content_type", "")),
        "page":            int(item.get("page") or 0),
        "section_heading": str(item.get("section_heading", ""))[:400],
        "fcp_number":      str(item.get("fcp_number", "")),
        "subsystem":       str(item.get("subsystem", "")),
        "is_sub_chunk":    bool(item.get("is_sub_chunk", False)),
        "parent_chunk_id": str(item.get("parent_chunk_id", "")),
        "sub_index":       int(item.get("sub_index", 0)),
        "date":            str(item.get("date", "")),
        "time":            str(item.get("time", "")),
        "cid":             str(item.get("cid", "")),
        "mnemonic":        str(item.get("mnemonic", "")),
        "orbit_no":        str(item.get("orbit_no", "")),
        "status":          str(item.get("status", "")),
        "source_station":  str(item.get("source_station", "")),
        "dest_station":    str(item.get("dest_station", "")),
        "has_images":      bool(images),
        "image_count":     int(len(images)),
        "image_file":      str(item.get("image_file", "")),
    }


def load_extracted(satellite) -> tuple:
    """Read the JSON that extraction produced for this satellite."""
    ex = satellite.extracted_dir
    knowledge, commands = [], []

    for filename in ("chunks.json", "image_captions.json"):
        path = ex / filename
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"    {filename:<24} {len(data):>6} items")
            knowledge.extend(data)
        else:
            print(f"    {filename:<24} {'missing':>6}")

    path = ex / "tch_records.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"    {'tch_records.json':<24} {len(data):>6} records")
        commands.extend(data)
    else:
        print(f"    {'tch_records.json':<24} {'missing':>6}")

    return knowledge, commands


def build_bm25(items: list, path: Path) -> None:
    if not HAS_BM25:
        return
    corpus = [tokenize(embed_text(it)) for it in items]
    ids    = [it.get("chunk_id") or it.get("record_id") for it in items]
    path.write_bytes(pickle.dumps({
        "bm25": BM25Okapi(corpus), "ids": ids, "items": items,
    }))


def upsert(collection, items: list, embedder, id_field: str,
           satellite) -> int:
    """Embed and store in batches, skipping ids already present."""
    if not items:
        return 0

    existing = set(collection.get(include=[])["ids"])
    pending  = [it for it in items if str(it.get(id_field, "")) not in existing]

    if not pending:
        print(f"    all {len(existing)} already stored")
        return 0

    done = 0
    step = embedder.batch_size
    for i in range(0, len(pending), step):
        batch = pending[i:i + step]
        collection.upsert(
            ids        = [str(it[id_field]) for it in batch],
            embeddings = embedder.encode_passages(
                            [embed_text(it) for it in batch]),
            documents  = [it.get("text", "")[:8000] for it in batch],
            metadatas  = [metadata_for(it, satellite) for it in batch],
        )
        done += len(batch)
        print(f"    {done}/{len(pending)}", end="\r", flush=True)

    print(f"    {done} embedded and stored          ")
    return done


def ingest(satellite, reset: bool = False, device: str = None) -> dict:
    """Build (or update) one satellite's vector store."""
    print(f"\n[{satellite.slug}] {satellite.name}")

    if not satellite.is_extracted():
        print(f"    nothing extracted yet at {satellite.extracted_dir}")
        print(f"    run extraction for this satellite first")
        return {"satellite": satellite.slug, "status": "no-extraction"}

    satellite.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    embedder = embeddings.load(satellite.embedding_model, device=device)
    print(f"    embedder: {embedder.describe()}")

    knowledge, commands = load_extracted(satellite)

    knowledge, dropped_k = chunking.deduplicate(knowledge, "chunk_id")
    commands,  dropped_c = chunking.deduplicate(commands,  "record_id")
    if dropped_k or dropped_c:
        print(f"    dropped {dropped_k + dropped_c} duplicate ids")

    knowledge, stats_k = chunking.apply(knowledge, satellite, "chunk_id")
    commands,  _       = chunking.apply(commands,  satellite, "record_id")

    print(f"    chunking: {stats_k['in']} -> {stats_k['out']} "
          f"(split {stats_k['split']}, avg {stats_k['avg_chars']}, "
          f"max {stats_k['max_chars']})")
    print(f"    per doc type: {stats_k['per_type']}")

    client = chromadb.PersistentClient(
        path     = str(satellite.vectorstore_dir),
        settings = Settings(anonymized_telemetry=False),
    )

    if reset:
        for name in (COLLECTION_KNOWLEDGE, COLLECTION_COMMANDS):
            try:
                client.delete_collection(name)
            except Exception:
                pass

    col_k = client.get_or_create_collection(
        COLLECTION_KNOWLEDGE, metadata={"hnsw:space": "cosine"})
    col_c = client.get_or_create_collection(
        COLLECTION_COMMANDS,  metadata={"hnsw:space": "cosine"})

    upsert(col_k, knowledge, embedder, "chunk_id",  satellite)
    upsert(col_c, commands,  embedder, "record_id", satellite)

    build_bm25(knowledge, satellite.vectorstore_dir / "bm25_knowledge.pkl")
    build_bm25(commands,  satellite.vectorstore_dir / "bm25_commands.pkl")

    # Record which model built this store, so retrieval can warn on mismatch.
    (satellite.vectorstore_dir / "build.json").write_text(json.dumps({
        "embedding_model": satellite.embedding_model,
        "hf_id":           embedder.spec.hf_id,
        "dim":             embedder.dim,
        "knowledge":       col_k.count(),
        "commands":        col_c.count(),
    }, indent=2), encoding="utf-8")

    print(f"    done: {col_k.count()} knowledge, {col_c.count()} commands")

    return {
        "satellite": satellite.slug,
        "status":    "ok",
        "knowledge": col_k.count(),
        "commands":  col_c.count(),
        "model":     satellite.embedding_model,
    }


def ingest_many(satellites: list, reset: bool = False,
                device: str = None) -> list:
    results = []
    for sat in satellites:
        try:
            results.append(ingest(sat, reset=reset, device=device))
        except Exception as exc:
            print(f"    FAILED: {exc}")
            results.append({"satellite": sat.slug, "status": f"error: {exc}"})
    return results
