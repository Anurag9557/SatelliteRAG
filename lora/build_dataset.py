"""
Build a QA fine-tuning dataset from the Aditya-L1 RAG chunks.

Reads satellites/aditya-l1/extracted/chunks.json (produced by core/extract.py)
and, for each usable chunk, asks the local model to synthesize 1-2
grounded question/answer pairs that use correct space and solar-mission
terminology. This reuses the same offline local-model infrastructure
core/generate.py already depends on, rather than calling an external API.

The model is reached through core/llm.py, so it is Ollama by default and a
local vLLM server with SATRAG_LLM=vllm.

Output: lora/data/adityal1_qa.jsonl (alpaca-style {instruction, input,
output} records) plus a held-out lora/data/adityal1_qa_eval.jsonl split.

If the model is unavailable, falls back to a template-based generator built
from each chunk's section_heading + text. This produces lower-quality,
more repetitive data and is clearly logged as a degraded path, not silently
used as if it were the LLM-synthesized data.

The reply shape is enforced by a JSON schema (QA_SCHEMA), not hoped for.
The earlier version passed format="json", which constrains the reply to
valid JSON of any shape. Measured on 50 chunks with qwen2.5 1.5B (bench/,
28 October 2026): not one reply came back as the array the prompt asked
for, 42 chunks survived only through a rescue branch that kept one pair
each, and 8 were dropped because the model used a "questions" key the
rescue did not check. 42 pairs in total. With the schema: 50 of 50 chunks
and 99 pairs. The rescue branches are kept as a fallback, but every chunk
that needs one, or yields nothing, is now counted in build_meta.json
instead of disappearing.
"""

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "satellites" / "aditya-l1" / "extracted" / "chunks.json"
DATA_DIR    = Path(__file__).resolve().parent / "data"

DEFAULT_MODEL = "qwen2.5:7b-instruct"
MIN_CHARS     = 200        # skip near-empty / noise chunks
MAX_CHARS     = 3000       # keep the source excerpt within context comfortably
EVAL_FRACTION = 0.15

SYNTH_SYSTEM_PROMPT = """You write flashcard-style question/answer pairs for \
training a space-domain QA assistant. Given an excerpt from a real ISRO \
Aditya-L1 mission document, write 1 to 2 question/answer pairs that:

- Use correct space, solar-physics, and spacecraft terminology (e.g.
  halo orbit, Lagrange point, coronagraph, spacecraft bus, payload,
  telemetry) wherever the excerpt supports it.
- Are answerable using ONLY the excerpt, with no outside knowledge.
- Have concise, factually precise answers (1-3 sentences).

Respond with ONLY a JSON object like:
{"pairs": [{"question": "...", "answer": "..."}]}
No other text."""

# The shape the decoder is held to. minItems/maxItems are not decoration:
# a guided decoder constrains shape, not length, and without an upper bound
# it can keep adding items until max_tokens cuts the JSON off unclosed.
QA_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer":   {"type": "string"},
                },
                "required": ["question", "answer"],
            },
        },
    },
    "required": ["pairs"],
}


def load_llm():
    """core/llm.py, loaded by path.

    Importing it as core.llm would run core/__init__.py, which imports the
    whole retrieval stack (ChromaDB, PyMuPDF, sentence-transformers). None
    of that is needed here, and it is not installed in the Linux venv that
    talks to vLLM. llm.py itself is standard library only.
    """
    path = REPO_ROOT / "core" / "llm.py"
    spec = importlib.util.spec_from_file_location("satrag_llm", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_chunks() -> list:
    if not CHUNKS_PATH.exists():
        sys.exit(f"No chunks found at {CHUNKS_PATH}. Run "
                  f"`python cli.py build aditya-l1` first.")
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def usable_chunks(chunks: list) -> list:
    out = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if len(text) < MIN_CHARS:
            continue
        out.append(c)
    return out


def parse_pairs(content: str) -> tuple:
    """
    (pairs, path). path says how the reply was read, so the build can
    report how many chunks needed a rescue and how many gave nothing.

    With the schema in force "schema" is the only path that should occur.
    The rest are what format="json" used to produce, kept so a server that
    ignores the schema degrades visibly rather than silently.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return [], "dropped:unparseable"

    path = "schema"
    if isinstance(data, dict):
        if isinstance(data.get("pairs"), list):
            data = data["pairs"]
        elif "question" in data and "answer" in data:
            data, path = [data], "rescued:single-object"
        else:
            got = data.get("qa") or data.get("questions")
            data, path = (got or []), ("rescued:wrapper-key" if got
                                       else "dropped:other-dict")
    elif isinstance(data, list):
        path = "rescued:bare-array"
    else:
        return [], "dropped:not-json-object"

    pairs = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            pairs.append((q, a))
    if not pairs and not path.startswith("dropped"):
        path = "dropped:empty"
    return pairs, path


def synthesize_with_llm(chunks: list, model: str, llm,
                        concurrency: int) -> tuple:
    """Every chunk in one batch; returns (pairs per chunk, path counts)."""
    batch = [[
        {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Excerpt (from {c.get('doc_filename','')}):\n{c['text'][:MAX_CHARS]}"},
    ] for c in chunks]

    replies = []
    step = 20   # report progress every 20 chunks without serialising them
    for start in range(0, len(batch), step):
        replies += llm.chat_many(batch[start:start + step], model=model,
                                 num_ctx=4096, max_tokens=512,
                                 schema=QA_SCHEMA, concurrency=concurrency)
        print(f"  {min(start + step, len(batch))}/{len(batch)} chunks")

    out, paths = [], {}
    for chunk, reply in zip(chunks, replies):
        if isinstance(reply, Exception):
            print(f"    warn: synthesis failed for {chunk['chunk_id']}: "
                  f"{reply}", file=sys.stderr)
            pairs, path = [], "dropped:request-failed"
        else:
            pairs, path = parse_pairs(reply)
        paths[path] = paths.get(path, 0) + 1
        out.append(pairs)
    return out, paths


def synthesize_with_template(chunk: dict) -> list:
    """Degraded fallback: no LLM. Turns a heading into a "what/describe"
    question and the chunk text (trimmed) into the answer."""
    heading = (chunk.get("section_heading") or "").strip()
    text = re.sub(r"\s+", " ", chunk["text"]).strip()
    answer = text[:400].rsplit(" ", 1)[0] + ("..." if len(text) > 400 else "")
    if heading and not heading.replace(" ", "").isdigit():
        question = f"What does the Aditya-L1 documentation say about {heading.strip('.')}?"
    else:
        question = f"Summarize this section from {chunk.get('doc_filename', 'the Aditya-L1 documents')}."
    return [(question, answer)]


def split_by_chunk(records: list, seed: int) -> tuple:
    """
    Hold out whole chunks, so every pair written from a chunk lands on
    one side.

    The split used to be made over pairs. Two questions written from the
    same chunk ask about the same few sentences, so with one in training
    and the other in eval the adapter was being scored on text it had
    been trained on: 71 of the 104 eval pairs shared a chunk with the
    training set.
    """
    chunk_ids = sorted({r["source_chunk_id"] for r in records})
    random.Random(seed).shuffle(chunk_ids)
    n_eval = max(1, int(len(chunk_ids) * EVAL_FRACTION)) if chunk_ids else 0
    held   = set(chunk_ids[:n_eval])

    train = [r for r in records if r["source_chunk_id"] not in held]
    evals = [r for r in records if r["source_chunk_id"] in held]
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(evals)
    return train, evals


def resplit() -> None:
    """Re-split the pairs already on disk, without asking the model again."""
    train_path = DATA_DIR / "adityal1_qa.jsonl"
    eval_path  = DATA_DIR / "adityal1_qa_eval.jsonl"
    records = []
    for path in (train_path, eval_path):
        if path.exists():
            records += [json.loads(line) for line in
                        path.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
    if not records:
        raise SystemExit(f"No pairs in {DATA_DIR}. Build the dataset first.")

    # Sorted first so the result does not depend on how the two files
    # happened to be ordered by the previous split.
    records.sort(key=lambda r: (r["source_chunk_id"], r["instruction"]))
    train, evals = split_by_chunk(records, seed=13)
    train_path.write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(r) for r in evals) + "\n", encoding="utf-8")

    meta_path = DATA_DIR / "build_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update({"n_train": len(train), "n_eval": len(evals),
                 "split": "by source chunk"})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    shared = {r["source_chunk_id"] for r in train} & {r["source_chunk_id"] for r in evals}
    print(f"Re-split {len(records)} pairs by chunk: {len(train)} train, "
          f"{len(evals)} eval, {len(shared)} chunks on both sides")


def build(model: str, limit: int, seed: int, concurrency: int) -> tuple:
    chunks = usable_chunks(load_chunks())
    random.Random(seed).shuffle(chunks)
    if limit:
        chunks = chunks[:limit]

    llm = load_llm()
    ok, why = llm.available(model)
    mode = llm.BACKEND if ok else "template (DEGRADED: no LLM synthesis)"
    print(f"Synthesizing QA pairs from {len(chunks)} chunks, mode={mode}")
    print(f"  {why}")
    if ok:
        print(f"  {llm.summary()}, concurrency={concurrency}")

    paths = {}
    if ok:
        per_chunk, paths = synthesize_with_llm(chunks, model, llm, concurrency)
    else:
        per_chunk = [synthesize_with_template(c) for c in chunks]

    records = []
    for chunk, pairs in zip(chunks, per_chunk):
        for q, a in pairs:
            records.append({
                "instruction": q,
                "input": "",
                "output": a,
                "source_chunk_id": chunk["chunk_id"],
                "source_doc": chunk.get("doc_filename", ""),
            })

    train_records, eval_records = split_by_chunk(records, seed)
    stats = {"n_chunks": len(chunks), "paths": paths,
             "split": "by source chunk",
             "chunks_dropped": sum(v for k, v in paths.items()
                                   if k.startswith("dropped")),
             "chunks_rescued": sum(v for k, v in paths.items()
                                   if k.startswith("rescued"))}
    return train_records, eval_records, mode, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="cap number of source chunks (0 = all)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="requests in flight (0 = SATRAG_LLM_CONCURRENCY, default 1). "
                         "Only helps with OLLAMA_NUM_PARALLEL set, or on vLLM.")
    ap.add_argument("--resplit", action="store_true",
                    help="re-split the pairs already in lora/data by chunk "
                         "instead of synthesizing again")
    args = ap.parse_args()

    if args.resplit:
        resplit()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_records, eval_records, mode, stats = build(
        args.model, args.limit, args.seed, args.concurrency or None)

    train_path = DATA_DIR / "adityal1_qa.jsonl"
    eval_path = DATA_DIR / "adityal1_qa_eval.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train_records) + "\n", encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_records) + "\n", encoding="utf-8")

    meta_path = DATA_DIR / "build_meta.json"
    meta_path.write_text(json.dumps({
        "mode": mode,
        "model": args.model,
        "limit": args.limit,
        "n_train": len(train_records),
        "n_eval": len(eval_records),
        **stats,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(train_records)} train pairs -> {train_path}")
    print(f"Wrote {len(eval_records)} eval pairs -> {eval_path}")
    if stats["paths"]:
        print(f"Reply shapes: {stats['paths']}")
    if stats["chunks_dropped"]:
        print(f"WARNING: {stats['chunks_dropped']} of {stats['n_chunks']} "
              f"chunks gave no pair; see build_meta.json")
    if not mode.startswith(("ollama", "vllm")):
        print("\nWARNING: used the degraded template fallback, not LLM-synthesized data.")


if __name__ == "__main__":
    main()
