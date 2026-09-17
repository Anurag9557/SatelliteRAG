"""
Build a QA fine-tuning dataset from the Aditya-L1 RAG chunks.

Reads satellites/aditya-l1/extracted/chunks.json (produced by core/extract.py)
and, for each usable chunk, asks the local Ollama model to synthesize 1-2
grounded question/answer pairs that use correct space and solar-mission
terminology. This reuses the same offline local-model infrastructure
core/generate.py already depends on, rather than calling an external API.

Output: lora/data/adityal1_qa.jsonl (alpaca-style {instruction, input,
output} records) plus a held-out lora/data/adityal1_qa_eval.jsonl split.

If Ollama is unavailable, falls back to a template-based generator built
from each chunk's section_heading + text. This produces lower-quality,
more repetitive data and is clearly logged as a degraded path, not silently
used as if it were the LLM-synthesized data.
"""

import argparse
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

Respond with ONLY a JSON array like:
[{"question": "...", "answer": "..."}]
No other text."""


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


def has_ollama(model: str) -> bool:
    try:
        import ollama
    except ImportError:
        return False
    try:
        names = {m.get("model", m.get("name", "")) for m in ollama.list().get("models", [])}
        return any(n.startswith(model.split(":")[0]) for n in names)
    except Exception:
        return False


def synthesize_with_ollama(chunk: dict, model: str) -> list:
    import ollama
    excerpt = chunk["text"][:MAX_CHARS]
    try:
        reply = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
                {"role": "user", "content": f"Excerpt (from {chunk.get('doc_filename','')}):\n{excerpt}"},
            ],
            options={"num_ctx": 4096},
            format="json",
        )
        content = reply["message"]["content"].strip()
        data = json.loads(content)
        if isinstance(data, dict):
            # format="json" constrains to valid JSON but not to the exact
            # requested shape, so the model often returns a single
            # {"question": ..., "answer": ...} object instead of a
            # one-item array. Treat that shape as a single pair; only
            # fall back to a wrapper-key list for other dict shapes.
            if "question" in data and "answer" in data:
                data = [data]
            else:
                data = data.get("pairs") or data.get("qa") or []
        pairs = []
        for item in data:
            q, a = (item.get("question") or "").strip(), (item.get("answer") or "").strip()
            if q and a:
                pairs.append((q, a))
        return pairs
    except Exception as exc:
        print(f"    warn: synthesis failed for {chunk['chunk_id']}: {exc}", file=sys.stderr)
        return []


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


def build(model: str, limit: int, seed: int) -> tuple:
    chunks = usable_chunks(load_chunks())
    random.Random(seed).shuffle(chunks)
    if limit:
        chunks = chunks[:limit]

    use_llm = has_ollama(model)
    mode = "ollama" if use_llm else "template (DEGRADED: no LLM synthesis)"
    print(f"Synthesizing QA pairs from {len(chunks)} chunks, mode={mode}")

    records = []
    for i, chunk in enumerate(chunks, 1):
        pairs = synthesize_with_ollama(chunk, model) if use_llm else synthesize_with_template(chunk)
        for q, a in pairs:
            records.append({
                "instruction": q,
                "input": "",
                "output": a,
                "source_chunk_id": chunk["chunk_id"],
                "source_doc": chunk.get("doc_filename", ""),
            })
        if i % 10 == 0 or i == len(chunks):
            print(f"  {i}/{len(chunks)} chunks -> {len(records)} pairs so far")

    random.Random(seed).shuffle(records)
    n_eval = max(1, int(len(records) * EVAL_FRACTION)) if records else 0
    eval_records, train_records = records[:n_eval], records[n_eval:]
    return train_records, eval_records, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="cap number of source chunks (0 = all)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_records, eval_records, mode = build(args.model, args.limit, args.seed)

    train_path = DATA_DIR / "adityal1_qa.jsonl"
    eval_path = DATA_DIR / "adityal1_qa_eval.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train_records) + "\n", encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_records) + "\n", encoding="utf-8")

    meta_path = DATA_DIR / "build_meta.json"
    meta_path.write_text(json.dumps({
        "mode": mode,
        "model": args.model,
        "n_train": len(train_records),
        "n_eval": len(eval_records),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(train_records)} train pairs -> {train_path}")
    print(f"Wrote {len(eval_records)} eval pairs -> {eval_path}")
    if mode != "ollama":
        print("\nWARNING: used the degraded template fallback, not LLM-synthesized data.")


if __name__ == "__main__":
    main()
