"""
Turn the closed-book QA pairs into the task the app actually has: answer
from five retrieved passages, cite the one used, and say not found when
none of them holds the answer.

The first adapter (lora/train.py) was trained on question -> answer with
no passages, so it could only learn the answers' style, not their facts
(walkthrough chapter 10). Here every example is laid out the way
core/generate.answer() lays out a real question:

    system   generate.SYSTEM_PROMPT, the prompt in use
    user     "Context:\n<5 passages>\n\nQuestion: <q>" (+ the context note)

The five passages are the pair's own source chunk plus four of the
passages the app's retriever returns for that question, in a shuffled
order. The target is the reference answer followed by [Source n], where n
is wherever the source chunk landed.

Refusal examples: the same question with the source chunk taken out, and
every other passage that holds the answer taken out too. The target is
generate.NOT_FOUND_MESSAGE. Without them a model fine-tuned only on
answerable questions learns to answer everything.

The train/eval split is the one already on disk (by source chunk, 595 /
101, nothing shared), so the eval questions come from chunks the model
never saw in training.

    python lora/build_rag_dataset.py              writes lora/data/rag_*.jsonl
    python lora/build_rag_dataset.py --limit 20   a quick look

Needs the working vector store. Retrieval runs on this machine; the
training and evaluation run wherever the GPU is.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import generate, grounding, registry, retrieve, router   # noqa: E402
from core.retrieve import Hit                                       # noqa: E402

DATA     = REPO / "lora" / "data"
CHUNKS   = REPO / "satellites" / "aditya-l1" / "extracted" / "chunks.json"
PASSAGES = 5
REFUSE_SHARE_TRAIN = 0.3      # of training questions, also given as a refusal
SEED = 13

STOP = set("a an the of to in on for and or is are was were be by with as at "
           "from that this it its which what how when".split())


def words(text: str) -> list:
    """Content words, lower case, with a plural s taken off."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w[-2].isdigit() else w)
    return out


def holds_answer(text: str, answer: str) -> bool:
    """
    True when a text carries the reference answer: used to score a reply
    as right.

    Short answers ("BUS-A", "16 bits") are matched as text; longer ones by
    how many of their content words the text has.
    """
    gold = words(answer)
    if not gold:
        return False
    if len(gold) <= 3:
        return " ".join(gold) in " ".join(words(text))
    have = set(words(text))
    return sum(w in have for w in gold) / len(gold) >= 0.7


def mentions_answer(passage: str, answer: str) -> bool:
    """
    A looser test, for keeping the answer out of refusal examples and for
    crediting a citation. The documents repeat the same material in
    different words (the report, the manual and the guidelines overlap),
    so a passage that shares half the answer's content words, or any of
    its identifiers or numbers, may well answer the question. The first
    build used holds_answer here, and reading the replies showed refusal
    examples whose passages did hold the answer.
    """
    gold = words(answer)
    if not gold:
        return False
    have = set(words(passage))
    keys = [w for w in gold if len(w) >= 3 and any(c.isdigit() for c in w)]
    if any(k in have for k in keys):
        return True
    return sum(w in have for w in gold) / len(gold) >= 0.5


def as_hit(chunk: dict) -> Hit:
    """A chunk from chunks.json, shaped like a hit from the store."""
    meta = {k: chunk.get(k) for k in
            ("content_type", "page", "section_heading", "fcp_number",
             "subsystem") if chunk.get(k)}
    meta["doc_type"] = chunk.get("source_doc", "")
    return Hit(chunk["chunk_id"], chunk["text"], 1.0, "aditya-l1", meta)


def same_chunk(hit: Hit, chunk_id: str) -> bool:
    # Long procedure bodies are stored in pieces named <id>_s1, _s2, ...
    return hit.chunk_id == chunk_id or hit.chunk_id.startswith(chunk_id + "_s")


def messages_for(question: str, hits: list) -> list:
    context = retrieve.build_context(hits)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    note = grounding.context_note(context, related=generate.NOTE_RELATED)
    if note:
        user += f"\n\n{note}"
    return [{"role": "system", "content": generate.SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def build(pairs: list, chunks: dict, sat, rng: random.Random,
          refuse_share: float, both: bool) -> tuple:
    """
    Examples for one side of the split. `both` gives every question as an
    answer and as a refusal (the eval set); otherwise a share of them is
    also given as a refusal (the training set).
    """
    out, skipped = [], {"no source chunk": 0, "no clean refusal": 0}
    for n, pair in enumerate(pairs, 1):
        source = chunks.get(pair["source_chunk_id"])
        if not source:
            skipped["no source chunk"] += 1
            continue
        question, answer = pair["instruction"], pair["output"]

        plan = router.plan(question, sat.retrieval)
        found = retrieve.search([sat], question, top_k=PASSAGES + 15, plan=plan)
        others = [h for h in found if not same_chunk(h, source["chunk_id"])]

        # Answer: the source among four retrieved passages.
        hits = [as_hit(source)] + others[:PASSAGES - 1]
        rng.shuffle(hits)
        cite = [i for i, h in enumerate(hits, 1)
                if same_chunk(h, source["chunk_id"]) or mentions_answer(h.text, answer)]
        first = next(i for i, h in enumerate(hits, 1)
                     if same_chunk(h, source["chunk_id"]))
        out.append({
            "kind": "answer", "question": question, "gold": answer,
            "messages": messages_for(question, hits),
            "target": f"{answer} [Source {first}]",
            "answer_sources": cite, "source_chunk_id": source["chunk_id"],
        })

        # Refusal: the same question, with nothing that holds the answer.
        if both or rng.random() < refuse_share:
            clean = [h for h in others if not mentions_answer(h.text, answer)]
            if len(clean) < PASSAGES:
                skipped["no clean refusal"] += 1
            else:
                hits = clean[:PASSAGES]
                rng.shuffle(hits)
                out.append({
                    "kind": "refuse", "question": question, "gold": answer,
                    "messages": messages_for(question, hits),
                    "target": generate.NOT_FOUND_MESSAGE,
                    "answer_sources": [], "source_chunk_id": source["chunk_id"],
                })
        if n % 50 == 0:
            print(f"  {n}/{len(pairs)}", flush=True)
    return out, skipped


def read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="pairs per side, for a quick look")
    args = ap.parse_args()

    train = read_jsonl(DATA / "adityal1_qa.jsonl")
    evals = read_jsonl(DATA / "adityal1_qa_eval.jsonl")
    if args.limit:
        train, evals = train[:args.limit], evals[:args.limit]
    chunks = {c["chunk_id"]: c for c in json.loads(CHUNKS.read_text(encoding="utf-8"))}
    sat = registry.load_satellite("aditya-l1")
    rng = random.Random(SEED)

    print(f"training side, {len(train)} pairs")
    train_rows, train_skip = build(train, chunks, sat, rng, REFUSE_SHARE_TRAIN, both=False)
    print(f"evaluation side, {len(evals)} pairs")
    eval_rows, eval_skip = build(evals, chunks, sat, rng, 0, both=True)
    rng.shuffle(train_rows)

    for name, rows in (("rag_train", train_rows), ("rag_eval", eval_rows)):
        (DATA / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
    meta = {
        "passages": PASSAGES, "refuse_share_train": REFUSE_SHARE_TRAIN,
        "prompt": generate.PROMPT_VERSION,
        "train": {k: sum(r["kind"] == k for r in train_rows) for k in ("answer", "refuse")},
        "eval": {k: sum(r["kind"] == k for r in eval_rows) for k in ("answer", "refuse")},
        "skipped_train": train_skip, "skipped_eval": eval_skip,
    }
    (DATA / "rag_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
