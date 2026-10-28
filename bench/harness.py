"""
The scenarios, and the context cache they all share.

The cache is the part that makes the numbers mean anything. Retrieval runs
once, its top-k passages are written to disk, and every generation after that
replays the same passages to both backends. Without it each backend would be
answering from a slightly different context, because retrieval has its own
variance, and the difference between the two runtimes would be buried in it.

The cache holds document text, so it is confidential in exactly the way
satellites/<slug>/extracted/ is. bench/cache/ and bench/results/ are both in
.gitignore and must stay there.
"""

import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BENCH_DIR  = Path(__file__).resolve().parent
CACHE_DIR  = BENCH_DIR / "cache"
RESULT_DIR = BENCH_DIR / "results"

# Matches the QA pairs lora/build_dataset.py asks the model to produce. The
# string "json" form that the project passes today gets valid JSON of any
# shape; this schema is what pins the shape down.
QA_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
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

QA_PROMPT = (
    "Write 1 to 2 question/answer pairs that are answerable using ONLY the "
    "excerpt below. Keep answers to 1-3 sentences.\n\nExcerpt:\n{excerpt}"
)


# ---------------------------------------------------------------- cache


def build_context_cache(slug: str, limit: int = 0, force: bool = False) -> Path:
    """
    Run retrieval once over the labelled questions and store what came back.

    Everything downstream reads this file, so retrieval never runs during a
    timed scenario and both backends see byte-identical context.
    """
    from core import registry, retrieve
    import testset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"context_{slug}.json"
    if out.exists() and not force:
        print(f"    cache already at {out.name}, use --force to rebuild")
        return out

    sats = registry.resolve([slug])
    if not sats:
        raise SystemExit(f"no satellite matched {slug!r}")

    # The negatives belong in here. They are the only cases that test
    # whether the model declines when the documents do not answer, and a
    # backend that answers them anyway has a generation problem no latency
    # figure would show. They are tagged, not mixed in, so the report can
    # separate a correct refusal from a refusal of a question that did have
    # an answer.
    cases = [dict(c, is_negative=False) for c in testset.TEST_SET]
    if limit:
        cases = cases[:limit]
    cases += [dict(c, is_negative=True) for c in testset.NEGATIVE_CASES]

    print(f"    retrieving context for {len(cases)} questions "
          f"({sum(1 for c in cases if c['is_negative'])} negative)")
    records = []
    for i, case in enumerate(cases, 1):
        hits = retrieve.search(sats, case["q"], top_k=5)
        records.append({
            "id":          case["id"],
            "question":    case["q"],
            "context":     retrieve.build_context(hits),
            "n_hits":      len(hits),
            "is_negative": case["is_negative"],
        })
        if i % 20 == 0 or i == len(cases):
            print(f"      {i}/{len(cases)}")

    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"    wrote {out}")
    return out


def load_context(slug: str) -> list:
    path = CACHE_DIR / f"context_{slug}.json"
    if not path.exists():
        raise SystemExit(
            f"No context cache at {path}.\n"
            f"Run:  python bench/run.py cache {slug}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(slug: str, limit: int) -> list:
    """Raw chunks, for the structured-output scenario."""
    path = REPO_ROOT / "satellites" / slug / "extracted" / "chunks.json"
    if not path.exists():
        raise SystemExit(f"No chunks at {path}. Run cli.py build first.")
    chunks = json.loads(path.read_text(encoding="utf-8"))
    usable = [c for c in chunks if len((c.get("text") or "").strip()) > 200]
    return usable[:limit] if limit else usable


def _messages(record: dict) -> list:
    """The real system prompt, so this measures the real workload."""
    from core.generate import SYSTEM_PROMPT
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Context:\n{record['context']}\n\n"
                    f"Question: {record['question']}"},
    ]


def _summarise(values: list) -> dict:
    """Median and p95, never the mean. One slow first call drags a mean."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    idx = min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)
    return {
        "n":      len(ordered),
        "median": round(statistics.median(ordered), 4),
        "p95":    round(ordered[idx], 4),
        "min":    round(ordered[0], 4),
        "max":    round(ordered[-1], 4),
    }


# ------------------------------------------------------------ scenarios


def scenario_latency(backend, records: list, max_tokens: int) -> dict:
    """
    One request at a time, which is how the CLI is actually used.

    This is the scenario where vLLM has the least to offer and may well lose.
    Continuous batching does nothing for a queue of one, and a warm Ollama
    model has no engine to spin up. Worth measuring precisely because it is
    the case the project really runs.
    """
    rows = []
    for i, rec in enumerate(records, 1):
        r = backend.chat(_messages(rec), max_tokens=max_tokens)
        rows.append({
            "id": rec["id"], "ok": r.ok, "error": r.error,
            "ttft_s": round(r.ttft_s, 4), "total_s": round(r.total_s, 4),
            "out_tokens": r.out_tokens, "tok_per_s": round(r.tok_per_s, 2),
            "chars": len(r.text),
        })
        print(f"      {i}/{len(records)}  ttft {r.ttft_s:5.2f}s  "
              f"total {r.total_s:6.2f}s", flush=True)

    good = [x for x in rows if x["ok"]]
    return {
        "rows":      rows,
        "n_ok":      len(good),
        "n_failed":  len(rows) - len(good),
        "ttft_s":    _summarise([x["ttft_s"] for x in good]),
        "total_s":   _summarise([x["total_s"] for x in good]),
        "tok_per_s": _summarise([x["tok_per_s"] for x in good]),
    }


def scenario_throughput(backend, records: list, max_tokens: int,
                        concurrency: int) -> dict:
    """
    The same requests fired with N in flight, which is the captioning and
    QA-synthesis shape: many independent jobs, nobody waiting on any one.

    Continuous batching should show here or nowhere.
    """
    print(f"      {len(records)} requests at concurrency {concurrency}")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        replies = list(pool.map(
            lambda rec: backend.chat(_messages(rec), max_tokens=max_tokens),
            records))
    wall = time.perf_counter() - start

    good   = [r for r in replies if r.ok]
    tokens = sum(r.out_tokens for r in good)
    return {
        "concurrency":     concurrency,
        "n_requests":      len(replies),
        "n_ok":            len(good),
        "n_failed":        len(replies) - len(good),
        "wall_clock_s":    round(wall, 2),
        "requests_per_s":  round(len(good) / wall, 3) if wall else 0,
        "out_tokens":      tokens,
        "tokens_per_s":    round(tokens / wall, 1) if wall else 0,
        "per_request_s":   _summarise([round(r.total_s, 4) for r in good]),
    }


def scenario_determinism(backend, records: list, max_tokens: int,
                         repeats: int) -> dict:
    """
    The same question several times at temperature 0 and a fixed seed.

    This decides whether answer quality can be measured at all. A pipeline
    that gives a different answer each run cannot be regression-tested, so a
    score from it means nothing. vLLM documents a seed; Ollama takes one but
    makes no promise about it across restarts.
    """
    rows = []
    for rec in records:
        texts = [backend.chat(_messages(rec), max_tokens=max_tokens,
                              seed=1234).text
                 for _ in range(repeats)]
        distinct = len(set(texts))
        rows.append({
            "id": rec["id"],
            "repeats": repeats,
            "distinct_outputs": distinct,
            "all_identical": distinct == 1,
        })
        print(f"      {rec['id']:<22} {distinct} distinct / {repeats}",
              flush=True)

    identical = sum(1 for x in rows if x["all_identical"])
    return {
        "rows": rows,
        "n_questions": len(rows),
        "n_fully_reproducible": identical,
        "reproducible_rate": round(identical / len(rows), 3) if rows else 0,
    }


def scenario_quality(backend, records: list, max_tokens: int) -> dict:
    """
    Answers dumped for scoring, with the two checks that can be automated.

    Faithfulness still needs a human read. What a script can tell you is
    whether the answer pointed at a source at all, and whether it refused.
    Both are worth counting before anyone starts reading.
    """
    # Named constant only in the newer core/generate.py. Older copies put
    # the same sentence in SYSTEM_PROMPT without naming it, so fall back to
    # the literal rather than making this scenario need a particular commit.
    import core.generate as gen
    not_found = getattr(gen, "NOT_FOUND_MESSAGE",
                        "Not found in available documents.")

    cite = re.compile(r"\b(FCP[-\s]?\d{4}|p\.\s?\d+|MOR|OPS_MANUAL|"
                      r"GUIDELINES|HANDBOOK)\b", re.I)
    rows = []
    for i, rec in enumerate(records, 1):
        r = backend.chat(_messages(rec), max_tokens=max_tokens)
        refused = not_found.lower() in r.text.lower()
        rows.append({
            "id": rec["id"], "question": rec["question"],
            "answer": r.text, "ok": r.ok,
            "is_negative": rec.get("is_negative", False),
            "refused": refused,
            "has_citation": bool(cite.search(r.text)),
            "chars": len(r.text), "out_tokens": r.out_tokens,
        })
        print(f"      {i}/{len(records)}  {rec['id']}", flush=True)

    good = [x for x in rows if x["ok"]]
    pos  = [x for x in good if not x["is_negative"]]
    neg  = [x for x in good if x["is_negative"]]

    # Refusal only means something split this way. Refusing the two
    # out-of-scope questions is the behaviour we want; refusing a question
    # the documents do answer is a failure, and one number covering both
    # would hide whichever way it went.
    return {
        "rows": rows,
        "n_ok": len(good),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "correct_refusals": sum(1 for x in neg if x["refused"]),
        "false_refusals": sum(1 for x in pos if x["refused"]),
        "false_refusal_rate": round(
            sum(1 for x in pos if x["refused"]) / len(pos), 3) if pos else 0,
        "cited_rate": round(
            sum(1 for x in pos if x["has_citation"]) / len(pos), 3)
        if pos else 0,
        "median_chars": int(statistics.median(
            [x["chars"] for x in pos])) if pos else 0,
    }


def scenario_structured(backend, chunks: list, arm: str,
                        max_tokens: int) -> dict:
    """
    The QA-synthesis call from lora/build_dataset.py, three ways.

    arm "json-string"  what the project passes today: valid JSON, any shape
    arm "schema"       the same server, given the schema it will honour
    arm "none"         no constraint, as a floor

    Counted: did it parse, and did the parsed object have the shape the
    caller expects without any fallback branching. The second number is the
    one that matters, because every chunk that misses it is a training pair
    the adapter never sees.
    """
    schema = {"json-string": "json", "schema": QA_SCHEMA,
              "none": None}[arm]

    rows = []
    for i, chunk in enumerate(chunks, 1):
        prompt = QA_PROMPT.format(excerpt=chunk["text"][:3000])
        r = backend.chat([{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, schema=schema)

        parsed, shaped, pairs = False, False, 0
        try:
            data = json.loads(r.text)
            parsed = True
            # The shape the caller wants with no rescue branching: an object
            # with a "pairs" list of question/answer objects.
            items = data.get("pairs") if isinstance(data, dict) else None
            if isinstance(items, list) and items and all(
                    isinstance(p, dict) and p.get("question") and
                    p.get("answer") for p in items):
                shaped, pairs = True, len(items)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        rows.append({
            "chunk_id": chunk.get("chunk_id", f"#{i}"),
            "ok": r.ok, "parsed": parsed, "shaped": shaped,
            "pairs": pairs, "total_s": round(r.total_s, 3),
        })
        if i % 10 == 0 or i == len(chunks):
            print(f"      {i}/{len(chunks)}", flush=True)

    n = len(rows) or 1
    return {
        "arm": arm,
        "rows": rows,
        "n_chunks": len(rows),
        "parsed_rate": round(sum(1 for x in rows if x["parsed"]) / n, 3),
        "shaped_rate": round(sum(1 for x in rows if x["shaped"]) / n, 3),
        "total_pairs": sum(x["pairs"] for x in rows),
        "per_chunk_s": _summarise([x["total_s"] for x in rows]),
    }
