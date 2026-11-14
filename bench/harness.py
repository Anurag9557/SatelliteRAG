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

import hashlib
import json
import os
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
# Overridable so the OLLAMA_NUM_PARALLEL arm can be written to its own
# directory by the script that runs it, instead of being moved there by hand.
RESULT_DIR = Path(os.environ.get("BENCH_RESULT_DIR", BENCH_DIR / "results"))

# Matches the QA pairs lora/build_dataset.py asks the model to produce. The
# string "json" form that the project passes today gets valid JSON of any
# shape; this schema is what pins the shape down.
# The prompt asks for one or two pairs. Without minItems/maxItems the
# grammar permits an array of any length, and a guided decoder will happily
# keep emitting items until it hits max_tokens, leaving JSON that never
# closes. Ollama's backend terminated anyway; vLLM's did not, and 17 of 50
# chunks came back as unparseable truncated output that took five times
# longer to produce. Guided decoding constrains shape, not length, and the
# bound has to be written into the schema.
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

# A grammar enforces shape but cannot supply intent. Asked plainly, a small
# instruct model answers this prompt in markdown; forced by the grammar to
# open with "{" it is off the distribution it was trained on and falls back
# to the highest-probability legal token, which in JSON is whitespace, for
# as long as the token budget lasts. vLLM failed a third of chunks that way.
# Bounding the array, disable_any_whitespace and whitespace_pattern all made
# no difference; this system turn fixed it outright. Sent to both backends,
# so the arms differ only in how the schema is passed.
JSON_SYSTEM = (
    "You output only a single JSON object. No prose, no markdown, no code "
    "fences, and no blank space between tokens."
)

QA_PROMPT = (
    "Write 1 to 2 question/answer pairs that are answerable using ONLY the "
    "excerpt below. Keep answers to 1-3 sentences.\n\nExcerpt:\n{excerpt}"
)


def _builder_prompt():
    """lora/build_dataset.py's own system prompt, read from the file.

    The "json-string" arm above uses the bench's prompt, which never names
    a "pairs" key, so it scores zero on shape by construction and says
    nothing about how often the real builder loses a chunk. The "builder"
    arm sends exactly what build_dataset.py sends and parses the reply with
    exactly its rescue branches, so its drop count is the real one.
    Imported rather than copied, so an edit to the builder shows up here.

    Note the date. The 28 October numbers (42 pairs, 8 of 50 dropped) were
    measured against the builder as it was then: format="json" and a
    prompt asking for a bare array. build_dataset.py now passes a schema
    and asks for {"pairs": [...]}, so this arm, which still sends
    format="json" without the schema, no longer reproduces "today".
    """
    import importlib.util
    path = REPO_ROOT / "lora" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("_bd", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # stdlib-only at module level
    return mod.SYNTH_SYSTEM_PROMPT, mod.MAX_CHARS


def builder_parse(text: str):
    """build_dataset.synthesize_with_ollama's parsing, branch for branch.

    Returns (pairs, path) where path names which branch produced them, so
    the report can say how many chunks only survived because of a rescue.
    """
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return [], "unparseable"
    path = "array"
    if isinstance(data, dict):
        if "question" in data and "answer" in data:
            data, path = [data], "rescued:single-object"
        else:
            got = data.get("pairs") or data.get("qa")
            data, path = (got or []), ("rescued:wrapper-key" if got
                                       else "dropped:other-dict")
    pairs = []
    try:
        for item in data:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if q and a:
                pairs.append((q, a))
    except AttributeError:
        # A list of strings, or similar: build_dataset's blanket except
        # catches this and returns [], silently.
        return [], "dropped:exception"
    if not pairs and not path.startswith("dropped"):
        path = "dropped:empty"
    return pairs, path


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


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

    # Freeze the prompt alongside the context. See prompt_meta().
    import core.generate as gen
    meta = {
        "system_prompt": gen.SYSTEM_PROMPT,
        "not_found": getattr(gen, "NOT_FOUND_MESSAGE",
                             "Not found in available documents."),
    }
    (CACHE_DIR / f"prompt_{slug}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"    wrote prompt_{slug}.json")
    return out


def load_context(slug: str) -> list:
    path = CACHE_DIR / f"context_{slug}.json"
    if not path.exists():
        raise SystemExit(
            f"No context cache at {path}.\n"
            f"Run:  python bench/run.py cache {slug}")
    return json.loads(path.read_text(encoding="utf-8"))


def take(records: list, n: int) -> list:
    """First n answerable questions, plus every negative.

    Slicing the list plainly drops the negatives, because they sit at the
    end, and the quality scenario then reports 0/0 correct refusals while
    looking for all the world like it ran. A capped run should still
    exercise the refusal path, which is half of what answer quality means
    here.
    """
    pos = [r for r in records if not r.get("is_negative")][:n]
    neg = [r for r in records if r.get("is_negative")]
    return pos + neg


def prompt_meta(slug: str) -> dict:
    """The system prompt and refusal string, frozen when the cache was built.

    Reading them from disk rather than importing core.generate buys two
    things. The run phase stays free of the project's heavy imports, which
    have nothing to do with generation but pull in PyMuPDF and ChromaDB and
    so cannot live in a bare vLLM environment. And the prompt is pinned to
    the moment the cache was built, so two runs days apart compare backends
    rather than prompt edits.

    BENCH_PROMPT names a prompt in bench/prompts/ (v1, v2, ...) to run
    instead, for comparing prompt versions on the same frozen passages.
    The prompt's name goes into every result file, so two runs cannot be
    mixed up later.
    """
    choice = os.environ.get("BENCH_PROMPT")
    if choice:
        path = BENCH_DIR / "prompts" / f"{choice}.json"
        if not path.exists():
            raise SystemExit(f"BENCH_PROMPT={choice}: no {path}")
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.setdefault("name", choice)
        return meta
    path = CACHE_DIR / f"prompt_{slug}.json"
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.setdefault("name", "cache")
        return meta
    try:
        import core.generate as gen
        return {"system_prompt": gen.SYSTEM_PROMPT,
                "not_found": getattr(gen, "NOT_FOUND_MESSAGE",
                                     "Not found in available documents.")}
    except Exception as exc:
        raise SystemExit(
            f"No prompt metadata at {path}, and core.generate will not "
            f"import here ({type(exc).__name__}: {exc}).\n"
            f"Rebuild the cache where the project's dependencies are "
            f"installed:  python bench/run.py cache {slug}")


def load_chunks(slug: str, limit: int) -> list:
    """Raw chunks, for the structured-output scenario."""
    path = REPO_ROOT / "satellites" / slug / "extracted" / "chunks.json"
    if not path.exists():
        raise SystemExit(f"No chunks at {path}. Run cli.py build first.")
    chunks = json.loads(path.read_text(encoding="utf-8"))
    usable = [c for c in chunks if len((c.get("text") or "").strip()) > 200]
    return usable[:limit] if limit else usable


_GROUNDING = None


def _grounding():
    """core/grounding.py, loaded by path: standard library only, so this
    keeps the run phase free of core/__init__ and its heavy imports."""
    global _GROUNDING
    if _GROUNDING is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "grounding", REPO_ROOT / "core" / "grounding.py")
        _GROUNDING = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_GROUNDING)
    return _GROUNDING


def _messages(record: dict, meta: dict) -> list:
    """The real system prompt, so this measures the real workload.
    A prompt file with "context_note": true also gets the line the app adds
    after the question (core/grounding.context_note)."""
    user = (f"Context:\n{record['context']}\n\n"
            f"Question: {record['question']}")
    if meta.get("context_note"):
        note = _grounding().context_note(record["context"])
        if note:
            user += f"\n\n{note}"
    return [
        {"role": "system", "content": meta["system_prompt"]},
        {"role": "user", "content": user},
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


def scenario_latency(backend, records: list, max_tokens: int,
                     meta: dict) -> dict:
    """
    One request at a time, which is how the CLI is actually used.

    This is the scenario where vLLM has the least to offer and may well lose.
    Continuous batching does nothing for a queue of one, and a warm Ollama
    model has no engine to spin up. Worth measuring precisely because it is
    the case the project really runs.
    """
    rows = []
    for i, rec in enumerate(records, 1):
        r = backend.chat(_messages(rec, meta), max_tokens=max_tokens)
        rows.append({
            "id": rec["id"], "ok": r.ok, "error": r.error,
            "ttft_s": round(r.ttft_s, 4), "total_s": round(r.total_s, 4),
            "out_tokens": r.out_tokens, "tok_per_s": round(r.tok_per_s, 2),
            "chars": len(r.text),
        })
        print(f"      {i}/{len(records)}  ttft {r.ttft_s:5.2f}s  "
              f"total {r.total_s:6.2f}s", flush=True)

    good = [x for x in rows if x["ok"]]
    # Total time is only comparable when both sides write answers of the
    # same length, and the first run's did not (median 131 vs 172 tokens).
    # Seconds per 100 output tokens, prefill included, is the figure a user
    # waiting on an answer of a given size actually feels.
    per100 = [100 * x["total_s"] / x["out_tokens"]
              for x in good if x["out_tokens"]]
    return {
        "rows":      rows,
        "n_ok":      len(good),
        "n_failed":  len(rows) - len(good),
        "ttft_s":    _summarise([x["ttft_s"] for x in good]),
        "total_s":   _summarise([x["total_s"] for x in good]),
        "tok_per_s": _summarise([x["tok_per_s"] for x in good]),
        "s_per_100_tok": _summarise(per100),
        "out_tokens": _summarise([x["out_tokens"] for x in good]),
    }


def scenario_throughput(backend, records: list, max_tokens: int,
                        concurrency: int, meta: dict) -> dict:
    """
    The same requests fired with N in flight, which is the captioning and
    QA-synthesis shape: many independent jobs, nobody waiting on any one.

    Continuous batching should show here or nowhere.
    """
    print(f"      {len(records)} requests at concurrency {concurrency}")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        replies = list(pool.map(
            lambda rec: backend.chat(_messages(rec, meta), max_tokens=max_tokens),
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
                         repeats: int, meta: dict) -> dict:
    """
    The same question several times at temperature 0 and a fixed seed.

    This decides whether answer quality can be measured at all. A pipeline
    that gives a different answer each run cannot be regression-tested, so a
    score from it means nothing. vLLM documents a seed; Ollama takes one but
    makes no promise about it across restarts.
    """
    rows = []
    for rec in records:
        texts = [backend.chat(_messages(rec, meta), max_tokens=max_tokens,
                              seed=1234).text
                 for _ in range(repeats)]
        distinct = len(set(texts))
        digests = [_digest(t) for t in texts]
        # Where the first divergent repeat sits matters. If only repeat 0
        # differs, the cause is the prefill path (a cold prefix cache on the
        # first call, cache hits after it), not randomness.
        first_diff = next((i for i, d in enumerate(digests)
                           if d != digests[0]), None)
        rows.append({
            "id": rec["id"],
            "repeats": repeats,
            "distinct_outputs": distinct,
            "all_identical": distinct == 1,
            "digests": digests,
            "first_divergent_repeat": first_diff,
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


def classify_refusal(text: str, not_found: str) -> tuple:
    """(refused, tacked_on) for one answer.

    Counting any answer that contains the not-found sentence as a refusal
    overcounted the 7B badly: it often gives a real answer and then adds
    "Not found in available documents." at the end (29 flagged at 7B fp16,
    19 real when read). A refusal is an answer that says nothing else,
    meaning under 20 characters remain once the sentence is taken out (an
    FCP number and title on their own are an answer, not a refusal). An answer
    that does say something AND carries the sentence is counted apart, as
    a style fault the v2 prompt is meant to remove.
    """
    low = text.lower()
    if not_found.lower() not in low:
        return False, False
    rest = re.sub(re.escape(not_found.lower()), " ", low)
    rest = re.sub(r"[\s\W_]+", " ", rest).strip()
    if len(rest) < 20:
        return True, False
    return False, True


def scenario_quality(backend, records: list, max_tokens: int,
                     meta: dict) -> dict:
    """
    Answers dumped for scoring, with the two checks that can be automated.

    Faithfulness still needs a human read. What a script can tell you is
    whether the answer pointed at a source at all, and whether it refused.
    Both are worth counting before anyone starts reading.
    """
    not_found = meta["not_found"]

    cite = re.compile(r"\b(FCP[-\s]?\d{4}|p\.\s?\d+|MOR|OPS_MANUAL|"
                      r"GUIDELINES|HANDBOOK)\b", re.I)
    rows = []
    for i, rec in enumerate(records, 1):
        r = backend.chat(_messages(rec, meta), max_tokens=max_tokens)
        refused, tacked_on = classify_refusal(r.text, not_found)
        rows.append({
            "id": rec["id"], "question": rec["question"],
            "answer": r.text, "ok": r.ok,
            "is_negative": rec.get("is_negative", False),
            "refused": refused,
            "not_found_tacked_on": tacked_on,
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
        "not_found_tacked_on": sum(1 for x in pos if x["not_found_tacked_on"]),
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
              "none": None, "builder": "json"}[arm]
    if arm == "builder":
        synth_system, max_chars = _builder_prompt()

    rows = []
    for i, chunk in enumerate(chunks, 1):
        if arm == "builder":
            # Byte for byte what build_dataset.py sends today.
            messages = [
                {"role": "system", "content": synth_system},
                {"role": "user", "content":
                    f"Excerpt (from {chunk.get('doc_filename','')}):\n"
                    f"{chunk['text'][:max_chars]}"},
            ]
        else:
            prompt = QA_PROMPT.format(excerpt=chunk["text"][:3000])
            messages = [{"role": "system", "content": JSON_SYSTEM},
                        {"role": "user", "content": prompt}]
        r = backend.chat(messages, max_tokens=max_tokens, schema=schema)

        parsed, shaped, pairs, path = False, False, 0, ""
        if arm == "builder":
            got, path = builder_parse(r.text)
            parsed = path != "unparseable"
            shaped = path == "array" and bool(got)  # no rescue needed
            pairs = len(got)
        else:
            try:
                data = json.loads(r.text)
                parsed = True
                # The shape the caller wants with no rescue branching: an
                # object with a "pairs" list of question/answer objects.
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
            "pairs": pairs, "path": path, "total_s": round(r.total_s, 3),
            # Kept so a surprising count can be checked by reading the
            # reply. Results are gitignored, like the cache.
            "text": r.text[:2000],
        })
        if i % 10 == 0 or i == len(chunks):
            print(f"      {i}/{len(chunks)}", flush=True)

    n = len(rows) or 1
    out = {
        "arm": arm,
        "rows": rows,
        "n_chunks": len(rows),
        "parsed_rate": round(sum(1 for x in rows if x["parsed"]) / n, 3),
        "shaped_rate": round(sum(1 for x in rows if x["shaped"]) / n, 3),
        "total_pairs": sum(x["pairs"] for x in rows),
        "chunks_with_pairs": sum(1 for x in rows if x["pairs"]),
        "per_chunk_s": _summarise([x["total_s"] for x in rows]),
    }
    if arm == "builder":
        paths = {}
        for x in rows:
            paths[x["path"]] = paths.get(x["path"], 0) + 1
        out["paths"] = paths
        out["chunks_dropped"] = sum(1 for x in rows if not x["pairs"])
    return out
