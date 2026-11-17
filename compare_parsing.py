"""
Compare the two ways of reading the documents, on the labelled test set.

    flat        core/extract.py. Sections of plain text, cut near a
                character count with an overlap. What results/ was
                measured on.
    structure   core/layout.py. Tables read row by row and kept in their
                section, report sections taken from the PDF's bookmarks,
                page headers and the contents pages removed, and chunks
                cut between paragraphs and rows.

The structure variant is extracted and embedded into folders of its own
(extracted_structure/ and vectorstore_structure/), so the working
extraction and the working store are not touched and the two can be
scored side by side.

    python compare_parsing.py               build the variant, score both
    python compare_parsing.py --eval-only   score the stores already built

Everything else is held equal: the same embedder, the same chunk budgets,
the same figure captions, the same retrieval settings. Each variant is
scored the way the pipeline ships (hybrid with the reranker) and with one
part at a time switched off, so a change can be traced to the pass it
came from.

Each run is scored twice. "As labelled" is the test set as it stands.
"Contents not counted" does not accept a chunk from the pages before a
PDF's first bookmark: those are the table of contents, a contents line
holds the words of every heading, and so it satisfies an accept condition
written for the section that heading opens without answering anything.
The flat reader keeps those pages and the structure reader drops them, so
only the second scoring compares like with like.

Writes results/parsing_comparison.csv and results/parsing_chunks.csv
(numbers only). Per-question ranks go to runs/parsing/, which is kept out
of git.
"""

import argparse
import csv
import json
import re
import shutil
import statistics
import time
from pathlib import Path

import pymupdf as fitz

from collect_results import interval
from core import chunking, embeddings, extract, ingest, registry, retrieve
from evaluate import aggregate, first_rank, label, relevant
from testset import TEST_SET

RESULTS = Path("results")
RUNS    = Path("runs") / "parsing"

VARIANT = "structure"
POOL    = 20

# (name, mode, reranker). The first is the pipeline as it ships.
SETTINGS = [
    ("hybrid + reranker", "hybrid", True),
    ("hybrid",            "hybrid", False),
    ("dense",             "dense",  False),
    ("bm25",              "bm25",   False),
]


def variant_of(slug: str):
    sat = registry.load_satellite(slug)
    sat.parsing_override = VARIANT
    sat.extract_suffix   = VARIANT
    sat.store_suffix     = VARIANT
    return sat


def build(base, sat) -> None:
    extract.extract(sat, force=True, images=False)
    # The figures are the same figures however the text around them is
    # read, so the variant takes the captions already made.
    captions = base.extracted_dir / "image_captions.json"
    if captions.exists():
        shutil.copyfile(captions, sat.extracted_dir / "image_captions.json")
    ingest.ingest(sat, reset=True)


def chunk_shape(sat, name: str) -> dict:
    """What the chunks look like, before any question is asked."""
    knowledge, _ = ingest.load_extracted(sat)
    knowledge, _ = chunking.deduplicate(knowledge, "chunk_id")
    pieces, _    = chunking.apply(knowledge, sat, "chunk_id")
    pieces = [p for p in pieces if p.get("content_type") not in
              ("image_caption", "toc_index")]

    embedder = embeddings.load(sat.embedding_model)
    window   = embedder._model.max_seq_length
    tokens   = [len(embedder._model.tokenizer(ingest.embed_text(p))["input_ids"])
                for p in pieces]

    later = [p for p in pieces if p.get("sub_index", 0) > 0]
    mid   = sum(1 for p in later
                if re.match(r"[a-z,;)\]]", p["text"].lstrip()[:1] or "x"))
    sizes = [len(p["text"]) for p in pieces]
    return {
        "parsing":            name,
        "text_chunks":        len(pieces),
        "median_chars":       int(statistics.median(sizes)),
        "max_chars":          max(sizes),
        "under_100_chars":    sum(1 for s in sizes if s < 100),
        "continuations":      len(later),
        "start_mid_sentence": mid,
        "embedder_window":    window,
        "longer_than_window": sum(1 for t in tokens if t > window),
    }


SCORINGS = ("as labelled", "contents not counted")


def first_bookmark(sat) -> dict:
    """{doc_type: page of the first bookmark} for each PDF that has them."""
    out = {}
    for doc_type, entry in sat.documents.items():
        path = entry["path"]
        if path.suffix.lower() != ".pdf" or not path.exists():
            continue
        doc   = fitz.open(str(path))
        pages = [page for _, _, page in doc.get_toc() if page >= 1]
        doc.close()
        if len(pages) >= 5 and min(pages) > 1:
            out[doc_type] = min(pages)
    return out


def is_contents(hit, front: dict) -> bool:
    meta  = hit.metadata or {}
    first = front.get(meta.get("doc_type", ""))
    return (bool(first) and meta.get("content_type") == "section"
            and 0 < int(meta.get("page") or 0) < first)


def score(sat, mode: str, rerank: bool, front: dict) -> dict:
    """{scoring: (rows, aggregate)} for one store under one setting."""
    cfg = dict(sat.config.get("retrieval") or {})
    cfg["rerank"] = rerank
    sat.config["retrieval"] = cfg
    # Both variants are the same satellite, and the store handles are
    # cached by satellite, so they are dropped before each run.
    retrieve._clients.clear()
    retrieve._command_values.clear()

    rows = {name: [] for name in SCORINGS}
    for case in TEST_SET:
        hits = retrieve.search([sat], case["q"], top_k=POOL, mode=mode)
        for name in SCORINGS:
            right = [bool(case["accept"]) and relevant(h, case["accept"])
                     and not (name != "as labelled" and is_contents(h, front))
                     for h in hits]
            rank = right.index(True) + 1 if True in right else 0
            rows[name].append({
                "id":     case["id"],
                "rank":   rank,
                "hit@1":  int(rank == 1),
                "hit@3":  int(1 <= rank <= 3),
                "hit@5":  int(1 <= rank <= 5),
                "hit@10": int(1 <= rank <= 10),
                "rr":     (1.0 / rank) if rank else 0.0,
                "top":    label(hits[0]) if hits else "",
                "found":  label(hits[rank - 1]) if rank else "",
                # How many of the top five are right, not whether one is.
                "right_in_5":    sum(right[:5]),
                "contents_in_5": sum(is_contents(h, front) for h in hits[:5]),
            })
    return {name: (rows[name], aggregate(rows[name])) for name in SCORINGS}


def main():
    ap = argparse.ArgumentParser(description="Compare parsing modes")
    ap.add_argument("--satellite", default="aditya-l1")
    ap.add_argument("--eval-only", action="store_true",
                    help="score the existing stores without rebuilding")
    args = ap.parse_args()

    base = registry.load_satellite(args.satellite)
    sat  = variant_of(args.satellite)
    if not base.is_ingested():
        raise SystemExit(f"{base.slug} has no working store to compare with. "
                         f"Run: python cli.py build {base.slug}")

    if not args.eval_only:
        build(base, sat)
    elif not sat.is_ingested():
        raise SystemExit(f"no {sat.vectorstore_dir.name}/ yet, "
                         f"run once without --eval-only")

    RESULTS.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    shapes = [chunk_shape(base, "flat"), chunk_shape(sat, VARIANT)]
    with (RESULTS / "parsing_chunks.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(shapes[0]))
        writer.writeheader()
        writer.writerows(shapes)
    print("\n  Chunks (captions and index rows left out; they do not change)")
    for key in shapes[0]:
        if key != "parsing":
            print(f"  {key:<22}{shapes[0][key]:>10}{shapes[1][key]:>12}")

    front = first_bookmark(base)
    table, ranks, scored = [], {}, {}
    for name, mode, rerank in SETTINGS:
        for parsing, which in (("flat", base), (VARIANT, sat)):
            start = time.time()
            scored[(name, parsing)] = score(which, mode, rerank, front)
            took  = round(time.time() - start, 1)
            for scoring, (rows, agg) in scored[(name, parsing)].items():
                n = len(rows)
                ranks[(scoring, name, parsing)] = {r["id"]: r["rr"] for r in rows}
                table.append({
                    "scoring": scoring, "setting": name, "parsing": parsing,
                    "pool": POOL,
                    **{k: round(agg[k], 4) for k in (
                        "hit@1", "hit@3", "hit@5", "hit@10", "recall",
                        "mrr", "mrr_strict")},
                    "right_in_top5": round(
                        sum(r["right_in_5"] for r in rows) / n, 3),
                    "contents_in_top5": round(
                        sum(r["contents_in_5"] for r in rows) / n, 3),
                    "seconds": took,
                })
                tag = (f"{parsing}_{mode}{'_rerank' if rerank else ''}"
                       f"{'' if scoring == SCORINGS[0] else '_no_contents'}")
                (RUNS / f"{tag}.json").write_text(
                    json.dumps(rows, indent=2), encoding="utf-8")

    for row in table:
        if row["parsing"] != VARIANT:
            continue
        key = (row["scoring"], row["setting"])
        mean, low, high = interval(ranks[(*key, VARIANT)], ranks[(*key, "flat")])
        row.update({"mrr_change": round(mean, 4), "low": round(low, 4),
                    "high": round(high, 4)})

    for scoring in SCORINGS:
        print(f"\n  Scoring: {scoring}")
        print(f"  {'setting':<20}{'parsing':<11}{'H@1':>6}{'H@5':>6}"
              f"{'H@10':>7}{'found':>7}{'MRR':>8}{'strict':>8}{'right/5':>9}")
        print("  " + "-" * 80)
        for row in table:
            if row["scoring"] != scoring:
                continue
            print(f"  {row['setting']:<20}{row['parsing']:<11}"
                  f"{row['hit@1']*100:5.0f}%{row['hit@5']*100:5.0f}%"
                  f"{row['hit@10']*100:6.0f}%{row['recall']*100:6.0f}%"
                  f"{row['mrr']:8.3f}{row['mrr_strict']:8.3f}"
                  f"{row['right_in_top5']:9.2f}")
            if "mrr_change" in row:
                mark = ("" if row["low"] > 0 or row["high"] < 0
                        else "   spans zero")
                print(f"  {'':<20}{VARIANT} minus flat, MRR "
                      f"{row['mrr_change']:+.3f}  [{row['low']:+.3f}, "
                      f"{row['high']:+.3f}]{mark}")

    cols = list(max(table, key=len))     # a structure row; flat has no change
    with (RESULTS / "parsing_comparison.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, restval="")
        writer.writeheader()
        writer.writerows(table)
    print(f"\n  tables -> {RESULTS / 'parsing_comparison.csv'}, "
          f"{RESULTS / 'parsing_chunks.csv'}\n")


if __name__ == "__main__":
    main()
