"""
Retrieval evaluation for the public ISRO demo dataset (testset_public_demo.py),
as opposed to evaluate.py, which is hardwired to testset.py and the
confidential real Aditya-L1 corpus (FCP/TCH document types this public
dataset does not have).

Reuses evaluate.py's scoring/reporting functions as-is; only the question
set and satellite default differ.

    python evaluate_public_demo.py
    python evaluate_public_demo.py --compare
    python evaluate_public_demo.py --negatives
"""

import argparse

from core import registry, retrieve
from evaluate import aggregate, label, report, rescue_view, run, set_rerank
from testset_public_demo import NEGATIVE_CASES, TEST_SET


def check_negatives(satellite, pool: int = 10) -> None:
    """
    Out-of-scope questions for this dataset. There is no right chunk, so
    the measure is whether the system declines rather than what it
    retrieves. Local copy of evaluate.check_negatives() because that one
    reads testset.py's module-level NEGATIVE_CASES, not this file's.
    """
    print(f"\n  Negative cases (no correct answer exists)")
    print("  " + "-" * 76)
    for case in NEGATIVE_CASES:
        hits = retrieve.search([satellite], case["q"], top_k=pool)
        top = label(hits[0]) if hits else "(nothing retrieved)"
        print(f"  {case['id']:<14} retrieved: {top[:56]}")
    print(f"  These should produce a not-found answer at generation time.\n")


def main():
    ap = argparse.ArgumentParser(description="Retrieval evaluation, public ISRO demo dataset")
    ap.add_argument("--satellite", default="aditya-l1")
    ap.add_argument("--pool", type=int, default=10)
    ap.add_argument("--rerank", choices=["on", "off"], default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--negatives", action="store_true")
    args = ap.parse_args()

    sat = registry.load_satellite(args.satellite)
    if not sat.is_ingested():
        raise SystemExit(f"{sat.slug} has no vector store. "
                         f"Run: python cli.py ingest {sat.slug}")

    print(f"\n  satellite : {sat.slug}")
    print(f"  embedder  : {sat.embedding_model}")
    print(f"  questions : {len(TEST_SET)}")

    if args.compare:
        results = {}
        for mode in ("off", "on"):
            set_rerank(sat, mode == "on")
            rows = run(sat, pool=args.pool, cases=TEST_SET)
            agg = aggregate(rows)
            results[mode] = (rows, agg)
            report(rows, agg, f"Reranker {mode.upper()}")
        (off_rows, _), (on_rows, _) = results["off"], results["on"]
        rescue_view(off_rows, on_rows)
    else:
        if args.rerank:
            set_rerank(sat, args.rerank == "on")
        rows = run(sat, pool=args.pool, cases=TEST_SET)
        agg = aggregate(rows)
        report(rows, agg, f"Retrieval evaluation (public demo) - {sat.name}")

    if args.negatives:
        check_negatives(sat, pool=args.pool)


if __name__ == "__main__":
    main()
