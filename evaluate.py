"""
Retrieval evaluation.

Scores the labelled questions in testset.py against a satellite's store
and reports Hit@1/3/5/10 and MRR.

    python evaluate.py                          aditya-l1, current settings
    python evaluate.py --satellite chandrayaan-3
    python evaluate.py --rerank off             measure without the reranker
    python evaluate.py --router soft            try another router
    python evaluate.py --compare                both, side by side
    python evaluate.py --routers                rules against soft
    python evaluate.py --csv results/eval.csv

--compare is the one that answers "is the reranker worth it on my data".
It runs the same questions twice and prints the difference.

Negative cases are reported separately: they have no correct answer, so
including them in Hit@k would understate the system.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from core import registry, retrieve
from testset import BROAD, NEGATIVE_CASES, TEST_SET

RESULTS_DIR = Path("results")


def matches(hit, cond: dict) -> bool:
    """True if the hit satisfies every key in one accept condition."""
    meta = hit.metadata or {}
    text = (hit.text or "").lower()
    for key, val in cond.items():
        if key == "text_contains":
            if str(val).lower() not in text:
                return False
        elif key == "text_contains_any":
            if not any(str(v).lower() in text for v in val):
                return False
        else:
            if str(meta.get(key, "")).strip().lower() != str(val).strip().lower():
                return False
    return True


def relevant(hit, accept: list) -> bool:
    return any(matches(hit, c) for c in accept)


def first_rank(hits: list, accept: list) -> int:
    """1-based rank of the first relevant hit, 0 if none."""
    for i, hit in enumerate(hits, start=1):
        if relevant(hit, accept):
            return i
    return 0


def label(hit) -> str:
    m = hit.metadata or {}
    bits = [m.get("doc_type", "")]
    if m.get("fcp_number"):
        bits.append(f"FCP-{m['fcp_number']}")
    if m.get("page"):
        bits.append(f"p.{m['page']}")
    bits.append(m.get("content_type", ""))
    return " / ".join(b for b in bits if b)


def run(satellite, pool: int = 10, cases: list = None) -> list:
    rows = []
    for case in (cases or TEST_SET):
        hits = retrieve.search([satellite], case["q"], top_k=pool)
        rank = first_rank(hits, case["accept"]) if case["accept"] else 0
        rows.append({
            "id":       case["id"],
            "question": case["q"],
            "intent":   case.get("intent", ""),
            "rank":     rank,
            "hit@1":    int(rank == 1),
            "hit@3":    int(1 <= rank <= 3),
            "hit@5":    int(1 <= rank <= 5),
            # Ranks 1-10 only. Found anywhere in a larger pool is recall.
            "hit@10":   int(1 <= rank <= 10),
            "rr":       (1.0 / rank) if rank else 0.0,
            "top":      label(hits[0]) if hits else "",
            "note":     case.get("note", ""),
        })
    return rows


def aggregate(rows: list) -> dict:
    n = len(rows) or 1
    # The same MRR without the questions nearly any chunk satisfies
    # (testset.BROAD). If the two disagree, the headline is leaning on
    # the easy ones.
    strict = [r for r in rows if r["id"] not in BROAD]
    return {
        "n":      len(rows),
        "n_strict":   len(strict),
        "mrr_strict": sum(r["rr"] for r in strict) / (len(strict) or 1),
        "hit@1":  sum(r["hit@1"]  for r in rows) / n,
        "hit@3":  sum(r["hit@3"]  for r in rows) / n,
        "hit@5":  sum(r["hit@5"]  for r in rows) / n,
        "hit@10": sum(r["hit@10"] for r in rows) / n,
        "recall": sum(1 for r in rows if r["rank"] >= 1) / n,
        "mrr":    sum(r["rr"]     for r in rows) / n,
        "misses": [r["id"] for r in rows if r["rank"] == 0],
    }


def report(rows: list, agg: dict, title: str) -> None:
    print("=" * 92)
    print(title)
    print("=" * 92)
    print(f"{'id':<22} {'rank':<6} {'H@1':<5} {'H@3':<5} {'H@5':<5} top result")
    print("-" * 92)
    for r in rows:
        rank = r["rank"] or "-"
        flag = "" if r["rank"] and r["rank"] <= 5 else "   miss@5"
        print(f"{r['id']:<22} {str(rank):<6} {r['hit@1']:<5} {r['hit@3']:<5} "
              f"{r['hit@5']:<5} {r['top'][:38]}{flag}")
    print("-" * 92)
    print(f"  Questions  : {agg['n']}")
    print(f"  Hit@1      : {agg['hit@1']*100:5.1f}%")
    print(f"  Hit@3      : {agg['hit@3']*100:5.1f}%")
    print(f"  Hit@5      : {agg['hit@5']*100:5.1f}%")
    print(f"  Hit@10     : {agg['hit@10']*100:5.1f}%")
    print(f"  MRR        : {agg['mrr']:.3f}")
    if agg["n_strict"] != agg["n"]:
        print(f"  MRR strict : {agg['mrr_strict']:.3f}   "
              f"(the {agg['n_strict']} questions outside testset.BROAD)")
    # Recall is the ceiling: a chunk that never enters the pool cannot be
    # ranked into first place, however good the reranker is. Reporting it
    # separates "retrieval did not find it" from "ranking put it low".
    print(f"  Recall     : {agg['recall']*100:5.1f}%   "
          f"(found in the pool; the ceiling for any reranker)")
    if agg["misses"]:
        print(f"  Missed     : {', '.join(agg['misses'])}")
    by_intent(rows)
    print("=" * 92)


def by_intent(rows: list) -> None:
    """The same metrics, split by what each question asks for."""
    intents = sorted({r["intent"] for r in rows if r.get("intent")})
    if not intents:
        return
    print(f"\n  {'intent':<18}{'n':>4}{'H@1':>8}{'H@5':>8}{'MRR':>8}")
    for intent in intents:
        sel = [r for r in rows if r["intent"] == intent]
        agg = aggregate(sel)
        print(f"  {intent:<18}{len(sel):>4}{agg['hit@1']*100:7.0f}%"
              f"{agg['hit@5']*100:7.0f}%{agg['mrr']:8.3f}")


def rescue_view(off_rows: list, on_rows: list) -> None:
    """
    The questions the unranked pipeline missed at 5, and where reranking
    put them. This is the measure of what the reranker is actually for:
    rescuing what fusion ordered badly, not padding what fusion already
    got right.
    """
    off = {r["id"]: r["rank"] for r in off_rows}
    on  = {r["id"]: r["rank"] for r in on_rows}
    missed = [q for q, rank in off.items() if not (1 <= rank <= 5)]

    if not missed:
        print("\n  Rescue: the pipeline missed nothing at 5 without reranking.\n")
        return

    fixed = sum(1 for q in missed if 1 <= on.get(q, 0) <= 5)
    print(f"\n  Rescue: {len(missed)} question(s) missed at 5 without reranking, "
          f"{fixed} recovered")
    print("  " + "-" * 52)
    print(f"  {'id':<24}{'no rerank':>12}{'rerank':>12}")
    for q in missed:
        a = off.get(q) or "-"
        b = on.get(q) or "-"
        mark = "" if not (1 <= on.get(q, 0) <= 5) else "  fixed"
        print(f"  {q:<24}{str(a):>12}{str(b):>12}{mark}")

    # A reranker should not push a good result down.
    broken = [q for q, rank in off.items()
              if 1 <= rank <= 5 and not (1 <= on.get(q, 0) <= 5)]
    if broken:
        print(f"\n  Regressed: reranking pushed {len(broken)} out of the top 5")
        for q in broken:
            print(f"  {q:<24}{off[q]:>12}{on.get(q) or '-':>12}")
    print()


def check_negatives(satellite, pool: int = 10) -> None:
    """
    Out-of-scope questions. There is no right chunk, so the measure is
    whether the system declines rather than what it retrieves.
    """
    print(f"\n  Negative cases (no correct answer exists)")
    print("  " + "-" * 76)
    for case in NEGATIVE_CASES:
        hits = retrieve.search([satellite], case["q"], top_k=3)
        top  = label(hits[0]) if hits else "(nothing retrieved)"
        print(f"  {case['id']:<14} retrieved: {top[:56]}")
    print(f"  These should produce a not-found answer at generation time.\n")


def set_rerank(satellite, on: bool) -> None:
    cfg = dict(satellite.config.get("retrieval") or {})
    cfg["rerank"] = on
    satellite.config["retrieval"] = cfg
    retrieve._clients.clear()


def set_router(satellite, mode: str) -> None:
    cfg = dict(satellite.config.get("retrieval") or {})
    cfg["router"] = mode
    satellite.config["retrieval"] = cfg


def widen(satellite, n: int) -> None:
    """
    Widen the candidate pool the reranker sees.

    Recall sets the ceiling: a chunk that never reaches the pool cannot be
    reranked into first place. Hit@10 is the honest measure of what
    retrieval found at all.
    """
    cfg = dict(satellite.config.get("retrieval") or {})
    cfg["n_dense"]     = n
    cfg["n_bm25"]      = n
    cfg["rerank_pool"] = n
    satellite.config["retrieval"] = cfg
    retrieve._clients.clear()


def compare_routers(satellite, pool: int) -> None:
    """
    Rules and soft, each with and without the reranker, on the same store.

    The model router is left out: it needs the text model running and is
    scored for accuracy by evaluate_router.py. Writes
    results/router_retrieval.csv, which is what the default router in
    core/registry.py rests on.
    """
    table = []
    print(f"\n  {'router':<8}{'rerank':<8}{'H@1':>6}{'H@3':>6}{'H@5':>6}"
          f"{'H@10':>7}{'MRR':>8}{'strict':>8}")
    print("  " + "-" * 56)
    for mode in ("rules", "soft"):
        set_router(satellite, mode)
        for on in (False, True):
            set_rerank(satellite, on)
            agg = aggregate(run(satellite, pool=pool))
            table.append({"router": mode, "rerank": "on" if on else "off",
                          "pool": pool,
                          **{k: round(agg[k], 4) for k in (
                              "hit@1", "hit@3", "hit@5", "hit@10",
                              "recall", "mrr", "mrr_strict")}})
            print(f"  {mode:<8}{'on' if on else 'off':<8}"
                  f"{agg['hit@1']*100:5.0f}%{agg['hit@3']*100:5.0f}%"
                  f"{agg['hit@5']*100:5.0f}%{agg['hit@10']*100:6.0f}%"
                  f"{agg['mrr']:8.3f}{agg['mrr_strict']:8.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "router_retrieval.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\n  table -> {path}\n")


def main():
    ap = argparse.ArgumentParser(description="Retrieval evaluation")
    ap.add_argument("--satellite", default="aditya-l1")
    ap.add_argument("--pool", type=int, default=10,
                    help="candidates considered per question")
    ap.add_argument("--rerank", choices=["on", "off"], default=None,
                    help="override the satellite's rerank setting")
    ap.add_argument("--compare", action="store_true",
                    help="run with and without the reranker, side by side")
    ap.add_argument("--router", choices=["rules", "soft", "llm"],
                    default=None,
                    help="override the satellite's router setting")
    ap.add_argument("--routers", action="store_true",
                    help="rules against soft, reranker off and on")
    ap.add_argument("--csv", default=None, help="write per-question rows here")
    ap.add_argument("--json", default=None, help="write the summary here")
    ap.add_argument("--negatives", action="store_true",
                    help="also show the out-of-scope cases")
    ap.add_argument("--widen", type=int, default=None,
                    help="candidate pool per pass (default 20); raising it "
                         "lifts the ceiling the reranker works within")
    args = ap.parse_args()

    sat = registry.load_satellite(args.satellite)
    if not sat.is_ingested():
        raise SystemExit(f"{sat.slug} has no vector store. "
                         f"Run: python cli.py ingest {sat.slug}")

    if args.widen:
        widen(sat, args.widen)
    if args.routers:
        compare_routers(sat, args.pool)
        return
    if args.router:
        set_router(sat, args.router)

    print(f"\n  satellite : {sat.slug}")
    print(f"  embedder  : {sat.embedding_model}")
    print(f"  router    : {sat.retrieval['router']}")
    print(f"  questions : {len(TEST_SET)}")
    print(f"  pool      : {sat.retrieval['n_dense']} per pass, "
          f"{args.pool} scored")

    if args.compare:
        results = {}
        for mode in ("off", "on"):
            set_rerank(sat, mode == "on")
            start = time.time()
            rows  = run(sat, pool=args.pool)
            agg   = aggregate(rows)
            agg["seconds"] = round(time.time() - start, 1)
            results[mode] = (rows, agg)
            report(rows, agg, f"Reranker {mode.upper()}")

        (off_rows, off), (on_rows, on) = results["off"], results["on"]
        rescue_view(off_rows, on_rows)
        print("  Comparison")
        print("  " + "-" * 62)
        print(f"  {'metric':<12} {'no rerank':>12} {'rerank':>12} {'change':>12}")
        for key in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr"):
            a, b = off[key], on[key]
            arrow = "+" if b > a else ""
            print(f"  {key:<12} {a:>12.3f} {b:>12.3f} {arrow}{b - a:>11.3f}")
        print(f"  {'seconds':<12} {off['seconds']:>12} {on['seconds']:>12}")
        print()
        rows, agg = on_rows, on
    else:
        if args.rerank:
            set_rerank(sat, args.rerank == "on")
        start = time.time()
        rows  = run(sat, pool=args.pool)
        agg   = aggregate(rows)
        agg["seconds"] = round(time.time() - start, 1)
        report(rows, agg, f"Retrieval evaluation - {sat.name}")

    if args.negatives:
        check_negatives(sat, pool=args.pool)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  per-question CSV -> {path}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"satellite": sat.slug, "embedder": sat.embedding_model,
             "router": sat.retrieval["router"],
             "rerank": sat.retrieval.get("rerank", True),
             "pool": args.pool, "summary": agg, "rows": rows},
            indent=2), encoding="utf-8")
        print(f"  summary JSON     -> {path}")


if __name__ == "__main__":
    main()
