"""
Compare rerankers, on top of one or more embedding models.

Stage 2 of the experiment. Stage 1 (compare_models.py) ranks the
embedders with no reranking; this takes the best of those and crosses
them against several cross-encoders.

    python compare_models.py --models ...        stage 1, pick the winners
    python compare_rerankers.py --top 3          stage 2, on stage 1's top 3
    python compare_rerankers.py --embedders minilm-l6 bge-base-en
    python compare_rerankers.py --rerankers none ms-marco-L-6 bge-base

The candidate pool is retrieved ONCE per embedder and every reranker is
applied to that same cached pool. Retrieval is the slow part, so this
makes the comparison both fair and much faster than re-running the
pipeline per pair: every reranker is judged on identical candidates.

Recall is fixed by the pool and printed per embedder. It is the ceiling:
no reranker can place a chunk that retrieval never returned.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from core import embeddings, ingest, registry, retrieve
from core import rerank as reranker_mod
from evaluate import aggregate, first_rank, label
from testset import TEST_SET

RESULTS = Path("results")

# Cross-encoders to try. "none" is the unranked baseline and must stay
# first so every table has its control.
RERANKERS = {
    "none":          None,
    "ms-marco-L-4":  "cross-encoder/ms-marco-MiniLM-L-4-v2",
    "ms-marco-L-6":  "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-L-12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-base":      "BAAI/bge-reranker-base",
    "mxbai-xsmall":  "mixedbread-ai/mxbai-rerank-xsmall-v1",
    # Heavier. Add explicitly with --rerankers if you want them; they cost
    # minutes per run rather than seconds.
    "bge-large":     "BAAI/bge-reranker-large",
    "bge-v2-m3":     "BAAI/bge-reranker-v2-m3",
}

DEFAULT_RERANKERS = ["none", "ms-marco-L-4", "ms-marco-L-6",
                     "ms-marco-L-12", "bge-base", "mxbai-xsmall"]

POOL = 20      # candidates retrieved per question, and reranked


def retrieve_pool(satellite, pool: int, mode: str) -> list:
    """
    Retrieve once for every question, with reranking off.
    Returns [(case, hits), ...] to be reordered by each reranker.
    """
    cfg = dict(satellite.config.get("retrieval") or {})
    cfg["rerank"] = False
    satellite.config["retrieval"] = cfg
    retrieve._clients.clear()

    cases = []
    for case in TEST_SET:
        hits = retrieve.search([satellite], case["q"], top_k=pool, mode=mode)
        cases.append((case, hits))
    return cases


def score_ranked(cases: list, reorder=None) -> tuple:
    rows = []
    for case, hits in cases:
        ordered = reorder(case["q"], hits) if reorder else hits
        rank = first_rank(ordered, case["accept"]) if case["accept"] else 0
        rows.append({
            "id":     case["id"],
            "rank":   rank,
            "hit@1":  int(rank == 1),
            "hit@3":  int(1 <= rank <= 3),
            "hit@5":  int(1 <= rank <= 5),
            "hit@10": int(rank >= 1),
            "rr":     (1.0 / rank) if rank else 0.0,
            "top":    label(ordered[0]) if ordered else "",
        })
    return rows, aggregate(rows)


def make_reorder(hf_id: str):
    """A callable that reranks one question's hits, or None if unavailable."""
    model = reranker_mod.load(hf_id)
    if model is None:
        return None

    def reorder(query, hits):
        if len(hits) < 2:
            return hits
        pairs  = [(query, h.text[:2000]) for h in hits]
        scores = model.predict(pairs, show_progress_bar=False)
        paired = sorted(zip(hits, scores), key=lambda z: -float(z[1]))
        return [h for h, _ in paired]

    return reorder


def run_embedder(satellite, embedder: str, reranker_keys: list,
                 pool: int, mode: str, build: bool) -> list:
    tag = embedder.split("/")[-1]
    print(f"\n{'=' * 70}")
    print(f"EMBEDDER: {tag}")
    print("=" * 70)

    satellite.model_override = embedder
    satellite.store_suffix   = tag
    retrieve._clients.clear()

    if build and not satellite.is_ingested():
        print(f"  building {satellite.vectorstore_dir.name}/ ...")
        try:
            if ingest.ingest(satellite, reset=True).get("status") != "ok":
                print("  BUILD FAILED, skipping"); return []
        except Exception as exc:
            print(f"  BUILD FAILED: {type(exc).__name__}: {str(exc)[:110]}")
            return []
    elif not satellite.is_ingested():
        print(f"  no {satellite.vectorstore_dir.name}/ yet. "
              f"Run compare_models.py first, or drop --eval-only.")
        return []

    print(f"  retrieving pool of {pool} once for {len(TEST_SET)} questions ...")
    start = time.time()
    cases = retrieve_pool(satellite, pool, mode)
    retrieve_s = round(time.time() - start, 1)

    ceiling = sum(1 for c, h in cases
                  if c["accept"] and first_rank(h, c["accept"]) >= 1) / len(cases)
    print(f"  done in {retrieve_s}s | recall ceiling in top-{pool} = "
          f"{ceiling*100:.0f}%")
    print(f"\n  {'reranker':<16}{'H@1':>7}{'H@3':>7}{'H@5':>7}{'H@10':>7}"
          f"{'MRR':>8}{'sec':>8}")
    print("  " + "-" * 60)

    out = []
    for key in reranker_keys:
        hf_id = RERANKERS.get(key, key if "/" in str(key) else None)

        if key == "none":
            reorder = None
        else:
            if not hf_id:
                print(f"  {key:<16}  unknown reranker, skipping")
                continue
            reorder = make_reorder(hf_id)
            if reorder is None:
                print(f"  {key:<16}  could not load, skipping")
                continue

        start = time.time()
        try:
            rows, agg = score_ranked(cases, reorder)
        except Exception as exc:
            print(f"  {key:<16}  FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:70]}")
            continue
        secs = round(time.time() - start, 1)

        agg.update({"embedder": tag, "reranker": key, "mode": mode,
                    "seconds": secs, "recall": ceiling,
                    "dim": embeddings.spec_for(embedder).dim})
        out.append(agg)

        print(f"  {key:<16}{agg['hit@1']*100:6.0f}%{agg['hit@3']*100:6.0f}%"
              f"{agg['hit@5']*100:6.0f}%{agg['hit@10']*100:6.0f}%"
              f"{agg['mrr']:8.3f}{secs:8.1f}")

        RESULTS.mkdir(exist_ok=True)
        # A key given as a raw HuggingFace id contains a slash, which
        # would be read as a folder that does not exist.
        safe = str(key).replace("/", "__")
        (RESULTS / f"rerank_{tag}_{safe}.json").write_text(
            json.dumps({"aggregate": agg, "per_question": rows},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    if out:
        best = max(out, key=lambda a: a["mrr"])
        base = next((a for a in out if a["reranker"] == "none"), None)
        line = f"  -> best: {best['reranker']}  MRR {best['mrr']:.3f}"
        if base and best["reranker"] != "none":
            line += f"  (baseline {base['mrr']:.3f}, {best['mrr']-base['mrr']:+.3f})"
        print(line)

    return out


def top_from_stage1(n: int) -> list:
    """Read the best embedders out of compare_models.py's output."""
    path = RESULTS / "model_comparison.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run compare_models.py first, or name the "
            f"embedders with --embedders.")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows = [r for r in rows if r.get("mode") == "dense"] or rows
    rows.sort(key=lambda r: float(r["mrr"]), reverse=True)
    seen, out = set(), []
    for r in rows:
        if r["model"] not in seen:
            seen.add(r["model"])
            out.append(r["model"])
        if len(out) >= n:
            break
    return out


def summary(results: list) -> None:
    if not results:
        return
    RESULTS.mkdir(exist_ok=True)
    cols = ["embedder", "dim", "reranker", "mode", "recall",
            "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "seconds"]
    path = RESULTS / "reranker_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(results)

    print(f"\n{'=' * 70}")
    print("ALL COMBINATIONS, ranked by MRR")
    print("=" * 70)
    print(f"  {'embedder':<16}{'reranker':<16}{'H@1':>7}{'H@5':>7}"
          f"{'MRR':>8}{'sec':>8}")
    print("  " + "-" * 62)
    for a in sorted(results, key=lambda a: a["mrr"], reverse=True):
        print(f"  {a['embedder']:<16}{a['reranker']:<16}"
              f"{a['hit@1']*100:6.0f}%{a['hit@5']*100:6.0f}%"
              f"{a['mrr']:8.3f}{a['seconds']:8.1f}")

    best = max(results, key=lambda a: a["mrr"])
    best5 = max(results, key=lambda a: (a["hit@5"], a["mrr"]))
    print(f"\n  best MRR   : {best['embedder']} + {best['reranker']}  "
          f"{best['mrr']:.3f}")
    print(f"  best Hit@5 : {best5['embedder']} + {best5['reranker']}  "
          f"{best5['hit@5']*100:.0f}%")
    print(f"\n  Hit@5 is what the language model reads, MRR is what a person")
    print(f"  scanning the citation list sees. They need not agree.")
    print(f"\n  table -> {path}\n")


def main():
    ap = argparse.ArgumentParser(description="Compare rerankers per embedder")
    ap.add_argument("--satellite", default="aditya-l1")
    ap.add_argument("--embedders", nargs="*", default=None,
                    help="embedders to test (default: top from stage 1)")
    ap.add_argument("--top", type=int, default=3,
                    help="how many stage-1 winners to use")
    ap.add_argument("--rerankers", nargs="*", default=DEFAULT_RERANKERS,
                    help=f"choose from: {', '.join(RERANKERS)}")
    ap.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid",
                    help="hybrid is what ships and is the right mode here")
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--eval-only", action="store_true",
                    help="use existing per-model stores, do not build")
    args = ap.parse_args()

    embedders = args.embedders or top_from_stage1(args.top)

    sat = registry.load_satellite(args.satellite)
    if not sat.is_extracted():
        raise SystemExit(f"{sat.slug} has nothing extracted.")

    print(f"  satellite : {sat.slug}")
    print(f"  questions : {len(TEST_SET)}")
    print(f"  embedders : {', '.join(embedders)}")
    print(f"  rerankers : {', '.join(args.rerankers)}")
    print(f"  mode      : {args.mode}   pool: {args.pool}")

    results = []
    for embedder in embedders:
        results.extend(run_embedder(sat, embedder, args.rerankers,
                                    args.pool, args.mode,
                                    build=not args.eval_only))
    summary(results)


if __name__ == "__main__":
    main()
