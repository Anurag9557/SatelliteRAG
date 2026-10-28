"""
Compare embedding models on the labelled test set.

For each model it builds that model's own vector store, evaluates it, and
records the scores. Stores are kept in separate folders
(satellites/<slug>/vectorstore_<tag>/), so every model persists side by
side and the working store is never touched.

    python compare_models.py                        the shortlist, dense mode
    python compare_models.py --mode both            dense and hybrid
    python compare_models.py --models minilm-l6 bge-base-en gte-base
    python compare_models.py --models BAAI/bge-m3   any HuggingFace id works
    python compare_models.py --eval-only            score existing stores, no rebuild
    python compare_models.py --rerank               with the cross-encoder too

Use DENSE to compare embedders. Hybrid mixes in BM25, which does not
depend on the embedding model at all, so it flattens the difference
between one model and another. Report hybrid as well, since that is what
ships.

Building embeds the whole corpus once per model. On a GPU that is seconds
for a small model and a couple of minutes for a 1024-dimension one; on
CPU, considerably longer. Use --eval-only to re-score without rebuilding.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from core import embeddings, ingest, registry, retrieve
from evaluate import aggregate, first_rank, label
from testset import TEST_SET

RESULTS = Path("results")

# A reasonable first pass: the current default, three English models of
# different sizes, and the multilingual one for contrast.
SHORTLIST = ["minilm-l6", "bge-small-en", "bge-base-en", "gte-base",
             "mpnet-base", "bge-m3"]


def score(satellite, pool: int, mode: str) -> tuple:
    rows = []
    for case in TEST_SET:
        hits = retrieve.search([satellite], case["q"], top_k=pool, mode=mode)
        rank = first_rank(hits, case["accept"]) if case["accept"] else 0
        rows.append({
            "id":     case["id"],
            "rank":   rank,
            "hit@1":  int(rank == 1),
            "hit@3":  int(1 <= rank <= 3),
            "hit@5":  int(1 <= rank <= 5),
            "hit@10": int(1 <= rank <= 10),
            "rr":     (1.0 / rank) if rank else 0.0,
            "top":    label(hits[0]) if hits else "",
        })
    return rows, aggregate(rows)


def run_model(satellite, name: str, modes: list, pool: int,
              rerank: bool, eval_only: bool) -> list:
    tag = name.split("/")[-1]
    print(f"\n=== {tag}  ({embeddings.spec_for(name).hf_id}) ===")

    satellite.model_override = name
    satellite.store_suffix   = tag
    retrieve._clients.clear()

    cfg = dict(satellite.config.get("retrieval") or {})
    cfg["rerank"] = rerank
    satellite.config["retrieval"] = cfg

    build_s = 0.0
    if eval_only:
        if not satellite.is_ingested():
            print(f"  no {satellite.vectorstore_dir.name}/ yet, "
                  f"build it once without --eval-only. Skipping.")
            return []
        print(f"  [eval-only] scoring existing {satellite.vectorstore_dir.name}/")
    else:
        start = time.time()
        try:
            result = ingest.ingest(satellite, reset=True)
            if result.get("status") != "ok":
                print(f"  BUILD FAILED ({result.get('status')}), skipping")
                return []
        except Exception as exc:
            print(f"  BUILD FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            return []
        build_s = round(time.time() - start, 1)

    out = []
    for mode in modes:
        start = time.time()
        try:
            rows, agg = score(satellite, pool, mode)
        except Exception as exc:
            print(f"  EVAL ({mode}) FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:120]}")
            continue
        agg.update({
            "tag": tag, "model": name, "mode": mode,
            "rerank": rerank, "build_s": build_s,
            "eval_s": round(time.time() - start, 1),
            "dim": embeddings.spec_for(name).dim,
        })
        print(f"  [{mode}]  Hit@1 {agg['hit@1']*100:.0f}%  "
              f"Hit@5 {agg['hit@5']*100:.0f}%  MRR {agg['mrr']:.3f}  "
              f"({agg['eval_s']}s)")

        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"eval_{tag}_{mode}.json").write_text(
            json.dumps({"aggregate": agg, "per_question": rows},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        out.append(agg)
    return out


def write_table(results: list) -> None:
    RESULTS.mkdir(exist_ok=True)
    cols = ["tag", "model", "dim", "mode", "rerank", "hit@1", "hit@3",
            "hit@5", "hit@10", "mrr", "build_s", "eval_s"]
    path = RESULTS / "model_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    for mode in sorted({a["mode"] for a in results}):
        rows = sorted([a for a in results if a["mode"] == mode],
                      key=lambda a: a["mrr"], reverse=True)
        print(f"\n[{mode}]  ranked by MRR")
        print(f"  {'model':<16}{'dim':>6}{'H@1':>7}{'H@3':>7}{'H@5':>7}"
              f"{'H@10':>7}{'MRR':>8}{'build':>8}")
        print("  " + "-" * 66)
        for a in rows:
            print(f"  {a['tag']:<16}{a['dim']:>6}"
                  f"{a['hit@1']*100:6.0f}%{a['hit@3']*100:6.0f}%"
                  f"{a['hit@5']*100:6.0f}%{a['hit@10']*100:6.0f}%"
                  f"{a['mrr']:8.3f}{a['build_s']:8.1f}")
        best = rows[0]
        print(f"  -> best ({mode}): {best['tag']}  MRR {best['mrr']:.3f}  "
              f"Hit@5 {best['hit@5']*100:.0f}%")

    print(f"\n  table -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Compare embedding models")
    ap.add_argument("--satellite", default="aditya-l1")
    ap.add_argument("--models", nargs="*", default=None,
                    help="short names or HuggingFace ids (default: shortlist)")
    ap.add_argument("--mode", choices=["dense", "hybrid", "both"],
                    default="dense",
                    help="dense isolates the embedder and is the fair test")
    ap.add_argument("--pool", type=int, default=10)
    ap.add_argument("--rerank", action="store_true",
                    help="also apply the cross-encoder")
    ap.add_argument("--eval-only", action="store_true",
                    help="score existing per-model stores without rebuilding")
    args = ap.parse_args()

    modes  = ["dense", "hybrid"] if args.mode == "both" else [args.mode]
    models = args.models or SHORTLIST

    sat = registry.load_satellite(args.satellite)
    if not sat.is_extracted():
        raise SystemExit(f"{sat.slug} has nothing extracted. "
                         f"Run: python cli.py extract {sat.slug}")

    print(f"  satellite : {sat.slug}")
    print(f"  questions : {len(TEST_SET)}")
    print(f"  models    : {', '.join(models)}")
    print(f"  modes     : {', '.join(modes)}")
    print(f"  reranker  : {'on' if args.rerank else 'off'}")

    results = []
    for name in models:
        results.extend(run_model(sat, name, modes, args.pool,
                                 args.rerank, args.eval_only))

    if not results:
        print("\n  No successful runs.")
        return

    write_table(results)
    print("\n  Each model's store persists in "
          "satellites/<slug>/vectorstore_<tag>/ .")
    print("  The working store at vectorstore/ was not touched.")
    print("  To adopt a model, set it in satellite.yaml and run:")
    print(f"    python cli.py ingest {sat.slug} --reset\n")


if __name__ == "__main__":
    main()
