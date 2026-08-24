"""
Merge every result file in results/ into one table.

The reranker comparison was run in two halves to keep VRAM in check, and
the second half overwrote reranker_comparison.csv. The per-combination
JSON files survive intact, so this rebuilds the full picture from those
rather than from the CSVs.

    python collect_results.py

Writes results/all_results.csv and prints the combined ranking.
"""

import csv
import json
from pathlib import Path

RESULTS = Path("results")


def load_all() -> list:
    rows = []

    # Stage 1: embedders alone, dense.
    for path in sorted(RESULTS.glob("eval_*_dense.json")):
        try:
            agg = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
        except Exception:
            continue
        rows.append({
            "stage":    "embedder",
            "embedder": agg.get("tag", path.stem),
            "reranker": "none",
            "mode":     agg.get("mode", "dense"),
            "dim":      agg.get("dim", ""),
            "recall":   agg.get("hit@10", ""),
            "hit@1":    agg["hit@1"], "hit@3": agg["hit@3"],
            "hit@5":    agg["hit@5"], "hit@10": agg["hit@10"],
            "mrr":      agg["mrr"],
            "seconds":  agg.get("eval_s", agg.get("seconds", "")),
            "build_s":  agg.get("build_s", ""),
        })

    # Stage 2: embedder x reranker, hybrid.
    for path in sorted(RESULTS.glob("rerank_*.json")):
        try:
            agg = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
        except Exception:
            continue
        rows.append({
            "stage":    "reranker",
            "embedder": agg.get("embedder", ""),
            "reranker": agg.get("reranker", ""),
            "mode":     agg.get("mode", "hybrid"),
            "dim":      agg.get("dim", ""),
            "recall":   agg.get("recall", ""),
            "hit@1":    agg["hit@1"], "hit@3": agg["hit@3"],
            "hit@5":    agg["hit@5"], "hit@10": agg["hit@10"],
            "mrr":      agg["mrr"],
            "seconds":  agg.get("seconds", ""),
            "build_s":  "",
        })
    return rows


def show(rows: list, stage: str, title: str) -> None:
    sel = sorted([r for r in rows if r["stage"] == stage],
                 key=lambda r: -r["mrr"])
    if not sel:
        return
    print(f"\n{title}")
    print(f"  {'embedder':<15}{'reranker':<15}{'H@1':>6}{'H@3':>6}"
          f"{'H@5':>6}{'H@10':>7}{'MRR':>8}{'sec':>7}")
    print("  " + "-" * 70)
    for r in sel:
        secs = f"{float(r['seconds']):.1f}" if r["seconds"] != "" else "-"
        print(f"  {r['embedder']:<15}{r['reranker']:<15}"
              f"{r['hit@1']*100:5.0f}%{r['hit@3']*100:5.0f}%"
              f"{r['hit@5']*100:5.0f}%{r['hit@10']*100:6.0f}%"
              f"{r['mrr']:8.3f}{secs:>7}")


def noise_floor(rows: list) -> None:
    """
    The same configuration was measured more than once. The spread
    between those repeats is the smallest difference worth believing:
    ChromaDB's index is approximate and fp16 rounds, so two runs of one
    setup do not return identical numbers.
    """
    seen = {}
    for r in rows:
        if r["stage"] != "reranker":
            continue
        key = (r["embedder"], r["reranker"])
        seen.setdefault(key, []).append(r["mrr"])

    repeats = {k: v for k, v in seen.items() if len(v) > 1}
    if not repeats:
        return
    spread = max(max(v) - min(v) for v in repeats.values())
    print(f"\n  Noise floor")
    print(f"  " + "-" * 46)
    for (emb, rr), vals in repeats.items():
        if max(vals) - min(vals) > 0:
            print(f"  {emb} + {rr}: " +
                  ", ".join(f"{v:.3f}" for v in sorted(vals)))
    print(f"\n  Largest spread between repeats of one configuration: "
          f"{spread:.3f} MRR.")
    print(f"  Differences smaller than this are not real.")


def main():
    rows = load_all()
    if not rows:
        raise SystemExit("No result files found in results/.")

    RESULTS.mkdir(exist_ok=True)
    cols = ["stage", "embedder", "reranker", "mode", "dim", "recall",
            "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "seconds", "build_s"]
    path = RESULTS / "all_results.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    print("=" * 78)
    print(f"ALL RESULTS  ({len(rows)} runs)")
    print("=" * 78)
    show(rows, "embedder", "STAGE 1 - embedders alone, dense mode")
    show(rows, "reranker", "STAGE 2 - embedder x reranker, hybrid mode")
    noise_floor(rows)
    print(f"\n  merged table -> {path}\n")


if __name__ == "__main__":
    main()
