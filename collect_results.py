"""
Merge every result file in results/ into one table.

The reranker comparison was run in two halves to keep VRAM in check, and
the second half overwrote reranker_comparison.csv. The per-combination
JSON files survive intact, so this rebuilds the full picture from those
rather than from the CSVs.

    python collect_results.py

Writes results/all_results.csv and results/intervals.csv, and prints the
combined ranking with the uncertainty on the differences that matter.
"""

import csv
import json
import random
from pathlib import Path

RESULTS = Path("results")

# The embedder and reranker the satellite ships with. The intervals are
# reported around this configuration.
DEPLOYED          = "minilm-l6"
DEPLOYED_RERANKER = "ms-marco-L-6"


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
            "pool":     agg.get("pool", ""),
            "recall":   agg.get("recall", agg.get("hit@10", "")),
            "hit@1":    agg["hit@1"], "hit@3": agg["hit@3"],
            "hit@5":    agg["hit@5"], "hit@10": agg["hit@10"],
            "mrr":      agg["mrr"],
            "mrr_strict": agg.get("mrr_strict", ""),
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
            "pool":     agg.get("pool", ""),
            "recall":   agg.get("recall", ""),
            "hit@1":    agg["hit@1"], "hit@3": agg["hit@3"],
            "hit@5":    agg["hit@5"], "hit@10": agg["hit@10"],
            "mrr":      agg["mrr"],
            "mrr_strict": agg.get("mrr_strict", ""),
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
          f"{'H@5':>6}{'H@10':>7}{'found':>7}{'MRR':>8}{'strict':>8}{'sec':>7}")
    print("  " + "-" * 85)
    for r in sel:
        secs   = f"{float(r['seconds']):.1f}" if r["seconds"] != "" else "-"
        found  = f"{r['recall']*100:.0f}%" if r["recall"] != "" else "-"
        strict = f"{r['mrr_strict']:.3f}" if r["mrr_strict"] != "" else "-"
        print(f"  {r['embedder']:<15}{r['reranker']:<15}"
              f"{r['hit@1']*100:5.0f}%{r['hit@3']*100:5.0f}%"
              f"{r['hit@5']*100:5.0f}%{r['hit@10']*100:6.0f}%"
              f"{found:>7}{r['mrr']:8.3f}{strict:>8}{secs:>7}")


def reciprocal_ranks(name: str) -> dict:
    """{question id: reciprocal rank} from one per-question file."""
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))["per_question"]
    return {r["id"]: r["rr"] for r in rows}


def interval(a: dict, b: dict, draws: int = 10000, seed: int = 7) -> tuple:
    """
    MRR(a) - MRR(b) with a 95% interval, by paired bootstrap.

    It is the questions that are resampled, not the runs. Retrieval here
    is deterministic: a second run of one configuration returns the same
    rank for every question, so there is no run-to-run noise to measure.
    What is uncertain is which hundred questions happened to be asked,
    and with only a hundred that is worth several hundredths of MRR.
    """
    ids  = [q for q in a if q in b]
    diff = [a[q] - b[q] for q in ids]
    n    = len(diff)
    rng  = random.Random(seed)
    means = sorted(sum(rng.choice(diff) for _ in range(n)) / n
                   for _ in range(draws))
    return (sum(diff) / n, means[int(0.025 * draws)],
            means[int(0.975 * draws)])


def uncertainty(rows: list) -> list:
    """
    The differences the write-up leans on, each with its interval.

    A difference whose interval spans zero has not been shown on this
    test set, whatever the two point estimates look like side by side.
    """
    stage1 = sorted((r for r in rows if r["stage"] == "embedder"),
                    key=lambda r: -r["mrr"])
    stage2 = [r for r in rows if r["stage"] == "reranker"]
    if not stage1 or not stage2:
        return []

    none  = f"rerank_{DEPLOYED}_none"
    pairs = [(f"{DEPLOYED}: hybrid over dense",
              none, f"eval_{DEPLOYED}_dense")]
    for key in sorted({r["reranker"] for r in stage2
                       if r["embedder"] == DEPLOYED and r["reranker"] != "none"}):
        pairs.append((f"{DEPLOYED}: {key} over no reranker",
                      f"rerank_{DEPLOYED}_{key}", none))

    best = max(stage2, key=lambda r: r["mrr"])
    pairs.append((f"best pair ({best['embedder']} + {best['reranker']}) "
                  f"over {DEPLOYED} + {DEPLOYED_RERANKER}",
                  f"rerank_{best['embedder']}_{best['reranker']}",
                  f"rerank_{DEPLOYED}_{DEPLOYED_RERANKER}"))

    top, last = stage1[0]["embedder"], stage1[-1]["embedder"]
    for a, b in ((top, DEPLOYED), (DEPLOYED, "bge-m3"), (top, last)):
        if a != b:
            pairs.append((f"dense: {a} over {b}",
                          f"eval_{a}_dense", f"eval_{b}_dense"))

    out = []
    for label, a, b in pairs:
        ra, rb = reciprocal_ranks(a), reciprocal_ranks(b)
        if not ra or not rb:
            continue
        mean, low, high = interval(ra, rb)
        out.append({"comparison": label, "difference": round(mean, 4),
                    "low": round(low, 4), "high": round(high, 4),
                    "excludes_zero": "yes" if low > 0 or high < 0 else "no"})

    if out:
        width = max(len(r["comparison"]) for r in out)
        print(f"\n  Differences in MRR, 95% interval over the questions")
        print("  " + "-" * (width + 34))
        for r in out:
            mark = "" if r["excludes_zero"] == "yes" else "   spans zero"
            print(f"  {r['comparison']:<{width}}  {r['difference']:+.3f}  "
                  f"[{r['low']:+.3f}, {r['high']:+.3f}]{mark}")
        print(f"\n  An interval that spans zero is a difference this test "
              f"set cannot tell from none.")
    return out


def main():
    rows = load_all()
    if not rows:
        raise SystemExit("No result files found in results/.")

    RESULTS.mkdir(exist_ok=True)
    cols = ["stage", "embedder", "reranker", "mode", "dim", "pool", "recall",
            "hit@1", "hit@3", "hit@5", "hit@10", "mrr", "mrr_strict",
            "seconds", "build_s"]
    path = RESULTS / "all_results.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    print("=" * 88)
    print(f"ALL RESULTS  ({len(rows)} runs)")
    print("=" * 88)
    show(rows, "embedder", "STAGE 1 - embedders alone, dense mode")
    show(rows, "reranker", "STAGE 2 - embedder x reranker, hybrid mode")

    table = uncertainty(rows)
    if table:
        ipath = RESULTS / "intervals.csv"
        with ipath.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader(); w.writerows(table)
        print(f"  intervals    -> {ipath}")
    print(f"\n  merged table -> {path}\n")


if __name__ == "__main__":
    main()
