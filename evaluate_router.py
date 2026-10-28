"""
Routing accuracy.

Scores the routers on what each question asks for. Nothing is retrieved,
so the rules finish in a moment and the model router in a few minutes.

    python evaluate_router.py                     rules and llm
    python evaluate_router.py --routers rules
    python evaluate_router.py --model qwen2.5:7b-instruct

Two sets are scored separately:

    benchmark   the 100 questions in testset.py, labelled in INTENT_LABELS
    routing     testset_router.py: paraphrases and known failure cases

The store follows from the intent (command_history reads the log,
everything else the documents), and store accuracy is reported on its
own: a question sent to the wrong store is the one mistake ranking can
never repair.

The soft router makes the same decisions as the rules and differs only in
how the procedure document is searched, so it is not scored here.

Writes results/router_accuracy.csv (metrics only). Per-question decisions
go to runs/router/, which is kept out of git.
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from core import router
from testset import TEST_SET
from testset_router import ROUTER_CASES

RESULTS = Path("results")
RUNS    = Path("runs") / "router"

SHORT = {"procedure": "proc", "which_procedure": "which", "figure": "fig",
         "command_history": "cmds", "concept": "concept"}


def uses_log(intent: str) -> bool:
    return intent == "command_history"


def score(cases: list, mode: str, model: str) -> list:
    cfg  = {"router": mode, "router_model": model}
    rows = []
    for case in cases:
        start = time.perf_counter()
        plan  = router.plan(case["q"], cfg)
        ms    = (time.perf_counter() - start) * 1000
        rows.append({
            "id":        case["id"],
            "gold":      case["intent"],
            "predicted": plan.intent,
            "correct":   int(plan.intent == case["intent"]),
            "store_ok":  int(("commands" in plan.stores)
                             == uses_log(case["intent"])),
            "source":    plan.source,
            "note":      plan.note,
            "ms":        round(ms, 1),
        })
    return rows


def summarise(rows: list) -> dict:
    n   = len(rows) or 1
    out = {
        "n":         len(rows),
        "accuracy":  sum(r["correct"] for r in rows) / n,
        "store_acc": sum(r["store_ok"] for r in rows) / n,
        "fallbacks": sum(1 for r in rows
                         if r["note"].startswith("model unavailable")),
        "mean_ms":   sum(r["ms"] for r in rows) / n,
    }
    for intent in router.INTENTS:
        gold = [r for r in rows if r["gold"] == intent]
        pred = [r for r in rows if r["predicted"] == intent]
        hits = sum(r["correct"] for r in gold)
        out[f"recall_{SHORT[intent]}"]    = hits / len(gold) if gold else ""
        out[f"precision_{SHORT[intent]}"] = hits / len(pred) if pred else ""
    return out


def show(rows: list, summ: dict, title: str) -> None:
    print(f"\n{title}")
    print("-" * 72)
    print(f"  intent accuracy {summ['accuracy']*100:5.1f}%   "
          f"store accuracy {summ['store_acc']*100:5.1f}%   "
          f"{summ['mean_ms']:.0f} ms per question")
    if summ["fallbacks"]:
        print(f"  model unavailable for {summ['fallbacks']} question(s); "
              f"the rules answered instead")

    counts = Counter((r["gold"], r["predicted"]) for r in rows)
    print(f"\n  {'gold / predicted':<18}"
          + "".join(f"{SHORT[i]:>9}" for i in router.INTENTS))
    for gold in router.INTENTS:
        if any(r["gold"] == gold for r in rows):
            print(f"  {SHORT[gold]:<18}"
                  + "".join(f"{counts[(gold, p)]:>9}" for p in router.INTENTS))

    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        print(f"\n  {len(wrong)} wrong:")
        for r in wrong:
            print(f"    {r['id']:<22}{r['gold']:<17} -> {r['predicted']}")


def main():
    ap = argparse.ArgumentParser(description="Routing accuracy")
    ap.add_argument("--routers", nargs="+", default=["rules", "llm"],
                    choices=["rules", "llm"])
    ap.add_argument("--model", default=router.LLM_MODEL,
                    help="Ollama model for the llm router")
    args = ap.parse_args()

    sets = {"benchmark": TEST_SET, "routing": ROUTER_CASES}
    RUNS.mkdir(parents=True, exist_ok=True)

    table = []
    for mode in args.routers:
        for name, cases in sets.items():
            rows = score(cases, mode, args.model)
            summ = summarise(rows)
            show(rows, summ, f"{mode.upper()} router, {name} set "
                             f"({len(cases)} questions)")
            table.append({"set": name, "router": mode,
                          "model": args.model if mode == "llm" else "",
                          **summ})
            (RUNS / f"router_{mode}_{name}.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "router_accuracy.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\n  summary -> {path}\n")


if __name__ == "__main__":
    main()
