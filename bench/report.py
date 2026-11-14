"""
Turn the scenario JSONs into one table.

    python bench/report.py
    python bench/report.py --profile matched-fp16

Writes bench/results/comparison.csv and bench/results/comparison.md. The
markdown is the one to paste into the deck or the write-up.

Medians and p95 throughout, never means. The first request to either server
pays for a cold cache and would drag a mean into nonsense.
"""

import argparse
import csv
import json
from pathlib import Path

import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

# BENCH_RESULT_DIR points the report at another run's folder (the 7B run
# writes to results-7b), the same variable bench/run.py writes with.
RESULT_DIR = Path(os.environ.get("BENCH_RESULT_DIR",
                                 Path(__file__).resolve().parent / "results"))

NOT_FOUND = "Not found in available documents."


def refusals(result: dict) -> tuple:
    """(refused, tacked on, answerable), recounted from the saved answers.

    The count stored in a result file is whatever the classifier was on
    the day of the run, and the early one called any answer containing the
    not-found sentence a refusal. That put 29 of 100 against the 7B at
    fp16 when 20 were refusals and 9 were real answers with the sentence
    left on the end. Recounting here keeps every table on one definition.
    """
    pos = [x for x in result.get("rows", [])
           if x.get("ok", True) and not x.get("is_negative")]
    marks = [harness.classify_refusal(x.get("answer", ""), NOT_FOUND)
             for x in pos]
    return (sum(1 for m in marks if m[0]),
            sum(1 for m in marks if m[1]), len(pos))


# (row label, scenario, how to pull the number, lower is better)
METRICS = [
    ("Single-query TTFT, median (s)", "latency",
     lambda r: r["ttft_s"].get("median"), True),
    ("Single-query TTFT, p95 (s)", "latency",
     lambda r: r["ttft_s"].get("p95"), True),
    ("Single-query total, median (s)", "latency",
     lambda r: r["total_s"].get("median"), True),
    ("Decode rate, median (tok/s)", "latency",
     lambda r: r["tok_per_s"].get("median"), False),
    # Answers differ in length between runtimes, so raw total time is not
    # like for like. This one is.
    ("Seconds per 100 output tokens, median", "latency",
     lambda r: (r.get("s_per_100_tok") or {}).get("median"), True),
    ("Output tokens per answer, median", "latency",
     lambda r: (r.get("out_tokens") or {}).get("median"), None),
    ("Prefix-cache hit rate, latency run", "latency",
     lambda r: r.get("prefix_cache_hit_rate"), None),
    ("Batch wall clock (s)", "throughput",
     lambda r: r.get("wall_clock_s"), True),
    ("Batch throughput (req/s)", "throughput",
     lambda r: r.get("requests_per_s"), False),
    ("Batch throughput (tok/s)", "throughput",
     lambda r: r.get("tokens_per_s"), False),
    ("Fully reproducible answers", "determinism",
     lambda r: r.get("reproducible_rate"), False),
    ("Answers carrying a citation", "quality",
     lambda r: r.get("cited_rate"), False),
    ("Correct refusals (out-of-scope)", "quality",
     lambda r: f"{r.get('correct_refusals')}/{r.get('n_negative')}", None),
    ("False refusals (answerable)", "quality",
     lambda r: "{0}/{2}".format(*refusals(r)), None),
    ("Real answers ending in the not-found line", "quality",
     lambda r: "{1}/{2}".format(*refusals(r)), None),
    ("Median answer length (chars)", "quality",
     lambda r: r.get("median_chars"), None),
    # A greedy decoder with no repetition penalty can loop until the cap.
    # Counted, because a looping answer inflates length and token counts.
    ("Answers that hit the token cap", "quality",
     lambda r: (f"{sum(1 for x in r['rows'] if x.get('out_tokens', 0) >= 512)}"
                f"/{len(r['rows'])}") if r.get("rows") else None, None),
]


def load(profile: str, extra_dir: str = "") -> dict:
    """{(backend, scenario, arm): envelope}

    extra_dir adds a second Ollama arm, normally the one run with
    OLLAMA_NUM_PARALLEL set. Reporting vLLM against Ollama's shipped default
    alone overstates the batching gap by an order of magnitude, because the
    default serialises. Both arms go in the table and the verdict is taken
    against whichever Ollama did better.
    """
    found = {}
    if extra_dir:
        tuned = RESULT_DIR.parent / extra_dir
        for path in sorted(tuned.glob("*.json")) if tuned.exists() else []:
            try:
                env = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if profile and env.get("profile") != profile:
                continue
            arm = (env.get("result") or {}).get("arm", "")
            found[("ollama-tuned", env["scenario"], arm)] = env
    if not RESULT_DIR.exists():
        return found
    for path in sorted(RESULT_DIR.glob("*.json")):
        if path.name.startswith("comparison"):
            continue
        try:
            env = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {path.name}")
            continue
        if profile and env.get("profile") != profile:
            continue
        arm = (env.get("result") or {}).get("arm", "")
        found[(env["backend"], env["scenario"], arm)] = env
    return found


def fmt(value) -> str:
    if value is None:
        return "not run"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def delta(ollama, vllm, lower_is_better) -> str:
    """How vLLM compares, in the direction that is actually better."""
    if ollama is None or vllm is None or lower_is_better is None:
        return ""
    try:
        o, v = float(ollama), float(vllm)
    except (TypeError, ValueError):
        return ""
    if o == 0:
        return ""
    change = (v - o) / o * 100
    if abs(change) < 1:
        return "level"
    better = (change < 0) if lower_is_better else (change > 0)
    who = "vLLM" if better else "Ollama"
    return f"{abs(change):.0f}% to {who}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="matched-fp16")
    ap.add_argument("--tuned-dir", default="results-wsl-ollama-parallel",
                    help="directory holding the OLLAMA_NUM_PARALLEL arm")
    args = ap.parse_args()

    found = load(args.profile, args.tuned_dir)
    if not found:
        raise SystemExit(
            f"No results for profile {args.profile!r} in {RESULT_DIR}.\n"
            f"Run:  python bench/run.py all ollama")

    any_env = next(iter(found.values()))
    header = [
        f"Profile: {args.profile} ({any_env.get('note','')})",
        f"Machine: {any_env['machine'].get('gpu','?')} | "
        f"{any_env['machine'].get('platform','?')}"
        f"{' | WSL2' if any_env['machine'].get('wsl') else ''}",
    ]
    models = {b: e["model"] for (b, _, _), e in found.items()}
    for name, model in sorted(models.items()):
        header.append(f"Model ({name}): {model}")
    samplings = {json.dumps(e.get("sampling"), sort_keys=True)
                 for e in found.values()}
    if len(samplings) == 1 and None not in [e.get("sampling") for e in found.values()]:
        header.append(f"Sampling (both, sent explicitly): {samplings.pop()}")
    else:
        header.append("Sampling: NOT pinned on every file. Results mix "
                      "server defaults and are not a like-for-like comparison.")

    rows = []
    for label, scenario, pull, lower in METRICS:
        o_env = found.get(("ollama", scenario, ""))
        t_env = found.get(("ollama-tuned", scenario, ""))
        v_env = found.get(("vllm", scenario, ""))
        o = pull(o_env["result"]) if o_env else None
        tn = pull(t_env["result"]) if t_env else None
        v = pull(v_env["result"]) if v_env else None
        if o is None and tn is None and v is None:
            continue
        # Compare vLLM against Ollama's better arm, not its default.
        best = o
        if tn is not None and o is not None and lower is not None:
            try:
                best = min(float(o), float(tn)) if lower else max(float(o), float(tn))
            except (TypeError, ValueError):
                best = o
        elif tn is not None and o is None:
            best = tn
        rows.append([label, fmt(o), fmt(tn), fmt(v), delta(best, v, lower)])

    # Structured output gets its own block: three arms, not two backends.
    struct = []
    for (backend, scenario, arm), env in sorted(found.items()):
        if scenario != "structured":
            continue
        r = env["result"]
        n = r.get("n_chunks") or 0
        with_pairs = r.get("chunks_with_pairs")
        if with_pairs is None:   # files from before this column existed
            with_pairs = sum(1 for x in r.get("rows", []) if x.get("pairs"))
        struct.append([
            f"{backend} / {arm}",
            fmt(r.get("parsed_rate")),
            fmt(r.get("shaped_rate")),
            f"{n - with_pairs}/{n}",
            str(r.get("total_pairs", "")),
            fmt(r["per_chunk_s"].get("median")),
        ])

    print()
    for line in header:
        print(line)
    print()
    width = max((len(r[0]) for r in rows), default=10)
    print(f"{'':<{width}}   {'Ollama':>11} {'Ollama+par':>11} {'vLLM':>11}   verdict")
    for label, o, tn, v, d in rows:
        print(f"{label:<{width}}   {o:>11} {tn:>11} {v:>11}   {d}")
    print("\nVerdict compares vLLM against Ollama's better arm.")
    if struct:
        print("\nStructured output")
        print(f"{'arm':<28} {'parsed':>8} {'shaped':>8} {'dropped':>8} "
              f"{'pairs':>7} {'s/chunk':>8}")
        for s in struct:
            print(f"{s[0]:<28} {s[1]:>8} {s[2]:>8} {s[3]:>8} {s[4]:>7} {s[5]:>8}")
        print("shaped = right shape with no rescue branch; dropped = chunks "
              "that yielded no pair at all")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULT_DIR / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for line in header:
            w.writerow([line])
        w.writerow([])
        w.writerow(["metric", "ollama_default", "ollama_tuned", "vllm", "verdict"])
        w.writerows(rows)
        if struct:
            w.writerow([])
            w.writerow(["arm", "parsed_rate", "shaped_rate",
                        "chunks_dropped", "total_pairs", "median_s_per_chunk"])
            w.writerows(struct)

    md = ["# Ollama vs vLLM", ""]
    md += [f"- {line}" for line in header]
    md += ["", "| Metric | Ollama (default) | Ollama (`NUM_PARALLEL=4`) | vLLM | Verdict |",
           "| --- | ---: | ---: | ---: | --- |"]
    md += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    if struct:
        md += ["", "## Structured output", "",
               "| Arm | Parsed | Correct shape, no rescue | Chunks dropped | Pairs | s/chunk |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"]
        md += [f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} | {s[5]} |"
               for s in struct]
        md += ["", "`builder` is lora/build_dataset.py's own prompt and "
               "parsing, rescue branches included; `schema` passes the "
               "QA schema to the server."]
    md += ["", "Verdict compares vLLM against Ollama's better arm, so the "
           "batching figure is measured against Ollama with parallelism "
           "enabled rather than against its shipped default.", ""]
    if any(backend == "ollama-tuned" for backend, _, _ in found):
        md += ["", "The client kept 8 requests in flight. Ollama's tuned "
               "arm had 4 parallel slots, so it served 4 at a time while "
               "vLLM batched all 8. The batch verdict is therefore against "
               "Ollama at 4 slots. bench/README.md has the paired runs "
               "against Ollama at 8 slots, its better setting.", ""]
    md += ["", "Medians and p95, not means. Retrieval ran once and the same "
           "cached passages were replayed to both backends, so these numbers "
           "describe generation only.", ""]
    md_path = RESULT_DIR / "comparison.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
