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

RESULT_DIR = Path(__file__).resolve().parent / "results"

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
     lambda r: f"{r.get('false_refusals')}/{r.get('n_positive')}", None),
    ("Median answer length (chars)", "quality",
     lambda r: r.get("median_chars"), None),
]


def load(profile: str) -> dict:
    """{(backend, scenario, arm): envelope}"""
    found = {}
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
    args = ap.parse_args()

    found = load(args.profile)
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

    rows = []
    for label, scenario, pull, lower in METRICS:
        o_env = found.get(("ollama", scenario, ""))
        v_env = found.get(("vllm", scenario, ""))
        o = pull(o_env["result"]) if o_env else None
        v = pull(v_env["result"]) if v_env else None
        if o is None and v is None:
            continue
        rows.append([label, fmt(o), fmt(v), delta(o, v, lower)])

    # Structured output gets its own block: three arms, not two backends.
    struct = []
    for (backend, scenario, arm), env in sorted(found.items()):
        if scenario != "structured":
            continue
        r = env["result"]
        struct.append([
            f"{backend} / format={arm}",
            fmt(r.get("parsed_rate")),
            fmt(r.get("shaped_rate")),
            str(r.get("total_pairs", "")),
            fmt(r["per_chunk_s"].get("median")),
        ])

    print()
    for line in header:
        print(line)
    print()
    width = max((len(r[0]) for r in rows), default=10)
    print(f"{'':<{width}}   {'Ollama':>12} {'vLLM':>12}   verdict")
    for label, o, v, d in rows:
        print(f"{label:<{width}}   {o:>12} {v:>12}   {d}")
    if struct:
        print("\nStructured output")
        print(f"{'arm':<28} {'parsed':>8} {'shaped':>8} {'pairs':>7} {'s/chunk':>8}")
        for s in struct:
            print(f"{s[0]:<28} {s[1]:>8} {s[2]:>8} {s[3]:>7} {s[4]:>8}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULT_DIR / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for line in header:
            w.writerow([line])
        w.writerow([])
        w.writerow(["metric", "ollama", "vllm", "verdict"])
        w.writerows(rows)
        if struct:
            w.writerow([])
            w.writerow(["arm", "parsed_rate", "shaped_rate",
                        "total_pairs", "median_s_per_chunk"])
            w.writerows(struct)

    md = ["# Ollama vs vLLM", ""]
    md += [f"- {line}" for line in header]
    md += ["", "| Metric | Ollama | vLLM | Verdict |",
           "| --- | ---: | ---: | --- |"]
    md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    if struct:
        md += ["", "## Structured output", "",
               "| Arm | Parsed | Correct shape | Pairs | s/chunk |",
               "| --- | ---: | ---: | ---: | ---: |"]
        md += [f"| {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} |"
               for s in struct]
    md += ["", "Medians and p95, not means. Retrieval ran once and the same "
           "cached passages were replayed to both backends, so these numbers "
           "describe generation only.", ""]
    md_path = RESULT_DIR / "comparison.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
