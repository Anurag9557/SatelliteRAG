"""
Captioning, the batch job vLLM was supposed to be best at, on both runtimes.

Uses the project's own captioning code (core/caption.py: the same prompt,
the same image normalisation, the same page context), so what is timed is
the real call, not a re-implementation. The figures are a fixed sample of
the ones already captioned by qwen2.5vl 7B, and those 7B captions are kept
as a reference to compare against.

Why 3B and not the 7B vision model the project uses: Qwen2.5-VL-7B is
about 16 GB at fp16 and over 6 GB even at 4 bits once the vision encoder is
counted, so vLLM cannot hold it on a 6 GB card. Both runtimes here run
Qwen2.5-VL-3B at 4 bits: Ollama's qwen2.5vl:3b (GGUF Q4_K_M) and vLLM's
Qwen2.5-VL-3B-Instruct-AWQ. Same model, different 4-bit schemes, so read
caption differences as quantiser plus runtime, and speed as runtime.

    python bench/caption_bench.py run ollama --arm serial --model qwen2.5vl:3b
    python bench/caption_bench.py run vllm   --arm batched --concurrency 8
    python bench/caption_bench.py report

Writes bench/results-caption/. Captions describe confidential figures, so
that directory is gitignored like bench/results/.
"""

import argparse
import importlib
import json
import os
import re
import statistics
import sys
import time
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

REPO    = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "bench" / "results-caption"
SLUG    = "aditya-l1"


def load_caption(backend: str):
    """core.caption without core/__init__.py.

    The package __init__ imports the whole retrieval stack, which is not
    installed in the vLLM venv and is not needed to caption. Registering a
    bare 'core' package first lets caption.py's own 'from . import llm'
    resolve to core/llm.py and nothing else.
    """
    os.environ["SATRAG_LLM"] = backend
    pkg = types.ModuleType("core")
    pkg.__path__ = [str(REPO / "core")]
    sys.modules["core"] = pkg
    return importlib.import_module("core.caption")


def satellite():
    ext = REPO / "satellites" / SLUG / "extracted"
    return SimpleNamespace(slug=SLUG, extracted_dir=ext, images_dir=ext / "images")


def sample(state: dict, n: int) -> list:
    """Every k-th figure that already has a 7B caption, so the sample spans
    all the documents rather than the first one alphabetically."""
    names = sorted(k for k, v in state.items() if v.get("caption"))
    if n and n < len(names):
        step = len(names) / n
        names = [names[int(i * step)] for i in range(n)]
    return names


def cmd_run(args):
    cap = load_caption(args.backend)
    sat = satellite()
    state = cap.load_state(sat)
    pages = cap.build_page_index(sat)
    names = sample(state, args.n)

    jobs = []
    for name in names:
        doc_type, page = cap.parse_filename(name)
        chunk = cap.nearest_chunk(pages, doc_type, page)
        jobs.append((sat.images_dir / name, cap.page_context(chunk), args.model))

    ok, why = cap.llm.available()
    if not ok:
        raise SystemExit(why)
    print(f"[{args.backend}/{args.arm}] {len(jobs)} figures, model {args.model}, "
          f"concurrency {args.concurrency}; {why}")

    def one(job):
        start = time.perf_counter()
        try:
            text, err = cap.describe(*job), ""
        except Exception as exc:
            text, err = "", f"{type(exc).__name__}: {str(exc)[:160]}"
        return text, round(time.perf_counter() - start, 3), err

    # Warm the model so a cold load is not billed to the first figure.
    one(jobs[0])

    start = time.perf_counter()
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(one, jobs))
    else:
        results = [one(j) for j in jobs]
    wall = time.perf_counter() - start

    rows = []
    for (path, _, _), (text, dt, err) in zip(jobs, results):
        rows.append({"name": path.name, "caption": text, "s": dt, "error": err,
                     "reference_7b": state[path.name]["caption"]})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.backend}_{args.arm}.json"
    out.write_text(json.dumps({
        "backend": args.backend, "arm": args.arm, "model": args.model,
        "concurrency": args.concurrency, "n": len(rows),
        "wall_clock_s": round(wall, 1), "sampling": cap.llm.summary(),
        "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = sum(1 for r in rows if r["error"])
    print(f"    wall {wall:.1f}s, {failed} failed, wrote {out}")


def tokens(t):
    return re.findall(r"[a-z0-9]+", (t or "").lower())


def f1(a, b):
    p, g = tokens(a), tokens(b)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def cmd_report(args):
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(OUT_DIR.glob("*.json")) if p.stem != "summary"}
    if not runs:
        raise SystemExit(f"nothing in {OUT_DIR}")
    summary = {}
    for name, r in runs.items():
        good = [x for x in r["rows"] if not x["error"] and x["caption"]]
        summary[name] = {
            "model": r["model"], "concurrency": r["concurrency"], "n": r["n"],
            "wall_clock_s": r["wall_clock_s"],
            "s_per_figure_effective": round(r["wall_clock_s"] / r["n"], 2),
            "failed": r["n"] - len(good),
            "median_caption_chars": int(statistics.median(len(x["caption"]) for x in good)) if good else 0,
            "f1_vs_7b_caption": round(statistics.mean(f1(x["caption"], x["reference_7b"]) for x in good), 3) if good else 0,
        }
    # Agreement between runtimes on the same figure.
    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = {x["name"]: x["caption"] for x in runs[names[i]]["rows"]}
            b = {x["name"]: x["caption"] for x in runs[names[j]]["rows"]}
            both = [k for k in a if a[k] and b.get(k)]
            if both:
                summary[f"f1 {names[i]} vs {names[j]}"] = round(
                    statistics.mean(f1(a[k], b[k]) for k in both), 3)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("backend", choices=["ollama", "vllm"])
    p.add_argument("--arm", default="serial")
    p.add_argument("--model", default="qwen2.5vl:3b")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("-n", type=int, default=100)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("report")
    p.set_defaults(func=cmd_report)
    a = ap.parse_args()
    a.func(a)
