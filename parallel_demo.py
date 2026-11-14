#!/usr/bin/env python3
"""
Parallel demo: the same questions answered one at a time, then all at once.

Runs the app's real path (router, retrieval, prompt, grounding guard) on
whichever server SATRAG_LLM points at, so the two runs can be compared on
the same questions:

    $env:SATRAG_LLM="vllm"; $env:SATRAG_VLLM_URL="http://127.0.0.1:8000/v1"
    .venv\\Scripts\\python.exe parallel_demo.py aditya-l1 -n 8

    $env:SATRAG_LLM="ollama"; $env:OLLAMA_HOST="http://127.0.0.1:11600"
    .venv\\Scripts\\python.exe parallel_demo.py aditya-l1 -n 8 --model qwen2.5:7b-instruct-fp16

The search runs first, on this machine, for every question, and is timed
separately. Only the answering is compared: once with one question in
flight, once with all n in flight. vLLM batches in-flight requests on the
GPU. Ollama does so only when its server was started with
OLLAMA_NUM_PARALLEL above 1; with the default of 1 it queues them.

--parallel-only skips the one-at-a-time round.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from core import generate, llm, registry, retrieve, router

QUESTIONS = [
    "How do I perform an OBC system reset?",
    "OBC CPU reset procedure",
    "telecommand link test procedure",
    "How do you interrupt one OBC's power using the other OBC?",
    "What happens when the spacecraft enters safe mode?",
    "What are the Aditya-L1 orbit constraints?",
    "What is on the hard emergency list?",
    "tone ranging of transponder-1 procedure",
    "thermal requirements and special heater needs",
    "What are the guidelines for Payload operations?",
    "Stellar calibration guidelines",
    "The X-band system",
    "What are the safety logics?",
    "OBC watchdog timer limit and reset procedure",
    "EEPROM management procedure",
    "energy balance analysis",
]

FAILED = "[Generation unavailable"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("satellite")
    ap.add_argument("-n", type=int, default=8, help="questions (max %d)" % len(QUESTIONS))
    ap.add_argument("--model", default=generate.DEFAULT_MODEL)
    ap.add_argument("--parallel-only", action="store_true")
    args = ap.parse_args()

    sats = registry.resolve([args.satellite])
    if not sats:
        raise SystemExit(f"no satellite {args.satellite}")
    qs = QUESTIONS[:max(1, min(args.n, len(QUESTIONS)))]
    where = llm.VLLM_URL if llm.BACKEND == "vllm" else llm.OLLAMA_URL
    print(f"\nserver: {llm.BACKEND} at {where}   questions: {len(qs)}\n")

    # 1. search, on this machine, once per question
    t0 = time.perf_counter()
    jobs = []
    for q in qs:
        plan = router.plan(q, sats[0].retrieval)
        hits = retrieve.search(sats, q, top_k=5, plan=plan)
        jobs.append((q, hits, plan))
    t_search = time.perf_counter() - t0
    print(f"search (laptop): {t_search:.1f} s for {len(qs)} questions\n")

    def answer(job):
        q, hits, plan = job
        s = time.perf_counter()
        a = generate.generate(q, hits, model=args.model,
                              show_satellite=len(sats) > 1, plan=plan)
        return a, time.perf_counter() - s

    # 2. one at a time
    seq = None
    if not args.parallel_only:
        print("answering ONE AT A TIME ...")
        s = time.perf_counter()
        seq = [answer(j) for j in jobs]
        t_seq = time.perf_counter() - s
        print(f"  done in {t_seq:.1f} s\n")

    # 3. all at once
    print(f"answering ALL {len(qs)} AT ONCE ...")
    s = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        par = list(pool.map(answer, jobs))
    t_par = time.perf_counter() - s
    print(f"  done in {t_par:.1f} s\n")

    # 4. table
    w = 46
    head = f"  {'question':<{w}}" + ("  one-by-one" if seq else "") + "  all-at-once  first line of answer"
    print(head)
    print("  " + "-" * (len(head) + 10))
    for i, q in enumerate(qs):
        a, tp = par[i]
        first = next((ln.strip() for ln in a.splitlines() if ln.strip()), "")[:48]
        row = f"  {q[:w]:<{w}}"
        if seq:
            row += f"  {seq[i][1]:>9.1f} s"
        row += f"  {tp:>9.1f} s  {first}"
        print(row)

    bad = sum(a.startswith(FAILED) for a, _ in par) + sum(a.startswith(FAILED) for a, _ in (seq or []))
    print()
    if seq:
        print(f"  TOTAL answering time:  one at a time {t_seq:.1f} s   |   all at once {t_par:.1f} s"
              f"   |   {t_seq / t_par:.1f}x faster together")
    else:
        print(f"  TOTAL answering time, all at once: {t_par:.1f} s")
    print(f"  server: {llm.BACKEND} at {where}" + (f"   |   FAILED answers: {bad}" if bad else ""))
    print()


if __name__ == "__main__":
    main()
