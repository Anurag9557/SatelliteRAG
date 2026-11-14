"""
Benchmark runner: Ollama against vLLM on the same work.

    python bench/run.py cache aditya-l1
    python bench/run.py run ollama latency
    python bench/run.py run vllm   latency
    python bench/run.py all ollama
    python bench/run.py all vllm
    python bench/report.py

Every result file records the model, the profile, and what the machine was,
because the first question anyone asks about a benchmark is what it ran on.
"""

import argparse
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone

import backends
import harness

SCENARIOS = ["latency", "throughput", "determinism", "quality", "structured"]


def machine() -> dict:
    """Whatever can be read about the host without extra dependencies."""
    info = {
        "python":   platform.python_version(),
        "platform": platform.platform(),
        "wsl":      "microsoft" in platform.release().lower(),
    }
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20)
            info["gpu"] = out.stdout.strip().splitlines()[0]
        except (subprocess.SubprocessError, IndexError):
            info["gpu"] = "nvidia-smi failed"
    else:
        info["gpu"] = "no nvidia-smi on PATH"
    return info


def save(backend_name: str, profile: str, scenario: str,
         payload: dict, suffix: str = "") -> None:
    harness.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag  = f"{backend_name}_{profile}_{scenario}{suffix}"
    path = harness.RESULT_DIR / f"{tag}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"    wrote {path.relative_to(harness.BENCH_DIR.parent)}")


def envelope(backend, profile: str, scenario: str, body: dict,
             elapsed: float) -> dict:
    return {
        "backend":   backend.name,
        "model":     backend.model,
        "profile":   profile,
        "note":      backends.PROFILES[profile]["note"],
        "scenario":  scenario,
        "num_ctx":   backend.num_ctx,
        "run_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine":   machine(),
        "elapsed_s": round(elapsed, 1),
        "sampling":  backends.sampling_note(),
        "result":    body,
    }


def single_instance():
    """Refuse to start if another copy of this runner is already timing.

    The first full run went wrong exactly here: phase C was launched four
    times, all four copies drove one vLLM engine at once, and each wrote its
    numbers over the others. Nothing in the output said so; it took the
    engine log (Running: 4 reqs during a sequential test) to show it. The
    lock lives in /tmp on the Linux side because /mnt/c does not honour
    flock. On Windows there is no fcntl and the guard is skipped.
    """
    try:
        import fcntl
    except ImportError:
        return None
    fh = open("/tmp/satrag_bench.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("Another bench/run.py is already running. "
                         "Two runs on one server corrupt both results.")
    return fh   # held until this process exits


def cmd_cache(args):
    # cli.py takes the satellite positionally everywhere, so anyone coming
    # from it types "cache aditya-l1" and argparse rejects it. Accept both:
    # the positional wins when given, otherwise --satellite, which already
    # defaults to aditya-l1.
    slug = getattr(args, "satellite_pos", None) or args.satellite
    harness.build_context_cache(slug, args.n, args.force)


def run_one(backend, scenario: str, args) -> None:
    records = harness.load_context(args.satellite)
    meta    = harness.prompt_meta(args.satellite)
    if args.n:
        records = harness.take(records, args.n)

    print(f"\n[{backend.name}] {scenario}  model={backend.model}")
    health = backend.health()
    if health != "up":
        raise SystemExit(f"    {backend.name} is not answering: {health}")

    before = backends.VLLM.metrics() if backend.name == "vllm" else {}
    start = time.perf_counter()
    suffix = ""

    if scenario == "latency":
        body = harness.scenario_latency(backend, records, args.max_tokens,
                                        meta)
    elif scenario == "throughput":
        body = harness.scenario_throughput(backend, records, args.max_tokens,
                                           args.concurrency, meta)
    elif scenario == "determinism":
        body = harness.scenario_determinism(
            backend, records[:args.determinism_questions],
            args.max_tokens, args.repeats, meta)
    elif scenario == "quality":
        body = harness.scenario_quality(backend, records, args.max_tokens,
                                        meta)
    elif scenario == "structured":
        chunks = harness.load_chunks(args.satellite, args.structured_chunks)
        body   = harness.scenario_structured(backend, chunks, args.arm,
                                             args.max_tokens)
        suffix = f"_{args.arm}"
    else:
        raise SystemExit(f"unknown scenario {scenario!r}")

    elapsed = time.perf_counter() - start
    if backend.name == "vllm":
        after = backends.VLLM.metrics()
        body["prefix_cache_hit_rate"] = backends.prefix_hit_rate(before, after)
        body["server_metrics_seen"] = bool(after)
    if scenario != "structured":
        body["prompt"] = meta.get("name", "cache")
    save(backend.name, args.profile, scenario,
         envelope(backend, args.profile, scenario, body, elapsed), suffix)


def cmd_run(args):
    _lock = single_instance()
    backend = backends.build(args.backend, args.profile, args.num_ctx)
    run_one(backend, args.scenario, args)


def cmd_all(args):
    _lock = single_instance()
    backend = backends.build(args.backend, args.profile, args.num_ctx)
    for scenario in SCENARIOS:
        if scenario == "structured":
            # "builder" is what lora/build_dataset.py does today, parsed
            # with its own rescue branches; "schema" is the fix. Running
            # both on Ollama answers whether vLLM is needed for the fix.
            arms = (["builder", "schema"] if args.backend == "ollama"
                    else ["schema"])
            for arm in arms:
                args.arm = arm
                run_one(backend, scenario, args)
        else:
            run_one(backend, scenario, args)


def main():
    # The shared options live on a parent parser that every subcommand
    # inherits, so they can be written after the subcommand. Putting them
    # only on the top-level parser means "run ollama latency --n 5" is
    # rejected and the flag has to go before the word "run", which nobody
    # guesses and nothing in the help text suggests.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--satellite", default="aditya-l1")
    common.add_argument("--profile", default="matched-fp16",
                        choices=sorted(backends.PROFILES))
    common.add_argument("--num-ctx", type=int, default=8192)
    common.add_argument("--max-tokens", type=int, default=512)
    common.add_argument("-n", type=int, default=0,
                        help="cap the number of questions (0 = all)")
    common.add_argument("--concurrency", type=int, default=8)
    common.add_argument("--repeats", type=int, default=5)
    common.add_argument("--determinism-questions", type=int, default=10)
    common.add_argument("--structured-chunks", type=int, default=50)
    common.add_argument("--arm", default="schema",
                        choices=["builder", "json-string", "schema", "none"])
    common.add_argument("--force", action="store_true")

    ap = argparse.ArgumentParser(
        description=__doc__, parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cache", parents=[common],
                       help="retrieve once and store the context")
    p.add_argument("satellite_pos", nargs="?", default=None,
                   metavar="satellite",
                   help="satellite slug (optional; --satellite also works)")
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("run", parents=[common],
                       help="one backend, one scenario")
    p.add_argument("backend", choices=["ollama", "vllm"])
    p.add_argument("scenario", choices=SCENARIOS)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("all", parents=[common],
                       help="one backend, every scenario")
    p.add_argument("backend", choices=["ollama", "vllm"])
    p.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
