"""
Side-by-side demo: one question, or a batch of them, answered by Ollama
and by vLLM from the same retrieved passages, in a browser page.

    python demo/server.py aditya-l1
    python demo/server.py aditya-l1 --ollama-model qwen2.5:7b-instruct-fp16

Then open http://127.0.0.1:8765

The two servers are found the way core/llm.py finds them:

    OLLAMA_HOST        default http://localhost:11434
    SATRAG_VLLM_URL    default http://localhost:8000/v1

Routing, search, the prompt and the grounding guard are the app's own
(core/) and run once per question on this machine. Only the answering
differs, so anything the page shows between the two columns is the
server and nothing else. Sampling is pinned the same on both: greedy,
fixed seed, no repetition penalty.

The page is served on 127.0.0.1 only. The answers quote the mission
documents, so it is not to be opened to the network.

Standard library only on top of what the project already needs.
"""

import argparse
import csv
import json
import os
import queue
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import generate, grounding, registry, retrieve, router   # noqa: E402
from parallel_demo import QUESTIONS                                # noqa: E402

PAGE   = Path(__file__).resolve().parent / "page.html"
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
VLLM   = os.environ.get("SATRAG_VLLM_URL", "http://localhost:8000/v1").rstrip("/")
SEED   = 1234
SIDES  = ("ollama", "vllm")

# Filled in by main().
SAT    = None
MODELS = {"ollama": "", "vllm": ""}

# One search at a time. The embedder, the reranker and the store are
# shared, and the handler runs a thread per request.
SEARCH_LOCK = threading.Lock()

# Shown under an answer the app gives without a model.
_SAME = "without a model, so there is nothing for the servers to differ on."
WHY = {
    "caption":  "A figure request is answered from the figure's caption, " + _SAME,
    "commands": "A command-log lookup by date, command id or mnemonic is "
                "answered in code from every matching record, " + _SAME,
    "none":     "Nothing was retrieved, so the app says so " + _SAME,
}


# ------------------------------------------------------------ the servers


def get_json(url: str, timeout: float = 4):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def post_lines(url: str, payload: dict):
    """Lines of a streaming POST, as they arrive."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                yield line


def stream_ollama(messages: list):
    """("text", piece) for each piece, then ("tokens", n) at the end."""
    payload = {
        "model": MODELS["ollama"], "messages": messages, "stream": True,
        "options": {"temperature": 0.0, "seed": SEED,
                    "num_ctx": generate.NUM_CTX,
                    "num_predict": generate.MAX_TOKENS,
                    "repeat_penalty": 1.0},
    }
    for line in post_lines(f"{OLLAMA}/api/chat", payload):
        chunk = json.loads(line)
        if chunk.get("error"):
            raise RuntimeError(str(chunk["error"]))
        piece = (chunk.get("message") or {}).get("content", "")
        if piece:
            yield "text", piece
        if chunk.get("done"):
            yield "tokens", chunk.get("eval_count", 0)


def stream_vllm(messages: list):
    payload = {
        "model": vllm_model(), "messages": messages, "stream": True,
        "temperature": 0.0, "seed": SEED,
        "max_tokens": generate.MAX_TOKENS, "repetition_penalty": 1.0,
        "stream_options": {"include_usage": True},
    }
    for line in post_lines(f"{VLLM}/chat/completions", payload):
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            yield "tokens", chunk["usage"].get("completion_tokens", 0)
        for choice in chunk.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content") or ""
            if piece:
                yield "text", piece


STREAMS = {"ollama": stream_ollama, "vllm": stream_vllm}


def vllm_model() -> str:
    """The name given on the command line, else whatever vLLM is serving."""
    if not MODELS["vllm"]:
        served = get_json(f"{VLLM}/models").get("data") or []
        if served:
            MODELS["vllm"] = served[0]["id"]
    return MODELS["vllm"]


def answer(side: str, job: dict, on_text=None) -> dict:
    """
    One answer from one server, timed from the moment the request leaves.

    first_s is the wait for the first piece of text, total_s the wait for
    the last. The rate is over the time spent writing, with the wait for
    the first token left out, which is how bench/ reports it too.
    """
    parts, tokens, first = [], 0, None
    start = time.perf_counter()
    try:
        for kind, value in STREAMS[side](job["messages"]):
            if kind == "tokens":
                tokens = value
                continue
            if first is None:
                first = time.perf_counter() - start
            parts.append(value)
            if on_text:
                on_text(value)
    except (urllib.error.URLError, OSError, RuntimeError,
            json.JSONDecodeError) as exc:
        return {"side": side, "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    total = time.perf_counter() - start
    text  = "".join(parts)
    # The same steps the app runs after the model (core/generate.answer):
    # no identifier the passages do not hold, no "steps not in the
    # documents" under a procedure whose steps are there, and citations
    # checked. The count shows how often the guard had to step in.
    text, removed = grounding.guard(text, job["context"])
    text, _       = grounding.drop_false_no_steps(text, job["context"])
    text, _       = generate.drop_trailing_not_found(text)
    text, _       = generate.check_citations(text, job.get("n_sources", 5))
    tokens  = tokens or len(parts)
    writing = total - (first or 0)
    return {
        "side": side, "ok": True, "text": text,
        "first_s": round(first or 0, 3), "total_s": round(total, 3),
        "tokens": tokens,
        "tokens_per_s": round(tokens / writing, 1) if writing > 0 else 0,
        "ids_removed": len(removed or []),
    }


# ------------------------------------------------------------ the question


def prepare(question: str) -> dict:
    """Route and search once, and build the messages both servers get."""
    with SEARCH_LOCK:
        start = time.perf_counter()
        plan  = router.plan(question, SAT.retrieval)
        hits  = retrieve.search([SAT], question, top_k=5, plan=plan)
        took  = time.perf_counter() - start

    fixed, path = generate.answer_without_model(hits, plan, question)
    context = retrieve.build_context(hits)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    note = grounding.context_note(context, related=generate.NOTE_RELATED)
    if note:
        user += f"\n\n{note}"
    return {
        "question": question,
        "context":  context,
        "messages": [{"role": "system", "content": generate.SYSTEM_PROMPT},
                     {"role": "user", "content": user}],
        "sources":  [h.cite() for h in hits],
        "n_sources": len(hits),
        "route":    plan.describe(),
        "search_s": round(took, 2),
        # What the app answers in code: a figure from its caption, a
        # command-log lookup from every matching record, and nothing
        # found. No model is asked, so there is nothing for the two
        # servers to differ on.
        "fixed":    fixed,
        "why":      WHY.get(path, ""),
    }


def status() -> dict:
    out = {"satellite": SAT.name, "questions": QUESTIONS,
           "sampling": "greedy, seed fixed, no repetition penalty, "
                       f"up to {generate.MAX_TOKENS} tokens"}
    try:
        version = get_json(f"{OLLAMA}/api/version").get("version", "")
        have = [m["name"] for m in get_json(f"{OLLAMA}/api/tags").get("models", [])]
        out["ollama"] = {"up": True, "url": OLLAMA, "version": version,
                         "model": MODELS["ollama"],
                         "has_model": MODELS["ollama"] in have}
    except Exception as exc:
        out["ollama"] = {"up": False, "url": OLLAMA, "model": MODELS["ollama"],
                         "error": type(exc).__name__}
    try:
        out["vllm"] = {"up": True, "url": VLLM, "model": vllm_model()}
    except Exception as exc:
        out["vllm"] = {"up": False, "url": VLLM, "model": MODELS["vllm"],
                       "error": type(exc).__name__}
    return out


def measured() -> dict:
    """What the benchmark recorded, for the third tab: the numbers to
    fall back on when the shared card is too busy for a live run to mean
    much."""
    out = {"paired": [], "quiet": []}
    path = ROOT / "bench" / "results-gpu-paired" / "paired.csv"
    if path.exists():
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        for series in sorted({r["series"] for r in rows}):
            arms = {}
            for r in rows:
                if r["series"] == series:
                    arms.setdefault(r["arm"], []).append(float(r["tokens_per_s"]))
            out["paired"].append({
                "series": series,
                "arms": [{"arm": a, "runs": v,
                          "median": statistics.median(v)} for a, v in arms.items()]})
    path = ROOT / "bench" / "results-gpu-quiet" / "results-gpu__comparison.csv"
    if path.exists():
        for row in csv.reader(path.open(encoding="utf-8")):
            if len(row) == 5 and row[0] in (
                    "Single-query total, median (s)",
                    "Single-query TTFT, median (s)",
                    "Batch throughput (tok/s)",
                    "False refusals (answerable)"):
                out["quiet"].append({"metric": row[0], "ollama": row[1],
                                     "ollama_4_slots": row[2], "vllm": row[3]})
    return out


# ------------------------------------------------------------ the page


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):       # keep the terminal quiet
        pass

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def open_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def emit(self, event: str, data: dict) -> None:
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def do_GET(self):
        url   = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/status":
                self.send_json(status())
            elif url.path == "/api/measured":
                self.send_json(measured())
            elif url.path == "/api/ask":
                self.ask(query.get("q", "").strip(), query.get("mode", "together"))
            elif url.path == "/api/batch":
                self.batch(int(query.get("n", 8)), int(query.get("flight", 8)),
                           query.get("first", "ollama"))
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                              # the tab was closed mid-answer

    # One question, both servers.

    def ask(self, question: str, mode: str) -> None:
        self.open_stream()
        if not question:
            self.emit("end", {"error": "no question"})
            return
        job = prepare(question)
        self.emit("search", {k: job[k] for k in ("sources", "route", "search_s")})
        if job["fixed"]:
            self.emit("caption", {"text": job["fixed"], "why": job["why"]})
            self.emit("end", {})
            return

        events = queue.Queue()

        def run(side):
            events.put(("start", {"side": side}))
            result = answer(side, job, lambda piece: events.put(
                ("token", {"side": side, "text": piece})))
            events.put(("done", result))

        def drain(waiting: int) -> None:
            while waiting:
                event, data = events.get()
                self.emit(event, data)
                if event == "done":
                    waiting -= 1

        if mode == "turns":
            for side in SIDES:
                threading.Thread(target=run, args=(side,), daemon=True).start()
                drain(1)
        else:
            for side in SIDES:
                threading.Thread(target=run, args=(side,), daemon=True).start()
            drain(len(SIDES))
        self.emit("end", {})

    # A batch, one server and then the other.

    def batch(self, n: int, in_flight: int, first: str) -> None:
        self.open_stream()
        n = max(1, min(n, len(QUESTIONS)))
        in_flight = max(1, min(in_flight, n))
        jobs, search_s = [], 0.0
        for i, question in enumerate(QUESTIONS[:n]):
            job = prepare(question)
            search_s += job["search_s"]
            jobs.append(job)
            self.emit("searching", {"done": i + 1, "of": n})
        self.emit("search", {"search_s": round(search_s, 1), "n": n,
                             "questions": QUESTIONS[:n]})

        order = [first] + [s for s in SIDES if s != first]
        totals = {}
        for side in order:
            self.emit("start", {"side": side, "n": n, "in_flight": in_flight})
            start = time.perf_counter()
            tokens = failed = 0
            with ThreadPoolExecutor(max_workers=in_flight) as pool:
                futures = {pool.submit(answer, side, job): i
                           for i, job in enumerate(jobs)}
                for future in as_completed(futures):
                    result = future.result()
                    tokens += result.get("tokens", 0)
                    failed += not result["ok"]
                    self.emit("answer", {
                        "side": side, "i": futures[future], "ok": result["ok"],
                        "total_s": result.get("total_s", 0),
                        "at_s": round(time.perf_counter() - start, 1)})
            wall = time.perf_counter() - start
            totals[side] = {"side": side, "wall_s": round(wall, 1),
                            "tokens": tokens, "failed": failed,
                            "tokens_per_s": round(tokens / wall, 1) if wall else 0}
            self.emit("side_done", totals[side])
        self.emit("end", totals)


def main():
    global SAT
    ap = argparse.ArgumentParser(description="Ollama and vLLM, side by side")
    ap.add_argument("satellite")
    ap.add_argument("--ollama-model", default=generate.DEFAULT_MODEL)
    ap.add_argument("--vllm-model", default="",
                    help="default: the first model the vLLM server lists")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    sats = registry.resolve([args.satellite])
    if not sats:
        raise SystemExit(f"no satellite {args.satellite}")
    SAT = sats[0]
    MODELS["ollama"], MODELS["vllm"] = args.ollama_model, args.vllm_model

    # The first search loads the embedder and the reranker. Paying for
    # that now keeps it out of the first question someone is watching.
    print("loading the retrieval models ...", flush=True)
    prepare(QUESTIONS[0])

    # A server that has not answered since it started loads the model on
    # its first request. One throwaway reply from each keeps that wait out
    # of the first comparison, where it would count against the cold side.
    ready = {"messages": [{"role": "user", "content": "Reply with: ready"}],
             "context": ""}
    for side in SIDES:
        answer(side, ready)

    state = status()
    for side in SIDES:
        s = state[side]
        print(f"  {side:<7} {'up  ' if s['up'] else 'DOWN'} {s['url']}  "
              f"{s.get('model') or ''}")
    print(f"\n  open http://127.0.0.1:{args.port}   (Ctrl+C to stop)\n", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
