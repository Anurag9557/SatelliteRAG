"""
One request interface over Ollama and vLLM, so the two are timed the same way.

Both servers are driven over HTTP with streaming on, for one reason: time to
first token cannot be measured any other way, and it has to be measured at the
same point in both paths or the comparison is meaningless. Using the vLLM
Python class for one side and HTTP for the other would compare an in-process
call against a network round trip.

That choice costs vLLM something. Its offline LLM.generate() over a list of
prompts is its fastest batch path, and the server is not quite as fast. So the
throughput numbers here are a floor for vLLM, not a ceiling. Said plainly in
the report rather than buried.

Only the standard library is used. This project runs offline on machines that
are set up once, and a benchmark that needs its own dependency tree is a
benchmark nobody runs.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Ports are overridable because the lab GPU box is shared: its system Ollama
# already owns 11434 and is used by other people, so the bench runs its own
# servers on spare ports there (bench/gpu/run_gpu.sh).
_OLLAMA = os.environ.get("BENCH_OLLAMA", "http://localhost:11434")
_VLLM   = os.environ.get("BENCH_VLLM", "http://localhost:8000")
OLLAMA_URL = f"{_OLLAMA}/api/chat"
VLLM_URL   = f"{_VLLM}/v1/chat/completions"
VLLM_METRICS_URL = f"{_VLLM}/metrics"

# Sent explicitly on every request to both servers, so neither side falls
# back to a default the other does not share. The first full run did not do
# this: vLLM loaded the model's generation_config.json and applied
# repetition_penalty=1.1 to every request (its startup log says so), while
# the Ollama side looped to the 512-token cap on 14 of 100 answers. Answer
# length, citation rate and refusals then measured the penalty, not the
# runtime. 1.0 means "no penalty", the only setting both implement the same
# way: Ollama penalises the last 64 tokens, vLLM the whole sequence, so 1.1
# on both would still not be the same computation.
REPETITION_PENALTY = float(os.environ.get("BENCH_REPETITION_PENALTY", "1.0"))


def sampling_note() -> dict:
    """Recorded in every result file, so a reader can see what was pinned."""
    return {"temperature": 0.0, "repetition_penalty": REPETITION_PENALTY,
            "sent_explicitly_to_both": True}

# Long enough that a slow first load does not look like a failure, short
# enough that a wedged server does not hang the whole run.
TIMEOUT_S = 300


@dataclass
class Reply:
    """One generation, with the timings taken around it."""

    text:          str   = ""
    ttft_s:        float = 0.0   # wall clock to the first content token
    total_s:       float = 0.0   # wall clock to the last token
    out_tokens:    int   = 0
    prompt_tokens: int   = 0
    # What the server itself reported, where it reports anything. Ollama
    # returns its own timings; vLLM does not, so this stays None there and
    # the report falls back to wall clock for both.
    server_eval_s: float = None
    ok:            bool  = True
    error:         str   = ""

    @property
    def decode_s(self) -> float:
        """Time spent generating, with the prefill excluded."""
        return max(self.total_s - self.ttft_s, 0.0)

    @property
    def tok_per_s(self) -> float:
        return self.out_tokens / self.decode_s if self.decode_s > 0 else 0.0


def _post_stream(url: str, payload: dict):
    """Yield raw lines from a streaming POST, decoded and stripped."""
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                yield line


class Backend:
    """Shared shape. Subclasses build the payload and read the stream."""

    name = "base"

    def __init__(self, model: str, num_ctx: int = 8192):
        self.model   = model
        self.num_ctx = num_ctx

    def chat(self, messages: list, max_tokens: int = 512,
             temperature: float = 0.0, seed: int = 0,
             schema=None) -> Reply:
        raise NotImplementedError

    def health(self) -> str:
        """A one-token request, to prove the server answers before timing it."""
        r = self.chat([{"role": "user", "content": "ok"}], max_tokens=1)
        return "up" if r.ok else f"down: {r.error}"


class Ollama(Backend):
    """
    Ollama's /api/chat, streamed as newline-delimited JSON.

    'format' takes either the string "json" or a whole JSON schema. Both are
    worth measuring: the project's own code passes the string today, and the
    schema form is what Ollama can actually do. Passing the string and then
    branching over whatever shape came back is the thing being compared, not
    a limit of the server.
    """

    name = "ollama"

    def chat(self, messages, max_tokens=512, temperature=0.0, seed=0,
             schema=None) -> Reply:
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   True,
            "options": {
                "temperature":    temperature,
                "seed":           seed,
                "num_ctx":        self.num_ctx,
                "num_predict":    max_tokens,
                "repeat_penalty": REPETITION_PENALTY,
            },
        }
        if schema is not None:
            payload["format"] = schema

        reply = Reply()
        parts = []
        start = time.perf_counter()
        try:
            for line in _post_stream(OLLAMA_URL, payload):
                chunk = json.loads(line)
                if chunk.get("error"):
                    reply.ok, reply.error = False, str(chunk["error"])
                    break

                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    if not parts:
                        reply.ttft_s = time.perf_counter() - start
                    parts.append(piece)

                if chunk.get("done"):
                    reply.out_tokens    = chunk.get("eval_count", 0)
                    reply.prompt_tokens = chunk.get("prompt_eval_count", 0)
                    ns = chunk.get("eval_duration")
                    if ns:
                        reply.server_eval_s = ns / 1e9
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            reply.ok, reply.error = False, f"{type(exc).__name__}: {exc}"

        reply.total_s = time.perf_counter() - start
        reply.text    = "".join(parts)
        return reply


class VLLM(Backend):
    """
    vLLM's OpenAI-compatible /v1/chat/completions, streamed as SSE.

    Structured output goes through 'structured_outputs', which the server
    documents as extra kwargs and which takes json / regex / choice / grammar.
    response_format is only documented for the completions endpoint, so this
    uses the field that is documented for chat.

    stream_options.include_usage is part of the OpenAI streaming spec but is
    not in vLLM's own server docs, so the token count is asked for and then
    counted by hand if it does not arrive. A missing usage block should not
    silently report zero tokens per second.
    """

    name = "vllm"

    def chat(self, messages, max_tokens=512, temperature=0.0, seed=0,
             schema=None) -> Reply:
        payload = {
            "model":       self.model,
            "messages":    messages,
            "stream":      True,
            "temperature": temperature,
            "seed":        seed,
            "max_tokens":  max_tokens,
            "repetition_penalty": REPETITION_PENALTY,
            "stream_options": {"include_usage": True},
        }
        if schema is not None:
            # disable_any_whitespace is a per-request field on
            # StructuredOutputsParams, not a server flag in this build.
            # Left off, the JSON grammar admits arbitrary whitespace
            # between tokens, and the decoder can emit "{" followed by
            # spaces and carriage returns until max_tokens runs out. That
            # output never violates the schema, so the constraint cannot
            # stop it, and a third of requests came back as unparseable
            # whitespace with finish_reason "length".
            if schema == "json":
                # The "any valid JSON" mode, vLLM's equivalent of Ollama's
                # format="json", for the builder arm.
                payload["structured_outputs"] = {
                    "json_object": True,
                    "disable_any_whitespace": True,
                }
            else:
                payload["structured_outputs"] = {
                    "json": schema,
                    "disable_any_whitespace": True,
                }

        reply = Reply()
        parts = []
        deltas = 0
        start = time.perf_counter()
        try:
            for line in _post_stream(VLLM_URL, payload):
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage:
                    reply.out_tokens    = usage.get("completion_tokens", 0)
                    reply.prompt_tokens = usage.get("prompt_tokens", 0)

                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        if not parts:
                            reply.ttft_s = time.perf_counter() - start
                        parts.append(piece)
                        deltas += 1
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            reply.ok, reply.error = False, f"{type(exc).__name__}: {exc}"

        reply.total_s = time.perf_counter() - start
        reply.text    = "".join(parts)
        if not reply.out_tokens:
            # One content delta is one token often enough for a rate, and a
            # rate from counted deltas beats a rate of zero. The report marks
            # which runs fell back to this.
            reply.out_tokens = deltas
        return reply

    # ------------------------------------------------------------ metrics

    @staticmethod
    def metrics() -> dict:
        """Sum of each vllm:* gauge/counter on the Prometheus endpoint.

        Used for two things. Before a scenario: refuse to time anything if
        the engine already has requests running or waiting, because that
        means another client is on the server. The first full run had four
        copies of the benchmark on one engine and nothing noticed. After a
        scenario: the prefix-cache hit rate, so a low TTFT can be read as
        prefill speed or as cache hits rather than guessed at.
        """
        try:
            with urllib.request.urlopen(VLLM_METRICS_URL, timeout=10) as r:
                text = r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError):
            return {}
        out = {}
        for line in text.splitlines():
            if not line.startswith("vllm:"):
                continue
            m = re.match(r"(vllm:[A-Za-z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        return out

    def health(self) -> str:
        m = self.metrics()
        busy = (m.get("vllm:num_requests_running", 0)
                + m.get("vllm:num_requests_waiting", 0))
        if busy:
            return (f"down: server already has {int(busy)} request(s) in "
                    f"flight, so another client is using it. Stop it first.")
        return super().health()


def prefix_hit_rate(before: dict, after: dict):
    """Hit rate over one scenario, from the counter deltas. None if absent."""
    def pick(m, *names):
        for n in names:
            if n in m:
                return m[n]
        return None
    q0 = pick(before, "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
    q1 = pick(after,  "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
    h0 = pick(before, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
    h1 = pick(after,  "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
    if None in (q0, q1, h0, h1) or q1 - q0 <= 0:
        return None
    return round((h1 - h0) / (q1 - q0), 3)


# Model pairs. The point of the fp16 profile is that both servers run the
# same weights at the same precision, so a quality difference is the runtime
# and not the quantisation. Ollama's default qwen2.5 tags are 4-bit, and
# comparing 4-bit output against fp16 output measures the quantiser.
#
# The as-deployed profile keeps Ollama on the 4-bit tag it would really run,
# which answers a different and equally fair question: what would actually be
# experienced, rather than which runtime is faster at equal precision.
PROFILES = {
    "matched-fp16": {
        "ollama": "qwen2.5:1.5b-instruct-fp16",      # 3.1 GB
        "vllm":   "Qwen/Qwen2.5-1.5B-Instruct",
        "note":   "same weights, same precision, both fit in 6 GB",
    },
    "matched-fp16-small": {
        "ollama": "qwen2.5:0.5b-instruct-fp16",      # 994 MB
        "vllm":   "Qwen/Qwen2.5-0.5B-Instruct",
        "note":   "the base Abhinav's LoRA adapter was trained on",
    },
    # The project's real generator size. Neither side can hold 7B at fp16 in
    # 6 GB, so both run 4-bit, in different formats: Ollama's GGUF Q4_K_M
    # (which already spills 4 of 29 layers to the CPU on this card) and
    # vLLM's AWQ, which needs --cpu-offload-gb to fit at all. Read it as
    # "7B on this laptop", not as a clean runtime comparison.
    "laptop-7b": {
        "ollama": "qwen2.5:7b-instruct",             # Q4_K_M, 4.7 GB
        "vllm":   "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "note":   "7B at 4-bit on both (Q4_K_M vs AWQ), partly offloaded to CPU on both",
    },
    # The lab GPU box (48 GB). 7B at fp16 on both: same weights, same
    # precision, everything on the card. This is the comparison the laptop
    # could not make.
    "gpu-7b-fp16": {
        "ollama": "qwen2.5:7b-instruct-fp16",        # 15 GB
        "vllm":   "Qwen/Qwen2.5-7B-Instruct",
        "note":   "7B, same weights, fp16 on both, fully on a 48 GB card (shared box)",
    },
    "as-deployed": {
        "ollama": "qwen2.5:1.5b-instruct",           # 4-bit
        "vllm":   "Qwen/Qwen2.5-1.5B-Instruct",      # fp16
        "note":   "DIFFERENT precision on each side, latency only, not quality",
    },
}


def build(name: str, profile: str, num_ctx: int = 8192) -> Backend:
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; "
                         f"have {', '.join(PROFILES)}")
    spec = PROFILES[profile]
    if name == "ollama":
        return Ollama(spec["ollama"], num_ctx)
    if name == "vllm":
        return VLLM(spec["vllm"], num_ctx)
    raise SystemExit(f"unknown backend {name!r}; have ollama, vllm")
