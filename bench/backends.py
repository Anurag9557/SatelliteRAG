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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

OLLAMA_URL = "http://localhost:11434/api/chat"
VLLM_URL   = "http://localhost:8000/v1/chat/completions"

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
                "temperature": temperature,
                "seed":        seed,
                "num_ctx":     self.num_ctx,
                "num_predict": max_tokens,
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
            "stream_options": {"include_usage": True},
        }
        if schema is not None:
            payload["structured_outputs"] = {"json": schema}

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
