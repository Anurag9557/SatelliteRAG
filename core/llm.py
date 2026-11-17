"""
One way to call a language model, whichever server is behind it.

generate.py, caption.py and lora/build_dataset.py each used to talk to
Ollama directly, each with its own options. They now call chat() here, and
the backend is one environment variable:

    SATRAG_LLM=ollama   (default) the local Ollama server, OLLAMA_HOST
    SATRAG_LLM=vllm     a vLLM OpenAI-compatible server, SATRAG_VLLM_URL

Both backends get the same sampling settings, sent explicitly on every
request. The benchmark in bench/ found that leaving them to server defaults
is not neutral: vLLM applies the model's generation_config.json (a
repetition penalty of 1.1 for Qwen2.5) unless told otherwise, and Ollama
does not, so the same weights gave different answers for a reason that had
nothing to do with the runtime.

This file uses the standard library only. Both servers are reached over
HTTP, so neither the ollama package nor vLLM has to be installed where it
runs, and it can be loaded on its own by path (lora/build_dataset.py does,
because importing the core package pulls in ChromaDB and PyMuPDF). That
matters because vLLM runs only under Linux, in its own venv with its own
pinned torch.
"""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Kept for callers that used to check for the ollama package. Ollama is
# now reached over its HTTP API, so there is nothing to be missing.
HAS_OLLAMA = True

BACKEND    = os.environ.get("SATRAG_LLM", "ollama").strip().lower()
VLLM_URL   = os.environ.get("SATRAG_VLLM_URL", "http://localhost:8000/v1").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

# Pinned, not left to the server. Temperature 0 and a fixed seed make an
# answer reproducible, which is what lets answer quality be measured at all
# (Ollama reproduced 10/10 questions over 5 repeats with these settings).
# A repetition penalty of 1.0 is "none", the behaviour Ollama already had
# by default, so pinning it changes no current answer; it only stops vLLM
# from quietly applying a different one.
TEMPERATURE        = 0.0
SEED               = 1234
REPETITION_PENALTY = 1.0

# How many requests a batch job keeps in flight. 1 by default because the
# win depends on the server: Ollama only overlaps requests when started
# with OLLAMA_NUM_PARALLEL above 1, and each parallel slot reserves its own
# context, which a 7B model on a 6 GB card may not have room for. Measured
# at 1.5B: NUM_PARALLEL=4 with 8 in flight was 2.8x the shipped default.
CONCURRENCY = int(os.environ.get("SATRAG_LLM_CONCURRENCY", "1"))

TIMEOUT_S = 600

# A request that fails because the server is busy loading a model, or has
# just restarted, usually works a moment later. One more try after a short
# wait covers that. Seen on the laptop: the first question after start-up
# came back as an error after 20 seconds and succeeded when asked again.
# A request the server understood and refused (a missing model, a bad
# payload) is not tried again; asking twice changes nothing.
RETRIES      = 1
RETRY_WAIT_S = 3

# Ollama tags to the Hugging Face ids vLLM serves. Only consulted when the
# vLLM server is serving more than one base model; normally it serves one
# and that one is used. SATRAG_VLLM_MODEL overrides both.
HF_NAMES = {
    "qwen2.5:7b-instruct":        "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:1.5b-instruct":      "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:1.5b-instruct-fp16": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:0.5b-instruct":      "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:0.5b-instruct-fp16": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5vl":                  "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5vl:latest":           "Qwen/Qwen2.5-VL-7B-Instruct",
}


class LLMError(RuntimeError):
    """The server was not reachable, or it refused the request."""

    # True when trying again may help: no answer at all, or a 5xx reply.
    transient = False


# ------------------------------------------------------------------ public


def chat(messages: list, *, model: str, max_tokens: int = 768,
         temperature: float = TEMPERATURE, num_ctx: int = 8192,
         schema=None, images: list = None, adapter: str = None) -> str:
    """
    One chat completion. Returns the reply text.

    schema   "json" for any valid JSON, or a JSON schema dict. A schema is
             enforced by the decoder on both backends, so the reply has the
             shape asked for rather than merely being JSON.
    images   base64-encoded images, attached to the last user turn.
    adapter  a LoRA adapter name. vLLM only: the server must have been
             started with --enable-lora --lora-modules name=path. Ollama
             binds an adapter to a model tag instead, so there pass the
             tag as model and leave this empty.
    num_ctx  Ollama only. vLLM fixes the context once, at server start.
    """
    if BACKEND not in ("ollama", "vllm"):
        raise LLMError(f"SATRAG_LLM={BACKEND!r}; expected ollama or vllm")
    if adapter and BACKEND == "ollama":
        raise LLMError("Ollama cannot take an adapter per request; create a "
                       "model tag with an ADAPTER line and pass it as model")

    for attempt in range(RETRIES + 1):
        try:
            if BACKEND == "vllm":
                return _vllm_chat(messages, model=model, max_tokens=max_tokens,
                                  temperature=temperature, schema=schema,
                                  images=images, adapter=adapter)
            return _ollama_chat(messages, model=model, max_tokens=max_tokens,
                                temperature=temperature, num_ctx=num_ctx,
                                schema=schema, images=images)
        except LLMError as exc:
            if attempt == RETRIES or not exc.transient:
                raise
            time.sleep(RETRY_WAIT_S)


def chat_many(batch: list, *, concurrency: int = None, **kwargs) -> list:
    """
    Many independent chats, in order. Each item of batch is a messages
    list. Returns one entry per item: the reply text, or the exception that
    request raised, so one failure does not lose the rest of a batch job.

    With concurrency above 1 the requests overlap. vLLM batches them on the
    GPU; Ollama does too if its server was started with OLLAMA_NUM_PARALLEL.
    """
    workers = max(1, concurrency or CONCURRENCY)

    def one(messages):
        try:
            return chat(messages, **kwargs)
        except Exception as exc:          # returned, not raised
            return exc

    if workers == 1:
        return [one(m) for m in batch]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, batch))


def available(model: str = None) -> tuple:
    """(ok, reason). Cheap; used before a long batch job starts."""
    if BACKEND == "vllm":
        try:
            served = _vllm_models()
        except LLMError as exc:
            return False, str(exc)
        return bool(served), f"vLLM serving {', '.join(served) or 'nothing'}"
    try:
        listed = _get(OLLAMA_URL + "/api/tags", "Ollama")
    except LLMError as exc:
        return False, str(exc)
    if model is None:
        return True, "Ollama reachable"
    names = {m.get("model") or m.get("name", "")
             for m in listed.get("models", [])}
    base  = model.split(":")[0]
    if any(n == model or n.split(":")[0] == base for n in names):
        return True, f"Ollama has {model}"
    return False, f"Ollama is running but has no {model}: ollama pull {model}"


def summary() -> str:
    """One line for logs and result files: what generated this."""
    where = VLLM_URL if BACKEND == "vllm" else OLLAMA_URL
    return (f"{BACKEND} ({where}), temperature={TEMPERATURE}, seed={SEED}, "
            f"repetition_penalty={REPETITION_PENALTY}")


# ------------------------------------------------------------------ ollama


def _ollama_chat(messages, *, model, max_tokens, temperature, num_ctx,
                 schema, images):
    payload = {
        "model":    model,
        "messages": _attach_images(messages, images, style="ollama"),
        "stream":   False,
        "options": {
            "temperature":    temperature,
            "seed":           SEED,
            "repeat_penalty": REPETITION_PENALTY,
            "num_ctx":        num_ctx,
            "num_predict":    max_tokens,
        },
    }
    if schema is not None:
        payload["format"] = schema
    data = _post(OLLAMA_URL + "/api/chat", payload, "Ollama")
    if data.get("error"):
        raise LLMError(f"Ollama: {data['error']}")
    return ((data.get("message") or {}).get("content") or "").strip()


# ------------------------------------------------------------------ vllm


_served_cache = None


def _vllm_models() -> list:
    global _served_cache
    if _served_cache is None:
        data = _get(VLLM_URL + "/models", "vLLM")
        _served_cache = [m["id"] for m in data.get("data", [])]
    return _served_cache


def _vllm_model_name(model: str, adapter: str) -> str:
    """
    The name to put in the request. vLLM addresses a LoRA adapter by the
    name it was registered under, as if it were a model; otherwise the
    served base model is used, whatever Ollama tag the caller passed.
    """
    if adapter:
        return adapter
    override = os.environ.get("SATRAG_VLLM_MODEL")
    if override:
        return override
    served = _vllm_models()
    if model in served:
        return model
    if HF_NAMES.get(model) in served:
        return HF_NAMES[model]
    if served:
        return served[0]
    raise LLMError(f"vLLM at {VLLM_URL} is serving no model")


def _vllm_chat(messages, *, model, max_tokens, temperature, schema,
               images, adapter):
    payload = {
        "model":              _vllm_model_name(model, adapter),
        "messages":           _attach_images(messages, images, style="openai"),
        "max_tokens":         max_tokens,
        "temperature":        temperature,
        "seed":               SEED,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if schema is not None:
        # disable_any_whitespace matters: without it the JSON grammar
        # admits unlimited whitespace, and a small model forced to open
        # with "{" can emit blanks until max_tokens runs out.
        so = {"json_object": True} if schema == "json" else {"json": schema}
        so["disable_any_whitespace"] = True
        payload["structured_outputs"] = so
    data = _post(VLLM_URL + "/chat/completions", payload, "vLLM")
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"unexpected vLLM reply: {str(data)[:200]}")


# ------------------------------------------------------------------ http


def _get(url: str, who: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LLMError(f"cannot reach {who} at {url}: {exc}")


def _post(url: str, payload: dict, who: str) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        body  = exc.read()[:300].decode("utf-8", "replace")
        error = LLMError(f"{who} HTTP {exc.code}: {body}")
        error.transient = exc.code >= 500
        raise error
    except (urllib.error.URLError, OSError) as exc:
        error = LLMError(f"cannot reach {who} at {url}: {exc}")
        error.transient = True
        raise error
    except ValueError as exc:
        raise LLMError(f"{who} at {url} did not reply with JSON: {exc}")


def _attach_images(messages, images, style):
    """Put base64 images on the last user turn, in each server's format."""
    if not images:
        return messages
    out = [dict(m) for m in messages]
    idx = max(i for i, m in enumerate(out) if m["role"] == "user")
    if style == "ollama":
        out[idx]["images"] = list(images)
    else:
        parts = [{"type": "text", "text": out[idx]["content"]}]
        parts += [{"type": "image_url",
                   "image_url": {"url": f"data:image/png;base64,{b64}"}}
                  for b64 in images]
        out[idx]["content"] = parts
    return out
