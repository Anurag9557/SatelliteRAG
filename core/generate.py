"""
Answer generation.

Retrieval finds the passages; this turns them into a written answer with
citations, using a local Ollama text model. Nothing leaves the machine.

The text model is deliberately not the vision model used for captioning.
qwen2.5vl can answer from text, but it lists procedure steps poorly and
is heavier on the GPU, where it intermittently crashes the llama-server
process. A dedicated instruct model is both better at this and steadier.

A diagram question is answered from the caption directly rather than
through the model: a text model asked to "show the recovery diagram"
tends to reply that it cannot, even though the figure was retrieved and
is about to be displayed.
"""

import re

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from . import retrieve, router

DEFAULT_MODEL = "qwen2.5:7b-instruct"
NUM_CTX       = 8192

SYSTEM_PROMPT = """You are a spacecraft operations assistant for mission control.
Answer using ONLY the provided context from official mission documents.

Rules:
- For procedures: cite the FCP number and list the steps explicitly,
  including command mnemonics and CID codes exactly as written.
- For figures: name the diagram and its source document and page.
- If the answer is not in the context, say "Not found in available
  documents." Do not guess and do not fill gaps from general knowledge.
- Never speculate about spacecraft operations beyond what the documents
  state.
- Say which document and section each claim comes from.
- Keep answers concise and operationally precise."""


def answer_from_caption(hits: list) -> str:
    """
    Compose a reply directly from the top-ranked hit when it is a figure
    caption. A caption further down the list is not what the question is
    about, so it is not answered from.
    """
    if not hits or hits[0].content_type != "image_caption":
        return ""
    best = hits[0]
    meta = best.metadata or {}
    where = meta.get("doc_type") or "the documents"
    if meta.get("page"):
        where += f", page {meta['page']}"
    return (f"Here is the figure matching your request (from {where}):\n\n"
            f"{best.text.strip()}\n\n"
            f"The image file is listed in the sources below.")


def generate(query: str, hits: list, model: str = DEFAULT_MODEL,
             show_satellite: bool = False, plan=None) -> str:
    """Turn retrieved passages into a cited answer."""
    if not hits:
        return "No relevant documents found."

    # A figure request is answered from the caption, not the text model.
    plan = plan or router.rule_plan(query)
    if plan.intent == "figure":
        reply = answer_from_caption(hits)
        if reply:
            return reply

    if not HAS_OLLAMA:
        return ("[Generation unavailable] ollama package not installed. "
                "Run: pip install ollama\n"
                "The retrieved sources are listed below.")

    context = retrieve.build_context(hits, show_satellite=show_satellite)

    try:
        reply = ollama.chat(
            model    = model,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            options  = {"num_ctx": NUM_CTX},
        )
        return reply["message"]["content"].strip()
    except Exception as exc:
        # A transient Ollama or GPU error should not lose the retrieval.
        return (f"[Generation unavailable: {type(exc).__name__}] "
                f"Retrieval succeeded but the local model did not produce an "
                f"answer. This is usually transient and re-running works. "
                f"The most relevant sources are listed below.")


def ask(satellites: list, query: str, top_k: int = 5, doc_types=None,
        subsystem=None, model: str = DEFAULT_MODEL, plan=None) -> tuple:
    """
    Route, retrieve, then answer. Returns (answer, hits).

    The question is routed once and the same plan drives retrieval and
    generation. Pass a plan to reuse one the caller already made.
    """
    plan  = plan or router.plan(query, satellites[0].retrieval)
    hits  = retrieve.search(satellites, query, top_k=top_k,
                            doc_types=doc_types, subsystem=subsystem,
                            plan=plan)
    reply = generate(query, hits, model=model,
                     show_satellite=len(satellites) > 1, plan=plan)
    router.log_plan(query, plan, hits)
    return reply, hits
