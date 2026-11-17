"""
Answer generation.

Retrieval finds the passages; this turns them into a written answer with
citations, using a local text model. Nothing leaves the machine. The model
is reached through core/llm.py, so it is Ollama by default and a local
vLLM server with SATRAG_LLM=vllm.

The text model is deliberately not the vision model used for captioning.
qwen2.5vl can answer from text, but it lists procedure steps poorly and
is heavier on the GPU, where it intermittently crashes the llama-server
process. A dedicated instruct model is both better at this and steadier.

Two kinds of question never reach the model, because the answer can be
read straight off what was retrieved:

    a figure     answered from its caption. A text model asked to "show
                 the recovery diagram" tends to reply that it cannot, even
                 though the figure was retrieved and is about to be shown.
    a log lookup "what was sent on <date>" returns every record for that
                 day. The model was given as many as fitted, took most of
                 a minute, and could only describe the part it saw. The
                 records are counted and grouped in code instead, which is
                 complete and immediate.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from . import grounding, llm, retrieve, router

DEFAULT_MODEL = "qwen2.5:7b-instruct"
NUM_CTX       = 8192

# A cap on the answer, which there was not before. Greedy decoding with no
# repetition penalty occasionally loops on a numbered procedure (14 of 100
# benchmark answers ran to a 512-token cap that way), and without a cap a
# loop runs until the context window is full.
MAX_TOKENS    = 768

# The literal refusal the model is instructed to give when nothing in the
# retrieved context answers the query. cli.py matches on this to avoid
# printing a "Sources:" list under an answer that cites none of them.
NOT_FOUND_MESSAGE = "Not found in available documents."

# Version 11, 18 November. v10 run on 102 questions: the code step did
# its job (every answer that copied a table carries its caution and
# notes; the demo answer is right), but the two new rules hurt more than
# they helped. "Name another procedure by its FCP number and title only"
# made four index-only replies drop the procedure's own index line, and
# one step table (5204) was described instead of copied; the rule about
# naming changed nothing for pcs. v11 is v9 plus one plain rule about
# mixing tables, with the conditions still added in code.
#
# Version 10, 18 November (tried, not kept). Reading v9's 30 answers (17 November) left two
# faults that a prompt has to handle and one it should not:
#   - obc_on gave FCP 5203's system-reset steps under "For OBC-2" in a
#     switch-on answer. v10: steps from one procedure table only, the one
#     asked about; any other is named by number and title only.
#   - pcs said the command sequencer handles battery thresholds, citing a
#     passage that never names it. v10: say a passage describes something
#     only if it names it.
#   - every step-table answer left out the procedure's caution and most of
#     its conditions, though v8 and v9 asked for them. That is now done in
#     code (grounding.add_conditions, ADD_CONDITIONS below), not asked for.
#
# Version 9, 17 November. v8 tightened, after its run (102 questions):
#   - with "also give the conditions", the model rewrote the demo
#     procedure's steps in its own words and copied none of its 15 command
#     ids. v9 says to copy the steps first, then add the conditions.
#   - asked for "what other passages say", it twice gave another FCP's
#     step table as the steps asked for. v9 forbids that by number.
#   - answers ending in the not-found line after a real answer rose from 7
#     to 12. v9 says not to.
# v8 refused 2 instead of 7 and invented no command id, so the direction
# was kept.
#
# Version 8, 17 November (tried, not kept). Version 7 plus one rule, and a
# longer note from the code. 30 v7 answers were read against their passages that day: 13
# right, 7 partly, 3 wrong, 7 over-refused, and no invented command. The
# faults were mostly of saying too little:
#   - the 5 answers that copied a step table had the commands right and
#     all left out when the procedure applies, what must be on first, and
#     its cautions. The new rule asks for those with the steps.
#   - 6 of 11 fixed "steps are not in the documents" replies gave nothing
#     else, while other passages held what the procedure does or its
#     commands. grounding.context_note() now asks for that, cited, after
#     the fixed line (NOTE_RELATED).
#
# Version 7, 17 November. Version 6 plus one rule: say which source each
# statement came from. With v6, 51 of 91 answers named a document, page or
# FCP somewhere and the rest named nothing, so a reader could not tell
# which of five passages an answer rested on. The passages are already
# numbered in the context ("[Source 2 | MOR | p.41 | ...]"), so the model
# is asked to use those numbers and check_citations() below verifies them.
#
# Version 6, 31 October. The procedure document holds steps for only 6
# procedures; the other ~220 were not released and exist as one index line
# each. So for most procedure questions the right answer is "this FCP, and
# its steps are not in the documents". The original prompt (v1, below)
# often refused instead, or wrote steps and command IDs found in no
# passage. Prompts that asked the model to tell a step table from an index
# line (v2 to v5, kept in bench/prompts/) either refused answerable
# questions or said a table that was there was not. So the code decides:
# core/grounding.context_note() reads which FCPs have step tables in the
# passages and says so after the question, and the prompt is v1 plus one
# rule. Results: bench/results-prompt-gpu/ (7B fp16, lab GPU).
PROMPT_VERSION = "v11-one-table"

# What the note after the question asks for about an index-only procedure
# (grounding.context_note): "" for v7 and before, "any" for v8, "named"
# for v9 to v11. To go back to v7, set this to "" and SYSTEM_PROMPT below
# to SYSTEM_PROMPT_V7.
NOTE_RELATED = "named"

# Copy a used procedure's applicability, pre-requisites, caution and notes
# into the answer in code (grounding.add_conditions). On from v10; off
# reproduces v9 and earlier exactly.
ADD_CONDITIONS = True

_HEAD = """You are a spacecraft operations assistant for mission control.
Answer using ONLY the provided context from official mission documents.

Rules:
- For procedures: cite the FCP number and list the steps explicitly,
  including command mnemonics and CID codes exactly as written."""

_COPY = """
- Copy every command mnemonic, CID code and value from the context.
  If you cannot copy it from the context, leave it out."""

_WHOLE = """
- With the steps of a procedure, also give what the context says about
  when it is used, what must be on or done first, and its cautions and
  notes."""

_STEPS_FIRST = """
- For a procedure with steps: first copy its steps from the table, with
  every command mnemonic and CID exactly as written. Then add, from the
  same procedure, when it is used, what must be on or done first, and its
  cautions and notes."""

_CITE = """
- After each statement, give the source it came from in square brackets,
  like [Source 2]. Use only the source numbers shown in the context."""

_TAIL = f"""
- For figures: name the source document and page or section.
- You may combine multiple facts from the context to answer a question
  (comparisons, differences, summaries, lists, counts, etc.), as long as
  every fact used is individually stated somewhere in the context.
  Combining grounded facts is not guessing.
- Only say "{NOT_FOUND_MESSAGE}" for the specific part of the question
  the context genuinely does not address, not for a question the
  context answers once its stated facts are combined.
- Never speculate about spacecraft operations beyond what the documents
  state.
- Keep answers concise and operationally precise."""

_END = f"""
- Do not end an answer with "{NOT_FOUND_MESSAGE}" when you have answered."""

_ONE_PROCEDURE = """
- Give steps from one procedure table only: the one the question asks
  about. Never put another procedure's steps under it. If another
  procedure is relevant, name it by its FCP number and title only."""

_NAMED = """
- Say that a passage describes something only if that passage names it."""

_ONE_TABLE = """
- If the context holds step tables for more than one procedure, give the
  steps of the procedure the question asks about, not another's."""

SYSTEM_PROMPT_V11 = _HEAD + _COPY + _STEPS_FIRST + _ONE_TABLE + _CITE + _TAIL + _END
SYSTEM_PROMPT_V10 = (_HEAD + _COPY + _ONE_PROCEDURE + _STEPS_FIRST + _CITE
                     + _NAMED + _TAIL + _END)
SYSTEM_PROMPT_V9 = _HEAD + _COPY + _STEPS_FIRST + _CITE + _TAIL + _END
SYSTEM_PROMPT_V8 = _HEAD + _COPY + _WHOLE + _CITE + _TAIL
SYSTEM_PROMPT_V7 = _HEAD + _COPY + _CITE + _TAIL
SYSTEM_PROMPT    = SYSTEM_PROMPT_V11
SYSTEM_PROMPT_V6 = _HEAD + _COPY + _TAIL
SYSTEM_PROMPT_V1 = _HEAD + _TAIL


@dataclass
class Answer:
    """
    A reply, and how it was arrived at.

    path         model, caption, commands, none (nothing retrieved) or
                 failed (the model could not be reached)
    removed_ids  identifiers the grounding guard took out
    cited        the source numbers the reply names
    context      what the model was given; for a caption reply, the
                 caption and its image file, which is what the reply has
                 to be checked against; empty for the other paths
    """
    text:        str
    path:        str = "model"
    removed_ids: list = field(default_factory=list)
    cited:       list = field(default_factory=list)
    context:     str = ""


def is_refusal(text: str) -> bool:
    """
    The not-found sentence and nothing else. Citations after it do not
    count: with prompt v7 the model sometimes cites the passages it found
    nothing in, and that is still a refusal.
    """
    bare = CITATION.sub("", text).strip().rstrip(".").strip()
    return bare == NOT_FOUND_MESSAGE.rstrip(".")


# Figures

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


def caption_source(hit) -> str:
    """
    What a caption reply rests on. The caption was written by the vision
    model from the image, so the reply is only as right as the caption,
    and the image is what a reader checks it against.
    """
    image = (hit.metadata or {}).get("image_file", "")
    return (f"[Source 1] {hit.cite()}\n"
            f"Image file: {image or 'not recorded'}\n\n"
            f"Caption:\n{hit.text.strip()}")


# The command log

# How many distinct commands the summary lists before saying "and N more".
COMMANDS_LISTED = 40


def is_log_lookup(hits: list, plan) -> bool:
    """
    True when the hits are every record matching a date, command id or
    mnemonic, as opposed to the few records most like the question.

    A date or command id only becomes a filter if the log holds it, so
    when every hit carries the value the plan asked for, the set is the
    complete one. A mnemonic in the plan is only a candidate; it counts
    here if all the hits share it.
    """
    if not hits or any(h.content_type != "command_log" for h in hits):
        return False
    wanted = {k: getattr(plan, k) for k in ("date", "cid", "mnemonic")
              if getattr(plan, k, None)}
    if not wanted:
        return False
    for key, value in wanted.items():
        same = all(str(h.metadata.get(key, "")).upper() == value.upper()
                   for h in hits)
        if not same and (key != "mnemonic" or len(wanted) == 1):
            return False
    return True


def _spread(counts: Counter) -> str:
    return ", ".join(f"{value or 'not recorded'} ({n})"
                     for value, n in counts.most_common())


def answer_from_commands(hits: list, plan, query: str = "") -> str:
    """
    A reply counted off the records themselves: how many, over what
    period, with what status, between which stations, and which commands.
    Every figure in it is a count, so none of it can be invented.
    """
    metas = [h.metadata or {} for h in hits]
    total = len(metas)
    asked = [f"{label} {getattr(plan, key)}"
             for label, key in (("date", "date"), ("CID", "cid"),
                                ("mnemonic", "mnemonic"))
             if getattr(plan, key, None)
             and all(str(m.get(key, "")).upper() == getattr(plan, key).upper()
                     for m in metas)]
    lines = [f"{total} command record{'s' if total != 1 else ''} in the "
             f"telecommand log for {', '.join(asked)}."]

    stamps = sorted(f"{m.get('date', '')} {m.get('time', '')}".strip()
                    for m in metas)
    if total > 1 and stamps[0] != stamps[-1]:
        lines.append(f"First at {stamps[0]}, last at {stamps[-1]}.")
    lines.append(f"Status: {_spread(Counter(m.get('status', '') for m in metas))}.")
    lines.append("Uplinked from " + _spread(Counter(
        f"{m.get('source_station', '') or '?'} to "
        f"{m.get('dest_station', '') or '?'}" for m in metas)) + ".")
    orbits = Counter(m.get("orbit_no", "") for m in metas)
    if any(orbits):
        lines.append(f"Orbit: {_spread(orbits)}.")

    # "OBC commands on <date>" is the day's records narrowed to one
    # subsystem. The narrowing is shown, not applied silently: the count
    # for the whole day stays above it.
    shown = metas
    subsystem = router.find_subsystem(query) if query else None
    if subsystem:
        mine = [m for m in metas if m.get("subsystem") == subsystem]
        if mine and len(mine) < total:
            lines.append(f"{len(mine)} of the {total} are {subsystem} "
                         f"commands, listed here.")
            shown = mine
        elif not mine:
            lines.append(f"None of the {total} is tagged as a {subsystem} "
                         f"command.")

    by_command = {}
    for m in shown:
        key = (m.get("mnemonic", ""), m.get("cid", ""))
        by_command.setdefault(key, []).append(m.get("time", ""))
    lines.append("")
    lines.append("By command:")
    ordered = sorted(by_command.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for (mnemonic, cid), times in ordered[:COMMANDS_LISTED]:
        times = sorted(t for t in times if t)
        when  = ""
        if times:
            when = (f", at {times[0]}" if len(times) == 1
                    else f", {times[0]} to {times[-1]}")
        lines.append(f"  {mnemonic or '?'} (CID {cid or '?'}): "
                     f"{len(by_command[(mnemonic, cid)])} time"
                     f"{'s' if len(by_command[(mnemonic, cid)]) != 1 else ''}{when}")
    if len(ordered) > COMMANDS_LISTED:
        lines.append(f"  and {len(ordered) - COMMANDS_LISTED} more commands")

    lines.append("")
    lines.append("Every matching record is listed in the sources below.")
    return "\n".join(lines)


# Citations

CITATION     = re.compile(r"\[\s*Sources?\s+([\d,\s and]+?)\]", re.I)
SOURCE_NUM   = re.compile(r"\bSources?\s+(\d+)", re.I)
UNCITED_NOTE = ("Note: this answer does not say which source each statement "
                "came from. Check it against the sources listed below.")


def drop_trailing_not_found(text: str) -> tuple:
    """
    (text, whether a line was taken off). The not-found sentence as the
    last paragraph of an answer that has already answered is taken off.

    The prompt allows the sentence for the part of a question the passages
    do not address, but on its own at the end it does not say which part,
    and to a reader it contradicts the answer above it. With prompts v8
    and v9, 12 and 14 of 88 answers ended that way, 13 of the 14 as a
    paragraph alone. A sentence that names what is missing is kept, as is
    a reply that is only the refusal.
    """
    paras = text.rstrip().split("\n\n")
    if len(paras) < 2 or not is_refusal(paras[-1]):
        return text, False
    rest = "\n\n".join(paras[:-1]).rstrip()
    if not rest or is_refusal(rest):
        return text, False
    return rest, True


def check_citations(text: str, n_sources: int) -> tuple:
    """
    (text, the source numbers it cites).

    A citation to a source number that was never in the context is taken
    out: it points at nothing. An answer that makes statements and cites
    nothing gets a line saying so, so the reader knows to check it. A
    refusal is left alone, as is the fixed reply for an FCP whose steps
    are not in the documents; neither asserts anything to cite. That reply
    followed by what other passages say does, and is checked.
    """
    def keep(match):
        good = [n for n in re.findall(r"\d+", match.group(1))
                if 1 <= int(n) <= n_sources]
        return f"[Source {', '.join(good)}]" if good else ""

    # A refusal that cites sources reads as if those sources were checked
    # and came up empty, which is all it means; the sources are listed
    # below the answer anyway.
    if is_refusal(text):
        return NOT_FOUND_MESSAGE, []

    text  = CITATION.sub(keep, text)
    text  = re.sub(r"[ \t]+([.,;])", r"\1", text)
    cited = sorted({int(n) for n in SOURCE_NUM.findall(text)
                    if 1 <= int(n) <= n_sources})
    states_nothing = is_refusal(text) or grounding.only_no_steps(text)
    if not cited and not states_nothing and UNCITED_NOTE not in text:
        text = f"{text.rstrip()}\n\n{UNCITED_NOTE}"
    return text, cited


# The model

def answer_without_model(hits: list, plan, query: str = "") -> tuple:
    """
    (reply, path) for the questions answered in code, else ("", "").

    The app and the browser demo both ask this first, so a question the
    app answers without a model is never sent to one in the demo either.
    """
    if not hits:
        return "No relevant documents found.", "none"
    if plan.intent == "figure":
        reply = answer_from_caption(hits)
        if reply:
            return reply, "caption"
    if is_log_lookup(hits, plan):
        return answer_from_commands(hits, plan, query), "commands"
    return "", ""


def answer(query: str, hits: list, model: str = DEFAULT_MODEL,
           show_satellite: bool = False, plan=None) -> Answer:
    """Turn retrieved passages into a cited answer, with how it was made."""
    if not hits:
        return Answer("No relevant documents found.", path="none")

    plan = plan or router.rule_plan(query)
    reply, path = answer_without_model(hits, plan, query)
    if path == "caption":
        return Answer(reply, path=path, cited=[1],
                      context=caption_source(hits[0]))
    if path:
        return Answer(reply, path=path)

    context = retrieve.build_context(hits, show_satellite=show_satellite)
    user = f"Context:\n{context}\n\nQuestion: {query}"
    note = grounding.context_note(context, related=NOTE_RELATED)
    if note:
        user += f"\n\n{note}"

    try:
        reply = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            model      = model,
            num_ctx    = NUM_CTX,
            max_tokens = MAX_TOKENS,
        )
    except Exception as exc:
        # A transient Ollama or GPU error should not lose the retrieval.
        return Answer(
            f"[Generation unavailable: {type(exc).__name__}] "
            f"Retrieval succeeded but the local model did not produce an "
            f"answer. This is usually transient and re-running works. "
            f"The most relevant sources are listed below.",
            path="failed", context=context)

    # No identifier the passages do not contain reaches the reader, and no
    # "steps not in the documents" under a procedure whose steps are.
    reply, removed = grounding.guard(reply, context)
    reply, _       = grounding.drop_false_no_steps(reply, context)
    reply, _       = drop_trailing_not_found(reply)
    if ADD_CONDITIONS:
        reply, _   = grounding.add_conditions(reply, context)
        reply, _   = grounding.name_the_procedure(reply, context)
    reply, cited   = check_citations(reply, len(hits))
    return Answer(reply, removed_ids=removed, cited=cited, context=context)


def generate(query: str, hits: list, model: str = DEFAULT_MODEL,
             show_satellite: bool = False, plan=None) -> str:
    """The reply text alone. answer() returns it with how it was made."""
    return answer(query, hits, model=model, show_satellite=show_satellite,
                  plan=plan).text


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
