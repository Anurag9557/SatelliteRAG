"""
A check, in code, that the identifiers in an answer came from its passages.

The failure that matters most in operations is a made-up command: an
answer that reads like a procedure and carries a CID that does not exist.
A prompt can make that rarer but cannot rule it out. This can, for the
identifiers it recognises: anything shaped like a command or telemetry ID
(letters then digits, 0x hex, or any value written after "CID") that does
not appear in the passages the model was given cannot have come from them,
so it is removed from the answer before anyone reads it, with a note.

It does not catch a made-up name with no digits in it, and it does not
check that a real ID is used in the right place. It is a floor under the
prompt, not a faithfulness score.

Standard library only, so the benchmark can load it without the project's
heavy imports.
"""

import re

# OBCS2370, DTGS0064, AOE00115, OBC0C072: three or more capitals, then at
# least three digits somewhere. Also hex literals.
ID_RE = re.compile(r"\b(?:[A-Z]{3,}[A-Z0-9]*\d{3,}[A-Z0-9]*|0x[0-9A-Fa-f]{2,})\b")

# A value written straight after "CID", in whatever shape the model chose
# ("CID code 1234", "CID-0123", "CID: X12"). It must contain a digit, so
# "CID codes" in a sentence is not taken for a value.
CID_VALUE_RE = re.compile(
    r"\bCID(?:\s+code)?\s*[:=-]?\s*`?((?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{2,})",
    re.I)

PLACEHOLDER = "[ID not in documents]"

# The heading of a procedure table in the FCP document: "OBC- 5215: TELECOMMAND
# LINK TESTS", followed by a PROCEDURE: section. An index row instead reads
# "FCP 5215 - OBC - TELECOMMAND LINK TESTS" and carries no steps.
BODY_HEAD_RE = re.compile(r"\b[A-Z_]+-\s*(\d{4}):")
INDEX_RE = re.compile(r"^FCP (\d{4}) - ", re.M)
NO_STEPS_LINE = "The steps of this procedure are not in the available documents."


def procedure_tables(context: str) -> list:
    """FCP numbers whose step tables are in the context."""
    if "PROCEDURE:" not in context:
        return []
    return sorted(set(BODY_HEAD_RE.findall(context)))


# The labelled parts of a procedure table besides its steps, in the order
# the document prints them. NOTES comes after the steps, often before a
# page break that repeats the table header, so it runs only to the next
# header, page footer or end of the procedure.
SOURCE_RE  = re.compile(r"^\[Source (\d+) \|[^\n]*\]\n", re.M)
SECTIONS   = (("APPLICABILITY", "Applicability"), ("PRE-REQUISITES", "Pre-requisites"),
              ("CAUTION", "Caution"), ("NOTES", "Notes"))
SECTION_END_RE = re.compile(
    r"\n\s*(?:APPLICABILITY:|PRE-REQUISITES:|CAUTION:|PROCEDURE:|NOTES:|END OF PROCEDURE"
    r"|Evt\s*\n\s*Stp|ADITYA-L1 Flight Control Procedures)")


def procedure_conditions(context: str) -> list:
    """
    For each procedure table in the context: its source number, FCP, the
    identifiers in it, and its applicability, pre-requisites, caution and
    notes, as written. Empty parts ("-") are left out.
    """
    out = []
    starts = list(SOURCE_RE.finditer(context))
    for k, m in enumerate(starts):
        block = context[m.end(): starts[k + 1].start() if k + 1 < len(starts) else len(context)]
        head = BODY_HEAD_RE.search(block)
        if not head or "PROCEDURE:" not in block:
            continue
        parts = {}
        for label, name in SECTIONS:
            at = block.find(label + ":")
            if at < 0:
                continue
            body = block[at + len(label) + 1:]
            end = SECTION_END_RE.search(body)
            text = " ".join((body[:end.start()] if end else body).split())
            if text.strip("- "):
                parts[name] = text
        out.append({"source": int(m.group(1)), "fcp": head.group(1),
                    "ids": set(ID_RE.findall(block)), "parts": parts})
    return out


INDEX_LINE_RE = re.compile(r"^FCP \d{4} - .+$", re.M)


def name_the_procedure(answer: str, context: str) -> tuple:
    """
    (answer, index line added or ""). A no-steps reply that does not say
    which procedure it is about gets that procedure's index line in front.

    The note asks for "its index line followed by" the no-steps sentence.
    With v9 every such reply had it; with v10 and v11, 3 to 4 of 43 gave the
    sentence alone, citing the index line without quoting it. The line to
    use is the one in the source the reply cites; failing that, the first
    index line in the context, which retrieval ranked highest.
    """
    at = answer.find(NO_STEPS_LINE)
    if at < 0 or fcp_numbers_named(answer[:at], context):
        return answer, ""
    starts = list(SOURCE_RE.finditer(context))
    blocks = {int(m.group(1)): context[m.end(): starts[k + 1].start() if k + 1 < len(starts) else len(context)]
              for k, m in enumerate(starts)}
    cited = [int(n) for n in re.findall(r"\[Source (\d+)\]", answer[at:])]
    for n in cited + sorted(blocks):
        line = INDEX_LINE_RE.search(blocks.get(n, ""))
        if line:
            return f"{line.group(0).strip()} [Source {n}]\n\n{answer}", line.group(0).strip()
    return answer, ""


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def add_conditions(answer: str, context: str) -> tuple:
    """
    (answer, FCP numbers given conditions). When an answer copies the steps
    of a procedure table in the context, add that procedure's
    applicability, pre-requisites and caution before it and its notes
    after it, copied as written and cited, unless the answer already has
    them.

    Prompts v8 and v9 asked the model for these with the steps. Read on 17
    November, 5 answers copied a step table with every command right, and
    one added the applicability and none the caution: the demo procedure
    lost "NOT a nominal operation". The parts are labelled in the table,
    so the code can copy them, and a caution is not something to leave to
    a model's judgement.

    An answer counts as using a table when it carries two of its
    identifiers (a command id and its telemetry, or two commands).
    """
    plain = _plain(answer)
    before, after, given = [], [], []
    for proc in procedure_conditions(context):
        if len([i for i in proc["ids"] if i in answer]) < 2:
            continue
        missing = {name: text for name, text in proc["parts"].items()
                   if _plain(text)[:60] not in plain}
        if not missing:
            continue
        cite = f"[Source {proc['source']}]"
        lead = [f"{name}: {missing[name]}" for name in ("Applicability", "Pre-requisites", "Caution")
                if name in missing]
        if lead:
            before.append(f"FCP {proc['fcp']} {cite}\n" + "\n".join(lead))
        if "Notes" in missing:
            after.append(f"Notes, FCP {proc['fcp']}: {missing['Notes']} {cite}")
        given.append(proc["fcp"])
    if not given:
        return answer, []
    text = "\n\n".join(before + [answer.strip()] + after)
    return text, given


def context_note(context: str, related: str = "") -> str:
    """
    One line, added after the question, saying which procedures the
    passages hold steps for.

    related: what to say about a procedure that is only an index line,
    beyond the fixed reply. Read on 17 November, 6 of 11 such replies were
    the fixed line alone while the passages held the operation the
    procedure carries out, or its commands in another document.
        ""       the fixed reply only (v7 and before)
        "any"    give what other passages say about it (v8). Tried and
                 not kept: twice it gave another FCP's step table as the
                 steps asked for.
        "named"  the same, but never another FCP's table (v9)

    The documents released so far hold steps for 6 procedures; the other
    ~220 appear only as index rows. Whether a passage set holds a step
    table is a fact the code can read off the text, and the model gets it
    wrong when asked to judge it (prompts v3 to v5, 31 October: it either
    refused answerable questions or said a table that was there was not).
    So the code decides, and tells the model.
    """
    tables = procedure_tables(context)
    indexed = sorted(set(INDEX_RE.findall(context)) - set(tables))
    parts = []
    if tables:
        parts.append("The context contains the procedure table (numbered steps) "
                     f"for FCP {', '.join(tables)}. For a question about "
                     f"{'that procedure' if len(tables) == 1 else 'these procedures'}, "
                     "list the steps from the table.")
    if indexed:
        lead = "Other FCPs in the context" if tables else "The context has no procedure tables. FCPs in it"
        parts.append(f"{lead} appear as index lines only. If the question asks "
                     "for the steps of one of those, reply with its index line "
                     f'followed by: "{NO_STEPS_LINE}" Do not write steps for it.')
        if related == "any":
            parts.append("Then, if other passages describe that procedure, the "
                         "operation it carries out, or commands for it, give "
                         "what they say, with their source numbers. If none "
                         "does, stop after that sentence.")
        elif related == "named":
            others = f" (FCP {', '.join(tables)})" if tables else ""
            parts.append("Then, if a passage that is not a procedure table names "
                         "that FCP or describes the operation it carries out, "
                         "give what that passage says, with its source number. "
                         f"Never give the steps of a different FCP{others} for it. "
                         "If no passage does, stop after that sentence.")
    return ("Note: " + " ".join(parts)) if parts else ""


def fcp_numbers_named(text: str, context: str) -> list:
    """
    The FCP numbers text mentions, in order, counting only numbers that
    are an FCP in the context (a step table or an index line). A year or a
    value in the answer is then never taken for a procedure number, and
    nothing depends on how a satellite numbers its procedures.
    """
    known = set(procedure_tables(context)) | set(INDEX_RE.findall(context))
    return [n for n in re.findall(r"(?<!\d)(\d{4})(?!\d)", text) if n in known]


def only_no_steps(text: str) -> bool:
    """
    The fixed reply and nothing more: the no-steps sentence, with at most
    the procedure's index line before it. It states nothing to cite.
    """
    if not NO_STEPS_RE.search(text):
        return False
    rest = [ln for ln in NO_STEPS_RE.sub("\n", text).splitlines() if ln.strip()]
    return len(rest) <= 1 and sum(len(ln) for ln in rest) < 160


NO_STEPS_RE   = re.compile(r"[ \t]*" + re.escape(NO_STEPS_LINE)
                           .replace(r"\ ", r"\s+") + r"[ \t]*\n?", re.I)


def drop_false_no_steps(answer: str, context: str) -> tuple:
    """
    (answer, how many times the no-steps sentence was taken out).

    context_note() tells the model to follow an index-only FCP with the
    sentence saying its steps are not in the documents. Asked about one
    procedure whose table is present and shown another that is only an
    index line, the model sometimes lists the first one's steps and then
    writes that sentence under them, without the other's index line. The
    answer then contradicts itself.

    Which FCPs have a table in the passages is a fact, so the sentence is
    checked against it: it belongs to the nearest FCP number written before
    it, and if that FCP's table is in the passages the sentence is wrong
    and is removed. With no FCP number before it, it is left alone.
    """
    tables = set(procedure_tables(context))
    if not tables:
        return answer, 0
    out, removed, last = [], 0, 0
    for match in NO_STEPS_RE.finditer(answer):
        named = fcp_numbers_named(answer[:match.start()], context)
        if named and named[-1] in tables:
            out.append(answer[last:match.start()])
            last = match.end()
            removed += 1
    out.append(answer[last:])
    return "".join(out), removed


def _in_context(token: str, context_upper: str) -> bool:
    pattern = r"(?<![A-Z0-9])" + re.escape(token.upper()) + r"(?![A-Z0-9])"
    return re.search(pattern, context_upper) is not None


def unverified_ids(answer: str, context: str) -> list:
    """Identifiers in the answer that appear nowhere in the context."""
    found = set(ID_RE.findall(answer)) | set(CID_VALUE_RE.findall(answer))
    upper = context.upper()
    return sorted(t for t in found if not _in_context(t, upper))


def guard(answer: str, context: str) -> tuple:
    """(answer with unverified IDs replaced, list of the IDs removed)."""
    bad = unverified_ids(answer, context)
    if not bad:
        return answer, []
    out = answer
    for token in sorted(bad, key=len, reverse=True):
        out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])",
                     PLACEHOLDER, out)
    note = (f"\n\nNote: {len(bad)} command or telemetry ID(s) in this answer "
            f"did not appear in the retrieved documents and were removed. "
            f"Do not act on the steps that carried them.")
    return out + note, bad
