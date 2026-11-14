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


def context_note(context: str) -> str:
    """
    One line, added after the question, saying which procedures the
    passages hold steps for.

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
    return ("Note: " + " ".join(parts)) if parts else ""


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
