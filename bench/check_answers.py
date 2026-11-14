"""
Automatic checks on the answers in quality result files.

Reading every answer against its passages is the real faithfulness score,
and it takes hours. These checks take a second and catch the failures that
matter most in operations, so every run gets them:

    refused        the answer is the not-found sentence and nothing else
    tacked_on      a real answer that still ends in the not-found sentence
    invented_ids   command or telemetry IDs (letters then digits, 0x hex, or
                   any value written after "CID") that appear nowhere in the
                   passages the model was given
    invented_fcps  FCP numbers that appear nowhere in those passages
    title_only     the passages hold FCP index lines but no procedure steps
                   (the documents hold steps for only 6 procedures; the rest
                   were not released)
    wrong_no_steps the answer says an FCP's steps are not in the documents
                   while that FCP's procedure table is in its passages

invented_ids is core/grounding.py's check, the one the app now applies to
every answer before showing it (it removes such IDs, with a note). Counted
here on the raw model output, it shows how often the guard has to step in.
It misses made-up names without digits, so it is a floor, not a full
faithfulness score.

    python bench/check_answers.py bench/results-prompt/v1/*.json bench/results-prompt/v2/*.json
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
import harness  # noqa: E402

# The same check the app runs on every answer (core/grounding.py), loaded
# by path so the bench does not pull in core/__init__ and its heavy imports.
_spec = importlib.util.spec_from_file_location(
    "grounding", BENCH.parent / "core" / "grounding.py")
grounding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grounding)

NOT_FOUND = "Not found in available documents."
FCP_RE = re.compile(r"\bFCP[-\s]?(\d{4})\b", re.I)
BODY_RE = re.compile(r"END OF PROCEDURE|PROCEDURE:|\bEvt\b", re.I)
# The heading of a procedure table, "OBC- 5215: TELECOMMAND LINK TESTS".
BODY_HEAD_RE = re.compile(r"\b[A-Z_]+-\s*(\d{4}):")
NO_STEPS = "steps of this procedure are not in the available documents"


def contexts(slug: str = "aditya-l1") -> dict:
    return {r["id"]: r["context"] for r in harness.load_context(slug)}


def check(row: dict, ctx: str) -> dict:
    text = row["answer"]
    refused, tacked = harness.classify_refusal(text, NOT_FOUND)
    ids = grounding.unverified_ids(text, ctx)
    fcps = sorted({n for n in FCP_RE.findall(text) if n not in ctx})
    has_index = "FCP Index" in ctx
    has_body = bool(BODY_RE.search(ctx))
    bodies = set(BODY_HEAD_RE.findall(ctx))
    wrong_no_steps = NO_STEPS in text.lower() and bool(bodies & set(FCP_RE.findall(text)))
    return {
        "id": row["id"],
        "negative": row.get("is_negative", False),
        "refused": refused,
        "tacked_on": tacked,
        "invented_ids": ids,
        "invented_fcps": fcps,
        "title_only": has_index and not has_body,
        "names_fcp": bool(FCP_RE.search(text)),
        "wrong_no_steps": wrong_no_steps,
    }


def summarise(path: Path, ctxs: dict) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["result"]["rows"] if r.get("ok", True)]
    checks = [check(r, ctxs.get(r["id"], "")) for r in rows]
    pos = [c for c in checks if not c["negative"]]
    neg = [c for c in checks if c["negative"]]
    title = [c for c in pos if c["title_only"]]
    return {
        "file": f"{path.parent.name}/{path.name}",
        "prompt": data["result"].get("prompt", "?"),
        "answerable": len(pos),
        "refused": sum(c["refused"] for c in pos),
        "tacked_on": sum(c["tacked_on"] for c in pos),
        "answers_with_invented_ids": sum(bool(c["invented_ids"]) for c in pos),
        "answers_with_invented_fcps": sum(bool(c["invented_fcps"]) for c in pos),
        "title_only_questions": len(title),
        "title_only_named_fcp": sum(c["names_fcp"] and not c["refused"] for c in title),
        "title_only_refused": sum(c["refused"] for c in title),
        "wrong_no_steps": sum(c["wrong_no_steps"] for c in pos),
        "out_of_scope_refused": f"{sum(c['refused'] for c in neg)}/{len(neg)}",
        "checks": checks,
    }


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    ctxs = contexts()
    cols = ["answerable", "refused", "tacked_on", "answers_with_invented_ids",
            "answers_with_invented_fcps", "title_only_questions",
            "title_only_named_fcp", "title_only_refused", "wrong_no_steps",
            "out_of_scope_refused"]
    results = [summarise(p, ctxs) for p in paths]
    width = max(len(r["file"]) for r in results)
    print(f"{'':<{width}}  prompt  " + "  ".join(c.replace('answers_with_', '') for c in cols))
    for r in results:
        print(f"{r['file']:<{width}}  {r['prompt']:<6}  " +
              "  ".join(f"{str(r[c]):>{len(c.replace('answers_with_', ''))}}" for c in cols))
    for r in results:
        bad = [c for c in r["checks"] if c["invented_ids"] or c["invented_fcps"]]
        if bad:
            print(f"\n{r['file']}: answers with identifiers not in their passages")
            for c in bad:
                print(f"  {c['id']:<22} ids={len(c['invented_ids'])} fcps={c['invented_fcps']}")


if __name__ == "__main__":
    main()
