"""
Answer evaluation.

evaluate.py scores retrieval: was a right passage found. This runs every
test question through the whole app and looks at the answer that comes
out. It counts what a script can count:

    refused          the not-found sentence and nothing else
    tacked on        a real answer that still ends in that sentence
    names the FCP    for a question with one right procedure, whether the
                     answer names its number
    wrong FCP        it names a procedure number and not the right one
    cited            it says which source a statement came from
    ids removed      the grounding guard took an identifier out
    said no steps    it says a procedure's steps are not in the documents

    python evaluate_answers.py                    the app as it is
    python evaluate_answers.py --tag v6 --prompt v6
    python evaluate_answers.py --limit 10         a quick look

None of these says an answer is right. For that a person has to read it:
review_answers.py turns a run into a page for marking.

Writes one row of numbers to results/answer_checks.csv. The answers and
the passages they were written from go to runs/answers/<tag>.json, which
holds document text and is kept out of git.
"""

import argparse
import csv
import json
import re
import statistics
import time
from datetime import date
from pathlib import Path

from core import generate, grounding, llm, registry, retrieve, router
from evaluate import relevant
from testset import NEGATIVE_CASES, TEST_SET

RESULTS = Path("results")
RUNS    = Path("runs") / "answers"

FCP_RE    = re.compile(r"\b(?:FCP[-\s]?)?([56]\d{3})\b")
PROMPTS   = {"v9": generate.SYSTEM_PROMPT_V9, "v8": generate.SYSTEM_PROMPT_V8,
             "v7": generate.SYSTEM_PROMPT_V7,
             "v6": generate.SYSTEM_PROMPT_V6, "v1": generate.SYSTEM_PROMPT_V1}
# v8 also changed the note the code adds after the question.
NOTE_RELATED = {"v8": "any", "v9": "named"}


def classify_refusal(text: str) -> tuple:
    """(refused, tacked_on), as bench/harness.py counts them."""
    sentence = generate.NOT_FOUND_MESSAGE.lower()
    low = generate.CITATION.sub(" ", text).lower()
    if sentence not in low:
        return False, False
    rest = re.sub(r"[\s\W_]+", " ", low.replace(sentence, " ")).strip()
    return (True, False) if len(rest) < 20 else (False, True)


SOURCE_HEAD = re.compile(r"(?m)^\[Source \d+ \|[^\n]*\]$")


def table_codes(context: str, number: str) -> set:
    """The command and telemetry ids in one FCP's step table, if present."""
    heads  = list(SOURCE_HEAD.finditer(context))
    codes  = set()
    for i, head in enumerate(heads):
        end   = heads[i + 1].start() if i + 1 < len(heads) else len(context)
        block = context[head.end():end]
        if f"FCP-{number}" in head.group(0) and "PROCEDURE:" in block:
            codes |= set(grounding.ID_RE.findall(block))
    return codes


def checks(text: str, context: str, wanted: list) -> dict:
    """
    What can be counted about one answer.

    Whether a procedure's steps came out is judged by what was copied, not
    by layout: the step tables reach the model one cell per line, so a
    faithful answer is often not a numbered list. Copying two or more of
    the right table's command ids counts as listing its steps.

    The no-steps sentence counts against an answer only when it sits under
    one of the FCPs asked about (the nearest FCP number before it).
    """
    named   = sorted(set(FCP_RE.findall(text)))
    tables  = grounding.procedure_tables(context)
    refused, tacked = classify_refusal(text)
    copied  = set()
    for number in wanted:
        copied |= table_codes(context, number) & set(grounding.ID_RE.findall(text))
    no_steps_for_wanted = False
    for match in grounding.NO_STEPS_RE.finditer(text):
        before = grounding.fcp_numbers_named(text[:match.start()], context)
        if before and before[-1] in wanted:
            no_steps_for_wanted = True
    return {
        "wanted_has_steps": any(w in tables for w in wanted),
        "names_wanted_fcp": bool(wanted) and any(w in named for w in wanted),
        "names_other_fcp":  bool(wanted) and bool(named) and not any(w in named for w in wanted),
        "refused":   refused,
        "tacked_on": tacked,
        "codes_copied": len(copied),
        "no_steps_for_wanted": no_steps_for_wanted,
        "said_no_steps": grounding.NO_STEPS_LINE.lower() in text.lower(),
        "no_steps_and_more": (grounding.NO_STEPS_LINE.lower() in text.lower()
                              and not grounding.only_no_steps(text)),
        "uncited_note": generate.UNCITED_NOTE in text,
        "chars": len(text),
    }


def run_one(sat, case: dict, negative: bool, model: str) -> dict:
    query  = case["q"]
    start  = time.perf_counter()
    plan   = router.plan(query, sat.retrieval)
    hits   = retrieve.search([sat], query, top_k=sat.retrieval["top_k"],
                             plan=plan)
    found  = time.perf_counter()
    reply  = generate.answer(query, hits, model=model, plan=plan)
    done   = time.perf_counter()

    accept = case.get("accept") or []
    wanted = sorted({c["fcp_number"] for c in accept if "fcp_number" in c})
    return {
        "id":        case["id"],
        "question":  query,
        "negative":  negative,
        "intent":    case.get("intent", ""),
        "path":      reply.path,
        "answer":    reply.text,
        "context":   reply.context,
        "sources":   [h.cite() for h in hits],
        "right_in_context": any(relevant(h, accept) for h in hits) if accept else False,
        "wanted_fcp": wanted,
        "cited":     len(reply.cited),
        "ids_removed": len(reply.removed_ids),
        "search_s":  round(found - start, 2),
        "answer_s":  round(done - found, 2),
        **checks(reply.text, reply.context, wanted),
    }


def rescore(rows: list, repair: bool) -> list:
    """
    Count again from saved answers, with the checks as they are now.

    With repair, the answer is first passed through the code steps that run
    after the model today (grounding.drop_false_no_steps,
    generate.drop_trailing_not_found), so a change to
    those can be measured on answers already written, without asking the
    model again.
    """
    out = []
    for row in rows:
        text = row["answer"]
        if repair and row["path"] == "model":
            text, _ = grounding.drop_false_no_steps(text, row["context"])
            text, _ = generate.drop_trailing_not_found(text)
        out.append({**row, "answer": text,
                    **checks(text, row["context"], row["wanted_fcp"])})
    return out


def summarise(rows: list, tag: str, prompt: str) -> dict:
    pos   = [r for r in rows if not r["negative"]]
    neg   = [r for r in rows if r["negative"]]
    model = [r for r in pos if r["path"] == "model"]
    one   = [r for r in pos if r["wanted_fcp"] and r["path"] == "model"]
    held  = [r for r in model if r["right_in_context"]]
    return {
        "tag": tag, "date": date.today().isoformat(), "prompt": prompt,
        "backend": llm.BACKEND, "questions": len(pos),
        "by_model": len(model),
        "by_caption": sum(1 for r in pos if r["path"] == "caption"),
        "by_log_summary": sum(1 for r in pos if r["path"] == "commands"),
        "failed": sum(1 for r in pos if r["path"] == "failed"),
        "refused": sum(r["refused"] for r in model),
        "refused_with_right_passage": sum(r["refused"] for r in held),
        "tacked_on": sum(r["tacked_on"] for r in model),
        "one_right_fcp": len(one),
        "names_right_fcp": sum(r["names_wanted_fcp"] for r in one),
        "names_other_fcp": sum(r["names_other_fcp"] for r in one),
        "steps_in_passages": sum(r["wanted_has_steps"] for r in one),
        "steps_copied": sum(1 for r in one if r["wanted_has_steps"]
                            and r["codes_copied"] >= 2),
        "no_steps_under_a_listed_fcp": sum(1 for r in one if r["wanted_has_steps"]
                                           and r["no_steps_for_wanted"]),
        "said_no_steps": sum(r.get("said_no_steps", False) for r in model),
        "no_steps_and_more": sum(r.get("no_steps_and_more", False) for r in model),
        "cited": sum(1 for r in model if r["cited"]),
        "uncited_note": sum(r["uncited_note"] for r in model),
        "answers_with_ids_removed": sum(1 for r in model if r["ids_removed"]),
        "out_of_scope_refused": f"{sum(r['refused'] for r in neg)}/{len(neg)}",
        "median_chars": int(statistics.median(r["chars"] for r in model)) if model else 0,
        "median_answer_s": round(statistics.median(r["answer_s"] for r in model), 1) if model else 0,
        "slowest_answer_s": max((r["answer_s"] for r in pos), default=0),
    }


def main():
    ap = argparse.ArgumentParser(description="Answer evaluation")
    ap.add_argument("--satellite", default="aditya-l1")
    ap.add_argument("--tag", default=None,
                    help="name for this run (default: the prompt version)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="v9",
                    help="system prompt to answer with (default: the app's)")
    ap.add_argument("--model", default=generate.DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rescore", default=None, metavar="TAG",
                    help="count a saved run again instead of asking the model")
    ap.add_argument("--repair", action="store_true",
                    help="with --rescore: apply today's post-processing first")
    args = ap.parse_args()

    if args.rescore:
        src  = RUNS / f"{args.rescore}.json"
        rows = rescore(json.loads(src.read_text(encoding="utf-8")), args.repair)
        tag  = args.tag or f"{args.rescore}{'-repaired' if args.repair else '-rescored'}"
        prompt = rows and src.stem.split("-")[0] or "?"
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{tag}.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        write_summary(summarise(rows, tag, prompt), tag)
        return

    sat = registry.load_satellite(args.satellite)
    if not sat.is_ingested():
        raise SystemExit(f"{sat.slug} has no vector store. "
                         f"Run: python cli.py ingest {sat.slug}")
    ok, why = llm.available(args.model)
    if not ok:
        raise SystemExit(why)

    generate.SYSTEM_PROMPT = PROMPTS[args.prompt]
    generate.NOTE_RELATED  = NOTE_RELATED.get(args.prompt, "")
    tag   = args.tag or args.prompt
    cases = [(c, False) for c in TEST_SET] + [(c, True) for c in NEGATIVE_CASES]
    if args.limit:
        cases = cases[:args.limit]

    RUNS.mkdir(parents=True, exist_ok=True)
    out, rows = RUNS / f"{tag}.json", []
    print(f"\n  {len(cases)} questions, prompt {args.prompt}, {llm.summary()}")
    for i, (case, negative) in enumerate(cases, 1):
        rows.append(run_one(sat, case, negative, args.model))
        # Saved as it goes, so a long run that stops keeps what it has.
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        print(f"  {i}/{len(cases)}  {case['id']:<22}{rows[-1]['path']}",
              flush=True)

    write_summary(summarise(rows, tag, args.prompt), tag)
    print(f"  answers -> {out}  (document text, not for git)")
    print(f"  to read them: python review_answers.py {tag}\n")


def write_summary(summary: dict, tag: str) -> None:
    """Print one run's numbers and keep them in results/answer_checks.csv."""
    print()
    for key, value in summary.items():
        print(f"  {key:<28}{value}")

    RESULTS.mkdir(exist_ok=True)
    path  = RESULTS / "answer_checks.csv"
    known = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            known = [r for r in csv.DictReader(fh) if r.get("tag") != tag]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary))
        writer.writeheader()
        writer.writerows(known + [summary])
    print(f"\n  numbers -> {path}")


if __name__ == "__main__":
    main()
