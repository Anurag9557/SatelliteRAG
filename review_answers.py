"""
Read answers against the passages they were written from.

evaluate_answers.py counts refusals, citations and removed identifiers. It
cannot tell whether an answer is right. This turns one of its runs into a
page: each question, the answer the app gave, and the passages the model
was given, with buttons to mark it.

    python evaluate_answers.py --tag today        make a run
    python review_answers.py today                30 answers, at random
    python review_answers.py today --all
    python review_answers.py today --only refused   refusals only
    python review_answers.py today --only figures   figure answers only
    python review_answers.py --tally answer_review_<run>.json

Open the page it names, mark each answer, and press Save. That downloads
answer_review_<run>.json (with _refused or _procedures for a filtered page);
--tally counts it.

The page holds document text. It is written to runs/answers/, which is
kept out of git, and is not to be shared.

Marking guide:
    right          every statement is supported by the passages, and it
                   answers what was asked
    partly         answers the question but a detail is wrong, missing or
                   not in the passages
    wrong          says something the passages do not, or answers about
                   the wrong thing
    over-refused   said not found, and the passages do hold the answer
"""

import argparse
import html
import json
import random
from collections import Counter
from pathlib import Path

RUNS  = Path("runs") / "answers"
MARKS = ("right", "partly", "wrong", "over-refused")

STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f4f4f1; color: #1c1c1a; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 80px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.lead { color: #5a5a55; margin: 0 0 24px; }
.item { background: #fff; border: 1px solid #dcdcd6; border-radius: 6px; padding: 16px; margin: 0 0 16px; }
.q { font-weight: 600; margin: 0 0 4px; }
.how { font-size: 13px; color: #5a5a55; margin: 0 0 10px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font: 14px/1.5 ui-monospace, Consolas, monospace;
      background: #f7f7f4; border: 1px solid #e6e6e0; border-radius: 4px; padding: 10px; margin: 0 0 10px; }
details { margin: 0 0 12px; }
summary { cursor: pointer; color: #33506e; }
label { margin-right: 16px; cursor: pointer; }
input[type=text] { width: 100%; box-sizing: border-box; margin-top: 8px; padding: 6px 8px;
                   border: 1px solid #c8c8c2; border-radius: 4px; font: inherit; }
.bar { position: fixed; left: 0; right: 0; bottom: 0; background: #1c1c1a; color: #fff;
       padding: 10px 16px; display: flex; gap: 16px; align-items: center; }
button { font: inherit; padding: 6px 14px; border: 0; border-radius: 4px; cursor: pointer; }
"""

SCRIPT = """
const items = [...document.querySelectorAll('.item')];
function marks() {
  return items.map(el => ({
    id: el.dataset.id,
    mark: (el.querySelector('input[type=radio]:checked') || {}).value || '',
    note: el.querySelector('input[type=text]').value.trim(),
  }));
}
function count() {
  const done = marks().filter(m => m.mark).length;
  document.getElementById('count').textContent = done + ' of ' + items.length + ' marked';
}
document.addEventListener('change', count);
document.getElementById('save').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(marks(), null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = document.body.dataset.file;
  a.click();
});
count();
"""

HOW = {"model": "written by the model",
       "caption": "built from a figure caption; check the caption against "
                  "the image file named in the passages",
       "commands": "counted from the command log", "failed": "the model could not be reached",
       "none": "nothing was retrieved"}


def build(tag: str, n: int, seed: int, everything: bool, only: str) -> Path:
    path = RUNS / f"{tag}.json"
    if not path.exists():
        raise SystemExit(f"no run called {tag!r} in {RUNS}. "
                         f"Make one: python evaluate_answers.py --tag {tag}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if only == "refused":
        rows = [r for r in rows if r["refused"] and not r["negative"]]
    elif only == "procedures":
        rows = [r for r in rows if r["intent"] == "procedure"]
    elif only == "figures":
        rows = [r for r in rows if r["path"] == "caption"]
    if not everything and not only:
        rows = random.Random(seed).sample(rows, min(n, len(rows)))

    parts = []
    for i, row in enumerate(rows):
        how = HOW.get(row["path"], row["path"])
        if row["negative"]:
            how += "; this question has no answer in the documents"
        sources = "\n".join(f"[{k}] {s}" for k, s in enumerate(row["sources"], 1))
        passages = row["context"] or sources
        buttons = "".join(
            f'<label><input type="radio" name="m{i}" value="{m}"> {m}</label>'
            for m in MARKS)
        parts.append(
            f'<div class="item" data-id="{html.escape(row["id"])}">'
            f'<p class="q">{html.escape(row["question"])}</p>'
            f'<p class="how">{html.escape(row["id"])} | {html.escape(how)}</p>'
            f'<pre>{html.escape(row["answer"])}</pre>'
            f'<details><summary>The passages it was given</summary>'
            f'<pre>{html.escape(passages)}</pre></details>'
            f'{buttons}'
            f'<input type="text" placeholder="What is wrong, if anything">'
            f'</div>')

    # Each page, and the marks saved from it, is named for its run and its
    # filter, so the refusals page does not overwrite the full one and two
    # sets of marks cannot be mixed up in the downloads folder.
    name = f"{tag}{'_' + only if only else ''}"
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Answer review</title><style>{STYLE}</style></head>'
        f'<body data-file="answer_review_{html.escape(name)}.json"><main>'
        f'<h1>Answer review: run {html.escape(tag)}</h1>'
        f'<p class="lead">{len(rows)} answers. Open the passages, mark the '
        f'answer against them, then save.</p>'
        + "".join(parts) +
        '</main><div class="bar"><span id="count"></span>'
        f'<button id="save">Save marks</button></div><script>{SCRIPT}</script>'
        '</body></html>')
    out = RUNS / f"review_{name}.html"
    out.write_text(page, encoding="utf-8")
    return out


def tally(path: Path) -> None:
    marks  = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(m["mark"] or "not marked" for m in marks)
    done   = sum(n for mark, n in counts.items() if mark != "not marked")
    print(f"\n  {len(marks)} answers, {done} marked")
    for mark in MARKS + ("not marked",):
        if counts[mark]:
            share = (f"{counts[mark] / done * 100:5.1f}%"
                     if done and mark != "not marked" else "")
            print(f"  {mark:<13}{counts[mark]:>4}  {share}")
    notes = [m for m in marks if m.get("note")]
    if notes:
        print(f"\n  {len(notes)} with a note:")
        for m in notes:
            print(f"    {m['id']:<22}{m['mark']:<14}{m['note']}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Read answers against passages")
    ap.add_argument("tag", nargs="?", default=None,
                    help="the run to review, as given to evaluate_answers.py")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", choices=["refused", "procedures", "figures"],
                    default="")
    ap.add_argument("--tally", default=None, help="count a saved review file")
    args = ap.parse_args()

    if args.tally:
        tally(Path(args.tally))
        return
    if not args.tag:
        raise SystemExit("name a run: python review_answers.py <tag>")
    out = build(args.tag, args.n, args.seed, args.all, args.only)
    print(f"\n  open in a browser: {out.resolve()}\n")


if __name__ == "__main__":
    main()
