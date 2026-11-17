"""
Score models on the passage-grounded eval set (lora/data/rag_eval.jsonl).

Two steps, so the GPU machine needs nothing but the standard library:

    python lora/eval_rag.py ask --url http://localhost:8124/v1 --model rag-0.5b
        sends every eval example to an OpenAI-style server (vLLM), the
        same way the app does: greedy, seed 1234, no repetition penalty.
        Writes lora/rag_results/<model>.json with the raw replies.

    python lora/eval_rag.py report
        run where core/ is importable. Scores every results file:

        answerable questions (the source passage is among the five)
          answered right   the reply holds the reference answer
          token F1         against the reference, citations removed
          right citation   cites a passage that holds the answer
          refused          said not found although the passage was there
        unanswerable questions (no passage holds the answer)
          refused          the right reply
          answered anyway  the failure the grounding guard exists for
"""

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE / "data" / "rag_eval.jsonl"
OUT  = HERE / "rag_results"


def ask_one(url: str, model: str, messages: list) -> tuple:
    """(reply, seconds, tokens). A refused request is kept as an error
    reply, so one prompt the server will not take does not lose the run."""
    body = {"model": model, "messages": messages, "temperature": 0.0,
            "seed": 1234, "max_tokens": 300, "repetition_penalty": 1.0}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
    except urllib.error.HTTPError as exc:
        return f"[request failed: HTTP {exc.code}]", time.perf_counter() - start, 0
    reply = data["choices"][0]["message"]["content"] or ""
    return reply, time.perf_counter() - start, data.get("usage", {}).get("completion_tokens", 0)


def cmd_ask(args):
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    ask_one(args.url, args.model, [{"role": "user", "content": "Reply with: ready"}])
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.flight) as pool:
        replies = list(pool.map(lambda r: ask_one(args.url, args.model, r["messages"]), rows))
    wall = time.perf_counter() - start
    OUT.mkdir(exist_ok=True)
    label = args.label or args.model.split("/")[-1]
    (OUT / f"{label}.json").write_text(json.dumps({
        "model": args.model, "label": label, "wall_s": round(wall, 1),
        "rows": [{"i": i, "kind": r["kind"], "reply": t, "seconds": round(s, 2), "tokens": n}
                 for i, (r, (t, s, n)) in enumerate(zip(rows, replies))],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{label}: {len(rows)} replies in {wall:.0f} s")


# ------------------------------------------------------------- report

def cmd_report(args):
    sys.path.insert(0, str(HERE.parent))
    from core import generate
    sys.path.insert(0, str(HERE))
    from build_rag_dataset import holds_answer, words

    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    cites = re.compile(r"\[Source[s]?\s*([\d,\s and]+)\]", re.I)

    def cited(text):
        return {int(n) for m in cites.finditer(text) for n in re.findall(r"\d+", m.group(1))}

    def refused(text):
        return generate.is_refusal(text) or generate.NOT_FOUND_MESSAGE.rstrip(".") in text

    def f1(pred, gold):
        p, g = words(generate.CITATION.sub(" ", pred)), words(gold)
        common = sum((Counter(p) & Counter(g)).values())
        if not p or not g or not common:
            return 0.0
        prec, rec = common / len(p), common / len(g)
        return 2 * prec * rec / (prec + rec)

    table = []
    for path in sorted(OUT.glob("*.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        ans = [(r, x) for r, x in zip(rows, res["rows"]) if r["kind"] == "answer"]
        neg = [(r, x) for r, x in zip(rows, res["rows"]) if r["kind"] == "refuse"]
        a_ref = sum(refused(x["reply"]) for _, x in ans)
        right = sum(not refused(x["reply"]) and holds_answer(x["reply"], r["gold"]) for r, x in ans)
        good_cite = sum(bool(cited(x["reply"]) & set(r["answer_sources"])) for r, x in ans)
        n_ref = sum(refused(x["reply"]) for _, x in neg)
        table.append({
            "model": res["label"],
            "answerable": len(ans),
            "answered_right": right,
            "token_f1": round(statistics.mean(f1(x["reply"], r["gold"]) for r, x in ans), 3),
            "right_citation": good_cite,
            "refused_answerable": a_ref,
            "unanswerable": len(neg),
            "refused_unanswerable": n_ref,
            "answered_anyway": len(neg) - n_ref,
            "median_tokens": statistics.median(x["tokens"] for x in res["rows"]),
            "wall_s": res["wall_s"],
        })
    (OUT / "summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    keys = list(table[0])
    print(" | ".join(keys))
    for t in table:
        print(" | ".join(str(t[k]) for k in keys))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("--url", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--label", default="")
    a.add_argument("--flight", type=int, default=16)
    sub.add_parser("report")
    args = ap.parse_args()
    {"ask": cmd_ask, "report": cmd_report}[args.cmd](args)
