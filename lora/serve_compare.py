"""
Serve the trained LoRA adapter two ways and compare them on the same prompts.

    transformers + PEFT   what lora/query.py does: load the base, generate,
                          wrap it with PeftModel, generate again
    vLLM, enable_lora     one server holds the base model and the adapter,
                          and each request names which one it wants

Three questions, in this order:
  1. Does vLLM's LoRA path compute the same thing as PEFT? Greedy outputs
     compared prompt by prompt. Different kernels and bf16 rounding mean
     exact agreement is not guaranteed, so the shared-prefix length is
     reported alongside exact matches.
  2. Did the adapter learn anything? Token F1 of base and adapter answers
     against the held-out eval split's reference answers.
  3. What does serving cost? Per-prompt latency on both paths, and one
     mixed batch where base and adapter requests are in flight together on
     a single vLLM engine, which is the thing PEFT-in-a-script cannot do.

    python lora/serve_compare.py reference    # needs torch, transformers, peft
    python lora/serve_compare.py vllm         # stdlib; needs the server up
    python lora/serve_compare.py report

The vLLM server must be started with the adapter registered, for example:

    vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype bfloat16 \\
        --enable-lora --max-lora-rank 16 \\
        --lora-modules adityal1=lora/adapters/adityal1-qwen2.5-0.5b \\
        --generation-config vllm --port 8000

Both paths decode greedily with no repetition penalty, set explicitly.
Note that lora/query.py does not set one, so it inherits the 1.1 in the
model's generation_config.json; its answers will not match these exactly.
"""

import json
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE        = Path(__file__).resolve().parent
OUT_DIR     = HERE / "serve_results"
ADAPTER     = HERE / "adapters" / "adityal1-qwen2.5-0.5b"
EVAL_PATH   = HERE / "data" / "adityal1_qa_eval.jsonl"
VLLM_URL    = "http://localhost:8000/v1"
ADAPTER_NAME = "adityal1"
MAX_NEW     = 150
N_EVAL      = 40

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# The five fixed questions lora/query.py uses, so the two are comparable.
FIXED_QUESTIONS = [
    "What payloads does Aditya-L1 carry, and what does each one measure?",
    "What is a halo orbit around the L1 Lagrange point, and why does Aditya-L1 use one?",
    "What launch vehicle placed Aditya-L1 into orbit, and from where was it launched?",
    "What are Aditya-L1's main science objectives?",
    "What is the difference between the VELC coronagraph and the SUIT imager?",
]


def base_model_name() -> str:
    results = HERE / "results.json"
    if results.exists():
        return json.loads(results.read_text(encoding="utf-8"))["base_model"]
    return "Qwen/Qwen2.5-0.5B-Instruct"


def prompt_set() -> list:
    """[{id, instruction, gold}] - the fixed five, then eval-split pairs."""
    items = [{"id": f"fixed{i}", "instruction": q, "gold": None}
             for i, q in enumerate(FIXED_QUESTIONS)]
    if EVAL_PATH.exists():
        lines = [l for l in EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, line in enumerate(lines[:N_EVAL]):
            r = json.loads(line)
            items.append({"id": f"eval{i}", "instruction": r["instruction"],
                          "gold": r["output"]})
    return items


# ------------------------------------------------------------- reference


def cmd_reference():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not ADAPTER.exists():
        raise SystemExit(f"No adapter at {ADAPTER}. Run lora/train.py first.")
    name   = base_model_name()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    t0 = time.perf_counter()
    tok   = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    load_base_s = time.perf_counter() - t0

    def gen(m, instruction, chat=False):
        if chat:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False, add_generation_prompt=True)
        else:
            prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        enc = tok(prompt, return_tensors="pt").to(device)
        start = time.perf_counter()
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - start
        new = out[0][enc["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip(), dt, len(new)

    items = prompt_set()
    rows = {it["id"]: dict(it) for it in items}
    # Base answers first: PeftModel attaches its modules to the base in
    # place, so once wrapped there is no clean base left to ask.
    for it in items:
        text, dt, n = gen(model, it["instruction"])
        rows[it["id"]].update(base=text, base_s=round(dt, 3), base_tokens=n)
        print(f"  base    {it['id']:<7} {dt:5.2f}s", flush=True)

    # The base model again, asked the way it was trained to be asked. The
    # "### Instruction" layout is the adapter's; the base model has never
    # seen it and runs on to the token limit, and a 150-token ramble
    # scores badly on token F1 whatever it knows. Its chat-template answer
    # is the baseline the adapter's F1 should be read against. Not timed
    # into the serving totals: those compare the same prompt on both paths.
    for it in items:
        text, dt, n = gen(model, it["instruction"], chat=True)
        rows[it["id"]].update(base_chat=text, base_chat_tokens=n)
        print(f"  base/chat {it['id']:<7} {dt:5.2f}s", flush=True)

    t0 = time.perf_counter()
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    load_adapter_s = time.perf_counter() - t0
    for it in items:
        text, dt, n = gen(model, it["instruction"])
        rows[it["id"]].update(adapter=text, adapter_s=round(dt, 3), adapter_tokens=n)
        print(f"  adapter {it['id']:<7} {dt:5.2f}s", flush=True)

    save("peft", {"base_model": name, "device": device, "dtype": str(dtype),
                  "load_base_s": round(load_base_s, 2),
                  "load_adapter_s": round(load_adapter_s, 2),
                  "rows": list(rows.values())})


# ------------------------------------------------------------- vllm


def complete(model: str, prompt: str) -> tuple:
    payload = {"model": model, "prompt": prompt, "max_tokens": MAX_NEW,
               "temperature": 0.0, "seed": 1234, "repetition_penalty": 1.0}
    req = urllib.request.Request(
        VLLM_URL + "/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    dt = time.perf_counter() - start
    choice = data["choices"][0]
    return choice["text"].strip(), dt, data.get("usage", {}).get("completion_tokens", 0)


def cmd_vllm():
    ref = load("peft")
    with urllib.request.urlopen(VLLM_URL + "/models", timeout=30) as r:
        served = [m["id"] for m in json.load(r)["data"]]
    if ADAPTER_NAME not in served:
        raise SystemExit(f"vLLM is serving {served}; the adapter "
                         f"{ADAPTER_NAME!r} is not registered (--lora-modules)")
    base = ref["base_model"] if ref["base_model"] in served else served[0]

    rows = {r["id"]: {"id": r["id"], "instruction": r["instruction"],
                      "gold": r["gold"]} for r in ref["rows"]}
    complete(base, "warm-up")
    complete(ADAPTER_NAME, "warm-up")

    # One at a time, the way lora/query.py runs.
    for rid, row in rows.items():
        p = PROMPT_TEMPLATE.format(instruction=row["instruction"])
        t, dt, n = complete(base, p)
        row.update(base=t, base_s=round(dt, 3), base_tokens=n)
        t, dt, n = complete(ADAPTER_NAME, p)
        row.update(adapter=t, adapter_s=round(dt, 3), adapter_tokens=n)
        print(f"  {rid:<7} base {row['base_s']:5.2f}s  adapter {row['adapter_s']:5.2f}s", flush=True)

    # Mixed: every prompt for both models, interleaved, 8 in flight on one
    # engine. Each request picks its model by name; nothing is reloaded.
    jobs = []
    for row in rows.values():
        p = PROMPT_TEMPLATE.format(instruction=row["instruction"])
        jobs += [(base, p), (ADAPTER_NAME, p)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        mixed = list(pool.map(lambda j: complete(*j), jobs))
    mixed_wall = time.perf_counter() - start
    # Concurrency changes the batch, which can change bf16 rounding; record
    # whether the mixed run still agrees with the sequential one.
    agree = sum(1 for (m, _), (t, _, _), i in zip(jobs, mixed, range(len(jobs)))
                if t == list(rows.values())[i // 2]["base" if m == base else "adapter"])

    save("vllm", {"base_model": base, "served": served,
                  "mixed_requests": len(jobs), "mixed_concurrency": 8,
                  "mixed_wall_s": round(mixed_wall, 2),
                  "mixed_tokens": sum(n for _, _, n in mixed),
                  "mixed_agree_with_sequential": agree,
                  "rows": list(rows.values())})


# ------------------------------------------------------------- report


def tokens(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def shared_prefix(a: str, b: str) -> float:
    """Fraction of the longer string the two agree on before diverging."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    longest = max(len(a), len(b)) or 1
    return n / longest


def shared_with_training() -> str:
    """How many of the scored eval pairs come from a chunk that also has a
    pair in the training file. Anything above zero means the F1 below is
    partly measured on text the adapter was trained on."""
    train_path = HERE / "data" / "adityal1_qa.jsonl"
    if not (train_path.exists() and EVAL_PATH.exists()):
        return "unknown"
    read = lambda p: [json.loads(l) for l in
                      p.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen  = {r.get("source_chunk_id") for r in read(train_path)}
    evals = read(EVAL_PATH)[:N_EVAL]
    return f"{sum(r.get('source_chunk_id') in seen for r in evals)}/{len(evals)}"


def cmd_report():
    peft, vl = load("peft"), load("vllm")
    P = {r["id"]: r for r in peft["rows"]}
    V = {r["id"]: r for r in vl["rows"]}
    ids = [i for i in P if i in V]
    evals = [i for i in ids if P[i]["gold"]]

    out = {"n_prompts": len(ids), "n_eval_with_gold": len(evals)}
    for side in ("base", "adapter"):
        out[f"{side}_exact_match"] = sum(P[i][side] == V[i][side] for i in ids)
        out[f"{side}_shared_prefix_median"] = round(statistics.median(
            shared_prefix(P[i][side], V[i][side]) for i in ids), 3)
        out[f"{side}_peft_s_median"] = round(statistics.median(P[i][f"{side}_s"] for i in ids), 3)
        out[f"{side}_vllm_s_median"] = round(statistics.median(V[i][f"{side}_s"] for i in ids), 3)
        if evals:
            out[f"{side}_f1_vs_gold_peft"] = round(statistics.mean(f1(P[i][side], P[i]["gold"]) for i in evals), 3)
            out[f"{side}_f1_vs_gold_vllm"] = round(statistics.mean(f1(V[i][side], V[i]["gold"]) for i in evals), 3)
    if evals and all("base_chat" in P[i] for i in evals):
        out["base_chat_f1_vs_gold_peft"] = round(statistics.mean(
            f1(P[i]["base_chat"], P[i]["gold"]) for i in evals), 3)
    out["base_at_token_cap"] = sum(P[i]["base_tokens"] >= MAX_NEW for i in ids)
    out["adapter_at_token_cap"] = sum(P[i]["adapter_tokens"] >= MAX_NEW for i in ids)
    out["eval_pairs_sharing_a_chunk_with_training"] = shared_with_training()
    out["adapter_changed_answer"] = sum(V[i]["base"] != V[i]["adapter"] for i in ids)
    seq_peft = sum(P[i]["base_s"] + P[i]["adapter_s"] for i in ids)
    seq_vllm = sum(V[i]["base_s"] + V[i]["adapter_s"] for i in ids)
    out["all_prompts_both_models_peft_sequential_s"] = round(seq_peft, 1)
    out["all_prompts_both_models_vllm_sequential_s"] = round(seq_vllm, 1)
    out["all_prompts_both_models_vllm_mixed_s"] = vl["mixed_wall_s"]
    out["mixed_agree_with_sequential"] = f"{vl['mixed_agree_with_sequential']}/{vl['mixed_requests']}"
    out["peft_load_base_s"] = peft["load_base_s"]
    out["peft_load_adapter_s"] = peft["load_adapter_s"]

    save("summary", out)
    width = max(len(k) for k in out)
    for k, v in out.items():
        print(f"  {k:<{width}}  {v}")
    print("\n  sample, fixed0:")
    print(f"    base    : {V['fixed0']['base'][:160]!r}")
    print(f"    adapter : {V['fixed0']['adapter'][:160]!r}")


# ------------------------------------------------------------- io


def save(name: str, payload: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


def load(name: str) -> dict:
    path = OUT_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"No {path}; run the earlier step first.")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    cmds = {"reference": cmd_reference, "vllm": cmd_vllm, "report": cmd_report}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]]()
