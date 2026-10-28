"""
Compare base-model vs. base+LoRA-adapter answers on fixed space-terminology
questions about Aditya-L1. This is the concrete proof the adapter actually
loads and shifts model behavior, and doubles as before/after content for
the methodology slide deck.

Usage:
    python lora/query.py
    python lora/query.py --adapter-name adityal1-qwen2.5-0.5b --json-out lora/comparison.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

DEFAULT_QUESTIONS = [
    "What payloads does Aditya-L1 carry, and what does each one measure?",
    "What is a halo orbit around the L1 Lagrange point, and why does Aditya-L1 use one?",
    "What launch vehicle placed Aditya-L1 into orbit, and from where was it launched?",
    "What are Aditya-L1's main science objectives?",
    "What is the difference between the VELC coronagraph and the SUIT imager?",
]


def generate(model, tokenizer, question: str, max_new_tokens: int = 150) -> str:
    prompt = PROMPT_TEMPLATE.format(instruction=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-name", default="adityal1-qwen2.5-0.5b")
    ap.add_argument("--json-out", default=str(Path(__file__).resolve().parent / "comparison.json"))
    args = ap.parse_args()

    adapter_dir = ADAPTERS_DIR / args.adapter_name
    if not adapter_dir.exists():
        raise SystemExit(f"No adapter at {adapter_dir}. Run lora/train.py first.")

    base_model_name = json.loads(RESULTS_PATH.read_text())["base_model"] if RESULTS_PATH.exists() \
        else "Qwen/Qwen2.5-0.5B-Instruct"

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"
    print(f"Device: {device}, dtype: {dtype}")

    print(f"Loading base model {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype).to(device)

    # Generate the base-only answers BEFORE wrapping the model: peft
    # attaches adapter modules to base_model's layers in place, so once
    # wrapped, base_model is no longer a clean "no adapter" reference.
    print("Generating base-model (no adapter) answers...")
    base_answers = {q: generate(base_model, tokenizer, q) for q in DEFAULT_QUESTIONS}

    print(f"Loading adapter from {adapter_dir}...")
    # A fresh PeftModel.from_pretrained load in this new process is the
    # concrete check that the adapter persisted correctly, not just that
    # it worked in the same process that trained it.
    adapted_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    results = []
    for q in DEFAULT_QUESTIONS:
        print(f"\nQ: {q}")
        t0 = time.time()
        base_answer = base_answers[q]
        adapted_answer = generate(adapted_model, tokenizer, q)
        dt = time.time() - t0

        print(f"  base:    {base_answer[:200]}")
        print(f"  adapted: {adapted_answer[:200]}")

        results.append({
            "question": q,
            "base_answer": base_answer,
            "adapted_answer": adapted_answer,
            "seconds": round(dt, 1),
        })

    Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote comparison to {args.json_out}")


if __name__ == "__main__":
    main()
