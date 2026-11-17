"""
LoRA fine-tune on the passage-grounded dataset (lora/build_rag_dataset.py).

Differences from lora/train.py, and why:

- The model's own chat template, with the app's system prompt, so the
  adapter is trained on exactly what core/generate.py sends and can be
  served behind the same chat endpoint.
- Up to 3,072 tokens per example: five passages and the system prompt are
  about 2,000; the old limit of 384 would cut the passages off.
- LoRA on every linear layer (attention and MLP), not attention only.
  Reading passages and copying from them is the skill being taught, and
  the MLP layers carry most of a small model's capacity.
- Loss on the answer tokens only, as before.

    python lora/train_rag.py --base Qwen/Qwen2.5-0.5B-Instruct --name rag-0.5b
    python lora/train_rag.py --base Qwen/Qwen2.5-1.5B-Instruct --name rag-1.5b

Writes lora/adapters/<name>/ and lora/adapters/<name>/train_results.json.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

HERE    = Path(__file__).resolve().parent
DATA    = HERE / "data" / "rag_train.jsonl"
ADAPTERS = HERE / "adapters"
MAX_LEN = 3072
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(row: dict, tok) -> dict:
    prompt = tok.apply_chat_template(row["messages"], tokenize=False,
                                     add_generation_prompt=True)
    # The template closes an assistant turn with <|im_end|>; ending the
    # target the same way teaches the model where to stop.
    full = prompt + row["target"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids   = tok(full, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_LEN:
        return None
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--name", default="rag-0.5b")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    encoded = [e for e in (encode(r, tok) for r in rows) if e]
    dropped = len(rows) - len(encoded)
    lengths = sorted(len(e["input_ids"]) for e in encoded)
    print(f"{len(encoded)} examples ({dropped} over {MAX_LEN} tokens dropped); "
          f"median {lengths[len(lengths) // 2]} tokens, longest {lengths[-1]}")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=TARGETS, task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    out = ADAPTERS / args.name
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "checkpoints"), per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum, num_train_epochs=args.epochs,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=5,
            logging_steps=5, save_strategy="no", report_to=[], bf16=True,
            remove_unused_columns=False, seed=13),
        train_dataset=Dataset.from_list(encoded),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    start = time.time()
    result = trainer.train()
    seconds = time.time() - start
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    losses = [{"step": e["step"], "loss": e["loss"]}
              for e in trainer.state.log_history if "loss" in e]
    (out / "train_results.json").write_text(json.dumps({
        "base_model": args.base, "examples": len(encoded), "dropped_too_long": dropped,
        "median_tokens": lengths[len(lengths) // 2], "longest_tokens": lengths[-1],
        "rank": args.rank, "alpha": 2 * args.rank, "targets": TARGETS,
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "accum": args.accum,
        "trainable": trainable, "total": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "final_loss": result.training_loss, "seconds": round(seconds, 1),
        "gpu": torch.cuda.get_device_name(0), "loss_history": losses,
    }, indent=2), encoding="utf-8")
    print(f"done in {seconds:.0f} s, loss {result.training_loss:.4f}, saved {out}")


if __name__ == "__main__":
    main()
