"""
Build slides/LoRA_Finetuning_Methodology.pptx from the actual LoRA run's
own output files: lora/data/build_meta.json, lora/data/adityal1_qa.jsonl,
lora/results.json, lora/comparison.json. Numbers and examples shown are
real measurements from the run that produced them, not asserted figures.

Run this AFTER lora/build_dataset.py, lora/train.py, and lora/query.py
have all completed at least once.

Usage: python slides/build_lora_methodology_pptx.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_utils import add_bullet_slide, add_text_slide, new_deck

LORA_DIR = Path(__file__).resolve().parent.parent / "lora"
OUT_PATH = Path(__file__).resolve().parent / "LoRA_Finetuning_Methodology.pptx"


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    build_meta = load_json(LORA_DIR / "data" / "build_meta.json", {})
    results = load_json(LORA_DIR / "results.json", {})
    comparison = load_json(LORA_DIR / "comparison.json", [])
    train_examples = load_jsonl(LORA_DIR / "data" / "adityal1_qa.jsonl")

    missing = [name for name, v in [
        ("lora/data/build_meta.json", build_meta),
        ("lora/results.json", results),
    ] if not v]
    if missing:
        sys.exit("Missing run output(s), run the LoRA pipeline first: " + ", ".join(missing))

    prs = new_deck(
        "LoRA Fine-Tuning for Space-Domain QA",
        "Freezing a base model and training only low-rank adapters — SatelliteRAG",
    )

    # What LoRA is.
    add_bullet_slide(prs, "What LoRA Does", [
        "Full fine-tuning updates every weight in the base model — expensive, and",
        ("risky on a small dataset (the model can forget general ability).", 1),
        "LoRA freezes the base model's weights entirely.",
        "A small pair of low-rank matrices (A, B) is added alongside chosen",
        ("attention projections; only A and B are trained.", 1),
        "The update to a frozen weight W becomes W + BA, where B and A are",
        ("far smaller than W — typically well under 1% of total parameters.", 1),
        "Because so few parameters train, it is realistic to fine-tune on CPU,",
        ("which this project has no GPU for.", 1),
    ])

    # This project's actual configuration.
    add_bullet_slide(prs, "This Project's Configuration", [
        f"Base model: {results.get('base_model', '?')} (frozen)",
        f"LoRA rank (r): {results.get('lora_r', '?')}, alpha: {results.get('lora_alpha', '?')}, "
        f"dropout: {results.get('lora_dropout', '?')}",
        f"Target modules: {', '.join(results.get('target_modules', []))}",
        f"Trainable parameters: {results.get('trainable_params', 0):,} / "
        f"{results.get('total_params', 0):,} ({results.get('trainable_pct', '?')}%)",
        f"Training data: {build_meta.get('n_train', '?')} QA pairs "
        f"(synthesis mode: {build_meta.get('mode', '?')})",
        f"Epochs: {results.get('epochs', '?')}, batch size: {results.get('batch_size', '?')} "
        f"x grad-accum {results.get('grad_accum', '?')}",
    ])

    # Data pipeline with one real example.
    example_block = ["Training data was synthesized from the same document chunks the RAG",
                      "pipeline extracts (satellites/aditya-l1/extracted/chunks.json), via the",
                      "project's local Ollama model — no external API calls.", ""]
    if train_examples:
        ex = train_examples[0]
        example_block += [
            f"Example (source: {ex.get('source_doc', '?')}):",
            f"Q: {ex['instruction']}",
            f"A: {ex['output'][:220]}{'...' if len(ex['output']) > 220 else ''}",
        ]
    add_text_slide(prs, "Data Pipeline", "\n".join(example_block), font_size=14)

    # Real measured results.
    loss_hist = results.get("loss_history", [])
    loss_line = (f"Loss: {loss_hist[0]['loss']:.3f} → {loss_hist[-1]['loss']:.3f} "
                 f"over {len(loss_hist)} logged steps" if loss_hist
                 else f"Final training loss: {results.get('final_loss', '?')}")
    add_bullet_slide(prs, "Measured Training Results", [
        loss_line,
        f"Wall-clock time: {results.get('wall_clock_seconds', '?')} seconds (CPU-only)",
        f"Adapter saved to: {results.get('adapter_dir', '?')}",
        "This is a demonstration-scale fine-tune (small model, small dataset,",
        ("short training) — it shows the LoRA mechanism working end-to-end,", 1),
        ("not a production-quality fine-tune.", 1),
    ])

    # Before/after answer comparison.
    if comparison:
        c = comparison[0]
        add_text_slide(prs, "Before / After: One Example", "\n".join([
            f"Q: {c['question']}", "",
            f"Base model: {c['base_answer'][:300]}", "",
            f"Base + LoRA adapter: {c['adapted_answer'][:300]}",
        ]), font_size=13)

    prs.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
