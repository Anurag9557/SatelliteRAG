# Paired runs on a shared card

Lab GPU (RTX PRO 5000, 48 GB), 13 and 14 November 2026. Qwen2.5-7B at fp16
on both runtimes, the same 102 questions with 8 in flight, sampling pinned
as in every other run.

The card is shared and was at 94 to 100% from other jobs throughout, so a
quiet-card timing was not possible. Each series alternates two arms three
times, so both arms of a round see the same background load. Read the
ratio within a round, not the absolute speed.

`paired.csv` holds the twelve runs:

- `slots`: Ollama with 4 parallel slots against 8. Eight is 1.5x faster
  in every round (ratios 1.47, 1.43, 1.58).
- `runtime`: Ollama with 8 slots against vLLM (`--max-num-seqs 32`).
  vLLM is 2.1x faster in every round (ratios 2.03, 2.09, 2.09).

The figures were copied from the run logs on the lab machine
(`slots_paired.log`, `vllm_paired.log`). The per-run JSON files are still
there, under `~/satrag_raghav/bench/results-gpu-paired*/`, and have not
been fetched. Ollama was loaded with one short request before each timed
run; vLLM loads its model before it reports ready.
