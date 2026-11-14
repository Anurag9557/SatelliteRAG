# Ollama vs vLLM

- Profile: matched-fp16 (same weights, same precision, both fit in 6 GB)
- Machine: NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB, 581.86 | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 | WSL2
- Model (ollama): qwen2.5:1.5b-instruct-fp16
- Model (ollama-tuned): qwen2.5:1.5b-instruct-fp16
- Model (vllm): Qwen/Qwen2.5-1.5B-Instruct
- Sampling (both, sent explicitly): {"repetition_penalty": 1.0, "sent_explicitly_to_both": true, "temperature": 0.0}

| Metric | Ollama (default) | Ollama (`NUM_PARALLEL=4`) | vLLM | Verdict |
| --- | ---: | ---: | ---: | --- |
| Single-query TTFT, median (s) | 0.478 | not run | 0.175 | 63% to vLLM |
| Single-query TTFT, p95 (s) | 1.095 | not run | 0.308 | 72% to vLLM |
| Single-query total, median (s) | 3.547 | not run | 2.817 | 21% to vLLM |
| Decode rate, median (tok/s) | 46.25 | not run | 49.31 | 7% to vLLM |
| Seconds per 100 output tokens, median | 2.632 | not run | 2.218 | 16% to vLLM |
| Output tokens per answer, median | 131.5 | not run | 127 |  |
| Prefix-cache hit rate, latency run | not run | not run | 0.29 |  |
| Batch wall clock (s) | 435.68 | 157.62 | 76.32 | 52% to vLLM |
| Batch throughput (req/s) | 0.234 | 0.647 | 1.337 | 107% to vLLM |
| Batch throughput (tok/s) | 44.4 | 122.5 | 244.7 | 100% to vLLM |
| Fully reproducible answers | 1 | not run | 0.8 | 20% to Ollama |
| Answers carrying a citation | 0.37 | not run | 0.38 | 3% to vLLM |
| Correct refusals (out-of-scope) | 2/2 | not run | 2/2 |  |
| False refusals (answerable) | 9/100 | not run | 9/100 |  |
| Real answers ending in the not-found line | 0/100 | not run | 0/100 |  |
| Median answer length (chars) | 545 | not run | 514 |  |
| Answers that hit the token cap | 14/102 | not run | 12/102 |  |

## Structured output

| Arm | Parsed | Correct shape, no rescue | Chunks dropped | Pairs | s/chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| ollama / builder | 1 | 0 | 8/50 | 42 | 1.694 |
| ollama / schema | 1 | 1 | 0/50 | 99 | 2.599 |
| vllm / schema | 1 | 1 | 0/50 | 100 | 2.176 |

`builder` is lora/build_dataset.py's own prompt and parsing, rescue branches included; `schema` passes the QA schema to the server.

Verdict compares vLLM against Ollama's better arm, so the batching figure is measured against Ollama with parallelism enabled rather than against its shipped default.


The client kept 8 requests in flight. Ollama's tuned arm had 4 parallel slots, so it served 4 at a time while vLLM batched all 8. The batch verdict is therefore against Ollama at 4 slots. bench/README.md has the paired runs against Ollama at 8 slots, its better setting.


Medians and p95, not means. Retrieval ran once and the same cached passages were replayed to both backends, so these numbers describe generation only.
