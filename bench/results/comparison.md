# Ollama vs vLLM

- Profile: matched-fp16 (same weights, same precision, both fit in 6 GB)
- Machine: NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB, 581.86 | Windows-10-10.0.26200-SP0
- Model (ollama): qwen2.5:1.5b-instruct-fp16

| Metric | Ollama | vLLM | Verdict |
| --- | ---: | ---: | --- |
| Single-query TTFT, median (s) | 2.404 | not run |  |
| Single-query TTFT, p95 (s) | 4.467 | not run |  |
| Single-query total, median (s) | 7.68 | not run |  |
| Decode rate, median (tok/s) | 44.075 | not run |  |
| Batch wall clock (s) | 59.96 | not run |  |
| Batch throughput (req/s) | 0.167 | not run |  |
| Batch throughput (tok/s) | 42.6 | not run |  |
| Fully reproducible answers | 1 | not run |  |
| Answers carrying a citation | 0.6 | not run |  |
| Correct refusals (out-of-scope) | 0/0 | not run |  |
| False refusals (answerable) | 0/10 | not run |  |
| Median answer length (chars) | 831 | not run |  |

## Structured output

| Arm | Parsed | Correct shape | Pairs | s/chunk |
| --- | ---: | ---: | ---: | ---: |
| ollama / format=json-string | 0.96 | 0 | 0 | 2.689 |
| ollama / format=schema | 1 | 1 | 100 | 4.933 |

Medians and p95, not means. Retrieval ran once and the same cached passages were replayed to both backends, so these numbers describe generation only.
