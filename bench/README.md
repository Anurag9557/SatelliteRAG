# bench — Ollama against vLLM on this pipeline

Measures the two generation backends on the same work: the same questions,
the same retrieved passages, the same model weights at the same precision.
Retrieval runs once up front and its output is cached, so nothing here is
measuring the retriever.

Five scenarios, because "is vLLM faster" is not one question:

| Scenario | What it answers |
| --- | --- |
| `latency` | One question at a time, which is how the CLI is used. Time to first token and total. |
| `throughput` | Many requests in flight, which is the captioning and QA-synthesis shape. |
| `determinism` | Same question repeated. Decides whether answer quality can be measured at all. |
| `quality` | Answers dumped for scoring, plus citation and refusal rates. |
| `structured` | The QA-synthesis call three ways: `format:"json"`, a real schema on Ollama, and vLLM's `structured_outputs`. |

Expect Ollama to win `latency`. A queue of one gets nothing from continuous
batching, and a warm Ollama model has no engine to start. If the report says
otherwise, check the vLLM server was already warm before the run.

---

## Why this needs WSL2

vLLM ships no Windows build. Its kernels are compiled CUDA and Triton against
a Linux toolchain and PyPI carries manylinux wheels only, so `pip install
vllm` on Windows finds nothing to install.

The second reason matters more. `qwen2.5:7b-instruct` only fits in 6 GB
because Ollama runs it as a 4-bit GGUF, about 4.7 GB. vLLM's default is fp16:
7.6 billion parameters at two bytes each is roughly 15 GB of weights before a
single KV-cache block is allocated, and vLLM pre-allocates that cache at
startup. So the benchmark runs a **1.5B** model, where fp16 weights are about
3.1 GB and both servers can hold the same thing.

That is a real limitation of the result, and the deck should say so: it
compares the two runtimes, not the 7B latency the project actually has.

---

## Setup

### 1. WSL2 with GPU passthrough

In PowerShell as administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Then inside Ubuntu:

```bash
nvidia-smi
```

That must print the RTX 3050 and a driver version. If it does, CUDA
passthrough is working. **Do not install an NVIDIA display driver inside
WSL** — the Windows driver supplies CUDA to the Linux side, and installing a
second one inside breaks it.

### 2. Both servers, inside WSL

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b-instruct-fp16          # 3.1 GB

python3 -m venv ~/.venvs/vllm
source ~/.venvs/vllm/bin/activate
pip install vllm                                 # needs Python 3.10 - 3.14
```

Install vLLM into a **fresh** virtualenv. Its wheels bundle a pinned PyTorch
built against CUDA 12.9, and vLLM's own docs warn that mixing them with an
existing torch install causes binary incompatibility. Do not reuse the
project's `.venv`.

> **Watch the port.** WSL2 forwards `localhost`, so if Ollama is also running
> on Windows, `localhost:11434` from inside WSL may reach the *Windows*
> Ollama and you will benchmark the wrong process. Stop the Windows Ollama
> service first, then confirm with `ollama ps` inside WSL.

### 3. The model download is a one-time online step

vLLM pulls weights from Hugging Face on first run, the same way
`cli.py prefetch` does for the embedders. Once cached, it runs offline. If
the machine is air-gapped, download on a connected machine and copy
`~/.cache/huggingface` across.

---

## Running it

The repo lives on the Windows disk, reachable from WSL:

```bash
cd /mnt/c/Users/anura/Desktop/SatelliteRAG
```

**Cache the retrieval once.** This uses the project's own venv, not the vLLM
one, because it imports `core.retrieve`:

```bash
python bench/run.py cache aditya-l1
```

**Start Ollama and measure it:**

```bash
python bench/run.py all ollama
```

**Start the vLLM server** in a second terminal, with flags sized for 6 GB:

```bash
source ~/.venvs/vllm/bin/activate
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --port 8000
```

Wait for it to report that it is serving, then in the first terminal:

```bash
python bench/run.py all vllm
python bench/report.py
```

`bench/results/comparison.md` is the table to paste.

Only one server can hold the GPU at a time at these settings. Stop Ollama
(`ollama stop <model>`) before starting vLLM, or the second one will fail to
allocate.

---

## Profiles

```bash
python bench/run.py all ollama --profile matched-fp16
```

| Profile | Ollama | vLLM | Question it answers |
| --- | --- | --- | --- |
| `matched-fp16` *(default)* | `qwen2.5:1.5b-instruct-fp16` | `Qwen/Qwen2.5-1.5B-Instruct` | Which runtime is faster at equal precision |
| `matched-fp16-small` | `qwen2.5:0.5b-instruct-fp16` | `Qwen/Qwen2.5-0.5B-Instruct` | Same, on the base the LoRA adapter was trained from |
| `as-deployed` | `qwen2.5:1.5b-instruct` (4-bit) | `Qwen/Qwen2.5-1.5B-Instruct` (fp16) | What would actually be experienced today |

Use `matched-fp16` for anything you intend to call a runtime comparison. In
`as-deployed` the two sides run different weights, so a quality difference
measures the quantiser rather than the server, and the report should only be
read for latency.

---

## Confidentiality

`bench/cache/` holds the retrieved passages and `bench/results/*.json` holds
generated answers. Both contain document text, exactly as
`satellites/<slug>/extracted/` does, and both are in `.gitignore`.

`comparison.csv` and `comparison.md` hold aggregate numbers only and are safe
to commit and to share.

---

## Reading the output

Medians and p95, never means: the first request to either server pays for a
cold cache and would drag a mean into nonsense.

Both backends are driven over HTTP with streaming on, because time to first
token has to be measured at the same point in both paths. That costs vLLM
something — its offline `LLM.generate()` over a list is its fastest batch
path and the server is not quite as quick — so the throughput figures here
are a floor for vLLM rather than its best.
