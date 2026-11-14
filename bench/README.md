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

**Use `bench/wsl/clean_rerun.sh`.** It runs every scenario on both backends
in one guarded pass (about 75 minutes) and is how the numbers in the deck
were produced after 28 October. The manual steps below still work, but the
first full run done by hand went wrong in three ways that nothing in the
output showed:

- **Four copies of the vLLM phase ran at once** against one engine. The
  engine log showed `Running: 4 reqs` through the sequential latency test.
  `run.py` now holds a lock and refuses to time against a vLLM engine that
  already has requests in flight.
- **Sampling was not matched.** vLLM applies the model's
  `generation_config.json` by default (`repetition_penalty=1.1` for
  Qwen2.5), and says so in a startup warning. Both backends now send
  `repetition_penalty=1.0` explicitly, the server runs with
  `--generation-config vllm`, and every result file records the settings.
- **One long-lived engine carried its prefix cache from scenario to
  scenario**, so throughput replayed prompts latency had already cached.
  The script starts a fresh engine per scenario and records the hit rate.

The structured scenario's `builder` arm sends exactly what
`lora/build_dataset.py` sends and parses the reply with its own rescue
branches, so its "chunks dropped" figure is the builder's real loss rate.
The older `json-string` arm uses the bench's prompt, which never names the
`pairs` key, and scores zero on shape by construction.

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

"Tuned Ollama" means `OLLAMA_NUM_PARALLEL=4` for text and `=2` for
captioning. The client kept 8 requests in flight in the text runs (4
against Ollama when captioning), and vLLM batched all 8. So the batch
ratios in the comparison tables are against Ollama at 4 and 2 slots, and
should be quoted with the slot count beside them.

Four slots turned out not to be Ollama's best setting. On 13 and 14
November the lab card was shared and never quiet, so the question was put
as paired runs instead: two arms alternated three times, each pair seeing
the same background load (the card read 94 to 100% from other jobs before
every run). Same model, same 102 questions, 8 in flight
(`bench/results-gpu-paired/paired.csv`):

    Ollama, 4 slots     54.0   55.0   52.7  tok/s
    Ollama, 8 slots     79.4   78.5   83.2          1.5x the 4-slot arm
    Ollama, 8 slots     79.0   77.1   76.9  tok/s
    vLLM               160.2  161.0  160.7          2.1x Ollama at 8 slots

Absolute speeds on a loaded card mean little: these are a third of the
quiet-card figures. The ratios within a pair are the result. Against
Ollama at its best measured setting vLLM is 2.1x on batch work, not far
from the 2.2x measured against 4 slots on the quiet card, because under
load vLLM's lead over the 4-slot arm widens to about 3x. Captioning has
not been repeated this way and still stands against Ollama at 2 slots.

The cached passages in `bench/cache/` were retrieved on 28 October, with
the rules router and before the command log's columns were corrected.
They are frozen on purpose: every generation run in these folders saw the
same passages, which is what makes them comparable. Rebuilding the cache
with today's retrieval would start a new series, not extend this one.

False refusals in `comparison.md` are recounted by `report.py` from the
saved answers each time. The count stored inside older result files used
a looser rule and overstated the 7B (29 of 100 where 20 were refusals).

Both backends are driven over HTTP with streaming on, because time to first
token has to be measured at the same point in both paths. That costs vLLM
something — its offline `LLM.generate()` over a list is its fastest batch
path and the server is not quite as quick — so the throughput figures here
are a floor for vLLM rather than its best.
