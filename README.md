# Satellite RAG

Retrieval over spacecraft mission documents, one satellite at a time or
across all of them. Each satellite owns its documents, chunks and vector
store. Nothing is shared except the model cache in `models/`.

Fully offline: extraction, figure captioning, embedding, retrieval and
answer generation all run locally.

Retrieval is hybrid — a dense semantic pass and a BM25 keyword pass,
fused by Reciprocal Rank Fusion, with an optional cross-encoder reranker.
Every component was chosen by measurement on a 100-question labelled set;
see `PROJECT_CONTEXT.md` for the numbers and `COMPARISON.md` for what
changed against the earlier pipeline.

    Hit@1  0.84     Hit@5  0.93     Hit@10  0.97     MRR  0.883

## Documentation

| File | Contents |
|---|---|
| `SETUP.txt` | installation, step by step |
| `GIT_SETUP.md` | pushing to GitHub, and how to run it after cloning |
| `PROJECT_CONTEXT.md` | architecture, design decisions, every measured number |
| `COMPARISON.md` | this system versus the previous pipeline |
| `results/` | raw measurements |
| `testset_public_demo.py` / `evaluate_public_demo.py` | retrieval benchmark for the public demo dataset below (`testset.py` / `evaluate.py` are written for the confidential real corpus and score zero against it) |


## Demo dataset

The real Aditya-L1 flight-ops documents are confidential and are not in
this repo. `satellites/aditya-l1/` ships instead with 4 public ISRO/ISSDC
documents about the mission itself (mission booklet, spacecraft/orbit
booklet, Announcement of Opportunity, mission brochure), so the pipeline
works out of the box. Re-fetch them with:

    python scripts/fetch_adityal1_docs.py

This dataset only has MOR / OPS_MANUAL / GUIDELINES / HANDBOOK documents —
no FCP or telecommand log — so it demonstrates retrieval and generation,
not the procedure/command-log features described below. `lora/` and
`slides/` hold a LoRA fine-tuning demo and auto-generated PPTX decks built
from this same public data.


## Adding a satellite

Scaffold the folder (or just create it yourself — nothing below needs it):

    python cli.py new chandrayaan-3

Copy the documents in, named so each one's type is recognised
automatically (see "Which filenames are recognised" below):

    satellites\chandrayaan-3\documents\
        CH3_FCP_v2.pdf
        Chandrayaan3_MissionOperationsReport.pdf
        CH3_Operations_Manual.docx
        CH3_Guidelines_NormalPhase.doc
        telecommand_history_ch3.txt

Check they were recognised, then build:

    python cli.py scan chandrayaan-3
    python cli.py build chandrayaan-3

`scan` only reports what matched; nothing is written until `build` runs.
`build` is extract + caption + ingest in one pass (see "What build does"
below) and prints `ok` per satellite when done. Then:

    python cli.py query chandrayaan-3 "your question"

There is no configuration file to fill in — the document type of each
file is recognised from its name. `satellite.yaml` is only needed to
override a default (embedding model, chunk sizes) or to map a filename
the patterns cannot recognise.


## What build does

    extract    Reads each document with the right parser. Procedure
               documents give index rows plus procedures; report PDFs
               give sections; Word files give sections and tables; the
               telecommand log gives one record per command. Figures are
               saved for the next stage.

    caption    Describes each figure with a local vision model and stores
               the description as a searchable chunk. That is what lets
               "show me the recovery diagram" find an image: the diagram
               is not searchable, its description is. Resumes if
               interrupted.

    ingest     Applies the per-document-type chunk sizes, embeds
               everything, writes the vector store and keyword index.

Each stage runs alone when only that part needs redoing:

    python cli.py extract aditya-l1 --force
    python cli.py caption aditya-l1
    python cli.py ingest  aditya-l1 --reset


## Which filenames are recognised

    FCP           "FCP", "flight control procedure"
    MOR           "MOR", "mission operations", "mission plan", "mission report"
    OPS_MANUAL    "operations manual", "ops manual", "manual"
    GUIDELINES    "guidelines", "guidance", "normal phase"
    TCH           "TCH", "telecommand", "command history", or any .txt

Case, underscores and camel case do not matter: AL1OperationsManual.docx,
ops_manual.docx and CH3-Operations-Manual.docx all resolve to OPS_MANUAL.

`scan` prints what matched and lists anything it could not place, so a
mismatch is visible before building rather than after. For a file the
patterns cannot recognise, rename it after its type or map it in
satellite.yaml:

    documents:
      FCP:
        file: some_unusual_name.pdf


## Layout

    SatelliteRAG\
        cli.py
        core\
            registry.py      satellite discovery, document detection
            extract.py       documents to structured JSON
            caption.py       figure descriptions via a vision model
            chunking.py      per-document-type splitting
            embeddings.py    offline embedding model registry
            ingest.py        embed into the vector store
            retrieve.py      hybrid search, one satellite or many
        models\              embedding models, downloaded once, shared
        satellites\
            aditya-l1\
                satellite.yaml   optional
                documents\       the source files
                extracted\       chunks, captions, commands, images
                vectorstore\     ChromaDB + BM25
            chandrayaan-3\
                ... same shape
        scripts\              fetch_adityal1_docs.py, one-time public-data fetch
        lora\                 LoRA fine-tuning demo: build_dataset.py, train.py, query.py
        slides\               auto-generated PPTX decks from real pipeline output


## Install

Python 3.11 or 3.12. Newer versions often have no matching CUDA wheel.

PyTorch goes first, from the index matching the CUDA version that
`nvidia-smi` reports:

    pip install torch --index-url https://download.pytorch.org/whl/cu128
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

The version must end in `+cuXXX` and print True. Then:

    pip install -r requirements.txt
    python cli.py prefetch minilm-l6
    ollama pull qwen2.5vl

Order matters. pip resolves torch against PyPI, which serves CPU-only
wheels, so installing requirements.txt first pulls the CPU build. torch
is deliberately absent from requirements.txt for the same reason.

LibreOffice is needed only for legacy `.doc` files, and is found
automatically once installed.


## Which model server

Every model call goes through `core/llm.py`. Ollama is the default and
needs nothing set. A local vLLM server (Linux or WSL2 only) is one
environment variable away:

    set SATRAG_LLM=vllm                          (PowerShell: $env:SATRAG_LLM="vllm")
    set SATRAG_VLLM_URL=http://localhost:8000/v1

Temperature 0, a fixed seed and a repetition penalty of 1.0 are sent
explicitly to either server, so an answer is reproducible and does not
depend on a server default. Answers are capped at 768 tokens.

Batch jobs (`caption`, `lora/build_dataset.py`) can keep several requests
in flight:

    set SATRAG_LLM_CONCURRENCY=4

This only helps if the server overlaps requests: start Ollama with
`OLLAMA_NUM_PARALLEL=4`, or use vLLM. Measured on the RTX 3050 with a
1.5B model, that made a 102-request batch 2.8x faster than Ollama's
default. Each parallel slot reserves its own context, so with the 7B text
model or the vision model on a 6 GB card it may not fit; the default stays
at 1 for that reason. `bench/README.md` has the measurements.


## Querying

Three ways to ask a question, depending on how many you have.

### One-shot: a single prompt

    python cli.py query aditya-l1 "What are Aditya-L1's science objectives?"

Retrieves, generates one cited answer, and exits. Useful flags:

    -k <n>              how many chunks to retrieve (default 5)
    --doc-types T ...   restrict to these document types
    --subsystem S       restrict to one subsystem
    --no-answer         show retrieved sources only, skip generation
    --show-text         print the retrieved snippet text, not just citations
    --model NAME        a different Ollama text model for the answer

Against the public demo data that ships in `satellites/aditya-l1/`:

    python cli.py query aditya-l1 "What payloads does Aditya-L1 carry?"
    python cli.py query aditya-l1 "What launch vehicle was used, and from where was it launched?"
    python cli.py query aditya-l1 "What is the deadline for the first AO cycle proposal?"
    python cli.py query aditya-l1 "VELC payload" --doc-types OPS_MANUAL MOR
    python cli.py query all "Aditya-L1 payloads"

The examples below need a satellite with procedure and telecommand
documents (FCP / TCH), which the shipped public demo data does not
include — see "Adding a satellite":

    python cli.py query chandrayaan-3 "how to switch on OBC"
    python cli.py query chandrayaan-3 "which FCP covers gyro power on"
    python cli.py query chandrayaan-3 "what commands were uplinked on 2026-06-09"
    python cli.py query chandrayaan-3 "MRS recovery diagram"

The query is routed before it is run. A date or a mention of commands
goes to the telecommand store and comes back as records; everything else
goes to the knowledge store. Asking for a diagram lifts figure captions;
asking "how do I" pushes index rows below the procedures, while asking
"which FCP" leaves them where they are.

Three routers are available through `retrieval.router`. `soft`, the
default, searches the procedure document for every question and fuses it
in; `rules` decides from the wording whether a procedure is wanted and
puts procedure chunks first; `llm` asks the local text model. Soft scored
highest on the benchmark (`results/router_retrieval.csv`).

### Interactive: several questions in a row

    python cli.py shell aditya-l1

Loads the model and vector store once and takes one question after
another, which avoids paying that startup cost per question. In-shell
commands:

    :sat <slug|all>    switch satellite (or search across all of them)
    :k <n>             change how many chunks are retrieved
    :doc <TYPE ...>    filter by document type, "off" to clear
    :text              toggle showing retrieved snippet text
    :exit              quit

### Question sets: a file of questions, one answer each

    python cli.py query aditya-l1 --file questions.txt

`questions.txt` is one question per line; blank lines and lines starting
with `#` are skipped. Runs each line through the same one-shot path as a
single query — retrieve, generate a cited answer, print it — one after
another. All the `query` flags above (`-k`, `--doc-types`, `--no-answer`,
etc.) apply to every question in the file.


## Chunking

Each document type has its own splitting policy:

    FCP           2500 chars, cuts only at end-of-procedure or notes
    MOR           1000 chars, paragraph boundaries
    OPS_MANUAL    1200 chars
    GUIDELINES    1200 chars
    toc_index     atomic, never split
    image_caption atomic, never split
    command_log   atomic, never split

FCP gets a larger budget because a procedure cut in half produces a chunk
that starts mid-sequence. Index rows, captions and command records are
atomic regardless of length.

Extraction emits whole sections and chunking applies the size policy at
ingest, so changing a chunk budget never requires re-extraction.

Chunks are never mixed across document types. Every chunk carries its
doc_type, and `--doc-types` filters both the semantic and the keyword
pass. Override per satellite in satellite.yaml:

    chunking:
      MOR:
        max_chars: 1500
        overlap: 250


## Storage

Two collections per satellite, split by how they are accessed:

    knowledge   procedures, guidelines, manual prose, image captions
    commands    the telecommand log, filtered by date, CID or subsystem

Document types stay separated inside knowledge through the doc_type field
rather than through separate collections, so ranking can put all five
results in FCP when that is where the answer is.


## Embedding models

    minilm-l6       384   fastest; level with the best pair in the confidential-corpus 100-question benchmark
    bge-m3         1024   multilingual, long context, heavy
    bge-large-en   1024   strong English-only
    mxbai-large    1024   strong English-only
    gte-large-en   1024   needs trust_remote_code
    e5-large       1024   multilingual, prefixes handled automatically

Set per satellite in satellite.yaml:

    embedding:
      model: bge-m3

Satellites may use different models. Cross-satellite results are fused by
rank rather than by score, so mixed models still merge correctly.

Batch size is chosen from the card's VRAM and the model's dimension:

    VRAM     minilm-l6 (384)   1024-dim models
    4 GB           32                16
    6 GB           48                24
    10 GB          64                32
    24 GB         128                64
    CPU            16                16

If a batch still runs out of memory the size halves and retries.


## Commands

    python cli.py models                    embedding model registry
    python cli.py prefetch <name> ...       download models
    python cli.py new <slug>                scaffold a satellite folder
    python cli.py list                      state of every satellite
    python cli.py scan [<slug>]             check document detection
    python cli.py build <slug|all>          extract + caption + embed
    python cli.py extract <slug|all>        documents to JSON
    python cli.py caption <slug|all>        describe figures
    python cli.py ingest <slug|all>         embed into the vector store
    python cli.py query <slug|all> "..."    one-shot search
    python cli.py shell <slug|all>          interactive search
