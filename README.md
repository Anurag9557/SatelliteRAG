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

    Hit@1  0.83     Hit@5  0.92     Recall  0.96     MRR  0.871

## Documentation

| File | Contents |
|---|---|
| `SETUP.txt` | installation, step by step |
| `GIT_SETUP.md` | pushing to GitHub, and how to run it after cloning |
| `PROJECT_CONTEXT.md` | architecture, design decisions, every measured number |
| `COMPARISON.md` | this system versus the previous pipeline |
| `results/` | raw measurements |


## Adding a satellite

Make a folder, copy the documents in, run one command.

    satellites\chandrayaan-3\documents\
        CH3_FCP_v2.pdf
        Chandrayaan3_MissionOperationsReport.pdf
        CH3_Operations_Manual.docx
        CH3_Guidelines_NormalPhase.doc
        telecommand_history_ch3.txt

    python cli.py build chandrayaan-3

There is no configuration to fill in. The document type of each file is
recognised from its name.


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


## Querying

    python cli.py query aditya-l1 "how to switch on OBC"
    python cli.py query aditya-l1 "which FCP covers gyro power on"
    python cli.py query aditya-l1 "what commands were uplinked on 2026-06-09"
    python cli.py query aditya-l1 "MRS recovery diagram"
    python cli.py query aditya-l1 "safe mode" --doc-types GUIDELINES MOR
    python cli.py query all "safe mode recovery"
    python cli.py shell aditya-l1

The query is routed before it is run. A date or a mention of commands
goes to the telecommand store and comes back as records; everything else
goes to the knowledge store. Asking for a diagram lifts figure captions;
asking "how do I" pushes index rows below the procedures, while asking
"which FCP" leaves them where they are.


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

    minilm-l6       384   fastest, best hybrid MRR in the Aditya-L1 benchmark
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
