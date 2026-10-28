# SatelliteRAG — project context and handoff

> Paste or upload this at the start of a new chat to restore context. It
> describes the system as it stands, the measurements behind the current
> configuration, and what is still open. No document contents are
> included, only methodology and numbers.

Supersedes the earlier `PROJECT_CONTEXT.md` / `ISRO_RAG_PROJECT_CONTEXT.md`,
which describe `ISRO_THESIS\Isro_data` (the four-script single-satellite
pipeline). That project still exists and is untouched; SatelliteRAG
replaced it.


## 1. What this is

A fully-local retrieval assistant over spacecraft mission-operations
documents. An operator asks a question in plain language and gets a cited
answer. Nothing leaves the machine.

Built for Aditya-L1 first, then generalised: the same five document types
exist for roughly 24 satellites, so the system takes one folder per
satellite and shares no state between them.

**Constraints, unchanged:**
- Source documents are confidential. Never share them, never reproduce
  their contents, describe methodology only.
- Offline. Ollama plus sentence-transformers, no cloud, no API keys.
- Repository private.
- Code should read as hand-written.

**Author:** Anurag Prakash, IIT (BHU) Varanasi.
**Guide:** Prof. Ravi Shankar Singh.

**Stack:** Python, PyMuPDF, python-docx, LibreOffice, ChromaDB,
BM25 (`rank_bm25`), sentence-transformers, Reciprocal Rank Fusion
(RRF, k=60), a ms-marco cross-encoder for optional reranking, Qwen2.5-VL
for figure captioning and Qwen2.5-7B-Instruct for generation, both via
Ollama.

Retrieval is **hybrid**: a dense semantic pass and a BM25 keyword pass,
fused by RRF. That fusion is the largest single contributor to retrieval
quality in this system, ahead of both the embedding model and the
reranker (section 6).

**Location:** `C:\Users\anura\Desktop\SatelliteRAG`
**Previous project:** `C:\Users\anura\Desktop\ISRO_THESIS\Isro_data`


## 2. Adding a satellite

Make a folder, copy the documents in, run one command.

    satellites\chandrayaan-3\documents\
        CH3_FCP_v2.pdf
        Chandrayaan3_MissionOperationsReport.pdf
        CH3_Operations_Manual.docx
        CH3_Guidelines_NormalPhase.doc
        telecommand_history_ch3.txt

    python cli.py scan chandrayaan-3      check they were recognised
    python cli.py build chandrayaan-3     extract, caption, embed

No configuration is required. Each document's type is recognised from its
filename: "FCP" or "flight control procedure", "MOR" / "mission
operations" / "mission report", "operations manual" / "manual",
"guidelines" / "guidance" / "normal phase", "TCH" / "telecommand" or any
`.txt`. Case, underscores and camel case do not matter. `satellite.yaml`
is optional and only holds overrides.


## 3. Layout

    SatelliteRAG\
        cli.py                 one entry point, all commands
        evaluate.py            100-question retrieval evaluation
        testset.py             the labelled questions
        compare_models.py      embedder comparison, stage 1
        compare_rerankers.py   embedder x reranker, stage 2
        collect_results.py     merge every result file into one table
        core\
            registry.py        satellite discovery, filename detection
            extract.py         documents to structured JSON
            caption.py         figure descriptions via a vision model
            chunking.py        per-document-type splitting
            embeddings.py      offline embedding registry
            ingest.py          embed into the vector store
            rerank.py          cross-encoder reranking
            retrieve.py        hybrid search, one satellite or many
            generate.py        cited answers via a local text model
        models\                every model, downloaded once, shared
        satellites\<slug>\
            satellite.yaml     optional
            documents\         the source files
            extracted\         chunks, captions, commands, images
            vectorstore\       ChromaDB + BM25
        results\               all measurements

`cli.py` holds almost no logic; it reads the command and calls `core/`.
The modules import cleanly, which is how a UI would use them:

    from core import registry, generate
    sat = registry.load_satellite("aditya-l1")
    answer, hits = generate.ask([sat], "how to switch on OBC")


## 4. Pipeline

**extract** reads each document with the right parser. The procedure
document yields index rows and procedure bodies separately; report PDFs
yield sections; Word files yield sections and tables; the telecommand log
yields one record per command. Figures are saved for the next stage.
Extraction parses structure only and emits whole sections, so changing a
chunk size later does not require re-extracting.

**caption** describes each figure with a local vision model and stores
the description as a searchable chunk. That is what lets "show me the
recovery diagram" reach an image: the diagram is not searchable, its
description is. Captions inherit the section heading, procedure number
and subsystem of the nearest chunk, located by bisect so it works when a
figure's page has no chunk of its own. Resumes if interrupted.

**ingest** applies the per-document-type chunk budgets, embeds
everything, and writes the vector store and keyword index.

**retrieve** routes the query (a date or an explicit command phrase goes
to the command store, everything else to knowledge), runs a dense pass
and a BM25 pass fused by Reciprocal Rank Fusion, optionally reranks with
a cross-encoder, and prepends a targeted search when the question asks
for a procedure or a figure.

**generate** turns the retrieved passages into a cited answer with
`qwen2.5:7b-instruct`. Diagram questions are answered from the caption
directly, because a text model asked to "show" a figure tends to reply
that it cannot. Generation failure returns the sources rather than
crashing.


## 5. Design decisions worth knowing

**Hybrid retrieval: BM25 alongside the dense pass, fused by RRF.** This
is the single largest contributor to retrieval quality here, worth 3.7
times the reranker (section 6).

The reason is the corpus. These documents are dense with identifiers that
carry no semantic content and cannot be paraphrased: procedure numbers
like FCP-5201, command codes like OBCS2370, hardware names like DC-DC-11
and NCF1. An embedding model places "OBCS2370" near other alphanumeric
tokens; BM25 matches it exactly. Conversely a question phrased "how do I
bring the onboard computer up" shares no keywords with a procedure titled
"OBC-1 ON / OBC-2 ON", and only the dense pass finds it. The two failure
modes barely overlap, which is why fusing them gains so much.

Reciprocal Rank Fusion at k=60 combines them by position rather than by
score, so a BM25 score of 7.4 and a cosine similarity of 0.62 never have
to be made commensurable. It is also what lets cross-satellite search
merge results from satellites embedded with different models.

**The BM25 tokenizer indexes compound identifiers whole and in parts.**
`OBC-1` becomes `obc-1`, `obc` and `1`; `L1/L2` becomes `l1/l2`, `l1` and
`l2`. Without the parts, a procedure titled "OBC-1 ON / OBC-2 ON" scores
zero against a search for OBC, because `obc-1` and `obc` are different
tokens. Without the whole, an exact search for `DC-DC-11` cannot be
distinguished from one for `DC`.

**Both passes get identical filters.** Letting the keyword pass be more
permissive than the semantic one means chunks the semantic filter
excluded return through the side door, which defeats the filter.

**Two collections per satellite, split by access pattern.** `knowledge`
holds prose and captions and is searched by meaning; `commands` holds the
telecommand log and is looked up by exact filter. The access patterns
conflict, and mixing them makes both noisier. A date query returns every
matching record in time order, not the five most semantically similar.

**Chunk budgets are per document type.**

    FCP           2500 chars, cuts at end-of-procedure or notes markers
    MOR           1000 chars, paragraph boundaries
    OPS_MANUAL    1200 chars
    GUIDELINES    1200 chars
    toc_index     atomic, never split
    image_caption atomic, never split
    command_log   atomic, never split

A procedure cut in half starts mid-sequence and cannot be followed, so it
gets more than double the prose budget. Index rows, captions and command
records are atomic by type rather than by length.

**No subsystem filtering.** Guessing the subsystem from the question and
filtering on the guess loses answers: the documents file "auto thruster
firing" under AOCS, not PROPULSION; "reaction wheel logics" under AOCS,
not WHEEL. A wrong guess deletes the correct chunk before anything scores
it. Removing the filter was worth roughly +0.05 MRR.

**Index subsystem is read, not inferred.** The procedure index states the
owning subsystem in its own column; that value is mapped directly,
including the `PAYLAOD` misspelling in the source.

**The procedure prepend searches the whole procedure document.** The
Aditya-L1 file holds 6 procedure bodies and 225 index rows, so an index
row is the only chunk that can answer most procedure questions.
Restricting the prepend to bodies cost about 30 points of Hit@1.

**Cross-satellite search fuses by rank, not score**, so satellites using
different embedding models still merge correctly.


## 6. Measured results

100 labelled questions, `results/all_results.csv`. Accept conditions were
tightened after an early run: several were loose enough to admit false
positives, so figures measured before that are not comparable.

### Embedders alone, dense mode

Dense isolates the embedding model. Hybrid mixes in BM25, which does not
depend on the embedder and flattens the differences.

| Model | Dim | Hit@1 | Hit@5 | Recall | MRR | Build |
|---|---|---|---|---|---|---|
| bge-small-en | 384 | 72% | 88% | 90% | **0.787** | 25s |
| bge-base-en | 768 | 72% | 86% | 90% | 0.786 | 41s |
| mpnet-base | 768 | 71% | 87% | 93% | 0.784 | 50s |
| bge-large-en | 1024 | 74% | 85% | 88% | 0.783 | 109s |
| minilm-l6 | 384 | 69% | 86% | **94%** | 0.774 | 12s |
| mxbai-large | 1024 | 71% | 82% | 87% | 0.758 | 96s |
| e5-large | 1024 | 68% | 84% | 87% | 0.750 | 93s |
| gte-base | 768 | 64% | 86% | 90% | 0.733 | 38s |
| bge-m3 | 1024 | 65% | 82% | 86% | 0.721 | 34s |
| e5-base | 768 | 55% | 74% | 80% | 0.635 | 50s |

Size predicts nothing. The 133 MB model matches the 1.34 GB one. BGE-M3,
the previous default at 2.3 GB, ranks ninth of ten.

### What BM25 and RRF contribute

Comparing each embedder in dense mode against the same embedder in hybrid
mode, with no reranker in either. The only difference is the BM25 pass
and the fusion.

| Embedder | MRR dense | MRR hybrid | Δ | Recall dense | Recall hybrid | Δ |
|---|---|---|---|---|---|---|
| minilm-l6 | 0.774 | 0.844 | **+0.070** | 94% | 97% | +3 |
| mpnet-base | 0.784 | 0.841 | +0.057 | 93% | 96% | +3 |
| bge-small-en | 0.787 | 0.833 | +0.046 | 90% | 96% | +6 |
| bge-base-en | 0.786 | 0.829 | +0.043 | 90% | 98% | **+8** |
| bge-large-en | 0.783 | 0.810 | +0.027 | 88% | 98% | **+10** |

Stacked for the recommended embedder:

    dense only                0.774
    + BM25 + RRF              0.844    +0.070
    + cross-encoder rerank    0.863    +0.019

**The keyword pass is worth 3.7 times the reranker**, and it is the only
stage that raises recall — the reranker moves Hit@10 by 0.000, because
reordering cannot find what retrieval missed. On bge-large-en, BM25 adds
ten points of recall: a tenth of the questions are answered by a chunk
the dense pass never returned at all.

Note the inversion. The embedders that look best in dense mode gain least
from fusion, and the one that looks worst gains most. Ranking embedders
in dense mode alone would have picked bge-small-en; measured in the
configuration that actually ships, minilm-l6 comes out ahead.

### Embedder x reranker, hybrid mode

Best of 30 combinations:

| Embedder | Reranker | Hit@1 | Hit@5 | MRR | sec |
|---|---|---|---|---|---|
| bge-base-en | ms-marco-L-6 | 82% | 92% | 0.867 | 9.6 |
| bge-base-en | ms-marco-L-12 | 81% | 92% | 0.866 | 18.7 |
| minilm-l6 | ms-marco-L-6 | 82% | 91% | 0.864 | 9.5 |
| minilm-l6 | ms-marco-L-4 | 81% | 92% | 0.863 | 6.7 |
| bge-base-en | ms-marco-L-4 | 81% | 93% | 0.863 | 6.8 |
| minilm-l6 | none | 77% | **94%** | 0.844 | 0.0 |

### The noise floor

`minilm-l6 + none` was measured twice and returned 0.839 and 0.844.
ChromaDB's index is approximate and fp16 rounds, so **differences under
about 0.005 MRR are not real**. That makes the top ten combinations
statistically tied.

### Negative results

**Larger rerankers were consistently worse.** On every embedder tested:

| Reranker | Size | MRR (minilm-l6) | Time |
|---|---|---|---|
| ms-marco-L-4 | 77 MB | **0.863** | 7s |
| ms-marco-L-6 | 90 MB | 0.864 | 10s |
| ms-marco-L-12 | 133 MB | 0.860 | 19s |
| bge-reranker-base | 1.1 GB | 0.775 | 42s |
| bge-reranker-v2-m3 | 2.3 GB | 0.846 | 113s |
| mxbai-rerank-xsmall | 142 MB | 0.776 | 19s |

`bge-reranker-base` scores below no reranker at all (0.844), on all five
embedders. Both BGE reranker generations lose to ms-marco here.

**The reranker's value shrank as retrieval improved.** An intermediate
version of this pipeline showed +0.39 MRR from reranking, which was the
symptom that led to finding a real bug in the fusion stage. With that
fixed the gain is +0.019, and reranking costs Hit@5: 94% to 91%.

Because the pipeline sends the top 5 to the language model and the model
reads all five, Hit@5 is the operationally relevant number, and the
reranker is slightly negative there.


## 7. Recommended configuration

    embedding:
      model: minilm-l6

    retrieval:
      rerank: false          # Hit@5 94%, no second model
      # rerank: true         # Hit@1 81%, MRR 0.863, +7s per 100 queries
      # rerank_model: cross-encoder/ms-marco-MiniLM-L-4-v2

167 MB of models with reranking, 90 MB without. Everything within 0.005
MRR of the best measured combination, at a fraction of the size.


## 8. Data, Aditya-L1

    chunks.json          781 items
    image captions       210
    command records      203
    after splitting      1,474 knowledge vectors
    per document type    FCP 235, MOR 698, OPS_MANUAL 255, GUIDELINES 286
    figures extracted    359 (210 captioned, 149 below the size threshold)
    FCP index rows       225
    FCP procedure bodies 6

The procedure document is a sample: it carries the complete 225-row index
but only the first six procedures. Six is correct for this file. With the
full document that number rises without any code change.


## 9. Environment

Python 3.11 in `.venv`. Python 3.13+ often has no matching CUDA wheel.

PyTorch must be installed **before** `requirements.txt`, from the wheel
index matching the CUDA version `nvidia-smi` reports. pip resolves torch
against PyPI, which serves CPU-only wheels, so the order matters and
torch is deliberately absent from `requirements.txt`.

    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    python cli.py prefetch minilm-l6
    ollama pull qwen2.5vl            # figure captioning
    ollama pull qwen2.5:7b-instruct  # answer generation

Currently: torch 2.11.0+cu128, RTX 3050 6 GB. Batch size is chosen from
VRAM and model dimension, with halve-and-retry on out-of-memory.

LibreOffice is needed only for legacy `.doc` files. If absent, open the
file and save it as `.docx`.


## 10. Commands

    python cli.py models                    embedding model registry
    python cli.py prefetch <name> ...       download models
    python cli.py new <slug>                scaffold a satellite folder
    python cli.py list                      state of every satellite
    python cli.py scan [<slug>]             check document detection
    python cli.py build <slug|all>          extract + caption + embed
    python cli.py extract <slug|all>        documents to JSON
    python cli.py caption <slug|all>        describe figures
    python cli.py ingest <slug|all>         embed into the vector store
    python cli.py query <slug|all> "..."    ask a question
    python cli.py shell <slug|all>          interactive search

    python evaluate.py --compare            reranker on vs off
    python evaluate.py --compare --widen 50 larger candidate pool
    python compare_models.py --models ...   embedder comparison
    python compare_rerankers.py --top 3     reranker comparison
    python collect_results.py               merge all results


## 11. Open items

- **Four questions miss at 5** in every configuration: `power_budget`,
  `odhs_diag`, `xband_system`, `cmd_station`. They sit outside the
  candidate pool, so no reranker can reach them. Try
  `evaluate.py --compare --widen 50` to see whether pool size is the
  ceiling; if recall rises, that is worth more than any model swap.
- **Chunk size has never been swept.** The per-type budgets were chosen
  by reasoning, not measured. Extraction and chunking are separate
  stages, so a sweep only needs re-ingesting.
- **The Streamlit interface is not ported.** The old `app.py` has chat
  with citations, a command table with CSV download, inline figures and
  audit logging. `generate.ask()` already returns what it needs; the work
  is a satellite dropdown and swapping the import.
- **LLM router with Pydantic** — replace the regex `route()` with a
  validated `QueryPlan`. Not started.
- **Multimodal retrieval** — captions only at present. Page-image
  retrieval was on the old roadmap and remains unexplored.
- **Verify the GitHub repository is private.** Flagged as urgent in the
  previous context file and never confirmed.
- **Only Aditya-L1 exists.** The multi-satellite design is tested with
  synthetic second satellites but has not been run on real documents for
  another mission.
- **Seminar date.** The previous context says 26 July 2026, which has
  passed. If the deck still carries the old figures (1,133 vectors, 201
  commands, BGE-M3, MRR 0.967 from the 15-question set), all four are
  now wrong.


## 12. Documents in this folder

- `README.md` — usage
- `SETUP.txt` — installation, step by step
- `COMPARISON.md` — SatelliteRAG versus the previous pipeline, including
  the bugs found in both
- `PROJECT_CONTEXT.md` — this file
