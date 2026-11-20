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
and subsystem of the section the figure sits in. Extraction reads that off
the document: the figure's position on its PDF page, or the paragraph that
holds it in a Word file. The model is also given the text printed beside
the figure. Resumes if interrupted.

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
is the single largest contributor to retrieval quality here, +0.06 MRR,
and the one gain the test set is large enough to confirm (section 6).

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

**Procedures are fused, not forced.** Under the default soft router the
procedure document is searched for every question and joins the fusion as
a third list. The rules router instead classifies the question and puts
two procedure chunks first; measured, that costs about 0.03 MRR, because
one question in five is classified wrongly (section 6, Routers).

**Cross-satellite search fuses by rank, not score**, so satellites using
different embedding models still merge correctly.


## 6. Measured results

100 labelled questions, `results/all_results.csv`, measured on 12 November
2026 with the pipeline as it ships: soft router, the top 20 results scored
in both stages. The August figures are kept locally in
`results/archive-2026-08/` (not in git) and are not comparable with these; the last part of this section says why.

**Note, 16 November 2026.** The figure captions were written again that
day with the right section as context (most had been given the following
section's). The working store now holds the new captions and scores Hit@1
84%, Hit@5 92%, Hit@10 97%, MRR 0.879 with the default settings, against
the 93% and 0.883 below; the difference is inside the interval. The tables
in this section, and the per-model stores they came from, were built with
the earlier captions and have not been run again. The earlier captions and
store are kept (`image_captions.before-context-fix.json`,
`vectorstore_before-caption-fix/`).

`MRR strict` in the result files is the same measure without the eleven
questions that a large share of the store satisfies (`testset.BROAD`). It
moves with the headline, 0.01 to 0.02 lower, so the ranking does not lean on
the easy questions. One question, `odhs_diag`, cannot be hit on this build,
so the ceiling is 99%.

### Embedders alone, dense mode

Dense isolates the embedding model. Hybrid mixes in BM25, which does not
depend on the embedder and flattens the differences.

| Model | Dim | Hit@1 | Hit@5 | Hit@10 | Found in 20 | MRR |
|---|---|---|---|---|---|---|
| bge-large-en | 1024 | 77% | 88% | 92% | 95% | **0.816** |
| bge-small-en | 384 | 74% | 89% | 92% | 95% | 0.808 |
| bge-base-en | 768 | 74% | 87% | 92% | 95% | 0.807 |
| mpnet-base | 768 | 73% | 88% | 94% | 94% | 0.800 |
| minilm-l6 | 384 | 71% | 87% | **96%** | **96%** | 0.792 |
| mxbai-large | 1024 | 73% | 84% | 89% | 93% | 0.781 |
| gte-base | 768 | 66% | 87% | 92% | 96% | 0.754 |
| e5-large | 1024 | 68% | 83% | 87% | 92% | 0.752 |
| bge-m3 | 1024 | 67% | 82% | 87% | 91% | 0.737 |
| e5-base | 768 | 57% | 76% | 82% | 83% | 0.655 |

Size predicts nothing. The 133 MB bge-small-en sits within 0.01 of the
1.3 GB bge-large-en, and BGE-M3, the previous default at 2.3 GB, ranks
ninth of ten. The order inside the top five is not established, though:
see the intervals below.

### What BM25 and RRF contribute

Each embedder in dense mode against the same embedder in hybrid mode, no
reranker in either, both scored on the top 20.

| Embedder | MRR dense | MRR hybrid | Change | Hit@10 dense | Hit@10 hybrid |
|---|---|---|---|---|---|
| minilm-l6 | 0.792 | 0.852 | **+0.060** | 96% | 97% |
| mpnet-base | 0.800 | 0.854 | +0.054 | 94% | 96% |
| bge-small-en | 0.808 | 0.820 | +0.012 | 92% | 95% |
| bge-base-en | 0.807 | 0.810 | +0.003 | 92% | 94% |
| bge-large-en | 0.816 | 0.817 | +0.001 | 92% | 96% |

Stacked for the embedder that ships:

    dense only                0.792
    + BM25 + RRF              0.852    +0.060    interval +0.014 to +0.108
    + cross-encoder (L-6)     0.884    +0.032    interval -0.015 to +0.079

The keyword pass is the one gain this test set can confirm. The reranker
adds about half as much and its interval reaches zero, so on a hundred
questions it is a likely improvement and not a shown one. It does what it
is for, though: Hit@1 goes from 77% to 84%.

Fusion helps the two small general-purpose embedders and does almost
nothing for the three BGE models. That is why minilm-l6, fifth in dense
mode, ends level with the best once BM25 is in.

On recall, BM25 adds one to four points of Hit@10. An earlier version of
this section credited it with three to ten, which came from scoring dense
on the top 10 and hybrid on the top 20.

### Embedder x reranker, hybrid mode

Best of 30 combinations, each run through the whole pipeline with the
reranker in its real place:

| Embedder | Reranker | Hit@1 | Hit@5 | MRR | sec |
|---|---|---|---|---|---|
| bge-base-en | ms-marco-L-4 | 84% | 94% | 0.887 | 9.8 |
| bge-base-en | ms-marco-L-12 | 84% | 93% | 0.886 | 19.9 |
| mpnet-base | ms-marco-L-4 | 83% | **95%** | 0.886 | 10.2 |
| minilm-l6 | ms-marco-L-4 | 83% | 94% | 0.885 | 9.3 |
| minilm-l6 | ms-marco-L-6 | 84% | 93% | 0.884 | 11.7 |
| minilm-l6 | none | 77% | **95%** | 0.852 | 3.6 |

The fifteen combinations of an embedder with an ms-marco reranker span
0.860 to 0.887. The best of them against the pair that ships is +0.003
with an interval of -0.028 to +0.036: a tie.

### How sure the differences are

Retrieval is deterministic. The same configuration run four times, in two
separate processes, returned the same rank for every question, so there is
no run-to-run noise to report. An earlier version of this file gave a
"noise floor" of 0.005 MRR from two runs that disagreed; that gap was
never reproduced and was not random.

What is uncertain is the sample: a hundred questions. `collect_results.py`
resamples the questions (paired bootstrap, 10,000 draws) and writes
`results/intervals.csv`:

| Difference in MRR | Estimate | 95% interval | Shown? |
|---|---|---|---|
| minilm-l6: hybrid over dense | +0.060 | +0.014 to +0.108 | yes |
| minilm-l6: ms-marco-L-6 over no reranker | +0.032 | -0.015 to +0.079 | no |
| minilm-l6: ms-marco-L-4 over no reranker | +0.033 | -0.018 to +0.084 | no |
| best pair over minilm-l6 + ms-marco-L-6 | +0.003 | -0.028 to +0.036 | no |
| dense: bge-large-en over minilm-l6 | +0.025 | -0.030 to +0.080 | no |
| dense: minilm-l6 over bge-m3 | +0.055 | -0.020 to +0.132 | no |
| dense: bge-large-en over e5-base | +0.161 | +0.082 to +0.243 | yes |

So the test set can separate a good embedder from a poor one and hybrid
from dense. It cannot rank the good embedders against each other, or the
ms-marco rerankers. Settling those needs more questions, not more runs.

### Routers

Retrieval under each router, working store, top 10 scored
(`results/router_retrieval.csv`):

| Router | Reranker | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| rules | off | 74% | 94% | 96% | 0.823 |
| rules | on | 79% | 94% | 97% | 0.850 |
| soft | off | 77% | 95% | 97% | 0.851 |
| soft | on | **84%** | 93% | 97% | **0.883** |

The rules router decides whether a question wants a procedure and, if so,
puts the two closest procedure chunks first. It gets that decision right
80% of the time on the benchmark questions, and when it is wrong the
forced chunks push the answer down. The soft router searches the procedure
document for every question and lets fusion place the result. Soft over
rules, reranker on, is +0.033 with an interval of +0.001 to +0.068, and
soft is the default.

Routing accuracy (`results/router_accuracy.csv`):

| Set | Router | Intent | Store | Per question |
|---|---|---|---|---|
| 100 benchmark questions | rules | 80% | 100% | under 1 ms |
| 100 benchmark questions | llm (qwen2.5:7b-instruct) | 93% | 99% | 1.3 s |
| 40 routing questions | rules | 52.5% | 90% | under 1 ms |
| 40 routing questions | llm (qwen2.5:7b-instruct) | 92.5% | 97.5% | 0.7 s |

The 40 routing questions were written after the rules, around their weak
spots, so they are a regression check and overstate how often the rules
fail in ordinary use. The model router is far more accurate and costs a
second per question; the soft router gets most of the retrieval benefit
for nothing, by not needing the procedure decision at all.

### Negative results

**The larger rerankers were no better.** On minilm-l6:

| Reranker | Size | Hit@1 | MRR | Time |
|---|---|---|---|---|
| none | | 77% | 0.852 | |
| ms-marco-L-4 | 77 MB | 83% | **0.885** | 9s |
| ms-marco-L-6 | 91 MB | 84% | 0.884 | 12s |
| ms-marco-L-12 | 134 MB | 83% | 0.879 | 20s |
| bge-reranker-base | 1.1 GB | 73% | 0.822 | 44s |
| mxbai-rerank-xsmall | 153 MB | 70% | 0.789 | 21s |

`bge-reranker-base` and `mxbai-rerank-xsmall` score below no reranker at
all on minilm-l6 and mpnet-base, and never reach the ms-marco family on
any embedder. Within the ms-marco family depth buys nothing: L-4 matches
L-12 at half the time.

**The reranker costs a little at Hit@5.** 95% without it, 93% with it on
minilm-l6. The pipeline hands the top 5 to the language model, so with the
reranker on the first citation is right more often and the context is
complete slightly less often.

### Why the August figures were retired

The figures this section used to carry (dense 0.774, hybrid 0.844,
reranked 0.863, "the keyword pass is worth 3.7 times the reranker") were
measured in August and do not describe the pipeline now. Five things were
wrong with them:

- **The procedure prepend was not running.** Retrieval filtered that
  second search on a metadata field called `source_doc`; ingestion has
  always written `doc_type`. The filter matched nothing, so the step did
  nothing, and every August number was measured without it. Switching the
  step off today reproduces the August file rank for rank. The field name
  was corrected with the routing work on 28 October, the step began to
  run, and under the rules router it cost about 0.03 MRR.
- **The reranker table used a shortcut.** It retrieved a pool once and
  reranked that saved list last. The pipeline reranks first and then
  applies the figure and procedure rules, which is a different order.
  `compare_rerankers.py` was changed in September to run the real pipeline
  but the table was never run again.
- **The Hit@10 column was wrong in all thirty reranker rows.** It held
  "found anywhere in the top 20". True Hit@10 was one to six points lower.
- **Dense was scored on the top 10 and hybrid on the top 20**, so the
  recall the keyword pass appeared to add was partly the deeper pool.
- **The command log was misread.** Every row ends with a pipe, and the
  columns were counted from the end without allowing for it, so each
  trailing field was one place out: the destination station came out
  blank and the status column held the source station. The page header
  that repeats every 30 rows was also read into the row above it, which
  corrupted six records. All 203 are now clean, and the one test question
  about the uplink station, which could never match, now does.

None of this changes what the August work found in outline: hybrid beats
dense, the small rerankers beat the large ones, and the embedders near the
top cannot be told apart. The sizes of the gains changed.


## 7. Recommended configuration

    embedding:
      model: minilm-l6

    retrieval:
      router: soft
      rerank: true           # Hit@1 84%, Hit@5 93%, MRR 0.883
      rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
      # rerank: false        # Hit@1 77%, Hit@5 95%, MRR 0.851, no second model

183 MB of models with reranking, 92 MB without. On this test set it is
level with every heavier combination measured.


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
    python evaluate.py --routers            rules router against soft

    python parallel_demo.py <slug>          questions one at a time, then all at once
    python demo/server.py <slug>            Ollama and vLLM side by side, see demo/README.md


## 11. Open items

- **Three questions are not found in the top 10** with the default
  settings: `power_budget`, `odhs_diag`, `xband_system`. `odhs_diag`
  cannot be hit at all, because no caption in the store names the ODHS.
  For the other two, try `evaluate.py --compare --widen 50` to see
  whether pool size is the ceiling. `cmd_station` was a fourth until the
  command log's columns were read in the right order.
- **The usual demo question ranks its procedure sixth.** "What is the
  procedure for switching on OBC?" returns FCP-5201 at rank 6 with the
  default settings and rank 4 with the reranker off, behind the other
  OBC procedures. The top 5 go to the language model, so the body it
  needs is just outside the context. Worth looking at before a demo; it
  is one question and should not be tuned for on its own.
- **A hundred questions is too few** to rank the leading embedders or the
  ms-marco rerankers against each other (section 6, intervals). Eleven of
  the hundred are also satisfied by a large share of the store
  (`testset.BROAD`). More and tighter questions would settle more than
  another model sweep.
- **Chunk size has never been swept.** The per-type budgets were chosen
  by reasoning, not measured. Extraction and chunking are separate
  stages, so a sweep only needs re-ingesting.
- **The Streamlit interface is not ported.** The old `app.py` has chat
  with citations, a command table with CSV download, inline figures and
  audit logging. `generate.ask()` already returns what it needs; the work
  is a satellite dropdown and swapping the import.
- **LLM router: built, retrieval effect not measured.** `router: llm`
  classifies 93% of the benchmark questions correctly against 80% for the
  rules (section 6). What that does to Hit@k has not been run:
  `evaluate.py --router llm --compare`, with the text model up.
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
