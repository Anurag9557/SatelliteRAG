# SatelliteRAG vs the original Aditya-L1 pipeline

What the previous system was, what changed, and what the numbers actually
say. Measured on the same 100-question labelled set in both cases.

Previous system: `ISRO_THESIS\Isro_data` (step1-step4 + evaluate + app)
Current system:  `SatelliteRAG` (core/ + cli.py + evaluate.py)

**Note, 12 November 2026.** The current-system figures in this file are
the August measurements and have been superseded. They were taken while
the procedure prepend was silently not running, and the reranker rows
used a shortcut the pipeline does not take. `PROJECT_CONTEXT.md` section 6
has the re-measured figures (Hit@1 84%, Hit@5 93%, MRR 0.883 with the
default settings) and what was wrong. The previous-system figures, the
structure comparison and the lists of bugs below are unaffected.


## 1. The headline numbers

Same 100 questions, same machine.

One caveat governs the whole table: the previous system's figures were
scored with the original accept conditions, several of which were loose
enough to admit false positives. `eeprom_mgmt` accepted any chunk
containing "eeprom", so it scored a hit on FCP-5203, whose notes merely
mention an EEPROM setting. Those conditions have since been tightened.
The current figures are therefore measured against a stricter standard
than the previous ones, and the gap between them is understated.

### Previous system

| Configuration | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| best embedder, no reranker (minilm-l6 hybrid) | 0.73 | 0.84 | 0.87 | 0.91 | 0.790 |
| bge-m3 hybrid, no reranker | 0.65 | 0.82 | 0.89 | 0.92 | 0.744 |
| with reranker (ms-marco-MiniLM-L-6, bge-m3 base) | 0.82 | 0.89 | 0.89 | 0.93 | 0.857 |

### Current system, tightened accept conditions

| Configuration | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | sec |
|---|---|---|---|---|---|---|
| **minilm-l6, no reranker** | 0.77 | **0.91** | **0.94** | 0.95 | 0.843 | 14 |
| **minilm-l6, reranker** | **0.83** | 0.90 | 0.92 | 0.96 | **0.871** | 25 |
| bge-m3, no reranker | 0.68 | 0.80 | 0.92 | **0.97** | 0.767 | 87 |
| bge-m3, reranker | 0.81 | 0.90 | 0.93 | **0.97** | 0.862 | 42 |

Misses at 5 fell from 12 to 4.

### Which configuration to run

minilm-l6 beats bge-m3 by +0.076 MRR unranked and +0.009 ranked, at a
quarter of the size and a fifth of the build time. This reproduces the
previous system's own embedder comparison, where minilm-l6 hybrid scored
0.790 against bge-m3's 0.744 and bge-m3 was the weakest of four models in
dense mode (0.648). Multilingual training does not help English technical
documents.

The reranker is now a judgement call rather than a clear win:

    Hit@1   0.77 -> 0.83   +0.06
    Hit@3   0.91 -> 0.90   -0.01
    Hit@5   0.94 -> 0.92   -0.02
    MRR    0.843 -> 0.871  +0.029

It sharpens the first position and loses a little from the 3-5 band. The
rescue view shows why: of 6 questions missed at 5 without it, 1 was
recovered and 3 others were pushed out.

Because the pipeline sends the top 5 to the language model, and the model
reads all five, **Hit@5 is the operationally relevant number** and the
reranker is slightly negative there. The default is therefore minilm-l6
with `rerank: false`: best Hit@5, fastest, one fewer model to load. Turn
reranking on when the first citation shown to a human matters more than
the context handed to the model.


## 2. What Hit@10 says that Hit@1 does not

Hit@10 is recall: how often the correct chunk reaches the candidate pool
at all. No reranker can exceed it, because a chunk that was never
retrieved cannot be reordered into first place.

    Previous:  0.91 - 0.93
    Current:   0.95 - 0.97

That improvement comes from extraction, not from ranking. It is the part
of the work that a better model cannot substitute for.

The reranker delta is the mirror of this. On the previous system it added
+0.11 MRR; on the current one, +0.029, and it costs Hit@5. During
development an intermediate version showed +0.39, which was the symptom
that led to finding a real bug in the fusion stage. A reranker that adds
a very large amount is not a strong reranker; it is a sign of weak
retrieval beneath it.


## 3. Structure

| | Previous | Current |
|---|---|---|
| Satellites | one, hardcoded | any number, one folder each |
| Documents | loose in the project root | `satellites/<slug>/documents/` |
| Filenames | typed into a `DOCS` dict in step1 | recognised from the name |
| Outputs | shared `extracted/`, `vectorstore/` | per satellite, never mixed |
| Entry point | four scripts run in order | `cli.py` + importable `core/` |
| Adding a satellite | copy the folder, edit the code | make a folder, copy documents in |

Adding a satellite is now:

    python cli.py new chandrayaan-3
    # copy that satellite's documents into satellites/chandrayaan-3/documents/
    python cli.py build chandrayaan-3

No configuration file is required. `satellite.yaml` exists for overrides
(embedding model, chunk sizes, retrieval settings) but a folder holding
documents is a valid satellite on its own.


## 4. Bugs found in the previous pipeline

**The procedure index was one unsearchable chunk.** The entire 225-row
table of contents was stored as a single 13,127-character chunk. Embedded
as one vector, no individual row could be retrieved, so "which FCP covers
gyro power on" had nothing to match. Now 225 separate records, each with
its procedure number and the subsystem the document assigns it.

**Duplicate chunk IDs silently dropped 92 chunks.** MOR IDs were
`MOR_{page}_{index}` with the index restarting per section, so two
sections beginning on the same page collided. ChromaDB rejects a batch
containing a duplicate, so those chunks were discarded at ingest.

**Six figures were never captioned.** Ollama was called without setting
the context window, so the 4096-token default applied and full-page
figures failed outright with `exceed context size`. Now 8192.

**Procedure notes lost their procedure number.** The extractor reset the
current FCP number on seeing `END OF PROCEDURE`, so the notes pages that
follow could not be traced back to their procedure.

**Word tables were skipped.** Only paragraphs were read from `.docx`
files; tables, which carry much of the operational content, were ignored.

**Two questions were unanswerable by construction.** `event_uplink_exec`
missed in every configuration because `route()` sent anything containing
"uplinked" to the command store, where FCP-5217 does not exist. Six more
(`manual_thruster`, `rw_logics`, `gyro_cal`, `power_safe_mode`,
`velc_occulter`, `momentum_bias`) missed because the subsystem filter
compared a guess from the question against the document's own label: the
documents file thruster firing under AOCS, not PROPULSION.


## 5. Bugs introduced during the rebuild, and fixed

Recorded because they cost more than they saved, and because the pattern
is worth remembering: every one came from changing something without
measuring it.

**The procedure prepend excluded the index rows.** The previous code
searched `source_doc == FCP` and prepended the best two hits. The rebuild
narrowed that to `content_type in (procedure, note)`, which removes 225
of 231 FCP chunks in a file that holds only 6 procedure bodies. This was
the single largest regression: baseline Hit@1 fell to 0.42 and the
reranker was quietly compensating for it.

**The index subsystem was inferred rather than read.** The previous code
mapped the document's own subsystem column, misspelling and all. The
rebuild ran the title through keyword inference, so "5010 AOCS THRUSTER
BLOCK SELECTION" became PROPULSION.

**The index text lost its "FCP" prefix.** `FCP 5601 - DTG - GYRO POWER
ON` became `5601 DTG GYRO POWER ON`, breaking the token match for
questions phrased "which FCP...".

**The subsystem filter became a hard gate.** Changing `$in [sub, UNKNOWN,
""]` to an exact match turned a soft signal into a filter that deleted
correct answers before scoring. It has since been removed entirely: an
auto-guessed subsystem is not reliable enough to gate on.

**BM25 was more permissive than the dense pass.** Keeping `UNKNOWN`
through the keyword filter let chunks the semantic filter had excluded
back in through the side door.

**Reranking bypassed the content-type rules.** Returning the reranked
list immediately meant a cross-encoder rated the FCP-5201 index row above
the procedure body, and the steps dropped out of the top five entirely.
Reranking now runs before those rules, not instead of them.

**The tokenizer did not split hyphenated compounds.** `OBC-1` and `OBC`
were different tokens, so a procedure titled "OBC-1 ON / OBC-2 ON" scored
zero against a search for OBC.

**Answer generation was dropped.** The port kept `build_context()` but
never called a model, so every query returned a source list rather than
an answer. Restored in `core/generate.py` using `qwen2.5:7b-instruct`.


## 6. Design changes kept

**Chunking moved to ingest time.** Extraction emits whole sections;
`core/chunking.py` applies the size policy when building the store.
Changing a chunk budget no longer requires re-extraction.

**Chunk sizes are per document type.**

    FCP           2500 chars, cuts at end-of-procedure or notes markers
    MOR           1000 chars, paragraph boundaries
    OPS_MANUAL    1200 chars
    GUIDELINES    1200 chars
    toc_index     atomic, never split
    image_caption atomic, never split
    command_log   atomic, never split

FCP gets more than double MOR's budget because a procedure cut in half
starts mid-sequence and cannot be followed. Index rows, captions and
command records are atomic by type rather than by length.

**Date queries return the whole day.** "What commands were uplinked on
2026-06-09" reads every matching record in time order rather than the
five most semantically similar.

**Captions inherit their metadata.** Section heading, procedure number
and subsystem are copied from the nearest chunk, found by bisect so it
works even when the figure's page has no chunk of its own.

**Batch size follows the card.** Chosen from VRAM and model dimension,
with a halve-and-retry on out-of-memory, instead of a fixed 128 that
overflows a laptop GPU part-way through a long run.

**Cross-satellite search.** `query all "..."` searches each satellite
independently and fuses the ranked lists, so a larger store cannot
dominate on size. Fusion is by rank, so satellites using different
embedding models still merge correctly.


## 7. Taken from the previous system

- The targeted re-query and prepend, in place of score multiplication.
  RRF scores all land between 1/61 and 1/80, so a multiplier fights over
  hundredths; a second filtered search puts the result where it belongs.
- The TOC subsystem map, including the `PAYLAOD` misspelling.
- `truststore.inject_into_ssl()` for campus HTTPS interception.
- `qwen2.5:7b-instruct` for generation rather than the vision model.
- Answering diagram questions from the caption instead of the text model.
- Returning sources when generation fails rather than crashing.
- Recall ceiling reporting and the rescue view from `rerank_compare.py`.
- The 100-question labelled set.


## 8. Deliberately not taken

- **Uniform 1500-char chunking.** Per-type budgets give Hit@10 0.98
  against 0.93.
- **Subsystem filtering, in any form.** Measured to cost more than it
  saves. The previous code was also inconsistent with itself: `$eq` in
  `evaluate.py`, `$in [sub, UNKNOWN, ""]` in `step4_query.py`.
- **The breadth of `route()`.** `\buplink(ed|s|ing)?\b` sending a query
  to the command store costs `event_uplink_exec` permanently, and
  `\b[A-Z]{3,5}\d{3,4}\b` would catch a bare `FCP5201`.
- **Pre-splitting MOR during extraction.** Splitting at ingest keeps the
  two concerns separate.


## 9. Chunk counts

| | Previous | Current |
|---|---|---|
| chunks.json | 872 | 781 |
| image captions | 210 | 210 |
| command records | 203 | 203 |
| after splitting | 1133 | 1474 |
| duplicate IDs dropped | 92 | 0 |

Fewer source chunks and more after splitting: the previous count included
the 13,127-character index blob and 92 chunks discarded at ingest, while
MOR now splits at 1000 chars rather than 1500.


## 10. Open items

- **Four questions still miss at 5** in the recommended configuration:
  `power_budget`, `odhs_diag`, `xband_system`, `cmd_station`. Worth
  checking whether the accept condition or the retrieval is at fault.
- **`README.md` in the previous project quotes Hit@1 93.3%, MRR 0.967.**
  Those are from the original 15-question set, not the 100-question one.
  They should not be carried into the seminar.
- **Not yet ported:** the Streamlit interface (`app.py`), the
  dense/hybrid/bm25 mode switch from `eval_modes.py`, per-model store
  folders from `compare_embeddings.py`, and the HTML evaluation report.
