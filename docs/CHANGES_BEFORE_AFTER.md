# SatelliteRAG: changes made on 16 and 17 November 2026, before and after

This file lists what was changed in the project over two working days,
how each change was carried out, and what the numbers were before and
after. It holds method and measurements only, no text from the mission
documents.

Nothing here is committed to git yet.

## Summary

| # | Change | State now |
| --- | --- | --- |
| 1 | Figures are placed in their document, and captions filed under the right section | Live |
| 2 | All 210 figure captions written again with the right context | Live |
| 3 | A page for checking captions against figures by eye | Used: 30 read, 14 right, 15 partly, 1 wrong |
| 4 | A second, structure-aware parser and chunker | Built and measured, switched off |
| 5 | Three measurements that changed no code | Recorded |
| 6 | Answer generation: eight rough edges worked on | 5 fixed, 1 partly; 30 answers read (13 right, 7 over-refused); prompt v9 from the reading; the demo answers in code where the app does |
| 7 | Documentation | Updated |

What did not change: the default parser, the chunk sizes, the retrieval
code, the router, the test set, and the sweep tables in `results/`.


## 1. Where a figure sits, and which section its caption belongs to

### Before

- The only thing known about a figure was its filename: a document and a
  page for PDF figures, a document and nothing else for Word figures.
- The section was looked up by page number: "the first section starting on
  or after this page".
- When no section starts on the figure's page, that rule returns the
  following section.
- Word chunks have no page number, so Word figures matched nothing.

| Measure | Before |
| --- | --- |
| Report figures given the following section instead of their own | 128 of 181 |
| Report figures with a section starting on their own page | 52 of 181 |
| Word figures with a heading | 0 of 29 |
| Captions filed under a heading | 181 of 210 |

### What we did

1. **Measured the fault first.** For each caption, compared its page with
   the pages where sections start, and counted which section the old rule
   picked.
2. **Checked that a better answer was available.** Every one of the 181
   report figures has a rectangle on its PDF page, and every one of the 29
   Word figures can be found in a paragraph under a heading.
3. **Added figure placement to extraction** (`core/extract.py`:
   `locate_pdf_figures`, `locate_docx_figures`, `locate_figures`). It
   writes `extracted/figures.json`: for each image, the section it is in
   and the text printed beside it.
   - PDF: a section starting on the figure's page counts only if its
     heading is printed above the figure. Otherwise the figure belongs to
     the section running on from an earlier page.
   - Word: the section under the nearest heading above the picture. Image
     files are matched to pictures by the order they were saved in.
4. **Changed captioning to use it** (`core/caption.py`: `load_figures`,
   `place`, `containing_chunk`). The fallback for a figure that cannot be
   placed now takes the last section starting on or before the page.
5. **Made the captions file rebuild without the model.** Heading,
   procedure number and subsystem are read from the section every time the
   file is written, so `python cli.py caption aditya-l1` with nothing new
   to describe re-files old captions.

### After

| Measure | Before | After |
| --- | --- | --- |
| Figures placed in a section | not recorded | 358 of 359 |
| Captions whose heading changed | | 188 of 210 |
| Report figures filed under a heading printed above them | | 24 |
| Report figures filed under the section running on from an earlier page | | 157 |
| Word figures with a heading | 0 of 29 | 28 of 29 |
| Captions filed under a heading | 181 | 209 |

Retrieval with only this change (old caption text, new headings) was
unchanged as shipped: MRR 0.884 to 0.882, figure questions 6 of 7 both
times.


## 2. The captions themselves

### Before

- The vision model was given the opening 700 characters of the section the
  lookup chose. For 128 report figures that was the wrong section. For the
  29 Word figures there was no context at all.
- Even with the right section, its opening could be pages away from the
  figure.

### What we did

1. **Changed the context** (`caption.page_context`). The model now gets the
   section's heading, procedure number and subsystem, then the text printed
   beside the figure: three blocks above and three below in a PDF, two
   paragraphs either side in a Word file. If that is under 200 characters,
   the opening of the section is added.
2. **Backed up the old captions**:
   `extracted/image_captions.before-context-fix.json` and
   `extracted/.captioning_state.before-context-fix.json`.
3. **Captioned all 210 figures again**:
   `python cli.py caption aditya-l1 --force`, with `qwen2.5vl` through
   Ollama on the laptop. About an hour, no failures.
4. **Scored before switching.** The new captions were embedded into a store
   of their own and compared with the working store on the 100 questions.
5. **Switched.** The new store became `vectorstore/`. The old one is kept
   as `vectorstore_before-caption-fix/`.

### After

| Measure | Before | After |
| --- | --- | --- |
| Captions using a word from their own section's heading | 54 of 191 | 178 of 191 |
| Caption length, median characters | 472 | 519 |
| Sentences, median | 3 | 4 |
| Captions quoting a command or telemetry identifier | 0 | 3 |
| Captions starting with a leftover label | 0 | 0 |
| Figure questions answered at rank 1 | 6 of 7 | 6 of 7 |

Retrieval on the 100 questions:

| Setting | Before | After | Change | 95% interval |
| --- | --- | --- | --- | --- |
| Hybrid + reranker (as shipped), MRR | 0.884 | 0.879 | -0.004 | -0.029 to +0.022 |
| Hybrid, MRR | 0.852 | 0.862 | +0.010 | -0.013 to +0.037 |
| Dense only, MRR | 0.792 | 0.825 | +0.033 | +0.001 to +0.071 |
| Hit@5 as shipped | 93% | 92% | | |
| Captions among the top 5 results, per question | 0.55 | 0.97 | | |

Read it this way:

- The captions now describe the section they sit in. That was the aim.
- Retrieval as shipped is level. Dense search alone is better.
- A caption now reaches the top 5 for almost every question. Each one
  takes a place a text passage would have had, which is the likely reason
  Hit@5 fell by one.

The headline on the working store is now Hit@1 84%, Hit@5 92%, Hit@10 97%,
MRR 0.879.

### To go back

Restore the two backup files in `extracted/`, and swap
`vectorstore_before-caption-fix/` back to `vectorstore/`.


## 3. Checking captions by eye

### Before

Nothing recorded a check of any caption against its figure. The only
measure was 7 test questions, which show a caption can be found, not that
it is right.

### What we did

Wrote `review_captions.py`. It builds one local page,
`extracted/caption_review.html`, with a random sample of figures, each
next to its caption and its section, and three buttons: right, partly,
wrong.

```text
python review_captions.py                    30 figures, picked at random
python review_captions.py --n 60 --seed 2    a different, larger sample
python review_captions.py --tally caption_review.json
```

### After

The tool works and the page for the new captions is built. On 17
November, with the documents cleared as dummy data, Claude read all 30
against their figures (one AI reader, one pass; marks in
`extracted/caption_review_claude.json`, out of git):

| Mark | Captions, of 30 |
| --- | --- |
| Right | 14 |
| Partly | 15 |
| Wrong | 1 (a signature, captioned from the text around it) |

The main fault is the surrounding text written up as if the figure showed
it (6 of the 15). Two causes are in the prompt: its figure kinds have no
"table", and it asks every caption for a mission phase and an operational
use. Walkthrough section 3.13 has the detail.


## 4. A structure-aware parser and chunker (built, not switched on)

### Before

- Parsing gave sections of plain text. Chunking cut them near a character
  count, preferring sentence ends, with an overlap.
- The chunker did not know what a table or a paragraph was.

| Measure | Before |
| --- | --- |
| Prose chunks | 1,039 |
| Chunks continuing a section that start mid-sentence | 221 of 483 |
| Pieces of long tables that begin or end inside a row | 98 of 108 |
| Contents-page lines in the store | 359 |
| Page header and footer lines inside chunks | about 600 |

### What we did

1. **Measured what the current chunking breaks** (the table above).
2. **Checked what the documents offer**: the report PDF has 188 bookmarks
   with levels; PyMuPDF finds 179 tables in its body; Word files can be
   walked in order.
3. **Wrote new readers** (`core/layout.py`). Report sections come from the
   bookmarks, headers, footers and contents lines are removed, tables are
   read row by row, and each section records its heading path. Word files
   are read in document order, so a table stays under its heading.
4. **Wrote a block-aware chunker** (`chunking.split_blocks`). Cuts fall
   only between paragraphs or table rows, and a long table repeats its
   header row on every chunk.
5. **Put it behind a switch**: `parsing: {mode: structure}` in a
   satellite's config. The default stays `flat`.
6. **Compared side by side** (`compare_parsing.py`). The new version is
   built into `extracted_structure/` and `vectorstore_structure/`, and the
   working store is not touched.
7. **Left the procedure document alone.** Its step tables cannot be rebuilt
   row by row from the PDF.

### After

| Measure | Flat | Structure |
| --- | --- | --- |
| Prose chunks | 1,039 | 779 |
| Continuation chunks starting mid-sentence | 221 | 40 |
| Table rows cut | 98 pieces | 0 |
| Contents-page chunks in the top 5, per question | 0.23 | 0 |

Retrieval, MRR, with contents-page chunks not counted as right answers:

| Setting | Flat | Structure | Change | 95% interval |
| --- | --- | --- | --- | --- |
| Hybrid + reranker (as shipped) | 0.881 | 0.873 | -0.008 | -0.037 to +0.021 |
| Hybrid | 0.830 | 0.829 | -0.001 | -0.049 to +0.044 |
| Dense only | 0.785 | 0.778 | -0.008 | -0.050 to +0.034 |
| BM25 only | 0.698 | 0.725 | +0.027 | -0.006 to +0.064 |

These were measured before the captions were rewritten.

**Decision:** the chunks are cleaner but retrieval is level, so the current
parser stays the default. The test set checks whether a right chunk was
found, not whether the answer written from it was better. To decide, the
written answers from both stores need comparing.


## 5. Three measurements that changed no code

| Finding | Before we looked | What was measured |
| --- | --- | --- |
| Contents pages count as right answers | Not known | With BM25 alone, the first "right" chunk was a contents page for 17 of 100 questions. BM25 MRR is 0.804 as labelled and 0.698 without them. As shipped it affects 2 questions: 0.884 to 0.881. |
| Chunk size | Never swept | Prose budgets of 800, 600 and 450 characters, on both parsers. None beat the current sizes; 450 was clearly worse. |
| Repeatability | Documented as deterministic | For the same query vector, ChromaDB sometimes returns different neighbours in a new process. Seen in 1 run of 5 on one command-log question (ranks 2, 5 and 14). Scores moved by up to 0.009 between identical runs. |

None of these was fixed. The contents pages are still in the working
store.


## 6. Answer generation (17 November)

Writing the answer-generation chapter turned up eight rough edges. Each is
below with what was done.

### How it was measured

- **Before:** every test question run through the app on the morning of
  17 November (100 questions and 2 with no answer in the documents), with
  counts recorded and no answer text kept.
- **After:** `evaluate_answers.py`, written for this, ran the same
  questions twice, once with prompt v6 and once with the new v7, both with
  all the code fixes. The saved answers were then counted again with
  `--rescore`, and once more with `--repair` after the last code step was
  added, so that step could be measured without asking the model again.
- **Retrieval** was scored on the 100 questions with and without the
  procedure change, in the same process, on the same store.
- **The code-decided parts** have 29 checks in `tests/test_answering.py`,
  on made-up data. All pass.

### 6.1 The demo question got "steps not in the documents"

**Before.** Each procedure was in the store as an index row and up to
three pieces of its step table. The pieces are long and full of codes, so
they ranked well for any question about their subsystem. For "how to
switch on the OBC", the bodies of two other procedures took places in the
top five, and the right one (5201) was 7th after reranking.

**What we did** (`core/retrieve.py`):

1. After fusion, every piece of a procedure is replaced by its index row,
   once, so a procedure is one candidate (`rows_for_bodies`).
2. The reranker reads both the row and the whole procedure, and the
   procedure keeps the better place (`bodies_of`).
3. After ranking, the top two rows with steps in the store are swapped
   for the whole procedure, joined from its pieces without the overlap
   (`steps_for_rows`, `_join_pieces`).
4. "Which FCP" questions keep the row.
5. A procedure goes to the model up to 6,000 characters, not cut at 2,000.

**After.**

| Measure | Before | After |
| --- | --- | --- |
| The demo question's procedure, place in the top 20 | 5th | 4th, inside the five passages |
| Questions whose procedure has steps, with that table in the passages | 3 of 4 | 4 of 4 |
| MRR, 100 questions, top 20 | 0.880 | 0.880 |
| MRR, the 50 procedure questions | 0.912 | 0.913 |

### 6.2 Steps often did not come out

**Before.** Of 8 answers with a step table in the passages, 2 gave a
numbered list.

**What we found.** The step tables are stored one cell per line, with each
column heading on its own line: 5201 is 203 lines with a median of 11
characters. The model sees fragments.

**What we did.**

- The procedure goes to the model whole (6.1), so its last steps are not
  cut off.
- A new code step, `grounding.drop_false_no_steps`, removes "the steps of
  this procedure are not in the available documents" when it sits under an
  FCP whose table is in the passages. It came from the demo answer, which
  listed 5201's steps and then wrote that sentence under them.
- The check itself was wrong. It counted `1.`-style lines; a faithful
  answer copies the table as it is. It now counts the command ids copied
  from the right table.
- Rebuilding the table rows from positions on the PDF page was tried and
  **rejected**: the column headings read back wrong and cells landed in the
  wrong columns. A rebuilt table that puts an id against the wrong step
  would look trustworthy and be wrong.

**After.**

| Measure | Before | After |
| --- | --- | --- |
| Answers copying command ids from the right table | not measured | 4 of 4 |
| No-steps sentence under a procedure whose table is there | 1 by the old check; 1 with v7 before the new step | 0 |

**Still open:** the steps come out as copied fragments, not readable
numbered steps. The fix is a clean copy of the procedure document.

### 6.3 Four refusals with a right passage

**What we did.** `review_answers.py <run> --only refused` builds a page
with each refusal next to the passages it was given, with marks: right,
partly, wrong, over-refused. It now shows 5 refusals with a right passage,
of 7 (first counted as 4 of 6; see 6.10).

**Still open:** 1 has been read (over-refused); 6 to read.

### 6.4 The guard had never acted

**Before.** No answer from the current prompt carried an id that was not
in its passages, so the guard had only ever been seen to act on older
prompts.

**What we did.** Five checks hand it made-up answers: a correct one (left
alone), a wrong telemetry id, a hex value and a value after "CID" (all
removed), an id one digit longer than a real one (removed), and a made-up
name with no digits (passes, as documented).

**After.** All pass. On real answers it still removed nothing, in either
run.

### 6.5 Nothing measured whether an answer was right

**What we did.**

- `evaluate_answers.py` runs every question through the app and counts:
  refusals, wrong procedure named, steps copied, citations, ids removed,
  timing. Numbers go to `results/answer_checks.csv`. The answers and their
  passages go to `runs/answers/`, which git ignores.
- `review_answers.py` turns a run into a page with each answer and its
  passages, to mark right, partly, wrong or over-refused, and tallies the
  marks.

**After.** 30 answers read on 17 November; see 6.10.

### 6.6 Citations were not enforced

**Before.** 51 of 91 answers named a document, page or FCP somewhere.
Nothing tied a statement to a passage.

**What we did.**

- Prompt v7 is v6 plus one rule: give the source after each statement, as
  `[Source 2]`, using only the numbers shown in the context. v6 and v1 are
  kept, word for word.
- `generate.check_citations` removes a citation to a source number that
  was never given, and adds a line to an answer that states things and
  cites nothing.

**After.**

| Measure | v6 | v7 |
| --- | --- | --- |
| Answers with a checked `[Source N]` citation, of 88 | 8 | 68 |
| Answers flagged as citing nothing | 27 | 2 |
| Refused | 7 | 7 |
| Named the wrong procedure | 1 | 0 |
| Real answer that still ends in the not-found line | 5 | 7 |

The other 18 of the 88 v7 answers are refusals or the fixed no-steps
reply, which assert nothing to cite. v7 is worse on one count, the last
row.

### 6.7 The first request after a start could fail

**Before.** 1 request failed in the morning run, the first one, after 20
seconds. It worked when asked again.

**What we did** (`core/llm.py`). A request that fails because the server
did not answer, or answered with a server error, is tried once more after
3 seconds. One the server understood and refused, such as a missing model,
is not.

**After.** 0 failed in either run. Two checks cover it: a request that
fails once succeeds on the second try, and a refused one is tried once.

### 6.8 A day of commands filled the context

**Before.** "What was sent on <date>" gave the model every record for the
day. The largest prompt was 4,098 tokens, hit the 16,000-character limit,
took 49 seconds and ran to the answer limit. The model could only describe
the records that fitted.

**What we did** (`generate.answer_from_commands`). When the passages are
every record for a date, command id or mnemonic, the answer is counted in
code: how many, first and last time, status, stations, orbit, and each
command with how often it was sent and when. A question that names a
subsystem ("OBC commands on ...") is narrowed to it, and the answer says
so. The records are listed as the sources.

**After.**

| Measure | Before | After |
| --- | --- | --- |
| Command-log lookups answered by the model | 4 | 0 |
| Slowest command-log answer | 48.9 s | 0.0 s |
| Records left out of the answer | some, whenever the day did not fit | none |

Two command questions still go to the model, because they name no date or
id: one asks whether a command was sent, the other about stations.

### 6.9 Also changed

- The browser demo (`demo/server.py`) runs the same steps after the model
  as the app does: guard, no-steps check, citation check.
- The benchmark (`bench/`) takes the app's system prompt by default, which
  is now v7.

### 6.10 The first reading of the answers

**Before.** No answer had been read against its passages.

**What we did.** 30 answers from the v7 run were read on the review page
and marked.

| Mark | Answers |
| --- | --- |
| right | 13 |
| partly | 7 |
| wrong | 3 |
| over-refused | 7 |

- 6 of the 7 over-refused are the fixed reply for an index-only procedure,
  which the code counts as correct, given where the passages held related
  material from other documents. The 7th is a plain refusal.
- The 5 answers that copied a step table had the commands right, and all
  left out the conditions, prerequisites and cautions.
- 3 of 28 made a statement the passages do not; none was a command id.

**Two faults found by reading, both fixed.**

| Fault | Before | After |
| --- | --- | --- |
| A figure answer's page showed only source labels | 2 figure answers could not be judged | The caption and its image file are kept as the passages; `--only figures` page |
| A refusal followed by citations was counted as an answer | v7 refused 6, 4 with a right passage, 8 tacked on | 7, 5, 7; citations are dropped from a bare refusal |

Two checks were added to `tests/test_answering.py` (31 in all, all pass).

### 6.11 Prompts v8 and v9

**Before.** v7: 7 refusals, 5 of them with a right passage; the fixed
no-steps reply gave nothing else; procedure answers left out conditions.

**What we did.**

- v8: ask for a procedure's conditions with its steps, and for what other
  passages say about an index-only procedure. Not kept: it rewrote the
  demo procedure without its command ids, and twice gave another
  procedure's steps.
- v9: copy the steps first, then the conditions; use only passages that
  are not another FCP's table, naming those FCPs; do not end on the
  not-found sentence.
- Code: the not-found sentence alone as the last paragraph of an answer
  is taken off (`generate.drop_trailing_not_found`), in the app and the
  demo.
- 6 more checks in `tests/test_answering.py` (37 in all, all pass).

**After.**

| Measure | v7 | v8 | v9 |
| --- | --- | --- | --- |
| Refused, of 88 | 7 | 2 | 2 |
| Refused with a right passage | 5 | 1 | 1 |
| Not-found line after a real answer, before the new code step | 7 | 12 | 14 |
| the same, after it | 4 | | 1 |
| Named the right procedure, of 23 | 19 | 17 | 21 |
| Named another procedure | 0 | 2 | 1 |
| Procedures with steps whose command ids were copied, of 4 | 4 | 3 | 4 |
| The demo question's command ids copied | 15 | 0 | 15 |
| Answers with a checked `[Source N]` | 68 | 86 | 85 |
| Command ids removed by the guard | 0 | 0 | 0 |
| Median answer length, characters | 150 | 512 | 446 |
| Median answer time | 5.0 s | 9.4 s | 8.3 s |

v9 is in use. **Read on 17 November** (Claude, against the passages; v7
marks are from a different reader): v9 right 16 of 30 (v7 13),
over-refused 3 (v7 7), wrong 3 (v7 3). With the reader held equal, v9
improved 7 answers and made 4 worse, two badly (`obc_on` gives the
system-reset steps as OBC-2's switch-on; `pcs` cites a passage that does
not mention it). Cautions are still left out of procedure answers.
Walkthrough 7.13 has the detail.

### 6.12 The browser demo answers in code where the app does

**Before.** The app answers three kinds of question without a model: a
figure from its caption, a command-log lookup by date, command id or
mnemonic from every matching record, and a question that retrieves
nothing. `demo/server.py` handled only the first, so it sent log lookups
and empty retrievals to both models.

**What we did.** The rule moved into one function,
`generate.answer_without_model()`, which `answer()` and the demo's
`prepare()` both call. The demo sends the reason with the reply, and
the page shows it (one line in `demo/page.html`). Two checks added to
`tests/test_answering.py`, one driving `prepare()` with made-up records.

**After.** 39 checks, all pass. On the laptop, a figure question
through the running demo was answered from its caption with the reason
shown, and no model was called. The batch tab is unchanged.

## 7. Files

**New**

| File | What it is |
| --- | --- |
| `core/layout.py` | Structure-aware readers for the report PDF and Word files |
| `compare_parsing.py` | Builds the structure variant beside the working store and scores both |
| `review_captions.py` | The caption review page and its tally |
| `results/parsing_comparison.csv`, `results/parsing_chunks.csv` | Numbers from the parser comparison |
| `evaluate_answers.py` | Runs every question through the app and counts what can be counted about the answers |
| `review_answers.py` | A page for reading answers against their passages, and its tally |
| `tests/test_answering.py` | 29 checks on the code-decided parts of answering, on made-up data |
| `results/answer_checks.csv` | Numbers from the answer runs |
| `PROJECT_WALKTHROUGH.md` | The project explained in build order; chapters 1 to 3 |
| `CHANGES_BEFORE_AFTER.md` | This file |

**Changed**

| File | Change |
| --- | --- |
| `core/extract.py` | Figure placement; `figures.json`; optional image saving; reader chosen by parsing mode |
| `core/caption.py` | Context from the figure's own section and the text beside it; captions file rebuilt without the model |
| `core/chunking.py` | Block-aware splitter, used only for sections that come with blocks |
| `core/ingest.py` | Heading path in the embedded text and metadata, when a section has one |
| `core/registry.py` | `parsing.mode` setting; folders for a side-by-side extraction |
| `core/retrieve.py` | A procedure is one candidate; its steps are attached after ranking; procedures not cut at 2,000 characters |
| `core/generate.py` | Prompts v7, v8 and v9 (v9 in use); the trailing not-found line taken off; command-log lookups counted in code; `answer_without_model()` shared with the demo; citations checked; `answer()` returns how a reply was made |
| `core/grounding.py` | The wrong no-steps sentence removed |
| `core/llm.py` | One retry for a busy or unreachable server |
| `lora/build_rag_dataset.py`, `lora/train_rag.py`, `lora/eval_rag.py` | The passage-grounded adapter: data laid out as the app's prompt, training, and scoring against base models and the app's 7B |
| `demo/server.py` | The same steps after the model as the app, and the same questions answered without one |
| `.gitignore` | Ignores `extracted_*/` |
| `PROJECT_CONTEXT.md` | Dated note on the new captions and the new headline; corrected description of caption metadata |
| `COMPARISON.md` | Corrected description of caption metadata |

**Local data, not in git**

| Path | What it is |
| --- | --- |
| `extracted/figures.json` | Where each figure sits |
| `extracted/image_captions.json` | The new captions |
| `extracted/*.before-context-fix.json` | The old captions and state |
| `vectorstore/` | The working store, with the new captions |
| `vectorstore_before-caption-fix/` | The working store as it was |
| `extracted_structure/`, `vectorstore_structure/` | The structure parser's output |
| `runs/answers/` | Answer runs and their review pages; they hold document text |


## 8. Still open

- Captions: add "table" and "equation" to the prompt, ask for phase and
  use only when shown, drop signatures; caption again and read a new
  sample (walkthrough 3.14).
- The ten per-model stores and the sweep tables in `results/` were built
  with the old captions. The README headline still shows Hit@5 93% and MRR
  0.883.
- Captions now crowd the top 5. If the review shows they do not help
  answers, limit them to one place unless the question asks for a figure.
- Take the contents pages out of the working store and check the test
  conditions they satisfy.
- Compare written answers from the flat and structure stores before
  deciding on the parser.
- Use exact search for the 203-record command log, so its results are
  repeatable.
- `odhs_diag` is still a miss.
- Prompt after v9: the caution must come with a procedure's steps, and
  another FCP's steps must never be given under the one asked for
  (`obc_on`, `pcs`; walkthrough 7.13).
- Recaption `GUIDELINES_img005`: its caption says direction cosines, so
  the thruster direction-cosine question gets a force/torque table
  instead of `img004` (walkthrough 7.13).
- Read the 5 refusals that had a right passage (1 done):
  `python review_answers.py v7-with-no-steps-check --only refused`.
- Get a clean (Word or text) copy of the procedure document, so steps come
  out as readable steps.
- 7 answers end with the not-found sentence after a real answer.
- LoRA (walkthrough chapter 10): the first adapter learned the
  documents' style, not their facts (2 of 40 right); correct the work
  log's "triples F1". A second adapter, trained on the app's own task
  (five passages, a cited answer, refusals), was trained and tested on
  the lab GPU on 17 November: a 1.5B with it cites the right passage 84
  times of 101 against the 7B's 69, and refuses correctly 74 of 101
  against 67, but its answers are too brief. Next: references written to
  be complete, then a run on the app's 102 questions. The README still
  says `lora/` is built from the public booklets.
- Run the browser demo once against the lab servers (walkthrough
  chapter 11).
- From the 16 November work log, unchanged: the repository's visibility,
  the lab password, and running the browser demo against the lab servers.
