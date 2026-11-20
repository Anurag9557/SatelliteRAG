# SatelliteRAG walkthrough, from the start

This file explains the project in the order it was built, one chapter at a
time. Each chapter says what we had, what the problem was, what we did, and
what came out. New chapters are appended as we go through the code.

It describes structure and method only. It holds no text from the mission
documents, and nothing in it should ever be copied from them.

| Chapter | Topic | State |
| --- | --- | --- |
| 1 | The source files and how we parsed them | written |
| 2 | Chunking, and the structure-aware parser we tried | written |
| 3 | Figure captioning | written |
| 4 | Embedding and the two stores | written |
| 5 | Retrieval: dense, BM25, fusion, reranking | written |
| 6 | Routing a question | written |
| 7 | Answer generation and the grounding guard | written |
| 8 | Evaluation and the November audit | written |
| 9 | Ollama against vLLM | written |
| 10 | LoRA fine-tuning | written |
| 11 | The browser demo | written |


## Chapter 1: The source files and how we parsed them

### 1.1 What parsing has to do

A language model cannot search a PDF. Before anything can be retrieved, each
document has to be turned into small pieces of text with labels on them:
which document it came from, which page, which section, which procedure.
One such piece is called a **chunk**.

Parsing ("extraction" in the code) is the first stage of the pipeline:

```text
documents/            extract              extracted/
5 source files   ------------------>   chunks.json        text pieces
                 core/extract.py       tch_records.json   one per command
                                       images/            every figure
```

Everything later (captioning, embedding, search, answers) reads only the
`extracted/` folder. If a fact is lost or mislabelled here, no later stage
can get it back. That is why most of the early bugs were parsing bugs.

Code: `core/extract.py` (647 lines), with file-type detection in
`core/registry.py`. Run it with:

```text
python cli.py scan aditya-l1              which file was taken as which type
python cli.py extract aditya-l1 --force   parse again from scratch
```


### 1.2 The five files we started with

| Type | What it is | Format | Size | Shape |
| --- | --- | --- | --- | --- |
| FCP | Flight control procedures | PDF | 0.35 MB | 25 pages, all with text, no figures |
| MOR | Mission operations report | PDF | 8.07 MB | 270 pages, 260 with text, 330 embedded images |
| OPS_MANUAL | Operations manual | Word (.docx) | 3.01 MB | 1,640 paragraphs, 173 headings, 80 tables, 8 images |
| GUIDELINES | Normal-phase guidelines | Word (.doc, converted) | 0.44 MB | 1,849 paragraphs, 103 headings, 65 tables, 21 images |
| TCH | Telecommand history log | Text (.txt) | 0.05 MB | 285 lines, 203 command rows |

The point to notice is that these are five different kinds of file, not five
copies of one kind. A procedure document, a long report, two Word manuals
and a machine-printed log have nothing in common in layout. So there is one
parser per kind, not one parser for everything.

The old pipeline (`ISRO_THESIS\Isro_data`) had the filenames typed into the
code and treated the text more uniformly. Several of its bugs came from that
(section 1.9).


### 1.3 How a file gets its type

Nothing is typed in. `registry.detect_doc_type()` reads the filename:

1. Drop the extension, split camelCase, turn every non-letter into a space,
   lower-case it. `AL1OperationsManual_latest.docx` becomes
   `al1 operations manual latest`.
2. Try the patterns in a fixed order, most specific first:
   TCH ("tch", "telecommand", "command history"), FCP ("fcp", "flight
   control proc"), MOR ("mor", "mission oper", "mission report"),
   GUIDELINES ("guide line", "guidance", "normal phase"), then OPS_MANUAL
   ("manual") last because it is the loosest word.
3. If nothing matches and the file is a `.txt`, call it TCH.
4. `satellite.yaml` can name a file explicitly, and that wins.

The file extension then picks the reader: `.pdf`, `.docx`, `.doc`, `.txt`.
`python cli.py scan` prints the result, so a wrong match is seen before
building and not after.


### 1.4 The procedure document (FCP, PDF)

**How the file was.** 25 pages in two parts:

- Pages 1 to 6 are an index: 225 rows, each with a serial number, a
  four-digit procedure number (5xxx or 6xxx), the owning subsystem, and the
  procedure's title.
- Pages 7 to 25 are the procedures themselves. Each opens with a header of
  the form `SUBSYSTEM-NNNN: TITLE`, has a `PROCEDURE:` step table, ends with
  `END OF PROCEDURE`, and is followed by a `NOTES:` page.

This file is a sample. It carries the full index but only 6 procedure
bodies. So for most procedures the index row is the only thing we have.

**What made it hard.**

- The index and the bodies answer different questions. "Which FCP covers X"
  needs an index row. "How do I do X" needs a body.
- The step tables inside a procedure contain numbered lines that look like
  index rows.
- The notes page comes after `END OF PROCEDURE`, so it is easy to cut it
  off from its procedure.
- The index reuses a few procedure numbers, and misspells one subsystem.

**How we parse it** (`extract_fcp`):

1. Read the text of every page with PyMuPDF. If fewer than half the pages
   have a text layer, print a warning that it is a scanned document.
2. Find where the bodies begin: the first page whose text contains a
   procedure header. Here that is page 7. Everything before it is index.
3. Read index rows only from the index pages. A row is found by the pattern
   "serial number, dot, procedure number, subsystem word". The title is the
   text between one row's match and the next.
4. The subsystem is taken from the row's own column through a small lookup
   table, misspelling included. It is not guessed from the title.
5. Each row becomes one chunk of type `toc_index`, with text in the form
   `FCP <number> - <SUBSYSTEM> - <title>` and an id built from the serial
   and the number, so repeated numbers do not collide.
6. For the bodies: every page from page 7 on is checked for a header. A
   procedure runs from its header page to the page before the next header.
   So the notes page stays inside its own procedure.
7. Each body becomes one chunk of type `procedure`, carrying its procedure
   number and its header as the section heading.
8. If no header or row matched at all, fall back to one chunk per page.
   Rough chunks are better than silently producing none.

**What comes out.** 231 chunks: 225 index rows (median 53 characters) and 6
procedure bodies (median 1,637 characters, the longest 4,530).


### 1.5 The mission operations report (MOR, PDF)

**How the file was.** 270 pages of prose under numbered headings (1, 1.2,
1.2.3 and so on, with titles in capitals), and 330 embedded images.

**What made it hard.**

- It is far too long to be one piece, and a blind cut every N characters
  would split sentences and separate a heading from its text.
- Lines such as page footers and table rows can look like headings.
- We need the page number of each section for the citation.

**How we parse it** (`extract_prose_pdf`):

1. Join the text of all pages into one long string, and note the character
   position where each page starts.
2. Find headings: a line that starts with a section number followed by a
   title in capitals at least five characters long. 208 lines match.
3. Throw out false headings: ones containing a pipe (a table row), ones
   containing the spaced-out page footer, titles under five characters, and
   titles that are only digits.
4. A section is the text from one heading up to the next. Sections under 40
   characters are dropped.
5. The section's page is the last page that starts at or before the
   heading's position.
6. The id is `MOR_s<n>`, where n counts across the whole document, so no
   two sections can share an id.
7. Every image on every page is saved as `MOR_p<page>_img<n>`. The page in
   the name is how a figure is later matched to the text near it.
8. If no heading matched, fall back to one chunk per page.

**What comes out.** 192 sections, median 690 characters, the longest
31,047.


### 1.6 The two Word documents (OPS_MANUAL and GUIDELINES)

**How the files were.** Word documents with real heading styles, a lot of
tables (80 and 65), and a few embedded drawings. The guidelines document
was originally a legacy `.doc`.

**What made it hard.**

- `.doc` is an old binary format that python-docx cannot open.
- Much of the operational content is in tables, not paragraphs.
- The drawings are EMF and WMF, vector formats that no vision model reads.
- A Word file has no fixed pages, so there is no page number to record.

**How we parse them** (`extract_docx`, `extract_doc`):

1. A `.doc` is first converted to `.docx` by LibreOffice running without a
   window, then read like any other `.docx`. The guidelines file in the
   folder today is already the converted copy.
2. Walk the paragraphs in order. A paragraph whose style starts with
   "Heading" closes the current section and opens a new one. Any other
   non-empty paragraph is added to the current section. Text before the
   first heading goes under "Preamble".
3. Each table with at least two non-empty rows becomes its own chunk. Cells
   are joined with ` | ` and rows with a newline, under the heading
   "Table N".
4. Images are pulled out of the file's internal parts. EMF and WMF are
   converted to PNG with ImageMagick.
5. If a section's text contains a four-digit number starting with 5 or 6,
   it is stored as that section's procedure number.

**What comes out.**

| Document | Section chunks | Table chunks | Total | Images |
| --- | --- | --- | --- | --- |
| OPS_MANUAL | 139 | 77 | 216 | 8 |
| GUIDELINES | 80 | 62 | 142 | 21 |

The longest guidelines section is 39,445 characters, which matters in the
next chapter.


### 1.7 The telecommand log (TCH, text)

**How the file was.** A printout of commands sent to the spacecraft: 203
rows over 7 pages, 30 rows to a page. Each row is 11 columns separated by
pipes:

```text
SL.NO | CODE | CID | MNEMONIC | TIME | DATA | DATA VALUE | ORBIT | STATUS | SRC | DST
```

Every page after the first starts with a form feed, a title and the column
header again. The DATA column holds raw command bytes, so the file contains
253 bytes that are not printable text.

**What made it hard.**

- The raw bytes mean the file is not valid UTF-8, and a byte in the data can
  be a newline or a pipe.
- Every row ends with a pipe, which leaves an empty cell at the end.
- The page header and footer sit between rows and get glued to the row
  above them.

**How we parse it** (`extract_tch`):

1. Read the file as bytes and decode as latin-1, which accepts any byte and
   never fails.
2. Find rows by their start, not by line: a serial number of 3 to 6 digits
   at the start of a line followed by a pipe. A record is everything from
   one such anchor to the next. So a row would survive even if its data
   broke it across two lines.
3. Cut the record at a page break: a rule of 40 or more `=` or `-`, or the
   column header line.
4. Split on pipes and drop the empty last cell.
5. Skip anything with fewer than 11 cells.
6. Take the first five columns from the left and the last five from the
   right. Whatever is left in the middle is the DATA column, joined back
   together. So an extra pipe inside the data cannot shift the other
   columns. One row in this file has such a pipe.
7. Split the timestamp into date, time and milliseconds.
8. Replace non-printable bytes in the data with `\xNN` and keep at most 400
   characters of it.
9. Work out the subsystem from the first letters of the CID.
10. Write one plain sentence per record, of the form "Command MNEMONIC
    (CID: ...) uplinked on DATE at TIME from SRC to DST. Orbit ... Status
    ... Subsystem ...". This sentence is what gets searched later; the
    separate fields are kept for exact filtering by date or CID.

**What comes out.** 203 records in `tch_records.json`, all with a date,
status, source and destination. They cover one date and 15 distinct
commands.


### 1.8 What every chunk carries

Each entry in `chunks.json` has the same ten fields:

| Field | Meaning |
| --- | --- |
| `chunk_id` | Unique id, of the form `MOR_s<n>` or `FCP_IDX_<serial>_<number>` |
| `text` | The section, table, index row or procedure |
| `source_doc` | FCP, MOR, OPS_MANUAL or GUIDELINES |
| `doc_filename` | The file it came from |
| `content_type` | `toc_index`, `procedure`, `note` or `section` |
| `page` | Page number; 0 for Word documents |
| `section_heading` | The heading it sits under |
| `fcp_number` | Procedure number, if any |
| `subsystem` | OBC, AOCS, POWER and so on, or UNKNOWN |
| `images` | Reserved; empty |

The subsystem label comes from three places. Index rows read it from the
document's own column. Command records take it from the CID prefix. All
other chunks get it from keywords in the heading and text (first match
wins), then from the procedure-number range, and otherwise UNKNOWN. 185 of
the 781 chunks are UNKNOWN. Retrieval does not filter on this label; an
earlier version did and it lost correct answers, which chapter 5 covers.

Two rules hold for the whole stage:

- **Extraction decides structure, not size.** It emits whole sections,
  however long. 119 of them are over the size we want to search on. Cutting
  them is done at ingest (chapter 2), so a chunk size can be changed
  without parsing the documents again.
- **A failure in one document does not stop the rest**, and a reader that
  recognises nothing falls back to page-by-page text.


### 1.9 Parsing mistakes, and what fixed them

| Where | What went wrong | Fix |
| --- | --- | --- |
| Old pipeline, FCP | The whole 225-row index was stored as one 13,127-character chunk, so no single row could be found | One chunk per row |
| Old pipeline, MOR | Ids restarted per section, so two sections on one page collided; ChromaDB rejected the batch and 92 chunks were lost | One running counter for the whole document |
| Old pipeline, FCP | The procedure number was reset at `END OF PROCEDURE`, so notes pages lost their procedure | A procedure block runs to the next header |
| Old pipeline, Word | Only paragraphs were read; tables were skipped | Each table is its own chunk |
| Rebuild, FCP | The subsystem of an index row was guessed from its title and came out wrong | Read it from the row's own column |
| Rebuild, FCP | The index text lost its "FCP" prefix, so "which FCP ..." questions stopped matching | Text starts with `FCP <number>` |
| Rebuild, MOR | The page lookup took the first page offset, so every section was stamped page 1 | Take the last offset at or before the heading |
| TCH, found 12 November | The empty cell after the closing pipe was counted, so each trailing column was one place out: destination blank, status holding the source station | Drop the empty last cell |
| TCH, found 12 November | The page header was read into the last row of each page, corrupting 6 records | Cut each record at the page break |

The lesson that repeats: each of these produced output that looked fine.
Nothing crashed. They were found by counting (92 missing chunks) or by a
test question that could never be answered (the uplink station).


### 1.10 Numbers to remember

| Item | Count |
| --- | --- |
| Source files | 5 |
| Chunks in `chunks.json` | 781 |
| of which FCP index rows / procedure bodies | 225 / 6 |
| of which MOR sections | 192 |
| of which OPS_MANUAL / GUIDELINES | 216 / 142 |
| Command records | 203 |
| Figures saved | 359 (MOR 330, GUIDELINES 21, OPS_MANUAL 8) |
| Figures large enough to caption (5,000 bytes or more) | 210 |
| Sections longer than their size budget | 119 |


### 1.11 Rough edges still in the parser

- **Word chunks have no page number.** Their citations show the heading
  only.
- **Tables are detached from their section.** A table chunk is labelled
  "Table N" and does not record which heading it sat under.
- **185 chunks have no subsystem.**
- **Procedure numbers in Word sections are loosely matched.** Any
  four-digit number starting with 5 or 6 counts. 8 chunks carry one; whether
  each is a real procedure reference has not been checked.
- **10 MOR pages have almost no text.** They are probably figure-only
  pages, which has not been checked.
- **MOR headings must be in capitals to be seen.** A heading in mixed case
  would be absorbed into the section before it. How many were missed has
  not been counted.
- **The FCP file is a sample**: 6 procedure bodies out of 225.


### 1.12 Saying it in a minute

"We started with five documents of five different kinds: a procedure PDF, a
270-page report PDF, two Word manuals and a pipe-separated command log. I
wrote one parser per kind. The procedure parser separates the index rows
from the procedure bodies, because one answers 'which procedure' and the
other 'how do I'. The report is cut at its numbered headings and each
section keeps its page. The Word files are cut at heading styles, and each
table becomes its own piece. The log is read row by row into 203 command
records with separate fields for date, command id and status. Every piece
is labelled with its document, page, heading and procedure number. The
parser only decides structure; sizes are decided later. Most of our early
bugs were here, and none of them crashed: they were found by counting what
came out."


## Chapter 2: Chunking, and the structure-aware parser we tried

Written 16 November 2026. The structure-aware parser described here is
built and measured but is not the default. Section 2.7 says why.

### 2.1 How chunking works today ("flat")

Parsing gives whole sections (chapter 1). `core/chunking.py` cuts each one
to a size at ingest time:

| Type | Budget | Overlap |
| --- | --- | --- |
| FCP procedure | 2,500 characters | 200 |
| MOR | 1,000 | 150 |
| OPS_MANUAL, GUIDELINES | 1,200 | 200 |
| Index rows, captions, command records | never cut | |

The cutter walks to the budget, then looks backwards for a good place to
stop: a blank line, then the end of a sentence, then a line break, then a
space. The next chunk starts a little before the cut, so the two overlap.
So it is overlap chunking that prefers sentence ends, not a blind cut
every N characters. But it sees a section as one string. It does not know
what a table is, or a paragraph, or a heading.

### 2.2 What that costs, measured

781 sections become 1,474 chunks. Counting the prose chunks only:

| Measure | Count |
| --- | --- |
| Chunks that continue a section | 483 |
| of which start in the middle of a sentence | 221 |
| Tables longer than a chunk | 18 |
| Pieces those tables were cut into | 108 |
| of which begin or end inside a row | 98 |
| Chunks longer than the embedder can read (256 tokens) | 340 of 1,039 |

Three more losses come from parsing, not cutting:

- **The report's contents pages are in the store.** Pages 1 to 15 hold 359
  contents lines, and they went in as 5 sections of 53,000 characters.
- **Page headers and footers are inside the chunks**: about 600 lines.
- **A Word table is cut off from its section.** It is stored under
  "Table N", with nothing to say what it is a table of.

### 2.3 What the documents offer that we were not using

- **The report PDF has 188 bookmarks**, with levels. They are the
  document's own list of its sections. The flat reader guesses headings by
  pattern (a number, then a title in capitals) and misses headings in
  mixed case.
- **PyMuPDF can find tables** and return them row by row: 179 in the body
  of the report.
- **Word files can be walked in order**, so a table stays under its
  heading, after the sentence that introduces it.
- **Heading levels** give a path such as "Power system > Battery".

One thing we could not use: the procedure step tables. The PDF places each
column's text at its own height, so rows cannot be rebuilt from positions.
Procedures are still read the old way.

### 2.4 What we built ("structure")

Switched on per satellite with `parsing: {mode: structure}`. The default
stays `flat`.

**Reading** (`core/layout.py`). A section comes out as an ordered list of
blocks, each a paragraph or a table.

- Report: sections are taken from the bookmarks (146 of 188 found as a
  heading on their page; the rest start at the top of the bookmarked
  page). Headers and footers are removed by position and repetition.
  Contents lines are removed. 165 tables are read by row. Each section
  records its heading path.
- Word: the body is walked in order. Each table stays in its section, with
  the caption above it as its title. Merged cells are read once, not once
  per column they span.

**Cutting** (`chunking.split_blocks`). Same budgets as before.

- A cut falls between paragraphs or between table rows, never inside one.
- A table that needs several chunks has its header row repeated on each.
- A short line ending in a colon moves to the next chunk, to stay with the
  list or table it introduces.
- Only a paragraph longer than a whole chunk is cut inside, between
  sentences, and only then is a sentence repeated as overlap.
- The heading path is put in front of the text that is embedded and
  indexed.

**Comparing** (`compare_parsing.py`). The new version is extracted and
embedded into `extracted_structure/` and `vectorstore_structure/`. The
working store is not touched.

### 2.5 The result

Chunks: 1,039 prose chunks became 779. Chunks starting mid-sentence fell
from 221 to 40. No table row is cut. The contents pages are gone.

Retrieval, 100 questions, top 20 scored, MRR:

| Setting | Flat | Structure | Change | 95% interval |
| --- | --- | --- | --- | --- |
| Hybrid + reranker (as shipped) | 0.881 | 0.873 | -0.008 | -0.037 to +0.021 |
| Hybrid | 0.830 | 0.829 | -0.001 | -0.049 to +0.044 |
| Dense only | 0.785 | 0.778 | -0.008 | -0.050 to +0.034 |
| BM25 only | 0.698 | 0.725 | +0.027 | -0.006 to +0.064 |

This table does not count a contents-page chunk as a right answer (2.6
explains). On this test set the two parsers are level: every interval
includes zero. Cleaner chunks did not move the retrieval score.

### 2.6 Three things this found

**The contents pages were flattering the old scores.** A contents line
holds the words of a heading, so it satisfies a test condition written for
the section under that heading, and answers nothing. With BM25 alone, the
first "right" chunk was a contents page for 17 of 100 questions, and BM25's
MRR falls from 0.804 to 0.698 when those are not counted. In the shipped
setting the effect is small: 2 questions (`power_budget`,
`mission_overview`), 0.884 to 0.881. But a contents chunk was reaching the
top 5 that go to the language model about once in every four questions.

**Chunk size is not the lever.** Prose budgets of 800, 600 and 450
characters were tried on both parsers. None beat the current budgets, and
450 was clearly worse (0.857 flat, 0.850 structure, against 0.885 and
0.880 with one unstable question left out).

**Retrieval is not fully repeatable.** For the same query vector, the
vector store's approximate search sometimes returns a different set of
neighbours in a different process. It was seen in 1 run of 5 on a
command-log question (`cmd_mep`, ranks 2, 5 and 14), where many records are
nearly identical. Between two scoring runs of the same structure store,
dense MRR read 0.769 and 0.778. This contradicts the note in
`PROJECT_CONTEXT.md` that retrieval is deterministic, and may explain the
old unreproduced "noise floor" of 0.005.

### 2.7 Where it stands

- The default is still `flat`. Nothing in `results/` or the docs changes.
- The structure parser is not shown to retrieve better, so there is no
  measured reason to switch. Its gains are in what the chunk looks like
  when it reaches the language model: whole rows with their header, whole
  sentences, no contents lines. The test set checks whether a right chunk
  was found, not whether the answer written from it was better, so it
  cannot see those gains.
- To decide, compare answers, not ranks: run the same questions through
  both stores and count wrong or missing values in the answers,
  especially for questions answered from tables.
- Whatever is decided, the contents pages should come out of the flat
  store too, and the test conditions that they satisfy should be checked.

### 2.8 Saying it in a minute

"Our chunking cut sections near a character count with an overlap. I
measured what that broke: 221 chunks started mid-sentence, and tables were
cut inside rows. So I built a second parser that uses the document's own
structure: the PDF's bookmarks for sections, tables read row by row with
the header repeated, cuts only between paragraphs. The chunks came out
much cleaner, but on our 100 questions retrieval was level, not better.
The useful result was a side finding: the table of contents was sitting in
the store and counting as a right answer for 17 questions under keyword
search. So I kept the old parser as the default, and the next step is to
compare the written answers, which is where cleaner chunks should show."


## Chapter 3: Figure captioning

Written 16 November 2026, and revised the same day after the context
fix in section 3.12. Code: `core/caption.py`, with figure placement in
`core/extract.py`. Run with:

```text
python cli.py caption aditya-l1              describe the figures not done yet
python cli.py caption aditya-l1 --limit 5    try it on five
python cli.py caption aditya-l1 --force      describe every figure again
```

### 3.1 Why we caption at all

Search works on text. A diagram has none, so a question such as "show me
the propulsion schematic" has nothing to match. The fix is to have a
vision model write a short description of each figure once, and store that
description as a chunk like any other. The figure is still not searchable;
its description is, and the description carries the path of the image
file.

### 3.2 Which figures get captioned

Parsing saved 359 images (chapter 1). Not all of them are figures:

| Source | Saved | 5,000 bytes or more | Captioned |
| --- | --- | --- | --- |
| MOR (PDF) | 330 | 181 | 181 |
| GUIDELINES (Word) | 21 | 21 | 21 |
| OPS_MANUAL (Word) | 8 | 8 | 8 |
| Total | 359 | 210 | 210 |

Two filters decide:

- **File size.** Anything under 5,000 bytes is skipped. Those 149 files are
  rules, bullets, logos and spacers.
- **Pixel size.** Anything under 60 pixels on a side is skipped and the
  skip is recorded, so it is not tried again. None of the 210 was this
  small.

The 28 Word drawings were EMF or WMF, vector formats a vision model cannot
read. Parsing had already turned them into PNG with ImageMagick.

### 3.3 How one figure is captioned

1. **Place the figure.** Extraction records, in `extracted/figures.json`,
   where each figure sits. In a PDF that is the figure's rectangle on its
   page. In a Word file it is the paragraph that holds the picture.
2. **Find its section.** A section that starts on the figure's page counts
   only if its heading is printed above the figure. Otherwise the figure
   belongs to the section running on from an earlier page. For a Word
   figure it is the section under the nearest heading above it.
3. **Collect the text beside it.** In a PDF, the three blocks of text above
   the figure and the three below, leaving out page headers and footers.
   In a Word file, the two paragraphs before and the two after.
4. **Build the context.** The document name, the section's heading,
   procedure number and subsystem, then the text beside the figure. If
   there is under 200 characters of that (a figure that fills its page),
   the opening of the section is added.
5. **Prepare the image.** Pillow opens it, converts odd colour modes to
   RGB, re-saves it as PNG in memory and base64-encodes it. This keeps
   progressive JPEGs and unusual colour spaces away from the model.
6. **Ask the vision model.** One request with the prompt, the context and
   the image.
7. **Clean the reply.** Labels such as "Caption:" or "The image shows" are
   stripped from the start.
8. **Save progress.** The caption goes into a state file, which is written
   to disk every 10 figures.

When all figures are done, the state is turned into
`extracted/image_captions.json`, one chunk per caption. That file is
rewritten on every run, even when there is nothing new to describe, so a
change in where figures are filed reaches old captions without the model.

### 3.4 What the prompt asks for

The model is told it is a documentation analyst for a spacecraft mission,
is given the context, and is asked for a caption "optimised for semantic
search". It must include:

1. The figure type (block diagram, orbit plot, telemetry chart, flow
   chart, schematic, photograph).
2. The mission phase or operation shown.
3. The subsystem, if it can be told.
4. Any labels, codes, mnemonics, parameter names and numbers that are
   legible.
5. What the figure is for in operations.

And it must: use operations terms, copy acronyms and codes exactly, not
guess beyond the figure and the context, stay within 5 sentences, and
start with the content, with no "Caption:" or "This figure".

The context matters most for plots. A graph with unlabelled axes can only
be described well if the surrounding text says what is being plotted.

### 3.5 Model and settings

| Setting | Value | Why |
| --- | --- | --- |
| Model | `qwen2.5vl` through Ollama | Local, so no figure leaves the machine |
| Context window | 8,192 tokens | The Ollama default of 4,096 is too small for a full-page figure; the request fails outright |
| Temperature | 0.2 | What the captions were first made with |
| Seed, repetition penalty | fixed, 1.0 | Set in `core/llm.py`, so a figure captioned twice gives the same text |
| Reply limit | 400 tokens | |
| In flight at once | 1 by default | A vision model on a 6 GB card may not fit two |

`SATRAG_LLM_CONCURRENCY` raises the number in flight. It only helps if the
server can overlap requests: Ollama started with `OLLAMA_NUM_PARALLEL`
above 1, or vLLM.

### 3.6 Stopping and resuming

Captioning 210 figures takes a long time on a laptop, so it is built to be
interrupted:

- The state file `extracted/.captioning_state.json` is keyed by image
  filename. A figure already in it is not sent again.
- A figure that fails (model error, empty reply) is counted and left out
  of the state, so the next run tries it again.
- A figure that is skipped as too small is written to the state with the
  reason, so it is not retried.
- With several in flight, figures are done in windows of that size, so an
  interrupted run loses at most one window.

### 3.7 What a caption chunk carries

| Field | Value |
| --- | --- |
| `chunk_id` | `aditya-l1_cap_<image name>` |
| `text` | The caption |
| `content_type` | `image_caption` |
| `source_doc`, `page` | From the image filename |
| `section_heading`, `fcp_number`, `subsystem` | Copied from the section found in step 2 |
| `image_file` | Full path of the image |

The heading and subsystem are copied from the document, not guessed from
the caption's wording. Captions are never split at ingest, whatever their
length.

### 3.8 How captions are used later

- **Embedding.** The string that is embedded starts with "Image caption",
  the source document and the page, then the heading, then the caption.
- **Routing.** A question with a word such as diagram, figure, plot or
  schematic gets the intent `figure`.
- **Retrieval.** For that intent, a second search runs over captions only
  (dense and BM25 fused), and its best two are placed first.
- **Answering.** If the top result is a caption, the answer is built from
  the caption directly, without the text model, and the image file is
  listed in the sources. A text model asked to "show" a diagram tends to
  reply that it cannot.

### 3.9 Numbers

These describe the captions written on 16 November with the corrected
context. The first set, written in August, is in the last column.

| Item | Now | Before |
| --- | --- | --- |
| Captions | 210 | 210 |
| Length, characters | 241 to 797, median 519 | 271 to 853, median 472 |
| Sentences, median | 4 | 3 |
| Captions past the limit of 5 sentences | 4 | 2 |
| Captions still starting with a label | 0 | 0 |
| Captions filed under a heading | 209 | 181 |
| Captions with a known subsystem | 108 | 109 |
| Captions that quote a command or telemetry identifier | 3 | 0 |
| Test questions that need a caption | 7 | 7 |
| of which answered at rank 1 | 6 | 6 |

The seventh, `odhs_diag`, is still a miss. One caption now satisfies its
condition, where none did before, but it is not retrieved in the top 20.
Whether that caption is the right figure has not been checked.

### 3.10 Problems met and fixed

| Problem | Fix |
| --- | --- |
| Six full-page figures failed with "exceed context size" in the old pipeline | Context window set to 8,192 |
| The 3B vision model began 70 of 100 captions with "**Caption:**" in bold, which the plain patterns missed | A pattern for the bold form; labels are stripped again at ingest |
| An exact page match often found no text, because the text near a figure belongs to a section that began on another page | Lookup by position in the sorted page list |
| "Show me" was a figure word, so "show me the steps" was answered from a caption | Removed from the figure pattern |
| A dense search alone missed some captions | Captions are searched by dense and BM25 together |
| A crash meant starting again | State file and resume |

### 3.11 Speed, measured

`bench/caption_bench.py` timed the real captioning call on 100 figures,
with Qwen2.5-VL 3B at 4 bits on the laptop card. The 7B vision model does
not fit there under vLLM.

| Server | In flight | Wall clock | Per figure |
| --- | --- | --- | --- |
| Ollama, one at a time | 1 | 418.8 s | 4.19 s |
| Ollama, 2 slots | 4 | 324.4 s | 3.24 s |
| vLLM | 8 | 80.6 s | 0.81 s |

vLLM is 4.0 times Ollama with 2 slots, and 5.2 times Ollama one at a time.
This has not been repeated against Ollama with more slots. The 3B captions
overlap the 7B captions with a token F1 of about 0.35 to 0.38, so they are
different text, and the stored captions are still the 7B ones.

### 3.12 The context was wrong, and the fix

Writing this chapter turned up two faults in how a figure's context was
found. Both were fixed on 16 November.

**What was wrong**

- **The wrong section for most report figures.** The lookup took the first
  section starting on or after the figure's page. When no section starts
  on that page, that is the following section, a median of 3 pages ahead.
  128 of 181 report figures got the next section's opening as context and
  were filed under its heading.
- **No context for Word figures.** Word chunks have no page number, so the
  lookup found nothing. All 29 were captioned from the image alone, with no
  heading and no subsystem.
- **The context was the section's opening**, 700 characters that could be
  pages away from the figure, even when the section was the right one.

**What was done**

1. Extraction now places every figure (section 3.3, steps 1 to 3). All 359
   images were placed, 358 with a section.
2. The old captions were re-filed under the right sections without calling
   the model. 188 of 210 changed heading: 160 report figures, and 28 of the
   29 Word figures, which had none.
3. All 210 figures were captioned again with the corrected context. It
   took about an hour on the laptop, with no failures.
4. The new captions were scored in a store of their own before replacing
   the working one.

Of the 181 report figures, 24 are filed under a section whose heading is
printed above them on their page and 157 under the section running on
from an earlier page.

**What changed**

| Measure | Before | After |
| --- | --- | --- |
| Captions using a word from their section's heading | 54 of 191 | 178 of 191 |
| Figure questions at rank 1 | 6 of 7 | 6 of 7 |
| MRR as shipped (hybrid + reranker) | 0.884 | 0.879 |
| Hit@5 as shipped | 93% | 92% |
| MRR, dense only | 0.792 | 0.825 |
| Captions among the top 5 results, per question | 0.55 | 0.97 |

The first row is the point of the fix: the captions now talk about the
section they are in. The old captions are compared against their correct
heading there, and almost none of them mentioned it.

Retrieval as shipped is level: the change of -0.004 has an interval of
-0.029 to +0.022. Dense search alone improved by +0.033, with an interval
of +0.001 to +0.071.

One side effect to watch. A caption now reaches the top 5 for almost every
question, up from about one question in two. The captions share words
with their sections, so they rank for ordinary questions too, and each one
takes a place a text passage would have had. That is the likely reason
Hit@5 went from 93 to 92.

**Where the old versions are**

- `extracted/image_captions.before-context-fix.json` and
  `extracted/.captioning_state.before-context-fix.json`: the August
  captions.
- `vectorstore_before-caption-fix/`: the working store as it was.

To go back, restore the two files and swap the store folders.

### 3.13 The captions read against their figures

On 17 November the documents were cleared as dummy data, so the 30
figures on the review page were read against their captions by Claude
(the assistant writing this), not by a person. One reader, one pass, with
the page's marking guide: **right** says what the figure is with nothing
false; **partly** has the kind of figure right but a detail wrong or
made up; **wrong** describes something else. The marks, with a note on
each, are in `extracted/caption_review_claude.json`, kept out of git.

| Mark | Captions, of 30 |
| --- | --- |
| Right | 14 |
| Partly | 15 |
| Wrong | 1 |

Only one caption describes something else, and only because there was
nothing to describe: `GUIDELINES_img016` is a signature, and the caption
is a power-limits diagram taken from the text around it. Plots of
measured data came out best: the velocity-difference plots were right in
their title, case and dates.

The 15 partly-right captions fail in a few repeated ways:

| Fault | Captions | Example |
| --- | --- | --- |
| The surrounding text written up as if the figure showed it | 6 | an exclusion zone on a plain orbit plot; a "stowed configuration" on an axes diagram |
| Labels read as the wrong kind of thing | 3 | four ground stations called subsystems or mnemonics; visibility windows called manoeuvre times |
| A number or detail misread | 4 | x-axis day markers given as angles; the solar disk and ring sizes swapped; an end date 9 days short; a point called highlighted that is not |
| An abbreviation expanded wrongly | 2 | a Sun exclusion zone called a "Sun-Earth Zonotope"; a correction manoeuvre called a translunar injection |

(Some captions have more than one fault; each is counted under its main
one.) Two faults run through the right ones too, without making them
false: tables are called "telemetry charts" or "command tables", and an
equation a "block diagram".

Two of these come from the prompt (section 3.4) rather than the model:

- **Its list of figure kinds has no table.** 6 of the 30 are tables, so
  the model picks the nearest kind it is offered.
- **It asks every caption for the mission phase and the operational
  use.** Where the figure does not show them, the model supplies them
  from the context or invents them ("during the Normal Phase", "the
  propulsion system's performance in drag-free mode"). That is the first
  fault in the table.

What it means for the app: a caption is good enough to be found, and the
title, kind and dates of a figure are mostly right, but a reader should
not take a number or a name from a caption without the image. A caption
answer already lists the image file for that reason (chapter 7).

### 3.14 Still open

- **Act on the reading (section 3.13).** Add "table" and "equation" to
  the prompt's figure kinds; ask for the phase and the operational use
  only when the figure or its own caption text shows them; drop
  signatures and logos before captioning. Then caption again and read a
  fresh sample. A person reading a few of the same 30 would also show
  whether the AI marks can be trusted.
- **Captions crowd the top 5** (section 3.12). If the review shows they
  help answers, leave it. If not, limit captions to one of the five places
  unless the question asks for a figure.
- **The ten per-model stores and the tables in `results/` were built with
  the August captions.** The headline on the working store is now Hit@1
  84%, Hit@5 92%, Hit@10 97%, MRR 0.879. The sweeps have not been run
  again.
- **102 of 210 captions have no subsystem.** The section they are in has
  none either.
- **8 guidelines captions carry a procedure number** that their section
  got from the loose matching described in section 1.11. It is not one of
  the test set's answers.
- **For a few report figures the heading on the page could not be found**,
  and the earlier section is assumed.
- **Seven figure questions is thin.**

### 3.15 Saying it in a minute

"A diagram cannot be searched, so I had a local vision model write a short
technical description of each one and stored it as text. Of 359 images
pulled from the documents, 210 were real figures; the rest were bullets
and rules, filtered by size. For each figure the model gets the heading of
the section it sits in and the text printed beside it, because a plot with
unlabelled axes cannot be described from pixels alone. At question time, a
word like 'diagram' triggers a search over captions only, and the answer
is built from the caption with the image attached. Reviewing this stage I
found a real fault: the code was giving most figures the following
section as context, and Word figures no context at all. I fixed the
placement by reading each figure's position from the document, captioned
all 210 again, and measured it. Captions that mention their own section's
heading went from 54 to 178 of 191, with retrieval level. Reading 30 captions
against their figures, 14 were right, 15 partly right and 1 wrong, and
the wrong one was a signature. The common fault is writing the
surrounding text up as if the figure showed it, partly because the prompt
asks every caption for a mission phase and has no 'table' in its list of
figure kinds; that is the next fix."



## Chapter 4: Embedding and the two stores

Written 17 November 2026. Code: `core/embeddings.py` (335 lines) and
`core/ingest.py` (about 290). Run with:

```text
python cli.py models                     which embedding models are downloaded
python cli.py prefetch minilm-l6         download one, once
python cli.py ingest aditya-l1           embed what is not in the store yet
python cli.py ingest aditya-l1 --reset   empty the store and build it again
```

### 4.1 What this stage does

Parsing and captioning left three files of text: sections, captions and
command records. Ingest turns them into something that can be searched in
two ways:

- **By meaning.** Each chunk is turned into a list of 384 numbers (an
  embedding, or vector). Two texts about the same thing get vectors that
  point the same way, even if they share no words.
- **By exact words.** A keyword index (BM25) records which words appear in
  which chunk.

Both are built here and saved in `satellites/<slug>/vectorstore/`.
Retrieval (chapter 5) reads them.

### 4.2 Ingest, step by step

1. **Load** `chunks.json`, `image_captions.json` and `tch_records.json`.
2. **Drop duplicate ids.** ChromaDB rejects a whole batch if two items in
   it share an id. The old pipeline lost 92 chunks that way. None are
   dropped now.
3. **Cut to size** (chapter 2). 991 sections and captions become 1,474
   chunks; 602 of them are pieces of a longer section and keep a link to
   their parent.
4. **Build the string to embed** for each chunk (section 4.4).
5. **Embed** in batches and write to ChromaDB with the text and metadata.
6. **Build the keyword index** over the same strings and save it.
7. **Write `build.json`**: which model built the store, its dimension, and
   the counts.

### 4.3 The embedding model

| Property | Value |
| --- | --- |
| Model | `all-MiniLM-L6-v2` ("minilm-l6") |
| Vector size | 384 numbers |
| Longest input it reads | 256 tokens; anything after that is ignored |
| Size on disk | 88 MB |
| Where it runs | GPU if there is one, in half precision |
| Vectors | Scaled to length 1, compared by cosine similarity |

It was chosen by measurement. In the ten-model comparison it was level
with models fifteen times its size once keyword search was added
(chapter 8).

Things the code takes care of:

- **Offline.** Models are downloaded once into `models/` and loaded from
  there. Nothing is fetched at query time.
- **A registry.** 15 models have short names. Any Hugging Face id also
  works as a name, so a new model can be tried without editing code.
- **Prefixes.** The E5 models score badly unless every query starts with
  "query: " and every passage with "passage: ". The registry adds them.
- **Batch size from the card.** 48 on this 6 GB laptop for a 384-number
  model, half that for the 1,024-number models. If a batch still runs out
  of memory, the size is halved and tried again.
- **Campus network.** The operating system's certificate store is used, so
  downloads work behind a network that inspects HTTPS.

### 4.4 What text is actually embedded

Not the chunk alone. `ingest.embed_text` puts labels in front of it, one
per line:

```text
Image caption            only for captions, with Source and Page
FCP <number>             if the chunk has a procedure number
Subsystem: <name>        if known
Command: <mnemonic>      command records
CID: <id>                command records
<section heading>
<the chunk's text>
```

This is why a question such as "OBC switch on procedure" can reach a
chunk whose own text never says "procedure". The labels cost a median of
9 to 25 tokens, depending on the type.

The same string is used for the keyword index. The text stored for
display, reranking and the language model is the chunk alone, without the
labels.

### 4.5 Two collections

| Collection | Holds | How it is searched |
| --- | --- | --- |
| `knowledge` | Sections, procedures, index rows, captions | By meaning and by keywords, best few |
| `commands` | The telecommand log | By exact date, command id or mnemonic when the question gives one; every match, in time order |

They are split by how they are used, not by document. "What was sent on
this date" wants the whole day, not the five records most like the
question. Mixing the log into the prose would make both searches noisier.

Document types are not split into separate collections. They share
`knowledge` and carry a `doc_type` field, so a search can be limited to
some types, or can return five results from one document when that is
where the answer is.

Each record carries 24 metadata fields: where it came from (satellite,
document type, file, page, heading, procedure number, subsystem), how it
was cut (whether it is a piece, its parent, its position), the command
fields (date, time, command id, mnemonic, orbit, status, stations), and
the image file for a caption. ChromaDB accepts only plain values, so
everything is flattened to text, numbers and true or false.

### 4.6 The keyword index

BM25 scores a chunk by how many of the question's words it contains,
giving more weight to rare words and less to long chunks. The library is
`rank_bm25`, and the index is saved with pickle.

What matters is how text is split into words. `ingest.tokenize` keeps a
compound identifier whole and also adds its parts:

```text
"OBC-1 ON / OBC-2 ON, DC-DC-11, L1/L2"
->  obc-1  obc-2  dc-dc-11  l1/l2  obc  1  on  2  dc  11  l1  l2
```

Without the parts, a search for "OBC" scores zero against a title that
says "OBC-1". Without the whole, a search for "DC-DC-11" cannot be told
from a search for "DC".

### 4.7 Numbers

| Item | Value |
| --- | --- |
| Vectors in `knowledge` | 1,474 |
| of which index rows / captions / procedure pieces / prose | 225 / 210 / 10 / 1,029 |
| by document: FCP / MOR / OPS_MANUAL / GUIDELINES | 235 / 698 / 255 / 286 |
| Vectors in `commands` | 203 |
| Store on disk | about 17 MB |
| Keyword index, knowledge | 11,523 distinct words, median 62 per chunk |
| Keyword index, commands | 118 distinct words |
| Embedding 1,474 chunks | 2.0 s on the laptop GPU (about 740 per second) |
| A full rebuild with `--reset` | 11.6 s |
| Characters per token in these documents | 3.8 |

A rebuild is cheap. That is what made the ten-model comparison and the
chunk-size sweep practical.

### 4.8 Design decisions

- **Embed labels with the text.** A short chunk says little on its own;
  its heading and procedure number say what it is about.
- **Two collections by access pattern** (section 4.5).
- **Identifiers indexed whole and in parts** (section 4.6).
- **One store per satellite, and the model's name recorded with it.**
  Satellites may use different models. Results across satellites are
  merged by rank, not by score, so that still works.
- **A separate store per model for experiments.** `compare_models.py`
  builds each model's store in its own folder
  (`vectorstore_<model>/`), so the working store is never overwritten.
- **Chunking at ingest, not at parsing.** A chunk size can be changed and
  the store rebuilt in seconds, without reading the documents again.

### 4.9 Rough edges found while writing this

Measured or read from the code on 17 November. None is fixed.

- **The embedder does not read all of a long chunk.** 340 of 1,474 chunks
  (23%) are longer than its 256 tokens, and the rest of each is ignored by
  the meaning search. The keyword search still sees all of it.

  | Type | Chunks | Over the limit | Share of a chunk read, median | Worst |
  | --- | --- | --- | --- | --- |
  | Procedure bodies | 10 | 10 | 40% | 26% |
  | GUIDELINES | 265 | 147 | 82% | 40% |
  | MOR | 517 | 129 | 100% | 29% |
  | OPS_MANUAL | 247 | 54 | 100% | 42% |
  | Captions | 210 | 0 | 100% | 100% |
  | Index rows | 225 | 0 | 100% | 100% |

  Procedures are the worst case: the meaning search sees the first 40% of
  each. The 2,500-character budget was chosen so a procedure is not cut
  in half, but the embedder reads about 1,000 characters. Smaller chunks
  did not score better in the sweep (section 2.6), so this is a known
  limit, not a proven loss.

- **`ingest` without `--reset` does not notice changed text.** It skips any
  id already in the store. If a caption or a section is rewritten but
  keeps its id, the old vector stays. The keyword index is rebuilt every
  time, so the two can disagree. After changing text, use `--reset`.

- **Nothing checks which model built a store.** `build.json` records it,
  and a comment says this is so retrieval can warn on a mismatch, but no
  code reads the file. Querying a store with a different model in the
  config would give wrong results with no warning, or an error if the
  vector sizes differ.

- **Underscores are not split.** The tokenizer splits compounds at `-`,
  `/` and `.`, but not at `_`. A made-up `FCP_5201` stays one word. Command
  mnemonics are words joined by underscores, so a keyword search for one
  part of a mnemonic does not match it.

- **The command log is full of near-copies.** Its 203 records make only 68
  distinct strings, and 14 once the digits are blanked. Their scores tie.
  This is where ChromaDB's approximate search was seen to return different
  neighbours from one run to the next (section 2.6). With 203 records an
  exact search would cost nothing and would always give the same answer.

- **The keyword index file holds the full text of every chunk**, as the
  store does. Both are kept out of git. The file is read from disk on every
  query, which takes 23 ms.

### 4.10 Saying it in a minute

"Every chunk is stored twice. Once as a vector from a small embedding
model, MiniLM, which finds text that means the same thing. And once in a
BM25 keyword index, which finds exact words, and that matters here because
the documents are full of codes and procedure numbers. Before embedding I
put the chunk's heading, procedure number and subsystem in front of its
text, so a short chunk still says what it is about. My tokenizer keeps an
identifier like OBC-1 whole and also adds its parts, so both an exact
search and a search for just OBC work. The telecommand log has its own
collection, because a date question wants every record for that day, not
the five most similar. Everything is local, and a full rebuild takes
about twelve seconds, which is what let me compare ten embedding models.
One limit I measured: the model reads only 256 tokens, so for about a
quarter of the chunks the meaning search sees only the first part."


## Chapter 5: Retrieval

Written 17 November 2026. Code: `core/retrieve.py` (about 500 lines) and
`core/rerank.py` (126). All numbers here are from the working store as it
is now, with the captions rewritten on 16 November. Try it with:

```text
python cli.py query aditya-l1 "your question" --no-answer --show-text
python diagnose.py            what each pass returns on its own
python evaluate.py --compare  reranker off against on
```

### 5.1 What retrieval has to do

Given a question, return the five passages most likely to answer it, out
of 1,474. Those five are all the language model sees, so a passage that
is not among them cannot be in the answer.

No single method finds everything in these documents:

- A question in ordinary words ("how do I bring the onboard computer up")
  shares no words with a procedure titled with abbreviations. Only a
  search by meaning finds it.
- A question with a code in it (a procedure number, a command id, a
  hardware name) needs that exact code. A search by meaning treats codes
  as near-noise. Only a keyword search finds it.

So both run, and their results are merged.

### 5.2 The whole path, in order

This is `retrieve.search_one`, step by step, for one question.

1. **Plan.** The router decides what the question wants and which store
   holds it (chapter 6). Everything below follows that plan.
2. **Command log, if the plan says so.** If the question gives a date, a
   command id or a known mnemonic, every matching record is read straight
   from the log, in time order. If the plan is the log alone and something
   was found, that is the result and the steps below are skipped.
3. **Meaning search.** The question is embedded and the 20 nearest chunks
   are taken from `knowledge`.
4. **Keyword search.** BM25 scores every chunk against the question's
   words and the best 20 are taken.
5. **Procedure search.** With the default (soft) router, the 20 nearest
   chunks from the procedure document alone are taken as a third list.
6. **Fuse.** The three lists are merged into one by Reciprocal Rank Fusion
   (section 5.4).
7. **Rerank.** The top 20 of the fused list are re-ordered by a second
   model that reads the question and each passage together (section 5.5).
8. **Figure rule.** If the question asks for a figure, a search over
   captions only is run and its best two are put first.
9. **Procedure rule.** Only with the rules router: the two nearest
   procedure chunks are put first. The soft router does not do this.
10. **Add command records**, if the plan asked for both stores.
11. **Cut** to the top 5.

### 5.3 The two passes

**Meaning ("dense").** The question goes through the same embedding model
as the chunks. ChromaDB returns the chunks whose vectors are closest by
cosine similarity. A filter on document type, subsystem or date can be
applied.

**Keywords (BM25).** The question is split into words by the same
tokenizer used at ingest, and every chunk is scored. Chunks scoring zero
are dropped.

One rule holds both together: **the same filters go to both passes.** An
early version let the keyword pass keep chunks the meaning pass had
filtered out, and they came back in through the keyword list.

There is no subsystem filter unless the user asks for one. Guessing the
subsystem from the question and filtering on the guess lost correct
answers, because the documents file things differently from how people
ask about them. A wrong guess removes the right chunk before anything
scores it.

### 5.4 Fusion: Reciprocal Rank Fusion

The two passes give scores that cannot be compared: a cosine similarity
near 0.6 and a BM25 score near 7. RRF ignores the scores and uses only the
positions. A chunk's fused score is the sum, over every list it appears
in, of

```text
1 / (60 + position)        position 1 is the top of a list
```

So a chunk that is 1st in one list and 3rd in another scores
1/61 + 1/63. A chunk that is 1st in one list and absent from the others
scores 1/61. Being found by two methods beats being top of one. The 60
keeps one very high position from dominating.

Because only positions are used, the same formula merges results from
several satellites, even if they were embedded with different models.

### 5.5 The reranker

Fusion orders by arithmetic; it has not read anything. The reranker is a
cross-encoder (`ms-marco-MiniLM-L-6-v2`). It takes the question and one
passage together and outputs one score for the pair. That is slower than
comparing vectors, so it only runs on the top 20 of the fused list.

It runs before the figure and procedure rules, not instead of them. A
cross-encoder rates an index row highly for a how-to question, because
the row reads like a title match, but a row has no steps. So relevance is
judged first, and the rules that know what kind of passage was asked for
are applied after.

If the reranker cannot be loaded, the fused order is used and nothing
fails. It can be switched off per satellite with `rerank: false`.

### 5.6 Why a second search, not a score multiplier

To push a figure or a procedure to the front, the code runs a second,
filtered search and puts its best two results first (`prepend`). The
obvious alternative, multiplying the scores of the wanted kind, does not
work with RRF: all fused scores lie between 1/61 and 1/80, so a multiplier
moves things by hundredths and the wanted item often does not move at
all.

### 5.7 What reaches the language model

- The top 5 passages (`top_k`).
- Each one is headed with its citation: document, procedure number,
  subsystem, page and heading.
- At most 2,000 characters of each passage, and 16,000 in total. A date
  lookup in the command log can return a whole day of records; the context
  stops at the limit and says how many sources were left out.
- For display, sources that share a procedure number and heading are shown
  once.

### 5.8 Numbers

**What each stage scores** (100 questions, top 20 scored):

| Stage | Hit@1 | Hit@5 | Hit@10 | Found in 20 | MRR |
| --- | --- | --- | --- | --- | --- |
| Meaning only | 77% | 89% | 94% | 96% | 0.825 |
| Keywords only | 76% | 90% | 94% | 97% | 0.824 |
| Both, fused | 80% | 95% | 97% | 98% | 0.862 |
| Both, fused, reranked (as shipped) | 84% | 92% | 97% | 98% | 0.879 |

The keyword figure is helped by contents pages counting as right answers
(section 2.6). `PROJECT_CONTEXT.md` gives the fusion gain as +0.060; that
was measured with the earlier captions, when the meaning search scored
0.792.

**How the two passes complement each other** (the 94 questions answered
from the documents; 6 go to the command log alone):

| Right chunk in the top 20 of | Questions |
| --- | --- |
| Both passes | 88 |
| Meaning only | 2 |
| Keywords only | 3 |
| Neither | 1 |

The two top-20 lists share little: a median of 21% of their chunks are in
both. They mostly agree on the right chunk and disagree on everything
else, which is why fusing them lifts the right one. After fusing the three
lists there are a median of 49 distinct chunks.

**What the reranker does** to the right chunk within the top 20:

| Effect | Questions |
| --- | --- |
| Moved it up | 14 |
| Moved it down | 8 |
| No change | 72 |

That is the trade seen in the table above: Hit@1 goes from 80 to 84, and
Hit@5 from 95 to 92.

**Time per question**, warm, on the laptop:

| What | Median |
| --- | --- |
| Whole search, reranker on | 224 ms |
| Whole search, reranker off | 139 ms |
| Reranking 20 passages, timed alone | 106 ms |
| Meaning search, timed alone | 15 ms |
| Keyword search, timed alone | 16 ms |

The first search in a process also loads the two models and is much
slower.

**What the top 5 is made of**, as shipped, over all 100 questions:

| Kind | Share of the 500 places |
| --- | --- |
| Prose sections | 47% |
| Procedure index rows | 24% |
| Figure captions | 19% |
| Command records | 6% |
| Procedure bodies | 3% |

### 5.9 Mistakes made here, and their fixes

| Mistake | What it did | Fix |
| --- | --- | --- |
| The procedure search filtered on a field the store does not have | The step silently did nothing, from August to 28 October; every August number was measured without it | Field name corrected |
| The procedure search was narrowed to procedure bodies | It excluded 225 of 231 procedure chunks; Hit@1 fell to 42% and the reranker hid it | Search the whole procedure document |
| The keyword pass was more permissive than the meaning pass | Filtered-out chunks came back through the keyword list | Identical filters on both |
| The reranked list was returned as it was | An index row outranked the procedure body, and the steps fell out of the top 5 | Rerank first, then apply the rules |
| The subsystem guess became a hard filter | Correct chunks were removed before scoring | No subsystem filter unless asked for |
| The command search ignored the dense or keyword mode | A "dense only" evaluation still used keywords for command questions | It follows the mode |

One lesson from development is worth keeping. At one point the reranker
appeared to add +0.39 MRR. A reranker that adds that much is not a strong
reranker; it is a sign that the retrieval under it is broken. That number
is what led to the second mistake in the table.

### 5.10 Rough edges found while writing this

Measured on 17 November. None is fixed.

- **Half the top 5 is not prose.** Index rows take 24% of the places and
  captions 19%. Measured by kind of question in chapter 6: the index rows
  are almost all on procedure questions (2.0 of the 5 places), where they
  belong, since most procedures exist only as an index row. Concept
  questions get 0.3. The captions are the ones out of place: concept
  questions get 1.6 captions in their 5. Whether that helps or hurts
  answers has not been measured.
- **The reranker costs three points of Hit@5** (95% to 92%). It moves the
  right chunk down for 8 questions. The language model reads all five, so
  Hit@5 matters more to the answer than Hit@1.
- **The reranker reads the passage without its heading.** The embedder and
  the keyword index both get the heading; the reranker gets the chunk's
  text alone.
- **7% of the pairs it reads are cut short.** It reads 512 tokens of
  question plus passage; 126 of 1,880 pairs were longer.
- **It sees 20 of about 49 fused chunks.** For 1 question the right chunk
  was fused but below place 20.
- **The keyword index is read from disk on every search**, twice for a
  figure question. It costs about 23 ms each time.
- **The parts timed alone add up to less than the whole search.** About
  100 ms of the 139 ms without the reranker was not traced.
- **Search across satellites has only been tried with made-up second
  satellites.**
- From earlier chapters: contents pages in the store (2.6), and ChromaDB's
  search not being fully repeatable (2.6, 4.9).

### 5.11 Saying it in a minute

"Retrieval has to pick five passages out of about fifteen hundred. I run
two searches: one by meaning, using embeddings, and one by exact keywords,
using BM25. They fail in different places. Meaning search finds a
procedure when the question uses ordinary words; keyword search finds
exact codes and procedure numbers. Their scores cannot be compared, so I
merge them by position with Reciprocal Rank Fusion: a chunk found by both
beats a chunk that is top of one. Then a cross-encoder reranks the top
twenty by reading the question and each passage together. Fusion takes MRR
from about 0.82 for either search alone to 0.86, and reranking to 0.88.
The reranker is a trade: the first result is right more often, 84 percent
against 80, but the right passage is in the top five slightly less often.
A figure question triggers a second search over captions only. The whole
thing takes about a quarter of a second."


## Chapter 6: Routing a question

Written 17 November 2026. Code: `core/router.py` (318 lines). Measured on
the working store. Try it with:

```text
python cli.py query aditya-l1 "your question" --no-answer    the first line printed is the route
python cli.py query aditya-l1 "your question" --router llm   another router, for one run
python evaluate_router.py                                    how often each router is right
python evaluate.py --routers                                 what each does to retrieval
```

### 6.1 What routing is for

Questions are not all the same kind. "How do I switch on the computer"
wants the steps of a procedure. "Which procedure covers gyro power on"
wants one line of the index. "Show the propulsion schematic" wants a
figure. "What was sent on 9 June" wants rows of the command log, which is
a different store altogether.

Routing looks at the question once, before any search, and writes down
what it is asking for. That note is called the plan. Retrieval and answer
generation both read the same plan, so they cannot disagree about whether
a question wants a figure.

### 6.2 The plan

| Field | Meaning |
| --- | --- |
| `intent` | One of: `procedure`, `which_procedure`, `figure`, `command_history`, `concept` |
| `stores` | `knowledge`, `commands`, or both |
| `date`, `cid`, `mnemonic` | Exact values found in the question, used as filters on the command log |
| `router` | Which router was configured |
| `source` | Which one actually decided (the model router can fall back to the rules) |
| `note` | Why, when there is something to say |

Every routed question is appended to `logs/routing.jsonl` with its plan
and the ids of what was retrieved. The log holds the questions people
asked, so it is kept out of git.

### 6.3 The rules router

A fixed list of checks. The first one that matches decides.

1. **Command log?** A date in year-month-day form, or "CID" followed by an
   id, or a phrase such as "commands sent", "commands uplinked",
   "uplinked on", "command history", or "orbit" with a number. Intent
   `command_history`, store `commands`.
2. **Figure?** One of: diagram, figure, chart, plot, image, picture,
   schematic, flow chart, block diagram, visual.
3. **Which procedure?** "Which FCP", "what procedure", "FCP number",
   "list procedures".
4. **A procedure?** A long list of action words: procedure, how to, steps,
   switch, power, turn, enable, disable, reset, select, activate, operate,
   execute, configure, uplink, command.
5. **Otherwise** `concept`.

Dates, command ids and mnemonics are pulled out by pattern whichever
branch is taken. A mnemonic only becomes a filter if the command log
really holds it, so a false match costs nothing.

It takes under a millisecond.

### 6.4 The soft router (the default)

The soft router makes the same decisions as the rules about the command
log and figures. The difference is what happens to procedure questions.

- **Rules:** if the question looks like a procedure question, the two
  nearest chunks of the procedure document are forced to the front of the
  results.
- **Soft:** the question is never classified as procedure or not. The
  procedure document is searched for every question, and that list joins
  the fusion as a third list. A procedure chunk that is relevant rises on
  its own; nothing is forced.

So a wrong guess cannot push a wrong chunk to the top, because no guess is
made.

### 6.5 The model router

The local text model (`qwen2.5:7b-instruct`) is asked to choose the intent
and the stores.

- It gets a short description of the two stores and one made-up example
  per intent. None of the examples comes from a test set.
- Its reply is forced into a fixed JSON shape by the server, so it cannot
  answer with anything else.
- **Identifiers never come from the model.** Dates, command ids and
  mnemonics are still taken from the question by pattern. A date that is
  not written in the question cannot end up in a filter.
- A question with an explicit command id skips the model.
- If the model is not running, is slow, or replies off-shape, the rules
  decide and the plan says so.

After classifying, it behaves like the rules router: procedure chunks are
forced to the front for procedure questions.

### 6.6 How often each is right

Each test question carries a label saying what it asks for.

| Set | Router | Intent right | Store right | Time per question |
| --- | --- | --- | --- | --- |
| 100 benchmark questions | Rules | 80% | 100% | under 1 ms |
| 100 benchmark questions | Model | 93% | 99% | 0.8 s |
| 40 routing questions | Rules | 52.5% | 90% | under 1 ms |
| 40 routing questions | Model | 92.5% | 97.5% | 0.8 s |

The 40 routing questions were written after the rules, around their weak
spots, so they overstate how often the rules fail in ordinary use.

Where the rules go wrong on the benchmark, all 20 mistakes are one
confusion: 11 procedure questions called concept, and 9 concept questions
called procedure. Figures and command questions are always right. The
action-word list is the cause: "power budget" contains "power".

The soft router is not in this table. It makes the rules' decisions for
the command log and figures, and makes no procedure decision to score.

### 6.7 What each does to retrieval

Working store, 100 questions, top 20 scored:

| Router | Reranker | Hit@1 | Hit@5 | Hit@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| Rules | on | 80% | 94% | 96% | 0.855 |
| Soft | on | 84% | 92% | 97% | 0.879 |
| Model | on | 80% | 93% | 95% | 0.856 |
| Rules | off | 77% | 94% | 96% | 0.834 |
| Soft | off | 80% | 95% | 97% | 0.862 |
| Model | off | 78% | 93% | 95% | 0.840 |

| Difference in MRR, reranker on | Estimate | 95% interval |
| --- | --- | --- |
| Soft over rules | +0.024 | -0.010 to +0.061 |
| Model over rules | +0.001 | -0.040 to +0.041 |
| Model over soft | -0.023 | -0.067 to +0.019 |

By what the question asks for, reranker on, MRR:

| Question kind | Questions | Rules | Soft | Model |
| --- | --- | --- | --- | --- |
| Procedure | 50 | 0.913 | 0.911 | 0.880 |
| Concept | 34 | 0.769 | 0.843 | 0.822 |
| Figure | 8 | 0.875 | 0.875 | 0.875 |
| Command history | 6 | 0.783 | 0.783 | 0.783 |
| Which procedure | 2 | 1.000 | 1.000 | 1.000 |

### 6.8 What this shows

**Being right about the intent is not what helps.** The model router
classifies 93% of questions correctly against the rules' 80%, and its
retrieval score is the same as the rules' (0.856 against 0.855). This is
the first time the model router's effect on retrieval has been measured;
it was an open item.

**Forcing is what costs.** Both the rules and the model router force
procedure chunks to the front when they think the question is a procedure
question. The soft router forces nothing and scores highest. The gain is
all in concept questions (0.843 against 0.769), which is where a wrong
"procedure" call puts two unwanted chunks on top.

**The lead is not proven on today's store.** `PROJECT_CONTEXT.md` gives
soft over rules as +0.033 with an interval of +0.001 to +0.068, measured
on 12 November with the earlier captions and the top 10 scored. On the
working store with the top 20 scored it is +0.024 and the interval
includes zero. Soft is still the best estimate and costs nothing, so it
stays the default.

### 6.9 Mistakes made here, and their fixes

| Mistake | What it did | Fix |
| --- | --- | --- |
| "Show me" was a figure phrase | "Show me the steps to switch on the OBC" was answered from a caption, and the text model never saw the procedure | Removed from the figure pattern |
| A bare "command" sent a question to the log | "Programmable command sequencer" and "telecommand link test" are procedure questions; in the command store they had no answer | The wording must be about commands that were sent, a command id, or a date |
| Any "uplinked" sent a question to the log | One test question about a procedure could never be answered | Only "uplinked on / from / to" |
| The subsystem pattern for POWER matched a bare "power" | Every "power on" question was tagged POWER | Generic patterns moved last; and retrieval no longer filters on subsystem at all |
| Retrieval and generation each decided for themselves whether a question wanted a figure | They could disagree | One plan, made once, read by both |

### 6.10 Rough edges found while writing this

Measured on 17 November. None is fixed.

- **A date in words is not understood.** "Commands sent on 9 June 2026"
  goes to the log but no date is extracted, so the whole day is not
  returned; the log is searched by similarity instead. Only
  year-month-day is recognised.
- **Any date sends the question to the log alone.** A question about the
  mission report that mentions a date in that form is routed to
  `commands`. If the log returns anything, the documents are not searched.
- **A mnemonic on its own does not reach the log.** "Was XYZ_HEATER_ON
  sent" is routed to the documents as a concept question. The mnemonic is
  found but never used, because the wording does not match a command
  phrase.
- **"Power budget" is a procedure question to the rules.** The action-word
  list was left broad on purpose, so the other routers could be measured
  against it. The soft router is not affected.
- **The model router does not use `core/llm.py`.** It calls the Ollama
  package directly. So it ignores `SATRAG_LLM=vllm` and always needs
  Ollama, and it does not send the fixed seed the rest of the project
  uses.
- **Across satellites, the question is routed once**, with the first
  satellite's settings.
- **Only 2 test questions ask "which procedure".** That intent is barely
  tested.
- **The routing log has 12 entries.** Real questions from users would be
  the best test set, and almost none have been collected.

### 6.11 Saying it in a minute

"Before searching, a router reads the question once and writes a plan:
what it is asking for, which store has the answer, and any exact values
such as a date or a command id. Retrieval and answer generation both
follow that one plan. I built three routers. The rules router uses
patterns and is right 80 percent of the time. The model router asks the
local language model and is right 93 percent, at about a second per
question. The third, which I call soft, does not classify procedure
questions at all: it searches the procedure document for every question
and lets ranking decide. Soft scores highest on retrieval. And the
interesting result is that the model router, although far more accurate,
retrieves no better than the rules. What hurts is not a wrong
classification; it is forcing chunks to the top because of a
classification. So the default is the one that forces nothing."


## Chapter 7: Answer generation and the grounding guard

Written 17 November 2026. Code: `core/generate.py` (161 lines),
`core/grounding.py` (102) and `core/llm.py` (287). Try it with:

```text
python cli.py query aditya-l1 "your question"
python cli.py query aditya-l1 --file questions.txt
python cli.py shell aditya-l1
```

### 7.1 What this stage does

Retrieval hands over five passages. This stage turns them into a written
answer that cites where it came from. It has two jobs that pull against
each other: answer whenever the passages hold the answer, and say "not
found" whenever they do not. In operations the worst failure is the second
one going wrong in a particular way: an answer that reads like a procedure
and contains a command id that does not exist.

### 7.2 The path of one answer

This is `generate.ask`.

1. **Route** the question (chapter 6) and **retrieve** five passages
   (chapter 5). The same plan drives both.
2. **Figure shortcut.** If the question asks for a figure and the top
   passage is a caption, the answer is built from the caption directly:
   where the figure is from, the caption, and a pointer to the image file.
   The text model is not called. A text model asked to "show" a diagram
   tends to reply that it cannot, even though the figure was found.
   **Command-log shortcut** (added 17 November). If the passages are every
   record for a date, command id or mnemonic, the answer is counted from
   the records in code: how many, first and last time, status, stations,
   and each command with how often it was sent. No model call.
3. **Build the context.** Each passage is headed with its citation
   (document, procedure number, subsystem, page, heading). At most 2,000
   characters per passage and 16,000 in all.
4. **Add the context note** (section 7.5): one line, written by code,
   saying which procedures the passages hold steps for.
5. **Call the model** with the system prompt, the context, the question
   and the note.
6. **Run the guard** (section 7.6): any identifier in the answer that is
   not in the passages is removed.
7. **Check the no-steps sentence** (added 17 November): if it sits under
   an FCP whose step table is in the passages, it is wrong and is removed.
8. **Check citations** (added 17 November): a citation to a source number
   that was never given is removed; an answer that states things and cites
   nothing gets a line telling the reader to check it.
9. **Log** the route and the sources.

If the model call fails, the reply says generation is unavailable and the
sources are still listed. Retrieval is not lost to a model error. If the
answer is the not-found sentence and nothing else, the command line does
not print a list of sources under it.

### 7.3 The model and its settings

| Setting | Value | Why |
| --- | --- | --- |
| Model | `qwen2.5:7b-instruct` | A text model, not the vision model: it lists steps better and is steadier on the GPU |
| Context window | 8,192 tokens | The largest prompt measured was 4,098 |
| Answer limit | 768 tokens | Without a limit, greedy decoding sometimes loops on a numbered list until the window is full |
| Temperature | 0 | The same question gives the same answer |
| Seed | 1234, fixed | Same reason |
| Repetition penalty | 1.0, meaning none | Sent explicitly, so no server applies its own |

Every model call in the project goes through `core/llm.py`. It talks to
Ollama by default and to a vLLM server when `SATRAG_LLM=vllm` is set. The
sampling settings are sent on every request to either server. That was
learned the hard way: vLLM applies the model's own default penalty of 1.1
unless told otherwise, so the same weights gave different answers on the
two servers for a reason that had nothing to do with the server.

### 7.4 What the prompt says

The system prompt tells the model it is an operations assistant and must
answer only from the context. Its rules, in short:

- For a procedure, give the FCP number and list the steps, with command
  mnemonics and ids exactly as written.
- Copy every mnemonic, id and value from the context. If it cannot be
  copied from the context, leave it out.
- After each statement, give the source in square brackets, such as
  `[Source 2]` (version 7, added 17 November).
- For a figure, name the document and the page or section.
- Facts from several passages may be combined, as long as each one is
  stated in the context. Combining stated facts is not guessing.
- Say "Not found in available documents." only for the part of the
  question the context does not address.
- Do not speculate. Be concise.

### 7.5 The context note

The procedure document here is a sample: 6 procedures have their steps, and
about 220 exist only as one line of the index. So for most procedure
questions the honest answer is "it is this FCP, and its steps are not in
the documents".

The model could not be relied on to work that out. Prompts that asked it
to tell a step table from an index line either refused questions that
could be answered, or said a table that was there was not (versions 3 to
5 in section 7.7).

So the code decides. `grounding.context_note` reads the passages:

- If it finds a procedure heading and a `PROCEDURE:` section, that FCP has
  a step table. The note says: list the steps from the table.
- Other FCPs that appear only as index lines get: reply with the index
  line, followed by a fixed sentence saying the steps are not in the
  available documents. Do not write steps for it.

The note goes after the question. Whether a table is present is a fact
that can be read off the text, so it is read off the text.

### 7.6 The grounding guard

A prompt can make an invented id rarer. It cannot rule one out. The guard
can, for the identifiers it recognises.

`grounding.guard` looks in the answer for anything shaped like a command
or telemetry id: three or more capital letters with at least three digits,
a hex value such as `0x1F`, or any value written straight after "CID".
Each one is looked up in the passages the model was given. One that is not
there cannot have come from them. It is replaced with
`[ID not in documents]`, and a note is added saying how many were removed
and not to act on the steps that carried them.

What it does not do:

- It does not catch a made-up name that has no digits in it.
- It does not check numbers such as limits, voltages or times.
- It does not check that a real id is used in the right step.

It is a floor under the prompt, not a measure of whether the answer is
faithful.

### 7.7 The six prompt versions

All six were run on the lab GPU on 31 October: the 7B model at full
precision, 100 questions, the same saved passages each time.

| Version | What changed | Refused | Real answer that still ends in the not-found line | Answers with an invented id | Said "no steps" when the table was there |
| --- | --- | --- | --- | --- | --- |
| v1 | The original | 20 | 9 | 4 | 0 |
| v2 | The revised system prompt of late October | 18 | 11 | 3 | 0 |
| v3 | The version recorded as "v3-template" | 26 | 4 | 1 | 2 |
| v4 | v1 plus two rules: the exact reply for an index-only FCP, and copy ids or leave them out | 10 | 2 | 1 | 3 |
| v5 | v4 with the order spelled out: look for the table first | 28 | 6 | 0 | 2 |
| v6, in use | v1 plus the copy rule; the code writes the context note | 11 | 6 | 0 | 2 |

Versions 2 and 5 each lost 3 requests, so they are out of 97.

Version 6 was chosen because it has no invented ids and nearly the fewest
refusals. Of the 56 questions whose passages held only index lines, it
named the right kind of answer (the FCP) for 44, the most of any version.
The answers also became much shorter: a median of 414 characters with v1
and 131 with v6.

### 7.8 Numbers today

Run on 17 November: every test question through the app as it stands, on
the laptop (the 7B model at 4 bits, partly on CPU), with the working
store. This differs from the table above in the model's precision, the
passages and the router, so the two are not a like-for-like comparison.

| Item | Value |
| --- | --- |
| Questions | 100, plus 2 that have no answer in the documents |
| Answered from a caption, no model call | 8 (the caption was a right one for 7) |
| Answered by the model | 91 |
| Request failed | 1 (the very first, while the model was loading; it worked when repeated) |
| Refused | 6 of 91 |
| Real answer that still ends in the not-found line | 5 |
| Answers carrying a citation | 51 of 91 |
| Answers where the guard removed an id | 0 |
| Answers naming an FCP that was not in the passages | 0 |
| Answers that ran to the 768-token limit | 1 |
| Both no-answer questions refused | 2 of 2 |

| Size and time | Median | Largest |
| --- | --- | --- |
| Context, characters | 2,938 | 16,068 |
| Prompt, tokens | 1,043 | 4,098 |
| Answer, characters | 152 | |
| Search time | 0.16 s | |
| Answer time | 5.2 s | 48.9 s |

One answer in ten took over 17.7 seconds. The slowest was a command-log
question that returned a whole day of records.

What the context note said, and what the model then did:

| Passages held | Questions | Answer said "steps not in the documents" | Refused |
| --- | --- | --- | --- |
| A step table for some FCP | 8 | 3 | 1 |
| Index lines only | 47 | 42 | 2 |
| No procedure at all | 36 | 0 | 3 |

So 45 of 91 answers are "this FCP; its steps are not in the documents".
That is the correct answer for a document that holds 6 procedures out of
225, and it will change when the full procedure document is loaded.

Refusals against retrieval:

| The five passages | Questions | Refused |
| --- | --- | --- |
| Held a right chunk | 85 | 4 |
| Did not | 6 | 2 |

### 7.9 Mistakes made here, and their fixes

| Mistake | What it did | Fix |
| --- | --- | --- |
| Answer generation was dropped in the rebuild | Every query returned a list of sources and no answer | Restored in `core/generate.py` |
| Figure questions went to the text model | It replied that it could not show a figure | Answered from the caption |
| No limit on answer length | 14 of 100 benchmark answers looped up to the 512-token cap of that test | A 768-token limit |
| Sampling left to each server's defaults | vLLM applied a penalty of 1.1 and Ollama none, so answers differed | Settings sent explicitly to both |
| Refusals counted loosely | Any answer containing the not-found sentence counted: 29 of 100, where 20 were refusals | A refusal is the sentence and nothing else |
| Sources hidden whenever the answer contained the not-found sentence | A real answer that ended with it lost its sources | Hidden only for a pure refusal |
| The model asked to judge whether a step table was present | It refused answerable questions or denied a table that was there | The code decides (section 7.5) |

### 7.10 Rough edges, and what was done about them

Section 7.8 was measured on the morning of 17 November and listed eight
rough edges. All eight were worked on the same day. The full set of
questions was then answered twice, with prompt v6 and with the new v7,
and the saved answers were counted again after the last code step was
added.

**What changed, item by item**

| Rough edge | What was done | State |
| --- | --- | --- |
| The demo question got "steps not in the documents" | Retrieval: a procedure is one candidate, not up to four (section 7.14) | Fixed |
| Steps often did not come out | The procedure goes to the model whole; a wrong no-steps sentence is removed in code | Partly: the steps come out, but as copied table fragments |
| 4 refusals with a right passage (5, once counted right) | `review_answers.py --only refused` puts each beside its passages | 1 read: over-refused; 6 to read |
| The guard never fired | 5 checks in `tests/test_answering.py` hand it made-up wrong ids | Proved |
| Nothing measures whether an answer is right | `evaluate_answers.py` counts; `review_answers.py` builds a page to mark | 30 read (section 7.12) |
| Citations not enforced | Prompt v7 asks for `[Source N]`; code checks them | Fixed |
| The first request could fail | One more try after 3 seconds, only for a busy or unreachable server | Fixed |
| A day of commands filled the context | Counted in code, no model call | Fixed |

**Before and after**

Before is the run of section 7.8. After is prompt v7 with every fix.

| Measure | Before | After |
| --- | --- | --- |
| Requests that failed | 1 | 0 |
| Command-log lookups answered by the model | 4 | 0 (counted in code) |
| Slowest command-log answer | 48.9 s | 0.0 s |
| Questions whose procedure has steps in the store, with that table in the passages | 3 of 4 | 4 of 4 |
| of which the answer copies command ids from that table | not measured | 4 of 4 |
| Answers saying "steps not in the documents" under a procedure whose table is there | 1 (by the older check) | 0 |
| Answers naming the wrong procedure, of 23 with one right FCP | not measured | 0 |
| Refused | 6 of 91 | 7 of 88 |
| Answers with a checked `[Source N]` citation | not asked for | 68 of 88 |
| Answers flagged as citing nothing | | 2 |
| Identifiers removed by the guard | 0 | 0 |
| Out-of-scope questions refused | 2 of 2 | 2 of 2 |

"Before" counted an answer as cited if it named a document, page or FCP
anywhere (51 of 91). "After" counts only `[Source N]` that points at a
passage the model was given. The two are different measures. Of the 88
answers by the model, 68 carry a checked citation, 2 are flagged, and the
other 18 are refusals or the fixed no-steps reply, which assert nothing to
cite.

**Prompt v6 against v7**, both with the retrieval and code fixes:

| Measure | v6 | v7 |
| --- | --- | --- |
| Refused | 7 | 7 |
| Refused with a right passage | 5 | 5 |
| Real answer that still ends in the not-found line | 5 | 7 |
| Named the wrong procedure | 1 | 0 |
| No-steps sentence under a listed procedure, before the code step | 2 | 1 |
| `[Source N]` citations | 8 | 68 |

v7 was the one in use until v9 (section 7.13). It is worse on one count: more answers end with the
not-found sentence after a real answer. (These v7 counts were corrected
on 17 November; section 7.12 says why.)

**Retrieval** with the procedure change, 100 questions, top 20 scored,
same store: MRR 0.880 before and 0.880 after; procedure questions 0.912
and 0.913. The demo question's procedure moved from 5th to 4th, which is
inside the five passages.

### 7.11 Still open

- **The steps come out as fragments.** The step tables are stored one
  cell per line, with the column headings as separate lines (5201 has 203
  lines, median 11 characters). The model copies them faithfully, command
  ids included, but not as readable numbered steps. Rebuilding the rows
  from the PDF was tried and rejected: the headings read back wrong and
  cells landed in the wrong columns, and a rebuilt table that puts an id
  against the wrong step is worse than fragments. A clean copy of the
  procedure document, as Word or text, is the fix.
- **Answers gave too little.** The first reading (section 7.12) marked 7
  of 30 over-refused and 7 partly right. v9's 30 have been read (section
  7.13): 16 right, 3 over-refused, but conditions and cautions are still
  left out, and `obc_on` and `pcs` got worse. The next prompt change is
  narrower: the caution must come with the steps, and another FCP's steps
  must never be given under the procedure asked for.
- **5 refusals had a right passage**, 1 read. Run
  `python review_answers.py v7-with-no-steps-check --only refused`.
- **The 2 figure answers were read against their images** (section
  7.13): the MRS diagram is right; the thruster answer returned the wrong
  table because its caption is wrong.
- **7 answers end with the not-found sentence** after a real answer, up
  from 5 with v6.
- **The guard has still never acted on a real answer.** That is the prompt
  working, and the tests show the guard would act.
- **The slowest answer is a long procedure**, 49 seconds.

### 7.12 The first reading of the answers

On 17 November 30 answers from the v7 run were read against the passages
each was given, and marked on the review page (seed 1, the same 30 for
any run). This is the first measure of whether answers are right.

| Mark | Answers | By the model | From a caption | From the command log |
| --- | --- | --- | --- | --- |
| right | 13 | 12 | | 1 |
| partly | 7 | 6 | 1 | |
| wrong | 3 | 2 | 1 | |
| over-refused | 7 | 7 | | |

**What the marks say**

- **Over-refused is the largest fault, and the code did not see it.** Of
  the 7, 6 are the fixed reply for a procedure that is only an index row
  ("the steps of this procedure are not in the available documents"). The
  code counts that reply as correct, and on the steps it is. But the
  reader found that the passages held related material from the other
  documents (the operation the procedure carries out, or the commands
  for it in another document) and the reply gave none of it. 5 other
  replies of the same kind were marked right, where there was nothing
  else to give. The 7th is a plain refusal with the procedure's table in
  the passages.
- **Procedures with steps: the commands are right, the conditions are
  left out.** All 5 answers that copied a step table had the command
  sequence right. All 5 were marked down, because they left out when the
  procedure applies, what must be on first, and the cautions and notes
  around the steps. One also added part of a different procedure.
- **Statements not in the passages: 3 of 28.** One turned a statement
  with two cases into one case. One completed a sentence that the passage
  cut off at a chunk boundary. One added the not-found sentence after a
  full answer. None was an invented command id.
- **The 2 figure answers could not be judged.** The page showed only the
  source labels for an answer built from a caption, not the caption, so
  the reader had nothing to check against. Fixed: a caption answer now
  keeps the caption and its image file as its passages, the 8 saved
  figure answers were filled in after checking each caption matched the
  one the answer quoted, and `--only figures` builds a page for them. The
  2 marks stand until they are read again against the image.

**A counting error found on the way.** One v7 refusal was the not-found
sentence followed by three citations, and the check took the citations
for an answer. With that fixed, v7 refused 7, not 6, and 5 of those had
a right passage, not 4; answers with the not-found sentence tacked on are
7, not 8. The app now drops citations from a bare refusal. v6 is
unchanged.

**What to do next, in order**

1. When a procedure is only an index row, still give what the passages say
   about it, cited, after the fixed line. (Done as v9, section 7.13.)
2. For a procedure with steps, ask for the conditions, prerequisites and
   cautions with the steps. (Done as v9.)
3. Read the 2 figure answers against their images (done in 7.13), and
   the 6 refusals not yet read.

### 7.13 Prompts v8 and v9: acting on the reading

The reading in 7.12 gave two changes to make. Each was tried on all 102
questions, the same way as v6 and v7.

**v8** added one rule, "with the steps of a procedure, also give when it
is used, what must be on first, and its cautions", and made the code's
note ask, for an index-only procedure, for what other passages say about
it. Refusals fell from 7 to 2, but two faults came in. On the demo
question the model rewrote 5201's steps in its own words and copied none
of its 15 command ids. And twice it gave another procedure's step table
as the steps asked for. v8 was not kept.

**v9** keeps the direction and tightens the wording:

- "first copy its steps from the table, with every command mnemonic and
  CID exactly as written; then add when it is used, what must be on or
  done first, and its cautions and notes";
- the note now says to use only a passage that is not a procedure table
  and names that FCP or the operation it carries out, and to never give
  the steps of a different FCP, naming the FCPs whose tables are present;
- "do not end an answer with the not-found sentence when you have
  answered". This did not work on its own (14 answers still did), so a
  code step now takes off the not-found sentence when it is the last
  paragraph alone after an answer (`generate.drop_trailing_not_found`). A
  sentence that names what is missing, and a bare refusal, are kept.

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

v9 is the prompt in use. v7 and v8 are kept word for word, and
`evaluate_answers.py --prompt v7` reproduces v7 with its note.

**What the counts cannot say.** v9 answers are three times as long, and
the extra is exactly what 7.12 asked for: what other passages say, and a
procedure's conditions. Whether that extra is supported has to be read.
One v9 answer (`eeprom_mgmt`, not in the 30 below) lists two other
procedures' tables under their own numbers for a question about an
index-only one; it labels them, but it is the kind of answer the note
forbids.

**Reading v9's 30 answers.** On 17 November, with the documents cleared
as dummy data, Claude read the same 30 questions' v9 answers against
their passages, with the page's marking guide. The v7 marks are from the
first reading (7.12), by a different reader, so the table compares two
readers as well as two prompts. Marks and a note on each are in
`runs/answers/answer_review_v9-steps-first-repaired_claude.json`.

| Mark | v7 (first reading) | v9 |
| --- | --- | --- |
| Right | 13 | 16 |
| Partly | 7 | 8 |
| Wrong | 3 | 3 |
| Over-refused | 7 | 3 |

To separate the prompt from the reader, every answer whose mark changed
was compared with its v7 text. Two answers are word for word the same in
both runs and changed mark only with the reader (one of them a figure
answer whose caption the first page did not show), and three more carry
the same faults in both. That leaves the changes v9 made:

- **7 better.** Four of the over-refusals now give what the passages say
  (`gyro_cal`, `auto_firing`, the 21 ODHS commands for `odhs_record`;
  `obc_scrub` now names the related procedures instead of a bare
  refusal). The tacked-on not-found line is gone from `orbit_det`;
  `suit_desc` no longer makes up the content of the filter slots; `orbit`
  states the class that was proposed.
- **4 worse, two of them badly.** For `obc_on`, v9 copies OBC-1's
  switch-on steps and then gives FCP 5203's *system reset* steps under
  the heading "For OBC-2", citing the wrong sources; v7 at least labelled
  them as the reset. For `pcs`, v9 says the command sequencer handles
  battery thresholds, citing a passage that never mentions it. The two
  small ones: `moi` now calls the inverse-inertia values "cross
  products", and `transfer_traj` repeats a caption's wrong expansion of
  EPO.

Three faults remain across v9:

1. **Conditions are still left out.** The v9 rule asks for when a
   procedure is used, what must be done first, and its cautions. Of the
   five answers that copy a step table, one added the applicability and
   none the caution: the demo question (`obc_reset`) still drops "not a
   nominal operation, execute in consultation with the subsystem". The
   steps and command ids were exact every time.
2. **Three index-only procedures are still over-refused** (`tm_prog`,
   `latch_valve`, `rw_logics`): the passages describe the operation, but
   the answer stops at the no-steps line.
3. **Figure captions leak into text answers.** Three partly-right
   answers repeat a claim that exists only in a caption: "L1 is
   highlighted", which the caption reading (3.13) found false; EPO as
   "Earth-Proximity Orbit", where the text says Earth Parking Orbit; and
   a sun sensor's "redundancy for critical operations". And one figure answer
   is wrong because a caption is: asked for the thruster direction
   cosines, the app returned a force-and-torque table whose caption says
   it shows direction cosines (`thruster_cosines`); the right table is
   another image in the same section.

No answer invented a command id, in either run.

### 7.14 The procedure change in retrieval

Each procedure is in the store as one index row and, for the 6 with steps,
one to three pieces of its step table. The pieces are long and full of
codes, so they scored well for any question about their subsystem, and two
or three procedures' bodies often took places in the top five meant for
the procedure asked about.

Now:

1. After fusion, every piece of a procedure is replaced by its index row,
   once, so a procedure is one candidate.
2. The reranker reads both the row and the whole procedure, and the
   procedure keeps the better place of the two.
3. After ranking, the top two rows that have steps in the store are
   swapped for the whole procedure, joined from its pieces in order.
4. A "which FCP" question keeps the index row; it asks for the line.
5. A procedure is given to the model up to 6,000 characters, not cut at
   the 2,000 used for prose, so its last steps and notes are not lost.

### 7.15 Saying it in a minute

"The last stage writes the answer. A local 7B model gets the five
retrieved passages, each headed with its source, and is told to answer
only from them and to say 'not found' otherwise. Sampling is fixed, so the
same question gives the same answer. Two things are done in code, not left
to the model. First, the procedure document I have is a sample with steps
for only six procedures, so the code checks whether the passages actually
contain a step table and tells the model; before that, the model either
refused or wrote steps that were in no document. Second, a guard checks
every command id in the answer against the passages and removes any that
is not there, because an invented command is the most dangerous failure.
I tried six prompt versions and chose the one with no invented ids and
about half the refusals of the original, then added one rule: cite the
source number after each statement, which the code checks. Reading 30
answers showed they said too little, so v9 copies a procedure's steps
and then its conditions, and gives what other passages say about a
procedure whose steps are missing; refusals fell from 7 to 2. Figure
questions and command-log lookups skip the model: one is answered from the
caption, the other is counted from the records. Reading the same 30
answers again under v9, 16 were right, against 13, and none invented a
command; what is still missing is a procedure's cautions, and two answers
mixed in the wrong passage, which is the next thing to fix."


## Chapter 8: Evaluation and the November audit

Written 17 November 2026. Code: `testset.py` (251 lines), `evaluate.py`
(377), `collect_results.py` (212), `compare_models.py` (201),
`compare_rerankers.py` (267), `evaluate_router.py` (149),
`evaluate_answers.py` (282), `review_answers.py` (191). Try it with:

```text
python evaluate.py                    the current settings, 100 questions
python evaluate.py --compare          with and without the reranker
python evaluate.py --routers          rules against soft
python collect_results.py             every result file in one table, with intervals
python evaluate_answers.py            every question through the app, answers counted
```

### 8.1 What this stage is for

Every chapter so far ends with a number: a parser that was "level", a
router that "scores best", a prompt that "refuses less". This chapter is
where those numbers come from, and how far they can be trusted. It has
three layers, each checking something the one before cannot:

| Layer | Asks | Tool | Who decides |
| --- | --- | --- | --- |
| Retrieval | Is a right passage among the ones found, and how high? | `evaluate.py` | Code, against labels |
| Answers, counted | Did it refuse, copy the right ids, cite, name the right procedure? | `evaluate_answers.py` | Code, against the passages |
| Answers, read | Is what it says right? | `review_answers.py` | A person |

Routing has its own accuracy check (`evaluate_router.py`, chapter 6), and
the parser and captions had theirs (chapters 2 and 3).

### 8.2 The test set

`testset.py` holds 100 questions written against the Aditya-L1 documents,
plus 2 out-of-scope questions with no answer in them.

| Kind (as labelled) | Questions |
| --- | --- |
| procedure | 50 |
| concept | 34 |
| figure | 8 |
| command history | 6 |
| which procedure | 2 |

Each question carries `accept` conditions: what a passage must hold to
count as a right one. A passage is right if it meets any one condition,
and within a condition every key must match.

| What the condition checks | Questions |
| --- | --- |
| A distinctive phrase in the text (any of several) | 66 |
| The FCP number, with a phrase as a fallback | 13 |
| The FCP number alone | 10 |
| The kind of passage (a caption, say) and a phrase | 7 |
| The kind of passage and the subsystem | 2 |
| The document type and a phrase | 1 |
| A date in the command log | 1 |

87 questions have one condition, 13 have two.

Three rules came out of getting this wrong:

- **A condition must name the thing asked for, not the topic.** An early
  condition accepted any passage mentioning EEPROMs, and scored a hit
  when a question about one procedure found a different procedure whose
  notes mentioned an EEPROM setting. A condition that any passage on the
  subject meets measures nothing.
- **Figure questions need the caption to name the thing.** "Any caption
  from this report" would be met by 200 passages.
- **FCP numbers carry a text fallback**, so the labels hold if
  extraction changes and a number moves.

Two honest weaknesses are written into the file rather than hidden:

- **One question cannot be answered on this build.** No caption in the
  store names the subsystem `odhs_diag` asks about. It stays in, so the
  ceiling is 99, not 100.
- **Eleven questions are broad**: a large share of the store meets their
  condition (the three command-log questions are met by 201 to 203 of 203
  records; the others by 23 to 62 passages). They are listed in
  `testset.BROAD`, and every report gives MRR with and without them.

There is a second, separate test set (`testset_public_demo.py`, 32
questions) for the public booklets used in demos, run by
`evaluate_public_demo.py`. The confidential set's procedure and command
questions mean nothing on the public files, so the two are never mixed.

### 8.3 The measures

For each question, `evaluate.py` runs the real search with the
satellite's settings and records the rank of the first right passage (0
if none in the pool).

| Measure | Meaning |
| --- | --- |
| Hit@k | Share of questions with a right passage in the top k (1, 3, 5, 10) |
| MRR | Mean of 1/rank, 0 if not found. Rank 1 scores 1, rank 2 scores 0.5 |
| MRR strict | MRR over the 89 questions outside `BROAD` |
| Recall | Found anywhere in the pool of 20: the ceiling any reranker works within |

Hit@5 matters most for answers, because the top five passages are what
the model sees. Hit@1 matters for the first citation. MRR sums up both
in one number, and is what comparisons are made on.

The out-of-scope questions are not scored by rank, because there is no
right passage. They are checked at the answer stage: both were refused
in every one of the ten answer runs on 17 November.

**Today's figures**, 17 November, working store (new captions and the
procedure change of chapter 7), default settings:

| Measure | Value |
| --- | --- |
| Hit@1 | 84% |
| Hit@3 | 91% |
| Hit@5 | 93% |
| Hit@10 | 97% |
| MRR | 0.880 |
| MRR strict | 0.866 |
| Recall in the pool of 20 | 98% |

| Kind | Questions | Hit@1 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| procedure | 50 | 88% | 96% | 0.913 |
| concept | 34 | 79% | 88% | 0.843 |
| figure | 8 | 88% | 88% | 0.875 |
| command history | 6 | 67% | 100% | 0.783 |
| which procedure | 2 | 100% | 100% | 1.000 |

84 questions find a right passage first. Of the other 16, 13 find one
by rank 8, one at rank 14, and two not at all: `odhs_diag`, which cannot
be found, and one concept question whose passage never reaches the pool.

### 8.4 Repeat runs, and what is actually uncertain

Retrieval is deterministic on a given store. The evaluation was run
twice today in separate processes, and all 100 questions came back at
the same rank. The same was found in November with four runs. So there
is no run-to-run noise to average away.

The uncertainty is in the sample. A hundred questions is a small set:
had a different hundred been asked, the score would be different. That
is measured by resampling the questions (bootstrap, 10,000 draws).
Today's MRR of 0.880 has a 95% interval of **0.822 to 0.932**.

To compare two settings, the same idea is applied to the difference,
question by question (paired bootstrap, `collect_results.interval`). A
difference counts as **shown** only if its interval does not include
zero. From `results/intervals.csv`, measured 12 November:

| Difference in MRR | Estimate | 95% interval | Shown? |
| --- | --- | --- | --- |
| Hybrid over dense (MiniLM) | +0.060 | +0.014 to +0.108 | yes |
| Reranker L-6 over none | +0.032 | -0.015 to +0.079 | no |
| Reranker L-4 over none | +0.033 | -0.018 to +0.084 | no |
| Best embedder and reranker over the deployed pair | +0.003 | -0.028 to +0.036 | no |
| bge-large over MiniLM, dense | +0.025 | -0.030 to +0.080 | no |
| MiniLM over bge-m3, dense | +0.055 | -0.020 to +0.132 | no |
| bge-large over e5-base, dense | +0.161 | +0.082 to +0.243 | yes |
| Soft router over rules, reranker on | +0.033 | +0.001 to +0.068 | yes, just |

So this test set can tell a good embedder from a poor one and hybrid from
dense, and it just shows the soft router over the rules. It cannot rank
the good embedders against each other, and it cannot confirm the
reranker, even though the reranker lifts Hit@1 from 77% to 84%. Settling
those needs more questions, not more runs.

This is also why every "level" in earlier chapters is called level, not
"slightly better": the structure parser (0.881 against 0.873), the new
captions (0.879 against 0.883) and the procedure change (0.880 against
0.880) all sit well inside an interval this wide.

### 8.5 The audit, 12 November

**What we had.** Headline figures from August: dense 0.774, hybrid
0.844, reranked 0.863, and the claim that "the keyword pass is worth 3.7
times the reranker". They were in the README, the comparison document,
and the slides.

**What prompted it.** After the routing work in October, the rules
router was scoring lower than it had in August, on the same questions.
Something in between had changed what was being measured.

**What it found.** One root cause and several smaller errors.

The root cause: **the procedure prepend had never run.** Retrieval does
a second search restricted to the procedure document, and puts the
closest procedure passages first (chapter 5). The August code filtered
that search on a metadata field called `source_doc`. Ingestion has
always written `doc_type`. The filter matched nothing, so the step did
nothing, silently, and every August number was measured without it.
Switching the step off today reproduces the August file rank for rank.
The field name was corrected with the routing work on 28 October; the
step started running, and under the rules router it cost about 0.03 MRR.
That is why the rules router appeared to get worse, and why the soft
router, which does not force procedure passages first, became the
default.

The smaller errors:

| Area | What was wrong | What was done |
| --- | --- | --- |
| Reranker table | Retrieved a pool once and reranked the saved list last. The pipeline reranks first, then applies the figure and procedure rules | Run again through the real pipeline |
| Hit@10, all 30 reranker rows | Held "found anywhere in the top 20" | Corrected; true Hit@10 was 1 to 6 points lower |
| Dense against hybrid | Dense scored on the top 10, hybrid on the top 20, so part of the recall credited to BM25 was the deeper pool | Both on the top 20. BM25's recall gain fell from 3 to 10 points to 1 to 4 |
| Command log | Every row ends in a pipe, and columns counted from the end were one place out: the destination station came out blank and the status held the source. A header repeated every 30 rows was read into six records | Parser fixed; all 203 records clean; the command collection rebuilt in all 11 stores. The one question about the uplink station, which could never match, now does |
| Command search | Ignored the dense / BM25 mode switch | Fixed |
| Broad questions | 11 questions with many right passages lifted MRR | `MRR strict` added |
| Differences | Quoted with no measure of how sure | Paired bootstrap intervals, `results/intervals.csv` |
| "Noise floor" | 0.005 MRR, from two runs that disagreed once | Never reproduced, and not random; removed |
| Refusal counts | An older, looser check gave 29 of 100 | Recounted from saved answers with the strict rule: 20 of 100, on both runtimes |
| Captioning speed-up | Described as against "Ollama, 4 in parallel" | It was 2 slots and 4 clients; wording corrected, not re-tested |
| LoRA split | Train and evaluation pairs could share a source passage | Split by passage (595 train, 101 eval, 0 shared), retrained (chapter 10) |

The August files were moved to `results/archive-2026-08/` (45 files,
kept locally, not in git), and the full sweep was run again: 40 result files and the comparison
tables.

**What came out.**

| | August | 12 November |
| --- | --- | --- |
| Dense only | 0.774 | 0.792 |
| Hybrid | 0.844 | 0.852 |
| Hybrid and reranker | 0.863 | 0.884 |
| Default router | rules | soft |
| Hybrid's gain, shown? | not tested | yes |
| Reranker's gain, shown? | not tested | no |

The outline of the August work held: hybrid beats dense, the small
ms-marco rerankers beat the large ones, and the embedders near the top
cannot be told apart. The sizes of the gains changed, and "3.7 times"
was dropped.

### 8.6 Measuring answers

Retrieval scores say nothing about the answer the user reads. Until 17
November nothing measured answers except a count of refusals. Chapter 7
tells that story. In brief:

- `evaluate_answers.py` runs every question through the app and counts
  what code can judge against the passages: refusals, refusals with a
  right passage, a not-found line tacked onto a real answer, the right
  or a wrong procedure named, command ids copied from the right table,
  `[Source N]` citations checked, ids removed by the guard, and timing.
  It can run any saved prompt version, and `--rescore` counts a saved run
  again when a check changes, without asking the model again.
- `review_answers.py` turns a run into a page: each question, the answer,
  the exact passages the model was given, and marks (right, partly,
  wrong, over-refused). The same 30 questions are drawn for every run,
  so two prompts can be read side by side.
- `tests/test_answering.py` holds 37 checks on made-up data for every
  part of answering that code decides.

The first reading found two faults in the counting, which is the point of
reading: the page did not show figure answers' captions, and a refusal
followed by citations was counted as an answer. Both are fixed.

### 8.7 Mistakes made here, and their fixes

- **A filter on a field that does not exist** made a whole stage a no-op
  for two months, with no error. The lesson: a step that changes nothing
  when switched off is a step to check, not a step that is harmless.
- **Comparing two things measured differently** (top 10 against top 20)
  credited the deeper pool to BM25. Both stages are now scored on the
  same pool.
- **A shortcut harness** (rerank a saved list) measured a pipeline that
  does not exist. Every comparison now calls the real search.
- **A "noise floor" from two runs** gave false confidence about small
  gaps. Replaced by intervals over questions.
- **Labels that any passage on the topic meets** scored hits for the
  wrong procedure. Labels now name the thing asked for.

### 8.8 Rough edges found while writing this

- **A hundred questions is too few** to confirm the reranker or rank the
  good embedders. More questions, especially concept questions (the
  weakest kind at 0.843), would settle more than any other change.
- **The test set was written by the people who built the system.** The
  routing set (chapter 6) shows the risk: written around the rules'
  weak spots, it overstates how often they fail. Questions from an
  operator would be a fairer test.
- **The headline in the README is stale.** It still quotes 12 November
  (Hit@5 93%, MRR 0.883). Today's is 0.880, inside the interval; the
  per-model stores and sweep tables were built with the old captions and
  have not been run again.
- **The contents pages are still in the working store**, and they lift
  BM25 scores for some questions (chapter 2).
- **Answer reading has covered the same 30 questions in two runs**, read
  by two different readers. The other 72 answers of each run have not
  been read.

### 8.9 Saying it in a minute

"I evaluate at three levels. Retrieval: a hundred questions written
against the documents, each with a condition saying what a right passage
must contain, scored by Hit@k and MRR, with a strict MRR that leaves out
the eleven questions too many passages satisfy. Today it finds a right
passage first for 84 and in the top five for 93. Runs repeat exactly, so
the uncertainty is in the sample, and I resample the questions to put an
interval on every difference: hybrid search over dense is shown, the
reranker is not, even though it lifts the first hit from 77 to 84. In
November I audited every number and found that a filter on a field that
didn't exist had switched off a whole retrieval step since August, so I
re-measured everything and changed the default router. For answers, a
script counts what code can check against the passages, and a review
page lets a person mark what code can't; the first reading found 13 of
30 right, no invented commands, and two errors in my own counting."


## Chapter 9: Ollama against vLLM

Written 17 November 2026. Code: `core/llm.py` (313 lines), and the
benchmark in `bench/`: `harness.py` (607), `backends.py` (372),
`report.py` (297), `run.py` (229), `caption_bench.py` (184),
`check_answers.py` (128). Results in `bench/results*/`. Try it with:

```text
set SATRAG_LLM=vllm                  the app, answered by a vLLM server
set SATRAG_VLLM_URL=http://127.0.0.1:8000/v1
python bench/run.py cache aditya-l1  retrieve once, freeze the passages
python bench/run.py all ollama       every scenario against Ollama
python bench/run.py all vllm         the same against vLLM
python bench/report.py               the comparison table
```

### 9.1 The question

The app runs its models through Ollama. The question put to the project
was whether vLLM, the server built for high-throughput serving, would do
better. "Better" turned out to be several questions, and the answer is
different for each.

**Ollama** wraps llama.cpp. It is one install on Windows, loads a model
on first request, and is built for one person at a time; it can serve
several requests at once only if started with more "parallel slots",
each of which reserves its own context memory.

**vLLM** is a Python server for Linux. Its two main ideas are paged
attention, which stores the model's working memory in small blocks so
many requests fit at once, and continuous batching, which adds a new
request to the running batch instead of waiting for it to finish. It
also caches the work done on a shared prompt prefix.

The expectation going in: no difference for one question at a time, a
large one for many.

### 9.2 One switch in the app

Before the benchmark, the app talked to Ollama in three places
(answering, captioning, building the LoRA dataset), each with its own
options. All three now call `llm.chat()` in `core/llm.py`, and the server
is one environment variable: `SATRAG_LLM=ollama` (the default) or
`SATRAG_LLM=vllm`. It uses the standard library only, over HTTP, so
neither server's Python package needs to be installed where the app
runs. That matters because vLLM runs only under Linux, in its own
environment with its own pinned PyTorch.

Sampling is sent on every request, never left to the server:
temperature 0, seed 1234, repetition penalty 1.0. Section 9.9 says why.

### 9.3 Five questions, not one

| Scenario | What it answers |
| --- | --- |
| latency | One question at a time, as the command line is used: time to first token, and total |
| throughput | 102 questions with 8 in flight, the shape of captioning and dataset building |
| determinism | The same 10 questions, 5 times each: are answers reproducible? |
| quality | Every answer kept, for refusal, citation and faithfulness counts |
| structured | The dataset builder's JSON call: does it parse? |

Held equal in every run:

- **The same weights at the same precision** on both sides, or the
  comparison measures the quantiser, not the server.
- **The same passages.** Retrieval ran once, on 28 October, and the
  passages were saved (`bench/cache/`). Every run replays them, so
  nothing here measures the retriever.
- **The same sampling**, sent explicitly to both.
- **The same path**: both are driven over HTTP with streaming on, so
  time to first token is taken at the same point. This costs vLLM a
  little (its offline batch call is faster), so its figures are a floor.
- **Medians and the 95th percentile**, never means: the first request to
  either server pays for a cold start.

### 9.4 Where it ran

| Machine | GPU | What fits | Used for |
| --- | --- | --- | --- |
| Laptop | RTX 3050, 6 GB, under WSL2 | 1.5B at fp16 (about 3.1 GB) on both; 7B only at 4-bit, partly on the CPU | First runs, the method |
| Lab machine | RTX PRO 5000, 48 GB, shared | 7B at fp16 on both, fully on the card | The figures quoted |

vLLM has no Windows build, so on the laptop both servers ran inside
WSL2. The app's own 7B fits in 6 GB only because Ollama runs it at 4-bit
(about 4.7 GB); at fp16 the weights alone are about 15 GB. So the laptop
could compare the two runtimes only on a smaller model, and the real
comparison waited for the lab card.

### 9.5 One question at a time

7B at fp16, lab card, quiet, 31 October:

| Measure | Ollama | vLLM |
| --- | --- | --- |
| Time to first token, median | 0.105 s | 0.084 s |
| Total per question, median | 1.501 s | 1.471 s |
| Decode speed, median | 79.6 tokens/s | 80.0 tokens/s |

Level. A queue of one gets nothing from batching, and the model is the
same, so it decodes at the same speed. vLLM starts answering a little
sooner.

### 9.6 Many at once

Same card, same model, 102 questions, 8 in flight:

| Server | Wall clock | Throughput |
| --- | --- | --- |
| Ollama as installed (1 slot) | 197.5 s | 76.3 tokens/s |
| Ollama, 4 slots | 80.5 s | 190.6 tokens/s |
| vLLM | 34.4 s | 422.9 tokens/s |

vLLM is 2.2 times Ollama at 4 slots. But 4 slots was not Ollama's best
setting, and with 8 requests in flight it served only 4 at a time. So
the fair question was: Ollama at 8 slots against vLLM.

By 13 November the lab card was shared and never quiet: other jobs held
it at 94 to 100% before every run. A quiet timing was not possible, so
the question was asked as **paired runs**: two settings alternated three
times, back to back, so that both arms of each round saw the same
background load. The ratio within a round is the result; the absolute
speeds, a third of the quiet-card ones, mean little.

| Round | Ollama 4 slots | Ollama 8 slots | | Ollama 8 slots | vLLM |
| --- | --- | --- | --- | --- | --- |
| 1 | 54.0 | 79.4 | | 79.0 | 160.2 |
| 2 | 55.0 | 78.5 | | 77.1 | 161.0 |
| 3 | 52.7 | 83.2 | | 76.9 | 160.7 |
| Ratio | | 1.47, 1.43, 1.58 | | | 2.03, 2.09, 2.09 |

Tokens per second. Eight slots is 1.5 times four, every round. vLLM is
2.1 times Ollama at eight, every round. Against Ollama at its best
measured setting, vLLM does twice the batch work.

### 9.7 Quality: the same

The same weights should give the same answers, and they nearly do.

| Measure, 7B fp16, lab card | Ollama | vLLM |
| --- | --- | --- |
| Answers word for word identical | 84 of 102 | |
| False refusals (strict count) | 20 of 100 | 20 of 100 |
| Real answers ending in the not-found line | 9 of 100 | 9 of 100 |
| Answers carrying a citation | 48% | 50% |
| Out-of-scope questions refused | 2 of 2 | 2 of 2 |
| Answers that hit the token cap | 3 | 2 |
| Faithfulness: supported / partly / unsupported | 38 / 28 / 14 | 38 / 28 / 14 |

The faithfulness rows were scored once, by an AI assistant against the
same passages, with a written rubric (`bench/results-gpu/
faithfulness_7b_fp16.json`). Four answers scored differently between the
runtimes; the totals came out equal. It is a single rater and not a
person, and the file says to spot-check before quoting. These answers
were written on 29 October, with the prompt from before v6, not today's.

**Reproducibility** differs, and the reason was found. With the same
question asked five times, Ollama gave the same answer 10 of 10 times;
vLLM 8 of 10, on the lab's 7B and the laptop's 1.5B alike. On the
laptop, with vLLM's prefix cache turned off, it was 10 of 10 (not
repeated on the 7B). The likely reason is that a reused prefix is not
computed in exactly the same arithmetic order as a fresh one, and at
fp16 a tiny difference can change a later token. It does not make vLLM
worse, but "the same question gives the same answer" held for vLLM only
without the cache.

### 9.8 Two other results

**The 7B on the laptop.** At 4-bit on both (Ollama's GGUF against
vLLM's AWQ), partly offloaded to the CPU, vLLM took 28.1 s a question
against Ollama's 8.6 s. vLLM is built for a model that fits on the card;
when it does not, Ollama's CPU offload is far better. Batch throughput
was still slightly ahead for vLLM (19.3 against 16.7 tokens/s). On the
laptop, Ollama is the right choice.

**Captioning**, 100 figures with the 3B vision model:

| Server | Wall clock | Per figure |
| --- | --- | --- |
| Ollama, one at a time | 418.8 s | 4.19 s |
| Ollama, 2 slots, 4 in flight | 324.4 s | 3.24 s |
| vLLM, 8 in flight | 80.6 s | 0.81 s |

4 times Ollama at 2 slots. It was not run against Ollama at more slots,
so this ratio is against a weaker Ollama setting than the 2.1 times
above, and should be quoted with the slot count beside it.

**Structured output.** The dataset builder asks the model for question
and answer pairs as JSON. With its own prompt and parsing, Ollama
dropped 8 of 50 passages that would not parse. Given the JSON schema
directly, both servers parsed 50 of 50 (Ollama 99 pairs, vLLM 100).
That is a builder fix, not a runtime one.

### 9.9 Mistakes made here, and their fixes

The first full run on the laptop went wrong in three ways that nothing in
its output showed:

- **Four copies of the vLLM phase ran at once** against one server. The
  server log showed 4 requests running through what was meant to be the
  one-at-a-time test. `run.py` now holds a lock, and refuses to time
  against a vLLM server that already has requests in flight.
- **Sampling was not matched.** vLLM applies the model's own generation
  settings by default, which for Qwen2.5 include a repetition penalty of
  1.1; Ollama applies none. The same weights gave different answers for a
  reason that had nothing to do with the runtime. Both now get the
  settings explicitly, the server is started with `--generation-config
  vllm`, and every result file records what was sent.
- **One long-lived server carried its prefix cache between scenarios**,
  so the throughput run replayed prompts the latency run had already
  cached. A fresh server is now started for each scenario, and the cache
  hit rate is recorded.

And three found later:

- **The refusal count was too high.** An older check gave 29 of 100 for
  the 7B. Reading the answers showed the model often adds the not-found
  sentence after an answer it gave; the strict count, of replies that
  are only a refusal, is 20. `report.py` now recounts from saved answers
  every time.
- **One paired run was discarded.** The first 8-slot run read 110.9
  tokens/s because it included loading the model on a busy card. Every
  run now loads the model with one short request first.
- **The captioning speed-up was described as against "Ollama, 4 in
  parallel".** It was 2 slots with 4 clients. The wording was corrected;
  the run was not repeated.

### 9.10 What was decided

| Use | Server | Why |
| --- | --- | --- |
| One person asking questions | Ollama (default) | Level on speed and quality, one install, works on Windows and on a 6 GB card |
| Captioning, building the dataset, several users | vLLM | Twice the batch throughput at Ollama's best setting |
| A model that does not fit on the card | Ollama | vLLM is three times slower per question when offloaded |
| Answers that must repeat exactly | Ollama, or vLLM without the prefix cache | 10 of 10 against 8 of 10 with the cache |

Because the switch is one variable, nothing in the app had to choose.

### 9.11 Rough edges found while writing this

- **The passages in `bench/cache/` are from 28 October**: the rules
  router, and before the command log was fixed. That is deliberate (every
  run saw the same passages), but these runs do not describe today's
  retrieval or today's prompt. Rebuilding the cache would start a new
  series.
- **The paired runs' per-run files are still on the lab machine.** Only
  the summary (`bench/results-gpu-paired/paired.csv`) was copied from the
  logs.
- **Captioning was not repeated against Ollama at more slots.**
- **Faithfulness was scored by an AI assistant**, once. A person has not
  checked it.
- **vLLM at a context of 8,192 tokens cut the two longest command-log
  answers short**; Ollama trimmed the prompt instead. Since 17 November
  command-log lookups by date or id are answered in code (chapter 7),
  so in the app this mostly no longer arises.
- **The browser demo** (chapter 11) that shows the two side by side has
  run on the laptop with small models, but not yet against the lab
  servers.
- **The lab login is written into 14 helper scripts** in `bench/gpu/`.
  That folder is kept out of git, but the scripts should not be copied or
  shared, and the login should move out of them.

### 9.12 Saying it in a minute

"The app's models run through Ollama, and I was asked whether vLLM would
be better. I put both behind one switch in the code and benchmarked them
on the same 7B weights at the same precision, the same frozen passages
and the same sampling, on the lab GPU. For one question at a time they
are level: 1.5 seconds each, the same decode speed. With many questions
at once, vLLM does about twice the work of Ollama at its best setting;
because the lab card was shared, I measured that with paired runs, two
settings back to back three times, and the ratio was 2.1 every round.
Answer quality is the same: 20 false refusals on both, and the same
faithfulness scores. On my 6 GB laptop, where the model doesn't fit,
vLLM is three times slower, so Ollama stays the default and vLLM is for
batch work like captioning, where it was four times faster. Along the
way I found that vLLM quietly applies a repetition penalty Ollama
doesn't, which would have made the comparison meaningless, so sampling
is now always sent explicitly."


## Chapter 10: LoRA fine-tuning

Written 17 November 2026. Code in `lora/`: `build_dataset.py` (348
lines), `train.py` (176), `serve_compare.py` (330), `query.py` (107), and
`bench/wsl/lora_pipeline.sh` (148), which runs them in order under WSL.
The second adapter (section 10.8): `build_rag_dataset.py` (220),
`train_rag.py` (118), `eval_rag.py` (139). Results in `lora/results.json`,
`lora/serve_results/` and `lora/rag_results/`, all kept out of git. Try
it with:

```text
bash bench/wsl/lora_pipeline.sh dataset   pairs written by the 7B, 400 chunks
bash bench/wsl/lora_pipeline.sh train     the adapter, about 2 minutes
bash bench/wsl/lora_pipeline.sh serve     PEFT, then vLLM, then the report
python lora/build_dataset.py --resplit    re-split the pairs on disk by chunk
python lora/build_rag_dataset.py          the same pairs, laid out with passages
python lora/train_rag.py --base Qwen/Qwen2.5-1.5B-Instruct --name rag-1.5b
python lora/eval_rag.py ask --url http://localhost:8124/v1 --model rag-1.5b
python lora/eval_rag.py report
```

### 10.1 What this part is, and what it is not

The rest of the project answers a question by retrieving passages and
handing them to a 7B model, which is told to use nothing else. This
chapter is a separate demonstration: can a small model be fine-tuned on
the documents, cheaply, on the laptop, and served next to its base model?

It is not part of the app. No answer the app gives goes through the
adapter. `llm.chat()` can name an adapter on a vLLM server (chapter 9),
but nothing calls it with one. The reason is in what the adapter was
trained to do: answer a question with **no passages in front of it**,
from what it absorbed in training. The app is built on the opposite rule,
that the model answers only from passages it is shown, and says so when
they are missing. A model that answers from memory is what the grounding
guard in chapter 7 exists to catch.

### 10.2 What LoRA is

Fine-tuning normally updates every weight in a model. LoRA (low-rank
adaptation) freezes them all and adds, next to chosen weight matrices, a
pair of thin matrices whose product is a small correction. Only those are
trained. The base model is unchanged on disk; the adapter is a separate
file that is loaded on top of it.

| Setting | Value |
| --- | --- |
| Base model | Qwen2.5-0.5B-Instruct, 496,195,456 weights |
| Matrices adapted | the four attention projections (q, k, v, o) in all 24 layers |
| Rank, scaling (alpha), dropout | 16, 32, 0.05 |
| Weights trained | 2,162,688, which is 0.44% |
| Adapter file | 8.7 MB, against about 1 GB for the base model |

The 2,162,688 can be checked by hand. A rank-16 pair beside a matrix
that maps `a` numbers to `b` numbers adds 16 × (a + b) weights. In this
model q and o map 896 to 896, and k and v map 896 to 128, so one layer
adds 16 × (1792 + 1792 + 1024 + 1024) = 90,112, and 24 layers make
2,162,688.

### 10.3 The dataset

A fine-tune needs examples. There is no set of real questions and answers
about these documents, so `build_dataset.py` writes one: for each chunk
over 200 characters, it asks the app's 7B (through `core/llm.py`) for one
or two question and answer pairs answerable from that chunk alone, in the
mission's terminology, each answer one to three sentences.

The reply's shape is held by a JSON schema, not asked for in the prompt.
Section 9.8 has the measurement that forced this: with plain "JSON
mode", 8 of 50 chunks gave nothing, and the builder's docstring adds
that the other 42 survived only through a rescue branch. With the schema, the build recorded in
`lora/data/build_meta.json`:

| Count | Value |
| --- | --- |
| Chunks sampled | 400 |
| Read through the schema | 400 |
| Rescued, dropped | 0, 0 |
| Pairs | 696 |

If no model is running, the builder falls back to a template (the
section heading turned into a question, the chunk's text as the answer)
and says so loudly; that path was not used.

**Where the pairs came from.** All 696 come from the confidential
corpus: almost all from three of the source files, 12 from the
procedures document, none from the telecommand log (which is not
chunked). The README still says `lora/` is built from the public
booklets; that was true of the demo as it first came into the
repository, not of these runs. So the pairs, the comparison answers and
the adapter itself all fall under the rule at the top of `.gitignore`:
`lora/data/`, `lora/adapters/`, `lora/results.json` and
`lora/serve_results/` are ignored. The adapter counts as derived text:
its weights hold traces of what it was trained on.

### 10.4 The split

Some pairs are held back to score the adapter. The first split held back
pairs at random. But the builder writes two pairs from most chunks, and
two questions about the same few sentences share most of their answer.
When one landed in training and the other in evaluation, the adapter was
scored on text it had been trained on: **71 of the 104 evaluation pairs
shared a chunk with training.**

`split_by_chunk()` now holds back whole chunks, so both pairs from a
chunk land on the same side. `--resplit` re-splits the pairs already on
disk without asking the model again.

| | First split (29 Oct) | By chunk (12 Nov) |
| --- | --- | --- |
| Training pairs | 592 | 595 |
| Evaluation pairs | 104 | 101 |
| Chunks, training / evaluation | | 340 / 60 |
| Evaluation pairs sharing a chunk with training | 71 | 0 |

The same 696 pairs are on disk under both splits (checked while writing
this), so the only change is which side each falls on. The adapter was
then trained again from scratch.

### 10.5 Training

| Setting | Value |
| --- | --- |
| Prompt layout | `### Instruction:` question, `### Response:` answer |
| Loss | on the answer's tokens only; the question's are masked out |
| Longest example | 384 tokens |
| Batch | 2, with 4 steps accumulated, so 8 per update |
| Learning rate, epochs | 0.0002, 3 |
| Updates | 225 |
| Where | the laptop's RTX 3050 (6 GB) under WSL2, bf16 |
| Time | 104.8 s |
| Loss | 3.34 at step 5, 1.80 at step 225, 2.185 averaged over the run |

Masking the question means the model is taught to produce answers, not
to repeat questions. One run, one seed.

### 10.6 Did it learn anything?

`serve_compare.py` asks 45 prompts: the five fixed questions
`query.py` uses, which have no reference answer, and the first 40
evaluation pairs, which do. Answers are scored by **token F1**: the words
an answer shares with the reference, balanced between precision (how
much of the answer is in the reference) and recall (how much of the
reference is in the answer).

| Model | Token F1, 40 pairs | Answers at the 150-token limit, of 45 |
| --- | --- | --- |
| Base, asked in the adapter's layout | 0.123 | 43 |
| Base, asked through its own chat template | 0.140 | 29 |
| Base with the adapter | 0.404 | 2 |

The work log's headline is that the adapter triples F1, and that is
what was measured. Splitting F1 into its two halves while writing this
chapter, from the saved answers, says what the tripling is made of:

| Model | Precision | Recall | Median answer, tokens |
| --- | --- | --- | --- |
| Base, chat template | 0.097 | 0.473 | 150 |
| Base with the adapter | 0.418 | 0.448 | 17 |

The reference answers have a median of 11 words. The base model writes
until it is cut off, so most of what it says is not in the reference,
and precision is low. The adapter learned to stop: its answers are the
reference's length. **Recall is level.** Over the 40 pairs the
adapter's recall is lower than the base model's on 16, higher on 10,
equal on 14; the difference, with a 95% interval from 10,000 paired
resamples, is −0.025 (−0.088 to +0.041), which includes zero. The F1
difference over the chat baseline is +0.264 (+0.193 to +0.336).

What this supports: the adapter learned the dataset's form, a short
answer in the documents' terms, and that is a real change (it changed
every one of the 45 answers). What it does not show: that the adapter
knows more about the documents than the base model. Word overlap cannot
tell "knows more" from "says it shorter", and the evaluation chunks were
held out, so the only facts the adapter could know about them are ones
repeated elsewhere in the same documents.

**Reading the 40 answers settles it.** On 17 November, with the
documents cleared as dummy data, Claude read each of the 40 scored
answers, the adapter's and the base model's (chat template), against its
reference. An answer counts as right if it states the reference's fact.

| | Base, chat template | With the adapter |
| --- | --- | --- |
| Right | 1 | 2 |
| Partly (a command named but its id missing, or paraphrased) | 2 | 2 |
| Wrong, or no answer | 37 | 36 |

The adapter's two right answers could be given from the question alone
("16 bits" for a 16-bit data word; "ON" for the state of a unit after a
power-on procedure), and the base model got the first of them too. Every
number and every command id the adapter gave was wrong: a spacecraft of
1,000,000 kg, uplinks "every 12 hours", a command mnemonic one word off
the real one. Two answers loop on a phrase until the token limit.

The two models fail differently, and that is what the adapter learned.
The base model does not know what the questions are about: it explains
the terms as cloud services, Linux tools, drones or galaxy models, or
declines. The adapter answers in the documents' register, short and in
their vocabulary, with invented facts. That is the precision gain in the
table above, and it is why a closed-book adapter is the wrong tool for
this app (section 10.1): its wrong answers sound right.

The reading also checked the references. 2 of the 40 are wrong in
themselves (one answers a different question, one is not a sentence that
means anything), and one gives a date as "the upcoming Wednesday". They
were written by the 7B and the adapter was trained on 595 more like them.

The clean split did not lower the score (0.388 under the first split,
0.404 now). That fits the same reading: a score driven by answer length
does not depend much on whether the facts were seen.

### 10.7 Serving it two ways

**PEFT** is the Hugging Face library that trained the adapter. Serving
with it, as `query.py` does, means loading the base model in a script,
generating, wrapping the model with the adapter, generating again, one
prompt at a time.

**vLLM with LoRA enabled** keeps one copy of the base model and the
adapter on the card. Each request names `Qwen/Qwen2.5-0.5B-Instruct` or
`adityal1`, and the server batches requests for both together; nothing
is reloaded between them.

Both decode greedily, repetition penalty 1.0, 150 tokens at most, on the
same laptop card.

**Do they compute the same thing?**

| | Base | With adapter |
| --- | --- | --- |
| Answers word for word the same, PEFT and vLLM, of 45 | 10 | 41 |
| Median share of the answer before they diverge | 0.29 | 1.00 |
| Token F1, PEFT / vLLM | 0.123 / 0.123 | 0.404 / 0.401 |

The adapter path agrees. The base model's low agreement is its length:
it runs to the limit on almost every prompt, and over 150 tokens a small
rounding difference between the two libraries' kernels eventually picks
a different word, after which the texts part. Its score is unchanged.

**What does it cost?** All 45 prompts, through both models (90 answers):

| Method | Time |
| --- | --- |
| PEFT, one after another | 172.2 s |
| vLLM, one after another | 58.8 s |
| vLLM, base and adapter requests mixed, 8 in flight | 10.0 s |

The 17 times in the work log is 172.2 s against 10.0 s. It is two
effects multiplied: vLLM is 2.9 times faster on the same one-at-a-time
work, and batching then gives 5.9 times more. In the mixed batch, 50 of
the 90 answers match the one-at-a-time run word for word; batching
changes the order of the arithmetic, which changes bf16 rounding, which
matters most again for the base model's long answers. Loading took 0.9 s
for the base model and 0.21 s for the adapter.

### 10.8 A second adapter, trained on passages

Section 10.6 showed the first adapter could not have learned the facts:
it was trained to answer with no passages in front of it. The app works
the other way, so on 17 November a second adapter was trained on the
app's own task, on the lab GPU (RTX PRO 5000, 48 GB), next to the app's
7B already being served there.

**The data** (`lora/build_rag_dataset.py`) reuses the same 696 pairs and
the same split by chunk, with no new model calls. Each question is laid
out exactly as `generate.answer()` lays out a real one: the v9 system
prompt, then five passages under their `[Source n | ...]` headers, then
the question. The five are the pair's source chunk and four passages the
app's retriever returns for that question, shuffled; the target is the
reference answer and `[Source n]` for wherever the source landed. For
refusal examples the same question gets five passages with nothing that
mentions the answer, and the target is the not-found sentence: 205 in
training (a third of the questions), and every one of the 101 test
questions once each way.

**The training** (`lora/train_rag.py`): the model's own chat template,
up to 3,072 tokens per example (median 1,335), LoRA rank 16 on every
linear layer, two passes, loss on the answer only. 0.5B: 154 s, 1.75% of
the weights. 1.5B: 355 s, 1.18%.

**The test** (`lora/eval_rag.py`) serves each base model with its
adapter on vLLM and asks the 202 test prompts with the app's sampling;
the 7B is asked the same prompts on its own server. Scored on the laptop:

| Model | Right, of 101 answerable | Token F1 | Cites a right passage | Refused although answerable | Refused, of 101 unanswerable | Did not refuse |
| --- | --- | --- | --- | --- | --- | --- |
| The app's 7B | 72 | 0.495 | 69 | 5 | 67 | 34 |
| 1.5B | 65 | 0.509 | 6 | 1 | 25 | 76 |
| **1.5B + adapter** | 59 | **0.597** | **84** | 7 | **74** | **27** |
| 0.5B | 30 | 0.304 | 0 | 3 | 18 | 83 |
| 0.5B + adapter | 29 | 0.417 | 50 | 15 | 37 | 64 |

"Right" means the reply holds 70% of the reference's content words.
Both automatic columns were checked by reading replies (Claude, about 25
of them), and both need a word of explanation:

- **The adapter's "right" count is too low.** Of the 14 answers the 1.5B
  got right and the adapter did not, most are right but short ("16
  bits", "~1 hr", "to achieve higher accuracy"). Some are short in a way
  that matters: "Issue PB Q Clear" without the three commands that do
  it, half of a list of units switched off. The references are one to
  three sentences, so the adapter learned to be that brief.
- **"Did not refuse" is not "invented".** Most of these replies answer a
  neighbouring question from a passage that is there: the other door's
  purpose, the description of the fault logic, how long a set is valid.
  A few pick the wrong value from a passage ("6 weeks" from a different
  test case). The 7B does the same thing more often.

What it shows:

1. **Trained on its real task, a 1.5B model cites better and refuses
   better than the app's 7B**: the right passage 84 times against 69,
   the right refusal 74 against 67. That is the skill the first adapter
   could not learn, learned in six minutes of training.
2. **It answers less fully.** It inherits the references' length, and
   for operations that is a fault: a procedure answer needs every
   command.
3. **0.5B is too small.** The adapter teaches it to cite and to refuse
   more, but not to answer more questions right.

It is still not part of the app. Before it could be: references written
to be complete (every command and value the passage gives, not one
sentence), a run on the app's own 102 questions with the grounding guard
and the reading in chapter 7, and more than one training run.

### 10.9 Mistakes made here, and their fixes

- **The evaluation set overlapped training** (section 10.4). 71 of 104
  pairs shared a chunk with training. Split by chunk, 0 shared, retrained
  and re-scored. The `.pair-split` files on disk are the old run, kept to
  compare.
- **The builder lost pairs** to replies of the wrong shape. Fixed with
  the schema; 400 of 400 chunks now read cleanly (section 10.3, and 9.8).
- **The base model was asked in a layout it had never seen.** The
  `### Instruction` layout is the adapter's; asked that way, the base
  model ran to the limit on 43 of 45 prompts. `serve_compare.py` now
  also asks it through its own chat template, and that (0.140) is the
  baseline to read the adapter against, not 0.123.
- **On the second adapter (10.8)**, five, all caught before any figure
  was written: transformers 5 no longer takes `warmup_ratio`; the run
  started before the 1.5B model had downloaded; the first refusal
  examples were not clean (reading the 7B's replies showed passages that
  held the answer in other words, so the filter now drops any passage
  sharing half the answer's words or any of its numbers); some test
  prompts were longer than the 4,096 tokens the small models were served
  with (now 8,192); and a tmux wait matched its own session's name. The
  first, unclean results were thrown away.
- **Sampling was not matched.** The model's own settings carry a
  repetition penalty of 1.1. `serve_compare.py` sends 1.0 to both paths
  and starts vLLM with `--generation-config vllm`, as in chapter 9.

### 10.10 Rough edges found while writing this

- **"Triples F1" overstates it.** The gain is in precision, from shorter
  answers; recall is level, and reading the 40 answers found 2 right,
  both answerable from the question (section 10.6). The work log's
  sentence should say that the adapter learned the documents' style, not
  their facts.
- **17 times compares a script with a server.** PEFT ran one prompt at
  a time. Transformers can batch prompts, and PEFT can mix base and
  adapter requests in one batch; neither was tried. The like-for-like
  figure is 2.9 times, one at a time. The docstring of `serve_compare.py`
  says a mixed batch "is the thing PEFT-in-a-script cannot do", which is
  not right.
- **`query.py` does not set the repetition penalty**, so it inherits 1.1
  and its answers will not match `serve_compare.py`'s. Its output file,
  `lora/comparison.json`, is not on disk.
- **The README's line about `lora/`** says it is built from the public
  booklets. These runs were built from the confidential corpus.
- **Small and single.** One training run, one seed, 40 scored pairs, of
  101 held out. The five fixed questions ask about the mission in
  general (payloads, orbit, launch) and have no reference answer, so they
  show the adapter changes answers, not whether it is right.
- **The pairs were written by a model.** Reading 40 of the evaluation
  pairs found 2 references wrong in themselves. The 595 training pairs
  have not been read; the adapter was trained on whatever errors they
  hold.
- **The second adapter (10.8) is one run.** One seed, two sizes, scored
  automatically; the reading that checked the scoring covered about 25
  replies, by Claude, not a person. 11 training examples longer than
  3,072 tokens were left out.
- **What counts as "unanswerable" is a word-overlap test.** A passage
  that gives the answer in entirely different words would pass it, so a
  few refusal examples may still hold the answer.
- **The trained adapters are on the lab machine only**
  (`~/satrag_raghav/lora_rag/lora/adapters/`); only the scores and
  replies were copied back. The lab disk is 97% full.
- **Lab access is now by SSH key** (`ssh labgpu`), set up on 17
  November. The old password was shared in a chat and is still in the
  `bench/gpu/` scripts; it should be changed and those scripts deleted.
  The 7B demo server from 14 November is still running there, holding
  22 GB of the card.

### 10.11 Saying it in a minute

"Separately from the app, I fine-tuned a small model on the documents
with LoRA, which freezes the model and trains a thin correction beside
its attention layers: 2.2 million weights out of 496 million, an 8.7 MB
file, trained in under two minutes on my laptop GPU. There was no
question-and-answer data, so the 7B wrote 696 pairs from 400 chunks,
held to a JSON schema so none were lost. My first split leaked: 71 of
104 test pairs came from chunks the adapter had trained on, so I split
by chunk and retrained. The adapter's token F1 went from 0.14 to 0.40,
but when I split that into precision and recall, recall was level, and
when the 40 answers were read, 2 were right, both answerable from the
question. What it learned was to answer briefly in the documents' terms,
not their facts, which makes its wrong answers sound right. Then I
served it two ways: vLLM agreed with the PEFT reference on 41 of 45
answers, and serving the base and the adapter together from one vLLM
server took 10 seconds against 172 for PEFT in a script, though about 3
times of that is like for like and the rest is batching. Finally, I
trained a second one on the app's real task, five passages in and a
cited answer out, on the lab GPU: a 1.5B with that adapter cites the
right passage and refuses correctly more often than the app's 7B, though
its answers are too brief. The app doesn't use the first adapter,
because it answers from memory, and the app is built to answer only from
the passages it shows."


## Chapter 11: The browser demo

Written 17 November 2026. Code: `demo/server.py` (436 lines),
`demo/README.md` (54), and `parallel_demo.py` (128), its command-line
forerunner. The page itself (`demo/page.html`) is not covered here; this
chapter is about what runs behind it. Try it with:

```text
python demo\server.py aditya-l1                                   laptop, local servers
python demo\server.py aditya-l1 --ollama-model qwen2.5:7b-instruct-fp16
then open http://127.0.0.1:8765
python parallel_demo.py aditya-l1 -n 8     the same idea in the terminal, one server
```

### 11.1 What it is for

Chapter 9 settled Ollama against vLLM with a benchmark. The demo is the
same comparison made visible: one question, or a batch, answered by both
servers from the same passages, with the text arriving live and the
timings beside it. It was built because a working prototype was asked
for, one that shows the difference rather than a table of it.

What it is meant to show is chapter 9's result: the answers are the
same, one question takes about the same time on either, and vLLM pulls
ahead only when several questions arrive at once.

### 11.2 How a question travels

1. The browser sends the question to `demo/server.py`, which listens on
   `127.0.0.1:8765` only.
2. The server routes it and retrieves the top 5 passages **once**, on
   the machine that holds the store, with the app's own `router` and
   `retrieve`.
3. It builds the prompt the app builds: `generate.SYSTEM_PROMPT` (v9),
   the passages, the question, and the code's note for index-only
   procedures (`grounding.context_note`, with `generate.NOTE_RELATED`).
4. The same messages go to both servers, which stream their answers
   back.
5. Each answer goes through the app's post-model steps, the same on both
   sides: the grounding guard, `drop_false_no_steps`,
   `drop_trailing_not_found`, `check_citations`. The count of
   identifiers the guard removed is sent with the answer.

So the only thing that differs between the two columns is the server.
The exceptions are the questions the app answers in code, with no model:
a figure, from its caption (chapter 3); a command-log lookup by date,
command id or mnemonic, from every matching record (chapter 7); and a
question that retrieves nothing. There is nothing to compare, so the
page shows the reply once, with the reason. Both the app and the demo
decide this with one function, `generate.answer_without_model()`, so
the two cannot drift apart.

### 11.3 What the server offers

| Address | What it returns |
| --- | --- |
| `/` | the page |
| `/api/status` | each server's address, whether it is up, the model it will use, and the 16 fixed questions |
| `/api/ask?q=…&mode=together` or `mode=turns` | one question, both servers, as a live event stream |
| `/api/batch?n=…&flight=…&first=ollama` | the first `n` fixed questions, all through one server and then the other, `flight` at a time |
| `/api/measured` | the recorded figures from `bench/results-gpu-quiet/` (31 October) and `bench/results-gpu-paired/` (13 and 14 November) |

The live text uses Server-Sent Events, a plain HTTP response that the
server keeps writing to: `search` (sources, route, search time), then
`start`, `token` and `done` per side, then `end`. A batch reports
`searching` per question, an `answer` as each finishes, and `side_done`
with the wall time and tokens per second. A server that is down gives a
`done` with an error, not a hung page.

Everything is the Python standard library: a threading HTTP server and
`urllib` to the two model servers. Only the app's own retrieval stack is
imported.

### 11.4 What is held equal

- **Sampling**, sent on every request: temperature 0, seed 1234,
  repetition penalty 1.0, the app's token limit (768).
- **The prompt**, built once per question and given to both.
- **No cold start in front of people.** At start-up the server runs one
  search, which loads the embedder and the reranker, and sends each model
  server one throwaway request, so neither loads its model during the
  first comparison.
- **One search at a time.** The embedder, reranker and store are shared
  and the handler runs a thread per request, so searches take a lock.
  Search time is reported apart from answer time.
- **The weights.** Both servers should hold the same model at the same
  precision, or the page compares two models, not two runtimes. This one
  is left to whoever starts the servers: `--ollama-model` names Ollama's
  tag, and vLLM's model is whatever it lists first.

### 11.5 How it times

Per answer, from the moment the request leaves:

- `first_s`, the wait for the first piece of text;
- `total_s`, the wait for the last;
- tokens per second over the writing time only (total minus the first
  wait), which is how `bench/` reports it.

**Both at once** (`together`) starts the two servers together. They
share the card while they write, so it looks like a race but neither has
the card to itself. **One after the other** (`turns`) gives each the
card alone, and is the fair reading of speed for one question.

A batch sends the questions to one server, waits for all of them, then
sends them to the other. Its figure is wall time and tokens over wall
time. Ollama answers as many at once as it has slots
(`OLLAMA_NUM_PARALLEL`); vLLM batches whatever arrives.

### 11.6 Running it against the lab

Retrieval stays on the laptop, where the store is. The model servers run
on the lab card and listen only inside the lab machine, so the laptop
reaches them through an ssh login with two port forwards, and points
`OLLAMA_HOST` and `SATRAG_VLLM_URL` at the forwarded local ports.
`bench/gpu/demo_start.sh` starts vLLM on the lab machine (7B at fp16,
context 8,192) and `ollama_slots.sh 8` restarts Ollama with 8 slots. The
step-by-step version is in `RUN_DEMO.txt`, kept out of git with the
other lab notes.

Two rules from the README worth repeating. The page is served to the
local machine only, because the answers quote the documents. And the
retrieved passages travel to the model servers with every question, so
those servers must be on a network the documents are allowed on.

### 11.7 parallel_demo.py

Before the page there was a terminal script. It runs the app's real path
(`router.plan`, `retrieve.search`, `generate.generate`, so log lookups
are answered in code there) on whichever server `SATRAG_LLM` points at:
the fixed questions one at a time, then all of them at once, with search timed
apart. Run it once per server and compare. The page took its 16 fixed
questions from it, and imports them from there, so the two use the same
set.

### 11.8 Tested today

Run on the laptop on 17 November, after this session's prompt changes,
with Ollama's 7B (4-bit, the laptop's default, one slot) and no vLLM
server running:

| Check | Result |
| --- | --- |
| Start-up | retrieval models loaded, Ollama up, vLLM reported DOWN |
| One question, `turns` | 5 passages in 0.44 s; Ollama first text 9.1 s, done 22.2 s, 267 tokens, 20.5 tokens/s, 1 citation, 0 ids removed; vLLM side a clean connection error |
| A figure question | answered from the caption, no model called |
| A batch of 4, 4 in flight | Ollama 83.8 s, 1,414 tokens, 16.9 tokens/s (its one slot queues them); vLLM 4 failed, reported as failed |
| `/api/measured` | returned at once |
| `/api/status` | 8.3 s, because the check of the absent vLLM server is slow to fail |

That proves the server's path works with today's prompt and post-model
steps. It says nothing about speed: one server, a 4-bit model, a 6 GB
card. On 16 November, before this session, the page had been run with
small models on both sides on the laptop. **It has not been run against
the lab servers.** The first two attempts failed: the start scripts were
not on the lab machine, and the laptop was pointed at the lab address
instead of the forwarded ports.

### 11.9 Rough edges found while writing this

- **Command-log lookups took a different path from the app** (fixed the
  same day). The app answers a lookup by date, command id or mnemonic in
  code, but the demo handled only the caption case, so it sent those
  questions, and ones that retrieved nothing, to both models. The rule
  now lives in `generate.answer_without_model()`, which `answer()` and
  the demo's `prepare()` both call; the page shows the reason sent with
  the reply. Two checks in `tests/test_answering.py` (39 in all, all
  pass), one of them driving `prepare()` with made-up records. The batch
  tab still sends every question to the models, which is right for a
  speed test; none of the 16 fixed questions is answered in code.
- **Repeating a batch on a long-lived server replays cached prompts.**
  The benchmark started a fresh server for each scenario so no run
  inherited another's cache (section 9.9). `demo_start.sh` leaves vLLM's
  prefix cache on, and the batch tab always sends the same questions, so
  a second run of the same batch is not a clean timing. The Measured tab
  is the number to quote.
- **Nothing checks the weights match.** The status shows both model
  names, but a 4-bit Ollama tag against an fp16 vLLM model would run
  without complaint. On the laptop, Ollama's default tag is the 4-bit
  one.
- **The status call waits on an absent server**, 8.3 s on the laptop
  with vLLM down, and the page waits with it.
- **A live run is one run, on a shared card.** The page says so under a
  batch result, and the Measured tab holds the repeated figures.
- **The lab login is in the `bench/gpu/` scripts** the demo's lab run
  uses (section 9.11). They are out of git, but should not be copied
  until the login is moved out.

### 11.10 Saying it in a minute

"To show the Ollama-against-vLLM result instead of just tabling it, there
is a small web demo. A local Python server, standard library only, routes
and retrieves once on the laptop, builds exactly the prompt the app
builds, sends it to both model servers, and streams the two answers back
side by side with their timings. Both answers go through the app's own
grounding guard and citation check, sampling is pinned the same, and each
server gets a warm-up request so neither loads its model in front of the
audience. You can ask one question with both at once or one after the
other, run a batch with a set number in flight, which is where vLLM pulls
ahead, or show the recorded benchmark figures, which are the ones to
quote because the lab card is shared. It works end to end on the laptop
with today's prompt; it still has to be run once against the lab
servers."
