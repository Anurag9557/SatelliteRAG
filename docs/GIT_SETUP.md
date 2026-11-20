# Putting this on GitHub

Two audiences: the first half is for pushing the repository, the second
is what your guide needs in order to run it.


## Before anything else

The source documents are confidential, and so is everything derived from
them. Three kinds of file must never be committed:

| | Why |
|---|---|
| `satellites/*/documents/` | the documents themselves |
| `satellites/*/extracted/` | their text, captions and command records |
| `satellites/*/vectorstore*/` | ChromaDB keeps the raw text of every chunk beside its embedding, and the BM25 pickles keep it too — a store is the documents in another form |

`.gitignore` covers all three, including the ten `vectorstore_<model>/`
folders the model comparison produced. Verify before the first push, not
after.


## One decision to make first

`testset.py` holds the 100 benchmark questions. The questions themselves
are innocuous, but the accept conditions quote short phrases from the
documents — procedure numbers, section titles, terms like `crab source`
or `master mep`. Nothing substantial, but it is document-derived.

Judge it against the confidentiality rule, or ask your guide. If it
should stay local:

    echo testset.py >> .gitignore

Then `evaluate.py` will not run for anyone who clones the repository, so
say so in the README. Committing it is defensible if the repository is
private; leaving it out is the safer default.


## Pushing, first time

    cd C:\Users\anura\Desktop\SatelliteRAG

    git init
    git branch -M main

Check what git would actually take, before adding anything:

    git status --short

Read the list. If any of these appear, stop and fix `.gitignore`:
`documents/`, `extracted/`, `vectorstore`, `.venv`, `models/`, `.pdf`,
`.docx`.

An explicit check for the dangerous ones:

    git status --short | Select-String "documents|extracted|vectorstore|\.pdf|\.docx|\.venv|models/"

No output means clear. Then:

    git add .
    git status --short

Look again — this is the last point at which a mistake is cheap. Then:

    git commit -m "SatelliteRAG: multi-satellite retrieval pipeline with evaluation"

Create the repository on GitHub as **private**, without a README (you
have one), then:

    git remote add origin https://github.com/<your-user>/SatelliteRAG.git
    git push -u origin main

**Confirm it is private.** GitHub → the repository → Settings →
General → Danger Zone → it should read "Change to public", meaning it is
currently private. If it reads "Change to private", change it now.


## If something confidential is committed by accident

A commit is not undone by deleting the file — the content stays in the
history and remains readable.

Before pushing:

    git reset --soft HEAD~1        # undo the commit, keep the files
    # fix .gitignore, then commit again

After pushing: assume it is exposed. Delete the repository on GitHub,
fix `.gitignore`, and push a fresh one.


## Everyday use

    git add -A
    git commit -m "what changed"
    git push

Before each commit, the same one-line check:

    git status --short | Select-String "documents|extracted|vectorstore|\.pdf|\.docx"


## What ends up in the repository

    cli.py                    one entry point for everything
    core/                     13 modules, the pipeline
    demo/                     browser demo, Ollama and vLLM side by side
    lora/                     fine-tuning: closed-book and RAFT-style adapters
    bench/                    the Ollama-vs-vLLM benchmark
    tests/                    checks on the code-decided parts of answering
    docs/                     walkthrough, before/after record, context, this file
    evaluate*.py, testset*.py retrieval, answer and router evaluation
    compare_*.py              embedder, reranker and parser comparisons
    review_*.py               pages for reading answers and captions by eye
    collect_results.py        merge result files into one table
    results/                  summary tables of the measurements
    README.md                 usage, flow diagrams, results
    SETUP.txt                 installation
    requirements.txt
    .gitignore

The documents, the models, the vector stores, extracted text, run answers,
LoRA data and adapters stay on the machine; `.gitignore` covers them.


## For your guide: running it after cloning

Send this section on its own.

**1. Clone**

    git clone https://github.com/<your-user>/SatelliteRAG.git
    cd SatelliteRAG

**2. Python 3.11 or 3.12.** Newer versions often have no matching CUDA
wheel.

    py -3.11 -m venv .venv
    .venv\Scripts\activate

**3. PyTorch first, before requirements.txt.** pip resolves torch
against PyPI, which serves CPU-only wheels, so the order matters. Check
what CUDA version the driver reports:

    nvidia-smi

Then install the matching wheel:

    CUDA 12.8   pip install torch --index-url https://download.pytorch.org/whl/cu128
    CUDA 12.1   pip install torch --index-url https://download.pytorch.org/whl/cu121
    CPU only    pip install torch

Verify — the version must end in `+cuXXX` and print True:

    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

**4. Everything else**

    pip install -r requirements.txt
    python cli.py prefetch minilm-l6

**5. Ollama**, for figure captioning and answer generation:

    ollama pull qwen2.5vl
    ollama pull qwen2.5:7b-instruct

**6. Add a satellite.** Make a folder, copy its documents in, run one
command. No configuration: each document's type is recognised from its
filename.

    satellites\<name>\documents\
        ...FCP...pdf
        ...MOR...pdf
        ...operations manual...docx
        ...guidelines...doc
        ...telecommand...txt

    python cli.py scan <name>       check they were recognised
    python cli.py build <name>      extract, caption, embed

Captioning is the slow step, roughly 20 to 40 minutes for 200 figures on
a laptop GPU. It resumes if interrupted.

**7. Ask questions**

    python cli.py query <name> "what is the procedure for switching on OBC?"
    python cli.py shell <name>

**8. Reproduce the measurements** (needs `testset.py` and a built store)

    python evaluate.py --compare
    python compare_models.py --models minilm-l6 bge-small-en bge-base-en
    python compare_rerankers.py --top 3 --eval-only
    python collect_results.py

**Where to look first:** `PROJECT_CONTEXT.md` for the architecture and
every measured number, `COMPARISON.md` for what changed against the
previous pipeline and why, `results/all_results.csv` for the raw figures.
