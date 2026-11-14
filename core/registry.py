"""
Satellite registry.

Every satellite lives in its own folder under satellites/ and owns its
documents, extracted JSON, and vector store. Nothing is shared between
satellites except the embedding model cache in models/.

Layout per satellite:

    satellites/<slug>/
        satellite.yaml      optional. Defaults apply when it is absent.
        documents/          the 5 source files
        extracted/          chunks.json, image_captions.json, tch_records.json
        vectorstore/        ChromaDB collections + BM25 pickles

Adding a satellite is nothing more than making a folder under satellites/
and copying the documents in. The document type of each file is worked
out from its name, so nothing has to be typed into a config file.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    raise SystemExit("Missing dependency: pip install pyyaml")


PROJECT_ROOT   = Path(__file__).resolve().parent.parent
SATELLITES_DIR = PROJECT_ROOT / "satellites"
MODELS_DIR     = PROJECT_ROOT / "models"

CONFIG_NAME = "satellite.yaml"

# Document types every satellite is expected to have. A satellite may
# have a subset; missing types are simply skipped during ingestion.
# HANDBOOK is a generic prose type (same extract_prose_pdf() reader as
# MOR/OPS_MANUAL/GUIDELINES) for satellites with more prose documents than
# those three slots hold, e.g. Aditya-L1's public ISRO handbooks.
DOC_TYPES = ["FCP", "MOR", "OPS_MANUAL", "GUIDELINES", "HANDBOOK", "TCH"]

# Parser to use per file extension.
PARSER_BY_SUFFIX = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".doc":  "doc",
    ".txt":  "txt",
}

# Filename patterns used to work out which of the five document types a
# file is, so a new satellite needs no configuration at all: drop the
# documents in the folder and the type is recognised from the name.
# Most specific patterns first; OPS_MANUAL is last because "manual" is
# the loosest match.
DOC_TYPE_PATTERNS = [
    ("TCH",        [r"\btch\b", r"telecommand", r"command\s*history",
                    r"cmd\s*hist", r"\btc\s*history"]),
    ("FCP",        [r"\bfcp\b", r"flight\s*control\s*proc",
                    r"flight\s*proc"]),
    ("MOR",        [r"\bmor\b", r"mission\s*oper", r"mission\s*plan",
                    r"mission\s*report"]),
    ("GUIDELINES", [r"guide\s*line", r"guidance", r"normal\s*phase",
                    r"\bgl\b"]),
    ("OPS_MANUAL", [r"operation\s*s?\s*manual", r"\bops\s*manual",
                    r"\bmanual\b", r"manual"]),
]

# Used only when the filename gives nothing away.
FALLBACK_BY_SUFFIX = {".txt": "TCH"}

# Helper files that live in documents/ but are not documents. Without
# this the placeholder note would be picked up as the telecommand log,
# because any .txt falls back to TCH.
IGNORED_FILENAMES = {
    "put_documents_here.txt",
    "readme.txt",
    "notes.txt",
    "thumbs.db",
    "desktop.ini",
}

# Chunking defaults, applied when satellite.yaml does not override them.
# FCP procedures are kept large so a procedure is not split mid-sequence.
# Captions and command records are atomic and never split.
DEFAULT_CHUNKING = {
    "FCP":        {"max_chars": 2500, "overlap": 200},
    "MOR":        {"max_chars": 1000, "overlap": 150},
    "OPS_MANUAL": {"max_chars": 1200, "overlap": 200},
    "GUIDELINES": {"max_chars": 1200, "overlap": 200},
    "HANDBOOK":   {"max_chars": 1200, "overlap": 200},
}

DEFAULT_RETRIEVAL = {
    "top_k":   5,
    "rrf_k":   60,
    "n_dense": 20,
    "n_bm25":  20,
    # A cross-encoder re-scores the fused candidates by reading each one
    # against the question. It sharpens the first result and can push a
    # correct passage out of the top five; results/ has both sides. Set
    # false to fall back to fusion order alone.
    "rerank":      True,
    "rerank_pool": 20,
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # How a question is routed before searching: rules, soft or llm.
    # See core/router.py. Soft is the default because it scored highest
    # on the 100-question bench (results/router_retrieval.csv): forcing
    # procedure chunks to the front, as the rules router does, costs MRR.
    "router":       "soft",
    "router_model": "qwen2.5:7b-instruct",
}

DEFAULT_MODEL = "minilm-l6"


def normalise_filename(name: str) -> str:
    """Underscores and camel case become spaces so word patterns match."""
    name = re.sub(r"\.[^.]+$", "", name)              # drop the extension
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)  # camelCase -> camel Case
    name = re.sub(r"[^A-Za-z0-9]+", " ", name)
    return name.lower().strip()


def detect_doc_type(filename: str) -> Optional[str]:
    """
    Work out the document type from a filename. Returns None when nothing
    matches, so the caller can report it rather than guess wrong.
    """
    text = normalise_filename(filename)
    for doc_type, patterns in DOC_TYPE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return doc_type
    return FALLBACK_BY_SUFFIX.get(Path(filename).suffix.lower())


def slugify(text: str) -> str:
    """Turn a satellite name into a folder-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return slug.strip("-")


def prettify(slug: str) -> str:
    """aditya-l1 -> Aditya-L1, used when no config supplies a name."""
    return "-".join(part.capitalize() for part in slug.split("-"))


def is_document_file(path: Path) -> bool:
    """True for a real source document, false for helper and temp files."""
    return (
        path.is_file()
        and not path.name.startswith((".", "~$"))
        and path.name.lower() not in IGNORED_FILENAMES
        and path.suffix.lower() in PARSER_BY_SUFFIX
    )


@dataclass
class Satellite:
    slug: str
    name: str
    root: Path
    config: dict = field(default_factory=dict)

    # Set only when comparing models. store_suffix keeps each model's store
    # in its own folder so they persist side by side and the working store
    # is never overwritten; model_override picks the embedder without
    # editing satellite.yaml.
    store_suffix: str = ""
    model_override: str = ""

    # Paths

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    @property
    def extracted_dir(self) -> Path:
        return self.root / "extracted"

    @property
    def vectorstore_dir(self) -> Path:
        if self.store_suffix:
            return self.root / f"vectorstore_{self.store_suffix}"
        return self.root / "vectorstore"

    @property
    def images_dir(self) -> Path:
        return self.extracted_dir / "images"

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_NAME

    # Documents

    def files_on_disk(self) -> list:
        """Every source file sitting in documents/."""
        if not self.documents_dir.exists():
            return []
        return sorted(
            p for p in self.documents_dir.iterdir() if is_document_file(p))

    @property
    def documents(self) -> dict:
        """
        Returns {doc_type: {"file", "parser", "path", "source"}}.

        Files in documents/ are matched to a document type by name, so a
        new satellite needs no configuration. Anything named explicitly in
        satellite.yaml wins over the detected match, which is the escape
        hatch for files the patterns cannot recognise.
        """
        out = {}

        # Detected from the folder.
        for path in self.files_on_disk():
            doc_type = detect_doc_type(path.name)
            if not doc_type or doc_type in out:
                continue
            out[doc_type] = {
                "file":   path.name,
                "parser": PARSER_BY_SUFFIX.get(path.suffix.lower(), ""),
                "path":   path,
                "source": "detected",
            }

        # Explicit config overrides detection.
        for doc_type, entry in (self.config.get("documents") or {}).items():
            if not entry:
                continue
            filename = (entry.get("file") or "").strip()
            if not filename:
                continue
            path = self.documents_dir / filename
            # A mapping to a file that is not in the folder must not hide
            # one that is. satellite.yaml travels with the repo while the
            # documents do not, so the same mapping meets different files
            # on different machines; without this the detected document
            # was dropped and its type left pointing at nothing.
            if not path.exists() and doc_type in out:
                continue
            out[doc_type] = {
                "file":   filename,
                "parser": entry.get("parser")
                          or PARSER_BY_SUFFIX.get(path.suffix.lower(), ""),
                "path":   path,
                "source": "config",
            }
        return out

    def unassigned_files(self) -> list:
        """Files in documents/ that no pattern recognised."""
        claimed = {d["file"] for d in self.documents.values()}
        return [p for p in self.files_on_disk() if p.name not in claimed]

    def missing_types(self) -> list:
        """Document types with no file, in the canonical order."""
        have = set(self.documents)
        return [t for t in DOC_TYPES if t not in have]

    def missing_documents(self) -> list:
        """Document types whose file is named but not present on disk."""
        return [
            doc_type for doc_type, entry in self.documents.items()
            if not entry["path"].exists()
        ]

    # Configuration

    @property
    def embedding_model(self) -> str:
        if self.model_override:
            return self.model_override
        return (self.config.get("embedding") or {}).get("model", DEFAULT_MODEL)

    def chunking_for(self, doc_type: str) -> dict:
        """Chunking parameters for a document type, config over defaults."""
        base = dict(DEFAULT_CHUNKING.get(
            doc_type, {"max_chars": 1200, "overlap": 200}))
        override = (self.config.get("chunking") or {}).get(doc_type) or {}
        base.update(override)
        return base

    @property
    def retrieval(self) -> dict:
        cfg = dict(DEFAULT_RETRIEVAL)
        cfg.update(self.config.get("retrieval") or {})
        return cfg

    # State

    def is_extracted(self) -> bool:
        return (self.extracted_dir / "chunks.json").exists()

    def is_ingested(self) -> bool:
        return (self.vectorstore_dir / "chroma.sqlite3").exists()


def satellite_paths() -> list:
    """
    Every satellite folder. A folder counts if it has a config file or
    simply some documents, so creating a satellite is nothing more than
    making a folder and dropping the files in.
    """
    if not SATELLITES_DIR.exists():
        return []

    found = []
    for p in sorted(SATELLITES_DIR.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / CONFIG_NAME).exists():
            found.append(p)
            continue
        docs = p / "documents"
        if docs.is_dir() and any(is_document_file(f) for f in docs.iterdir()):
            found.append(p)
    return found


def list_satellites() -> list:
    """Load every satellite found under satellites/."""
    return [load_satellite(p.name) for p in satellite_paths()]


def list_slugs() -> list:
    return [p.name for p in satellite_paths()]


def load_satellite(slug: str) -> Satellite:
    """
    Load one satellite by folder slug.

    The config file is optional. A folder holding documents is a valid
    satellite on its own and picks up the defaults, which keeps adding a
    new satellite down to: make the folder, copy the documents in.
    """
    root = SATELLITES_DIR / slug
    cfg  = root / CONFIG_NAME

    if not root.is_dir():
        known = ", ".join(list_slugs()) or "(none)"
        raise FileNotFoundError(
            f"No satellite folder at {root}\n"
            f"Known satellites: {known}\n"
            f"Create one with: python cli.py new {slug}\n"
            f"or just make the folder and put the documents in it."
        )

    data = {}
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}

    return Satellite(
        slug   = slug,
        name   = data.get("name") or prettify(slug),
        root   = root,
        config = data,
    )


def resolve(slugs) -> list:
    """
    Turn a CLI satellite argument into a list of Satellite objects.
    Accepts a list of slugs, or the string 'all'.
    """
    if slugs in ("all", ["all"], None):
        return list_satellites()
    if isinstance(slugs, str):
        slugs = [slugs]
    return [load_satellite(s) for s in slugs]


def create_satellite(slug: str, name: Optional[str] = None) -> Satellite:
    """
    Scaffold a satellite folder. Safe to re-run: an existing folder is
    left alone and simply returned.
    """
    slug = slugify(slug)
    root = SATELLITES_DIR / slug

    for sub in ("documents", "extracted", "vectorstore"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    if (root / CONFIG_NAME).exists():
        return load_satellite(slug)

    header = (
        f"# {name or prettify(slug)}\n"
        f"#\n"
        f"# This file is optional. Put the documents in documents/ and the\n"
        f"# type of each one is recognised from its name.\n"
        f"#\n"
        f"# To point a document type at a specific file, add:\n"
        f"#\n"
        f"# documents:\n"
        f"#   FCP:\n"
        f"#     file: my_procedures.pdf\n"
        f"\n"
    )

    template = {
        "name": name or prettify(slug),
        "slug": slug,
        "embedding": {"model": DEFAULT_MODEL},
        "chunking":  DEFAULT_CHUNKING,
        "retrieval": DEFAULT_RETRIEVAL,
    }

    (root / CONFIG_NAME).write_text(
        header + yaml.safe_dump(template, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return load_satellite(slug)
