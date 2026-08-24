"""
Offline embedding model registry.

Models are downloaded once into models/ and loaded from there afterwards,
so ingestion and querying never touch the network. Pick a model by its
short name; the HuggingFace id, dimension and any required query/passage
prefixes are looked up here.

Typical use:

    python cli.py prefetch minilm-l6      once, with network
    python cli.py models                  see what is cached

    from core.embeddings import load
    embedder = load("minilm-l6")
    vectors  = embedder.encode_passages(["some text"])
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Use the operating system trust store for HTTPS. On a campus or corporate
# network that inspects TLS, model downloads otherwise fail with
# "self-signed certificate in certificate chain". No effect if truststore
# is not installed.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from .registry import MODELS_DIR


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    dim: int
    note: str = ""
    # E5-family models score badly without these. Everything else uses "".
    query_prefix: str = ""
    passage_prefix: str = ""


# Short name -> spec. Add new models here; nothing else needs to change.
# Any HuggingFace id also works directly as a name, so a model that is not
# listed can still be tested without editing this file.
MODEL_REGISTRY = {
    # Measured best on the Aditya-L1 100-question bench
    "minilm-l6": ModelSpec(
        "sentence-transformers/all-MiniLM-L6-v2", 384,
        "fastest; best measured on this corpus",
    ),

    # English-specialised, the family worth trying first on technical text
    "bge-small-en": ModelSpec(
        "BAAI/bge-small-en-v1.5", 384, "English, small"),
    "bge-base-en": ModelSpec(
        "BAAI/bge-base-en-v1.5", 768, "English, mid-size"),
    "bge-large-en": ModelSpec(
        "BAAI/bge-large-en-v1.5", 1024, "English, strong"),

    "gte-base": ModelSpec(
        "thenlper/gte-base", 768, "English, mid-size"),
    "gte-large": ModelSpec(
        "thenlper/gte-large", 1024, "English, strong"),

    "mpnet-base": ModelSpec(
        "sentence-transformers/all-mpnet-base-v2", 768,
        "long-standing general-purpose baseline"),

    "mxbai-large": ModelSpec(
        "mixedbread-ai/mxbai-embed-large-v1", 1024, "English, strong"),

    "arctic-m": ModelSpec(
        "Snowflake/snowflake-arctic-embed-m", 768,
        "retrieval-tuned, mid-size"),

    # Multilingual. Weakest of the four originally tested on this corpus,
    # in dense mode by a wide margin: English technical text gains nothing
    # from multilingual training.
    "bge-m3": ModelSpec(
        "BAAI/bge-m3", 1024, "multilingual, long context, heavy"),

    # E5 needs its prefixes or it scores unfairly low. Handled here.
    "e5-base": ModelSpec(
        "intfloat/e5-base-v2", 768, "English; prefixes handled",
        query_prefix="query: ", passage_prefix="passage: "),
    "e5-large": ModelSpec(
        "intfloat/e5-large-v2", 1024, "English; prefixes handled",
        query_prefix="query: ", passage_prefix="passage: "),
    "e5-multi": ModelSpec(
        "intfloat/multilingual-e5-large", 1024,
        "multilingual; prefixes handled",
        query_prefix="query: ", passage_prefix="passage: "),

    # Ships custom code. Known to fail on some torch builds with a RoPE
    # IndexError; keep it last and expect it to be skipped sometimes.
    "gte-large-en": ModelSpec(
        "Alibaba-NLP/gte-large-en-v1.5", 1024,
        "needs trust_remote_code; breaks on some torch builds"),
    "nomic-v1.5": ModelSpec(
        "nomic-ai/nomic-embed-text-v1.5", 768,
        "needs trust_remote_code"),
}

DEFAULT_MODEL = "minilm-l6"

CPU_BATCH_SIZE = 16
MIN_BATCH_SIZE = 8


def needs_trust(hf_id: str) -> bool:
    """Models that ship their own modelling code on the Hub."""
    lower = hf_id.lower()
    return any(t in lower for t in
               ("gte-large-en-v1.5", "nomic", "new-impl", "jina", "alibaba-nlp"))


def spec_for(name: str) -> ModelSpec:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    # Allow a raw HuggingFace id as an escape hatch.
    if "/" in name:
        return ModelSpec(name, 0, "custom (not in registry)")
    known = ", ".join(MODEL_REGISTRY)
    raise KeyError(f"Unknown embedding model '{name}'. Known: {known}")


def cache_dir_for(name: str) -> Path:
    """Where a given model's files live inside models/."""
    return MODELS_DIR / spec_for(name).hf_id.replace("/", "__")


def is_cached(name: str) -> bool:
    d = cache_dir_for(name)
    if not d.exists():
        return False
    return any(d.rglob("*.safetensors")) or any(d.rglob("*.bin"))


def pick_batch_size(vram_gb: float, dim: int) -> int:
    """
    Choose a batch size that fits the card.

    A fixed 128 overflows a 4 GB laptop GPU as soon as the model is one of
    the 1024-dimension ones, and the failure shows up part-way through a
    long ingest rather than at the start. Scale by VRAM, then halve for
    the larger models.
    """
    if vram_gb >= 20:
        base = 128
    elif vram_gb >= 10:
        base = 64
    elif vram_gb >= 6:
        base = 48
    else:
        base = 32

    if dim >= 1024:
        base //= 2

    return max(MIN_BATCH_SIZE, base)


def prefetch(name: str) -> Path:
    """
    Download a model into models/. Needs network access; run once per
    model. Safe to re-run: an already-cached model is left alone.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("Missing dependency: pip install sentence-transformers")

    sp     = spec_for(name)
    target = cache_dir_for(name)

    if is_cached(name):
        print(f"  {name:<14} already cached at {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    print(f"  {name:<14} downloading {sp.hf_id} ...")

    kwargs = {}
    if needs_trust(sp.hf_id):
        kwargs["trust_remote_code"] = True

    model = SentenceTransformer(sp.hf_id, cache_folder=str(MODELS_DIR), **kwargs)
    model.save(str(target))
    try:
        dim = model.get_embedding_dimension()
    except AttributeError:
        dim = model.get_sentence_embedding_dimension()
    print(f"  {name:<14} cached  dim={dim}")
    return target


def cached_models() -> list:
    return [n for n in MODEL_REGISTRY if is_cached(n)]


class Embedder:
    """
    Thin wrapper over SentenceTransformer that pins the model offline,
    picks GPU when available, sizes the batch to the card, and applies
    per-model prefixes.
    """

    def __init__(self, name: str, device: Optional[str] = None):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit(
                "Missing dependency: pip install sentence-transformers torch"
            )

        self._torch = torch
        self.name   = name
        self.spec   = spec_for(name)

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            props           = torch.cuda.get_device_properties(0)
            self.gpu_name   = props.name
            self.vram_gb    = props.total_memory / 1e9
            self.batch_size = pick_batch_size(self.vram_gb, self.spec.dim)
        else:
            self.gpu_name   = ""
            self.vram_gb    = 0.0
            self.batch_size = CPU_BATCH_SIZE

        local  = cache_dir_for(name)
        kwargs = {}
        if needs_trust(self.spec.hf_id):
            kwargs["trust_remote_code"] = True

        if local.exists():
            # Fully offline path.
            self._model = SentenceTransformer(
                str(local), device=self.device, **kwargs)
        else:
            # Not prefetched. Falls back to the shared cache folder, which
            # still works offline once HuggingFace has it cached.
            self._model = SentenceTransformer(
                self.spec.hf_id, device=self.device,
                cache_folder=str(MODELS_DIR), **kwargs)

        if self.device == "cuda":
            self._model.half()

        try:
            self.dim = self._model.get_embedding_dimension()
        except AttributeError:
            self.dim = self._model.get_sentence_embedding_dimension()

    def describe(self) -> str:
        if self.device == "cuda":
            return (f"{self.name} ({self.spec.hf_id}) dim={self.dim} "
                    f"GPU {self.gpu_name} {self.vram_gb:.1f}GB fp16 "
                    f"batch={self.batch_size}")
        return (f"{self.name} ({self.spec.hf_id}) dim={self.dim} "
                f"CPU fp32 batch={self.batch_size}")

    def _encode(self, texts: list, prefix: str) -> list:
        if prefix:
            texts = [prefix + t for t in texts]

        # Halve the batch and retry if the card runs out of memory, rather
        # than losing a long ingest to a single oversized batch.
        batch = self.batch_size
        while True:
            try:
                vecs = self._model.encode(
                    texts,
                    batch_size           = batch,
                    show_progress_bar    = False,
                    normalize_embeddings = True,
                )
                return vecs.tolist()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if batch <= MIN_BATCH_SIZE:
                    raise
                if self.device == "cuda":
                    self._torch.cuda.empty_cache()
                batch = max(MIN_BATCH_SIZE, batch // 2)
                self.batch_size = batch
                print(f"    GPU out of memory, batch size -> {batch}")

    def encode_passages(self, texts: list) -> list:
        return self._encode(texts, self.spec.passage_prefix)

    def encode_query(self, text: str) -> list:
        return self._encode([text], self.spec.query_prefix)[0]


_loaded = {}


def load(name: str, device: Optional[str] = None) -> Embedder:
    """Load (and memoise) an embedder by short name."""
    key = (name, device)
    if key not in _loaded:
        _loaded[key] = Embedder(name, device=device)
    return _loaded[key]
