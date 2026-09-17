"""
Cross-encoder reranking.

Hybrid search is a recall stage: it casts a wide net cheaply, but its
ordering comes from fusion arithmetic rather than from reading anything.
A cross-encoder reads the question and the passage together and scores
the pair directly, which is why it fixes the case where the right
passage is retrieved but sits several places down.

Measured on the 100-question Aditya-L1 bench:

    baseline                  Hit@1 0.65   MRR 0.744
    ms-marco-MiniLM-L-6-v2    Hit@1 0.82   MRR 0.857    200 s
    ms-marco-MiniLM-L-12-v2   Hit@1 0.83   MRR 0.860    446 s

L-6 is the default: it matches L-12 within noise for less than half the
time. The larger BGE rerankers cost 40-60 minutes per run on CPU and did
not pay for themselves.

Reranking is optional. Turn it off per satellite in satellite.yaml:

    retrieval:
      rerank: false
"""

from typing import Optional

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates to rerank. Below about 20 the reranker has too
# little to work with; well above it, latency grows for no gain.
DEFAULT_POOL = 20

# How much of the passage the cross-encoder reads. 512 tokens covers a
# chunk at the sizes this pipeline produces.
MAX_LENGTH = 512

_loaded = {}


def available() -> bool:
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        return True
    except ImportError:
        return False


def load(name: str = DEFAULT_MODEL, device: Optional[str] = None):
    """Load and memoise a cross-encoder. Returns None if unavailable."""
    if name in _loaded:
        return _loaded[name]

    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError:
        _loaded[name] = None
        return None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from .registry import MODELS_DIR
        try:
            # Local-only first so a cached model never touches the network.
            model = CrossEncoder(name, device=device, max_length=MAX_LENGTH,
                                 cache_folder=str(MODELS_DIR),
                                 local_files_only=True)
        except Exception:
            model = CrossEncoder(name, device=device, max_length=MAX_LENGTH,
                                 cache_folder=str(MODELS_DIR))
    except TypeError:
        # Older sentence-transformers has no cache_folder on CrossEncoder.
        model = CrossEncoder(name, device=device, max_length=MAX_LENGTH)
    except Exception as exc:
        print(f"    reranker unavailable ({type(exc).__name__}), "
              f"using fusion order")
        model = None

    _loaded[name] = model
    return model


def rerank(query: str, hits: list, top_k: int,
           model_name: str = DEFAULT_MODEL,
           device: Optional[str] = None) -> list:
    """
    Re-score hits by reading each against the query.

    Returns the fusion order untouched if the reranker cannot be loaded,
    so retrieval never depends on it being installed.
    """
    if len(hits) < 2:
        return hits[:top_k]

    model = load(model_name, device=device)
    if model is None:
        return hits[:top_k]

    pairs = [(query, h.text[:2000]) for h in hits]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        print(f"    rerank failed ({type(exc).__name__}), using fusion order")
        return hits[:top_k]

    for hit, score in zip(hits, scores):
        hit.score = float(score)

    return sorted(hits, key=lambda h: -h.score)[:top_k]


def prefetch(name: str = DEFAULT_MODEL) -> bool:
    """Download the reranker once so later runs are offline."""
    if not available():
        print(f"  reranker needs: pip install sentence-transformers")
        return False
    print(f"  reranker       downloading {name} ...")
    model = load(name)
    if model is None:
        print(f"  reranker       FAILED")
        return False
    print(f"  reranker       cached")
    return True
