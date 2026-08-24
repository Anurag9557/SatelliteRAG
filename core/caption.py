"""
Image captioning.

Every figure pulled out during extraction is described by a local vision
model, and the description is stored as a searchable chunk. That is what
makes a request like "show me the recovery flow diagram" reach an image:
the diagram itself is not searchable, its description is.

Runs against Ollama on localhost, so nothing leaves the machine.

Captions are written to extracted/image_captions.json, and progress to
extracted/.captioning_state.json, so an interrupted run resumes instead
of describing the same images twice.
"""

import base64
import bisect
import json
import re
from pathlib import Path

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Pillow is optional. With it, figures are checked by pixel dimensions as
# well as file size and normalised to PNG, which avoids progressive JPEG
# and unusual colour spaces reaching the model. Without it, only the file
# size check applies.
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


DEFAULT_MODEL = "qwen2.5vl"

# Vision models need room for the image tokens plus the prompt. The
# Ollama default of 4096 is too small for a full-page figure and the
# request fails outright rather than degrading.
NUM_CTX = 8192

# Figures below this are rules, bullets and spacer graphics, not content.
MIN_BYTES = 5000

# A figure thinner than this in either direction is a rule or a bullet,
# whatever its file size.
MIN_PIXELS = 60

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

CAPTIONS_FILE = "image_captions.json"
STATE_FILE    = ".captioning_state.json"

PROMPT = """You are a technical documentation analyst for a spacecraft mission.

Context from the source document:
{context}

Write a technical caption for this figure, optimised for semantic search
and retrieval by an operations engineer.

Include:
1. Figure type: block diagram, orbit plot, telemetry chart, command
   table, flow chart, schematic, photograph.
2. The mission phase or operation shown.
3. The subsystem involved, if identifiable.
4. Labels, identifiers, CID codes, command mnemonics, parameter names
   and numerical values that are legible.
5. What the figure is operationally for.

Requirements:
- Use spacecraft operations terminology.
- Reproduce technical terms, acronyms and codes exactly as shown.
- Be factual. Do not speculate beyond the figure and the context.
- At most 5 sentences.
- Write the caption only. Do not begin with "Caption:", "The image
  shows" or "This figure". Start directly with the content."""

# The model sometimes ignores the instruction above.
PREFIXES = [
    r'^Image\s+Caption\s*:?\s*', r'^Caption\s*:?\s*',
    r'^Figure\s+\d+\s*[.:]\s*',  r'^Fig\.?\s*\d+\s*[.:]\s*',
    r'^(This|The)\s+(image|figure|diagram)\s+(shows|is|depicts)\s+',
]

FILENAME = re.compile(r'^([A-Z][A-Z_]*)_(?:p(\d+)_)?img(\d+)', re.IGNORECASE)


def clean(text: str) -> str:
    text = text.strip()
    for pattern in PREFIXES:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    return text[:1].upper() + text[1:] if text else text


def parse_filename(name: str) -> tuple:
    """MOR_p240_img1.jpeg -> ('MOR', 240). GUIDELINES_img007.png -> (..., 0)."""
    m = FILENAME.match(name)
    if not m:
        return "UNKNOWN", 0
    return m.group(1).upper(), int(m.group(2) or 0)


def page_context(chunk, limit: int = 700) -> str:
    """
    Context handed to the vision model: what the document says around the
    figure. A plot with unlabelled axes is described far better when the
    surrounding prose says what is being plotted, and the section heading
    and procedure number let the model name the right subsystem.
    """
    if not chunk:
        return "(no surrounding context available)"

    lines = []
    for label, key in (("Document ", "doc_filename"),
                       ("Section  ", "section_heading"),
                       ("FCP      ", "fcp_number"),
                       ("Subsystem", "subsystem")):
        value = chunk.get(key)
        if value and value != "UNKNOWN":
            lines.append(f"{label}: {value}")

    body = re.sub(r'\s+', ' ', chunk.get("text", "")).strip()[:limit]
    if body:
        lines.append(f"\nText near the figure:\n{body}")

    return "\n".join(lines) if lines else "(no surrounding context available)"


def build_page_index(satellite) -> dict:
    """
    {doc_type: [(page, chunk), ...]} sorted by page, for nearest lookup.

    Whole chunks are kept, not just their text, because a caption
    inherits the section heading, procedure number and subsystem of the
    passage it sits next to. Inferring those from the caption wording
    alone is far less reliable than reading them off the document.
    """
    path = satellite.extracted_dir / "chunks.json"
    if not path.exists():
        return {}
    index = {}
    for chunk in json.loads(path.read_text(encoding="utf-8")):
        key  = chunk.get("source_doc", "")
        page = chunk.get("page") or 0
        if key and page:
            index.setdefault(key, []).append((page, chunk))
    for key in index:
        index[key].sort(key=lambda t: t[0])
    return index


def nearest_chunk(page_index: dict, doc_type: str, page: int):
    """
    The chunk whose page is closest to this one.

    An exact page match fails often: a figure frequently sits on a page
    whose text went into a chunk starting a page or two earlier, and a
    strict match then yields no context at all.
    """
    entries = page_index.get(doc_type)
    if not entries:
        return None
    if not page:
        return entries[0][1]
    pages = [p for p, _ in entries]
    idx   = bisect.bisect_left(pages, page)
    idx   = max(0, min(idx, len(entries) - 1))
    return entries[idx][1]


def load_state(satellite) -> dict:
    path = satellite.extracted_dir / STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(satellite, state: dict) -> None:
    (satellite.extracted_dir / STATE_FILE).write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


class SkipImage(Exception):
    """The figure is not worth captioning. Not an error."""


def encode_image(path: Path) -> str:
    """
    Validate and base64-encode a figure.

    With Pillow available the image is checked by pixel dimensions and
    re-saved as PNG, so progressive JPEGs and unusual colour spaces do
    not reach the model in a form it handles badly.
    """
    if not HAS_PIL:
        return base64.b64encode(path.read_bytes()).decode()

    import io
    try:
        img  = PILImage.open(path)
        w, h = img.size
    except Exception as exc:
        raise SkipImage(f"could not open image: {exc}")

    if w < MIN_PIXELS or h < MIN_PIXELS:
        raise SkipImage(f"too small ({w}x{h} px)")

    buf = io.BytesIO()
    try:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        img.save(buf, format="PNG")
    except Exception as exc:
        raise SkipImage(f"could not normalise to PNG: {exc}")

    return base64.b64encode(buf.getvalue()).decode()


def describe(path: Path, context: str, model: str) -> str:
    reply = ollama.generate(
        model   = model,
        prompt  = PROMPT.format(context=context),
        images  = [encode_image(path)],
        options = {"num_ctx": NUM_CTX, "temperature": 0.2},
    )
    return clean(reply.get("response", ""))


def caption(satellite, model: str = DEFAULT_MODEL, force: bool = False,
            min_bytes: int = MIN_BYTES, limit: int = 0) -> dict:
    """Describe every figure this satellite has that is not done yet."""
    print(f"\n[{satellite.slug}] caption")

    if not HAS_OLLAMA:
        print(f"    ollama package missing: pip install ollama")
        return {"satellite": satellite.slug, "status": "no-ollama"}

    if not satellite.images_dir.exists():
        print(f"    no images at {satellite.images_dir}, run extract first")
        return {"satellite": satellite.slug, "status": "no-images"}

    try:
        ollama.list()
    except Exception as exc:
        print(f"    cannot reach Ollama: {exc}")
        print(f"    start it, then: ollama pull {model}")
        return {"satellite": satellite.slug, "status": "ollama-unreachable"}

    images = sorted(p for p in satellite.images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not images:
        print(f"    no usable images found")
        return {"satellite": satellite.slug, "status": "no-images"}

    state = {} if force else load_state(satellite)
    pages = build_page_index(satellite)

    todo  = [p for p in images
             if p.name not in state and p.stat().st_size >= min_bytes]
    small = sum(1 for p in images if p.stat().st_size < min_bytes)

    print(f"    model {model}, {len(images)} images "
          f"({len(state)} done, {small} too small, {len(todo)} to do)")

    if limit:
        todo = todo[:limit]

    done, failed, skipped = 0, 0, 0
    for i, path in enumerate(todo, start=1):
        doc_type, page = parse_filename(path.name)
        chunk          = nearest_chunk(pages, doc_type, page)
        context        = page_context(chunk)
        try:
            text = describe(path, context, model)
            if not text:
                raise ValueError("empty response")
            # Inherit the document's own metadata rather than guessing it
            # from the caption wording.
            near = chunk or {}
            state[path.name] = {
                "caption":         text,
                "doc_type":        doc_type,
                "page":            page,
                "doc_filename":    near.get("doc_filename", ""),
                "section_heading": near.get("section_heading", ""),
                "fcp_number":      near.get("fcp_number", ""),
                "subsystem":       near.get("subsystem", "UNKNOWN"),
            }
            done += 1
            print(f"    [{i}/{len(todo)}] {path.name}")
            print(f"        {text[:90]}")
        except SkipImage as exc:
            # Record the skip so it is not retried on every run.
            skipped += 1
            state[path.name] = {"caption": "", "skipped": str(exc),
                                "doc_type": doc_type, "page": page}
            print(f"    [{i}/{len(todo)}] {path.name}  skipped: {exc}")
        except Exception as exc:
            failed += 1
            print(f"    [{i}/{len(todo)}] {path.name}  FAILED: "
                  f"{str(exc)[:110]}")

        if done and done % 10 == 0:
            save_state(satellite, state)

    save_state(satellite, state)

    captions = []
    for name, entry in sorted(state.items()):
        text = entry.get("caption", "")
        if not text:
            continue
        doc_type = entry.get("doc_type", "UNKNOWN")
        page     = entry.get("page", 0)

        # State written by an older run has no inherited fields. Fall back
        # to the nearest chunk so a re-caption is not needed just to pick
        # the metadata up.
        if "subsystem" not in entry:
            near = nearest_chunk(pages, doc_type, page) or {}
        else:
            near = entry

        captions.append({
            "chunk_id":        f"{satellite.slug}_cap_{Path(name).stem}",
            "text":            text,
            "source_doc":      doc_type,
            "doc_filename":    near.get("doc_filename", name),
            "content_type":    "image_caption",
            "page":            page,
            "section_heading": near.get("section_heading", ""),
            "fcp_number":      near.get("fcp_number", ""),
            "subsystem":       near.get("subsystem", "UNKNOWN"),
            "image_file":      str(satellite.images_dir / name),
            "images":          [],
        })

    (satellite.extracted_dir / CAPTIONS_FILE).write_text(
        json.dumps(captions, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"    {done} captioned this run, {skipped} skipped, "
          f"{failed} failed, {len(captions)} total")

    return {
        "satellite": satellite.slug, "status": "ok",
        "captioned": done, "failed": failed, "total": len(captions),
    }


def caption_many(satellites: list, model: str = DEFAULT_MODEL,
                 force: bool = False, min_bytes: int = MIN_BYTES,
                 limit: int = 0) -> list:
    out = []
    for sat in satellites:
        try:
            out.append(caption(sat, model=model, force=force,
                               min_bytes=min_bytes, limit=limit))
        except Exception as exc:
            print(f"    FAILED: {exc}")
            out.append({"satellite": sat.slug, "status": f"error: {exc}"})
    return out
