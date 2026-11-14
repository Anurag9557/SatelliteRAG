"""
Document extraction.

Reads the five source documents in satellites/<slug>/documents/ and writes
the structured JSON that ingestion consumes:

    extracted/chunks.json         text sections, procedures, index rows
    extracted/tch_records.json    one record per telecommand
    extracted/images/             every figure, for captioning

Extraction parses structure only. It does not decide chunk sizes: whole
sections are emitted here and core/chunking.py applies the size policy at
ingest time, so changing a chunk budget never requires re-extraction.

Each document type has its own reader, chosen from the type that
core/registry.py detected from the filename. A reader that recognises
nothing falls back to plain page or paragraph extraction rather than
returning nothing, because silently producing zero chunks is far worse
than producing rough ones.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    raise SystemExit("Missing dependency: pip install pymupdf")

try:
    from docx import Document as DocxDocument
except ImportError:
    raise SystemExit("Missing dependency: pip install python-docx")


# Windows rarely has these on PATH, so check the usual install locations.
SOFFICE_CANDIDATES = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice",
    "/usr/bin/libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
]

MAGICK_CANDIDATES = [
    r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
    r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
    "/usr/bin/magick",
    "/usr/bin/convert",
]

VECTOR_SUFFIXES = {".emf", ".wmf"}


def find_tool(names, candidates):
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def find_soffice():
    return find_tool(["soffice", "libreoffice"], SOFFICE_CANDIDATES)


def find_magick():
    return find_tool(["magick", "convert"], MAGICK_CANDIDATES)


# Subsystem tagging. These names are standard across ISRO spacecraft, so
# the same map serves every satellite.

SUBSYSTEM_KEYWORDS = [
    ("OBC",        r'\bOBC\b'),
    ("AOCS",       r'\bAOCS\b'),
    ("DTG",        r'\bDTG\b|\bGYRO\b|\bCSA\b'),
    ("SENSOR",     r'\bSTAR SENSOR\b'),
    ("WHEEL",      r'\bWHEEL\b|\bWDE\b|\bSUSPENSION\b'),
    ("MECHANISM",  r'\bMECHANISM\b|\bHGA\b|\bSPDM\b|\bMCE\b|\bMEPU\b'),
    ("PROPULSION", r'\bPROPULSION\b|\bLATCH VALVE\b|\bTHRUSTER\b|\bOXIDISER\b'),
    ("POWER",      r'\bFPGA\b|\bBATTERY\b|\bTCR\b|\bDC-DC\b'),
    ("TTC_XBAND",  r'\bTTC\b|\bX-BAND\b|\bTRANSPONDER\b|\bTWTA\b|\bXBS\b'),
    ("PAYLOAD",    r'\bPAPA\b|\bASPEX\b|\bSOLEXS?\b|\bHELIOS\b|\bSUIT\b'
                   r'|\bVELC\b|\bMAGNETOMETER\b|\bPAYLOAD\b'),
    ("ODHS",       r'\bODHS\b|\bSSR\b|\bDFU\b|\bBDH\b'),
]

# Fallback by procedure number range, used when no keyword matches.
SUBSYSTEM_RANGES = [
    (5000, 5099, "AOCS"),      (5200, 5299, "OBC"),
    (5400, 5499, "SENSOR"),    (5550, 5560, "WHEEL"),
    (5600, 5620, "DTG"),       (5650, 5665, "PROPULSION"),
    (5700, 5730, "POWER"),     (5950, 5980, "MECHANISM"),
    (6000, 6050, "TTC_XBAND"), (6100, 6220, "PAYLOAD"),
    (6221, 6240, "ODHS"),
]

CID_PREFIXES = {
    "OBC": "OBC", "CPU": "OBC", "DTG": "DTG", "WDE": "WHEEL",
    "PWR": "POWER", "MEC": "MECHANISM", "TTC": "TTC_XBAND",
    "PRO": "PROPULSION", "AOC": "AOCS", "PLD": "PAYLOAD",
    "ODH": "ODHS", "SEN": "SENSOR", "SS": "SENSOR",
}


def find_procedure_number(text: str) -> str:
    m = re.search(r'\b([56]\d{3})\b', text)
    return m.group(1) if m else ""


def infer_subsystem(text: str, number: str = "") -> str:
    for subsystem, pattern in SUBSYSTEM_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return subsystem
    if number:
        try:
            n = int(number)
            for lo, hi, sub in SUBSYSTEM_RANGES:
                if lo <= n <= hi:
                    return sub
        except ValueError:
            pass
    return "UNKNOWN"


def subsystem_from_cid(cid: str) -> str:
    upper = cid.upper()
    for prefix, subsystem in CID_PREFIXES.items():
        if upper.startswith(prefix):
            return subsystem
    return "UNKNOWN"


def classify(text: str) -> str:
    # Procedure markers are checked first. A procedure block keeps its
    # trailing notes page, so it contains "NOTES:" as well; testing for
    # notes first would label every procedure a note.
    if re.search(r'\bEND OF PROCEDURE\b|\bPROCEDURE\s*:', text, re.IGNORECASE):
        return "procedure"
    if re.search(r'\bNOTES?\s*:', text, re.IGNORECASE):
        return "note"
    return "section"


def make_chunk(chunk_id, text, doc_type, filename, content_type,
               page=0, heading="", number="", subsystem=""):
    return {
        "chunk_id":        chunk_id,
        "text":            text,
        "source_doc":      doc_type,
        "doc_filename":    filename,
        "content_type":    content_type,
        "page":            page,
        "section_heading": heading,
        "fcp_number":      number,
        "subsystem":       subsystem
                           or infer_subsystem(heading + " " + text, number),
        "images":          [],
    }


# Images

def convert_vector_image(path: Path) -> Path:
    """EMF and WMF are vector formats no vision model reads. Rasterise."""
    if path.suffix.lower() not in VECTOR_SUFFIXES:
        return path
    magick = find_magick()
    if not magick:
        return path
    png = path.with_suffix(".png")
    try:
        subprocess.run([magick, str(path), str(png)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=60)
        if png.exists():
            path.unlink(missing_ok=True)
            return png
    except Exception:
        pass
    return path


def save_images_from_page(doc, page, page_num, doc_type, images_dir) -> int:
    saved = 0
    for idx, info in enumerate(page.get_images(full=True)):
        try:
            base = doc.extract_image(info[0])
            name = f"{doc_type}_p{page_num}_img{idx}.{base['ext']}"
            path = images_dir / name
            path.write_bytes(base["image"])
            convert_vector_image(path)
            saved += 1
        except Exception:
            continue
    return saved


# Procedure document

# The subsystem column as it appears in the index, mapped to the canonical
# name. Taken from the document rather than inferred: the index states the
# owning subsystem, and inferring it from the title gets it wrong. "5010
# AOCS THRUSTER BLOCK SELECTION" is AOCS in the document, but the word
# "thruster" would infer PROPULSION.
TOC_SUBSYSTEM_MAP = {
    "OBC": "OBC", "SENSOR": "SENSOR", "POWER": "POWER",
    "TTC&XBAND": "TTC_XBAND", "TTCXBAND": "TTC_XBAND", "TTC": "TTC_XBAND",
    "DTG": "DTG", "WHEEL": "WHEEL", "MECHANISM": "MECHANISM",
    "PROPULSION": "PROPULSION", "AOCS": "AOCS",
    "PAYLOAD": "PAYLOAD", "PAYLAOD": "PAYLOAD",   # misspelled in the source
    "ODHS": "ODHS",
}

FCP_HEADER = re.compile(
    r'([A-Z][A-Z&\s]{1,20}?)\s*-?\s*([56]\d{3})\s*:\s*([^\n]{3,120})')

# "1.  5201  OBC ...", "63.  6006  TTC&XBAND ...". The \s+ spans the
# newline that sometimes separates the number from the subsystem.
FCP_TOC_ROW = re.compile(r'(\d{1,3})\s*\.\s+([56]\d{3})\s+([A-Z][A-Z&]{1,})')


def extract_fcp(path: Path, doc_type: str, images_dir: Path) -> list:
    """
    Two kinds of record come out of a procedure document: the index rows
    at the front, which answer "which procedure covers X", and the
    procedures themselves.
    """
    chunks = []
    doc    = fitz.open(str(path))
    pages  = [p.get_text("text") for p in doc]

    text_pages = sum(1 for t in pages if len(t.strip()) > 50)
    print(f"      {len(doc)} pages, {text_pages} with text")
    if text_pages < len(doc) * 0.5:
        print(f"      warning: {len(doc) - text_pages} pages have no text "
              f"layer and will be missed (scanned document)")

    for n, page in enumerate(doc, start=1):
        save_images_from_page(doc, page, n, doc_type, images_dir)

    # The index sits at the front of the file, before the first procedure
    # body. Scanning the whole document instead would pick up false rows
    # from step tables inside the procedures, so find where the bodies
    # start and read index rows only from the pages before that.
    body_probe = re.compile(r'\b[A-Z]+\s*-?\s*[56]\d{3}\s*:\s*.+')
    body_start = len(pages) + 1
    toc_pages  = []
    for i, text in enumerate(pages):
        if body_probe.search(text):
            body_start = i + 1
            break
        toc_pages.append(text)

    full  = "\n".join(toc_pages)
    rows  = list(FCP_TOC_ROW.finditer(full))
    n_toc = 0
    seen_ids = set()
    for i, m in enumerate(rows):
        end   = rows[i + 1].start() if i + 1 < len(rows) else len(full)
        title = re.sub(r'\s+', ' ', full[m.end():end]).strip(" .\n\t")
        if not title:
            continue
        serial, number, raw_sub = m.group(1), m.group(2), m.group(3).upper()
        subsystem = TOC_SUBSYSTEM_MAP.get(raw_sub, "UNKNOWN")

        # The serial disambiguates: the source reuses a few numbers
        # (6201 and 6232 each appear twice).
        chunk_id = f"{doc_type}_IDX_{int(serial):03d}_{number}"
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        chunks.append(make_chunk(
            chunk_id,
            f"FCP {number} - {subsystem} - {title}",
            doc_type, path.name, "toc_index",
            page=1, heading="FCP Index", number=number,
            subsystem=subsystem))
        n_toc += 1

    # Procedures. A new header opens a block; the block runs until the
    # next header, so notes after END OF PROCEDURE stay with their
    # procedure instead of becoming an orphan chunk.
    starts = []
    for n, text in enumerate(pages, start=1):
        if n < body_start:
            continue          # index pages, already handled above
        m = FCP_HEADER.search(text)
        if m:
            starts.append((n, m.group(2), m.group(0).strip()))

    n_proc = 0
    for i, (page_no, number, heading) in enumerate(starts):
        last = starts[i + 1][0] - 1 if i + 1 < len(starts) else len(pages)
        body = "\n".join(pages[page_no - 1:last]).strip()
        if not body:
            continue
        chunks.append(make_chunk(
            f"{doc_type}_p{page_no}_{number}_{i}", body,
            doc_type, path.name, classify(body),
            page=page_no, heading=heading, number=number))
        n_proc += 1

    # Nothing matched: fall back to one chunk per page.
    if not chunks:
        print(f"      no procedure headers matched, using page chunks")
        for n, text in enumerate(pages, start=1):
            if len(text.strip()) < 50:
                continue
            chunks.append(make_chunk(
                f"{doc_type}_p{n}", text.strip(), doc_type, path.name,
                classify(text), page=n))

    doc.close()
    print(f"      {n_toc} index rows (bodies begin p.{body_start}), "
          f"{n_proc} procedures")
    return chunks


# Prose PDF

HEADING = re.compile(
    r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][A-Z0-9\s\-/(),&]{5,})$', re.MULTILINE)


def extract_prose_pdf(path: Path, doc_type: str, images_dir: Path) -> list:
    """Numbered-heading documents: emit whole sections, unsplit."""
    chunks = []
    doc    = fitz.open(str(path))

    full, offsets = "", []
    for n, page in enumerate(doc, start=1):
        offsets.append((len(full), n))
        full += "\n" + page.get_text("text")
        save_images_from_page(doc, page, n, doc_type, images_dir)

    matches = list(HEADING.finditer(full))
    for i, m in enumerate(matches):
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        heading = m.group(0).strip()
        title   = heading.split(maxsplit=1)[1] if " " in heading else ""

        if ("|" in heading or "P a g e" in heading
                or len(title) < 5 or title.replace(" ", "").isdigit()):
            continue

        body = full[m.start():end].strip()
        if len(body) < 40:
            continue

        # The page is the LAST offset at or before the match. next() would
        # return the first, and offsets[0] is (0, page 1), so every section
        # in the document would be stamped page 1.
        page_no = 1
        for off, p in offsets:
            if off <= m.start():
                page_no = p
            else:
                break

        chunks.append(make_chunk(
            f"{doc_type}_s{i}", body, doc_type, path.name, "section",
            page=page_no, heading=heading))

    if not chunks:
        print(f"      no headings matched, using page chunks")
        for n, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if len(text) < 50:
                continue
            chunks.append(make_chunk(
                f"{doc_type}_p{n}", text, doc_type, path.name, "section",
                page=n))

    doc.close()
    print(f"      {len(chunks)} sections")
    return chunks


# Word documents

def extract_docx(path: Path, doc_type: str, images_dir: Path) -> list:
    chunks   = []
    document = DocxDocument(str(path))

    heading, buffer, index = "Preamble", [], 0

    def flush():
        nonlocal buffer, index
        body   = "\n".join(buffer).strip()
        buffer = []
        if not body:
            return
        number = find_procedure_number(heading + " " + body)
        chunks.append(make_chunk(
            f"{doc_type}_{index:04d}", body, doc_type, path.name, "section",
            heading=heading, number=number))
        index += 1

    for para in document.paragraphs:
        style = ""
        try:
            style = (para.style.name or "") if para.style is not None else ""
        except Exception:
            style = ""
        if style.startswith("Heading"):
            flush()
            heading = para.text.strip() or heading
        elif para.text.strip():
            buffer.append(para.text.strip())
    flush()

    # Tables carry a lot of the operational content in these documents.
    for t_idx, table in enumerate(document.tables):
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if len(rows) < 2:
            continue
        chunks.append(make_chunk(
            f"{doc_type}_table_{t_idx:03d}", "\n".join(rows), doc_type,
            path.name, "section", heading=f"Table {t_idx + 1}"))

    saved = 0
    for rel in document.part.rels.values():
        if "image" not in rel.target_ref:
            continue
        try:
            ext  = rel.target_ref.rsplit(".", 1)[-1].lower()
            out  = images_dir / f"{doc_type}_img{saved:03d}.{ext}"
            out.write_bytes(rel.target_part.blob)
            convert_vector_image(out)
            saved += 1
        except Exception:
            continue

    print(f"      {len(chunks)} sections, {saved} images")
    return chunks


def extract_doc(path: Path, doc_type: str, images_dir: Path,
                work_dir: Path) -> list:
    """Legacy .doc: convert with LibreOffice, then read as .docx."""
    converted = work_dir / (path.stem + "_converted.docx")

    if not converted.exists():
        soffice = find_soffice()
        if not soffice:
            print(f"      LibreOffice not found, skipping {path.name}")
            print(f"      install it, or save the file as .docx by hand")
            return []
        print(f"      converting with LibreOffice")
        subprocess.run(
            [soffice, "--headless", "--convert-to", "docx",
             "--outdir", str(work_dir), str(path)],
            capture_output=True, text=True, timeout=180)
        produced = work_dir / (path.stem + ".docx")
        if not produced.exists():
            print(f"      conversion failed")
            return []
        produced.rename(converted)

    return extract_docx(converted, doc_type, images_dir)


# Telecommand log

TCH_ROW = re.compile(r'(?m)^[ \t]{0,4}(\d{3,6})[ \t]*\|')
# Where a page ends: the rule under its last row, or failing that the
# column header of the next page.
TCH_PAGE_BREAK = re.compile(
    r'(?m)^[ \t]*(?:[=\-]{40,}[ \t]*\r?$|SL\.NO[ \t]*\|)')


def extract_tch(path: Path, doc_type: str) -> list:
    """
    Pipe-delimited telecommand history.

    Rows are found by their leading serial-number anchor rather than by
    physical line, because binary bytes in the data column introduce
    newlines that split a row across two lines and drop it.

    Columns: SL.NO | CODE | CID | MNEMONIC | TIME | DATA | DATA VALUE |
             ORBIT | STATUS | SRC | DST
    """
    raw     = path.read_bytes().decode("latin-1")
    starts  = list(TCH_ROW.finditer(raw))
    records = []

    for i, m in enumerate(starts):
        end   = starts[i + 1].start() if i + 1 < len(starts) else len(raw)
        block = raw[m.start():end]
        # The log is paginated every 30 rows: a footer, a form feed, a
        # title and the column header again. Left in, all of that rides
        # on the end of the row above it and the header's own cells
        # (STATUS, SRCSTN, DSTSTN) get read as that row's values.
        page_break = TCH_PAGE_BREAK.search(block)
        if page_break:
            block = block[:page_break.start()]
        parts = [p.strip() for p in block.split("|")]
        # Every row closes with a pipe, which leaves an empty last cell.
        # Counting columns from the end without dropping it puts each
        # trailing field one place out: the destination station came out
        # blank and the status column held the source station.
        if parts and parts[-1] == "":
            parts.pop()
        if len(parts) < 11:
            continue

        # Fixed columns are taken from each end, so extra pipe characters
        # inside the data column cannot shift everything along.
        sl_no, code, cid, mnemonic, stamp = (parts[0], parts[1], parts[2],
                                             parts[3], parts[4])
        dst, src, status, orbit, data_val = (parts[-1], parts[-2], parts[-3],
                                             parts[-4], parts[-5])
        data = "|".join(parts[5:-5])

        stamp = re.sub(r'\s+', ' ', stamp).strip()
        dm = re.match(r'(\d{4})\D(\d{2})\D(\d{2})\s+'
                      r'(\d{2}:\d{2}:\d{2})(?::(\d{1,3}))?', stamp)
        if dm:
            date = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
            time = dm.group(4)
            ms   = dm.group(5) or ""
        else:
            date = time = ms = ""

        data = re.sub(r'[^\x20-\x7e]',
                      lambda b: f'\\x{ord(b.group()):02x}', data)
        subsystem = subsystem_from_cid(cid)

        records.append({
            "record_id":      f"{doc_type}_{sl_no}",
            "source_doc":     doc_type,
            "doc_filename":   path.name,
            "content_type":   "command_log",
            "sl_no":          sl_no,
            "code":           code,
            "cid":            cid,
            "mnemonic":       mnemonic,
            "command_time":   stamp,
            "date":           date,
            "time":           time,
            "milliseconds":   ms,
            "command_data":   data[:400],
            "data_cmd_value": data_val,
            "orbit_no":       orbit,
            "status":         status,
            "source_station": src,
            "dest_station":   dst,
            "subsystem":      subsystem,
            "images":         [],
            "text": (f"Command {mnemonic} (CID: {cid}) uplinked on {date} "
                     f"at {time} from {src} to {dst}. Orbit {orbit}. "
                     f"Status {status}. Subsystem {subsystem}."),
        })

    print(f"      {len(records)} command records")
    return records


# Orchestration

def extract(satellite, force: bool = False) -> dict:
    """Read every document this satellite has and write the JSON."""
    print(f"\n[{satellite.slug}] extract")

    documents = satellite.documents
    if not documents:
        print(f"    no documents in {satellite.documents_dir}")
        return {"satellite": satellite.slug, "status": "no-documents"}

    if satellite.is_extracted() and not force:
        print(f"    already extracted, use --force to redo")
        return {"satellite": satellite.slug, "status": "cached"}

    satellite.extracted_dir.mkdir(parents=True, exist_ok=True)
    satellite.images_dir.mkdir(parents=True, exist_ok=True)

    chunks, commands = [], []

    for doc_type in ["FCP", "MOR", "OPS_MANUAL", "GUIDELINES", "HANDBOOK", "TCH"]:
        entry = documents.get(doc_type)
        if not entry:
            continue
        path = entry["path"]
        if not path.exists():
            print(f"    {doc_type:<12} {entry['file']}  NOT FOUND")
            continue

        print(f"    {doc_type:<12} {entry['file']}")
        try:
            if doc_type == "TCH":
                commands += extract_tch(path, doc_type)
            elif doc_type == "FCP" and path.suffix.lower() == ".pdf":
                chunks += extract_fcp(path, doc_type, satellite.images_dir)
            elif path.suffix.lower() == ".pdf":
                chunks += extract_prose_pdf(path, doc_type,
                                            satellite.images_dir)
            elif path.suffix.lower() == ".docx":
                chunks += extract_docx(path, doc_type, satellite.images_dir)
            elif path.suffix.lower() == ".doc":
                chunks += extract_doc(path, doc_type, satellite.images_dir,
                                      satellite.extracted_dir)
            else:
                print(f"      unsupported format")
        except Exception as exc:
            print(f"      FAILED: {exc}")

    (satellite.extracted_dir / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    (satellite.extracted_dir / "tch_records.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=False), encoding="utf-8")

    images = len(list(satellite.images_dir.glob("*")))
    by_doc = {}
    for c in chunks:
        by_doc[c["source_doc"]] = by_doc.get(c["source_doc"], 0) + 1

    print(f"    {len(chunks)} chunks {by_doc}")
    print(f"    {len(commands)} commands, {images} images")

    return {
        "satellite": satellite.slug, "status": "ok",
        "chunks": len(chunks), "commands": len(commands), "images": images,
    }


def extract_many(satellites: list, force: bool = False) -> list:
    out = []
    for sat in satellites:
        try:
            out.append(extract(sat, force=force))
        except Exception as exc:
            print(f"    FAILED: {exc}")
            out.append({"satellite": sat.slug, "status": f"error: {exc}"})
    return out
