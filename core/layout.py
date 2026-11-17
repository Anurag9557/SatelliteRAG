"""
Structure-aware readers.

The readers in core/extract.py turn a document into sections of plain
text. That drops three things the documents themselves state:

    where a table is, and where each of its rows ends
    which headings a section sits under
    which lines are page furniture and not content

The readers here keep them. A section comes out as an ordered list of
blocks, and core/chunking.py then cuts between blocks instead of through
them:

    {"kind": "text",  "text": "one paragraph"}
    {"kind": "table", "title": "...", "header": "a | b", "rows": ["1 | 2"]}

Every section still carries a plain "text" rendering of its blocks, so
anything that reads chunks.json without knowing about blocks keeps working.

Used when a satellite sets parsing.mode: structure. Two readers:

    read_docx          Word files, walked in document order so a table
                       stays under the heading it was written under
    read_outline_pdf   report PDFs that carry bookmarks; sections come
                       from the bookmarks, not from guessing at headings

The procedure document is not read here. Its step tables come out of the
PDF with each column's text at its own height, so the rows cannot be put
back together from positions; core/extract.py still reads it.
"""

import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from . import extract as ex


# Blocks

def text_block(text: str) -> dict:
    return {"kind": "text", "text": text}


def table_block(rows: list, title: str = "") -> dict:
    """
    rows is a list of rows, each a list of cell strings. The first row is
    kept apart as the header, so a table cut into several chunks can carry
    it on every one. Columns that are empty all the way down are dropped;
    the PDF table finder invents a good many of those.

    Something with one row, or one column, is not a table for our purposes
    and comes back as a text block.
    """
    cleaned = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", c or "").strip() for c in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return text_block("")

    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    keep  = [j for j in range(width) if any(r[j] for r in cleaned)]
    lines = [" | ".join(r[j] for j in keep) for r in cleaned]

    if len(lines) < 2 or len(keep) < 2:
        body = "\n".join(lines)
        return text_block(f"{title}\n{body}" if title else body)
    return {"kind": "table", "title": title,
            "header": lines[0], "rows": lines[1:]}


def render(blocks: list) -> str:
    """The plain-text form of a section."""
    out = []
    for block in blocks:
        if block["kind"] == "table":
            out += [x for x in (block.get("title"), block["header"]) if x]
            out += block["rows"]
        elif block["text"]:
            out.append(block["text"])
    return "\n".join(out)


def heading_path(stack: list) -> str:
    return " > ".join(title for _, title in stack)


def push_heading(stack: list, level: int, title: str) -> None:
    """Keep the stack as the chain of headings above the current one."""
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


# Word

def heading_level(style: str) -> int:
    """'Heading 3' -> 3. A heading style with no number counts as level 1."""
    m = re.search(r"(\d+)\s*$", style)
    return int(m.group(1)) if m else 1


def style_name(para) -> str:
    try:
        return (para.style.name or "") if para.style is not None else ""
    except Exception:
        return ""


def docx_rows(table) -> list:
    """
    Cell text row by row. python-docx hands back a merged cell once for
    every column it spans, so the same text would be read two or three
    times over; each underlying cell is taken once.
    """
    rows = []
    for row in table.rows:
        cells, seen = [], []
        for cell in row.cells:
            if any(cell._tc is tc for tc in seen):
                continue
            seen.append(cell._tc)
            cells.append(cell.text)
        rows.append(cells)
    return rows


def read_docx(path: Path, doc_type: str, images_dir) -> list:
    """
    A Word file as sections of blocks, in the order the document has them.

    extract_docx reads every paragraph first and every table afterwards,
    so a table ends up as a chunk of its own under the heading "Table N",
    with nothing to say what it is a table of. Walking the body in order
    leaves it in its section, after the sentence that introduces it.
    """
    document = DocxDocument(str(path))
    chunks, stack = [], []
    heading, blocks, caption = "Preamble", [], ""

    def flush():
        nonlocal blocks
        body, blocks = blocks, []
        body = [b for b in body if b["kind"] == "table" or b["text"]]
        if not body:
            return
        # The procedure number is looked for in the prose only, as the
        # flat reader does. A table of command codes is full of four-digit
        # numbers that are not procedure numbers.
        prose  = " ".join(b["text"] for b in body if b["kind"] == "text")
        number = ex.find_procedure_number(heading + " " + prose)
        chunk  = ex.make_chunk(
            f"{doc_type}_{len(chunks):04d}", render(body), doc_type,
            path.name, "section", heading=heading, number=number)
        chunk["heading_path"] = heading_path(stack) or heading
        chunk["blocks"]       = body
        chunks.append(chunk)

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_Tbl):
            # A caption straight above a table is that table's title.
            blocks.append(table_block(docx_rows(Table(child, document)),
                                      title=caption))
            caption = ""
            continue
        if not isinstance(child, CT_P):
            continue

        para  = Paragraph(child, document)
        style = style_name(para)
        text  = para.text.strip()

        if caption:                      # a caption with no table under it
            blocks.append(text_block(caption))
            caption = ""

        if style.startswith("Heading"):
            flush()
            if text:
                heading = text
                push_heading(stack, heading_level(style), text)
        elif style == "Caption" and text:
            caption = text
        elif text:
            blocks.append(text_block(text))

    if caption:
        blocks.append(text_block(caption))
    flush()

    saved = ex.save_docx_images(document, doc_type, images_dir)
    tables = sum(1 for c in chunks for b in c["blocks"] if b["kind"] == "table")
    print(f"      {len(chunks)} sections holding {tables} tables, "
          f"{saved} images")
    return chunks


# Report PDF

# Fewer bookmarks than this and the outline is not a table of contents
# worth trusting; the caller falls back to finding headings by pattern.
MIN_OUTLINE = 5

# Running headers and footers sit in these bands of the page.
TOP_BAND, BOTTOM_BAND = 0.10, 0.90

NUMBER_PREFIX = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")
TOC_LEADER    = re.compile(r"\.{4,}\s*\d+\s*$")
LIST_MARKER   = re.compile(r"^(?:[-•▪●*]|\(?\w{1,3}[.)])\s+")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _shape(text: str) -> str:
    """A line with its numbers blanked, so 'Page 3' and 'Page 4' agree."""
    return re.sub(r"\d+", "9", re.sub(r"\s+", " ", text)).strip()


def unwrap(text: str) -> str:
    """
    Undo the line breaks a PDF puts at the right margin.

    A long line that does not end a sentence runs on into the next one.
    Short lines are left alone: they are list items, labels or cells, and
    joining them would turn a list into one unreadable sentence.
    """
    out = []
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if (out and len(out[-1]) >= 45
                and not out[-1].endswith((".", ":", ";", "?", "!"))
                and not LIST_MARKER.match(line)):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def page_blocks(doc, doc_type: str, images_dir) -> list:
    """
    Every page as blocks in reading order: [(page number, block)].

    Two things are taken out on the way. Headers and footers: a block in
    the top or bottom band whose wording, numbers aside, turns up on many
    pages. And the loose text of any table, which is replaced by the table
    read row by row.

    Text blocks are left as the PDF gives them ("raw"); the caller unwraps
    them once it knows what they are.
    """
    pages, seen = [], Counter()
    for page in doc:
        height, found = page.rect.height, []
        for x0, y0, x1, y1, text, _, kind in page.get_text("blocks", sort=True):
            if kind != 0 or not text.strip():
                continue
            in_band = y1 < TOP_BAND * height or y0 > BOTTOM_BAND * height
            found.append((x0, y0, x1, y1, text, _shape(text) if in_band else ""))
        for shape in {f[5] for f in found if f[5]}:
            seen[shape] += 1
        pages.append(found)

    limit     = max(5, len(pages) // 20)
    furniture = {shape for shape, n in seen.items() if n >= limit}

    out, n_tables, n_furniture = [], 0, 0
    for number, (page, found) in enumerate(zip(doc, pages), start=1):
        if images_dir is not None:
            ex.save_images_from_page(doc, page, number, doc_type, images_dir)

        height = page.rect.height
        tables = []
        try:
            candidates = page.find_tables().tables
        except Exception:
            candidates = []
        for table in candidates:
            # The document's header is itself a ruled box on every page.
            if (table.bbox[3] < (TOP_BAND + 0.04) * height
                    or table.bbox[1] > (BOTTOM_BAND - 0.02) * height):
                continue
            block = table_block(table.extract())
            if block["kind"] == "table":
                tables.append((table.bbox, block))

        placed = []
        for x0, y0, x1, y1, text, shape in found:
            if shape in furniture:
                n_furniture += 1
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
                   for b, _ in tables):
                continue
            placed.append((y0, x0, {"kind": "text", "raw": text.strip()}))
        for bbox, block in tables:
            placed.append((bbox[1], bbox[0], block))
            n_tables += 1

        placed.sort(key=lambda item: (item[0], item[1]))
        out += [(number, block) for _, _, block in placed]

    print(f"      {n_tables} tables read by row, "
          f"{n_furniture} header and footer lines removed")
    return join_split_tables(out)


def join_split_tables(seq: list) -> list:
    """
    A table that runs over a page break arrives as two. If the second
    starts on the next page straight after the first, with the same number
    of columns, it is the same table and its rows are added to the first.
    """
    out = []
    for page, block in seq:
        last_page, last = out[-1] if out else (0, None)
        if (last is not None and block["kind"] == "table"
                and last["kind"] == "table" and page == last_page + 1
                and block["header"].count(" | ") == last["header"].count(" | ")):
            if block["header"] != last["header"]:
                last["rows"].append(block["header"])
            last["rows"].extend(block["rows"])
            out[-1] = (page, last)
            continue
        out.append((page, block))
    return out


def heading_matches(raw: str, title: str) -> bool:
    """Is this block the heading the bookmark names?"""
    a, b = _norm(raw), _norm(title)
    if not a or not b:
        return False
    if a == b or a.startswith(b):
        return True
    rest = _norm(NUMBER_PREFIX.sub("", title))
    return len(rest) >= 4 and (a == rest or a.startswith(rest))


def locate_headings(toc: list, seq: list) -> tuple:
    """
    Where in seq each bookmark's section begins.

    A bookmark gives a title and a page. The heading is the first block on
    that page, after the previous heading, that reads as the title. When
    no block does (the heading is drawn as an image, or split oddly), the
    section starts at the top of the bookmark's page, which is still the
    right page for a citation.
    """
    marks, matched, pos = [], 0, 0
    for _, title, page in toc:
        found = None
        j = pos
        while j < len(seq) and seq[j][0] <= page:
            _, block = seq[j]
            if (seq[j][0] == page and block["kind"] == "text"
                    and heading_matches(block["raw"], title)):
                found = j
                break
            j += 1
        if found is None:
            found = next((k for k in range(pos, len(seq))
                          if seq[k][0] >= page), len(seq))
        else:
            matched += 1
        marks.append(found)
        pos = min(found + 1, len(seq))
    return marks, matched


def front_matter(seq: list, toc: list) -> list:
    """
    What comes before the first bookmark, without the table of contents.

    The contents pages repeat every heading in the document. Left in the
    store they answer to almost any search, and say nothing. A line is a
    contents line if it ends in dot leaders and a page number, or reads as
    a bookmark title followed by a number.
    """
    titles = {_norm(title) for _, title, _ in toc}
    blocks = []
    for _, block in seq:
        if block["kind"] == "table":
            rows = [r for r in [block["header"]] + block["rows"]
                    if not TOC_LEADER.search(r)]
            if len(rows) >= 2:
                blocks.append({**block, "header": rows[0], "rows": rows[1:]})
            continue
        kept = [line for line in block["raw"].splitlines()
                if line.strip()
                and not TOC_LEADER.search(line)
                and _norm(re.sub(r"\d+\s*$", "", line)) not in titles]
        if kept:
            blocks.append(text_block(unwrap("\n".join(kept))))
    return blocks


def read_outline_pdf(path: Path, doc_type: str, images_dir):
    """
    A report PDF cut into sections at its bookmarks.

    Returns None when the file has no usable outline, and the caller then
    falls back to extract_prose_pdf.

    extract_prose_pdf finds headings by pattern: a section number followed
    by a title in capitals. The Aditya-L1 report has headings in mixed
    case too, which that pattern cannot tell from a numbered list item, so
    their text was swept into whichever section came before. The bookmarks
    are the document's own list of its sections, levels included.
    """
    doc = fitz.open(str(path))
    toc = [(lv, title.strip(), page) for lv, title, page in doc.get_toc()
           if title.strip() and page >= 1]
    if len(toc) < MIN_OUTLINE:
        doc.close()
        return None

    seq = page_blocks(doc, doc_type, images_dir)
    doc.close()
    marks, matched = locate_headings(toc, seq)

    chunks = []
    front  = front_matter(seq[:marks[0]], toc) if marks else []
    if front:
        chunk = ex.make_chunk(f"{doc_type}_front", render(front), doc_type,
                              path.name, "section", page=1,
                              heading="Front matter")
        chunk["heading_path"] = "Front matter"
        chunk["blocks"]       = front
        chunks.append(chunk)

    stack = []
    for i, (level, title, page) in enumerate(toc):
        push_heading(stack, level, title)
        end  = marks[i + 1] if i + 1 < len(marks) else len(seq)
        body = []
        for _, block in seq[marks[i]:end]:
            if block["kind"] == "text":
                block = text_block(unwrap(block["raw"]))
            body.append(block)

        # The flat reader starts every section with its heading line, and
        # the test questions were written against that. Keep it so.
        first = body[0]["text"] if body and body[0]["kind"] == "text" else ""
        if not heading_matches(first, title):
            body.insert(0, text_block(title))

        text = render(body)
        if len(text) - len(title) < 40:      # a heading with nothing under it
            continue
        chunk = ex.make_chunk(f"{doc_type}_s{i}", text, doc_type, path.name,
                              "section", page=page, heading=title)
        chunk["heading_path"] = heading_path(stack)
        chunk["blocks"]       = body
        chunks.append(chunk)

    print(f"      {len(toc)} bookmarks, {matched} found as a heading on "
          f"their page, {len(chunks)} sections")
    return chunks
