"""
Check captions against their figures, by eye.

Retrieval scores say whether a caption can be found. They do not say
whether it is right: a caption that names the wrong subsystem, or reads a
plot's axes wrongly, is found just as easily. The only way to know is to
look. This writes one page with a sample of figures, each next to the
caption stored for it, and three buttons.

    python review_captions.py                  30 figures, picked at random
    python review_captions.py --n 60 --seed 2  a different, larger sample
    python review_captions.py --all            every figure
    python review_captions.py --tally caption_review.json

Open the page it names in a browser, mark each caption, and press Save at
the bottom. That downloads caption_review.json; --tally counts it.

The page sits in extracted/ beside the figures it shows, which is kept out
of git. It is a local file and is not to be shared.

Marking guide:
    right    says what the figure is, with nothing in it that is false
    partly   the kind of figure is right but a detail is wrong or made up
    wrong    describes something else
"""

import argparse
import html
import json
import random
from collections import Counter
from pathlib import Path

from core import registry

PAGE = "caption_review.html"

STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #f4f4f1; color: #1c1c1a; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 80px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.lead { color: #5a5a55; margin: 0 0 24px; }
.item { background: #fff; border: 1px solid #dcdcd6; border-radius: 6px; padding: 16px; margin: 0 0 16px; }
.item img { max-width: 100%; max-height: 520px; display: block; margin: 0 0 12px; border: 1px solid #e6e6e0; }
.where { font-size: 13px; color: #5a5a55; margin: 0 0 8px; }
.caption { margin: 0 0 12px; }
label { margin-right: 16px; cursor: pointer; }
input[type=text] { width: 100%; box-sizing: border-box; margin-top: 8px; padding: 6px 8px;
                   border: 1px solid #c8c8c2; border-radius: 4px; font: inherit; }
.bar { position: fixed; left: 0; right: 0; bottom: 0; background: #1c1c1a; color: #fff;
       padding: 10px 16px; display: flex; gap: 16px; align-items: center; }
button { font: inherit; padding: 6px 14px; border: 0; border-radius: 4px; cursor: pointer; }
"""

SCRIPT = """
const items = [...document.querySelectorAll('.item')];
function marks() {
  return items.map(el => ({
    image: el.dataset.image,
    mark: (el.querySelector('input[type=radio]:checked') || {}).value || '',
    note: el.querySelector('input[type=text]').value.trim(),
  }));
}
function count() {
  const done = marks().filter(m => m.mark).length;
  document.getElementById('count').textContent = done + ' of ' + items.length + ' marked';
}
document.addEventListener('change', count);
document.getElementById('save').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(marks(), null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'caption_review.json';
  a.click();
});
count();
"""


def build(satellite, n: int, seed: int, everything: bool) -> Path:
    path = satellite.extracted_dir / "image_captions.json"
    if not path.exists():
        raise SystemExit(f"no captions yet. Run: python cli.py caption {satellite.slug}")
    captions = json.loads(path.read_text(encoding="utf-8"))
    if not everything:
        captions = random.Random(seed).sample(captions, min(n, len(captions)))
    captions.sort(key=lambda c: (c.get("source_doc", ""), c.get("page", 0)))

    rows = []
    for i, cap in enumerate(captions):
        image = Path(cap["image_file"]).name
        where = " | ".join(x for x in (
            cap.get("source_doc", ""),
            f"page {cap['page']}" if cap.get("page") else "",
            cap.get("section_heading", ""),
            cap.get("subsystem", "") if cap.get("subsystem") != "UNKNOWN" else "",
        ) if x)
        buttons = "".join(
            f'<label><input type="radio" name="m{i}" value="{v}"> {v}</label>'
            for v in ("right", "partly", "wrong"))
        rows.append(
            f'<div class="item" data-image="{html.escape(image)}">'
            f'<img src="images/{html.escape(image)}" alt="">'
            f'<p class="where">{html.escape(where)}</p>'
            f'<p class="caption">{html.escape(cap["text"])}</p>'
            f'{buttons}'
            f'<input type="text" placeholder="What is wrong, if anything">'
            f'</div>')

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Caption review</title><style>{STYLE}</style></head><body><main>'
        f'<h1>Caption review: {html.escape(satellite.name)}</h1>'
        f'<p class="lead">{len(captions)} figures. Mark each caption against '
        f'the figure above it, then save.</p>'
        + "".join(rows) +
        '</main><div class="bar"><span id="count"></span>'
        f'<button id="save">Save marks</button></div><script>{SCRIPT}</script>'
        '</body></html>')

    out = satellite.extracted_dir / PAGE
    out.write_text(page, encoding="utf-8")
    return out


def tally(path: Path) -> None:
    marks  = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(m["mark"] or "not marked" for m in marks)
    done   = sum(n for mark, n in counts.items() if mark != "not marked")
    print(f"\n  {len(marks)} figures, {done} marked")
    for mark in ("right", "partly", "wrong", "not marked"):
        if counts[mark]:
            share = f"{counts[mark] / done * 100:5.1f}%" if done and mark != "not marked" else ""
            print(f"  {mark:<11}{counts[mark]:>4}  {share}")
    notes = [m for m in marks if m.get("note")]
    if notes:
        print(f"\n  {len(notes)} with a note:")
        for m in notes:
            print(f"    {m['image']:<28}{m['mark']:<8}{m['note']}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Check captions against figures")
    ap.add_argument("--satellite", default="aditya-l1")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tally", default=None, help="count a saved review file")
    args = ap.parse_args()

    if args.tally:
        tally(Path(args.tally))
        return
    out = build(registry.load_satellite(args.satellite), args.n, args.seed, args.all)
    print(f"\n  open in a browser: {out}\n")


if __name__ == "__main__":
    main()
