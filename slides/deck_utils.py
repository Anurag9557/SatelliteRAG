"""Small shared helpers for building the two SatelliteRAG decks with
python-pptx. Kept minimal: title slide + bullet slide + references slide
cover everything both decks need.
"""

from pptx import Presentation
from pptx.util import Pt


def new_deck(title: str, subtitle: str) -> Presentation:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle
    return prs


def add_bullet_slide(prs: Presentation, title: str, bullets: list, font_size: int = 18):
    """bullets: list of str (level 0) or (text, level) tuples."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    body = slide.placeholders[1]
    tf = body.text_frame
    tf.word_wrap = True

    first = True
    for item in bullets:
        text, level = item if isinstance(item, tuple) else (item, 0)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = text
        p.level = level
        for run in p.runs:
            run.font.size = Pt(font_size)
    return slide


def add_text_slide(prs: Presentation, title: str, body_text: str, font_size: int = 16):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    tf = slide.placeholders[1].text_frame
    tf.word_wrap = True
    tf.text = body_text
    for run in tf.paragraphs[0].runs:
        run.font.size = Pt(font_size)
    return slide
