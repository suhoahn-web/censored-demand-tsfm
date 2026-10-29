# -*- coding: utf-8 -*-
"""Post-process a pandoc-built manuscript docx for submission.

* page: A4, 2.54 cm margins (pandoc's reference leaves page size unset, so Word
  falls back to the locale default and figure widths become unpredictable)
* headings (Title, Heading 1-4): black, bold, body theme font, no theme colour
* figures: every inline picture scaled to the text width (aspect ratio kept)

Used by build_jors_r1.py and build_jors_r1_marked.py.
"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Mm

A4_W, A4_H, MARGIN = Mm(210), Mm(297), Cm(2.54)
HEADING_SIZES = {"Title": 16, "Heading 1": 16, "Heading 2": 14, "Heading 3": 12, "Heading 4": 12}
HEADING_STYLES = tuple(HEADING_SIZES)


def _set_page(doc) -> int:
    for sec in doc.sections:
        sec.orientation = WD_ORIENT.PORTRAIT
        sec.page_width, sec.page_height = A4_W, A4_H
        sec.left_margin = sec.right_margin = MARGIN
        sec.top_margin = sec.bottom_margin = MARGIN
    return A4_W - 2 * MARGIN  # text width in EMU


def _black_bold(style, size_pt: int) -> None:
    rpr = style.element.get_or_add_rPr()
    for tag in ("w:color", "w:b", "w:bCs", "w:sz", "w:szCs"):
        for el in rpr.findall(qn(tag)):
            rpr.remove(el)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "000000")
    rpr.append(color)
    rpr.append(OxmlElement("w:b"))
    rpr.append(OxmlElement("w:bCs"))
    for tag in ("w:sz", "w:szCs"):
        el = OxmlElement(tag)
        el.set(qn("w:val"), str(size_pt * 2))
        rpr.append(el)
    # headings in the body font (minor theme font) rather than the display font
    for el in rpr.findall(qn("w:rFonts")):
        rpr.remove(el)
    fonts = OxmlElement("w:rFonts")
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme"):
        fonts.set(qn(attr), "minorHAnsi")
    fonts.set(qn("w:eastAsiaTheme"), "minorEastAsia")
    rpr.insert(0, fonts)


def _scale_figures(doc, width_emu: int) -> int:
    n = 0
    for shape in doc.inline_shapes:
        if not shape.width or not shape.height:
            continue
        ratio = shape.height / shape.width
        shape.width = Emu(width_emu)
        shape.height = Emu(int(width_emu * ratio))
        n += 1
    return n


def finalize(path: Path) -> None:
    doc = Document(path)
    text_width = _set_page(doc)
    for name in HEADING_STYLES:
        try:
            _black_bold(doc.styles[name], HEADING_SIZES[name])
        except KeyError:
            pass
    n = _scale_figures(doc, text_width)
    doc.save(path)
    print(f"post-processed {Path(path).name}: A4, black bold headings, {n} figures at text width")


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        finalize(Path(p))
