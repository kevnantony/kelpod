"""Plain (model-free) loaders. Each returns a :class:`ParsedDocument` or raises.

Used for TXT/MD/CSV/Parquet/XML always, and as the fallback for PDF/DOCX/HTML/images when
Docling (or its layout artefacts) is unavailable.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from zllf.stages.ingest.model import ParsedBlock, ParsedDocument
from zllf.stages.ingest.normalize import (
    blocks_from_text,
    clean_inline,
    normalize_text,
    split_paragraphs,
    strip_repeated_edges,
)
from zllf.stages.ingest.ocr import OcrEngine, ocr_image
from zllf.stages.ingest.options import IngestOptions


class ParseError(Exception):
    """A loader-level failure whose message is a short, text-free reason code."""


# ---------------------------------------------------------------- text decoding
def read_text(path: Path) -> str:
    """Decode a text file: BOM-aware UTF-8/16, then cp1252, then latin-1."""
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:8192]:
        raise ParseError("binary_content")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


# ---------------------------------------------------------------- tables
def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return " ".join(str(value).replace("|", "/").split())


def rows_to_markdown(header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    head = [_cell(h) or f"col{i + 1}" for i, h in enumerate(header)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [_cell(c) for c in row]
        cells += [""] * (len(head) - len(cells))
        lines.append("| " + " | ".join(cells[: len(head)]) + " |")
    return "\n".join(lines)


def grid_text(rows: Iterable[Sequence[Any]]) -> str:
    """Plain-text rendering of a table grid (cells tab-joined, one row per line)."""
    return "\n".join("\t".join(_cell(c) for c in row) for row in rows)


def records_text(header: Sequence[str], rows: Iterable[Sequence[Any]], first_row: int) -> str:
    """CSV/Parquet rows as text records with column headers (P1.1.5)."""
    lines = []
    for i, row in enumerate(rows):
        fields = [f"{h}: {_cell(v)}" for h, v in zip(header, row, strict=False) if _cell(v)]
        lines.append(f"Row {first_row + i}: " + " | ".join(fields))
    return "\n".join(lines)


def _record_blocks(header: Sequence[str], rows: Iterator[Sequence[Any]],
                   opts: IngestOptions, warnings: list[str]) -> list[ParsedBlock]:
    header = [clean_inline(str(h)) or f"col{i + 1}" for i, h in enumerate(header)]
    blocks: list[ParsedBlock] = []
    group: list[Sequence[Any]] = []
    start = 1
    for n, row in enumerate(rows):
        if n >= opts.max_table_rows:
            warnings.append("table_rows_truncated")
            break
        group.append([clean_inline(_cell(v)) for v in row])
        if len(group) >= opts.rows_per_block:
            blocks.append(ParsedBlock("table", records_text(header, group, start)))
            start += len(group)
            group = []
    if group:
        blocks.append(ParsedBlock("table", records_text(header, group, start)))
    return [b for b in blocks if b.text.strip()]


def load_csv(path: Path, opts: IngestOptions) -> ParsedDocument:
    text = read_text(path)
    sample = text[:16384]
    if path.suffix.lower() == ".tsv":
        dialect: Any = csv.excel_tab
    else:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    try:
        header = next(reader)
    except StopIteration:
        return ParsedDocument(parser="csv")
    warnings: list[str] = []
    rows = (r for r in reader if any(c.strip() for c in r))
    return ParsedDocument(blocks=_record_blocks(header, rows, opts, warnings), parser="csv",
                          warnings=warnings)


def load_parquet(path: Path, opts: IngestOptions) -> ParsedDocument:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    header = list(pf.schema_arrow.names)

    def rows() -> Iterator[list[Any]]:
        for batch in pf.iter_batches(batch_size=max(opts.rows_per_block, 1024)):
            cols = [c.to_pylist() for c in batch.columns]
            yield from (list(r) for r in zip(*cols, strict=True))

    warnings: list[str] = []
    return ParsedDocument(blocks=_record_blocks(header, rows(), opts, warnings),
                          parser="parquet", warnings=warnings)


# ---------------------------------------------------------------- txt / md
def load_txt(path: Path, opts: IngestOptions) -> ParsedDocument:
    text = normalize_text(read_text(path))
    return ParsedDocument(blocks=blocks_from_text(text), parser="plain-text")


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_SETEXT = re.compile(r"^(=+|-+)\s*$")
_MD_LIST = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_EMPH = re.compile(r"(\*\*|__|`)")
_MD_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _md_inline(text: str) -> str:
    return clean_inline(_MD_EMPH.sub("", _MD_LINK.sub(r"\1", text)))


def _md_table(lines: list[str]) -> ParsedBlock:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
            for i, ln in enumerate(lines) if i != 1]
    md = rows_to_markdown(rows[0], [[_md_inline(c) for c in r] for r in rows[1:]])
    return ParsedBlock("table", grid_text(rows), table_markdown=md)


def load_md(path: Path, opts: IngestOptions) -> ParsedDocument:
    text = normalize_text(_MD_COMMENT.sub("", read_text(path)))
    lines = text.split("\n")
    blocks: list[ParsedBlock] = []
    para: list[str] = []
    items: list[str] = []
    i = 0

    def flush() -> None:
        if para:
            blocks.append(ParsedBlock("paragraph", _md_inline(" ".join(para))))
            para.clear()
        if items:
            blocks.append(ParsedBlock("list", "\n".join(items)))
            items.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            code = "\n".join(lines[i + 1 : j]).strip()
            if code:
                blocks.append(ParsedBlock("other", code))
            i = j + 1
            continue
        if m := _MD_HEADING.match(line):
            flush()
            blocks.append(ParsedBlock("heading", _md_inline(m.group(2)), level=len(m.group(1))))
        elif (para and len(para) == 1 and not items and _MD_SETEXT.match(line)
              and i > 0 and lines[i - 1].strip()):
            level = 1 if line.startswith("=") else 2
            heading = para.pop()
            flush()
            blocks.append(ParsedBlock("heading", _md_inline(heading), level=level))
        elif line.lstrip().startswith("|") and i + 1 < len(lines) and \
                _MD_TABLE_SEP.match(lines[i + 1]):
            flush()
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append(_md_table(lines[i:j]))
            i = j
            continue
        elif _MD_LIST.match(line):
            if para:
                flush()
            items.append("- " + _md_inline(_MD_LIST.sub("", line)))
        elif not line.strip():
            flush()
        elif items:
            items[-1] = f"{items[-1]} {_md_inline(line)}"
        elif line.startswith(">"):
            para.append(line.lstrip("> "))
        else:
            para.append(line)
        i += 1
    flush()
    return ParsedDocument(blocks=[b for b in blocks if b.text], parser="markdown")


# ---------------------------------------------------------------- xml
_XML_HEADING_TAGS = {"title", "heading", "head", "h1", "h2", "h3", "h4", "h5", "h6"}
_XML_PARA_TAGS = {"p", "para", "paragraph", "li", "item", "note", "description", "text",
                  "abstract", "summary", "remark", "step"}
_XML_SECTION_TAGS = {"section", "sect", "sect1", "sect2", "sect3", "chapter", "part", "div"}


def _local(tag: Any) -> str:
    return str(tag).rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def load_xml(path: Path, opts: IngestOptions) -> ParsedDocument:
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False,
                             huge_tree=False, recover=False, remove_comments=True,
                             remove_pis=True)
    root = etree.parse(str(path), parser).getroot()
    blocks: list[ParsedBlock] = []

    def text_of(el: Any) -> str:
        return clean_inline(" ".join(t for t in el.itertext()))

    def walk(el: Any, depth: int) -> None:
        tag = _local(el.tag)
        if tag in _XML_HEADING_TAGS:
            if t := text_of(el):
                blocks.append(ParsedBlock("heading", t, level=max(1, depth)))
            return
        if tag in _XML_PARA_TAGS:
            if t := text_of(el):
                blocks.append(ParsedBlock("list" if tag in {"li", "item", "step"} else
                                          "paragraph", t))
            return
        children = [c for c in el if isinstance(c.tag, str)]
        leaves = [c for c in children if len(c) == 0 and _local(c.tag) not in
                  _XML_HEADING_TAGS | _XML_PARA_TAGS]
        if leaves and len(leaves) == len(children):
            # record-like element: "<field>: value" pairs
            fields = [f"{_local(c.tag)}: {clean_inline(c.text or '')}" for c in leaves
                      if (c.text or "").strip()]
            fields += [f"{_local(k)}: {clean_inline(v)}" for k, v in el.attrib.items()
                       if v.strip()]
            if fields:
                blocks.append(ParsedBlock("paragraph", f"{tag}: " + "; ".join(fields)))
            return
        if (el.text or "").strip():
            blocks.append(ParsedBlock("paragraph", clean_inline(el.text)))
        next_depth = depth + 1 if tag in _XML_SECTION_TAGS else depth
        for child in children:
            walk(child, next_depth)
            if (child.tail or "").strip():
                blocks.append(ParsedBlock("paragraph", clean_inline(child.tail)))

    walk(root, 1)
    return ParsedDocument(blocks=blocks, parser="xml")


# ---------------------------------------------------------------- html (fallback)
def load_html(path: Path, opts: IngestOptions) -> ParsedDocument:
    from bs4 import BeautifulSoup, Tag

    soup = BeautifulSoup(read_text(path), "html.parser")
    for el in soup(["script", "style", "noscript", "template", "head", "nav", "footer",
                    "header", "form", "svg"]):
        el.decompose()
    blocks: list[ParsedBlock] = []

    def walk(node: Tag) -> None:
        for el in node.children:
            if not isinstance(el, Tag):
                if (s := clean_inline(str(el))) and node.name in ("body", "div", "section",
                                                                 "article", "main"):
                    blocks.append(ParsedBlock("paragraph", s))
                continue
            name = el.name or ""
            if re.fullmatch(r"h[1-6]", name):
                if t := clean_inline(el.get_text(" ")):
                    blocks.append(ParsedBlock("heading", t, level=int(name[1])))
            elif name in ("p", "pre", "blockquote", "address"):
                if t := clean_inline(el.get_text(" ")):
                    blocks.append(ParsedBlock("paragraph", t))
            elif name == "figcaption" or name == "caption":
                if t := clean_inline(el.get_text(" ")):
                    blocks.append(ParsedBlock("caption", t))
            elif name in ("ul", "ol", "dl"):
                items = [clean_inline(li.get_text(" "))
                         for li in el.find_all(["li", "dt", "dd"], recursive=True)]
                if items := [f"- {t}" for t in items if t]:
                    blocks.append(ParsedBlock("list", "\n".join(items)))
            elif name == "table":
                rows = [[clean_inline(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
                        for tr in el.find_all("tr")]
                rows = [r for r in rows if any(r)]
                if rows:
                    blocks.append(ParsedBlock("table", grid_text(rows),
                                              table_markdown=rows_to_markdown(rows[0], rows[1:])))
            else:
                walk(el)

    walk(soup.body or soup)
    return ParsedDocument(blocks=blocks, parser="bs4")


# ---------------------------------------------------------------- docx (fallback)
def load_docx(path: Path, opts: IngestOptions) -> ParsedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    blocks: list[ParsedBlock] = []
    items: list[str] = []

    def flush_list() -> None:
        if items:
            blocks.append(ParsedBlock("list", "\n".join(items)))
            items.clear()

    for item in document.iter_inner_content():
        if isinstance(item, Table):
            flush_list()
            rows = [[clean_inline(c.text) for c in row.cells] for row in item.rows]
            rows = [r for r in rows if any(r)]
            if rows:
                blocks.append(ParsedBlock("table", grid_text(rows),
                                          table_markdown=rows_to_markdown(rows[0], rows[1:])))
            continue
        assert isinstance(item, Paragraph)
        text = clean_inline(item.text)
        if not text:
            continue
        style = (item.style.name if item.style is not None else "") or ""
        if style == "Title":
            flush_list()
            blocks.append(ParsedBlock("heading", text, level=0))
        elif m := re.match(r"Heading (\d)", style):
            flush_list()
            blocks.append(ParsedBlock("heading", text, level=int(m.group(1))))
        elif style.startswith("List") or (item._p.pPr is not None and \
                item._p.pPr.numPr is not None):
            items.append(f"- {text}")
        elif style == "Caption":
            flush_list()
            blocks.append(ParsedBlock("caption", text))
        else:
            flush_list()
            blocks.append(ParsedBlock("paragraph", text))
    flush_list()
    return ParsedDocument(blocks=blocks, parser="python-docx")


# ---------------------------------------------------------------- pdf / images
def pdf_page_texts(path: Path) -> list[str]:
    """Text layer per page via pypdfium2 (raises on broken/encrypted PDFs)."""
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise ParseError("pdf_unreadable") from exc
    try:
        texts = []
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            texts.append(tp.get_text_range())
            tp.close()
            page.close()
        return texts
    finally:
        pdf.close()


def scanned_pages(texts: Sequence[str], min_chars: int) -> list[int]:
    """0-based indices of pages with no/low text layer."""
    return [i for i, t in enumerate(texts) if len("".join(t.split())) < min_chars]


def _render_page(path: Path, index: int, dpi: int) -> Any:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        page = pdf[index]
        image = page.render(scale=dpi / 72).to_pil()
        page.close()
        return image
    finally:
        pdf.close()


def _paged_blocks(pages: list[str]) -> list[ParsedBlock]:
    pages = strip_repeated_edges(pages)
    blocks: list[ParsedBlock] = []
    for i, raw in enumerate(pages):
        blocks.extend(blocks_from_text(normalize_text(raw), page=i + 1))
    return blocks


def load_pdf(path: Path, opts: IngestOptions, engine: OcrEngine | None) -> ParsedDocument:
    texts = pdf_page_texts(path)
    warnings: list[str] = []
    scanned = scanned_pages(texts, opts.min_text_chars_per_page)
    n_ocr = 0
    for i in scanned:
        if engine is None:
            continue
        texts[i] = ocr_image(_render_page(path, i, opts.ocr_dpi), engine)
        n_ocr += 1
    if scanned and engine is None:
        warnings.append("ocr_unavailable")
    parser = "pypdfium2" + (f"+ocr:{engine.name}" if n_ocr and engine else "")
    return ParsedDocument(blocks=_paged_blocks(texts), n_pages=len(texts), n_ocr_pages=n_ocr,
                          parser=parser, warnings=warnings)


def load_image(path: Path, opts: IngestOptions, engine: OcrEngine | None) -> ParsedDocument:
    from PIL import Image, ImageSequence

    if engine is None:
        raise ParseError("ocr_unavailable")
    with Image.open(path) as img:
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
    texts = [ocr_image(frame, engine) for frame in frames]
    return ParsedDocument(blocks=_paged_blocks(texts), n_pages=len(frames),
                          n_ocr_pages=len(frames), parser=f"ocr:{engine.name}")


def paragraphs_of(doc: ParsedDocument) -> list[str]:
    """Non-heading, non-table block texts (used by tests/diagnostics)."""
    return [b.text for b in doc.blocks if b.type not in ("heading", "table")]


__all__ = [
    "ParseError", "grid_text", "load_csv", "load_docx", "load_html", "load_image", "load_md",
    "load_parquet", "load_pdf", "load_txt", "load_xml", "paragraphs_of", "pdf_page_texts",
    "read_text", "records_text", "rows_to_markdown", "scanned_pages", "split_paragraphs",
]
