"""Extract narratable plain text from a research paper (PDF or LaTeX).

The output is not meant to be pretty; it is meant to hand an LLM the
complete textual content of the paper (body, captions, table cells,
equations as source) so the narration stage misses nothing.
"""
from __future__ import annotations

import pathlib
import re


def extract_sections(path: str | pathlib.Path) -> list[tuple[str, str]]:
    """Extract the paper as an ordered list of (section_title, text).

    LaTeX: split on \\section markers (front matter/abstract becomes the
    first entry). PDF: heuristic split on numbered headings; falls back
    to a single section if none are found.
    """
    full = extract(path)
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in (".tex", ".latex"):
        return _split_detexed(full)
    return _split_pdf_text(full)


def _split_detexed(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"^== (.*?) ==$", text, flags=re.M)
    sections: list[tuple[str, str]] = []
    if parts[0].strip():
        sections.append(("Front matter and abstract", parts[0].strip()))
    for i in range(1, len(parts) - 1, 2):
        sections.append((parts[i].strip(), parts[i + 1].strip()))
    return sections or [("Full paper", text)]


_PDF_HEADING = re.compile(
    r"^(?:(?:[IVXL]+|\d+)\.\s+[A-Z][^\n]{2,60}|ABSTRACT|REFERENCES|"
    r"ACKNOWLEDGMENTS?|CONCLUSIONS?)\s*$",
    re.M,
)


def _split_pdf_text(text: str) -> list[tuple[str, str]]:
    matches = list(_PDF_HEADING.finditer(text))
    if len(matches) < 2:
        return [("Full paper", text)]
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        sections.append(("Front matter", text[: matches[0].start()].strip()))
    for m, nxt in zip(matches, matches[1:] + [None]):
        end = nxt.start() if nxt else len(text)
        title = re.sub(r"^(?:[IVXL]+|\d+)\.\s*", "", m.group(0)).strip().title()
        sections.append((title, text[m.end():end].strip()))
    return sections


def extract(path: str | pathlib.Path) -> str:
    p = pathlib.Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(p)
    if suffix in (".tex", ".latex"):
        return extract_latex(p)
    raise ValueError(f"unsupported input type {suffix!r}; expected .pdf or .tex")


def extract_pdf(path: pathlib.Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"[page {i}]\n{text}")
    return "\n\n".join(pages)


# LaTeX environments whose *content* should be kept verbatim so the
# narrator can read equations and table cells.
_KEEP_ENVS = ("equation", "align", "table", "tabular", "figure", "abstract")
# Environments to drop entirely.
_DROP_ENVS = ("tikzpicture",)


def extract_latex(path: pathlib.Path) -> str:
    src = path.read_text(encoding="utf-8", errors="replace")
    return detex(src)


def detex(src: str) -> str:
    """Best-effort LaTeX -> plain text. Keeps section structure, captions,
    table content, and equation source (an LLM reads LaTeX math fine)."""
    text = src

    # Strip comments (but not escaped \%).
    text = re.sub(r"(?<!\\)%.*", "", text)

    # Preamble: keep title if present, drop the rest.
    m = re.search(r"\\title\{(.*?)\}", text, re.S)
    title = m.group(1) if m else ""
    body = text.split(r"\begin{document}", 1)
    text = body[1] if len(body) == 2 else text

    text = re.sub(r"\\end\{document\}.*", "", text, flags=re.S)

    for env in _DROP_ENVS:
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", "", text, flags=re.S
        )

    # Bibliography: keep item text, drop markup.
    text = re.sub(r"\\begin\{thebibliography\}\{.*?\}", "\nREFERENCES\n", text)
    text = re.sub(r"\\end\{thebibliography\}", "", text)
    text = re.sub(r"\\bibitem\{.*?\}", "\n- ", text)

    # Sections -> headed lines.
    def _sec(m: re.Match) -> str:
        level = m.group(1)
        name = m.group(2)
        marker = {"section": "==", "subsection": "--", "subsubsection": "-"}.get(
            level, "-"
        )
        return f"\n\n{marker} {name} {marker}\n"

    text = re.sub(r"\\(section|subsection|subsubsection)\*?\{(.*?)\}", _sec, text)
    text = re.sub(r"\\paragraph\*?\{(.*?)\}", r"\n\1: ", text)

    # Captions and labels.
    text = re.sub(r"\\caption\{", "CAPTION: {", text)
    text = re.sub(r"\\label\{(.*?)\}", "", text)

    # Keep-env wrappers become plain markers; content stays.
    for env in _KEEP_ENVS:
        text = re.sub(rf"\\begin\{{{env}\*?\}}(\[.*?\])?", f"\n[{env}]\n", text)
        text = re.sub(rf"\\end\{{{env}\*?\}}", f"\n[end {env}]\n", text)

    # Common inline commands: keep the argument.
    for cmd in ("emph", "textbf", "textit", "texttt", "mathrm", "text", "mbox"):
        text = re.sub(rf"\\{cmd}\{{(.*?)\}}", r"\1", text)

    # Citations/refs -> readable placeholders.
    text = re.sub(r"~?\\cite\{(.*?)\}", r" [ref: \1]", text)
    text = re.sub(r"~?\\(eq)?ref\{(.*?)\}", r" [\2]", text)

    # booktabs / layout noise.
    text = re.sub(
        r"\\(toprule|midrule|bottomrule|centering|small|hline|maketitle|"
        r"IEEEpeerreviewmaketitle|noindent)\b",
        "",
        text,
    )
    text = re.sub(r"\\(IEEEauthorblock[NA]|author|title)\{", "{", text)

    # Unescape and tidy.
    text = text.replace(r"\%", "%").replace(r"\&", "&").replace(r"\_", "_")
    text = text.replace("~", " ").replace(r"\\", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    header = f"TITLE: {title}\n\n" if title else ""
    return header + text.strip()
