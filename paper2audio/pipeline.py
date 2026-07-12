"""Section-based project pipeline.

A *project directory* holds everything for one paper:

    project/
      manifest.json        section list, order, file paths, tts settings
      paper.txt            extracted full paper text
      sections/NN-slug.txt extracted source text per section
      scripts/NN-slug.txt  narration script per section (LLM or hand-written)
      audio/NN-slug.mp3    synthesized audio per section
      full.mp3             concatenation (via `concat`)

Any section's script or audio can be regenerated alone; `concat` stitches
the per-section MP3s in manifest order (all sections share one encoder
configuration, so plain frame concatenation is valid).
"""
from __future__ import annotations

import json
import pathlib
import re

from . import extract, narrate, tts

MANIFEST = "manifest.json"


def _slug(index: int, title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48]
    return f"{index:02d}-{s or 'section'}"


def load_manifest(project: str | pathlib.Path) -> dict:
    p = pathlib.Path(project) / MANIFEST
    return json.loads(p.read_text(encoding="utf-8"))


def save_manifest(project: str | pathlib.Path, manifest: dict) -> None:
    p = pathlib.Path(project) / MANIFEST
    p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def plan(
    paper: str | pathlib.Path,
    project: str | pathlib.Path,
    words_per_section: int = 700,
) -> dict:
    """Extract the paper, split into sections, write the manifest."""
    project = pathlib.Path(project)
    (project / "sections").mkdir(parents=True, exist_ok=True)
    (project / "scripts").mkdir(exist_ok=True)
    (project / "audio").mkdir(exist_ok=True)

    full = extract.extract(paper)
    (project / "paper.txt").write_text(full, encoding="utf-8")

    sections = []
    for i, (title, text) in enumerate(extract.extract_sections(paper), 1):
        slug = _slug(i, title)
        (project / "sections" / f"{slug}.txt").write_text(text, encoding="utf-8")
        # Scale each chapter's word target with its share of the paper.
        share = max(0.4, min(3.0, len(text) / max(1, len(full) / 10)))
        sections.append(
            {
                "index": i,
                "slug": slug,
                "title": title,
                "brief": f"Narrate the paper section titled '{title}' in full.",
                "words": int(words_per_section * share),
                "source": f"sections/{slug}.txt",
                "script": f"scripts/{slug}.txt",
                "audio": f"audio/{slug}.mp3",
            }
        )
    manifest = {
        "paper": "paper.txt",
        "paper_source": str(paper),
        "tts": {"engine": "auto", "voice": "en-us", "rate": 165, "bitrate": 64},
        "sections": sections,
    }
    save_manifest(project, manifest)
    return manifest


def select_sections(manifest: dict, selector: str | None) -> list[dict]:
    """Select sections by 1-based index list ("2,5"), range ("3-6"), or
    slug/title substring ("qae"). None selects all."""
    secs = manifest["sections"]
    if not selector:
        return secs
    sel = selector.strip().lower()
    if re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", sel):
        wanted: set[int] = set()
        for part in sel.split(","):
            if "-" in part:
                a, b = part.split("-")
                wanted.update(range(int(a), int(b) + 1))
            else:
                wanted.add(int(part))
        picked = [s for s in secs if s["index"] in wanted]
    else:
        picked = [
            s for s in secs
            if sel in s["slug"].lower() or sel in s["title"].lower()
        ]
    if not picked:
        raise ValueError(f"selector {selector!r} matched no sections")
    return picked


def narrate_sections(
    project: str | pathlib.Path,
    selector: str | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    log=print,
) -> None:
    project = pathlib.Path(project)
    manifest = load_manifest(project)
    paper_text = (project / manifest["paper"]).read_text(encoding="utf-8")
    outline = [s["title"] for s in manifest["sections"]]
    for s in select_sections(manifest, selector):
        log(f"narrating [{s['index']}/{len(outline)}] {s['title']} "
            f"({s['words']} words, {provider}) ...")
        source = None
        if s.get("source"):
            source = (project / s["source"]).read_text(encoding="utf-8")
        script = narrate.narrate_section(
            paper_text, outline, s["index"], s["title"], s["brief"],
            source_text=source, target_words=s["words"],
            provider=provider, model=model,
        )
        (project / s["script"]).write_text(script + "\n", encoding="utf-8")
        log(f"  wrote {s['script']} ({len(script.split())} words)")


def synth_sections(
    project: str | pathlib.Path,
    selector: str | None = None,
    log=print,
) -> None:
    project = pathlib.Path(project)
    manifest = load_manifest(project)
    cfg = manifest.get("tts", {})
    for s in select_sections(manifest, selector):
        script_path = project / s["script"]
        if not script_path.exists():
            log(f"  skip [{s['index']}] {s['title']}: no script yet")
            continue
        text = script_path.read_text(encoding="utf-8")
        dur = tts.text_to_mp3(
            text, project / s["audio"],
            engine=cfg.get("engine", "auto"),
            voice=cfg.get("voice", "en-us"),
            rate_wpm=cfg.get("rate", 165),
            bitrate_kbps=cfg.get("bitrate", 64),
            model=cfg.get("model"),
        )
        log(f"  wrote {s['audio']} ({dur / 60:.1f} min)")


def concat(
    project: str | pathlib.Path,
    out: str | pathlib.Path | None = None,
    log=print,
) -> pathlib.Path:
    """Concatenate per-section MP3s in manifest order. Valid because all
    sections are encoded with identical parameters (CBR mono frames)."""
    project = pathlib.Path(project)
    manifest = load_manifest(project)
    out = pathlib.Path(out) if out else project / "full.mp3"
    missing = [s["audio"] for s in manifest["sections"]
               if not (project / s["audio"]).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing section audio (run synth first): {missing}"
        )
    with open(out, "wb") as f:
        for s in manifest["sections"]:
            f.write((project / s["audio"]).read_bytes())
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
        f"{len(manifest['sections'])} sections)")
    return out
