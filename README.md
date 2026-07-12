# paper2audio

Turn a research paper (PDF or LaTeX) into a **long-form spoken audio
explanation** aimed at a listener with minimal background in the subject.
The narration explains, in painful detail, the paper's motivation, the
workings of every algorithm and piece of mathematics, the experiment
architecture, the results (figures and tables described in words), the
contributions, and the shortcomings — so you can genuinely learn the paper
by ear, e.g. on a flight.

> This directory is self-contained and designed to be extracted into its
> own repository later. It has no dependencies on the rest of this repo.

## Design: one MP3 per paper section

The unit of work is a **project directory** with one entry per paper
section, so any section can be re-narrated or re-synthesized alone and the
pieces stitched back together:

```
project/
  manifest.json          section list, order, word targets, tts settings
  paper.txt              extracted full paper text
  sections/NN-slug.txt   extracted source text per section
  scripts/NN-slug.txt    narration script per section (LLM or hand-edited)
  audio/NN-slug.mp3      synthesized audio per section
  full.mp3               concatenation of all sections
```

`manifest.json` is the source of truth and is meant to be edited: retitle
chapters, merge or insert sections (e.g. add primer chapters), tune word
targets, or point `script` at a hand-written file. Concatenation is plain
MP3-frame concatenation, valid because every section is encoded with
identical parameters (mono CBR, one sample rate).

## Pipeline

```
paper.pdf / paper.tex
   │  plan      extract text (pypdf / regex de-TeXer) and split into
   │            sections -> manifest.json + sections/*.txt
   │  narrate   LLM writes one TTS-ready teaching chapter per section
   │            (direct Anthropic or OpenAI API, per-section regenerable)
   │  synth     espeak-ng (bundled in the espeakng-loader wheel — no
   │            system install) -> 22 kHz mono PCM -> lameenc -> MP3
   │  concat    stitch audio/*.mp3 in manifest order -> full.mp3
   ▼
audio/NN-*.mp3 + full.mp3
```

Everything except `narrate` runs fully offline.

## Install

```bash
pip install -e .            # Anthropic narration backend
pip install -e .[openai]    # + OpenAI narration backend
```

## Usage

```bash
# 1. Extract + split into sections, write the manifest
paper2audio plan paper.tex -d project/

# 2. Generate narration scripts (choose your API)
export ANTHROPIC_API_KEY=...
paper2audio narrate -d project/                       # claude-opus-4-8
export OPENAI_API_KEY=...
paper2audio narrate -d project/ --provider openai     # gpt-5

# 3. Synthesize per-section MP3s (offline), 4. stitch
paper2audio synth -d project/
paper2audio concat -d project/ -o talk.mp3

# Or everything in one shot:
paper2audio all paper.tex -d project/ -o talk.mp3

# Regenerate ONE section after editing its script or brief:
paper2audio narrate -d project/ -s 7        # by index
paper2audio synth   -d project/ -s qae      # by title/slug substring
paper2audio concat  -d project/             # re-stitch
```

Section selectors accept an index (`3`), a list (`2,5`), a range (`3-6`),
or a case-insensitive title/slug substring (`qae`). Other knobs:
`--model` (override the per-provider default), `--words` (base word target
per section, scaled by section length), and the `tts` block in
`manifest.json` (`voice`, `rate` wpm, `bitrate` kbps).

## Library use

```python
from paper2audio import pipeline

pipeline.plan("paper.tex", "project/")
pipeline.narrate_sections("project/", "7", provider="openai")
pipeline.synth_sections("project/", "7")
pipeline.concat("project/", "talk.mp3")
```

## Example: the quantum-finance case-studies paper

`examples/quantum_finance/` is a complete project for this repository's
`paper/quantum_finance_case_studies.tex`: a ~52-minute, 7,420-word
novice-level walkthrough in ten chapters (two primer chapters were added
by hand-editing the manifest — the intended workflow). The per-section
MP3s are committed; rebuild any chapter or the full talk offline:

```bash
paper2audio synth  -d examples/quantum_finance -s 6   # redo one chapter
paper2audio concat -d examples/quantum_finance        # -> full.mp3
```

## Web player (GitHub Pages)

`player/index.html` is a dependency-free static web app that plays a
project's per-section MP3s in the browser:

- **Resumable**: playback position, per-chapter progress, and playback
  speed persist in `localStorage`; a Resume button picks up exactly where
  you left off.
- **Synced figures**: `site.json` maps narration timestamps to figures;
  the matching plot appears automatically as the audio reaches it, and
  any figure can be opened in a lightbox with pinch/scroll zoom and pan.
- **Spoken figure explanations**: every figure has its own short MP3
  ("Explain this figure"), which pauses the narration and resumes it
  afterwards.
- Auto-advance between chapters, playback speed cycling, lock-screen
  controls via the Media Session API.

The repository's GitHub Actions workflow (`.github/workflows/pages.yml`)
assembles the player + the quantum-finance example into a static site and
deploys it to GitHub Pages on every push to `main` (or manually via *Run
workflow*). The first run auto-enables Pages; the app is then live at
`https://<owner>.github.io/<repo>/`.

Run it locally with any static server that supports HTTP Range requests
(needed for audio seeking; GitHub Pages does):

```bash
npx http-server -p 8000 _site/   # after assembling like the workflow
```

`site.json` is generated data: chapter list with durations and
`{t, fig}` cues, plus a figure table (image, title, explanation audio).
See `examples/quantum_finance/site.json` for the shape.

## Notes & limitations

- The LaTeX extractor is a best-effort de-TeXer, not a TeX engine; exotic
  macros pass through raw (which the LLM handles fine). PDF section
  splitting is heuristic (numbered headings) and falls back to a single
  section.
- Two offline TTS engines: **piper** (neural, natural-sounding; default
  when `pip install .[neural]` is present — uses the en_US-joe-medium
  voice bundled in the `joe-us-piper-voice` wheel, or any Piper `.onnx`
  via the manifest's `tts.model`) and **espeak** (formant, tiny,
  robotic — the fallback). Select via `tts.engine` in `manifest.json`
  (`auto`/`piper`/`espeak`). Note espeak-ng can only be initialized
  once per process (`tts.get_synthesizer()` handles this).
- When section audio changes, bump `VERSION` in `player/sw.js` so
  installed PWAs re-download the updated cache.
- MP3s are mono 22.05 kHz, ~0.5 MB/min at the default 64 kbps.
