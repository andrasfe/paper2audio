"""Generate per-section spoken-word narration scripts via an LLM.

Each paper section becomes one chapter of a longer talk. Chapters are
generated independently (so any one can be regenerated alone) but share
the full paper text and the chapter outline for continuity.
"""
from __future__ import annotations

from . import providers

SYSTEM_PROMPT = """\
You write spoken-word scripts that explain research papers out loud to a
single listener who is smart but has NO background in the subject. The
listener cannot see anything: no figures, no tables, no equations. Your
script is their only channel.

You are writing ONE CHAPTER of a longer talk. You receive the full paper,
the outline of all chapters, and the assignment for this chapter. Rules:
- Cover ONLY this chapter's assignment; other chapters cover the rest.
- Open with a short spoken transition into the chapter's topic. Do NOT
  greet the listener or introduce the whole talk unless this is chapter
  one; do NOT wrap up the whole talk unless this is the final chapter.
- Assume minimal prior knowledge, but do not re-teach concepts the
  outline shows were covered in earlier chapters; a one-line reminder
  is enough.
- Be painfully detailed. Explain motivation, the workings of every
  method, experiment architecture, the actual numbers and their
  meaning, contributions, and shortcomings. Teach, don't summarize.
- Describe any figure or table in words: axes, trends, key numbers.
- Read equations as sentences, never as symbols.
- Output must be TTS-ready plain text: no markdown, no headers, no
  bullets, no symbols like %, ~, ^, or Greek letters. Spell acronyms
  as they should be spoken ("Q A O A"). Write numbers to be heard
  ("ten to the minus nine").
- Aim for the word target given; longer is better than shorter.
"""


def build_user_prompt(
    paper_text: str,
    outline: list[str],
    index: int,
    title: str,
    brief: str,
    source_text: str | None,
    target_words: int,
) -> str:
    outline_txt = "\n".join(
        f"  {i + 1}. {t}" for i, t in enumerate(outline)
    )
    src = (
        f"\n--- SOURCE TEXT FOR THIS CHAPTER ---\n{source_text}\n"
        if source_text
        else ""
    )
    return (
        f"--- FULL PAPER ---\n{paper_text}\n--- END PAPER ---\n\n"
        f"Chapter outline of the talk:\n{outline_txt}\n\n"
        f"Write chapter {index} of {len(outline)}: \"{title}\".\n"
        f"Assignment: {brief}\n"
        f"Target length: about {target_words} words.{src}"
    )


def narrate_section(
    paper_text: str,
    outline: list[str],
    index: int,
    title: str,
    brief: str,
    source_text: str | None = None,
    target_words: int = 700,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    user = build_user_prompt(
        paper_text, outline, index, title, brief, source_text, target_words
    )
    return providers.complete(
        SYSTEM_PROMPT,
        user,
        provider=provider,
        model=model,
        api_key=api_key,
    ).strip()
