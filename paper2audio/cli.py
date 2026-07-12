"""Command-line entry point (section-based pipeline).

    paper2audio plan paper.tex -d project/            # extract + manifest
    paper2audio narrate -d project/ [-s 5] [--provider openai]
    paper2audio synth -d project/ [-s 5]              # TTS, offline
    paper2audio concat -d project/ [-o full.mp3]
    paper2audio all paper.tex -d project/ [-o full.mp3]

Every step is independently re-runnable; -s regenerates single sections
(by index "3", list "2,5", range "3-6", or title substring "qae").
"""
from __future__ import annotations

import argparse
import sys

from . import pipeline, providers


def _add_select(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("-s", "--sections", default=None,
                    help='section selector: "3", "2,5", "3-6", or substring')


def _add_llm(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--provider", default="anthropic",
                    choices=sorted(providers.DEFAULT_MODELS),
                    help="LLM API to call directly (default anthropic)")
    ap.add_argument("--model", default=None,
                    help="model id (defaults: "
                    + ", ".join(f"{k}={v}"
                                for k, v in providers.DEFAULT_MODELS.items())
                    + ")")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="paper2audio",
        description="Turn a research paper (PDF or LaTeX) into per-section "
        "spoken-audio chapters plus a concatenated full talk.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="extract paper, split sections, write manifest")
    p.add_argument("paper")
    p.add_argument("-d", "--project", required=True)
    p.add_argument("--words", type=int, default=700,
                   help="base word target per section (scaled by section size)")

    p = sub.add_parser("narrate", help="generate narration scripts via LLM")
    p.add_argument("-d", "--project", required=True)
    _add_select(p)
    _add_llm(p)

    p = sub.add_parser("synth", help="synthesize per-section MP3s (offline)")
    p.add_argument("-d", "--project", required=True)
    _add_select(p)

    p = sub.add_parser("concat", help="stitch section MP3s into one file")
    p.add_argument("-d", "--project", required=True)
    p.add_argument("-o", "--out", default=None)

    p = sub.add_parser("all", help="plan + narrate + synth + concat")
    p.add_argument("paper")
    p.add_argument("-d", "--project", required=True)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--words", type=int, default=700)
    _add_llm(p)

    args = ap.parse_args(argv)

    if args.cmd == "plan":
        m = pipeline.plan(args.paper, args.project, words_per_section=args.words)
        for s in m["sections"]:
            print(f"  {s['index']:2d}. {s['title']}  (~{s['words']} words)")
        print(f"manifest written to {args.project}/manifest.json")
    elif args.cmd == "narrate":
        pipeline.narrate_sections(args.project, args.sections,
                                  provider=args.provider, model=args.model)
    elif args.cmd == "synth":
        pipeline.synth_sections(args.project, args.sections)
    elif args.cmd == "concat":
        pipeline.concat(args.project, args.out)
    elif args.cmd == "all":
        pipeline.plan(args.paper, args.project, words_per_section=args.words)
        pipeline.narrate_sections(args.project, None,
                                  provider=args.provider, model=args.model)
        pipeline.synth_sections(args.project, None)
        pipeline.concat(args.project, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
