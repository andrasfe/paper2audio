"""LLM provider abstraction: direct Anthropic or direct OpenAI API.

Both providers take (system, user) prompts and return the completion
text. Selection is by name so the CLI can expose --provider.
"""
from __future__ import annotations

import os

DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5",
}


def complete(
    system: str,
    user: str,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 16000,
    api_key: str | None = None,
) -> str:
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"unknown provider {provider!r}; choose from {sorted(DEFAULT_MODELS)}"
        )
    model = model or DEFAULT_MODELS[provider]
    if provider == "anthropic":
        return _anthropic(system, user, model, max_tokens, api_key)
    return _openai(system, user, model, max_tokens, api_key)


def _anthropic(
    system: str, user: str, model: str, max_tokens: int, api_key: str | None
) -> str:
    import anthropic

    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
    )
    # Streaming: narration chapters are long generations.
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        message = stream.get_final_message()
    return "".join(b.text for b in message.content if b.type == "text")


def _openai(
    system: str, user: str, model: str, max_tokens: int, api_key: str | None
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
