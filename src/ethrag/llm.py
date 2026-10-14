from __future__ import annotations

import os
from typing import Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"


def api_key_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 600):
        import anthropic  # optional: pip install ethrag[llm]

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ETHRAG_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in r.content if b.type == "text")


class ScriptedLLM:
    """Replays canned replies; records every (system, user) call. Used by tests and demos."""

    def __init__(self, replies: list[str]):
        self.replies, self.calls = list(replies), []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self.replies.pop(0)
