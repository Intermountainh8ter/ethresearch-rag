from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import RagPipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ethrag")
    p.add_argument("question")
    p.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data" / "eips"))
    p.add_argument("--llm", action="store_true", help="use Anthropic (needs ANTHROPIC_API_KEY)")
    a = p.parse_args(argv)
    llm = None
    if a.llm:
        from .llm import AnthropicLLM

        llm = AnthropicLLM()
    ans = RagPipeline.from_directory(a.data, llm).ask(a.question)
    print(json.dumps({"verified": ans.verified, "answer": ans.text, "retrieved": ans.retrieved,
                      "issues": ans.issues}, indent=2))
    return 0 if ans.verified else 2


if __name__ == "__main__":
    sys.exit(main())
