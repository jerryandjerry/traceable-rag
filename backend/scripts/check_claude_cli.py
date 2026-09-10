#!/usr/bin/env python
# ruff: noqa: T201 -- command-line probe reports its verdict to the operator
"""Prove the Claude CLI provider leaks no local context into its requests.

    python backend/scripts/check_claude_cli.py

The provider suppresses CLAUDE.md, the memory index, MCP servers and the tool
definitions using three env vars that the CLI reads but does not document. That
makes them a silent-failure risk: a future CLI release could ignore them and
every completion would quietly carry this machine's instructions. This asks the
model to enumerate its own context and fails if any of it came through.
"""
from __future__ import annotations

import asyncio
import re
import sys
import warnings

warnings.filterwarnings("ignore")

PROBE = (
    "Answer factually about your own context window. Is there any of the following "
    "present: a CLAUDE.md file, user or project instructions, a memory index or "
    "memory directory, MCP tools, tool definitions? Reply with one line per item "
    "in the form 'NAME: PRESENT' or 'NAME: ABSENT'. No other text."
)
# Wording varies, so match the leak rather than the phrasing.
LEAKS = {
    "CLAUDE.md": r"claude\.?md\s*:?\s*(present|yes)",
    "instructions": r"instructions?\s*:?\s*(present|yes)",
    "memory": r"memory[^:\n]*:?\s*(present|yes)",
    "MCP": r"mcp[^:\n]*:?\s*(present|yes)",
}


def main() -> int:
    from visionagent.providers.llm import build_llm

    llm = build_llm("claude-cli", effort="low")
    answer = asyncio.run(llm.complete(prompt=PROBE))
    print(answer, "\n")

    found = [n for n, pat in LEAKS.items() if re.search(pat, answer, re.I)]
    if found:
        print(f"LEAK: {', '.join(found)} reached the model.", file=sys.stderr)
        print("The CLI is ignoring the isolation env vars in "
              "visionagent/providers/llm/claude_cli.py:_ISOLATION_ENV.", file=sys.stderr)
        return 1
    print("clean: no CLAUDE.md, memory, instructions or MCP tools in context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
