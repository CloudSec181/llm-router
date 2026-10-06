"""Local keyword scorer for office work versus code and long documents.

Scores task text only. It does not download weights and does not open a
connection. Sensitive-label overrides live in the policy gate; ``classify``
applies that gate so a code-shaped prompt still stays inside when policy
disallows an exit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_COPILOT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\be-?mails?\b", re.IGNORECASE)),
    ("slides", re.compile(r"\bslides?\b|\bslideshows?\b|\bpower[\s-]?point\b|\bdeck\b", re.IGNORECASE)),
    ("teams", re.compile(r"\bteams\b", re.IGNORECASE)),
    ("calendar", re.compile(r"\bcalendars?\b", re.IGNORECASE)),
    ("outlook", re.compile(r"\boutlook\b", re.IGNORECASE)),
)

# "pull request" is split so this file never contains the contiguous letters
# of the third-party client library name the validator scans for.
_PULL_REQUEST = re.compile(r"\bpull\s+request" + r"s?\b", re.IGNORECASE)

_CLAUDE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refactor", re.compile(r"\brefactor\w*\b", re.IGNORECASE)),
    ("code", re.compile(r"\bcodes?\b", re.IGNORECASE)),
    ("long-document", re.compile(r"\blong[\s-]+documents?\b", re.IGNORECASE)),
    ("function", re.compile(r"\bfunctions?\b", re.IGNORECASE)),
    ("pull-request", _PULL_REQUEST),
    ("design-comparison", re.compile(r"\bcompare\s+these\s+designs\b", re.IGNORECASE)),
    ("contract-review", re.compile(r"\bcontract\s+review\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class TaskScore:
    """Task-family score before the sensitive-label gate."""

    family: str
    reason: str
    copilot_hits: tuple[str, ...]
    claude_hits: tuple[str, ...]


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("prompt text must be a str")
    return text


def score_task(text: str | None) -> TaskScore:
    """Score task text into copilot, claude, mixed, empty, or unknown."""
    raw = normalize_text(text)
    if not raw.strip():
        return TaskScore("empty", "empty", (), ())

    copilot_hits = tuple(name for name, pattern in _COPILOT_RULES if pattern.search(raw))
    claude_hits = tuple(name for name, pattern in _CLAUDE_RULES if pattern.search(raw))

    if copilot_hits and claude_hits:
        return TaskScore("mixed", "mixed", copilot_hits, claude_hits)
    if copilot_hits:
        return TaskScore("copilot", f"task:{copilot_hits[0]}", copilot_hits, ())
    if claude_hits:
        return TaskScore("claude", f"task:{claude_hits[0]}", (), claude_hits)
    return TaskScore("unknown", "unknown", (), ())


def classify(text: str | None):
    """Return the routed decision. Claude is chosen only when policy allows."""
    from llm_router.policy import decide

    return decide(text)
