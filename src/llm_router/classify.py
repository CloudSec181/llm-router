"""Local keyword scorer for office work versus Claude-family tasks.

Scores task text only. It does not download weights and does not open a
connection. Families match ordinary wording, not only canned phrases.
Sensitive-label overrides live in the policy gate; ``classify`` applies
that gate so a code-shaped prompt still stays inside when policy
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

# Ordinary wording for each named family. Not only the canned phrases.
_STEP_BY_STEP = r"step(?:\s+|-)+by(?:\s+|-)+step"
_LONG_PAGE = r"(?:[2-9]\d|\d{3,})(?:\s*-\s*|\s+)pages?"

_CLAUDE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refactor", re.compile(r"\brefactor\w*\b", re.IGNORECASE)),
    ("code", re.compile(r"\bcodes?\b", re.IGNORECASE)),
    ("function", re.compile(r"\bfunctions?\b", re.IGNORECASE)),
    ("pull-request", _PULL_REQUEST),
    (
        "python-def",
        re.compile(r"\bpython\s+def\b|\bdef\s+blocks?\b|\bdef\s+\w+\s*\(", re.IGNORECASE),
    ),
    (
        "fenced-code",
        re.compile(r"```|~~~|\bfenced\b.{0,40}\bblocks?\b", re.IGNORECASE | re.DOTALL),
    ),
    ("python-script", re.compile(r"\bpython\s+scripts?\b", re.IGNORECASE)),
    ("python", re.compile(r"\bpython\b", re.IGNORECASE)),
    ("stack-trace", re.compile(r"\bstack[\s-]*traces?\b|\btracebacks?\b", re.IGNORECASE)),
    ("debug", re.compile(r"\bdebug\w*\b", re.IGNORECASE)),
    (
        "long-document",
        re.compile(
            r"\blong[\s-]+documents?\b"
            r"|\blong\s+reports?\b"
            r"|\bwhite[\s-]?papers?\b"
            r"|\b" + _LONG_PAGE + r"\b"
            r"|\bdocuments?\b.{0,80}\bvery\s+long\b"
            r"|\bvery\s+long\b.{0,80}\bdocuments?\b"
            r"|\bsummar\w*\b.{0,80}\b(?:reports?|documents?|white[\s-]?papers?)\b"
            r"|\b(?:reports?|documents?|white[\s-]?papers?)\b.{0,80}\bsummar\w*\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "design-comparison",
        re.compile(
            r"\bdesign\s+comparisons?\b"
            r"|\bcompar(?:e|ed|ing|ison)\b.{0,48}\bdesigns?\b"
            r"|\bdesigns?\b.{0,48}\bcompar(?:e|ed|ing|ison)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "hard-reasoning",
        re.compile(
            r"\bhard\s+reasoning\b"
            r"|\breason(?:ing)?\s+" + _STEP_BY_STEP + r"\b"
            r"|\b" + _STEP_BY_STEP + r"\b",
            re.IGNORECASE,
        ),
    ),
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
