"""Policy gate. Runs before any backend call.

Sensitive labels force copilot and boundary_exit false, including when the
task text also looks like code or a long document. Empty, mixed, and
unrecognized text stay on copilot. Claude is allowed only for a clean
code or long-document score, and then boundary_exit is true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from llm_router.classify import normalize_text, score_task

SENSITIVE_LABELS: tuple[str, ...] = (
    "phi",
    "hipaa",
    "customer data",
    "confidential",
    "restricted",
    "ssn",
    "mrn",
)

_SENSITIVE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("phi", re.compile(r"\bphi\b", re.IGNORECASE)),
    ("hipaa", re.compile(r"\bhipaa\b", re.IGNORECASE)),
    ("customer data", re.compile(r"\bcustomer[\s_-]+data\b", re.IGNORECASE)),
    ("confidential", re.compile(r"\bconfidential\w*\b", re.IGNORECASE)),
    ("restricted", re.compile(r"\brestricted\w*\b", re.IGNORECASE)),
    ("ssn", re.compile(r"\bssns?\b", re.IGNORECASE)),
    ("mrn", re.compile(r"\bmrns?\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Decision:
    backend: str
    reason: str
    boundary_exit: bool


def find_sensitive(text: str | None) -> tuple[str, ...]:
    raw = normalize_text(text)
    return tuple(label for label, pattern in _SENSITIVE_RULES if pattern.search(raw))


def decide(text: str | None) -> Decision:
    """Route one prompt. Policy labels win before the task score."""
    hits = find_sensitive(text)
    if hits:
        return Decision("copilot", f"sensitive:{hits[0]}", False)

    scored = score_task(text)
    if scored.family == "claude":
        return Decision("claude", scored.reason, True)
    return Decision("copilot", scored.reason, False)


def route(text: str | None) -> Decision:
    """Alias used by later callers. Same gate as ``decide``."""
    return decide(text)


def route_and_record(text: str | None, audit_path: str | Path) -> Decision:
    """Decide, then append one audit line. The prompt text is not written."""
    from llm_router.audit import append_decision

    raw = normalize_text(text)
    decision = decide(raw)
    append_decision(
        audit_path,
        backend=decision.backend,
        reason=decision.reason,
        boundary_exit=decision.boundary_exit,
        prompt_chars=len(raw),
    )
    return decision
