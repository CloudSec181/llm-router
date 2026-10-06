"""JSONL decision log. Stores routing metadata, never the prompt text."""

from __future__ import annotations

import json
import os
from pathlib import Path

_FIELDS = ("backend", "reason", "boundary_exit", "prompt_chars")
_BACKENDS = frozenset({"copilot", "claude"})


def append_decision(
    path: str | Path,
    *,
    backend: str,
    reason: str,
    boundary_exit: bool,
    prompt_chars: int,
) -> dict[str, object]:
    """Append one JSON line with the four audit fields and nothing else."""
    if backend not in _BACKENDS:
        raise ValueError("backend must be copilot or claude")
    if not isinstance(reason, str) or not reason:
        raise ValueError("reason must be a non-empty string")
    if not isinstance(boundary_exit, bool):
        raise TypeError("boundary_exit must be a bool")
    if isinstance(prompt_chars, bool) or not isinstance(prompt_chars, int):
        raise TypeError("prompt_chars must be an int")
    if prompt_chars < 0:
        raise ValueError("prompt_chars must be >= 0")

    record = {
        "backend": backend,
        "reason": reason,
        "boundary_exit": boundary_exit,
        "prompt_chars": prompt_chars,
    }
    if tuple(record) != _FIELDS:
        raise RuntimeError("audit record fields drifted")

    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
    if "\n" in line or "\r" in line:
        raise RuntimeError("audit line must be a single JSON object")

    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def load_decisions(path: str | Path) -> list[dict[str, object]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError("audit line is not an object")
        rows.append(parsed)
    return rows
