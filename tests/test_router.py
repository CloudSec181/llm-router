"""Unit tests for the local classifier, policy gate, and audit log."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_router.audit import append_decision, load_decisions
from llm_router.classify import classify, score_task
from llm_router.policy import decide, route, route_and_record

ROOT = Path(__file__).resolve().parents[1]
CLASSIFY_PY = ROOT / "src" / "llm_router" / "classify.py"

COPILOT_TEXTS = (
    "draft an email to the team",
    "please send an e-mail",
    "build slides for the review",
    "make a powerpoint deck",
    "post an update in Teams",
    "put this on my calendar",
    "reply in Outlook",
)

CLAUDE_TEXTS = (
    "refactor this module",
    "refactor this function",
    "write a function that sorts a list",
    "please review this code",
    "please do a long-document review",
    "review this long document",
    "compare these designs",
    "open a pull request for the parser",
    "contract review of the API",
)

SENSITIVE_LABELS = (
    "phi",
    "HIPAA",
    "customer data",
    "customer-data",
    "Confidential",
    "confidentiality",
    "restricted",
    "ssn",
    "MRN",
)


def test_email_slides_teams_calendar_map_to_copilot():
    for text in COPILOT_TEXTS:
        decision = classify(text)
        assert decision.backend == "copilot", text
        assert decision.boundary_exit is False


@pytest.mark.parametrize("text", CLAUDE_TEXTS)
def test_refactor_code_long_document_map_to_claude_when_policy_allows(text):
    decision = classify(text)
    assert decision.backend == "claude"
    assert decision.boundary_exit is True
    assert decision.reason.startswith("task:")


def test_score_task_families_ignore_policy_labels():
    email = score_task("draft an email about PHI")
    assert email.family == "copilot"
    assert "email" in email.copilot_hits
    code = score_task("refactor this confidential function")
    assert code.family == "claude"


@pytest.mark.parametrize("label", SENSITIVE_LABELS)
def test_sensitive_label_blocks_code_and_long_document(label):
    text = f"Please refactor this function and review the long document. Label: {label}"
    decision = decide(text)
    assert decision.backend == "copilot"
    assert decision.boundary_exit is False
    assert decision.reason.startswith("sensitive:")
    assert classify(text) == decision


def test_sensitive_label_on_code_text_never_exits():
    text = "refactor this function; the spec is confidential"
    decision = route(text)
    assert decision.backend == "copilot"
    assert decision.boundary_exit is False
    assert decision.reason == "sensitive:confidential"


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None, "hello", "Hello!", "what time is it?"])
def test_empty_and_unknown_stay_on_copilot(text):
    decision = route(text)
    assert decision.backend == "copilot"
    assert decision.boundary_exit is False
    if text is None or not str(text).strip():
        assert decision.reason == "empty"
    else:
        assert decision.reason == "unknown"


def test_mixed_task_stays_on_copilot():
    decision = decide("refactor the email draft and update the slides")
    assert decision.backend == "copilot"
    assert decision.boundary_exit is False
    assert decision.reason == "mixed"


def test_short_sensitive_tokens_do_not_match_inside_words():
    assert decide("refactor this function about philosophy").backend == "claude"
    assert decide("refactor this function in a graphic design tool").backend == "claude"
    assert decide("refactor this function; access is unrestricted").backend == "claude"


def test_reason_does_not_echo_prompt_text():
    token = "Zzq-unique-token-991"
    decision = classify(f"refactor this function {token}")
    assert token not in decision.reason
    assert token not in decision.backend


def test_classify_source_has_no_network_surface():
    source = CLASSIFY_PY.read_text(encoding="utf-8")
    lowered = source.lower()
    for banned in (
        "socket",
        "urllib",
        "http://",
        "https://",
        "urlopen",
        "requests",
        "http.client",
        "aiohttp",
        "websocket",
        "ftp://",
        "www.",
    ):
        assert banned not in lowered
    tree = ast.parse(source)
    banned_roots = {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned_roots
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned_roots


def test_fresh_import_and_score_do_not_open_a_connection():
    code = (
        "import socket\n"
        "def deny(*_a, **_k):\n"
        "    raise SystemExit('connection attempted')\n"
        "socket.socket = deny\n"
        "socket.create_connection = deny\n"
        "socket.getaddrinfo = deny\n"
        "socket.create_server = deny\n"
        "from llm_router.classify import classify\n"
        "classify('draft an email')\n"
        "classify('refactor this function')\n"
        "classify('refactor this confidential function')\n"
        "print('ok')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_audit_line_has_required_fields_and_omits_canary(tmp_path):
    canary = "CANARY-unique-9f3a-do-not-log"
    text = f"Please refactor this function. Note {canary}"
    path = tmp_path / "audit" / "decisions.jsonl"
    decision = route_and_record(text, path)
    raw = path.read_text(encoding="utf-8")
    assert canary not in raw
    assert text not in raw
    lines = raw.splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row == {
        "backend": "claude",
        "reason": decision.reason,
        "boundary_exit": True,
        "prompt_chars": len(text),
    }
    assert isinstance(row["prompt_chars"], int)
    assert isinstance(row["boundary_exit"], bool)
    assert set(row) == {"backend", "reason", "boundary_exit", "prompt_chars"}


def test_sensitive_route_audit_omits_prompt(tmp_path):
    canary = "CANARY-confidential-body-44c1"
    text = f"refactor this function; confidential payload {canary}"
    path = tmp_path / "audit.jsonl"
    decision = route_and_record(text, path)
    assert decision.backend == "copilot"
    assert decision.boundary_exit is False
    raw = path.read_text(encoding="utf-8")
    assert canary not in raw
    assert "confidential payload" not in raw
    row = load_decisions(path)[0]
    assert row["backend"] == "copilot"
    assert row["boundary_exit"] is False
    assert row["prompt_chars"] == len(text)
    assert str(row["reason"]).startswith("sensitive:")


def test_each_successful_route_appends_one_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    route_and_record("hello", path)
    route_and_record("", path)
    route_and_record("refactor this function", path)
    rows = load_decisions(path)
    assert [row["backend"] for row in rows] == ["copilot", "copilot", "claude"]
    assert rows[0]["prompt_chars"] == len("hello")
    assert rows[1]["prompt_chars"] == 0
    assert rows[2]["boundary_exit"] is True
    raw = path.read_text(encoding="utf-8")
    assert "hello" not in raw


ORDINARY_CLAUDE_TEXTS = (
    "hard reasoning about caching",
    "reason step by step about the outage",
    "design comparison",
    "compare designs for the cache layer",
    "summarize this 80-page report",
    "this document is very long, please summarize it",
    "review the attached whitepaper",
    "a python def block",
    "a fenced python block",
    "write a python script that parses csv",
    "debug the stack trace in the parser module",
    "def add(a, b):\n    return a - b",
    "```python\nprint('ok')\n```",
)

STAY_COPILOT_TEXTS = (
    "draft an email to the team",
    "refactor this confidential customer data function",
)


@pytest.mark.parametrize("text", ORDINARY_CLAUDE_TEXTS)
def test_ordinary_claude_family_wording_exits_when_policy_allows(text):
    decision = classify(text)
    assert decision.backend == "claude", text
    assert decision.boundary_exit is True
    assert decision.reason.startswith("task:")
    scored = score_task(text)
    assert scored.family == "claude"


@pytest.mark.parametrize("text", STAY_COPILOT_TEXTS)
def test_email_and_sensitive_code_stay_on_copilot(text):
    decision = classify(text)
    assert decision.backend == "copilot", text
    assert decision.boundary_exit is False


@pytest.mark.parametrize("text", ORDINARY_CLAUDE_TEXTS)
@pytest.mark.parametrize("label", ("confidential", "customer data", "PHI", "HIPAA", "restricted", "ssn", "mrn"))
def test_sensitive_label_blocks_ordinary_claude_wording(text, label):
    decision = classify(f"{text}\nLabel: {label}")
    assert decision.backend == "copilot"
    assert decision.boundary_exit is False
    assert decision.reason.startswith("sensitive:")


def test_short_office_writing_stays_unknown_copilot():
    decision = classify("short Office writing for the all-hands")
    assert decision.backend == "copilot"
    assert decision.boundary_exit is False
    assert decision.reason == "unknown"


def test_append_decision_rejects_non_integer_length(tmp_path):
    path = tmp_path / "audit.jsonl"
    with pytest.raises(TypeError):
        append_decision(
            path,
            backend="copilot",
            reason="unknown",
            boundary_exit=False,
            prompt_chars=True,  # type: ignore[arg-type]
        )
    assert not path.exists()
