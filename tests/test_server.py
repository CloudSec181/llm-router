"""HTTP tests for the OpenAI-compatible fixture server."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from llm_router.server import HOST, make_server

ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = ROOT / "src" / "llm_router" / "server.py"


def _post(base: str, raw: bytes) -> tuple[int, dict[str, str], object]:
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read()
            headers = {key: value for key, value in response.headers.items()}
            return response.status, headers, json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw_body = exc.read()
        headers = {key: value for key, value in exc.headers.items()}
        try:
            parsed: object = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = raw_body
        return exc.code, headers, parsed


def _chat(base: str, content: str, model: str = "gpt-4") -> tuple[int, dict[str, str], dict]:
    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": content}]}
    ).encode("utf-8")
    status, headers, body = _post(base, payload)
    assert isinstance(body, dict)
    return status, headers, body


@pytest.fixture
def router(tmp_path):
    audit = tmp_path / "audit.jsonl"
    server = make_server(0, audit)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield {
            "base": f"http://{host}:{port}",
            "host": host,
            "port": port,
            "audit": audit,
            "server": server,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_source_binds_loopback_only():
    source = SERVER_PY.read_text(encoding="utf-8")
    assert HOST == "127.0.0.1"
    assert 'HOST = "127.0.0.1"' in source
    assert "0.0.0.0" not in source
    for banned in ("api.openai.com", "api.anthropic.com", "openai.azure.com"):
        assert banned not in source.lower()


def test_make_server_address_is_loopback(tmp_path):
    server = make_server(0, tmp_path / "audit.jsonl")
    try:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        assert port != 0
        assert server.server_address[0] != "0.0.0.0"
    finally:
        server.server_close()


def test_email_completion_is_copilot(router):
    status, headers, body = _chat(router["base"], "draft an email to the team")
    assert status == 200
    assert body["object"] == "chat.completion"
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert "copilot" in message["content"]
    assert "claude" not in message["content"]
    assert body["x_router"]["backend"] == "copilot"
    assert body["x_router"]["reason"] == "task:email"
    assert body["x_router"]["boundary_exit"] is False
    assert headers["X-LLM-Router-Backend"] == "copilot"
    assert headers["X-LLM-Router-Boundary-Exit"] == "false"
    row = json.loads(router["audit"].read_text(encoding="utf-8").splitlines()[0])
    assert row["backend"] == "copilot"
    assert "draft an email" not in router["audit"].read_text(encoding="utf-8")


def test_refactor_completion_is_claude(router):
    status, headers, body = _chat(router["base"], "please refactor this function")
    assert status == 200
    assert body["object"] == "chat.completion"
    content = body["choices"][0]["message"]["content"]
    assert "claude" in content
    assert "copilot" not in content
    assert body["x_router"]["backend"] == "claude"
    assert body["x_router"]["boundary_exit"] is True
    assert body["x_router"]["reason"] == "task:refactor"
    assert headers["X-LLM-Router-Backend"] == "claude"
    assert headers["X-LLM-Router-Boundary-Exit"] == "true"


def test_confidential_code_stays_on_copilot(router):
    canary = "CANARY-confidential-code-55e2"
    text = f"refactor this function; the spec is confidential {canary}"
    status, headers, body = _chat(router["base"], text)
    assert status == 200
    content = body["choices"][0]["message"]["content"]
    assert body["x_router"]["backend"] == "copilot"
    assert body["x_router"]["boundary_exit"] is False
    assert body["x_router"]["reason"] == "sensitive:confidential"
    assert "copilot" in content
    assert "claude" not in content
    assert headers["X-LLM-Router-Backend"] == "copilot"
    assert headers["X-LLM-Router-Boundary-Exit"] == "false"
    raw = router["audit"].read_text(encoding="utf-8")
    assert canary not in raw
    assert text not in raw
    row = json.loads(raw.splitlines()[0])
    assert row["backend"] == "copilot"
    assert row["boundary_exit"] is False
    assert row["reason"] == "sensitive:confidential"
    assert row["prompt_chars"] == len(text)


def test_bad_json_is_400_and_does_not_audit(router):
    canary = "CANARY-bad-json-77ab-do-not-log"
    status, _headers, body = _post(router["base"], b"{not-json " + canary.encode("ascii"))
    assert status == 400
    assert isinstance(body, dict)
    assert canary not in json.dumps(body)
    assert not router["audit"].exists()


def test_missing_messages_is_400_and_does_not_audit(router):
    canary = "CANARY-missing-messages-91c0"
    raw = json.dumps({"model": "gpt-4", "note": canary}).encode("utf-8")
    status, _headers, body = _post(router["base"], raw)
    assert status == 400
    assert isinstance(body, dict)
    assert body["error"]["message"] == "missing messages"
    assert canary not in json.dumps(body)
    assert not router["audit"].exists()

    empty = json.dumps({"messages": []}).encode("utf-8")
    status, _headers, body = _post(router["base"], empty)
    assert status == 400
    assert not router["audit"].exists()


def test_bad_request_after_success_does_not_append_body(router):
    status, _headers, _body = _chat(router["base"], "draft an email")
    assert status == 200
    before = router["audit"].read_text(encoding="utf-8")
    canary = "CANARY-second-bad-body-4401"
    status, _headers, body = _post(router["base"], f"[{canary}]".encode("utf-8"))
    assert status == 400
    assert canary not in json.dumps(body)
    after = router["audit"].read_text(encoding="utf-8")
    assert after == before
    assert canary not in after


def test_successful_canary_is_absent_from_audit(router):
    canary = "CANARY-http-unique-18de"
    text = f"please refactor this function {canary}"
    status, _headers, body = _chat(router["base"], text)
    assert status == 200
    assert body["x_router"]["backend"] == "claude"
    assert canary not in body["choices"][0]["message"]["content"]
    assert canary not in body["x_router"]["reason"]
    raw = router["audit"].read_text(encoding="utf-8")
    assert canary not in raw
    row = json.loads(raw.splitlines()[0])
    assert set(row) == {"backend", "reason", "boundary_exit", "prompt_chars"}
    assert row["prompt_chars"] == len(text)


def _listeners_for_port(port: int) -> list[str]:
    """Return local hex addresses from /proc/net/tcp listening on port."""
    hex_port = f"{port:04X}"
    found: list[str] = []
    text = Path("/proc/net/tcp").read_text(encoding="utf-8")
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[1]
        state = parts[3]
        ip_hex, port_hex = local.split(":")
        if port_hex.upper() == hex_port and state == "0A":
            found.append(ip_hex.upper())
    return found


def test_cli_process_listens_on_loopback(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    audit = tmp_path / "audit.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llm_router.server",
            "--port",
            str(port),
            "--audit",
            str(audit),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 5
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(f"server exited early: {proc.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    ready = True
                    break
            except OSError:
                time.sleep(0.05)
        assert ready

        listeners = _listeners_for_port(port)
        assert listeners, f"no listener on port {port}"
        assert all(item == "0100007F" for item in listeners)
        assert "00000000" not in listeners

        ss = subprocess.run(
            ["ss", "-ltn", f"sport = :{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if ss.returncode == 0:
            local_addrs = []
            for line in ss.stdout.splitlines():
                if "127.0.0.1:" not in line and "0.0.0.0:" not in line:
                    continue
                fields = line.split()
                # LISTEN lines: State Recv-Q Send-Q Local Peer
                if len(fields) >= 5 and fields[0] == "LISTEN":
                    local_addrs.append(fields[3])
            assert local_addrs, ss.stdout
            assert all(addr.startswith("127.0.0.1:") for addr in local_addrs)
            assert all(not addr.startswith("0.0.0.0:") for addr in local_addrs)

        status, headers, body = _chat(f"http://127.0.0.1:{port}", "draft an email")
        assert status == 200
        assert body["x_router"]["backend"] == "copilot"
        assert headers["X-LLM-Router-Backend"] == "copilot"
        assert "draft an email" not in audit.read_text(encoding="utf-8")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
