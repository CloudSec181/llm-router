"""OpenAI-compatible fixture server. Binds 127.0.0.1 only.

Fixture backends name themselves in the completion text. They do not call
Claude, Copilot, or any other vendor. No API key is required.

Start::

    python -m llm_router.server

POST /v1/chat/completions on http://127.0.0.1:8000
Audit file (metadata only): audit/decisions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_router.policy import Decision, route_and_record

# Conference laptops must not expose the router on the LAN.
HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_AUDIT_PATH = Path("audit") / "decisions.jsonl"
_COMPLETIONS_PATH = "/v1/chat/completions"
_MAX_BODY = 8_000_000
_AUDIT_LOCK = threading.Lock()


class LocalRouterServer(ThreadingHTTPServer):
    """Threading HTTP server that refuses any listen address except loopback."""

    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self) -> None:
        host = self.server_address[0]
        if host != HOST:
            raise OSError(f"refusing to bind {host}; listen address is {HOST}")
        super().server_bind()
        bound = self.server_address[0]
        if bound != HOST:
            raise OSError(f"refusing to listen on {bound}; listen address is {HOST}")


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    bits: list[str] = []
    for part in content:
        if isinstance(part, str):
            bits.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                bits.append(text)
    return "\n".join(bits)


def prompt_from_messages(messages: list[object]) -> str:
    """Join message text for the policy gate. Non-text parts are skipped."""
    chunks: list[str] = []
    for message in messages:
        if isinstance(message, str):
            chunks.append(message)
            continue
        if not isinstance(message, dict):
            continue
        text = _content_text(message.get("content"))
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def completion_payload(decision: Decision, model: object, prompt: str) -> dict[str, object]:
    """OpenAI chat.completion plus x_router. Content names the fixture backend."""
    name = decision.backend
    content = f"{name} fixture completion"
    if not isinstance(model, str) or not model.strip():
        model = f"fixture-{name}"
    prompt_tokens = len(prompt) // 4
    completion_tokens = len(content.split())
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "x_router": {
            "backend": decision.backend,
            "reason": decision.reason,
            "boundary_exit": decision.boundary_exit,
        },
    }


def _error_payload(message: str) -> dict[str, object]:
    return {"error": {"message": message, "type": "invalid_request_error"}}


def build_handler(audit_path: Path) -> type[BaseHTTPRequestHandler]:
    """Handler class closed over one audit path. The raw body is never logged."""

    class ChatHandler(BaseHTTPRequestHandler):
        server_version = "llm-router/0.1.0"
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
            path = self.path.split("?", 1)[0]
            if path != _COMPLETIONS_PATH and path != _COMPLETIONS_PATH + "/":
                self._send_json(404, _error_payload("not found"))
                return

            body = self._read_body()
            if body is None:
                self.close_connection = True
                self._send_json(400, _error_payload("invalid JSON"))
                return

            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Do not log the exception: the decoder message can quote the body.
                self._send_json(400, _error_payload("invalid JSON"))
                return

            messages = payload.get("messages") if isinstance(payload, dict) else None
            if not isinstance(messages, list) or not messages:
                self._send_json(400, _error_payload("missing messages"))
                return

            prompt = prompt_from_messages(messages)
            try:
                with _AUDIT_LOCK:
                    decision = route_and_record(prompt, audit_path)
            except Exception:
                self._send_json(500, _error_payload("audit write failed"))
                return

            model = payload.get("model") if isinstance(payload, dict) else None
            response = completion_payload(decision, model, prompt)
            self._send_json(
                200,
                response,
                {
                    "X-LLM-Router-Backend": decision.backend,
                    "X-LLM-Router-Boundary-Exit": "true" if decision.boundary_exit else "false",
                },
            )

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            self._send_json(405, _error_payload("method not allowed"))

        def log_message(self, fmt: str, *args: object) -> None:
            # Request line only. Never pass the body into this format string.
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _read_body(self) -> bytes | None:
            header = self.headers.get("Content-Length")
            if header is None:
                return None
            try:
                length = int(header)
            except ValueError:
                return None
            if length < 0 or length > _MAX_BODY:
                return None
            return self.rfile.read(length)

        def _send_json(
            self,
            status: int,
            payload: dict[str, object],
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw)

    return ChatHandler


def make_server(port: int, audit_path: str | Path) -> LocalRouterServer:
    """Bind 127.0.0.1:port. port 0 picks an ephemeral port for tests."""
    path = Path(audit_path).expanduser()
    handler = build_handler(path)
    return LocalRouterServer((HOST, port), handler)


def serve(port: int = DEFAULT_PORT, audit_path: str | Path = DEFAULT_AUDIT_PATH) -> None:
    """Serve until interrupted. The listen address is always 127.0.0.1."""
    try:
        server = make_server(port, audit_path)
    except OSError as exc:
        raise SystemExit(f"cannot listen on {HOST}:{port}: {exc}") from exc
    bound_host, bound_port = server.server_address[:2]
    if bound_host != HOST:
        server.server_close()
        raise SystemExit(f"refusing to listen on {bound_host}")
    audit = Path(audit_path).expanduser()
    print(f"llm-router listening on {HOST}:{bound_port}", flush=True)
    print(f"audit {audit.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Local OpenAI-compatible router. Listens on 127.0.0.1 only."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LLM_ROUTER_PORT", str(DEFAULT_PORT))),
    )
    parser.add_argument(
        "--audit",
        default=os.environ.get("LLM_ROUTER_AUDIT", str(DEFAULT_AUDIT_PATH)),
    )
    args = parser.parse_args(argv)
    if args.port < 0 or args.port > 65535:
        raise SystemExit("port must be between 0 and 65535")
    serve(args.port, args.audit)


if __name__ == "__main__":
    main()
