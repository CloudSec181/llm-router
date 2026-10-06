# llm-router

Local OpenAI-compatible router. Fixture backends answer the prompt. The server does not call Claude or Copilot.

No Claude or Copilot key is required.

## Start

From the repository root:

```bash
PYTHONPATH=src python -m llm_router.server
```

That listens on `http://127.0.0.1:8000`. The module is `llm_router.server`.

## Example

One curl for an email prompt:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"draft an email to the team"}]}'
```

The fixture routes that prompt to copilot. No API key is sent.
