# Project operational guidance

Local fixture router. It does not call Claude or Copilot and does not need an API key.

Start from the repository root:

    PYTHONPATH=src python -m llm_router.server

The process listens on 127.0.0.1:8000 only.

Verify:

    python -m pytest -q

Do not commit .env files, tokens, or audit logs.
