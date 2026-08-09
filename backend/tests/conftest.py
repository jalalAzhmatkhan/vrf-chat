"""Shared test environment setup.

Sets dummy-but-syntactically-valid environment variables required for
fail-fast provider validation (`app.llm_providers.factory`,
`app.storage.factory`) *before* any test module imports `app.main` (which
triggers `create_app()` at module import time). This keeps the test suite
independent of whatever a developer's local `.env` happens to contain, and
makes it reproducible in CI where no `.env` file exists at all.

Uses `setdefault` so a developer's real `.env`/shell env is respected if
already set (e.g. to exercise a specific provider locally).
"""

import os

os.environ.setdefault("CHAT_LLM_PROVIDER", "anthropic")
os.environ.setdefault("CHAT_LLM_MODEL", "claude-sonnet-4-5-20250929")
os.environ.setdefault("CHAT_LLM_API_KEY", "sk-ant-test-dummy")
os.environ.setdefault("EVAL_JUDGE_LLM_PROVIDER", "google")
os.environ.setdefault("EVAL_JUDGE_LLM_MODEL", "gemini-2.5-pro")
os.environ.setdefault("EVAL_JUDGE_LLM_API_KEY", "AIza-test-dummy")

os.environ.setdefault("OBJECT_STORAGE_BACKEND", "local")
os.environ.setdefault("OBJECT_STORAGE_LOCAL_BASE_PATH", "./data/object-storage-test")
os.environ.setdefault("OBJECT_STORAGE_LOCAL_TOKEN_SECRET", "test-only-local-storage-token-secret")
