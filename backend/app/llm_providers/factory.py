"""LLM provider abstraction factory.

See `Documentation/system-design/04-provider-abstractions.md` Bagian A. A
single generic factory, `build_model(prefix, settings)`, is called with two
different env prefixes — `"CHAT_LLM"` (the model that answers users, used by
`app/agent/`) and `"EVAL_JUDGE_LLM"` (LLM-as-judge, used by `app/evaluation/`)
— so switching either provider is purely a `.env` change, never a code
change.

Pydantic AI already provides the per-vendor `Model`/`Provider` abstraction;
this module does not reimplement it, it only wires env vars to the right
`Model`/`Provider` classes and fails fast on invalid combinations.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from app.core.config import Settings
from app.core.observability import get_logger

SUPPORTED_PROVIDERS = ("anthropic", "google", "openai", "local")

logger = get_logger(__name__)


class LLMProviderConfigError(ValueError):
    """Raised (fail-fast) when a `{prefix}_*` env combination is invalid."""


def _require(value: str | None, field_name: str) -> str:
    if not value:
        raise LLMProviderConfigError(f"'{field_name}' is required but not set.")
    return value


def build_model(prefix: str, settings: Settings) -> Model:
    """Build a Pydantic AI `Model` from `{prefix}_PROVIDER`/`{prefix}_MODEL`/etc.

    `prefix` is e.g. `"CHAT_LLM"` or `"EVAL_JUDGE_LLM"`. Raises
    `LLMProviderConfigError` for invalid/incomplete configuration instead of
    failing silently at first real use — call this during app startup (see
    `validate_llm_providers_or_raise`) so misconfiguration is caught
    immediately, not on the first chat/eval request.
    """
    provider_name = getattr(settings, f"{prefix}_PROVIDER")
    model_name = _require(getattr(settings, f"{prefix}_MODEL"), f"{prefix}_MODEL")
    api_key = getattr(settings, f"{prefix}_API_KEY")
    base_url = getattr(settings, f"{prefix}_BASE_URL") or None

    match provider_name:
        case "anthropic":
            api_key = _require(api_key, f"{prefix}_API_KEY")
            return AnthropicModel(
                model_name, provider=AnthropicProvider(api_key=api_key, base_url=base_url)
            )
        case "google":
            api_key = _require(api_key, f"{prefix}_API_KEY")
            return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key))
        case "openai":
            api_key = _require(api_key, f"{prefix}_API_KEY")
            return OpenAIChatModel(
                model_name, provider=OpenAIProvider(api_key=api_key, base_url=base_url)
            )
        case "local":
            # Local (Ollama/vLLM) is always OpenAI-compatible — reuse the
            # OpenAI adapter with a mandatory base_url override.
            base_url = _require(base_url, f"{prefix}_BASE_URL")
            return OpenAIChatModel(
                model_name,
                provider=OpenAIProvider(api_key=api_key or "not-needed", base_url=base_url),
            )
        case _:
            raise LLMProviderConfigError(
                f"{prefix}_PROVIDER: unknown provider '{provider_name}', "
                f"must be one of {SUPPORTED_PROVIDERS}"
            )


def build_model_settings(prefix: str, settings: Settings) -> ModelSettings:
    """Build the `ModelSettings` (temperature/max_tokens/timeout) for `prefix`."""
    return ModelSettings(
        temperature=getattr(settings, f"{prefix}_TEMPERATURE"),
        max_tokens=getattr(settings, f"{prefix}_MAX_TOKENS"),
        timeout=getattr(settings, f"{prefix}_TIMEOUT_SECONDS"),
    )


def validate_llm_providers_or_raise(settings: Settings) -> None:
    """Fail-fast validation of both LLM slots — call once at app startup.

    Also logs a (non-blocking) warning if `CHAT_LLM_*` and
    `EVAL_JUDGE_LLM_*` are configured identically, since using the same
    model to both answer and judge risks self-serving bias (see
    04-provider-abstractions.md Bagian A §2).
    """
    build_model("CHAT_LLM", settings)
    build_model("EVAL_JUDGE_LLM", settings)

    if (
        settings.CHAT_LLM_PROVIDER == settings.EVAL_JUDGE_LLM_PROVIDER
        and settings.CHAT_LLM_MODEL == settings.EVAL_JUDGE_LLM_MODEL
    ):
        logger.warning(
            "CHAT_LLM and EVAL_JUDGE_LLM are configured identically "
            "(provider=%s, model=%s) — evaluation runs will be self-judged, "
            "which risks self-serving bias. Acceptable temporarily before a "
            "second provider's credentials are available, but not ideal.",
            settings.CHAT_LLM_PROVIDER,
            settings.CHAT_LLM_MODEL,
        )
