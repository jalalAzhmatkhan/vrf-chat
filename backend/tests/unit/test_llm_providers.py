from types import SimpleNamespace

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel

from app.core.config import Settings
from app.llm_providers.factory import (
    LLMProviderConfigError,
    build_model,
    build_model_settings,
    validate_llm_providers_or_raise,
)


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "CHAT_LLM_PROVIDER": "anthropic",
        "CHAT_LLM_MODEL": "claude-sonnet-4-5-20250929",
        "CHAT_LLM_API_KEY": "sk-ant-test",
        "EVAL_JUDGE_LLM_PROVIDER": "google",
        "EVAL_JUDGE_LLM_MODEL": "gemini-2.5-pro",
        "EVAL_JUDGE_LLM_API_KEY": "AIza-test",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_build_model_anthropic() -> None:
    settings = _settings()

    model = build_model("CHAT_LLM", settings)

    assert isinstance(model, AnthropicModel)


def test_build_model_google() -> None:
    settings = _settings()

    model = build_model("EVAL_JUDGE_LLM", settings)

    assert isinstance(model, GoogleModel)


def test_build_model_openai() -> None:
    settings = _settings(
        CHAT_LLM_PROVIDER="openai", CHAT_LLM_MODEL="gpt-4.1", CHAT_LLM_API_KEY="sk-oai"
    )

    model = build_model("CHAT_LLM", settings)

    assert isinstance(model, OpenAIChatModel)


def test_build_model_local_reuses_openai_adapter() -> None:
    settings = _settings(
        CHAT_LLM_PROVIDER="local",
        CHAT_LLM_MODEL="llama3.1:8b-instruct",
        CHAT_LLM_API_KEY=None,
        CHAT_LLM_BASE_URL="http://localhost:11434/v1",
    )

    model = build_model("CHAT_LLM", settings)

    assert isinstance(model, OpenAIChatModel)


def test_build_model_local_requires_base_url() -> None:
    settings = _settings(CHAT_LLM_PROVIDER="local", CHAT_LLM_BASE_URL=None)

    with pytest.raises(LLMProviderConfigError, match="CHAT_LLM_BASE_URL"):
        build_model("CHAT_LLM", settings)


def test_build_model_requires_api_key_for_cloud_providers() -> None:
    settings = _settings(CHAT_LLM_API_KEY=None)

    with pytest.raises(LLMProviderConfigError, match="CHAT_LLM_API_KEY"):
        build_model("CHAT_LLM", settings)


def test_build_model_requires_model_name() -> None:
    settings = _settings(CHAT_LLM_MODEL="")

    with pytest.raises(LLMProviderConfigError, match="CHAT_LLM_MODEL"):
        build_model("CHAT_LLM", settings)


def test_build_model_unknown_provider_raises() -> None:
    fake_settings = SimpleNamespace(
        CHAT_LLM_PROVIDER="unsupported-vendor",
        CHAT_LLM_MODEL="some-model",
        CHAT_LLM_API_KEY="key",
        CHAT_LLM_BASE_URL=None,
    )

    with pytest.raises(LLMProviderConfigError, match="unknown provider"):
        build_model("CHAT_LLM", fake_settings)  # type: ignore[arg-type]


def test_build_model_settings_reads_generation_params() -> None:
    settings = _settings(
        CHAT_LLM_TEMPERATURE=0.3, CHAT_LLM_MAX_TOKENS=2048, CHAT_LLM_TIMEOUT_SECONDS=10
    )

    model_settings = build_model_settings("CHAT_LLM", settings)

    assert model_settings["temperature"] == 0.3
    assert model_settings["max_tokens"] == 2048
    assert model_settings["timeout"] == 10


def test_validate_llm_providers_or_raise_succeeds_for_valid_config() -> None:
    settings = _settings()

    validate_llm_providers_or_raise(settings)


def test_validate_llm_providers_or_raise_warns_on_identical_chat_and_judge(caplog) -> None:
    settings = _settings(
        EVAL_JUDGE_LLM_PROVIDER="anthropic",
        EVAL_JUDGE_LLM_MODEL="claude-sonnet-4-5-20250929",
        EVAL_JUDGE_LLM_API_KEY="sk-ant-test",
    )

    with caplog.at_level("WARNING"):
        validate_llm_providers_or_raise(settings)

    assert any("self-serving bias" in message for message in caplog.messages)


def test_validate_llm_providers_or_raise_propagates_config_error() -> None:
    settings = _settings(CHAT_LLM_API_KEY=None)

    with pytest.raises(LLMProviderConfigError):
        validate_llm_providers_or_raise(settings)
