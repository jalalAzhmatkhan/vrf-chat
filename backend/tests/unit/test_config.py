from app.core.config import Settings, get_settings


def test_default_settings_have_expected_app_name() -> None:
    settings = Settings(_env_file=None)

    assert settings.APP_NAME == "vrf-chat-backend"
    assert settings.APP_ENV == "development"
    assert settings.API_V1_PREFIX == "/api/v1"


def test_allowed_origins_accepts_native_list() -> None:
    settings = Settings(_env_file=None, ALLOWED_ORIGINS=["http://a", "http://b"])

    assert settings.ALLOWED_ORIGINS == ["http://a", "http://b"]


def test_allowed_origins_splits_comma_separated_string() -> None:
    settings = Settings(_env_file=None, ALLOWED_ORIGINS="http://a, http://b")

    assert settings.ALLOWED_ORIGINS == ["http://a", "http://b"]


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
    get_settings.cache_clear()
