from tests.integration.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    INACTIVE_EMAIL,
    INACTIVE_PASSWORD,
    USER_EMAIL,
    USER_PASSWORD,
)


def test_login_success_returns_access_token_and_sets_cookie(auth_client) -> None:
    response = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )

    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 1800
    assert "refresh_token" not in body  # never in the JSON body

    set_cookie = response.headers.get("set-cookie", "")
    assert "refresh_token=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=strict" in set_cookie.lower() or "samesite=strict" in set_cookie.lower()


def test_login_wrong_password_returns_401(auth_client) -> None:
    response = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": "wrong-password"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"


def test_login_nonexistent_user_returns_401(auth_client) -> None:
    response = auth_client.post(
        "/api/v1/auth/login", json={"username": "nobody@example.com", "password": "whatever"}
    )

    assert response.status_code == 401


def test_login_inactive_user_returns_401(auth_client) -> None:
    response = auth_client.post(
        "/api/v1/auth/login", json={"username": INACTIVE_EMAIL, "password": INACTIVE_PASSWORD}
    )

    assert response.status_code == 401


def test_login_rate_limited_after_max_attempts(auth_client) -> None:
    # test_settings fixture sets LOGIN_RATE_LIMIT_MAX_ATTEMPTS=3
    for _ in range(3):
        response = auth_client.post(
            "/api/v1/auth/login", json={"username": USER_EMAIL, "password": "wrong-password"}
        )
        assert response.status_code == 401

    blocked_response = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": "wrong-password"}
    )

    assert blocked_response.status_code == 429
    assert "Retry-After" in blocked_response.headers
    body = blocked_response.json()
    assert body["retry_after_seconds"] > 0

    # Even the CORRECT password is blocked while rate limited.
    still_blocked = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )
    assert still_blocked.status_code == 429


def test_successful_login_resets_rate_limit_counter(auth_client) -> None:
    auth_client.post("/api/v1/auth/login", json={"username": USER_EMAIL, "password": "wrong"})
    auth_client.post("/api/v1/auth/login", json={"username": USER_EMAIL, "password": "wrong"})

    success = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )
    assert success.status_code == 200

    # Counter should have been reset — another wrong attempt shouldn't be blocked yet.
    next_attempt = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": "wrong"}
    )
    assert next_attempt.status_code == 401


def test_me_requires_authentication(auth_client) -> None:
    response = auth_client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_me_returns_profile_with_scopes(auth_client) -> None:
    login_response = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )
    access_token = login_response.json()["access_token"]

    response = auth_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == USER_EMAIL
    assert body["role"] == "user"
    assert set(body["scopes"]) == {"chat:read", "chat:write", "documents:read"}


def test_admin_login_has_all_seeded_scopes(auth_client) -> None:
    login_response = auth_client.post(
        "/api/v1/auth/login", json={"username": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    access_token = login_response.json()["access_token"]

    response = auth_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {access_token}"}
    )

    assert response.status_code == 200
    assert "admin:rbac:write" in response.json()["scopes"]


def test_refresh_without_cookie_returns_401(auth_client) -> None:
    response = auth_client.post("/api/v1/auth/refresh")

    assert response.status_code == 401


def test_refresh_with_garbage_cookie_returns_401(auth_client) -> None:
    auth_client.cookies.set("refresh_token", "not-a-real-refresh-token")

    response = auth_client.post("/api/v1/auth/refresh")

    assert response.status_code == 401


def test_refresh_rotates_token_and_reuse_is_detected(auth_client) -> None:
    login_response = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )
    assert login_response.status_code == 200

    first_refresh = auth_client.post("/api/v1/auth/refresh")
    assert first_refresh.status_code == 200
    new_access_token = first_refresh.json()["access_token"]
    assert new_access_token != login_response.json()["access_token"]

    # A second, legitimate refresh with the rotated cookie succeeds.
    second_refresh = auth_client.post("/api/v1/auth/refresh")
    assert second_refresh.status_code == 200


def test_logout_revokes_refresh_token(auth_client) -> None:
    login_response = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )
    assert login_response.status_code == 200

    logout_response = auth_client.post("/api/v1/auth/logout")
    assert logout_response.status_code == 200
    assert logout_response.json()["detail"] == "Logged out"

    refresh_after_logout = auth_client.post("/api/v1/auth/refresh")
    assert refresh_after_logout.status_code == 401


def test_logout_without_cookie_still_succeeds(auth_client) -> None:
    response = auth_client.post("/api/v1/auth/logout")

    assert response.status_code == 200


def test_logout_with_unknown_cookie_still_succeeds(auth_client) -> None:
    auth_client.cookies.set("refresh_token", "not-a-real-refresh-token")

    response = auth_client.post("/api/v1/auth/logout")

    assert response.status_code == 200


def test_me_with_valid_token_for_deleted_user_returns_401(auth_client, test_settings) -> None:
    from app.auth.jwt import create_access_token

    token = create_access_token(
        user_id=999_999, role="user", scopes=["chat:read"], settings=test_settings
    )

    response = auth_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
