from tests.integration.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    ALL_SCOPES,
    USER_EMAIL,
    USER_PASSWORD,
    USER_SCOPES,
)


def _login(auth_client, username: str, password: str) -> str:
    response = auth_client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return str(response.json()["access_token"])


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_list_roles_requires_admin_rbac_read_scope(auth_client) -> None:
    unauthorized = auth_client.get("/api/v1/admin/rbac/roles")
    assert unauthorized.status_code == 401  # no token at all -> stage-1 AUTHENTICATION

    user_token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    forbidden = auth_client.get("/api/v1/admin/rbac/roles", headers=_auth_headers(user_token))
    # Valid token, missing scope -> stage-2 AUTHORIZATION -> 403 (F-8,
    # Documentation/qa-reports/phase-0-qa-report.md).
    assert forbidden.status_code == 403


def test_list_roles_as_admin(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.get("/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token))

    assert response.status_code == 200
    names = {role["name"] for role in response.json()}
    assert names == {"admin", "user"}


def test_list_scopes_as_admin(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.get("/api/v1/admin/rbac/scopes", headers=_auth_headers(admin_token))

    assert response.status_code == 200
    codes = {scope["code"] for scope in response.json()}
    assert codes == set(ALL_SCOPES)


def test_get_role_scopes(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    roles = auth_client.get(
        "/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token)
    ).json()
    user_role_id = next(r["id"] for r in roles if r["name"] == "user")

    response = auth_client.get(
        f"/api/v1/admin/rbac/roles/{user_role_id}/scopes", headers=_auth_headers(admin_token)
    )

    assert response.status_code == 200
    assert set(response.json()["scope_codes"]) == set(USER_SCOPES)


def test_get_role_scopes_404_for_unknown_role(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.get(
        "/api/v1/admin/rbac/roles/999999/scopes", headers=_auth_headers(admin_token)
    )

    assert response.status_code == 404


def test_update_role_scopes_replaces_full_set(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    roles = auth_client.get(
        "/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token)
    ).json()
    user_role_id = next(r["id"] for r in roles if r["name"] == "user")

    response = auth_client.put(
        f"/api/v1/admin/rbac/roles/{user_role_id}/scopes",
        headers=_auth_headers(admin_token),
        json={"scope_codes": ["chat:read"]},
    )

    assert response.status_code == 200
    assert response.json()["scope_codes"] == ["chat:read"]

    # Confirm persisted, not just returned.
    verify = auth_client.get(
        f"/api/v1/admin/rbac/roles/{user_role_id}/scopes", headers=_auth_headers(admin_token)
    )
    assert verify.json()["scope_codes"] == ["chat:read"]


def test_update_role_scopes_requires_write_scope(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    roles = auth_client.get(
        "/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token)
    ).json()
    user_role_id = next(r["id"] for r in roles if r["name"] == "user")

    user_token = _login(auth_client, USER_EMAIL, USER_PASSWORD)
    response = auth_client.put(
        f"/api/v1/admin/rbac/roles/{user_role_id}/scopes",
        headers=_auth_headers(user_token),
        json={"scope_codes": ["chat:read"]},
    )

    assert response.status_code == 403


def test_update_role_scopes_unknown_code_returns_400(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    roles = auth_client.get(
        "/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token)
    ).json()
    user_role_id = next(r["id"] for r in roles if r["name"] == "user")

    response = auth_client.put(
        f"/api/v1/admin/rbac/roles/{user_role_id}/scopes",
        headers=_auth_headers(admin_token),
        json={"scope_codes": ["not:a:real:scope"]},
    )

    assert response.status_code == 400


def test_update_role_scopes_404_for_unknown_role(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.put(
        "/api/v1/admin/rbac/roles/999999/scopes",
        headers=_auth_headers(admin_token),
        json={"scope_codes": []},
    )

    assert response.status_code == 404


def test_list_users_as_admin(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.get("/api/v1/admin/rbac/users", headers=_auth_headers(admin_token))

    assert response.status_code == 200
    usernames = {u["username"] for u in response.json()}
    assert ADMIN_EMAIL in usernames
    assert USER_EMAIL in usernames


def test_list_users_filters_by_search_and_is_active(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.get(
        "/api/v1/admin/rbac/users",
        headers=_auth_headers(admin_token),
        params={"search": "user@", "is_active": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["username"] == USER_EMAIL


def test_list_users_filters_by_role_id(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    roles = auth_client.get(
        "/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token)
    ).json()
    admin_role_id = next(r["id"] for r in roles if r["name"] == "admin")

    response = auth_client.get(
        "/api/v1/admin/rbac/users",
        headers=_auth_headers(admin_token),
        params={"role_id": admin_role_id},
    )

    assert response.status_code == 200
    assert all(u["role_id"] == admin_role_id for u in response.json())


def test_get_user_detail(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    users = auth_client.get(
        "/api/v1/admin/rbac/users", headers=_auth_headers(admin_token)
    ).json()
    target_id = next(u["id"] for u in users if u["username"] == USER_EMAIL)

    response = auth_client.get(
        f"/api/v1/admin/rbac/users/{target_id}", headers=_auth_headers(admin_token)
    )

    assert response.status_code == 200
    assert response.json()["username"] == USER_EMAIL


def test_get_user_detail_404(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.get(
        "/api/v1/admin/rbac/users/999999", headers=_auth_headers(admin_token)
    )

    assert response.status_code == 404


def test_create_user(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    roles = auth_client.get(
        "/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token)
    ).json()
    user_role_id = next(r["id"] for r in roles if r["name"] == "user")

    response = auth_client.post(
        "/api/v1/admin/rbac/users",
        headers=_auth_headers(admin_token),
        json={
            "username": "new-teknisi@example.com",
            "initial_password": "NewTeknisiPass123!",
            "role_id": user_role_id,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "new-teknisi@example.com"
    assert body["role_id"] == user_role_id
    assert body["is_active"] is True

    # New user can actually log in with the given password.
    login_response = auth_client.post(
        "/api/v1/auth/login",
        json={"username": "new-teknisi@example.com", "password": "NewTeknisiPass123!"},
    )
    assert login_response.status_code == 200


def test_create_user_duplicate_username_returns_409(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    roles = auth_client.get(
        "/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token)
    ).json()
    user_role_id = next(r["id"] for r in roles if r["name"] == "user")

    response = auth_client.post(
        "/api/v1/admin/rbac/users",
        headers=_auth_headers(admin_token),
        json={"username": USER_EMAIL, "initial_password": "Whatever123!", "role_id": user_role_id},
    )

    assert response.status_code == 409


def test_create_user_unknown_role_returns_400(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.post(
        "/api/v1/admin/rbac/users",
        headers=_auth_headers(admin_token),
        json={
            "username": "someone@example.com",
            "initial_password": "Whatever123!",
            "role_id": 999999,
        },
    )

    assert response.status_code == 400


def test_patch_user_role(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    roles = auth_client.get(
        "/api/v1/admin/rbac/roles", headers=_auth_headers(admin_token)
    ).json()
    admin_role_id = next(r["id"] for r in roles if r["name"] == "admin")
    users = auth_client.get(
        "/api/v1/admin/rbac/users", headers=_auth_headers(admin_token)
    ).json()
    target_id = next(u["id"] for u in users if u["username"] == USER_EMAIL)

    response = auth_client.patch(
        f"/api/v1/admin/rbac/users/{target_id}",
        headers=_auth_headers(admin_token),
        json={"role_id": admin_role_id},
    )

    assert response.status_code == 200
    assert response.json()["role_id"] == admin_role_id


def test_patch_user_unknown_role_returns_400(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)
    users = auth_client.get(
        "/api/v1/admin/rbac/users", headers=_auth_headers(admin_token)
    ).json()
    target_id = next(u["id"] for u in users if u["username"] == USER_EMAIL)

    response = auth_client.patch(
        f"/api/v1/admin/rbac/users/{target_id}",
        headers=_auth_headers(admin_token),
        json={"role_id": 999999},
    )

    assert response.status_code == 400


def test_patch_user_404(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    response = auth_client.patch(
        "/api/v1/admin/rbac/users/999999",
        headers=_auth_headers(admin_token),
        json={"is_active": False},
    )

    assert response.status_code == 404


def test_patch_user_deactivate_revokes_refresh_tokens(auth_client) -> None:
    admin_token = _login(auth_client, ADMIN_EMAIL, ADMIN_PASSWORD)

    # user logs in (issuing a refresh token cookie on the shared client)
    user_login = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )
    assert user_login.status_code == 200

    users = auth_client.get(
        "/api/v1/admin/rbac/users", headers=_auth_headers(admin_token)
    ).json()
    target_id = next(u["id"] for u in users if u["username"] == USER_EMAIL)

    deactivate = auth_client.patch(
        f"/api/v1/admin/rbac/users/{target_id}",
        headers=_auth_headers(admin_token),
        json={"is_active": False},
    )
    assert deactivate.status_code == 200
    assert deactivate.json()["is_active"] is False

    # The refresh token cookie from the user's earlier login is now revoked.
    refresh_attempt = auth_client.post("/api/v1/auth/refresh")
    assert refresh_attempt.status_code == 401

    # And the (now-inactive) user can no longer log in at all.
    relogin = auth_client.post(
        "/api/v1/auth/login", json={"username": USER_EMAIL, "password": USER_PASSWORD}
    )
    assert relogin.status_code == 401
