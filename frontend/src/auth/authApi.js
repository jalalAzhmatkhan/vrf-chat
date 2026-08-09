import { apiFetch } from '../lib/httpClient';

/**
 * Thin wrappers around the /auth/* contract —
 * Documentation/system-design/08-authentication-rbac.md §8.5.
 */

/** @returns {Promise<{access_token: string, token_type: string, expires_in: number}>} */
export function login(username, password) {
  return apiFetch('/auth/login', {
    method: 'POST',
    json: { username, password },
  });
}

/** @returns {Promise<{access_token: string, token_type: string, expires_in: number}>} */
export function refresh() {
  return apiFetch('/auth/refresh', { method: 'POST' });
}

/** @returns {Promise<{detail: string}>} */
export function logout() {
  return apiFetch('/auth/logout', { method: 'POST' });
}

/** @returns {Promise<{id: string|number, username: string, role: string, scopes: string[]}>} */
export function me() {
  return apiFetch('/auth/me', { method: 'GET' });
}
