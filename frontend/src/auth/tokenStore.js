/**
 * In-memory access-token store, deliberately outside React state/context so
 * the plain-fetch HTTP client (lib/httpClient.js) can read/write it without
 * importing React or causing circular imports with AuthContext.
 *
 * Per Documentation/system-design/08-authentication-rbac.md DIRECT MESSAGE
 * -> Frontend Engineer and ui-ux-design/02-login-page.md §1: the access
 * token lives ONLY in memory (this module's closure) — never localStorage,
 * never sessionStorage. It is lost on tab close/hard refresh by design, and
 * re-obtained transparently via POST /auth/refresh (httpOnly cookie).
 *
 * AuthContext is the only consumer that should mutate this via setToken();
 * everything else should treat it as read-only.
 */

let accessToken = null;
let onAuthFailure = null;

export function getAccessToken() {
  return accessToken;
}

export function setAccessToken(token) {
  accessToken = token;
}

export function clearAccessToken() {
  accessToken = null;
}

/**
 * Registered once by AuthProvider. Called by the HTTP client when a 401
 * survives a refresh attempt (refresh token itself invalid/expired/revoked)
 * — AuthProvider reacts by clearing state and redirecting to /login, per
 * Documentation/ui-ux-design/03-app-shell-navigation.md §5.
 */
export function setOnAuthFailure(callback) {
  onAuthFailure = callback;
}

export function notifyAuthFailure() {
  clearAccessToken();
  if (onAuthFailure) onAuthFailure();
}
