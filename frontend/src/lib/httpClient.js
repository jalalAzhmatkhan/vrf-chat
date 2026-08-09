import { env } from './env';
import { ApiError } from './apiError';
import { getAccessToken, setAccessToken, notifyAuthFailure } from '../auth/tokenStore';

/**
 * fetch-based HTTP client with an interceptor pattern ready for auth:
 * - Attaches `Authorization: Bearer <access_token>` automatically.
 * - Always sends `credentials: 'include'` so the httpOnly refresh-token
 *   cookie travels with requests that need it (refresh/logout) — harmless
 *   no-op for other endpoints since the cookie is scoped to
 *   `Path=/api/v1/auth` by the backend (§8.2/§8.5).
 * - On a 401 from any non-auth endpoint, transparently calls
 *   `POST /auth/refresh` (deduped across concurrent 401s) and retries the
 *   original request once. If refresh itself fails, clears the in-memory
 *   token and notifies AuthProvider (-> redirect to /login), per
 *   Documentation/system-design/08-authentication-rbac.md DIRECT MESSAGE ->
 *   Frontend Engineer and ui-ux-design/03-app-shell-navigation.md §5.
 *
 * Kept as a standalone module (not a React hook) so it can be used from
 * anywhere, including outside component tree (route loaders, etc).
 */

const AUTH_ENDPOINTS_NO_REFRESH_RETRY = new Set(['/auth/login', '/auth/refresh']);

let refreshPromise = null;

async function parseJsonSafe(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function buildApiError(response, body) {
  const retryAfterHeader = response.headers.get('Retry-After');
  const retryAfterSeconds =
    body?.retry_after_seconds ?? (retryAfterHeader ? Number(retryAfterHeader) : null);

  return new ApiError({
    status: response.status,
    detail: body?.detail ?? response.statusText,
    retryAfterSeconds,
    body,
  });
}

async function performRefresh() {
  if (!refreshPromise) {
    refreshPromise = fetch(`${env.API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      credentials: 'include',
    })
      .then(async (response) => {
        if (!response.ok) {
          throw buildApiError(response, await parseJsonSafe(response));
        }
        const body = await parseJsonSafe(response);
        setAccessToken(body?.access_token ?? null);
        return body;
      })
      .finally(() => {
        refreshPromise = null;
      });
  }
  return refreshPromise;
}

/**
 * @param {string} path Path relative to env.API_BASE_URL, e.g. "/auth/me".
 * @param {RequestInit & { json?: unknown, skipAuthHeader?: boolean }} options
 */
export async function apiFetch(path, options = {}) {
  const { json, skipAuthHeader = false, headers: customHeaders, ...rest } = options;

  const doRequest = () => {
    const headers = new Headers(customHeaders);
    if (json !== undefined) headers.set('Content-Type', 'application/json');
    const token = getAccessToken();
    if (token && !skipAuthHeader) headers.set('Authorization', `Bearer ${token}`);

    return fetch(`${env.API_BASE_URL}${path}`, {
      credentials: 'include',
      headers,
      body: json !== undefined ? JSON.stringify(json) : rest.body,
      ...rest,
    });
  };

  let response;
  try {
    response = await doRequest();
  } catch {
    throw new ApiError({ status: 'network', detail: 'Network request failed' });
  }

  if (response.status === 401 && !AUTH_ENDPOINTS_NO_REFRESH_RETRY.has(path)) {
    try {
      await performRefresh();
    } catch {
      notifyAuthFailure();
      throw new ApiError({ status: 401, detail: 'Session expired' });
    }

    let retryResponse;
    try {
      retryResponse = await doRequest();
    } catch {
      throw new ApiError({ status: 'network', detail: 'Network request failed' });
    }

    if (!retryResponse.ok) {
      const body = await parseJsonSafe(retryResponse);
      if (retryResponse.status === 401) notifyAuthFailure();
      throw buildApiError(retryResponse, body);
    }
    return parseJsonSafe(retryResponse);
  }

  if (!response.ok) {
    const body = await parseJsonSafe(response);
    throw buildApiError(response, body);
  }

  return parseJsonSafe(response);
}
