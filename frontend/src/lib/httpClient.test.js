import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../auth/tokenStore', () => ({
  getAccessToken: vi.fn(() => 'initial-access-token'),
  setAccessToken: vi.fn(),
  notifyAuthFailure: vi.fn(),
  notifyForbidden: vi.fn(),
}));

// Imported after the mock so httpClient picks up the mocked module.
const tokenStore = await import('../auth/tokenStore');
const { apiFetch } = await import('./httpClient');

function jsonResponse(status, body, headers = {}) {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  });
}

describe('apiFetch — 401 vs 403 interceptor behavior (§3.1/F-8)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('401 on a protected endpoint: attempts /auth/refresh, retries once, succeeds', async () => {
    const fetchMock = vi
      .fn()
      // 1st call: the original request -> 401
      .mockResolvedValueOnce(jsonResponse(401, { detail: 'Missing or invalid token' }))
      // 2nd call: POST /auth/refresh -> 200 with new access token
      .mockResolvedValueOnce(jsonResponse(200, { access_token: 'new-token', token_type: 'bearer', expires_in: 1800 }))
      // 3rd call: retried original request -> 200
      .mockResolvedValueOnce(jsonResponse(200, { id: 1, username: 'jalal', role: 'admin', scopes: ['chat:read'] }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await apiFetch('/auth/me');

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[1][0]).toContain('/auth/refresh');
    expect(tokenStore.setAccessToken).toHaveBeenCalledWith('new-token');
    expect(tokenStore.notifyAuthFailure).not.toHaveBeenCalled();
    expect(tokenStore.notifyForbidden).not.toHaveBeenCalled();
    expect(result).toEqual({ id: 1, username: 'jalal', role: 'admin', scopes: ['chat:read'] });
  });

  it('401 whose refresh also fails: notifies auth failure, does NOT notify forbidden', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(401, { detail: 'Missing or invalid token' }))
      .mockResolvedValueOnce(jsonResponse(401, { detail: 'Refresh token invalid, expired, or revoked' }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiFetch('/chat/stream')).rejects.toMatchObject({ status: 401 });

    expect(tokenStore.notifyAuthFailure).toHaveBeenCalledTimes(1);
    expect(tokenStore.notifyForbidden).not.toHaveBeenCalled();
  });

  it('403 (valid token, insufficient scope): never calls /auth/refresh, notifies forbidden instead', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(403, { detail: 'Missing required scope(s): admin:rbac:read' }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiFetch('/admin/rbac/users')).rejects.toMatchObject({ status: 403 });

    // Only the one original call — no /auth/refresh call at all.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(tokenStore.notifyForbidden).toHaveBeenCalledTimes(1);
    expect(tokenStore.notifyForbidden).toHaveBeenCalledWith('Missing required scope(s): admin:rbac:read');
    expect(tokenStore.notifyAuthFailure).not.toHaveBeenCalled();
  });

  it('401 on /auth/login (bad credentials): does NOT attempt refresh, does NOT notify auth failure', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(jsonResponse(401, { detail: 'Invalid username or password' }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(apiFetch('/auth/login', { method: 'POST', json: { username: 'a', password: 'b' } })).rejects.toMatchObject(
      { status: 401, detail: 'Invalid username or password' },
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(tokenStore.notifyAuthFailure).not.toHaveBeenCalled();
    expect(tokenStore.notifyForbidden).not.toHaveBeenCalled();
  });
});
