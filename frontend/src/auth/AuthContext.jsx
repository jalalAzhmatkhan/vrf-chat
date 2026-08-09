import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as authApi from './authApi';
import { AuthContext } from './AuthContextBase';
import { setAccessToken, clearAccessToken, setOnAuthFailure, setOnForbidden } from './tokenStore';

/**
 * Owns:
 *  - `user` ({id, username, role, scopes}) sourced from GET /auth/me.
 *  - `authMeStatus`: 'idle' | 'loading' | 'success' | 'error-network' | 'error-401'
 *    per Documentation/ui-ux-design/03-app-shell-navigation.md §4/§5.
 *  - login/logout actions.
 *
 * The access token itself lives in auth/tokenStore.js (plain module state,
 * outside React) so the non-React HTTP client can read it too — this
 * context only mirrors "do we currently believe we're authenticated" for
 * rendering purposes.
 */
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [authMeStatus, setAuthMeStatus] = useState('idle');
  // §3.1/F-8: 403 = "token valid, scope kurang" on ANY API call (not just
  // route access) — set true by the httpClient's notifyForbidden bridge,
  // surfaced by AppShell as the same in-shell "tidak punya akses" page used
  // by RouteGuard (03-app-shell-navigation.md §6).
  const [apiForbidden, setApiForbidden] = useState(false);
  const onSessionExpiredRef = useRef(null);

  useEffect(() => {
    // Registered once: fires when the HTTP client's refresh-on-401 flow
    // exhausts itself (refresh token invalid/expired/revoked) — per
    // 08-authentication-rbac.md DIRECT MESSAGE -> Frontend Engineer and
    // 03-app-shell-navigation.md §5 ("401 -> redirect langsung ke /login").
    setOnAuthFailure(() => {
      setUser(null);
      setAuthMeStatus('error-401');
      onSessionExpiredRef.current?.();
    });

    // Registered once: fires on any 403 from any API call. Deliberately
    // does NOT touch `user`/`authMeStatus`/the access token — the session
    // itself is still valid, just insufficiently scoped for this one
    // request (§3.1, revised 2026-08-09 per QA F-8).
    setOnForbidden(() => {
      setApiForbidden(true);
    });
  }, []);

  /** Registers the redirect-to-login side effect (wired from a router-aware component). */
  const registerOnSessionExpired = useCallback((callback) => {
    onSessionExpiredRef.current = callback;
  }, []);

  const clearApiForbidden = useCallback(() => setApiForbidden(false), []);

  const fetchMe = useCallback(async () => {
    setAuthMeStatus('loading');
    try {
      const profile = await authApi.me();
      setUser(profile);
      setAuthMeStatus('success');
      return profile;
    } catch (error) {
      if (error.status === 401) {
        // tokenStore's onAuthFailure already fired (redirect handled there).
        setUser(null);
        setAuthMeStatus('error-401');
      } else {
        // Network failure, 5xx, or (edge case) a 403 on /auth/me itself
        // (only possible for a user with literally zero scopes, since this
        // endpoint requires only the baseline scope every seeded role has)
        // — shown in-shell with retry, not a redirect
        // (03-app-shell-navigation.md §5). A 403 here also already
        // triggered the global apiForbidden signal above via httpClient.
        setAuthMeStatus('error-network');
      }
      throw error;
    }
  }, []);

  const loginWithCredentials = useCallback(async (username, password) => {
    const result = await authApi.login(username, password);
    setAccessToken(result.access_token);
    await fetchMe();
    return result;
  }, [fetchMe]);

  const logoutUser = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      clearAccessToken();
      setUser(null);
      setAuthMeStatus('idle');
    }
  }, []);

  const value = useMemo(
    () => ({
      user,
      scopes: user?.scopes ?? [],
      authMeStatus,
      apiForbidden,
      clearApiForbidden,
      fetchMe,
      loginWithCredentials,
      logoutUser,
      registerOnSessionExpired,
    }),
    [
      user,
      authMeStatus,
      apiForbidden,
      clearApiForbidden,
      fetchMe,
      loginWithCredentials,
      logoutUser,
      registerOnSessionExpired,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
