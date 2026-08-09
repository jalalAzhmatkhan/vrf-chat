import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import * as authApi from './authApi';
import { setAccessToken, clearAccessToken, setOnAuthFailure } from './tokenStore';

const AuthContext = createContext(null);

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
  }, []);

  /** Registers the redirect-to-login side effect (wired from a router-aware component). */
  const registerOnSessionExpired = useCallback((callback) => {
    onSessionExpiredRef.current = callback;
  }, []);

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
        // Network failure or 5xx — shown in-shell with retry, not a redirect
        // (03-app-shell-navigation.md §5).
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
      fetchMe,
      loginWithCredentials,
      logoutUser,
      registerOnSessionExpired,
    }),
    [user, authMeStatus, fetchMe, loginWithCredentials, logoutUser, registerOnSessionExpired],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
}
