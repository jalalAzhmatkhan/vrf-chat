import { useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../../auth/useAuth';
import { Button } from '../../components/Button/Button';
import { Input } from '../../components/Input/Input';
import { formatMinutesSeconds, useLoginRateLimitCountdown } from './useLoginRateLimitCountdown';
import styles from './LoginPage.module.css';

// bcrypt truncates input silently past 72 bytes (08-authentication-rbac.md
// §2) — this is a lightweight FE guard, not the source of truth for policy.
const PASSWORD_MAX_LENGTH = 72;
// §4.5 fallback: minimum client-side lockout when the backend 429 doesn't
// (yet) include retry_after_seconds — explicitly NOT synced to the real
// backend rate-limit window, documented as a deliberate stopgap.
const RATE_LIMIT_FALLBACK_SECONDS = 30;

export function LoginPage() {
  const { loginWithCredentials } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [status, setStatus] = useState('idle'); // idle | submitting | success
  const [errorKind, setErrorKind] = useState('none'); // none | 401 | 429 | network
  const [rateLimitSeconds, setRateLimitSeconds] = useState(null);
  const [cardErrorHighlight, setCardErrorHighlight] = useState(false);

  const passwordInputRef = useRef(null);

  const remainingSeconds = useLoginRateLimitCountdown(errorKind === '429' ? rateLimitSeconds : null, () => {
    setErrorKind('none');
    setRateLimitSeconds(null);
  });

  const bothFieldsFilled = username.trim().length > 0 && password.length > 0;
  const isRateLimited = errorKind === '429' && remainingSeconds !== null && remainingSeconds > 0;
  const canSubmit = bothFieldsFilled && status !== 'submitting' && !isRateLimited;

  const handleUsernameChange = (event) => {
    setUsername(event.target.value);
    if (errorKind === '401') setErrorKind('none');
  };

  const handlePasswordChange = (event) => {
    setPassword(event.target.value);
    if (errorKind === '401') setErrorKind('none');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;

    setStatus('submitting');
    try {
      await loginWithCredentials(username, password);
      setStatus('success');
      const redirectTo = searchParams.get('redirect') || '/chat';
      window.setTimeout(() => navigate(redirectTo, { replace: true }), 400);
    } catch (error) {
      setStatus('idle');
      if (error.status === 401) {
        setErrorKind('401');
        setPassword('');
        setCardErrorHighlight(true);
        window.setTimeout(() => setCardErrorHighlight(false), 400);
        window.setTimeout(() => passwordInputRef.current?.focus(), 0);
      } else if (error.status === 429) {
        setErrorKind('429');
        setRateLimitSeconds(error.retryAfterSeconds ?? RATE_LIMIT_FALLBACK_SECONDS);
      } else {
        setErrorKind('network');
      }
    }
  };

  const fieldsDisabled = status === 'submitting' || isRateLimited;

  return (
    <div className={`${styles.page} circuit-grid-bg`}>
      <div className={styles.glow} aria-hidden="true" />
      <div className={`${styles.card} ${cardErrorHighlight ? styles.errorHighlight : ''}`}>
        <div className={styles.wordmark}>
          <span className={styles.wordmarkIcon} aria-hidden="true">
            ◈
          </span>
          <span className={styles.wordmarkTitle}>VRF/VRV AI</span>
          <span className={styles.wordmarkSubtitle}>Technical Assistant</span>
        </div>

        <form className={styles.form} onSubmit={handleSubmit} noValidate>
          <div className={styles.field}>
            <Input
              label="Username"
              id="login-username"
              name="username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={handleUsernameChange}
              disabled={fieldsDisabled}
              autoFocus
            />
          </div>

          <div className={styles.field}>
            <Input
              ref={passwordInputRef}
              label="Password"
              id="login-password"
              name="password"
              type={showPassword ? 'text' : 'password'}
              autoComplete="current-password"
              maxLength={PASSWORD_MAX_LENGTH}
              value={password}
              onChange={handlePasswordChange}
              disabled={fieldsDisabled}
              trailing={
                <Button
                  variant="icon"
                  size="sm"
                  type="button"
                  aria-label={showPassword ? 'Sembunyikan password' : 'Tampilkan password'}
                  onClick={() => setShowPassword((v) => !v)}
                  disabled={fieldsDisabled}
                >
                  {showPassword ? '🙈' : '👁'}
                </Button>
              }
            />
          </div>

          <div className={styles.submitRow}>
            <Button
              type="submit"
              variant="primary"
              size="lg"
              className={styles.submitButton}
              disabled={!canSubmit}
              loading={status === 'submitting'}
            >
              {status === 'success' ? (
                <span className={styles.checkmark} aria-hidden="true">
                  ✓
                </span>
              ) : (
                'Masuk'
              )}
            </Button>
          </div>

          {errorKind === '401' && (
            <div className={`${styles.alert} ${styles.alertDanger}`} role="alert" aria-live="assertive">
              <span className={styles.alertIcon} aria-hidden="true">
                ⚠
              </span>
              <span>Username atau password salah.</span>
            </div>
          )}

          {errorKind === '429' && (
            <div className={`${styles.alert} ${styles.alertWarning}`} role="alert" aria-live="polite">
              <span className={styles.alertIcon} aria-hidden="true">
                🕐
              </span>
              <span>
                Terlalu banyak percobaan login. Coba lagi dalam{' '}
                <span className="mono">{formatMinutesSeconds(remainingSeconds)}</span>.
              </span>
            </div>
          )}

          {errorKind === 'network' && (
            <div className={`${styles.alert} ${styles.alertDanger}`} role="alert" aria-live="polite">
              <span className={styles.alertIcon} aria-hidden="true">
                ⚠
              </span>
              <span>Tidak dapat terhubung ke server. Periksa koneksi Anda dan coba lagi.</span>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
