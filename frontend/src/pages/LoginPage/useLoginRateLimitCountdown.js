import { useEffect, useRef, useState } from 'react';

/**
 * Live-updating countdown for the 429 rate-limit alert —
 * Documentation/ui-ux-design/02-login-page.md §4.5.
 *
 * @param {number|null} initialSeconds seconds remaining when the 429 was
 *   received (from `retry_after_seconds` / `Retry-After`, or the 30s
 *   client-side fallback per §4.5 when the backend doesn't provide one).
 * @param {() => void} onExpire called once when the countdown reaches 0.
 */
export function useLoginRateLimitCountdown(initialSeconds, onExpire) {
  const [remaining, setRemaining] = useState(initialSeconds);
  const onExpireRef = useRef(onExpire);
  onExpireRef.current = onExpire;

  useEffect(() => {
    setRemaining(initialSeconds);
    if (initialSeconds === null || initialSeconds <= 0) return undefined;

    const intervalId = window.setInterval(() => {
      setRemaining((prev) => {
        if (prev === null) return prev;
        const next = prev - 1;
        if (next <= 0) {
          window.clearInterval(intervalId);
          onExpireRef.current?.();
          return 0;
        }
        return next;
      });
    }, 1000);

    return () => window.clearInterval(intervalId);
  }, [initialSeconds]);

  return remaining;
}

export function formatMinutesSeconds(totalSeconds) {
  if (totalSeconds === null || totalSeconds === undefined) return '';
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}
