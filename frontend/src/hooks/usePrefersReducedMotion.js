import { useSyncExternalStore } from 'react';

const QUERY = '(prefers-reduced-motion: reduce)';

function subscribe(callback) {
  const mediaQueryList = window.matchMedia(QUERY);
  mediaQueryList.addEventListener('change', callback);
  return () => mediaQueryList.removeEventListener('change', callback);
}

function getSnapshot() {
  return window.matchMedia(QUERY).matches;
}

function getServerSnapshot() {
  return false;
}

/**
 * Single global flag for `prefers-reduced-motion: reduce`, per
 * Documentation/ui-ux-design/01-design-system.md §5.4 — implemented once
 * here and consumed by any component that animates, rather than re-derived
 * ad-hoc per component.
 */
export function usePrefersReducedMotion() {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
