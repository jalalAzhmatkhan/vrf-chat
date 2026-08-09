import { useEffect, useRef } from 'react';

const IDLE_DELAY_MS = 1200;

/**
 * Toggles the `.is-idle` class on a scrollable element after IDLE_DELAY_MS
 * of no scroll/pointer activity, per
 * Documentation/ui-ux-design/01-design-system.md §6 "Fade saat idle".
 *
 * Returns a ref to attach to the scrollable element (which should also have
 * the `.scroll-region` class from styles/scrollbar.css).
 */
export function useScrollbarIdle() {
  const ref = useRef(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return undefined;

    let timeoutId = null;

    const setIdle = () => node.classList.add('is-idle');
    const clearIdle = () => {
      node.classList.remove('is-idle');
      if (timeoutId) window.clearTimeout(timeoutId);
      timeoutId = window.setTimeout(setIdle, IDLE_DELAY_MS);
    };

    clearIdle();
    node.addEventListener('scroll', clearIdle, { passive: true });
    node.addEventListener('pointerenter', clearIdle);
    node.addEventListener('pointermove', clearIdle);

    return () => {
      node.removeEventListener('scroll', clearIdle);
      node.removeEventListener('pointerenter', clearIdle);
      node.removeEventListener('pointermove', clearIdle);
      if (timeoutId) window.clearTimeout(timeoutId);
    };
  }, []);

  return ref;
}
