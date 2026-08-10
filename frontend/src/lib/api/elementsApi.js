import { apiFetch } from '../httpClient';

/**
 * `GET /api/v1/elements/{element_id}` — per
 * Documentation/system-design/05-streaming-and-api-contract.md §6.
 *
 * @typedef {{
 *   id: number, document_id: number, page_number: number,
 *   element_type: string, text: string|null,
 *   bbox: {l:number,t:number,r:number,b:number,page_number:number,coord_origin:string}|null,
 *   section_path: unknown[]|null, image_uri: string|null,
 *   visual_description: Record<string, unknown>|null,
 *   content_structured: {rows: Array<Record<string, unknown>>, headers?: string[], truncated?: boolean}|null,
 * }} ElementDetail
 */

/**
 * Module-level cache (dedup in-flight + completed fetches) — deliberately
 * NOT React state: consumed both by `InlineVisualChip` (chat, C2.7) needing
 * element metadata *before* the `citations[]` payload arrives — see
 * §5.1/§10 note on citation timing in
 * `Documentation/system-design/05-streaming-and-api-contract.md` — and by
 * `VisualReferenceCard`'s table fallback (C2.8 companion,
 * `04-chat-ui.md` §7). Sharing one cache means a marker chip that already
 * triggered a fetch and a Visual Reference Card for the same element don't
 * double-fetch.
 *
 * @type {Map<string, Promise<ElementDetail>>}
 */
const cache = new Map();

/**
 * @param {number|string} elementId
 * @returns {Promise<ElementDetail>}
 */
export function getElementCached(elementId) {
  const key = String(elementId);
  let pending = cache.get(key);
  if (!pending) {
    pending = apiFetch(`/elements/${key}`, { method: 'GET' }).catch((error) => {
      // Don't poison the cache with a rejected promise — a transient
      // network error shouldn't permanently block retries for this id.
      cache.delete(key);
      throw error;
    });
    cache.set(key, pending);
  }
  return pending;
}

/** Test-only escape hatch (module-level cache would otherwise leak across tests). */
export function __clearElementCacheForTests() {
  cache.clear();
}
