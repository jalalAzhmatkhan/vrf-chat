/**
 * Streaming-safe boundary detection for the `{{el:<element_id>}}` inline
 * reference marker, per
 * Documentation/ui-ux-design/04-chat-ui.md §6.3 and
 * Documentation/system-design/05-streaming-and-api-contract.md §5.1
 * ("Streaming/buffering": marker bisa terpotong di batas event `token`).
 *
 * The raw accumulated answer text is always kept in full (nothing is
 * discarded) — `getSafeRenderLength` just tells the renderer how much of
 * that raw text is *safe to parse and display right now* without risking a
 * user glimpsing a raw `{{el:e` fragment mid-render. The withheld suffix is
 * re-evaluated on every new delta and is guaranteed to be flushed once the
 * stream reaches `done` (nothing can stay hidden forever).
 */

const MARKER_LITERAL = '{{el:';
// Generous bound: "{{el:" (5) + a very long element id + "}}" (2). Real
// element ids are small integers, but this is cheap to check either way.
const MAX_LOOKBACK = 32;

/**
 * True if `suffix` (assumed to end at the current end of the buffer) could
 * still turn into a complete `{{el:<digits>}}` marker with more incoming
 * text — i.e. it is a strict, still-open prefix of that pattern.
 * @param {string} suffix
 */
function isPotentialMarkerPrefix(suffix) {
  if (suffix.length === 0) return false;
  // Still typing out the literal "{{el:" itself (e.g. "{", "{{", "{{el").
  if (MARKER_LITERAL.startsWith(suffix)) return true;
  if (!suffix.startsWith(MARKER_LITERAL)) return false;
  // Past the literal: digits, then optionally one (but not two — a
  // complete "}}" would already have been extracted as a real match by the
  // caller before this function is ever consulted) closing brace.
  const rest = suffix.slice(MARKER_LITERAL.length);
  return /^\d+\}?$/.test(rest);
}

/**
 * Returns the index into `text` up to which it is safe to render — i.e. the
 * start of any still-ambiguous trailing marker attempt, or `text.length` if
 * there is none.
 * @param {string} text
 * @returns {number}
 */
export function getSafeRenderLength(text) {
  const searchStart = Math.max(0, text.length - MAX_LOOKBACK);
  for (let i = searchStart; i < text.length; i++) {
    if (text[i] !== '{') continue;
    if (isPotentialMarkerPrefix(text.slice(i))) return i;
  }
  return text.length;
}
