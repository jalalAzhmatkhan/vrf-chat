/**
 * Parser for the constrained markdown-lite subset backend guarantees for
 * `TechnicalAnswer.answer`, per
 * Documentation/system-design/05-streaming-and-api-contract.md §5.4:
 * paragraphs (`\n\n`), `1. `/`- `/`* ` lists, `**bold**`, and
 * `{{el:<element_id>}}` inline markers (§5.1). Deliberately NOT a general
 * markdown parser (`04-chat-ui.md` §4: "bukan full markdown parser") — the
 * contract is a closed set on both ends (backend never emits outside it, FE
 * never needs to handle more than this).
 *
 * Also applies the FE-side mono-identifier heuristic required by
 * `01-design-system.md` §3.1 / `04-chat-ui.md` §4 (error codes, connector
 * IDs, part numbers, value+unit — e.g. `P8`, `CN105`, `220V`).
 */

const MARKER_RE = /\{\{el:(\d+)\}\}/g;
const BOLD_RE = /\*\*(.+?)\*\*/g;
// Heuristic technical-identifier detector (MVP-level, per §4): short
// letter+digit codes (P8, U4, CN105, TH3) or a number immediately followed
// by a common engineering unit (220V, 4.7kΩ, 50Hz). Intentionally
// conservative to avoid flagging ordinary prose numbers/words.
const MONO_RE =
  /\b(?:[A-Za-z]{1,4}\d{1,4}[A-Za-z0-9]*|\d+(?:[.,]\d+)?\s?(?:V|A|W|Hz|kHz|MHz|k?Ω|kW|kPa|MPa|bar|mm|cm|°C|Nm|rpm)\b)/g;

/**
 * @param {string} text
 * @param {RegExp} regex Must be a global regex; `lastIndex` is reset here.
 * @param {(match: RegExpExecArray) => object} toNode
 * @returns {Array<{type: 'plain', value: string} | object>}
 */
function tokenize(text, regex, toNode) {
  const nodes = [];
  let lastIndex = 0;
  regex.lastIndex = 0;
  let match = regex.exec(text);
  while (match) {
    if (match.index > lastIndex) {
      nodes.push({ type: 'plain', value: text.slice(lastIndex, match.index) });
    }
    nodes.push(toNode(match));
    lastIndex = match.index + match[0].length;
    match = regex.exec(text);
  }
  if (lastIndex < text.length) nodes.push({ type: 'plain', value: text.slice(lastIndex) });
  return nodes;
}

/** Applies mono-identifier detection to a plain-text run. */
function applyMono(text) {
  return tokenize(text, MONO_RE, (m) => ({ type: 'mono', value: m[0] })).map((node) =>
    node.type === 'plain' ? { type: 'text', value: node.value } : node,
  );
}

/**
 * @param {string} text
 * @returns {Array<
 *   | { type: 'text', value: string }
 *   | { type: 'mono', value: string }
 *   | { type: 'bold', children: Array<{type:'text'|'mono', value: string}> }
 *   | { type: 'marker', elementId: string }
 * >}
 */
export function parseInline(text) {
  const afterMarkers = tokenize(text, MARKER_RE, (m) => ({ type: 'marker', elementId: m[1] }));

  const result = [];
  for (const node of afterMarkers) {
    if (node.type !== 'plain') {
      result.push(node);
      continue;
    }
    const afterBold = tokenize(node.value, BOLD_RE, (m) => ({ type: 'bold-raw', value: m[1] }));
    for (const boldNode of afterBold) {
      if (boldNode.type === 'bold-raw') {
        result.push({ type: 'bold', children: applyMono(boldNode.value) });
      } else {
        result.push(...applyMono(boldNode.value));
      }
    }
  }
  return result;
}

function classifyLine(line) {
  if (/^\d+\.\s+/.test(line)) return 'ol';
  if (/^[-*]\s+/.test(line)) return 'ul';
  return 'text';
}

function stripListMarker(line) {
  return line.replace(/^(?:\d+\.|[-*])\s+/, '');
}

/**
 * @param {string} text Full (or partially streamed, already
 *   `getSafeRenderLength`-trimmed) answer text.
 * @returns {Array<
 *   | { type: 'paragraph', inline: ReturnType<typeof parseInline> }
 *   | { type: 'ul'|'ol', items: Array<ReturnType<typeof parseInline>> }
 * >}
 */
export function parseAnswerBlocks(text) {
  const blocks = [];
  const paragraphChunks = text.split(/\n{2,}/);

  for (const chunk of paragraphChunks) {
    const lines = chunk
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length === 0) continue;

    let textBuffer = [];
    let listType = null;
    let listItems = [];

    const flushText = () => {
      if (textBuffer.length > 0) {
        blocks.push({ type: 'paragraph', inline: parseInline(textBuffer.join(' ')) });
        textBuffer = [];
      }
    };
    const flushList = () => {
      if (listItems.length > 0) {
        blocks.push({ type: listType, items: listItems });
        listItems = [];
        listType = null;
      }
    };

    for (const line of lines) {
      const kind = classifyLine(line);
      if (kind === 'text') {
        flushList();
        textBuffer.push(line);
      } else {
        flushText();
        if (listType && listType !== kind) flushList();
        listType = kind;
        listItems.push(parseInline(stripListMarker(line)));
      }
    }
    flushText();
    flushList();
  }

  return blocks;
}

/**
 * Extracts the set of `element_id`s referenced by `{{el:ID}}` markers
 * anywhere in `text` — used to distinguish inline-referenced citations from
 * ones that belong in the Visual Reference Card instead
 * (`04-chat-ui.md` §7).
 * @param {string} text
 * @returns {Set<string>}
 */
export function extractMarkerElementIds(text) {
  const ids = new Set();
  MARKER_RE.lastIndex = 0;
  let match = MARKER_RE.exec(text);
  while (match) {
    ids.add(match[1]);
    match = MARKER_RE.exec(text);
  }
  return ids;
}
