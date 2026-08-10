import { useMemo } from 'react';
import styles from './TableCard.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §7 "Table card" — renders a real
 * `<table>` from `Citation.content_structured.rows`
 * (`Documentation/system-design/05-streaming-and-api-contract.md` §5.2.1),
 * per `01-design-system.md` §7.3 (sticky header, mono/right-align for
 * identifier-or-number columns, zebra striping ≥6 rows, dense vertical
 * borders, fade-edge horizontal overflow on mobile, internal scroll +
 * `max-height: 320px` when >8 rows).
 *
 * Superseded the earlier FE deviation (fallback to an image crop or a raw
 * mono text block) now that the backend actually exposes structured rows —
 * see `VisualReferenceCard`'s docstring for the fallback chain that still
 * applies when `content_structured` is unavailable for a given table.
 */

// Cell-level heuristic for "identifier/angka teknis" columns — mirrors the
// inline mono-identifier detector in `lib/parseAnswer.js` (MONO_RE), but
// anchored to match a *whole* cell value rather than a substring, since
// table cells are short discrete values rather than prose.
const IDENTIFIER_CELL_RE =
  /^-?[A-Za-z]{1,4}\d{1,4}[A-Za-z0-9./-]*$|^-?\d+(?:[.,]\d+)?\s?(?:%|V|A|W|Hz|kHz|MHz|k?Ω|kW|kPa|MPa|bar|mm|cm|°C|Nm|rpm)?$/;

function cellText(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/** Derives ordered column keys: explicit `headers` if present, else the union of row keys in encounter order. */
function deriveColumns(contentStructured) {
  if (Array.isArray(contentStructured?.headers) && contentStructured.headers.length > 0) {
    return contentStructured.headers;
  }
  const columns = [];
  const seen = new Set();
  for (const row of contentStructured?.rows ?? []) {
    if (!row || typeof row !== 'object') continue;
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key);
        columns.push(key);
      }
    }
  }
  return columns;
}

/** A column is treated as an identifier/number column if most of its non-empty values look like one. */
function deriveIdentifierColumns(columns, rows) {
  const identifierCols = new Set();
  for (const col of columns) {
    let checked = 0;
    let matched = 0;
    for (const row of rows) {
      const text = cellText(row?.[col]).trim();
      if (!text) continue;
      checked += 1;
      if (IDENTIFIER_CELL_RE.test(text)) matched += 1;
    }
    if (checked > 0 && matched / checked >= 0.6) identifierCols.add(col);
  }
  return identifierCols;
}

/**
 * @param {{
 *   contentStructured: { rows?: Array<Record<string, unknown>>, headers?: string[], truncated?: boolean },
 * }} props
 */
export function TableCard({ contentStructured }) {
  const rows = useMemo(() => contentStructured?.rows ?? [], [contentStructured]);
  const columns = useMemo(() => deriveColumns(contentStructured), [contentStructured]);
  const identifierCols = useMemo(() => deriveIdentifierColumns(columns, rows), [columns, rows]);

  if (columns.length === 0 || rows.length === 0) return null;

  const scrollable = rows.length > 8;
  const zebra = rows.length >= 6;

  return (
    <div className={styles.wrapper}>
      <div className={`${styles.scrollInner} ${scrollable ? styles.scrollable : ''} scroll-region`}>
        <table className={`${styles.table} ${zebra ? styles.zebra : ''}`}>
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col}
                  className={`${styles.th} ${identifierCols.has(col) ? `${styles.identifierHeader} mono` : ''}`}
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              // eslint-disable-next-line react/no-array-index-key
              <tr key={rowIndex}>
                {columns.map((col) => {
                  const isIdentifier = identifierCols.has(col);
                  return (
                    <td key={col} className={`${styles.td} ${isIdentifier ? `${styles.identifierCell} mono` : ''}`}>
                      {cellText(row?.[col])}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
