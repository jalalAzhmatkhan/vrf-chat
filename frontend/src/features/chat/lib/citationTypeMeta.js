/**
 * Element/chunk type -> UI label mapping, Documentation/ui-ux-design/
 * 04-chat-ui.md §10.1. Glyphs are simple text/unicode markers (consistent
 * with the rest of the app's chrome — see shell/navConfig.js icons) rather
 * than an icon font/SVG set, kept intentionally lightweight for MVP.
 */
const TYPE_META = {
  procedure: { label: 'Prosedur', glyph: '☑' },
  list: { label: 'Prosedur', glyph: '☑' },
  table: { label: 'Tabel', glyph: '▦' },
  figure: { label: 'Gambar', glyph: '▧' },
  diagram: { label: 'Gambar', glyph: '▧' },
  figure_caption: { label: 'Gambar', glyph: '▧' },
  icon: { label: 'Ikon', glyph: '◉' },
  text: { label: 'Teks', glyph: '▤' },
  paragraph: { label: 'Teks', glyph: '▤' },
  entity: { label: 'Komponen', glyph: '#' },
  warning: { label: 'Catatan Manual', glyph: '⚠' },
  note: { label: 'Catatan Manual', glyph: 'ℹ' },
};

const DEFAULT_META = { label: 'Referensi', glyph: '▤' };

export function getCitationTypeMeta(elementType) {
  return TYPE_META[elementType] ?? DEFAULT_META;
}
