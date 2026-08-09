/**
 * Generic placeholder for routes not yet implemented in this phase —
 * Fase 0 (F0.1/F0.2) only needs the routing shell + scope-gated access to
 * exist; the real page content is built in later phases per
 * Documentation/project-milestones/.
 */
export function PlaceholderPage({ title, description }) {
  return (
    <div>
      <h1 style={{ fontSize: 'var(--font-size-xl)', marginBottom: 'var(--space-3)' }}>{title}</h1>
      <p style={{ color: 'var(--color-text-secondary)' }}>
        {description ?? 'Halaman ini akan diimplementasikan pada fase berikutnya.'}
      </p>
    </div>
  );
}
