/**
 * Placeholder for C2.8 (`features/citation-viewer`, chained on top of this
 * branch — `Documentation/project-milestones/03-phase-2-chat-core.md`
 * C2.7 depends_on/precedes C2.8). `ChatPage` and its children already call
 * `open(elementId)` wherever `05-citation-viewer.md` says a click should
 * open the panel (inline chips §6.2, citation rows §10, Visual Reference
 * Card §7) — this stub keeps those call sites functional (no-op + a
 * console warning) until C2.8 replaces this file with the real
 * `CitationViewerProvider`/panel implementation, without requiring any
 * changes to the call sites themselves.
 */
export function useCitationViewer() {
  return {
    open: (elementId) => {
      // eslint-disable-next-line no-console
      console.warn(
        `[citation-viewer] open(${elementId}) called before C2.8 implements the panel — no-op.`,
      );
    },
  };
}
