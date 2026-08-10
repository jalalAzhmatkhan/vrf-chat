import { useCallback, useRef, useState } from 'react';
import { CitationViewerPanel } from './CitationViewerPanel';
import { CitationViewerContext } from './CitationViewerContextBase';

function buildEffectiveCitations(elementId, citations) {
  const key = String(elementId);
  const hasMatch = citations.some((c) => String(c.element_id) === key);
  if (hasMatch) return citations;
  // Marker/citation-row click for an element the message's `citations[]`
  // doesn't (yet) know about — CitationViewerPanel resolves document/page
  // itself via GET /elements/{id} in this case (see its docstring).
  return [
    ...citations,
    {
      element_id: key,
      document_id: null,
      page: null,
      element_type: null,
      quote: null,
      image_uri: null,
      visual_description: null,
    },
  ];
}

/**
 * Documentation/ui-ux-design/05-citation-viewer.md — mounted once per
 * `ChatPage` (overlay, not a route, §2). Exposes only `open(elementId,
 * {citations, documentsMap})` to consumers (`InlineVisualChip`,
 * `CitationList`, `VisualReferenceCard`) — the panel itself is rendered
 * internally so call sites never need to know about its existence beyond
 * this one function.
 */
export function CitationViewerProvider({ children }) {
  const [state, setState] = useState({
    isOpen: false,
    citations: [],
    activeIndex: 0,
    documentsMap: new Map(),
  });
  const triggerElementRef = useRef(null);

  const open = useCallback((elementId, { citations = [], documentsMap = new Map() } = {}) => {
    triggerElementRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const effective = buildEffectiveCitations(elementId, citations);
    const activeIndex = effective.findIndex((c) => String(c.element_id) === String(elementId));
    setState({ isOpen: true, citations: effective, activeIndex: Math.max(0, activeIndex), documentsMap });
  }, []);

  const close = useCallback(() => {
    setState((s) => ({ ...s, isOpen: false }));
    // §10: focus returns to whatever triggered the panel (chip/citation row).
    window.requestAnimationFrame(() => triggerElementRef.current?.focus?.());
  }, []);

  const navigate = useCallback((index) => {
    setState((s) => (index >= 0 && index < s.citations.length ? { ...s, activeIndex: index } : s));
  }, []);

  return (
    <CitationViewerContext.Provider value={{ open }}>
      {children}
      <CitationViewerPanel
        isOpen={state.isOpen}
        citations={state.citations}
        activeIndex={state.activeIndex}
        documentsMap={state.documentsMap}
        onClose={close}
        onNavigate={navigate}
      />
    </CitationViewerContext.Provider>
  );
}
