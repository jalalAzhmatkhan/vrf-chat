import { useContext } from 'react';
import { CitationViewerContext } from './CitationViewerContextBase';

/**
 * C2.8. `ChatPage`/`AssistantMessageCard` (C2.7) already import
 * `useCitationViewer` from this exact path, so no call sites needed to
 * change once the real `CitationViewerProvider` was wired in.
 */
export function useCitationViewer() {
  const ctx = useContext(CitationViewerContext);
  if (!ctx) throw new Error('useCitationViewer must be used within a CitationViewerProvider');
  return ctx;
}
