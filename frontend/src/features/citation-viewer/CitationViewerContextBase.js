import { createContext } from 'react';

/**
 * Raw context object, split into its own module (no component/hook exports
 * here) so `CitationViewerProvider` (CitationViewerContext.jsx) and
 * `useCitationViewer` (useCitationViewer.js) can each live in files that
 * export exactly one thing — keeps Fast Refresh happy and resolves
 * oxlint's `react/only-export-components` warning (same pattern as
 * auth/AuthContextBase.js).
 */
export const CitationViewerContext = createContext(null);
