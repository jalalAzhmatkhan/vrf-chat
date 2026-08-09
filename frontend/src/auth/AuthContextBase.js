import { createContext } from 'react';

/**
 * Raw context object, split into its own module (no component/hook exports
 * here) so `AuthProvider` (AuthContext.jsx) and `useAuth` (useAuth.js) can
 * each live in files that export exactly one thing — keeps Fast Refresh
 * happy and resolves oxlint's `react/only-export-components` warning.
 */
export const AuthContext = createContext(null);
