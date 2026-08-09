/**
 * Centralized env config loader/validator.
 *
 * Vite only exposes `VITE_`-prefixed vars via `import.meta.env`, and inlines
 * them at build time (they are NOT read at container runtime) — see the
 * DIRECT MESSAGE to Backend Engineer about Dockerfile build args.
 *
 * Fails fast (throws at module load) if a required var is missing, mirroring
 * the fail-fast validation convention used in the backend provider
 * abstractions (Documentation/system-design/04-provider-abstractions.md).
 */

function readEnv(key, { required = false, fallback = undefined } = {}) {
  const value = import.meta.env[key];
  if (value === undefined || value === '') {
    if (required) {
      throw new Error(
        `[env] Missing required environment variable "${key}". ` +
          'Copy .env.example to .env and set it, or provide it as a Vite build arg.',
      );
    }
    return fallback;
  }
  return value;
}

export const env = {
  API_BASE_URL: readEnv('VITE_API_BASE_URL', {
    required: false,
    fallback: 'http://localhost:8000/api/v1',
  }),
  IS_DEV: import.meta.env.DEV,
  IS_PROD: import.meta.env.PROD,
};

if (env.IS_DEV && !import.meta.env.VITE_API_BASE_URL) {
  // eslint-disable-next-line no-console
  console.warn(
    `[env] VITE_API_BASE_URL not set — falling back to ${env.API_BASE_URL}. ` +
      'Set it in .env for a non-default backend location.',
  );
}
