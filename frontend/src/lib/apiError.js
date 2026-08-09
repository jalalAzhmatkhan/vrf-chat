/**
 * Structured error thrown by lib/httpClient.js for any non-2xx response (or
 * network failure), so callers (e.g. LoginForm) can branch on `status` /
 * `detail` / `retryAfterSeconds` without re-parsing the response.
 */
export class ApiError extends Error {
  /**
   * @param {{ status: number | 'network', detail?: string, retryAfterSeconds?: number | null, body?: unknown }} params
   */
  constructor({ status, detail, retryAfterSeconds = null, body = null }) {
    super(detail ?? `Request failed with status ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.retryAfterSeconds = retryAfterSeconds;
    this.body = body;
  }

  get isNetworkError() {
    return this.status === 'network';
  }
}
