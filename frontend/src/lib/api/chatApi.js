import { env } from '../env';
import { apiFetch } from '../httpClient';
import { ApiError } from '../apiError';
import { getAccessToken, notifyForbidden } from '../../auth/tokenStore';

/**
 * `POST /api/v1/chat/stream` (SSE) — per
 * Documentation/system-design/05-streaming-and-api-contract.md §4.
 *
 * Deliberately a raw `fetch` (not `lib/httpClient.js`'s `apiFetch`) because
 * `apiFetch` always parses the response as JSON — SSE needs the raw
 * `ReadableStream` body instead. To stay consistent with the rest of the
 * app's auth/session handling without duplicating the refresh-token dance,
 * a 401 on the *initial* response (before any bytes of the stream itself
 * have been read — chat.py's scope check runs before `StreamingResponse`
 * starts) is handled by delegating one refresh-and-retry attempt to
 * `apiFetch('/auth/me')`, which already implements exactly that flow
 * (lib/httpClient.js) including the terminal "refresh also failed ->
 * notifyAuthFailure -> redirect to /login" behavior. A 403 (valid token,
 * insufficient `chat:write` scope) is never retried, mirroring
 * `apiFetch`'s own 401-vs-403 split (§3.1/F-8).
 */

/**
 * @typedef {{ event: 'status'|'token'|'citation'|'done'|'error', data: any }} ChatStreamEvent
 */

function parseSseBlock(rawBlock) {
  let eventName = 'message';
  const dataLines = [];
  for (const line of rawBlock.split('\n')) {
    if (line.startsWith('event:')) eventName = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  const raw = dataLines.join('\n');
  try {
    return { event: eventName, data: JSON.parse(raw) };
  } catch {
    return { event: eventName, data: raw };
  }
}

/**
 * Parses a `ReadableStream<Uint8Array>` of `text/event-stream` bytes into
 * `ChatStreamEvent`s, handling chunk boundaries that split an event (or a
 * multi-byte UTF-8 character) across two `read()` calls.
 * @param {ReadableStream<Uint8Array>} body
 * @returns {AsyncGenerator<ChatStreamEvent>}
 */
async function* parseSseStream(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let separatorIndex = buffer.indexOf('\n\n');
      while (separatorIndex !== -1) {
        const rawBlock = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const parsed = parseSseBlock(rawBlock);
        if (parsed) yield parsed;
        separatorIndex = buffer.indexOf('\n\n');
      }
    }
    // Flush a final event that wasn't terminated by a trailing blank line.
    const trailing = parseSseBlock(buffer);
    if (trailing) yield trailing;
  } finally {
    reader.releaseLock();
  }
}

async function doStreamRequest(body, signal) {
  const headers = new Headers({ 'Content-Type': 'application/json' });
  const token = getAccessToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return fetch(`${env.API_BASE_URL}/chat/stream`, {
    method: 'POST',
    credentials: 'include',
    headers,
    body: JSON.stringify(body),
    signal,
  });
}

async function parseErrorBody(response) {
  try {
    const body = await response.json();
    return body?.detail;
  } catch {
    return undefined;
  }
}

/**
 * @param {{ conversation_id?: string|null, message: string, model_override?: string }} body
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {AsyncGenerator<ChatStreamEvent>}
 */
export async function* streamChat(body, { signal } = {}) {
  let response = await doStreamRequest(body, signal);

  if (response.status === 401) {
    // Reuse the existing refresh-and-retry flow via any authenticated JSON
    // endpoint (side effect: updates the in-memory access token, or
    // triggers notifyAuthFailure()/redirect-to-login if refresh itself
    // fails) — see module docstring.
    try {
      await apiFetch('/auth/me');
    } catch {
      /* apiFetch already notified auth failure; fall through to throw below */
    }
    response = await doStreamRequest(body, signal);
  }

  if (!response.ok) {
    const detail = await parseErrorBody(response);
    if (response.status === 403) notifyForbidden(detail);
    throw new ApiError({ status: response.status, detail: detail ?? response.statusText });
  }

  if (!response.body) {
    throw new ApiError({ status: 'network', detail: 'Empty stream response body' });
  }

  yield* parseSseStream(response.body);
}
