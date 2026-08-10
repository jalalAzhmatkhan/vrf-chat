import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../auth/tokenStore', () => ({
  getAccessToken: vi.fn(() => 'token-abc'),
  notifyForbidden: vi.fn(),
}));
vi.mock('../httpClient', () => ({ apiFetch: vi.fn() }));

const tokenStore = await import('../../auth/tokenStore');
const httpClient = await import('../httpClient');
const { streamChat } = await import('./chatApi');

function sseBody(chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(iterable) {
  const events = [];
  for await (const evt of iterable) events.push(evt);
  return events;
}

describe('streamChat (SSE)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('parses status/token/citation/done events, including one split across two chunks', async () => {
    const body = sseBody([
      'event: status\ndata: {"stage":"searching_manual"}\n\n',
      'event: token\ndata: {"delta":"Hal',
      'o dunia"}\n\n',
      'event: citation\ndata: {"element_id":"5","document_id":"1","page":2,"element_type":"icon"}\n\n',
      'event: done\ndata: {"answer":"Halo dunia","confidence":0.9,"citations":[],"warnings":[],"related_components":[],"related_error_codes":[],"refused":false}\n\n',
    ]);
    const fetchMock = vi.fn().mockResolvedValue(new Response(body, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const events = await collect(streamChat({ message: 'hi' }));

    expect(events.map((e) => e.event)).toEqual(['status', 'token', 'citation', 'done']);
    expect(events[1].data).toEqual({ delta: 'Halo dunia' });
    expect(events[3].data.answer).toBe('Halo dunia');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers.get('Authorization')).toBe('Bearer token-abc');
  });

  it('on 401, retries once via apiFetch("/auth/me") refresh-and-retry, then succeeds', async () => {
    const okBody = sseBody(['event: done\ndata: {"answer":"ok","confidence":1,"citations":[],"warnings":[],"related_components":[],"related_error_codes":[],"refused":false}\n\n']);
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 401 }))
      .mockResolvedValueOnce(new Response(okBody, { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    httpClient.apiFetch.mockResolvedValueOnce({ id: 1 });

    const events = await collect(streamChat({ message: 'hi' }));

    expect(httpClient.apiFetch).toHaveBeenCalledWith('/auth/me');
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(events[0].event).toBe('done');
  });

  it('on 403, throws and notifies forbidden without retrying', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify({ detail: 'Missing required scope(s): chat:write' }), { status: 403 }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(collect(streamChat({ message: 'hi' }))).rejects.toMatchObject({ status: 403 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(tokenStore.notifyForbidden).toHaveBeenCalledWith('Missing required scope(s): chat:write');
  });
});
