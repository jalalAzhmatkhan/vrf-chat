import { describe, expect, it } from 'vitest';
import { chatReducer, initialChatState } from './chatReducer';

function submit(state, { userId = 'u1', assistantId = 'a1', text = 'hello', timestamp = 1000 } = {}) {
  return chatReducer(state, { type: 'SUBMIT_USER_MESSAGE', userId, assistantId, text, timestamp });
}

describe('chatReducer', () => {
  it('SUBMIT_USER_MESSAGE appends a user bubble + a connecting assistant shell', () => {
    const state = submit(initialChatState());
    expect(state.messages).toHaveLength(2);
    expect(state.messages[0]).toMatchObject({ role: 'user', text: 'hello' });
    expect(state.messages[1]).toMatchObject({ role: 'assistant', status: 'connecting', rawAnswer: '' });
  });

  it('STREAM_STATUS/STREAM_TOKEN update only the targeted assistant message', () => {
    let state = submit(initialChatState());
    state = chatReducer(state, { type: 'STREAM_STATUS', assistantId: 'a1', stage: 'searching_manual' });
    expect(state.messages[1]).toMatchObject({ status: 'status', stage: 'searching_manual' });

    state = chatReducer(state, { type: 'STREAM_TOKEN', assistantId: 'a1', delta: 'Halo ' });
    state = chatReducer(state, { type: 'STREAM_TOKEN', assistantId: 'a1', delta: 'dunia' });
    expect(state.messages[1]).toMatchObject({ status: 'streaming', rawAnswer: 'Halo dunia' });
  });

  it('STREAM_STATUS captures conversation_id early (§5.6 — resolved before the first event, string id per F2-04)', () => {
    let state = submit(initialChatState());
    state = chatReducer(state, {
      type: 'STREAM_STATUS',
      assistantId: 'a1',
      stage: 'searching_manual',
      conversationId: 'conv-abc-123',
    });
    expect(state.conversationId).toBe('conv-abc-123');
  });

  it('STREAM_ERROR captures conversation_id when the backend had already created the row', () => {
    let state = submit(initialChatState());
    state = chatReducer(state, {
      type: 'STREAM_ERROR',
      assistantId: 'a1',
      error: { code: 'timeout', message: 'Timed out' },
      conversationId: 'conv-abc-123',
    });
    expect(state.conversationId).toBe('conv-abc-123');
    expect(state.messages[1]).toMatchObject({ status: 'error', error: { code: 'timeout', message: 'Timed out' } });
  });

  it('STREAM_ERROR without a conversation_id leaves the existing one untouched (never created / connection lost before status)', () => {
    let state = submit(initialChatState());
    state = { ...state, conversationId: 'conv-existing' };
    state = chatReducer(state, {
      type: 'STREAM_ERROR',
      assistantId: 'a1',
      error: { code: 'connection_lost', message: 'Koneksi terputus saat menerima jawaban.' },
    });
    expect(state.conversationId).toBe('conv-existing');
  });

  it('STREAM_CITATION dedupes by element_id', () => {
    let state = submit(initialChatState());
    const citation = { element_id: '5', document_id: '1', page: 2, element_type: 'icon' };
    state = chatReducer(state, { type: 'STREAM_CITATION', assistantId: 'a1', citation });
    state = chatReducer(state, { type: 'STREAM_CITATION', assistantId: 'a1', citation });
    expect(state.messages[1].citations).toHaveLength(1);
  });

  it('STREAM_DONE sets the final payload, reconciles citations, and captures conversation_id', () => {
    let state = submit(initialChatState());
    state = chatReducer(state, {
      type: 'STREAM_CITATION',
      assistantId: 'a1',
      citation: { element_id: 'stale', document_id: '1', page: 1, element_type: 'text' },
    });
    const payload = {
      answer: 'Jawaban final.',
      confidence: 0.9,
      citations: [{ element_id: '5', document_id: '1', page: 2, element_type: 'icon' }],
      warnings: [],
      related_components: [],
      related_error_codes: [],
      refused: false,
      conversation_id: 'conv-42',
    };
    state = chatReducer(state, { type: 'STREAM_DONE', assistantId: 'a1', payload });

    expect(state.conversationId).toBe('conv-42');
    expect(state.messages[1]).toMatchObject({ status: 'done', rawAnswer: 'Jawaban final.', final: payload });
    // The final citations[] list is authoritative; any progressive citation
    // not present in it is still kept (never silently dropped), per §10.
    expect(state.messages[1].citations.map((c) => c.element_id)).toEqual(['5', 'stale']);
  });

  it('STREAM_ERROR marks the message as errored without touching others', () => {
    let state = submit(initialChatState());
    state = chatReducer(state, {
      type: 'STREAM_ERROR',
      assistantId: 'a1',
      error: { code: 'timeout', message: 'Timed out' },
    });
    expect(state.messages[1]).toMatchObject({ status: 'error', error: { code: 'timeout', message: 'Timed out' } });
  });

  it('RETRY_RESET restores a connecting shell reusing the original user text', () => {
    let state = submit(initialChatState());
    state = chatReducer(state, { type: 'STREAM_ERROR', assistantId: 'a1', error: { code: 'x', message: 'y' } });
    state = chatReducer(state, { type: 'RETRY_RESET', assistantId: 'a1', timestamp: 2000 });
    expect(state.messages[1]).toMatchObject({
      status: 'connecting',
      rawAnswer: '',
      error: null,
      userMessageText: 'hello',
    });
    // The user's original bubble is untouched (not duplicated).
    expect(state.messages).toHaveLength(2);
  });

  it('RESET_CONVERSATION clears messages and conversationId', () => {
    let state = submit(initialChatState());
    state = { ...state, conversationId: 'conv-7' };
    state = chatReducer(state, { type: 'RESET_CONVERSATION' });
    expect(state).toEqual(initialChatState());
  });
});
