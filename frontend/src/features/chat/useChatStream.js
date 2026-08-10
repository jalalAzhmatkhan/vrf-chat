import { useCallback, useReducer, useRef } from 'react';
import { streamChat } from '../../lib/api/chatApi';
import { chatReducer, initialChatState } from './lib/chatReducer';

/**
 * Owns the SSE lifecycle for `ChatPage` (`04-chat-ui.md` §16): submits a
 * message, dispatches each `ChatStreamEvent` into `chatReducer`, and
 * exposes `retryMessage` for the `ErrorCard` "Coba Lagi" action (§12 —
 * replaces the failed assistant message in place, does NOT re-append the
 * user's question).
 */
export function useChatStream() {
  const [state, dispatch] = useReducer(chatReducer, undefined, initialChatState);
  const conversationIdRef = useRef(null);
  conversationIdRef.current = state.conversationId;
  const inFlightRef = useRef(false);

  const runStream = useCallback(async (assistantId, userText) => {
    inFlightRef.current = true;
    const controller = new AbortController();
    let sawTerminalEvent = false;
    try {
      for await (const evt of streamChat(
        { conversation_id: conversationIdRef.current, message: userText },
        { signal: controller.signal },
      )) {
        switch (evt.event) {
          case 'status':
            // §5.6: `conversation_id` is resolved server-side before the
            // *first* event (not just `done`) — capture it here so a
            // brand-new conversation is already known even if the stream
            // fails before `done`.
            dispatch({ type: 'STREAM_STATUS', assistantId, stage: evt.data?.stage, conversationId: evt.data?.conversation_id });
            break;
          case 'token':
            dispatch({ type: 'STREAM_TOKEN', assistantId, delta: evt.data?.delta ?? '' });
            break;
          case 'citation':
            dispatch({ type: 'STREAM_CITATION', assistantId, citation: evt.data });
            break;
          case 'done':
            sawTerminalEvent = true;
            dispatch({ type: 'STREAM_DONE', assistantId, payload: evt.data });
            break;
          case 'error':
            sawTerminalEvent = true;
            dispatch({
              type: 'STREAM_ERROR',
              assistantId,
              error: evt.data ?? { code: 'unknown', message: 'Terjadi kesalahan.' },
              // §5.6: included "if the conversation row was already
              // created" — undefined/null here just means it wasn't yet,
              // in which case retry correctly falls back to starting a new
              // conversation (state.conversationId unchanged).
              conversationId: evt.data?.conversation_id,
            });
            break;
          default:
            break;
        }
      }
      if (!sawTerminalEvent) {
        // §12 last bullet: SSE connection closed without an explicit
        // `error`/`done` event.
        dispatch({
          type: 'STREAM_ERROR',
          assistantId,
          error: { code: 'connection_lost', message: 'Koneksi terputus saat menerima jawaban.' },
        });
      }
    } catch (err) {
      dispatch({
        type: 'STREAM_ERROR',
        assistantId,
        error: {
          code: err?.status === 'network' ? 'network' : String(err?.status ?? 'client_error'),
          message: err?.detail || 'Koneksi terputus saat menerima jawaban.',
        },
      });
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  const sendMessage = useCallback(
    (text) => {
      const now = Date.now();
      const userId = `u-${now}-${Math.random().toString(36).slice(2, 8)}`;
      const assistantId = `a-${now}-${Math.random().toString(36).slice(2, 8)}`;
      dispatch({ type: 'SUBMIT_USER_MESSAGE', userId, assistantId, text, timestamp: now });
      return runStream(assistantId, text);
    },
    [runStream],
  );

  const retryMessage = useCallback(
    (assistantId, userText) => {
      dispatch({ type: 'RETRY_RESET', assistantId, timestamp: Date.now() });
      return runStream(assistantId, userText);
    },
    [runStream],
  );

  const resetConversation = useCallback(() => {
    dispatch({ type: 'RESET_CONVERSATION' });
  }, []);

  return {
    messages: state.messages,
    conversationId: state.conversationId,
    isStreamingActive: state.messages.some(
      (m) => m.role === 'assistant' && ['connecting', 'status', 'streaming'].includes(m.status),
    ),
    sendMessage,
    retryMessage,
    resetConversation,
  };
}
