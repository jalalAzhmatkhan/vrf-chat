/**
 * Pure state machine backing `ChatPage` (`04-chat-ui.md` §16 `ChatPage`
 * component spec: `messages`, `conversationId`, per-message
 * `status: 'idle' | 'connecting' | 'status' | 'streaming' | 'done' |
 * 'error'`). Kept framework-agnostic (no React) so it's independently
 * testable.
 *
 * `conversationId` is a `string` (F2-04,
 * `05-streaming-and-api-contract.md` §5.6 — **not** a number). It's also
 * now captured from the `status` and `error` events, not just `done`
 * (§5.6: the backend resolves/creates the conversation row *before* the
 * first event is emitted) — this means a request that fails mid-stream
 * still lets the retry/composer flow continue the *same* conversation
 * instead of silently starting a new one.
 */

/** @returns {{ messages: unknown[], conversationId: string|null }} */
export function initialChatState() {
  return { messages: [], conversationId: null };
}

function newAssistantMessage(id, userText, requestStartedAt) {
  return {
    id,
    role: 'assistant',
    status: 'connecting',
    stage: null,
    requestStartedAt,
    rawAnswer: '',
    citations: [],
    final: null,
    error: null,
    userMessageText: userText,
    timestamp: requestStartedAt,
  };
}

function mapMessage(state, id, updater) {
  return {
    ...state,
    messages: state.messages.map((message) => (message.id === id ? updater(message) : message)),
  };
}

/**
 * @param {ReturnType<typeof initialChatState>} state
 * @param {{ type: string, [key: string]: any }} action
 */
export function chatReducer(state, action) {
  switch (action.type) {
    case 'RESET_CONVERSATION':
      return initialChatState();

    case 'SUBMIT_USER_MESSAGE': {
      const { userId, assistantId, text, timestamp } = action;
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: userId, role: 'user', text, timestamp },
          newAssistantMessage(assistantId, text, timestamp),
        ],
      };
    }

    case 'RETRY_RESET': {
      return mapMessage(state, action.assistantId, (message) => ({
        ...newAssistantMessage(message.id, message.userMessageText, action.timestamp),
      }));
    }

    case 'STREAM_STATUS':
      return {
        ...mapMessage(state, action.assistantId, (message) => ({
          ...message,
          status: 'status',
          stage: action.stage,
        })),
        conversationId: action.conversationId ?? state.conversationId,
      };

    case 'STREAM_TOKEN':
      return mapMessage(state, action.assistantId, (message) => ({
        ...message,
        status: 'streaming',
        rawAnswer: message.rawAnswer + action.delta,
      }));

    case 'STREAM_CITATION':
      return mapMessage(state, action.assistantId, (message) => {
        const key = action.citation.element_id;
        if (message.citations.some((c) => c.element_id === key)) return message;
        return { ...message, citations: [...message.citations, action.citation] };
      });

    case 'STREAM_DONE': {
      const { assistantId, payload } = action;
      return {
        ...mapMessage(state, assistantId, (message) => {
          // Reconcile progressive `citation` events with the authoritative
          // final list, deduped by element_id (§10 "direkonsiliasi ...
          // sebagai sumber kebenaran akhir").
          const finalCitations = payload.citations ?? [];
          const finalIds = new Set(finalCitations.map((c) => c.element_id));
          const extra = message.citations.filter((c) => !finalIds.has(c.element_id));
          return {
            ...message,
            status: 'done',
            stage: null,
            rawAnswer: payload.answer ?? message.rawAnswer,
            citations: [...finalCitations, ...extra],
            final: payload,
          };
        }),
        conversationId: payload.conversation_id ?? state.conversationId,
      };
    }

    case 'STREAM_ERROR':
      return {
        ...mapMessage(state, action.assistantId, (message) => ({
          ...message,
          status: 'error',
          error: action.error,
        })),
        conversationId: action.conversationId ?? state.conversationId,
      };

    default:
      return state;
  }
}
