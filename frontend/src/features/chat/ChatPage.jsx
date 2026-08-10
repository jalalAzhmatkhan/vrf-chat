import { useEffect, useState } from 'react';
import { useChatStream } from './useChatStream';
import { MessageList } from './components/MessageList';
import { Composer } from './components/Composer';
import { Button } from '../../components/Button/Button';
import { listDocuments } from '../../lib/api/documentsApi';
import styles from './ChatPage.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §1, §16 `ChatPage`.
 * Route: `/chat` (routes/routes.jsx), rendered inside `AppShell`.
 */
export function ChatPage() {
  const { messages, sendMessage, retryMessage, resetConversation, isStreamingActive } = useChatStream();
  const [documentsMap, setDocumentsMap] = useState(new Map());
  const [composerValue, setComposerValue] = useState('');

  useEffect(() => {
    // §10.1: fetch once per chat page mount, cache in memory — never
    // per-citation.
    listDocuments()
      .then((docs) => {
        setDocumentsMap(new Map(docs.map((d) => [String(d.id), d])));
      })
      .catch(() => {
        /* Non-fatal — citation rows fall back to "Dokumen {id}" labels. */
      });
  }, []);

  const handleRetry = (assistantId) => {
    const message = messages.find((m) => m.id === assistantId);
    if (!message) return;
    retryMessage(assistantId, message.userMessageText);
  };

  return (
    <div className={styles.page}>
      <div className={styles.toolbar}>
        <Button
          variant="tertiary"
          size="sm"
          onClick={resetConversation}
          disabled={messages.length === 0 || isStreamingActive}
        >
          + Percakapan baru
        </Button>
      </div>
      <MessageList
        messages={messages}
        documentsMap={documentsMap}
        onRetry={handleRetry}
        onFillComposer={setComposerValue}
      />
      <Composer
        disabled={isStreamingActive}
        value={composerValue}
        onChange={setComposerValue}
        onSubmit={sendMessage}
      />
    </div>
  );
}
