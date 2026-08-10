import { useEffect, useRef, useState } from 'react';
import { UserMessageBubble } from './UserMessageBubble';
import { AssistantMessageCard } from './AssistantMessageCard';
import { EmptyState } from './EmptyState';
import { useScrollbarIdle } from '../../../hooks/useScrollbarIdle';
import styles from './MessageList.module.css';

const NEAR_BOTTOM_THRESHOLD_PX = 80;

/**
 * Documentation/ui-ux-design/04-chat-ui.md §1/§5/§14.
 *
 * @param {{
 *   messages: object[],
 *   documentsMap: Map<string, {title:string}>,
 *   onRetry: (assistantId: string) => void,
 *   onFillComposer: (text: string) => void,
 * }} props
 */
export function MessageList({ messages, documentsMap, onRetry, onFillComposer }) {
  const scrollRef = useScrollbarIdle();
  const isNearBottomRef = useRef(true);
  const [showNewMessagesPill, setShowNewMessagesPill] = useState(false);

  // Cheap "did content change" signal — total character volume across all
  // messages (grows monotonically while streaming, and on every new
  // message), avoids diffing message identity.
  const contentVersion = messages.reduce(
    (acc, m) => acc + (m.text?.length ?? 0) + (m.rawAnswer?.length ?? 0) + m.id.length,
    0,
  );

  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    if (isNearBottomRef.current) {
      node.scrollTop = node.scrollHeight;
      setShowNewMessagesPill(false);
    } else {
      setShowNewMessagesPill(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [contentVersion]);

  const handleScroll = () => {
    const node = scrollRef.current;
    if (!node) return;
    const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    isNearBottomRef.current = distanceFromBottom <= NEAR_BOTTOM_THRESHOLD_PX;
    if (isNearBottomRef.current) setShowNewMessagesPill(false);
  };

  const scrollToBottom = () => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
    isNearBottomRef.current = true;
    setShowNewMessagesPill(false);
  };

  return (
    <div className={styles.wrap}>
      <div ref={scrollRef} className={`${styles.scrollArea} scroll-region`} onScroll={handleScroll}>
        <div className={styles.inner}>
          {messages.length === 0 ? (
            <EmptyState onPickExample={onFillComposer} />
          ) : (
            messages.map((message) =>
              message.role === 'user' ? (
                <UserMessageBubble key={message.id} text={message.text} timestamp={message.timestamp} />
              ) : (
                <AssistantMessageCard
                  key={message.id}
                  message={message}
                  documentsMap={documentsMap}
                  onRetry={onRetry}
                  onFillComposer={onFillComposer}
                />
              ),
            )
          )}
        </div>
      </div>
      {showNewMessagesPill && (
        <button type="button" className={styles.newMessagesPill} onClick={scrollToBottom}>
          ↓ Pesan baru
        </button>
      )}
    </div>
  );
}
