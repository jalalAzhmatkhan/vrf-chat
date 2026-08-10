import { useMemo } from 'react';
import { InlineContent } from './InlineContent';
import { InlineVisualChip } from './InlineVisualChip';
import { getSafeRenderLength } from '../lib/markerBuffer';
import { parseAnswerBlocks } from '../lib/parseAnswer';
import styles from './AnswerBody.module.css';

/**
 * Renders `TechnicalAnswer.answer` (markdown-lite + inline markers) —
 * Documentation/ui-ux-design/04-chat-ui.md §4/§6.
 *
 * @param {{
 *   rawText: string,
 *   isStreaming: boolean,
 *   citationsByElementId: Map<string, object>,
 *   onOpenCitation: (elementId: string) => void,
 * }} props
 */
export function AnswerBody({ rawText, isStreaming, citationsByElementId, onOpenCitation }) {
  const visibleText = useMemo(() => {
    if (!isStreaming) return rawText;
    return rawText.slice(0, getSafeRenderLength(rawText));
  }, [rawText, isStreaming]);

  const blocks = useMemo(() => parseAnswerBlocks(visibleText), [visibleText]);

  const renderMarker = (elementId, key) => (
    <InlineVisualChip
      key={key}
      elementId={elementId}
      citation={citationsByElementId.get(elementId) ?? null}
      onOpenCitation={onOpenCitation}
    />
  );

  return (
    <div className={styles.root}>
      {blocks.map((block, index) => {
        const isLast = index === blocks.length - 1;
        if (block.type === 'ul' || block.type === 'ol') {
          const ListTag = block.type;
          return (
            <ListTag key={index} className={styles.list}>
              {block.items.map((inline, itemIndex) => (
                <li key={itemIndex} className={styles.listItem}>
                  <InlineContent nodes={inline} renderMarker={renderMarker} />
                  {isLast && itemIndex === block.items.length - 1 && isStreaming && (
                    <span className={styles.cursor} aria-hidden="true" />
                  )}
                </li>
              ))}
            </ListTag>
          );
        }
        return (
          <p key={index} className={styles.paragraph}>
            <InlineContent nodes={block.inline} renderMarker={renderMarker} />
            {isLast && isStreaming && <span className={styles.cursor} aria-hidden="true" />}
          </p>
        );
      })}
      {blocks.length === 0 && isStreaming && <span className={styles.cursor} aria-hidden="true" />}
    </div>
  );
}
