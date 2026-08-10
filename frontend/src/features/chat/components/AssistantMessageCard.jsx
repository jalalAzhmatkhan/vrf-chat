import { useCitationViewer } from '../../citation-viewer/useCitationViewer';
import { AssistantStatusIndicator } from './AssistantStatusIndicator';
import { AnswerBody } from './AnswerBody';
import { VisualReferenceCard } from './VisualReferenceCard';
import { WarningsBlock } from './WarningsBlock';
import { RelatedChips } from './RelatedChips';
import { CitationList } from './CitationList';
import { RefusedNotice } from './RefusedNotice';
import { ErrorCard } from './ErrorCard';
import { extractMarkerElementIds } from '../lib/parseAnswer';
import styles from './AssistantMessageCard.module.css';

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
}

const NON_INLINE_VISUAL_TYPES = new Set(['figure', 'diagram', 'table']);

/**
 * Documentation/ui-ux-design/04-chat-ui.md §1/§16 `AssistantMessage`.
 *
 * @param {{
 *   message: object,
 *   documentsMap: Map<string, {title:string}>,
 *   onRetry: (assistantId: string) => void,
 *   onFillComposer: (text: string) => void,
 * }} props
 */
export function AssistantMessageCard({ message, documentsMap, onRetry, onFillComposer }) {
  const { open: openCitationViewer } = useCitationViewer();

  const openCitation = (elementId) => {
    openCitationViewer(elementId, { citations: message.citations, documentsMap });
  };

  const citationsByElementId = new Map(message.citations.map((c) => [c.element_id, c]));

  const isStreamingActive = message.status === 'connecting' || message.status === 'status' || message.status === 'streaming';
  const hasText = message.rawAnswer.length > 0;

  let body;
  if (message.status === 'error') {
    body = (
      <ErrorCard code={message.error?.code} message={message.error?.message} onRetry={() => onRetry(message.id)} />
    );
  } else if (!hasText && isStreamingActive) {
    body = <AssistantStatusIndicator stage={message.stage} requestStartedAt={message.requestStartedAt} />;
  } else if (message.status === 'done' && message.final?.refused) {
    body = (
      <RefusedNotice
        answer={message.final.answer}
        relatedErrorCodes={message.final.related_error_codes ?? []}
        relatedComponents={message.final.related_components ?? []}
        citations={message.citations}
        documentsMap={documentsMap}
        onOpenCitation={openCitation}
        onFillComposer={onFillComposer}
      />
    );
  } else {
    const markerIds = extractMarkerElementIds(message.rawAnswer);
    const visualReferenceCitations =
      message.status === 'done'
        ? message.citations.filter(
            (c) => NON_INLINE_VISUAL_TYPES.has(c.element_type) && !markerIds.has(c.element_id),
          )
        : [];

    body = (
      <>
        <AnswerBody
          rawText={message.rawAnswer}
          isStreaming={isStreamingActive}
          citationsByElementId={citationsByElementId}
          onOpenCitation={openCitation}
        />
        {message.status === 'done' && (
          <>
            {visualReferenceCitations.length > 0 && (
              <div className={styles.visualReferenceSection}>
                <span className={styles.sectionLabel}>Referensi Visual</span>
                {visualReferenceCitations.map((citation) => (
                  <VisualReferenceCard
                    key={citation.element_id}
                    citation={citation}
                    documentsMap={documentsMap}
                    onOpenCitation={openCitation}
                  />
                ))}
              </div>
            )}
            <WarningsBlock warnings={message.final?.warnings ?? []} />
            <RelatedChips
              relatedErrorCodes={message.final?.related_error_codes ?? []}
              relatedComponents={message.final?.related_components ?? []}
              onFillComposer={onFillComposer}
            />
            <CitationList
              citations={message.citations}
              documentsMap={documentsMap}
              confidence={message.final?.confidence ?? 0}
              refused={false}
              onOpenCitation={openCitation}
            />
          </>
        )}
      </>
    );
  }

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <span className={styles.avatar} aria-hidden="true">
          ◈
        </span>
        <span className={styles.name}>VRF/VRV AI</span>
        <span className={`${styles.timestamp} mono`}>{formatTime(message.timestamp)}</span>
      </div>
      <div className={styles.body}>{body}</div>
    </div>
  );
}
