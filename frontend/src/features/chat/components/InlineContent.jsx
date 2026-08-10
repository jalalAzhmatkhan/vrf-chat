import { Fragment } from 'react';

/**
 * Renders the inline node tree produced by `lib/parseAnswer.js` — shared by
 * `AnswerBody` (which needs `renderMarker` for `{{el:ID}}` chips) and
 * `CitationList`/`VisualReferenceCard` captions (plain text/mono/bold only,
 * no markers expected there — `renderMarker` is simply omitted).
 *
 * @param {{
 *   nodes: Array<{type: string, value?: string, children?: unknown[], elementId?: string}>,
 *   renderMarker?: (elementId: string, key: string|number) => React.ReactNode,
 * }} props
 */
export function InlineContent({ nodes, renderMarker }) {
  return nodes.map((node, index) => {
    const key = `${node.type}-${index}`;
    switch (node.type) {
      case 'text':
        return <Fragment key={key}>{node.value}</Fragment>;
      case 'mono':
        return (
          <span key={key} className="mono">
            {node.value}
          </span>
        );
      case 'bold':
        return (
          <strong key={key}>
            <InlineContent nodes={node.children} renderMarker={renderMarker} />
          </strong>
        );
      case 'marker':
        return renderMarker ? renderMarker(node.elementId, key) : null;
      default:
        return null;
    }
  });
}
