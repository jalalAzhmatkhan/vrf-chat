import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { CitationViewerProvider } from './CitationViewerContext';
import { useCitationViewer } from './useCitationViewer';
import * as documentsApi from '../../lib/api/documentsApi';
import * as elementsApi from '../../lib/api/elementsApi';

vi.mock('../../lib/api/documentsApi');
vi.mock('../../lib/api/elementsApi');

const PAGE_DATA = {
  document_id: 1,
  page_number: 238,
  page_image_uri: 'https://example.test/page-238.png',
  page_width_pt: 609,
  page_height_pt: 793,
  image_width_px: 1691,
  image_height_px: 2203,
  elements: [
    { element_id: 12, element_type: 'icon', text: null, bbox: { l: 100, t: 700, r: 120, b: 680 }, image_uri: null },
    { element_id: 99, element_type: 'text', text: 'other', bbox: { l: 10, t: 10, r: 50, b: 5 }, image_uri: null },
  ],
};

const CITATIONS = [
  { element_id: '12', document_id: '1', page: 238, element_type: 'icon', quote: 'tekan tombol ini' },
];

function TriggerButton() {
  const { open } = useCitationViewer();
  return (
    <button type="button" onClick={() => open('12', { citations: CITATIONS, documentsMap: new Map([['1', { title: 'Service Manual A' }]]) })}>
      Open citation
    </button>
  );
}

function renderHarness() {
  return render(
    <CitationViewerProvider>
      <TriggerButton />
    </CitationViewerProvider>,
  );
}

describe('CitationViewerProvider + CitationViewerPanel', () => {
  it('opens the panel, fetches the page, and renders document/page info', async () => {
    documentsApi.getDocumentPage.mockResolvedValue(PAGE_DATA);
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByRole('button', { name: 'Open citation' }));

    await waitFor(() => expect(documentsApi.getDocumentPage).toHaveBeenCalledWith('1', 238));
    expect(await screen.findByText('Service Manual A')).toBeInTheDocument();
    expect(screen.getByText('Hal. 238')).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /Halaman 238/ })).toHaveAttribute('src', PAGE_DATA.page_image_uri);
  });

  it('resolves document_id/page via GET /elements/{id} when the citation lacks them', async () => {
    documentsApi.getDocumentPage.mockResolvedValue(PAGE_DATA);
    elementsApi.getElementCached.mockResolvedValue({
      id: 12,
      document_id: 1,
      page_number: 238,
      element_type: 'icon',
      image_uri: null,
      visual_description: null,
    });

    function UnresolvedTrigger() {
      const { open } = useCitationViewer();
      return (
        <button type="button" onClick={() => open('12', { citations: [], documentsMap: new Map() })}>
          Open unresolved
        </button>
      );
    }
    const user = userEvent.setup();
    render(
      <CitationViewerProvider>
        <UnresolvedTrigger />
      </CitationViewerProvider>,
    );

    await user.click(screen.getByRole('button', { name: 'Open unresolved' }));

    await waitFor(() => expect(elementsApi.getElementCached).toHaveBeenCalledWith('12'));
    await waitFor(() => expect(documentsApi.getDocumentPage).toHaveBeenCalledWith(1, 238));
  });

  it('closes on Escape', async () => {
    documentsApi.getDocumentPage.mockResolvedValue(PAGE_DATA);
    const user = userEvent.setup();
    renderHarness();

    await user.click(screen.getByRole('button', { name: 'Open citation' }));
    const dialog = await screen.findByRole('dialog', { hidden: true });
    await waitFor(() => expect(dialog).toHaveAttribute('aria-hidden', 'false'));

    await user.keyboard('{Escape}');
    await waitFor(() => expect(dialog).toHaveAttribute('aria-hidden', 'true'));
  });
});
