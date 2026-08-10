import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { VisualReferenceCard } from './VisualReferenceCard';
import { __clearElementCacheForTests } from '../../../lib/api/elementsApi';
import * as elementsApi from '../../../lib/api/elementsApi';

vi.mock('../../../lib/api/elementsApi', async () => {
  const actual = await vi.importActual('../../../lib/api/elementsApi');
  return { ...actual, getElementCached: vi.fn() };
});

const documentsMap = new Map([['1', { title: 'Service Manual A' }]]);

beforeEach(() => {
  __clearElementCacheForTests();
  vi.clearAllMocks();
});

describe('VisualReferenceCard — table rendering (05-streaming-and-api-contract.md §5.2.1)', () => {
  it('renders a real <table> straight from citation.content_structured, no roundtrip needed', async () => {
    const citation = {
      document_id: '1',
      page: 240,
      element_id: '77',
      element_type: 'table',
      quote: null,
      image_uri: null,
      content_structured: {
        rows: [{ Kode: 'P8', Deskripsi: 'Proteksi tekanan tinggi', Tindakan: 'Cek TH3' }],
      },
    };
    render(<VisualReferenceCard citation={citation} documentsMap={documentsMap} onOpenCitation={() => {}} />);

    expect(screen.getByRole('cell', { name: 'P8' })).toBeInTheDocument();
    expect(elementsApi.getElementCached).not.toHaveBeenCalled();
  });

  it('falls back to GET /elements/{id} when content_structured is absent on the citation', async () => {
    const citation = {
      document_id: '1',
      page: 240,
      element_id: '77',
      element_type: 'table',
      quote: 'raw text fallback',
      image_uri: null,
      content_structured: null,
    };
    elementsApi.getElementCached.mockResolvedValue({
      content_structured: { rows: [{ Kode: 'U4', Deskripsi: 'Komunikasi terputus' }] },
    });

    render(<VisualReferenceCard citation={citation} documentsMap={documentsMap} onOpenCitation={() => {}} />);

    expect(await screen.findByRole('cell', { name: 'U4' })).toBeInTheDocument();
    expect(elementsApi.getElementCached).toHaveBeenCalledWith('77');
  });

  it('falls back to the raw text block when neither content_structured nor an image is available', async () => {
    const citation = {
      document_id: '1',
      page: 240,
      element_id: '77',
      element_type: 'table',
      quote: 'Kode | Deskripsi\nP8 | Proteksi tekanan tinggi',
      image_uri: null,
      content_structured: null,
    };
    elementsApi.getElementCached.mockResolvedValue({ text: null, content_structured: null, image_uri: null });

    render(<VisualReferenceCard citation={citation} documentsMap={documentsMap} onOpenCitation={() => {}} />);

    expect(screen.getByText(/Kode \| Deskripsi/)).toBeInTheDocument();
    // The fallback fetch still fires (it resolves to nothing useful here) —
    // wait for it so the effect's state update is flushed inside `act`.
    await waitFor(() => expect(elementsApi.getElementCached).toHaveBeenCalledWith('77'));
  });

  it('shows a truncation notice and loads the untruncated table on demand', async () => {
    const citation = {
      document_id: '1',
      page: 240,
      element_id: '77',
      element_type: 'table',
      quote: null,
      image_uri: null,
      content_structured: {
        truncated: true,
        rows: [{ Kode: 'P8', Deskripsi: 'Baris terpotong' }],
      },
    };
    elementsApi.getElementCached.mockResolvedValue({
      content_structured: {
        truncated: false,
        rows: [
          { Kode: 'P8', Deskripsi: 'Baris terpotong' },
          { Kode: 'U4', Deskripsi: 'Baris lengkap tambahan' },
        ],
      },
    });

    const user = userEvent.setup();
    render(<VisualReferenceCard citation={citation} documentsMap={documentsMap} onOpenCitation={() => {}} />);

    expect(screen.getByText(/dipotong/)).toBeInTheDocument();
    expect(screen.queryByRole('cell', { name: 'U4' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Lihat semua baris' }));

    expect(await screen.findByRole('cell', { name: 'U4' })).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText(/dipotong/)).not.toBeInTheDocument());
  });
});
