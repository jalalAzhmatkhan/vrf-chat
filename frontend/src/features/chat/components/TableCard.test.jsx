import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TableCard } from './TableCard';

describe('TableCard', () => {
  it('renders nothing when there are no rows', () => {
    const { container } = render(<TableCard contentStructured={{ rows: [] }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing when contentStructured is null', () => {
    const { container } = render(<TableCard contentStructured={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('derives columns from the union of row keys and renders every row', () => {
    const contentStructured = {
      rows: [
        { Kode: 'P8', Deskripsi: 'Proteksi tekanan tinggi', Tindakan: 'Cek TH3' },
        { Kode: 'U4', Deskripsi: 'Komunikasi terputus' },
      ],
    };
    render(<TableCard contentStructured={contentStructured} />);

    expect(screen.getByRole('columnheader', { name: 'Kode' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Deskripsi' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Tindakan' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'P8' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'U4' })).toBeInTheDocument();
    // Missing cell for row 2 ("Tindakan") renders as an empty cell, not a dropped column.
    expect(screen.getAllByRole('row')).toHaveLength(3); // 1 header + 2 data rows
  });

  it('uses explicit headers order when provided', () => {
    const contentStructured = {
      headers: ['Tindakan', 'Kode'],
      rows: [{ Kode: 'P8', Tindakan: 'Cek TH3' }],
    };
    render(<TableCard contentStructured={contentStructured} />);
    const headers = screen.getAllByRole('columnheader').map((el) => el.textContent);
    expect(headers).toEqual(['Tindakan', 'Kode']);
  });

  it('right-aligns/mono-styles columns that look like identifiers or numeric+unit values', () => {
    const contentStructured = {
      rows: [
        { Kode: 'P8', Tegangan: '220V', Deskripsi: 'Proteksi tekanan tinggi pada outdoor unit' },
        { Kode: 'U4', Tegangan: '12V', Deskripsi: 'Komunikasi terputus antara indoor dan outdoor' },
      ],
    };
    render(<TableCard contentStructured={contentStructured} />);
    const kodeHeader = screen.getByRole('columnheader', { name: 'Kode' });
    const deskripsiHeader = screen.getByRole('columnheader', { name: 'Deskripsi' });
    expect(kodeHeader.className).toMatch(/mono/);
    expect(deskripsiHeader.className).not.toMatch(/mono/);
  });
});
