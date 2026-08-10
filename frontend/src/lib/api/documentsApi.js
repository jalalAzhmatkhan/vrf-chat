import { apiFetch } from '../httpClient';

/**
 * `GET /api/v1/documents` / `/documents/{id}` / `/documents/{id}/pages/{n}` —
 * per Documentation/system-design/05-streaming-and-api-contract.md §6.
 *
 * `listDocuments()` backs the `documentsMap` cache required by
 * `04-chat-ui.md` §10.1 ("FE WAJIB fetch GET /api/v1/documents SEKALI saat
 * halaman chat mount ... cache map id -> {title, manufacturer,
 * model_family}") — callers should fetch once and reuse, not per-citation.
 *
 * @typedef {{
 *   id: number, title: string, manufacturer: string|null,
 *   model_family: string|null, filename: string, page_count: number|null,
 *   status: string, uploaded_at: string,
 * }} DocumentSummary
 */

/** @returns {Promise<DocumentSummary[]>} */
export function listDocuments() {
  return apiFetch('/documents', { method: 'GET' });
}

/**
 * @typedef {{
 *   element_id: number, element_type: string, text: string|null,
 *   bbox: {l:number,t:number,r:number,b:number,page_number:number,coord_origin:string}|null,
 *   image_uri: string|null,
 * }} PageElement
 *
 * @typedef {{
 *   document_id: number, page_number: number, page_image_uri: string|null,
 *   page_width_pt: number|null, page_height_pt: number|null,
 *   image_width_px: number|null, image_height_px: number|null,
 *   elements: PageElement[],
 * }} PageDetail
 */

/**
 * @param {number|string} documentId
 * @param {number} pageNumber
 * @returns {Promise<PageDetail>}
 */
export function getDocumentPage(documentId, pageNumber) {
  return apiFetch(`/documents/${documentId}/pages/${pageNumber}`, { method: 'GET' });
}
