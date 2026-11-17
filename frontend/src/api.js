// Thin client for the TableQA service. VITE_API_URL is empty in development (the
// dev server proxies to the API) and the API's address in a deployed build.
const BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(message, status, retryAfter) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(BASE + path, options);
  } catch {
    throw new ApiError('The service could not be reached. Check that it is running.', 0);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}).`;
    throw new ApiError(detail, response.status, Number(response.headers.get('retry-after')) || null);
  }
  return body;
}

export const getTables = () => request('/tables');
export const getSystems = () => request('/systems');
export const getSchema = (tableId) => request(`/tables/${encodeURIComponent(tableId)}/schema`);

export function uploadTable(file) {
  const form = new FormData();
  form.append('file', file);
  return request('/tables', { method: 'POST', body: form });
}

export function ask(question, tableId, system) {
  return request('/ask', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question, table_id: tableId, system }),
  });
}
