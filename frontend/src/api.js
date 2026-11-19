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

// The static site has no service behind it: it replays answers recorded during the
// benchmark runs, from demo.json (built by scripts/build_demo.py).
export const RECORDED = import.meta.env.VITE_RECORDED === '1';
// Address of the other build (live <-> recorded) when both are published.
export const OTHER_URL = import.meta.env.VITE_OTHER_URL || '';
let bundle = null;
async function demo() {
  if (!bundle) {
    const r = await fetch(`${import.meta.env.BASE_URL}demo.json`);
    if (!r.ok) throw new ApiError('The recorded demo could not be loaded.', r.status);
    bundle = await r.json();
  }
  return bundle;
}

export async function getTables() {
  if (!RECORDED) return request('/tables');
  const b = await demo();
  return Object.values(b.tables).map(({ table_id, name, row_count, example_questions }) => ({
    table_id, name, row_count, example_questions,
  }));
}

export async function getSystems() {
  if (!RECORDED) return request('/systems');
  return (await demo()).systems;
}

export async function getSchema(tableId) {
  if (!RECORDED) return request(`/tables/${encodeURIComponent(tableId)}/schema`);
  const table = (await demo()).tables[tableId];
  if (!table) throw new ApiError('No such table in the recorded demo.', 404);
  return table;
}

export function uploadTable(file) {
  if (RECORDED) {
    return Promise.reject(new ApiError('Uploads need the live service; this page replays recorded answers.', 0));
  }
  const form = new FormData();
  form.append('file', file);
  return request('/tables', { method: 'POST', body: form });
}

export async function ask(question, tableId, system) {
  if (!RECORDED) {
    return request('/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question, table_id: tableId, system }),
    });
  }
  const entry = (await demo()).answers[tableId]?.[question];
  if (!entry) {
    throw new ApiError('This page only has recorded answers for the example questions. Pick one of those.', 0);
  }
  const r = entry.systems[system];
  // A short pause so three answers do not appear as one flash; the recorded latency is shown.
  await new Promise((resolve) => setTimeout(resolve, 250));
  return {
    answer: r.answer,
    error: r.error,
    cached: false,
    recorded: true,
    latency_ms: r.latency_ms,
    system_latency_ms: r.latency_ms,
    cost_usd: r.cost_usd,
    tokens: r.tokens,
    trace: r.trace,
    gold: entry.gold,
    correct: r.correct,
  };
}
