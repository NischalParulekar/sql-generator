import React, { useEffect, useRef, useState } from 'react';
import { ask, getSchema, getSystems, getTables, uploadTable } from './api';
import ResultCard from './ResultCard';
import TablePreview from './TablePreview';

const REPO = 'https://github.com/NischalParulekar/sql-generator';

export default function App() {
  const [tables, setTables] = useState([]);
  const [uploads, setUploads] = useState([]);
  const [systems, setSystems] = useState([]);
  const [selected, setSelected] = useState([]);
  const [tableId, setTableId] = useState('');
  const [schema, setSchema] = useState(null);
  const [question, setQuestion] = useState('');
  const [results, setResults] = useState({});
  const [notice, setNotice] = useState('');
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef(null);

  useEffect(() => {
    Promise.all([getTables(), getSystems()])
      .then(([tableList, systemList]) => {
        setTables(tableList);
        setSystems(systemList);
        setSelected(systemList.map((s) => s.system));
        if (tableList.length) setTableId(tableList[0].table_id);
      })
      .catch((e) => setNotice(e.message));
  }, []);

  useEffect(() => {
    if (!tableId) return;
    setSchema(null);
    setResults({});
    getSchema(tableId)
      .then(setSchema)
      .catch((e) => setNotice(e.message));
  }, [tableId]);

  const examples = schema?.example_questions || [];
  const asking = Object.values(results).some((r) => r.status === 'loading');
  const canAsk = question.trim() && tableId && selected.length && !asking;

  function toggleSystem(system) {
    setSelected((now) => (now.includes(system) ? now.filter((s) => s !== system) : [...now, system]));
  }

  async function submit(event) {
    event.preventDefault();
    if (!canAsk) return;
    setNotice('');
    const q = question.trim();
    const order = systems.map((s) => s.system).filter((s) => selected.includes(s));
    setResults(Object.fromEntries(order.map((s) => [s, { status: 'loading' }])));
    await Promise.all(
      order.map(async (system) => {
        try {
          const data = await ask(q, tableId, system);
          setResults((now) => ({ ...now, [system]: { status: 'done', data } }));
        } catch (e) {
          const message = e.retryAfter ? `${e.message} (about ${e.retryAfter} s)` : e.message;
          setResults((now) => ({ ...now, [system]: { status: 'error', message } }));
        }
      }),
    );
  }

  async function onFile(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setUploading(true);
    setNotice('');
    try {
      const table = await uploadTable(file);
      setUploads((now) => [
        { table_id: table.table_id, name: table.name, row_count: table.row_count },
        ...now.filter((t) => t.table_id !== table.table_id),
      ]);
      setTableId(table.table_id);
    } catch (e) {
      setNotice(e.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <h1>TableQA</h1>
          <p className="lede">
            Ask a question about a table and compare three ways of answering it: a rule-based
            SQL engine, TAPAS, and a Claude agent that writes and repairs SQL.
          </p>
        </div>
        <a className="repo" href={REPO} target="_blank" rel="noreferrer">
          Benchmark results and code
        </a>
      </header>

      {notice && <p className="notice" role="alert">{notice}</p>}

      <main className="layout">
        <section className="panel" aria-labelledby="table-heading">
          <div className="panel-head">
            <h2 id="table-heading">Table</h2>
            <button type="button" className="link" onClick={() => fileInput.current?.click()} disabled={uploading}>
              {uploading ? 'Uploading…' : 'Upload a CSV'}
            </button>
            <input ref={fileInput} type="file" accept=".csv,text/csv" hidden onChange={onFile} />
          </div>
          <label className="field">
            <span className="field-label">Choose a table</span>
            <select id="table" value={tableId} onChange={(e) => setTableId(e.target.value)}>
              {uploads.length > 0 && (
                <optgroup label="Your uploads">
                  {uploads.map((t) => (
                    <option key={t.table_id} value={t.table_id}>
                      {t.name} ({t.row_count.toLocaleString()} rows)
                    </option>
                  ))}
                </optgroup>
              )}
              <optgroup label="Samples">
                {tables.map((t) => (
                  <option key={t.table_id} value={t.table_id}>
                    {t.name} ({t.row_count.toLocaleString()} rows)
                  </option>
                ))}
              </optgroup>
            </select>
          </label>
          <TablePreview schema={schema} />
        </section>

        <section className="panel" aria-labelledby="ask-heading">
          <h2 id="ask-heading">Question</h2>
          <form onSubmit={submit} className="ask">
            <textarea
              id="question"
              rows={2}
              maxLength={500}
              placeholder="e.g. Which driver earned the most points?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) submit(e);
              }}
            />
            {examples.length > 0 && (
              <div className="examples" aria-label="Example questions">
                {examples.map((q) => (
                  <button type="button" key={q} className="chip" onClick={() => setQuestion(q)}>
                    {q}
                  </button>
                ))}
              </div>
            )}
            <fieldset className="systems">
              <legend className="field-label">Systems</legend>
              {systems.map((s) => (
                <label key={s.system} className="check">
                  <input
                    type="checkbox"
                    checked={selected.includes(s.system)}
                    onChange={() => toggleSystem(s.system)}
                  />
                  {s.name}
                  {s.paid && <span className="hint">uses the Claude API</span>}
                </label>
              ))}
            </fieldset>
            <button type="submit" className="primary" disabled={!canAsk}>
              {asking ? 'Answering…' : 'Ask'}
            </button>
          </form>
        </section>

        <section className="results" aria-label="Answers" aria-live="polite">
          {Object.keys(results).length === 0 ? (
            <p className="empty">
              Pick a table, ask a question, and each system's answer appears here with how it got
              there: the rule engine's SQL, the cells TAPAS chose, or every step the agent took.
            </p>
          ) : (
            systems
              .filter((s) => results[s.system])
              .map((s) => <ResultCard key={s.system} name={s.name} result={results[s.system]} />)
          )}
        </section>
      </main>

      <footer className="footer">
        Answers come from the TableQA service. Fresh answers from the Claude agent are rate limited
        and capped per day; repeated questions are served from a cache.
      </footer>
    </div>
  );
}
