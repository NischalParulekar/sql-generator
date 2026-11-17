import React from 'react';

function duration(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function money(usd) {
  if (!usd) return '$0';
  return usd < 0.01 ? `$${usd.toFixed(5)}` : `$${usd.toFixed(2)}`;
}

function clip(text, n = 600) {
  const s = typeof text === 'string' ? text : JSON.stringify(text);
  return s.length > n ? `${s.slice(0, n)} …` : s;
}

function SqlTrace({ sql }) {
  return (
    <>
      <p className="trace-label">Generated SQL</p>
      <pre className="code">{sql}</pre>
    </>
  );
}

function TapasTrace({ step }) {
  return (
    <dl className="facts">
      <dt>Operator</dt>
      <dd>{step.operator}</dd>
      <dt>Cells chosen</dt>
      <dd>{step.cells.length ? step.cells.map(([r, c]) => `row ${r + 1}, column ${c + 1}`).join('; ') : 'none'}</dd>
      <dt>Chunks</dt>
      <dd>{step.chunks}</dd>
      <dt>Rows it could not see</dt>
      <dd>{step.rows_dropped}</dd>
    </dl>
  );
}

function ToolCall({ call }) {
  if (call.name === 'final_answer') {
    return (
      <div className="call final">
        <span className="tool">final_answer</span> {call.input.values.join(' | ')}
      </div>
    );
  }
  if (call.name === 'run_sql') {
    return (
      <div className={`call${call.error ? ' failed' : ''}`}>
        <span className="tool">run_sql</span>
        <pre className="code">{call.input.query}</pre>
        <pre className="result">{clip(call.result)}</pre>
      </div>
    );
  }
  const args = call.input && Object.keys(call.input).length ? ` ${JSON.stringify(call.input)}` : '';
  return (
    <div className="call">
      <details>
        <summary>
          <span className="tool">{call.name}</span>
          {args}
        </summary>
        <pre className="result">{clip(call.result, 1500)}</pre>
      </details>
    </div>
  );
}

function AgentTrace({ steps }) {
  return (
    <ol className="steps">
      {steps.map((step) => (
        <li key={step.step}>
          <p className="trace-label">
            Turn {step.step} · {step.input_tokens + step.cache_read_tokens + step.cache_write_tokens} in,{' '}
            {step.output_tokens} out
          </p>
          {step.text.map((t, i) => (
            <p key={i} className="said">{t}</p>
          ))}
          {step.tool_calls.map((call, i) => (
            <ToolCall key={i} call={call} />
          ))}
        </li>
      ))}
    </ol>
  );
}

function Trace({ trace }) {
  if (typeof trace === 'string') return <SqlTrace sql={trace} />;
  if (Array.isArray(trace) && trace[0] && 'operator' in trace[0]) return <TapasTrace step={trace[0]} />;
  if (Array.isArray(trace) && trace.length) return <AgentTrace steps={trace} />;
  return <p className="muted">No trace recorded.</p>;
}

export default function ResultCard({ name, result }) {
  if (result.status === 'loading') {
    return (
      <article className="card loading">
        <h3>{name}</h3>
        <p className="muted">Working…</p>
      </article>
    );
  }
  if (result.status === 'error') {
    return (
      <article className="card">
        <h3>{name}</h3>
        <p className="problem">{result.message}</p>
      </article>
    );
  }
  const d = result.data;
  return (
    <article className="card">
      <div className="card-head">
        <h3>{name}</h3>
        {d.cached && <span className="badge">From cache</span>}
      </div>
      {d.answer.length ? (
        <ul className="answer">
          {d.answer.slice(0, 12).map((a, i) => (
            <li key={i}>{a}</li>
          ))}
          {d.answer.length > 12 && <li className="muted">and {d.answer.length - 12} more</li>}
        </ul>
      ) : (
        <p className="problem">No answer{d.error ? `: ${d.error}` : ''}</p>
      )}
      <p className="meta">
        {duration(d.latency_ms)}
        {d.cached && ` (first answered in ${duration(d.system_latency_ms)})`}
        {' · '}
        {money(d.cost_usd)}
        {d.tokens > 0 && ` · ${d.tokens.toLocaleString()} tokens`}
      </p>
      <details className="trace">
        <summary>How it got there</summary>
        <Trace trace={d.trace} />
      </details>
    </article>
  );
}
