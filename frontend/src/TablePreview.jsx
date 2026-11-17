import React from 'react';

export default function TablePreview({ schema }) {
  if (!schema) return <p className="muted">Loading the table…</p>;
  return (
    <div className="preview">
      <p className="muted">
        {schema.row_count.toLocaleString()} rows · {schema.columns.length} columns. First rows:
      </p>
      <div className="scroll">
        <table>
          <thead>
            <tr>
              {schema.columns.map((c) => (
                <th key={c.name} title={`${c.name} · ${c.type}`}>
                  {c.header || c.name}
                  <span className="type">{c.type === 'VARCHAR' ? 'text' : 'number'}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {schema.first_rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
