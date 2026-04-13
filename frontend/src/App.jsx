import React, { useState } from 'react';

const styles = {
  app: {
    minHeight: '100vh',
    backgroundColor: '#0f1117',
    color: '#e2e8f0',
    fontFamily: "'Segoe UI', system-ui, sans-serif",
    padding: '40px 24px',
  },
  container: {
    maxWidth: '860px',
    margin: '0 auto',
  },
  header: {
    marginBottom: '36px',
  },
  title: {
    fontSize: '28px',
    fontWeight: '700',
    color: '#f8fafc',
    margin: '0 0 6px 0',
  },
  subtitle: {
    fontSize: '14px',
    color: '#64748b',
    margin: 0,
  },
  label: {
    display: 'block',
    fontSize: '13px',
    fontWeight: '600',
    color: '#94a3b8',
    marginBottom: '8px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  textarea: {
    width: '100%',
    backgroundColor: '#1e2130',
    border: '1px solid #2d3148',
    borderRadius: '8px',
    color: '#e2e8f0',
    fontSize: '14px',
    padding: '12px 14px',
    resize: 'vertical',
    outline: 'none',
    boxSizing: 'border-box',
    fontFamily: 'inherit',
    lineHeight: '1.6',
    transition: 'border-color 0.15s',
  },
  schemaTextarea: {
    fontFamily: "'Consolas', 'Monaco', monospace",
    fontSize: '13px',
  },
  section: {
    marginBottom: '20px',
  },
  button: {
    backgroundColor: '#4f46e5',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    padding: '11px 28px',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'background-color 0.15s',
    marginTop: '4px',
  },
  buttonDisabled: {
    backgroundColor: '#312e81',
    cursor: 'not-allowed',
    opacity: 0.7,
  },
  resultBox: {
    backgroundColor: '#1e2130',
    border: '1px solid #2d3148',
    borderRadius: '8px',
    padding: '16px',
    marginTop: '24px',
    position: 'relative',
  },
  resultHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  resultLabel: {
    fontSize: '13px',
    fontWeight: '600',
    color: '#94a3b8',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  copyBtn: {
    backgroundColor: 'transparent',
    border: '1px solid #2d3148',
    color: '#94a3b8',
    borderRadius: '6px',
    padding: '4px 12px',
    fontSize: '12px',
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  pre: {
    margin: 0,
    fontFamily: "'Consolas', 'Monaco', monospace",
    fontSize: '13px',
    color: '#a5f3fc',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    lineHeight: '1.7',
  },
  error: {
    backgroundColor: '#2d1b1b',
    border: '1px solid #7f1d1d',
    borderRadius: '8px',
    padding: '12px 16px',
    marginTop: '16px',
    color: '#fca5a5',
    fontSize: '14px',
  },
  divider: {
    height: '1px',
    backgroundColor: '#1e2130',
    margin: '24px 0',
  },
  toggle: {
    background: 'none',
    border: 'none',
    color: '#4f46e5',
    fontSize: '13px',
    cursor: 'pointer',
    padding: '0',
    marginBottom: '10px',
    textDecoration: 'underline',
  },
};

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [schema, setSchema] = useState('');
  const [showSchema, setShowSchema] = useState(false);
  const [sql, setSql] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError('');
    setSql('');

    try {
      const res = await fetch('/generate-sql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, schema }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to generate SQL');
      setSql(data.sql);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleGenerate();
  };

  return (
    <div style={styles.app}>
      <div style={styles.container}>
        <div style={styles.header}>
          <h1 style={styles.title}>SQL Generator</h1>
          <p style={styles.subtitle}>Describe what you want in plain English — get exact SQL back.</p>
        </div>

        <div style={styles.section}>
          <label style={styles.label}>What do you want to query?</label>
          <textarea
            style={styles.textarea}
            rows={4}
            placeholder="e.g. Find the top 10 customers by total order value in the last 30 days"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>

        <div style={styles.section}>
          <button style={styles.toggle} onClick={() => setShowSchema(s => !s)}>
            {showSchema ? '− Hide schema' : '+ Add schema (optional)'}
          </button>
          {showSchema && (
            <>
              <label style={styles.label}>Database Schema</label>
              <textarea
                style={{ ...styles.textarea, ...styles.schemaTextarea }}
                rows={6}
                placeholder={`e.g.\nCREATE TABLE orders (id INT, customer_id INT, total DECIMAL, created_at TIMESTAMP);\nCREATE TABLE customers (id INT, name VARCHAR, email VARCHAR);`}
                value={schema}
                onChange={(e) => setSchema(e.target.value)}
              />
            </>
          )}
        </div>

        <button
          style={{ ...styles.button, ...(loading || !prompt.trim() ? styles.buttonDisabled : {}) }}
          onClick={handleGenerate}
          disabled={loading || !prompt.trim()}
        >
          {loading ? 'Generating…' : 'Generate SQL'}
        </button>

        {error && <div style={styles.error}>{error}</div>}

        {sql && (
          <div style={styles.resultBox}>
            <div style={styles.resultHeader}>
              <span style={styles.resultLabel}>Generated SQL</span>
              <button style={styles.copyBtn} onClick={handleCopy}>
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
            <pre style={styles.pre}>{sql}</pre>
          </div>
        )}

        <div style={styles.divider} />
        <p style={{ color: '#334155', fontSize: '12px', textAlign: 'center', margin: 0 }}>
          Press Ctrl+Enter to generate · Powered by Claude
        </p>
      </div>
    </div>
  );
}
