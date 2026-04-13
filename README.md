# SQL Generator

Convert plain English into SQL queries — no AI models, no API keys. Built with a rule-based NLP engine, FastAPI, and React.

## How It Works

The backend parses your natural language prompt using regex patterns and heuristics to extract:
- Table name (from schema or inferred from context)
- SELECT columns
- WHERE conditions (equality, comparisons, date ranges, LIKE)
- GROUP BY, ORDER BY, LIMIT, DISTINCT

If you provide a database schema, it maps your words to the exact table and column names.

## Examples

| Prompt | Generated SQL |
|--------|--------------|
| show all users | `SELECT * FROM users;` |
| count the number of orders | `SELECT COUNT(*) FROM orders;` |
| top 10 customers by revenue | `SELECT * FROM customers ORDER BY revenue DESC LIMIT 10;` |
| average salary of employees | `SELECT AVG(salary) FROM employees;` |
| orders where total greater than 100 | `SELECT * FROM orders WHERE total > 100;` |
| users grouped by country | `SELECT country, COUNT(*) FROM users GROUP BY country;` |
| orders from last 30 days | `SELECT * FROM orders WHERE created_at >= NOW() - INTERVAL '30 DAYS';` |

## Project Structure

```
sql-generator/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── nlp_sql.py       # Rule-based NLP engine
│   └── requirements.txt
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.jsx
    │   └── index.js
    └── package.json
```

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

App runs at `http://localhost:3000`. The frontend proxies `/generate-sql` requests to the backend on port 8000.

## Supported Query Patterns

**Aggregates:** count, how many, total, sum, average, avg, mean, max, min

**Comparisons:** greater than, less than, at least, at most, between, equals, is

**String matching:** contains, starts with, ends with, like

**Date ranges:** last N days/weeks/months/years, this month, this year, today

**Modifiers:** top N, order by, group by, distinct, limit
