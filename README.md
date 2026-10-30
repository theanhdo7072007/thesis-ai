# THESIS AI — MVP 0.1

A minimal cloud-ready investment research workspace. This starter includes:
- FastAPI backend and interactive web UI
- PostgreSQL persistence (Docker Compose)
- Watchlist CRUD
- AI research endpoint using an OpenAI-compatible API
- Saved research report endpoints
- Research message history
- Clear disclosure that live market data is not connected

## 1. Requirements
- Docker Desktop (recommended), or Python 3.12+
- An LLM API key for live AI responses

## 2. Run with Docker
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
docker compose up --build
```
Open http://localhost:8000

Without an API key, the research form runs in demo mode and explains that no model was called. Do not treat demo output as investment analysis.

## 3. Run locally without Docker
Create a virtual environment, install dependencies and run:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
The local default database is SQLite. Set environment variables as needed. API docs: http://localhost:8000/docs

## 4. API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/assets` | List watchlist |
| POST | `/api/assets` | Add ticker |
| DELETE | `/api/assets/{ticker}` | Remove ticker |
| POST | `/api/research` | Ask AI research question |
| GET | `/api/reports` | List saved reports |
| POST | `/api/reports` | Save report |
| GET | `/api/messages` | List chat messages |

Example research request:
```bash
curl -X POST http://localhost:8000/api/research \\
  -H 'Content-Type: application/json' \\
  -d '{"ticker":"FPT","question":"Liệt kê các yếu tố cần kiểm tra khi phân tích chất lượng lợi nhuận của doanh nghiệp này."}'
```

Example save report:
```bash
curl -X POST http://localhost:8000/api/reports \\
  -H 'Content-Type: application/json' \\
  -d '{"ticker":"FPT","title":"FPT initial research","thesis":"Research notes pending verification","risks":"Data not yet connected","assumptions":{"status":"illustrative"},"sources":[]}'
```

## 5. Database schema
- `assets`: ticker, name, market, currency, sector, notes
- `research_reports`: ticker, title, thesis, risks, assumptions (JSON), sources (JSON), created_at
- `research_messages`: role, content, ticker, created_at

Tables are created automatically at startup. For production, introduce Alembic migrations.

## 6. Important limitations before public cloud deployment
This is a runnable MVP scaffold, not a production investment platform.
1. Authentication and per-user authorization are not implemented. Keep it private; do not expose the service publicly as-is.
2. Live price feeds, financial statements, corporate actions, macro data, and news connectors are not implemented.
3. The AI cannot verify facts against external sources unless source data is added to the system. It is instructed not to invent realtime prices.
4. No PDF ingestion, DCF calculation engine, backtesting, or automated trading in this first cut.
5. Add TLS, secrets management, rate limiting, audit logs, backups, database migrations, and monitoring before public deployment.
6. Check API provider terms, market-data licenses, and local regulatory obligations.

## 7. Next development milestones
1. Authentication (single owner)
2. Licensed VN + global market data adapters
3. Financial statement normalization and provenance
4. PDF ingestion + pgvector retrieval
5. Deterministic valuation module (DCF, multiples, sensitivity)
6. Automated research scheduler and report QA
