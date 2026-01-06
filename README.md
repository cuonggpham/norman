# Norman - Japanese Legal RAG System

Hệ thống RAG để tra cứu và tìm kiếm văn bản pháp luật Nhật Bản.

## Project Structure

```
norman/
├── backend/          # FastAPI backend
│   ├── app/          # Application code
│   ├── scripts/      # CLI tools & utilities
│   └── tests/        # Python tests
│
├── frontend/         # React + Vite frontend
│   └── src/          # React components
│
├── data/             # Shared data (gitignored)
└── docs/             # Documentation
```

## Quick Start

### Backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### CLI Testing

```bash
cd backend
source venv/bin/activate
python scripts/cli.py health
python scripts/cli.py search "労働時間" --top-k 3
python scripts/cli.py chat "Quy định về thời gian làm việc" --top-k 5
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/search` | POST | Vector search |
| `/api/chat` | POST | RAG chat with LLM |

## Tech Stack

- **Backend**: FastAPI, Qdrant Cloud, OpenAI
- **Frontend**: React, Vite
- **Data**: 15,629 Japanese legal document chunks
