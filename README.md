# Norman - Japanese Financial Law RAG System

Hệ thống RAG chuyên về **luật pháp tài chính Nhật Bản**, hỗ trợ **người Việt Nam** sống và làm việc tại Nhật Bản.

**Phạm vi tư vấn:**
- 💰 **Thuế**: Thu nhập, tiêu dùng, cư trú, khai thuế cuối năm (確定申告)
- 🏥 **Bảo hiểm xã hội**: Y tế, lương hưu, thất nghiệp
- 📈 **Đầu tư & Tiết kiệm**: NISA, iDeCo, ふるさと納税
- 💵 **Tài chính cá nhân**: Chuyển tiền quốc tế, thuế cho người nước ngoài

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
python scripts/cli.py search "所得税" --top-k 3
python scripts/cli.py chat "Thuế thu nhập cá nhân ở Nhật tính như thế nào?" --top-k 5
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

thuế thu nhập cá nhân được tính như thế nào? 
thu nhập từ tiền mã hóa có phải đóng thuế không?
tôi đi xe vượt đèn đỏ thì bị phạt tiền bao nhiêu? 