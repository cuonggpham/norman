# Japanese Legal RAG System - Roadmap

Lộ trình phát triển hệ thống RAG cho văn bản pháp luật Nhật Bản.

---

## ✅ Phase 0: Data Collection & Processing (Current)
- [x] Download XML từ e-Gov API
- [x] Parse XML → JSON với cấu trúc hierarchical
- [ ] **Chunking data** để chuẩn bị cho embedding

---

## 📋 Phase 1: Vector Search with Qdrant

### 1.1 Setup Infrastructure
```bash
# Self-host Qdrant với Docker
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

### 1.2 Embedding với OpenAI
- Model: `text-embedding-3-small` (1536 dimensions) hoặc `text-embedding-3-large` (3072 dimensions)
- Batch processing để tối ưu API calls
- Caching embeddings để tránh duplicate calls

### 1.3 Indexing Pipeline
```
Chunks → OpenAI Embedding → Qdrant Upsert
```

### 1.4 Search Implementation
- Vector similarity search
- Metadata filtering (by law, category, article)
- Hybrid search (vector + keyword) nếu cần

---

## 📋 Phase 2: Reranking Integration

### 2.1 Reranker Options
| Model | Type | Pros | Cons |
|-------|------|------|------|
| Cohere Rerank | API | Chất lượng cao | Trả phí |
| BAAI/bge-reranker-large | Local | Miễn phí, đa ngôn ngữ | Cần GPU |
| cross-encoder/ms-marco | Local | Nhanh | Chủ yếu English |

### 2.2 Two-Stage Retrieval
```
Query → Vector Search (top 50) → Rerank → Final Results (top 5)
```

---

## 📋 Phase 3: Response Generation with Highlighting

### 3.1 LLM Integration
- Sử dụng retrieved chunks làm context
- Generate answer với citations

### 3.2 Highlight Response Format
```json
{
  "answer": "Theo Điều 1...",
  "sources": [
    {
      "law_title": "労働基準法",
      "article": "第一条",
      "text": "...",
      "highlight_path": ["労働基準法", "第一章", "第一条"]
    }
  ]
}
```

### 3.3 API Design
```
POST /api/search
{
  "query": "労働時間の制限",
  "top_k": 5,
  "filters": { "category": "労働" }
}
```

---

## 📋 Phase 4: Graph RAG with Neo4j

### 4.1 Graph Schema
```
(Law) -[:HAS_CHAPTER]-> (Chapter) -[:HAS_ARTICLE]-> (Article)
(Article) -[:REFERENCES]-> (Article)
(Article) -[:DEFINES]-> (LegalTerm)
(Law) -[:AMENDS]-> (Law)
```

### 4.2 Use Cases
- Tìm tất cả điều liên quan đến một điều cụ thể
- Trace lịch sử sửa đổi của một điều
- Tìm định nghĩa thuật ngữ pháp lý

### 4.3 Implementation
```bash
# Self-host Neo4j
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

---

## 📋 Phase 5: Production Deployment (Future)

### 5.1 API Server
- FastAPI backend
- Rate limiting
- Caching layer (Redis)

### 5.2 Frontend
- Search UI với highlight
- Law browser với navigation

### 5.3 Monitoring
- Search quality metrics
- Latency tracking
- Error monitoring

---

## Tech Stack Summary

| Component | Technology |
|-----------|------------|
| Embedding | OpenAI text-embedding-3 |
| Vector DB | Qdrant (self-hosted Docker) |
| Reranker | TBD (Cohere/BGE) |
| Graph DB | Neo4j (future) |
| Backend | FastAPI (Python) |
| LLM | OpenAI GPT-4 / Claude |

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 0 | 1 day | 🟡 In Progress |
| Phase 1 | 3-5 days | ⬜ Pending |
| Phase 2 | 2-3 days | ⬜ Pending |
| Phase 3 | 3-5 days | ⬜ Pending |
| Phase 4 | 5-7 days | ⬜ Pending |
