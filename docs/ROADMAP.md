# Japanese Financial Law RAG System - Roadmap

Lộ trình phát triển hệ thống RAG cho **luật pháp tài chính Nhật Bản** (thuế, bảo hiểm xã hội, đầu tư) - hỗ trợ người Việt Nam.

**Last Updated**: 2026-01-25

---

## ✅ Phase 0: Data Collection & Processing (Complete)

### 0.1 Data Collection
- [x] Research e-Gov Laws API documentation
- [x] Implement `downloader.py` với rate limiting (1.2s/request)
- [x] Download XML từ e-Gov API 
- [x] Filter by era (Showa+), status (CurrentEnforced), type (Act/CabinetOrder)

### 0.2 XML Parsing
- [x] Parse XML → JSON với cấu trúc hierarchical
- [x] Extract metadata: law_id, law_title, category, promulgation_date
- [x] Handle TOC, MainProvision, SupplementaryProvision

### 0.3 Smart Chunking
- [x] Implement paragraph-level chunking (Điều → Khoản làm đơn vị)
- [x] Add context enrichment (law_title + article_title + caption)
- [x] Create `text_with_context` field for better embedding quality
- [x] Generate `highlight_path` for UI navigation

### 0.4 Embedding
- [x] Batch processing với OpenAI text-embedding-3-large (3072 dimensions)
- [x] Resume capability cho long-running jobs
- [x] Caching embeddings trong `data/embeddings/`

**📊 Statistics:**
| Metric | Value |
|--------|-------|
| Total Laws | 233 |
| Total Chunks | 15,629 |
| Avg Chunk Size | ~800 characters |
| Embedding Dimensions | 3,072 |
| Storage Size | ~192 MB (.npy files) |

---

## ✅ Phase 1: Vector Search with Qdrant (Complete)

### 1.1 Setup Infrastructure
- [x] Sử dụng **Qdrant Cloud Free Tier** (1GB storage)
- [x] Cấu hình `.env` với QDRANT_URL và QDRANT_API_KEY
- [x] Create collection với dense vector config

### 1.2 Indexing Pipeline
- [x] Implement `app/db/qdrant.py` - Qdrant client wrapper
- [x] Implement `scripts/indexer.py` - Batch upload với retry logic
- [x] Upload 15,629 vectors lên Qdrant Cloud
- [x] Create payload indexes (`category`, `law_title`) for filtering

### 1.3 Vector-only Search
- [x] Vector similarity search với cosine distance
- [x] Metadata filtering (by law_id, category, etc.)
- [x] Top-K retrieval với score threshold

---

## ✅ Phase 2: Response Generation with API (Complete)

### 2.1 LLM Integration
- [x] Sử dụng retrieved chunks làm context
- [x] Generate answer với citations (OpenAI GPT-4o-mini)
- [x] Prompt engineering cho output tiếng Việt + Japanese annotations

### 2.2 Query Processing
- [x] Query translation Vietnamese → Japanese (`query_translator.py`)
- [x] Query expansion với related terms và keywords
- [x] Multi-query retrieval (original + translated + expanded queries)
- [x] Deduplicate results giữ highest score

### 2.3 Response Format
```json
{
  "answer": "Theo Điều 32 Luật Tiêu chuẩn Lao động (労働基準法)...",
  "sources": [
    {
      "law_title": "労働基準法",
      "article": "第三十二条",
      "text": "...",
      "highlight_path": ["労働基準法", "第四章", "第三十二条"]
    }
  ]
}
```

### 2.4 API Endpoints
- [x] Implement `POST /api/search` - Vector search endpoint
- [x] Implement `POST /api/chat` - RAG chat với LLM
- [x] Implement `GET /api/health` - Health check endpoint
- [x] Query validation và error handling

### 2.5 RAG Pipeline
- [x] Implement `app/pipelines/rag.py` - Full RAG pipeline
- [x] Score filtering (min_score threshold = 0.25)
- [x] Fallback logic khi không có kết quả đủ relevance

### 2.6 Frontend UI
- [x] React + Vite setup with HMR
- [x] Chat interface với message history
- [x] Source cards với expandable content
- [x] Loading states và error handling
- [x] Responsive dark theme design

---

## ✅ Phase 3: Hybrid Search & Reranking (Complete)

### 3.1 Hybrid Search Implementation
- [x] Implement BM25 sparse embedding với `fastembed`
- [x] Create hybrid collection với dense + sparse vectors
- [x] Implement `hybrid_indexer.py` cho cả hybrid indexing
- [x] Use Qdrant's native RRF (Reciprocal Rank Fusion)

**Hybrid Search Flow:**
```
Query → Dense Embedding (OpenAI 3072-dim)
      → Sparse Embedding (BM25/fastembed)
      → Prefetch both → RRF Fusion → Final Ranking
```

### 3.2 Reranker Implementation
| Model | Type | Status |
|-------|------|--------|
| **BAAI/bge-reranker-large** | Local (CPU) | ✅ Implemented |
| Cohere Rerank 3.5 | API | ❌ Skipped (cost) |

**Completed:**
- [x] Install torch CPU-only (~200MB) + FlagEmbedding
- [x] Implement `BGEReranker` class in `app/services/reranker.py`
- [x] Lazy loading để tiết kiệm memory (~2GB RAM khi load)
- [x] Integrate into RAGPipeline via `deps.py`
- [x] Configurable via `USE_RERANKER` env variable

### 3.3 Two-Stage Retrieval Results
```
Query → Hybrid Search (top 20-40) → BGE Rerank (CPU) → Final Results (top 5)
```

| Query | Before | After | Improvement |
|-------|--------|-------|-------------|
| Thời gian nghỉ giữa ca | 0.50 | 0.66 | +32% |
| Làm thêm giờ gấp đôi | 0.59 | 0.64 | +8% |
| Sa thải thử việc | 0.45 | 0.58 | +29% |

**Summary:** 60% queries improved với average +10-20% score gain.

---

## ✅ Phase 3.5: LangGraph Agent (Complete)

### Architecture
```
┌──────────┐    ┌──────────┐    ┌────────┐    ┌─────────┐    ┌──────────┐
│ Translate│ →  │ Retrieve │ →  │ Grade  │ →  │ Rerank  │ →  │ Generate │
└──────────┘    └──────────┘    └────────┘    └─────────┘    └──────────┘
                     ↑               │
                     └── Rewrite ────┘ (if docs weak)
```

### 3.5.1 Implementation Details

**New Files:**
| File | Purpose |
|------|---------|
| `app/agents/state.py` | TypedDict state definition |
| `app/agents/nodes.py` | 6 node functions + routing logic |
| `app/agents/graph.py` | StateGraph + LegalRAGAgent wrapper |

**Graph Nodes:**
1. **translate** - Vietnamese → Japanese translation + multi-query generation
2. **retrieve** - Multi-query vector search với deduplication
3. **grade** - LLM grades document relevance ("relevant"/"not_relevant")
4. **rerank** - BGE reranker for final ordering
5. **generate** - Answer generation with citations
6. **rewrite** - Query rewrite when < 2 relevant docs (max 2 retries)

### 3.5.2 Self-Correction Loop
```python
if relevant_docs < 2 and rewrite_count < 2:
    → Rewrite query with legal terminology
    → Re-retrieve documents
    → Re-grade
```

### 3.5.3 API Usage
```bash
# Default RAGPipeline (no grading, no self-correction)
curl -X POST /api/chat -d '{"query": "..."}'

# LangGraph Agent (with document grading + self-correction)
curl -X POST /api/chat -d '{"query": "...", "use_agent": true}'
```

---

## ✅ Phase 4: GraphRAG with Neo4j (Complete)

### 4.1 Graph Schema Design
```
(Law) -[:HAS_CHAPTER]-> (Chapter) -[:HAS_ARTICLE]-> (Article)
(Article) -[:HAS_PARAGRAPH]-> (Paragraph)
(Article) -[:REFERENCES]-> (Article)  // Cross-references
```

**Node Types:**
| Node | Properties |
|------|------------|
| Law | law_id, law_title, law_type, category, promulgation_date |
| Chapter | chapter_num, chapter_title |
| Article | article_num, article_title, article_caption |
| Paragraph | paragraph_num, text, chunk_id |

### 4.2 Graph Builder Implementation
- [x] Implement `scripts/graph_builder.py` - Build graph from JSON files
- [x] Implement `scripts/init_graph_schema.py` - Create indexes và constraints
- [x] Process 233 laws với hierarchical structure
- [x] Link Paragraph nodes với chunk_id từ vector store

**New Files:**
| File | Purpose |
|------|---------|
| `app/db/neo4j_client.py` | Neo4j client wrapper với connection pooling |
| `scripts/graph_builder.py` | Build graph từ processed JSON |
| `scripts/init_graph_schema.py` | Initialize schema, indexes, constraints |

### 4.3 Graph Service
- [x] Implement `app/services/graph_service.py` - GraphService class
- [x] `find_article(law_title, article_num)` - Direct article lookup
- [x] `find_related_articles(law_id, article_num, depth)` - Multi-hop traversal
- [x] `search_by_keyword(keyword)` - Fulltext search
- [x] `get_law_structure(law_id)` - Get hierarchical tree

### 4.4 Query Router
- [x] Implement `app/services/query_router.py` - QueryRouter class
- [x] Entity extraction (law names, article numbers)
- [x] Query type classification:
  - `SEMANTIC` - General question → Vector search only
  - `ENTITY_LOOKUP` - "第32条 là gì?" → Graph lookup
  - `MULTI_HOP` - "Điều liên quan đến..." → Graph traversal + Vector
  - `HYBRID` - Combination of both

**Entity Patterns:**
```python
# 労働基準法第32条 → (労働基準法, 32)
r'([ぁ-んァ-ン一-龯]+法)第(\d+)条'

# 第32条 (standalone)
r'第(\d+)条(?:の(\d+))?'

# Law names
r'([ぁ-んァ-ン一-龯]+法)'
```

### 4.5 GraphRAG Pipeline
- [x] Implement `app/pipelines/graph_rag.py` - GraphRAGPipeline class
- [x] Integrate QueryRouter for smart routing
- [x] Fusion strategy: Graph results + Vector results → Dedup → Rerank
- [x] Fallback to vector-only when graph has no results

**GraphRAG Flow:**
```
Query → Query Router → [Entities?]
        ├─ Yes → Graph Search (find_article, find_related)
        │        + Vector Search (semantic)
        │        → Fusion → Rerank → Generate
        └─ No  → Vector Search only → Generate
```

### 4.6 API Integration
- [x] Add `use_graph` parameter to `/api/chat` endpoint
- [x] Add graph stats to `/api/health` endpoint
- [x] Configurable via `USE_GRAPH_SEARCH` env variable

**📊 Graph Statistics:**
| Metric | Value |
|--------|-------|
| Total Law Nodes | 233 |
| Total Chapter Nodes | ~1,500 |
| Total Article Nodes | ~8,000 |
| Total Paragraph Nodes | ~15,600 |
| REFERENCES Relationships | (pending extraction) |

---

## ✅ Phase 4.5: Performance Optimization (Complete)

### 4.5.1 Latency Analysis
Initial performance breakdown (with all features enabled):
| Step | Duration |
|------|----------|
| Translation + Expansion | ~2s |
| Multi-Query (5x) Embedding | ~3s |
| Hybrid Search (5x) | ~5s |
| Reranking (BGE CPU) | ~20-40s |
| Generation | ~3s |
| **Total** | **60-80s** ❌ |

### 4.5.2 Optimization Steps

**Phase 1: Emergency Fix (Timeout Prevention)**
- [x] Disable Reranker by default (`USE_RERANKER=false`)
- [x] Enable Hybrid Search (`USE_HYBRID_SEARCH=true`)
- [x] Reduce Multi-Query count: 5 → 2 (`MULTI_QUERY_COUNT=2`)

**Phase 2: Query Optimization**
- [x] Merge translation + expansion into single LLM call
- [x] Batch embedding calls cho multi-query
- [x] Parallel search execution

### 4.5.3 Results After Optimization
| Optimization | Before | After | Impact |
|--------------|--------|-------|--------|
| Disable Reranker | 60s | 10s | -83% |
| Hybrid Search | 10s | 8s | -20% |
| Reduce Multi-Query (5→2) | 8s | 6s | -25% |
| Batch Embeddings | 6s | 5s | -17% |
| **Total** | **60s+** | **~5s** | **-90%** ✅ |

### 4.5.4 Current Configuration
```env
# Performance-optimized settings
USE_RERANKER=false          # Disable for speed
USE_HYBRID_SEARCH=true      # Better recall
MULTI_QUERY_COUNT=2         # Reduced parallel queries
MIN_SCORE_THRESHOLD=0.25    # Filter low-relevance docs
```

---

## ✅ Phase 5: Evaluation Framework (Complete)

### 5.1 RAGAS Integration
- [x] Implement `scripts/ragas_evaluation.py`
- [x] Create test dataset: `tests/data/ragas_test_questions.json`
- [x] Create ground truth: `tests/data/ragas_ground_truth.json`

### 5.2 Test Dataset
20+ câu hỏi về pháp luật tài chính Nhật Bản:
| Category | Sample Questions |
|----------|------------------|
| Thuế Thu Nhập | Thuế thu nhập cá nhân ở Nhật tính như thế nào? |
| Bảo Hiểm XH | Điều kiện hưởng lương hưu tại Nhật? |
| NISA | NISA là gì? Người nước ngoài có thể đăng ký không? |
| Lao Động | Thời gian làm việc tối đa mỗi tuần là bao nhiêu giờ? |

### 5.3 RAGAS Metrics
| Metric | Score | Description |
|--------|-------|-------------|
| **Context Precision** | 0.72 | Tỷ lệ relevant docs trong retrieved |
| **Context Recall** | 0.68 | Coverage of ground truth |
| **Faithfulness** | 0.85 | Answer grounded in context |
| **Answer Relevancy** | 0.78 | Answer addresses query |

### 5.4 Comparison Testing
- [x] Implement `scripts/compare_search_methods.py`
- [x] Vector-only vs Hybrid search comparison
- [x] With/without reranker comparison

---

## 📋 Phase 6: Production Deployment (Future)

### 6.1 Cloud Infrastructure
- [ ] Container packaging (Docker)
- [ ] Cloud deployment (AWS/GCP/Vercel)
- [ ] HTTPS với custom domain
- [ ] CDN cho frontend assets

### 6.2 API Hardening
- [ ] Rate limiting (requests/min per IP)
- [ ] API key authentication
- [ ] Request validation và sanitization
- [ ] Error logging và monitoring (Sentry)

### 6.3 Caching Layer
- [ ] Redis integration cho:
  - Query embedding cache
  - Translation cache
  - Popular query results cache
- [ ] Cache invalidation strategy

### 6.4 Observability
- [ ] Structured logging (JSON format)
- [ ] Metrics collection (Prometheus)
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Dashboard (Grafana)

---

## 📋 Phase 7: Enhanced Features (Future)

### 7.1 Conversation Memory
- [ ] Multi-turn chat support
- [ ] Context carry-over between turns
- [ ] Session management

### 7.2 User Feedback Loop
- [ ] Thumbs up/down on answers
- [ ] Feedback collection API
- [ ] Analytics dashboard

### 7.3 Extended Data Coverage
- [ ] More law categories (民法, 商法, etc.)
- [ ] Historical versions của luật
- [ ] Court case precedents (判例)

### 7.4 Mobile App
- [ ] React Native wrapper
- [ ] Push notifications
- [ ] Offline caching

---

## Tech Stack Summary

| Component | Technology | Version | Status |
|-----------|------------|---------|--------|
| **Data Pipeline** | Python scripts | 3.12 | ✅ Done |
| **Dense Embedding** | OpenAI text-embedding-3-large | 3072-dim | ✅ Done |
| **Sparse Embedding** | fastembed (BM25) | ≥0.3.0 | ✅ Done |
| **Vector DB** | Qdrant Cloud | Free Tier | ✅ Done |
| **Graph DB** | Neo4j | 5.x | ✅ Done |
| **LLM** | OpenAI GPT-4o-mini | - | ✅ Done |
| **Query Translation** | OpenAI (Vi → Ja) | - | ✅ Done |
| **Reranker** | BGE-reranker-large (CPU) | - | ✅ Done (optional) |
| **Agent Framework** | LangGraph | ≥0.2.0 | ✅ Done |
| **Backend** | FastAPI | ≥0.109.0 | ✅ Done |
| **Frontend** | React 18 + Vite | - | ✅ Done |
| **Evaluation** | RAGAS | 0.4.x | ✅ Done |
| **Cloud Deploy** | TBD | - | ⬜ Future |

---

## Project Structure

```
norman/
├── backend/
│   ├── app/
│   │   ├── agents/           # LangGraph Agent
│   │   │   ├── graph.py      # StateGraph definition
│   │   │   ├── nodes.py      # 6 agent nodes
│   │   │   └── state.py      # TypedDict state
│   │   ├── api/              # REST API
│   │   │   ├── routes.py     # /api/chat, /api/search
│   │   │   └── deps.py       # Dependency injection
│   │   ├── core/             # Configuration
│   │   │   ├── config.py     # Settings
│   │   │   └── protocols.py  # Abstract interfaces
│   │   ├── db/               # Database clients
│   │   │   ├── qdrant.py     # Qdrant vector store
│   │   │   └── neo4j_client.py  # Neo4j graph client
│   │   ├── llm/              # LLM modules
│   │   │   ├── base.py       # Abstract LLM
│   │   │   ├── openai_provider.py
│   │   │   ├── query_translator.py
│   │   │   └── prompts.py
│   │   ├── pipelines/        # RAG orchestration
│   │   │   ├── rag.py        # RAGPipeline class
│   │   │   └── graph_rag.py  # GraphRAGPipeline class
│   │   ├── services/         # Business logic
│   │   │   ├── embedding.py
│   │   │   ├── reranker.py
│   │   │   ├── sparse_embedding.py
│   │   │   ├── graph_service.py   # Neo4j queries
│   │   │   └── query_router.py    # Query routing
│   │   └── main.py
│   ├── scripts/              # Data pipeline
│   │   ├── downloader.py     # e-Gov API download
│   │   ├── xml_parser.py     # XML → JSON
│   │   ├── chunker.py        # Smart chunking
│   │   ├── embedder.py       # Batch embedding
│   │   ├── indexer.py        # Vector upload
│   │   ├── hybrid_indexer.py # Hybrid indexing
│   │   ├── graph_builder.py  # Build Neo4j graph
│   │   ├── init_graph_schema.py  # Graph schema
│   │   ├── ragas_evaluation.py   # RAGAS evaluation
│   │   └── compare_search_methods.py
│   └── tests/
│       └── data/
│           ├── ragas_test_questions.json
│           └── ragas_ground_truth.json
├── data/                     # Data storage
│   ├── raw/                  # Original XML files
│   ├── processed/            # Parsed JSON files
│   ├── chunks/               # Chunked data
│   └── embeddings/           # Cached vectors
├── frontend/                 # React UI
│   ├── src/
│   │   ├── components/
│   │   └── App.tsx
│   └── package.json
└── docs/
    ├── REPORT.md            # Main documentation
    └── ROADMAP.md           # This file
```

---

## Timeline & Progress

| Phase | Description | Duration | Status |
|-------|-------------|----------|--------|
| Phase 0 | Data Collection & Processing | 1 day | ✅ Complete |
| Phase 1 | Vector Search with Qdrant | 2 days | ✅ Complete |
| Phase 2 | Response Generation with API | 3-5 days | ✅ Complete |
| Phase 3 | Hybrid Search & Reranking | 2-3 days | ✅ Complete |
| Phase 3.5 | LangGraph Agent | 1 day | ✅ Complete |
| Phase 4 | GraphRAG with Neo4j | 3-4 days | ✅ Complete |
| Phase 4.5 | Performance Optimization | 1 day | ✅ Complete |
| Phase 5 | Evaluation Framework | 1 day | ✅ Complete |
| Phase 6 | Production Deployment | 5-7 days | ⬜ Future |
| Phase 7 | Enhanced Features | Ongoing | ⬜ Future |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/norman.git
cd norman

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with API keys

# Run backend
uvicorn app.main:app --reload --port 8000

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

## Environment Variables

```env
# Required - OpenAI
OPENAI_API_KEY=sk-...

# Required - Qdrant Cloud
QDRANT_URL=https://xxx.qdrant.tech
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=japanese_laws_hybrid

# Optional - Neo4j (for GraphRAG)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# Performance Settings
USE_RERANKER=false
USE_HYBRID_SEARCH=true
USE_GRAPH_SEARCH=true
MULTI_QUERY_COUNT=2
MIN_SCORE_THRESHOLD=0.25
```

---

**Norman - Japanese Financial Law RAG System**  
Version 1.1.0 | January 2026
