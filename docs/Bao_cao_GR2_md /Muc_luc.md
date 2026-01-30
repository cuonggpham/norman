# MỤC LỤC

## CHƯƠNG 1: TỔNG QUAN

### 1.1. Đặt vấn đề và bối cảnh thực tiễn
### 1.2. Mục tiêu nghiên cứu
### 1.3. Phạm vi và giới hạn đề tài
### 1.4. Đối tượng và phương pháp nghiên cứu

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

### 2.1. Retrieval-Augmented Generation (RAG)
- 2.1.1. Kiến trúc RAG cơ bản
- 2.1.2. So sánh RAG với Fine-tuning và Prompt Engineering
- 2.1.3. Các thách thức trong RAG

### 2.2. Vector Embedding và Semantic Search
- 2.2.1. Khái niệm vector embedding
- 2.2.2. Các mô hình embedding phổ biến
- 2.2.3. Vector similarity metrics

### 2.3. Hybrid Search (Dense + Sparse)
- 2.3.1. Dense retrieval vs Sparse retrieval (BM25)
- 2.3.2. Reciprocal Rank Fusion (RRF)
- 2.3.3. Ưu điểm của Hybrid Search

### 2.4. Reranking (Two-Stage Retrieval)
- 2.4.1. Bi-Encoder vs Cross-Encoder
- 2.4.2. Các mô hình reranker phổ biến

### 2.5. Knowledge Graph và Graph RAG ⭐ MỚI
- 2.5.1. Khái niệm Knowledge Graph
- 2.5.2. Graph Database và Neo4j
- 2.5.3. Graph RAG: Kết hợp Knowledge Graph với RAG
- 2.5.4. So sánh Vector RAG vs Graph RAG

### 2.6. AI Agent với LangGraph
- 2.6.1. Agent và State Machine
- 2.6.2. Self-Correction trong RAG

## CHƯƠNG 3: PHÂN TÍCH YÊU CẦU VÀ THIẾT KẾ HỆ THỐNG

### 3.1. Phân tích yêu cầu
- 3.1.1. Yêu cầu chức năng
- 3.1.2. Yêu cầu phi chức năng
- 3.1.3. Đặc thù bài toán xuyên ngôn ngữ (Việt-Nhật)

### 3.2. Kiến trúc tổng quan hệ thống
- 3.2.1. Sơ đồ kiến trúc multi-layer
- 3.2.2. Luồng xử lý dữ liệu (Data Flow)

### 3.3. Thiết kế Data Pipeline
- 3.3.1. Thu thập dữ liệu từ e-Gov API
- 3.3.2. XML Parser cho văn bản pháp luật
- 3.3.3. Chiến lược Chunking
- 3.3.4. Embedding Pipeline
- 3.3.5. Hybrid Indexing (Qdrant)

### 3.4. Thiết kế Knowledge Graph ⭐ MỚI
- 3.4.1. Mô hình dữ liệu Graph (Node Types: Law, Chapter, Article, Paragraph)
- 3.4.2. Các loại quan hệ (HAS_CHAPTER, HAS_ARTICLE, HAS_PARAGRAPH, REFERENCES)
- 3.4.3. Graph Schema Design
- 3.4.4. Connection với Vector Store (chunk_id linking)

### 3.5. Thiết kế RAG Pipeline
- 3.5.1. Query Translation & Multi-Query Expansion
- 3.5.2. Hybrid Retrieval (Vector Search)
- 3.5.3. Reranking
- 3.5.4. LLM Generation với citation

### 3.6. Thiết kế GraphRAG Pipeline ⭐ MỚI
- 3.6.1. Query Routing (Khi nào dùng Graph vs Vector)
- 3.6.2. Entity Extraction từ query
- 3.6.3. Graph Search (find_article, find_related_articles)
- 3.6.4. Result Fusion (Kết hợp Graph + Vector results)

### 3.7. Thiết kế LangGraph Agent
- 3.7.1. State Graph Definition
- 3.7.2. Các Node (Translate, Retrieve, Grade, Rerank, Generate, Rewrite)
- 3.7.3. Conditional Edge và Self-Correction Loop

### 3.8. Thiết kế API (REST Endpoints)

## CHƯƠNG 4: TRIỂN KHAI HỆ THỐNG

### 4.1. Công nghệ sử dụng
| Thành phần | Công nghệ |
| :--- | :--- |
| Backend | FastAPI (Python) |
| Vector Database | Qdrant Cloud |
| Graph Database | Neo4j |
| LLM | OpenAI GPT-4.1-mini |
| Embedding | text-embedding-3-large |
| Reranker | BGE Reranker |
| Agent | LangGraph |

### 4.2. Cấu trúc mã nguồn
### 4.3. Triển khai Data Pipeline
- 4.3.1. Downloader Module
- 4.3.2. XML Parser Module
- 4.3.3. Chunker Module
- 4.3.4. Embedder Module
- 4.3.5. Hybrid Indexer Module

### 4.4. Triển khai Knowledge Graph ⭐ MỚI
- 4.4.1. Neo4j Client Setup (app/db/neo4j.py)
- 4.4.2. Graph Schema Initialization (scripts/init_graph_schema.py)
- 4.4.3. Graph Builder (scripts/graph_builder.py)
- 4.4.4. Graph Service (app/services/graph_service.py)

### 4.5. Triển khai RAG Pipeline
- 4.5.1. Query Translation Service
- 4.5.2. Embedding Service
- 4.5.3. Hybrid Search Service
- 4.5.4. Reranker Service
- 4.5.5. LLM Provider

### 4.6. Triển khai GraphRAG Pipeline ⭐ MỚI
- 4.6.1. Query Router (app/services/query_router.py)
- 4.6.2. Entity Extraction
- 4.6.3. Graph Search Methods
- 4.6.4. Result Fusion Algorithm

### 4.7. Triển khai LangGraph Agent
### 4.8. Các kỹ thuật tối ưu
- 4.8.1. Caching (Query Cache)
- 4.8.2. Batch Processing
- 4.8.3. Lazy Loading

## CHƯƠNG 5: THỬ NGHIỆM VÀ ĐÁNH GIÁ

### 5.1. Phương pháp đánh giá RAG (RAGAS Framework)
- 5.1.1. Faithfulness
- 5.1.2. Answer Relevancy
- 5.1.3. Context Precision
- 5.1.4. Context Recall

### 5.2. Thử nghiệm thành phần
- 5.2.1. Đánh giá Hybrid Search vs Dense Search
- 5.2.2. So sánh các Reranker Models
- 5.2.3. Đánh giá Query Translation
- 5.2.4. Đánh giá Graph RAG vs Vector RAG ⭐ MỚI

### 5.3. Đánh giá tổng thể hệ thống
- 5.3.1. Dataset đánh giá (ground truth)
- 5.3.2. Kết quả RAGAS Metrics
- 5.3.3. Phân tích từng loại câu hỏi

### 5.4. Đánh giá hiệu năng
- 5.4.1. Thời gian response (Latency)
- 5.4.2. Graph Query Performance
- 5.4.3. So sánh trước/sau tối ưu

### 5.5. Hạn chế và vấn đề còn tồn tại

## CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết quả đạt được
### 6.2. Đóng góp của đồ án
### 6.3. Hạn chế
### 6.4. Hướng phát triển
- 6.4.1. Mở rộng nguồn dữ liệu
- 6.4.2. Cải thiện Knowledge Graph (thêm REFERENCES relationships)
- 6.4.3. Fine-tuning model riêng
- 6.4.4. Deploy production

## TÀI LIỆU THAM KHẢO

## PHỤ LỤC
