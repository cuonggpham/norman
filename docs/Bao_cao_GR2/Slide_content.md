# NỘI DUNG SLIDE BÁO CÁO ĐỒ ÁN GR2

## Hệ thống RAG Tư vấn Pháp luật Tài chính Nhật Bản cho Người Việt Nam

---

# PHẦN 1: GIỚI THIỆU

---

## SLIDE 1: Trang bìa

**Tiêu đề:** Hệ thống RAG Tư vấn Pháp luật Tài chính Nhật Bản cho Người Việt Nam

**Thông tin:**
- Sinh viên: Phạm Quốc Cường
- Mã số: [Mã SV]
- Giảng viên hướng dẫn: [Tên GVHD]
- Trường: Đại học Bách Khoa Hà Nội
- Thời gian: 2024-2025

---

## SLIDE 2: Đặt vấn đề

**Bối cảnh:**
- 500,000+ người Việt tại Nhật Bản (2024)
- Nhu cầu tìm hiểu pháp luật tài chính cao

**Thách thức:**
- Rào cản ngôn ngữ (tiếng Nhật pháp lý phức tạp)
- Thuật ngữ chuyên ngành: 確定申告, 源泉徴収, 厚生年金
- Thiếu công cụ tra cứu đa ngôn ngữ

**Câu hỏi nghiên cứu:**
> Xây dựng hệ thống cho phép hỏi tiếng Việt → tìm kiếm văn bản tiếng Nhật → trả lời kèm trích dẫn nguồn?

---

## SLIDE 3: Lựa chọn phương pháp

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| **Fine-tuning LLM** | Không cần retrieval | Chi phí cao, kiến thức "đóng băng" |
| **BM25 (từ khóa)** | Đơn giản, nhanh | Không hiểu ngữ nghĩa |
| **RAG ✅** | Cập nhật dễ, có trích dẫn | Phức tạp hơn |

**Lựa chọn:** RAG (Retrieval-Augmented Generation)
- ✅ Trích dẫn nguồn chính xác
- ✅ Cập nhật dữ liệu linh hoạt
- ✅ Hỗ trợ cross-lingual (Việt → Nhật)

---

## SLIDE 4: Mục tiêu và Phạm vi

**Mục tiêu:**
- Xây dựng chatbot tư vấn pháp luật Việt-Nhật
- Trả lời bằng tiếng Việt kèm trích dẫn nguồn

**Phạm vi dữ liệu:**

| Lĩnh vực | Văn bản pháp luật |
|----------|-------------------|
| Thuế | 所得税法, 地方税法 |
| Bảo hiểm xã hội | 健康保険法, 厚生年金保険法 |
| Đầu tư | NISA, iDeCo |
| Lao động | 労働基準法, 労働契約法 |

**Kết quả:** 431 văn bản luật • 206,014 chunks

---

# PHẦN 2: TỔNG QUAN HỆ THỐNG

---

## SLIDE 5: Kiến trúc tổng quan - Luồng hoạt động

```mermaid
flowchart TB
    subgraph USER["👤 NGƯỜI DÙNG"]
        Q[/"Câu hỏi tiếng Việt"/]
        A[/"Câu trả lời + Trích dẫn"/]
    end

    subgraph FRONTEND["🖥️ FRONTEND (Next.js)"]
        UI[Chat Interface]
    end

    subgraph API["⚡ API LAYER (FastAPI)"]
        EP["/api/chat<br/>/api/search<br/>/api/translate"]
    end

    subgraph AGENT["🤖 LANGGRAPH AGENT"]
        direction TB
        TR[Translate Node]
        RT[Retrieve Node]
        GR[Grade Documents]
        RR[Rerank Node]
        GN[Generate Node]
        RW[Rewrite Query]
        
        TR --> RT
        RT --> GR
        GR -->|"≥2 relevant"| RR
        GR -->|"<2 relevant"| RW
        RW -->|"retry ≤ 2"| RT
        RR --> GN
    end

    subgraph SERVICES["🔧 SERVICE LAYER"]
        direction LR
        EMB[Embedding<br/>Service]
        RANK[Reranker<br/>Service]
        TRANS[Translation<br/>Service]
        GRAPH[Graph<br/>Service]
    end

    subgraph SEARCH["🔍 HYBRID SEARCH"]
        direction TB
        DENSE[Dense Search<br/>Semantic]
        SPARSE[Sparse Search<br/>BM25]
        RRF[RRF Fusion]
        
        DENSE --> RRF
        SPARSE --> RRF
    end

    subgraph STORAGE["💾 DATA STORAGE"]
        direction LR
        QDRANT[(Qdrant Cloud<br/>Vector DB<br/>206K chunks)]
        NEO4J[(Neo4j Aura<br/>Graph DB<br/>50K nodes)]
    end

    subgraph LLM["🧠 LLM APIs"]
        GPT[GPT-4o-mini<br/>Translation<br/>Generation]
        OPENAI[OpenAI Embedding<br/>text-embedding-3-large]
    end

    Q --> UI
    UI --> EP
    EP --> AGENT
    
    TR -.->|"Dịch query"| TRANS
    TRANS -.-> GPT
    
    RT -.->|"Embed query"| EMB
    EMB -.-> OPENAI
    RT -.-> SEARCH
    
    SEARCH --> QDRANT
    RT -.->|"Graph lookup"| GRAPH
    GRAPH --> NEO4J
    
    RR -.-> RANK
    GN -.-> GPT
    
    AGENT --> EP
    EP --> UI
    UI --> A
```

---

## SLIDE 6: Kiến trúc chi tiết - Online vs Offline Pipeline

```mermaid
flowchart LR
    subgraph OFFLINE["📥 OFFLINE PIPELINE (Data Ingestion)"]
        direction TB
        EGOV[e-Gov API<br/>Laws Portal] --> XML[XML Parser<br/>lxml]
        XML --> CHUNK[Chunking<br/>Paragraph-level]
        CHUNK --> EMBED_OFF[Embedding<br/>Dense + Sparse]
        EMBED_OFF --> INDEX[Indexing<br/>Qdrant + Neo4j]
    end

    subgraph ONLINE["📤 ONLINE PIPELINE (Query Processing)"]
        direction TB
        QUERY[Query<br/>Tiếng Việt] --> TRANSLATE[Translation<br/>+ Expansion]
        TRANSLATE --> EMBED_ON[Query<br/>Embedding]
        EMBED_ON --> HYBRID[Hybrid<br/>Search]
        HYBRID --> RERANK[Cross-Encoder<br/>Reranking]
        RERANK --> GENERATE[LLM<br/>Generation]
        GENERATE --> RESPONSE[Response<br/>+ Citations]
    end

    INDEX -.->|"Vectors + Graph"| HYBRID
```

---

## SLIDE 7: Các thành phần chính

| Thành phần | Công nghệ | Chức năng |
|------------|-----------|-----------|
| **Frontend** | Next.js 14 | Chat UI, hiển thị nguồn |
| **Backend** | FastAPI | REST API, orchestration |
| **Vector DB** | Qdrant Cloud | Hybrid search |
| **Graph DB** | Neo4j Aura | Entity lookup, traversal |
| **LLM** | GPT-4o-mini | Translation, generation |
| **Embedding** | text-embedding-3-large | Multilingual vectors |
| **Reranker** | mMarco-mMiniLM | Cross-encoder ranking |
| **Agent** | LangGraph | Self-correction loop |

---

# PHẦN 3: XỬ LÝ DỮ LIỆU

---

## SLIDE 8: Thu thập dữ liệu từ e-Gov API

```mermaid
flowchart LR
    subgraph SOURCES["Nguồn dữ liệu"]
        CAT[Category Search<br/>国税, 労働, 社会保険]
        KEY[Keyword Search<br/>外国人, 所得税, 年金]
    end

    subgraph FILTER["Bộ lọc"]
        ERA[Era Filter<br/>Từ Showa trở đi]
        TYPE[Law Type<br/>Act, Cabinet Order]
        STATUS[Status Filter<br/>CurrentEnforced]
    end

    subgraph OUTPUT["Kết quả"]
        XML[(431 XML Files<br/>~80MB)]
    end

    CAT --> ERA
    KEY --> ERA
    ERA --> TYPE --> STATUS --> XML
```

**Rate Limiting:** 1.2s/request • Retry với exponential backoff

---

## SLIDE 9: XML Parsing và Chunking

**Cấu trúc văn bản luật Nhật:**
```
Law → Part (編) → Chapter (章) → Section (節) 
    → Article (条) → Paragraph (項) → Item (号)
```

**Chiến lược Chunking: Paragraph-level**

```mermaid
flowchart LR
    LAW[Văn bản luật] --> ART1[Điều 32]
    LAW --> ART2[Điều 33]
    ART1 --> P1[Khoản 1 → Chunk 1]
    ART1 --> P2[Khoản 2 → Chunk 2]
    ART2 --> P3[Khoản 1 → Chunk 3]
```

**Context Enrichment:**
```
text: "使用者は、労働者に..."
text_with_context: "労働基準法 第三十二条 (労働時間) 使用者は..."
```

---

## SLIDE 10: Embedding và Indexing

```mermaid
flowchart TB
    subgraph INPUT["206,014 Chunks"]
        CHUNKS[Chunks with<br/>context enrichment]
    end

    subgraph EMBEDDING["Dual Embedding"]
        DENSE[Dense Embedding<br/>OpenAI 3072-dim]
        SPARSE[Sparse Embedding<br/>BM25 fastembed]
    end

    subgraph INDEX["Indexing"]
        QDRANT[(Qdrant Cloud<br/>Hybrid Collection)]
        NEO4J[(Neo4j Aura<br/>Graph Structure)]
    end

    CHUNKS --> DENSE --> QDRANT
    CHUNKS --> SPARSE --> QDRANT
    CHUNKS -->|"Structure"| NEO4J
```

| Loại | Model | Đặc điểm |
|------|-------|----------|
| Dense | text-embedding-3-large | 3072-dim, multilingual |
| Sparse | Qdrant/bm25 | IDF weighting, exact match |

---

## SLIDE 11: Knowledge Graph Schema

```mermaid
graph TB
    LAW[("🏛️ LAW<br/>law_id, law_title")]
    CHAP[("📖 CHAPTER<br/>chapter_num, title")]
    ART[("📄 ARTICLE<br/>article_num, title")]
    PARA[("📝 PARAGRAPH<br/>paragraph_num, text<br/>chunk_id")]

    LAW -->|HAS_CHAPTER| CHAP
    CHAP -->|HAS_ARTICLE| ART
    ART -->|HAS_PARAGRAPH| PARA
    ART -->|REFERENCES| ART

    subgraph LINK["Link to Vector Store"]
        PARA -.->|chunk_id| QDRANT[(Qdrant)]
    end
```

**chunk_id** liên kết Graph ↔ Vector Store

---

# PHẦN 4: HỆ THỐNG TRUY VẤN

---

## SLIDE 12: Query Translation & Expansion

```mermaid
flowchart LR
    subgraph INPUT
        VN["Query tiếng Việt<br/>'Thời gian làm việc tối đa?'"]
    end

    subgraph GPT["GPT-4o-mini"]
        TRANS["Translation"]
        EXPAND["Query Expansion"]
    end

    subgraph OUTPUT
        JP["週の最大労働時間は?"]
        KW["Keywords:<br/>法定労働時間, 週40時間"]
        QUERIES["Search Queries:<br/>1. 法定労働時間とは<br/>2. 週の労働時間制限"]
    end

    VN --> GPT
    GPT --> JP
    GPT --> KW
    GPT --> QUERIES
```

**Multi-Query:** 1 query gốc → 3-5 search queries

---

## SLIDE 13: Hybrid Search với RRF Fusion

```mermaid
flowchart TB
    subgraph QUERY["Search Queries"]
        Q1["Query 1"]
        Q2["Query 2"]
    end

    subgraph DENSE["Dense Search"]
        D1["Semantic matching<br/>Top-20"]
    end

    subgraph SPARSE["Sparse Search"]
        S1["Keyword matching<br/>Top-20"]
    end

    subgraph FUSION["RRF Fusion"]
        RRF["score = Σ 1/(k + rank)"]
    end

    subgraph RESULT["Merged Results"]
        TOP["Top-K unique<br/>documents"]
    end

    Q1 --> D1 & S1
    Q2 --> D1 & S1
    D1 --> RRF
    S1 --> RRF
    RRF --> TOP
```

| Search | Ưu điểm | Nhược điểm |
|--------|---------|------------|
| Dense | Hiểu ngữ nghĩa | Miss exact keywords |
| Sparse | Khớp từ chính xác | Không hiểu synonym |
| **Hybrid** | **Kết hợp cả hai** | ✅ |

---

## SLIDE 14: Two-Stage Retrieval với Reranking

```mermaid
flowchart LR
    subgraph STAGE1["Stage 1: Recall"]
        BI["Bi-Encoder<br/>~1ms/doc"]
        TOP20["Top-20<br/>candidates"]
    end

    subgraph STAGE2["Stage 2: Precision"]
        CROSS["Cross-Encoder<br/>mMarco-mMiniLM"]
        TOP5["Top-5<br/>reranked"]
    end

    BI --> TOP20 --> CROSS --> TOP5
```

**Cross-Encoder improvement:**

| Query Type | Without | With | Δ |
|------------|---------|------|---|
| Semantic matching | 0.65 | 0.81 | **+16%** |
| Cross-lingual | 0.52 | 0.84 | **+32%** |
| Multi-concept | 0.58 | 0.79 | **+21%** |

---

# PHẦN 5: SINH CÂU TRẢ LỜI

---

## SLIDE 15: LangGraph Agent - State Machine

```mermaid
stateDiagram-v2
    [*] --> Translate
    Translate --> Retrieve
    Retrieve --> Grade
    
    Grade --> Rerank: ≥2 relevant OR retry=2
    Grade --> Rewrite: <2 relevant AND retry<2
    
    Rewrite --> Retrieve: retry++
    
    Rerank --> Generate
    Generate --> [*]
```

**Self-correction loop:** Nếu retrieval kém → Rewrite query (max 2 lần)

---

## SLIDE 16: LangGraph Nodes

```mermaid
flowchart TB
    subgraph NODES["Agent Nodes"]
        TR["🌐 Translate<br/>Dịch + Mở rộng query"]
        RT["🔍 Retrieve<br/>Hybrid search"]
        GR["✅ Grade<br/>Đánh giá relevance"]
        RR["📊 Rerank<br/>Cross-encoder scoring"]
        GN["💬 Generate<br/>LLM tạo câu trả lời"]
        RW["✏️ Rewrite<br/>Viết lại query"]
    end

    TR -->|"translated_query<br/>search_queries"| RT
    RT -->|"documents"| GR
    GR -->|"document_grades"| RR
    GR -->|"< 2 relevant"| RW
    RW -->|"new queries"| RT
    RR -->|"reranked_docs"| GN
    GN -->|"answer + sources"| OUT[Response]
```

---

## SLIDE 17: LLM Generation với Citation

**Prompt Engineering:**
```
Bạn là chuyên gia tư vấn pháp luật Nhật Bản.
- Trả lời bằng tiếng Việt
- Giữ thuật ngữ Nhật trong ngoặc []
- Trích dẫn nguồn bằng [1], [2]...

Nguồn:
[1] 労働基準法 第三十二条: "使用者は..."
```

**Output:**
> Theo Luật Tiêu chuẩn Lao động [労働基準法], thời gian làm việc tối đa là **40 giờ/tuần** [1].
>
> **Nguồn:** Điều 32 [第三十二条]

---

# PHẦN 6: ĐÁNH GIÁ

---

## SLIDE 18: RAGAS Evaluation Framework

**RAGAS Metrics:**

| Metric | Đo lường | Score |
|--------|----------|-------|
| Context Precision | % retrieved docs relevant | 0.72 |
| Context Recall | % ground truth covered | 0.68 |
| **Faithfulness** | **% answer grounded** | **0.85** ✅ |
| Answer Relevancy | % answer addresses query | 0.78 |

**Test Dataset:** 50 samples • 5 lĩnh vực • 3 mức độ khó

---

## SLIDE 19: So sánh Configurations

```mermaid
xychart-beta
    title "Faithfulness theo Configuration"
    x-axis ["Vector only", "Hybrid", "+Rerank", "+Agent"]
    y-axis "Faithfulness" 0.6 --> 0.9
    bar [0.72, 0.78, 0.82, 0.85]
```

| Configuration | Faithfulness | Latency |
|---------------|--------------|---------|
| Vector only | 0.72 | 2.8s |
| Hybrid search | 0.78 (+8%) | 3.5s |
| + Reranking | 0.82 (+5%) | 7.2s |
| **+ Agent** | **0.85 (+4%)** | **9.5s** |

---

## SLIDE 20: Latency Breakdown

```mermaid
pie title Phân bổ thời gian xử lý (~10s)
    "Translation & Expansion" : 20
    "Query Embedding" : 10
    "Hybrid Search" : 30
    "Reranking" : 10
    "Generate Answer" : 30
```

| Bước | Thời gian | Tỷ lệ |
|------|-----------|-------|
| Translation | 2s | 20% |
| Embedding | 1s | 10% |
| Hybrid Search | 3s | 30% |
| Reranking | 1s | 10% |
| Generation | 3s | 30% |

---

# PHẦN 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

---

## SLIDE 21: Những gì đã đạt được

**Kết quả:**
- ✅ End-to-end RAG pipeline hoàn chỉnh
- ✅ 431 văn bản luật, 206,014 chunks
- ✅ Cross-lingual retrieval (Việt → Nhật)
- ✅ Hybrid search + Reranking + LangGraph
- ✅ **Faithfulness 0.85** trên RAGAS

**Bài học kỹ thuật:**
1. **Data quality > Model size**
2. **Hybrid approach > Single method**
3. **Right model selection** quan trọng

---

## SLIDE 22: Hạn chế và Hướng phát triển

**Hạn chế:**
- Latency: 8-10s (chậm hơn ChatGPT)
- Coverage: 431 luật, còn thiếu nhiều lĩnh vực
- Không có conversation memory

**Hướng phát triển:**

| Hướng | Cải tiến |
|-------|----------|
| Retrieval | Adaptive chunking, fine-tuned embedding |
| RAG Architecture | CRAG, Self-RAG, Agentic RAG |
| GraphRAG | NER enhancement, multi-hop reasoning |
| Optimization | Redis caching, speculative retrieval |

---

## SLIDE 23: Tổng kết

> **Mục tiêu đã hoàn thành:** Xây dựng công cụ giúp người Việt Nam tại Nhật Bản tiếp cận thông tin pháp luật dễ dàng hơn, bằng tiếng Việt, với trích dẫn nguồn chính xác.

**Đóng góp chính:**
- Pipeline thu thập & xử lý 431 văn bản luật từ e-Gov API
- Hybrid search với RRF fusion
- Two-stage retrieval với cross-encoder
- LangGraph agent với self-correction loop
- Đạt **Faithfulness 0.85**

---

## SLIDE 24: Q&A

**Cảm ơn thầy/cô và các bạn đã lắng nghe!**

**Câu hỏi?**

---

# PHỤ LỤC

## Danh sách hình cần chèn

| Slide | Nội dung hình |
|-------|---------------|
| 5 | Kiến trúc tổng quan - Luồng hoạt động (Mermaid) |
| 6 | Online vs Offline Pipeline (Mermaid) |
| 8 | Thu thập dữ liệu flowchart (Mermaid) |
| 9 | Chunking strategy (Mermaid) |
| 10 | Embedding & Indexing (Mermaid) |
| 11 | Knowledge Graph Schema (Mermaid) |
| 12 | Query Translation flow (Mermaid) |
| 13 | Hybrid Search với RRF (Mermaid) |
| 14 | Two-Stage Retrieval (Mermaid) |
| 15 | LangGraph State Machine (Mermaid) |
| 16 | Agent Nodes flowchart (Mermaid) |
| 19 | Faithfulness comparison chart |
| 20 | Latency pie chart (Mermaid) |
