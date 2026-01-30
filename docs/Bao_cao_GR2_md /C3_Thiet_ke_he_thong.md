# CHƯƠNG 3: PHÂN TÍCH YÊU CẦU VÀ THIẾT KẾ HỆ THỐNG

Chương 2 đã trình bày các cơ sở lý thuyết về RAG, text embedding, hybrid search, và knowledge graph. Chương này sẽ phân tích yêu cầu bài toán và mô tả thiết kế chi tiết của hệ thống, bao gồm: (i) phân tích yêu cầu trong phần 3.1, (ii) kiến trúc tổng quan trong phần 3.2, (iii) thiết kế Data Pipeline trong phần 3.3, (iv) thiết kế Knowledge Graph trong phần 3.4, (v) thiết kế RAG Pipeline trong phần 3.5, (vi) thiết kế GraphRAG Pipeline trong phần 3.6, (vii) thiết kế LangGraph Agent trong phần 3.7, và (viii) thiết kế API trong phần 3.8.

---

## 3.1. Phân tích yêu cầu

### 3.1.1. Yêu cầu chức năng

Hệ thống Norman - Chatbot tư vấn pháp luật Nhật Bản cho người dùng Việt Nam cần đáp ứng các yêu cầu chức năng sau:

| STT | Yêu cầu | Mô tả |
|-----|---------|-------|
| F1 | Truy vấn pháp luật bằng tiếng Việt | Người dùng đặt câu hỏi bằng tiếng Việt, hệ thống tự động dịch sang tiếng Nhật để tìm kiếm |
| F2 | Tìm kiếm văn bản pháp luật | Tìm kiếm trong cơ sở dữ liệu luật Nhật Bản theo nội dung semantic |
| F3 | Trả lời câu hỏi với trích dẫn | Sinh câu trả lời bằng tiếng Việt kèm trích dẫn nguồn (điều, khoản cụ thể) |
| F4 | Tra cứu điều luật cụ thể | Tra cứu trực tiếp điều luật theo số (ví dụ: "Điều 32 Luật Tiêu chuẩn Lao động") |
| F5 | Tìm điều luật liên quan | Tìm các điều luật có quan hệ tham chiếu với điều luật đang xem |
| F6 | Dịch văn bản pháp luật | Dịch đoạn văn bản luật tiếng Nhật sang tiếng Việt với thuật ngữ chính xác |

### 3.1.2. Yêu cầu phi chức năng

| STT | Yêu cầu | Mô tả | Mức độ ưu tiên |
|-----|---------|-------|----------------|
| NF1 | Thời gian phản hồi | Trả lời trong vòng 3-5 giây | Cao |
| NF2 | Độ chính xác | Trích dẫn đúng nguồn, không "hallucination" | Cao |
| NF3 | Khả năng mở rộng | Dễ dàng thêm luật mới từ e-Gov API | Trung bình |
| NF4 | Tính sẵn sàng | Hệ thống hoạt động 24/7 trên cloud | Trung bình |
| NF5 | Giao diện thân thiện | UI đơn giản, dễ sử dụng cho người không chuyên | Trung bình |
| NF6 | Bảo mật | Không lưu trữ thông tin cá nhân người dùng | Thấp |

### 3.1.3. Đặc thù bài toán xuyên ngôn ngữ (Việt-Nhật)

Bài toán tư vấn pháp luật Nhật Bản cho người Việt có các đặc thù sau:

**Thách thức ngôn ngữ:**
- **Khoảng cách ngữ nghĩa**: Tiếng Việt và tiếng Nhật thuộc hai ngữ hệ khác nhau, embedding model cần hỗ trợ multilingual
- **Thuật ngữ pháp lý**: Cần bảo toàn thuật ngữ gốc tiếng Nhật (ví dụ: 労働基準法 - Luật Tiêu chuẩn Lao động)
- **Cấu trúc văn bản**: Văn bản luật Nhật có cấu trúc riêng (編/章/節/条/項/号)

**Giải pháp thiết kế:**

| Thách thức | Giải pháp |
|------------|-----------|
| Query tiếng Việt | Query Translation: GPT-4o-mini dịch query → tiếng Nhật trước khi tìm kiếm |
| Embedding xuyên ngữ | OpenAI text-embedding-3-large hỗ trợ multilingual |
| Thuật ngữ pháp lý | Prompt engineering: giữ nguyên thuật ngữ Nhật trong ngoặc [] |
| Citation format | Mapping paragraph-level chunking với cấu trúc pháp lý |

---

## 3.2. Kiến trúc tổng quan hệ thống

### 3.2.1. Sơ đồ kiến trúc multi-layer

Hệ thống được thiết kế theo kiến trúc **RAG Pipeline** với nhiều layer:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Next.js Frontend                          │    │
│  │         (Chat UI, Search Results, Source Display)            │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          API LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 FastAPI REST Endpoints                        │    │
│  │     /api/search    /api/chat    /api/translate    /api/health│    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   RAGPipeline    │  │ GraphRAGPipeline │  │ LangGraph Agent  │   │
│  │ (Vector Search)  │  │ (Hybrid Search)  │  │(Self-Correction) │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SERVICE LAYER                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐    │
│  │ Embedding  │ │  Reranker  │ │Query Router│ │ Graph Service  │    │
│  │  Service   │ │  Service   │ │  Service   │ │    Service     │    │
│  └────────────┘ └────────────┘ └────────────┘ └────────────────┘    │
│  ┌────────────┐ ┌────────────┐                                      │
│  │   Query    │ │   Sparse   │                                      │
│  │ Translator │ │ Embedding  │                                      │
│  └────────────┘ └────────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE LAYER                              │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐     │
│  │     Qdrant Cloud        │    │        Neo4j Aura           │     │
│  │   (Vector Database)     │    │    (Graph Database)         │     │
│  │  Dense + Sparse Vectors │    │  Law → Article → Paragraph  │     │
│  └─────────────────────────┘    └─────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐    │
│  │  e-Gov API │ │ XML Parser │ │  Chunking  │ │   Embedding    │    │
│  │  Fetcher   │ │            │ │  Strategy  │ │   Pipeline     │    │
│  └────────────┘ └────────────┘ └────────────┘ └────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2.2. Luồng xử lý dữ liệu (Data Flow)

Hệ thống có hai luồng xử lý chính:

**Offline Pipeline (Data Ingestion)**:
```
e-Gov API → XML Parser → Chunking → Embedding → Qdrant + Neo4j
```

**Online Pipeline (Query Processing)**:
```
User Query → Translation → Hybrid Search → Reranking → LLM Generation → Response
```

| Pipeline | Tính chất | Thời gian | Mục đích |
|----------|-----------|-----------|----------|
| Offline | Batch processing | Phút đến giờ | Chuẩn bị dữ liệu, chạy định kỳ |
| Online | Real-time | < 5 giây | Phục vụ query người dùng |

---

## 3.3. Thiết kế Data Pipeline

### 3.3.1. Thu thập dữ liệu từ e-Gov API

Dữ liệu văn bản pháp luật được thu thập từ **e-Gov API** của Chính phủ Nhật Bản:

**Thông số API:**
- Endpoint: `https://elaws.e-gov.go.jp/api/1/lawdata/{law_id}`
- Format: XML
- Rate limit: ~1.2 giây/request (để tránh bị block)

**Quy trình thu thập:**
```
1. Category Search → Lấy danh sách law_id theo category
2. Keyword Search  → Bổ sung thêm luật theo keyword
3. Download XML   → Tải về từng file XML
4. Lưu trữ local  → Lưu vào thư mục data/laws/
```

**Danh mục luật được thu thập:**

| Category | Số luật | Ví dụ |
|----------|---------|-------|
| Lao động (労働) | ~50 | 労働基準法, 労働契約法 |
| Thuế (税金) | ~30 | 所得税法, 消費税法 |
| An sinh xã hội | ~40 | 国民健康保険法, 年金法 |
| Nhập cư | ~20 | 出入国管理法, 入管特例法 |

### 3.3.2. XML Parser cho văn bản pháp luật

Cấu trúc văn bản pháp luật Nhật Bản theo XML:

```xml
<LawBody>
  <LawTitle>労働基準法</LawTitle>
  <MainProvision>
    <Part Num="1">           <!-- 編 (Phần) -->
      <Chapter Num="4">      <!-- 章 (Chương) -->
        <Section Num="1">    <!-- 節 (Mục) -->
          <Article Num="32"> <!-- 条 (Điều) -->
            <ArticleTitle>労働時間</ArticleTitle>
            <Paragraph Num="1">  <!-- 項 (Khoản) -->
              <ParagraphSentence>
                使用者は、労働者に...
              </ParagraphSentence>
            </Paragraph>
          </Article>
        </Section>
      </Chapter>
    </Part>
  </MainProvision>
</LawBody>
```

**Mapping cấu trúc:**

| Thẻ XML | Tiếng Nhật | Tiếng Việt | Cấp độ |
|---------|------------|------------|--------|
| Part | 編 | Phần | 1 |
| Chapter | 章 | Chương | 2 |
| Section | 節 | Mục | 3 |
| Article | 条 | Điều | 4 |
| Paragraph | 項 | Khoản | 5 |
| Item | 号 | Điểm | 6 |

### 3.3.3. Chiến lược Chunking

**Quyết định thiết kế**: Paragraph-level chunking với context enrichment

| Phương án | Ưu điểm | Nhược điểm | Quyết định |
|-----------|---------|------------|------------|
| Fixed-size (512 tokens) | Đồng đều | Phá vỡ semantic coherence | ❌ |
| Sentence-level | Granular | Quá nhỏ, mất context | ❌ |
| Article-level | Context đầy đủ | Quá lớn (>2000 tokens) | ❌ |
| **Paragraph-level** | Mapping citation format | Phù hợp với cấu trúc pháp lý | ✅ |

**Context Enrichment**:

Mỗi chunk có hai phiên bản text:
- `text`: Nội dung thuần túy, dùng để hiển thị
- `text_with_context`: Prefix với law_title + article_title, dùng để embedding

```
text: "使用者は、労働者に、休憩時間を除き一週間について四十時間を超えて、労働させてはならない。"

text_with_context: "労働基準法 第三十二条 (労働時間) 使用者は、労働者に、休憩時間を除き一週間について四十時間を超えて、労働させてはならない。"
```

### 3.3.4. Embedding Pipeline

**Model lựa chọn**: OpenAI `text-embedding-3-large`

| Tiêu chí | text-embedding-3-large | Lý do chọn |
|----------|------------------------|------------|
| Dimension | 3072 | Độ chính xác cao |
| Multilingual | ✅ | Hỗ trợ Việt-Nhật |
| Benchmark (MTEB) | Top-tier | Performance tốt |
| Cost | $0.00013/1K tokens | Chấp nhận được |

**Quy trình Embedding:**
```
1. Load chunks từ JSON
2. Batch embedding (100 chunks/batch)
3. Dense vector: text-embedding-3-large
4. Sparse vector: SPLADE (cho BM25-style search)
5. Upsert to Qdrant với payload metadata
```

### 3.3.5. Hybrid Indexing (Qdrant)

**Collection Schema:**

```python
collection_config = {
    "name": "japanese_laws_hybrid",
    "vectors": {
        "dense": {
            "size": 3072,           # OpenAI embedding dimension
            "distance": "Cosine"
        }
    },
    "sparse_vectors": {
        "sparse": {
            "modifier": "IDF"       # BM25 style weighting
        }
    }
}
```

**Payload Schema:**

| Field | Type | Mô tả |
|-------|------|-------|
| `chunk_id` | string | Unique identifier |
| `law_id` | string | e-Gov law ID |
| `law_title` | string | Tên luật |
| `article_num` | string | Số điều |
| `article_title` | string | Tiêu đề điều |
| `paragraph_num` | string | Số khoản |
| `text` | string | Nội dung chunk |
| `text_with_context` | string | Nội dung + context |
| `category` | string | Phân loại (労働, 税金, etc.) |

**Hybrid Search với RRF Fusion:**
```python
# Prefetch từ cả dense và sparse
prefetch = [
    Prefetch(query=dense_vector, using="dense", limit=20),
    Prefetch(query=sparse_vector, using="sparse", limit=20),
]

# Fusion bằng Reciprocal Rank Fusion
query = Query(fusion=Fusion.RRF, prefetch=prefetch)
```

---

## 3.4. Thiết kế Knowledge Graph ⭐

### 3.4.1. Mô hình dữ liệu Graph (Node Types)

Hệ thống sử dụng **Neo4j Aura** để lưu trữ knowledge graph với các node types sau:

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH SCHEMA                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────┐                                                   │
│    │   LAW   │  (law_id, law_title, category, promulgation_date)│
│    └────┬────┘                                                   │
│         │ HAS_CHAPTER                                            │
│         ▼                                                        │
│    ┌─────────┐                                                   │
│    │ CHAPTER │  (chapter_num, chapter_title)                    │
│    └────┬────┘                                                   │
│         │ HAS_ARTICLE                                            │
│         ▼                                                        │
│    ┌─────────┐                                                   │
│    │ ARTICLE │  (article_num, article_title, article_caption)   │
│    └────┬────┘                                                   │
│         │ HAS_PARAGRAPH                                          │
│         ▼                                                        │
│    ┌─────────┐                                                   │
│    │PARAGRAPH│  (paragraph_num, text, chunk_id)                 │
│    └─────────┘                                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Node Properties:**

| Node Type | Properties | Mô tả |
|-----------|------------|-------|
| Law | law_id, law_title, category, promulgation_date | Luật (cấp cao nhất) |
| Chapter | chapter_num, chapter_title, law_id | Chương trong luật |
| Article | article_num, article_title, article_caption, law_id | Điều luật |
| Paragraph | paragraph_num, text, chunk_id | Khoản (liên kết với Qdrant) |

### 3.4.2. Các loại quan hệ (Relationship Types)

```cypher
// Hierarchical relationships (Quan hệ phân cấp)
(:Law)-[:HAS_CHAPTER]->(:Chapter)
(:Chapter)-[:HAS_ARTICLE]->(:Article)
(:Article)-[:HAS_PARAGRAPH]->(:Paragraph)

// Cross-reference relationships (Quan hệ tham chiếu chéo)
(:Article)-[:REFERENCES {ref_type: "explicit"}]->(:Article)
(:Article)-[:REFERENCES {ref_type: "implicit"}]->(:Article)

// Same-law relationship
(:Article)-[:SAME_LAW]->(:Article)
```

**Mô tả các quan hệ:**

| Relationship | Mô tả | Ví dụ |
|--------------|-------|-------|
| HAS_CHAPTER | Luật chứa chương | 労働基準法 → 第四章 |
| HAS_ARTICLE | Chương chứa điều | 第四章 → 第三十二条 |
| HAS_PARAGRAPH | Điều chứa khoản | 第三十二条 → 第一項 |
| REFERENCES | Điều tham chiếu điều khác | 第36条 → 第32条 |

### 3.4.3. Graph Schema Design

**Cypher Schema:**

```cypher
// Law node
(:Law {
    law_id: "322AC0000000049",
    law_title: "労働基準法",
    category: "労働",
    promulgation_date: "1947-04-07"
})

// Article node
(:Article {
    article_num: "第三十二条",
    article_title: "労働時間",
    article_caption: "法定労働時間",
    law_id: "322AC0000000049"
})

// Paragraph node với linking to Vector Store
(:Paragraph {
    paragraph_num: "1",
    text: "使用者は、労働者に...",
    chunk_id: "chunk_32_1"  // ← Links to Qdrant
})
```

### 3.4.4. Connection với Vector Store (chunk_id linking)

**Chiến lược liên kết Graph ↔ Vector:**

```
┌──────────────────┐         ┌──────────────────┐
│     Neo4j        │         │     Qdrant       │
│   (Structure)    │  ←───→  │    (Content)     │
├──────────────────┤         ├──────────────────┤
│ :Paragraph {     │         │ Point {          │
│   chunk_id: X    │ ──────→ │   id: X          │
│ }                │         │   payload: {...} │
│                  │         │   vector: [...]  │
└──────────────────┘         └──────────────────┘
```

**Quy trình tích hợp:**
1. Graph search trả về `chunk_id`
2. Dùng `chunk_id` để fetch full text từ Qdrant
3. Kết hợp thông tin cấu trúc (từ Graph) với nội dung (từ Vector)

---

## 3.5. Thiết kế RAG Pipeline

### 3.5.1. Query Translation & Multi-Query Expansion

**Vấn đề**: Query tiếng Việt cần được dịch sang tiếng Nhật để tìm kiếm hiệu quả trong corpus tiếng Nhật.

**Giải pháp**: Sử dụng GPT-4o-mini cho translation + expansion

```python
# Input
query = "Thời gian làm việc tối đa mỗi tuần là bao nhiêu?"

# Output từ Query Translator
{
    "translated_query": "週の最大労働時間は何時間ですか",
    "keywords": ["法定労働時間", "週40時間", "労働基準法"],
    "search_queries": [
        "法定労働時間とは",
        "週の労働時間制限",
        "労働基準法第32条"
    ]
}
```

**Multi-Query Expansion:**
- Từ 1 query gốc → 3-5 search queries
- Mỗi query tập trung vào khía cạnh khác nhau
- Tăng recall, đa dạng hóa kết quả

### 3.5.2. Hybrid Retrieval (Vector Search)

**Hybrid Search = Dense + Sparse:**

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| Dense (Semantic) | Hiểu ý nghĩa | Có thể miss exact match |
| Sparse (BM25) | Exact keyword match | Không hiểu synonym |
| **Hybrid (RRF)** | Kết hợp cả hai | Tốt nhất cho legal domain |

**Reciprocal Rank Fusion (RRF):**
```
score(d) = Σ 1 / (k + rank_i(d))
```
Trong đó k = 60 (constant), rank_i(d) là rank của document d trong kết quả tìm kiếm thứ i.

**Luồng xử lý:**
```
Search Queries → [Dense Search] → Top-20 candidates
              → [Sparse Search] → Top-20 candidates
              → RRF Fusion → Top-K merged results
```

### 3.5.3. Reranking

**Model**: Cross-encoder `mMarco-mMiniLM`

**Tại sao cần Reranking?**
- Bi-encoder (embedding) nhanh nhưng approximate
- Cross-encoder chậm hơn nhưng chính xác hơn
- Two-stage: Retrieval (recall) → Reranking (precision)

**Quy trình:**
```
Top-20 candidates → Cross-encoder scoring → Top-5 reranked
```

### 3.5.4. LLM Generation với citation

**Model**: GPT-4o-mini

**Prompt Engineering:**
```
Bạn là chuyên gia tư vấn pháp luật Nhật Bản.
Trả lời bằng tiếng Việt, giữ thuật ngữ Nhật trong ngoặc [].
Trích dẫn nguồn bằng [1], [2], etc.

Nguồn:
[1] 労働基準法 第三十二条: "使用者は、労働者に..."
[2] ...

Câu hỏi: {query}
```

**Output format:**
```
Theo quy định tại Điều 32 [第三十二条] của Luật Tiêu chuẩn Lao động 
[労働基準法], thời gian làm việc tối đa là 40 giờ/tuần [1].
```

---

## 3.6. Thiết kế GraphRAG Pipeline ⭐

### 3.6.1. Query Routing (Khi nào dùng Graph vs Vector)

**Query Router**: Phân tích query để quyết định phương pháp tìm kiếm

```python
class QueryType(Enum):
    SEMANTIC = "semantic"           # → Vector search only
    ENTITY_LOOKUP = "entity_lookup" # → Graph search only
    MULTI_HOP = "multi_hop"         # → Graph traversal
    HYBRID = "hybrid"               # → Both Graph + Vector
```

**Routing Logic:**

| Query Pattern | Query Type | Search Method |
|---------------|------------|---------------|
| "労働時間について教えて" | SEMANTIC | Vector only |
| "第32条の内容は?" | ENTITY_LOOKUP | Graph lookup → Vector |
| "第32条に関連する条文は?" | MULTI_HOP | Graph traversal |
| "労働基準法第32条について" | HYBRID | Graph + Vector |

### 3.6.2. Entity Extraction từ query

**Pattern Matching:**
```python
ENTITY_PATTERNS = [
    # 労働基準法第32条 → (労働基準法, 32)
    (r'([ぁ-んァ-ン一-龯]+法)第(\d+)条', 'law_article'),
    
    # 第32条 (standalone article)
    (r'第(\d+)条(?:の(\d+))?', 'article_only'),
    
    # Law names
    (r'([ぁ-んァ-ン一-龯]+法)', 'law_name'),
]
```

**Ví dụ extraction:**
```
Query: "労働基準法第32条の内容を教えてください"
Entities: [("労働基準法第32条", "law_article")]
```

### 3.6.3. Graph Search (find_article, find_related_articles)

**GraphService Methods:**

```python
class GraphService:
    def find_article(self, law_title: str, article_num: str) -> GraphResult:
        """Tìm điều luật cụ thể theo tên luật và số điều"""
        
    def find_related_articles(self, law_id: str, article_num: str, 
                              depth: int = 2) -> List[GraphResult]:
        """Tìm các điều có quan hệ REFERENCES"""
        
    def search_by_keyword(self, keyword: str) -> List[GraphResult]:
        """Tìm theo keyword trong article_title/caption"""
        
    def get_law_structure(self, law_id: str) -> dict:
        """Lấy cấu trúc phân cấp của luật"""
```

**Cypher Query Example:**
```cypher
// Find article
MATCH (l:Law)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_ARTICLE]->(a:Article)
WHERE l.law_title CONTAINS $law_title AND a.article_num = $article_num
RETURN l, a

// Find related articles
MATCH (a:Article {article_num: $article_num})
      -[:REFERENCES*1..2]->(related:Article)
RETURN related
```

### 3.6.4. Result Fusion (Kết hợp Graph + Vector results)

**Fusion Strategy:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESULT FUSION FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌──────────────┐        ┌──────────────┐                     │
│    │ Graph Search │        │Vector Search │                     │
│    │  (Structure) │        │  (Semantic)  │                     │
│    └──────┬───────┘        └──────┬───────┘                     │
│           │                       │                              │
│           ▼                       ▼                              │
│    ┌──────────────┐        ┌──────────────┐                     │
│    │ chunk_ids    │        │ chunk_ids    │                     │
│    │ + graph_score│        │ + vector_score│                    │
│    └──────┬───────┘        └──────┬───────┘                     │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│              ┌──────────────┐                                    │
│              │ Score Fusion │                                    │
│              │ graph_boost  │                                    │
│              └──────┬───────┘                                    │
│                     ▼                                            │
│              ┌──────────────┐                                    │
│              │Deduplicated  │                                    │
│              │   Results    │                                    │
│              └──────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Score Calculation:**
```python
# Graph results get a boost
final_score = vector_score * (1 + graph_boost)
# where graph_boost = 0.3 if found in graph, else 0
```

---

## 3.7. Thiết kế LangGraph Agent

### 3.7.1. State Graph Definition

**State Definition:**
```python
class LegalRAGState(TypedDict):
    query: str                      # Original Vietnamese query
    translated_query: str           # Japanese translation
    search_queries: list[str]       # Expanded search queries
    documents: list[dict]           # Retrieved documents
    document_grades: list[str]      # "relevant" | "not_relevant"
    reranked_documents: list[dict]  # After cross-encoder
    answer: str                     # Final answer
    sources: list[dict]             # Citation sources
    rewrite_count: int              # Retry counter (max 2)
```

### 3.7.2. Các Node (Translate, Retrieve, Grade, Rerank, Generate, Rewrite)

| Node | Input | Output | Chức năng |
|------|-------|--------|-----------|
| `translate` | query | translated_query, search_queries | Dịch và mở rộng query |
| `retrieve` | search_queries | documents | Tìm kiếm hybrid |
| `grade` | documents | document_grades | Đánh giá relevance |
| `rerank` | documents | reranked_documents | Xếp hạng lại |
| `generate` | reranked_documents | answer, sources | Sinh câu trả lời |
| `rewrite` | query | search_queries | Viết lại query |

### 3.7.3. Conditional Edge và Self-Correction Loop

**Graph Structure:**
```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  translate  │
                    └──────┬──────┘
                           ▼
            ┌──────────────────────────┐
            │       retrieve_node      │◀──────────┐
            └────────────┬─────────────┘           │
                         ▼                         │
            ┌──────────────────────────┐           │
            │   grade_documents_node   │           │
            └────────────┬─────────────┘           │
                         │                         │
           ┌─────────────┴─────────────┐          │
           │                           │          │
     relevant >= 2              relevant < 2      │
     OR retry >= 2              AND retry < 2     │
           │                           │          │
           ▼                           ▼          │
    ┌─────────────┐           ┌─────────────┐    │
    │ rerank_node │           │rewrite_query│────┘
    └──────┬──────┘           │    node     │
           │                  └─────────────┘
           ▼
    ┌─────────────┐
    │generate_node│
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │    END      │
    └─────────────┘
```

**Decision Function:**
```python
def should_rewrite(state: LegalRAGState) -> str:
    relevant_count = sum(1 for g in state["document_grades"] if g == "relevant")
    
    if relevant_count >= 2 or state["rewrite_count"] >= 2:
        return "rerank"  # Proceed to reranking
    else:
        return "rewrite"  # Try rewriting query
```

---

## 3.8. Thiết kế API (REST Endpoints)

### API Schema

**Base URL:** `https://api.norman.example.com/api`

| Method | Endpoint | Mô tả | Request Body | Response |
|--------|----------|-------|--------------|----------|
| GET | `/health` | Health check | - | HealthResponse |
| POST | `/search` | Vector search | SearchQuery | SearchResponse |
| POST | `/chat` | RAG Chat | ChatQuery | ChatResponse |
| POST | `/translate` | Dịch văn bản | TranslateRequest | TranslateResponse |

### Request/Response Schemas

**SearchQuery:**
```json
{
    "query": "労働時間の制限",
    "top_k": 5,
    "filters": {"category": "労働"}
}
```

**ChatQuery:**
```json
{
    "query": "Thời gian làm việc tối đa mỗi tuần?",
    "top_k": 5,
    "use_agent": false
}
```

**ChatResponse:**
```json
{
    "answer": "Theo quy định tại Điều 32 [第三十二条]...",
    "sources": [
        {
            "law_title": "労働基準法",
            "article": "第三十二条",
            "text": "使用者は、労働者に...",
            "score": 0.85
        }
    ],
    "query": "Thời gian làm việc tối đa mỗi tuần?",
    "processing_time_ms": 1250.5
}
```

**TranslateResponse:**
```json
{
    "original": "使用者は、労働者に...",
    "translated": "Người sử dụng lao động không được...",
    "processing_time_ms": 850.2
}
```

---

## Kết chương

Chương này đã trình bày phân tích yêu cầu và thiết kế chi tiết của hệ thống Norman - Chatbot tư vấn pháp luật Nhật Bản. Hệ thống được thiết kế với kiến trúc multi-layer, bao gồm:

- **Data Pipeline**: Thu thập từ e-Gov API, chunking paragraph-level, hybrid indexing
- **Knowledge Graph**: Neo4j với cấu trúc Law → Chapter → Article → Paragraph
- **RAG Pipeline**: Query translation, hybrid search, reranking, LLM generation
- **GraphRAG Pipeline**: Query routing, entity extraction, graph search, result fusion
- **LangGraph Agent**: Self-correction loop với document grading
- **REST API**: 4 endpoints cho search, chat, translate, và health check

Chương tiếp theo sẽ trình bày chi tiết quá trình triển khai và cài đặt từng thành phần của hệ thống.
