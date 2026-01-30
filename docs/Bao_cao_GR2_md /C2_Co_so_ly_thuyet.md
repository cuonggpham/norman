# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

Chương này trình bày các cơ sở lý thuyết làm nền tảng cho hệ thống, bao gồm: (i) Retrieval-Augmented Generation trong phần 2.1, (ii) vector embedding và semantic search trong phần 2.2, (iii) hybrid search trong phần 2.3, (iv) reranking trong phần 2.4, (v) knowledge graph và GraphRAG trong phần 2.5, và (vi) AI Agent trong phần 2.6.

---

## 2.1. Retrieval-Augmented Generation (RAG)

### 2.1.1. Kiến trúc RAG cơ bản

Retrieval-Augmented Generation (RAG) là kiến trúc kết hợp giữa hệ thống truy xuất thông tin (Information Retrieval) và mô hình sinh ngôn ngữ (Language Generation). RAG được giới thiệu bởi Lewis et al. (2020) trong bài báo *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* tại hội nghị NeurIPS.

**Ý tưởng cốt lõi**: Thay vì buộc LLM ghi nhớ tất cả kiến thức trong tham số (parametric memory), hệ thống truy xuất tài liệu liên quan từ kho dữ liệu bên ngoài (non-parametric memory) và đưa vào context để LLM sinh câu trả lời.

**Các thành phần chính**:

1. **Indexing Pipeline** (Offline):
   - Thu thập và xử lý dữ liệu nguồn
   - Chia documents thành các chunks
   - Tạo embedding vectors và lưu vào vector database

2. **Retrieval Pipeline** (Runtime):
   - Chuyển đổi query thành embedding vector
   - Tìm kiếm chunks tương tự trong database
   - Trả về top-k chunks liên quan nhất

3. **Generation Pipeline** (Runtime):
   - Xây dựng prompt từ query và retrieved chunks
   - LLM sinh câu trả lời dựa trên context

**Luồng xử lý**:
```
Query → Embedding → Vector Search → Retrieved Chunks → Prompt → LLM → Answer
```

### 2.1.2. So sánh RAG với Fine-tuning và Prompt Engineering

Có ba cách tiếp cận chính để cung cấp domain knowledge cho LLM:

| Tiêu chí | Prompt Engineering | Fine-tuning | RAG |
|----------|-------------------|-------------|-----|
| **Chi phí** | Thấp | Rất cao | Trung bình |
| **Cập nhật kiến thức** | Manual | Re-train | Chỉ re-index |
| **Trích dẫn nguồn** | Có thể | Không | Có |
| **Hallucination** | Cao | Trung bình | Thấp |
| **Scalability** | Kém (context limit) | Tốt | Tốt |
| **Minh bạch** | Thấp | Rất thấp | Cao |

**Prompt Engineering**: Cung cấp thông tin trực tiếp trong prompt. Đơn giản nhưng giới hạn bởi context window.

**Fine-tuning**: Huấn luyện lại model trên dữ liệu domain-specific. Model "nhớ" kiến thức trong parameters nhưng khó cập nhật và tốn kém.

**RAG**: Truy xuất động từ knowledge base. Dễ cập nhật, có citations, nhưng phức tạp hơn trong triển khai.

### 2.1.3. Các thách thức trong RAG

**1. Chunk Quality**: 
- Chunking quá nhỏ mất context, quá lớn thì nhiễu
- Chunk boundaries có thể cắt giữa câu, mất ngữ nghĩa

**2. Retrieval Quality**:
- Vocabulary mismatch: query dùng từ khác với document
- Semantic gap: dense embedding có thể miss exact keywords

**3. Generation Quality**:
- Lost in the middle: LLM bỏ qua thông tin giữa context dài
- Conflicting information từ các chunks khác nhau

**4. Cross-lingual Challenges**:
- Query và documents khác ngôn ngữ
- Cần translation hoặc cross-lingual embeddings

---

## 2.2. Vector Embedding và Semantic Search

### 2.2.1. Khái niệm Vector Embedding

**Định nghĩa**: Vector embedding là quá trình chuyển đổi dữ liệu phi cấu trúc (text, image, audio) thành vector số học trong không gian nhiều chiều. Dữ liệu có ngữ nghĩa tương tự sẽ có vector gần nhau.

**Lịch sử phát triển**:
- **Word2Vec (2013)**: Mikolov et al., học word embeddings từ co-occurrence
- **GloVe (2014)**: Stanford, kết hợp matrix factorization và local context
- **ELMo (2018)**: Contextualized embeddings từ bidirectional LSTM
- **BERT (2018)**: Transformer-based, pre-trained bidirectional representations
- **Sentence-BERT (2019)**: Optimize BERT cho sentence embeddings

**Các loại Embedding**:

1. **Word-level Embedding**: Mỗi từ có vector cố định, không capture context
2. **Contextual Embedding**: Vector khác nhau tùy context (ELMo, BERT)
3. **Sentence/Document Embedding**: Một vector cho toàn bộ câu/đoạn

### 2.2.2. Các mô hình Embedding phổ biến

**Commercial Models**:
- OpenAI text-embedding-3-large/small
- Cohere embed-multilingual-v3
- Google text-embedding-004

**Open-source Models**:
- BGE-M3: Multi-lingual, multi-granularity
- multilingual-e5-large: Microsoft, 100+ languages
- jina-embeddings-v2: Long context (8K tokens)

**Tiêu chí lựa chọn**:
- Dimension size: cao hơn thường capture nhiều thông tin hơn
- Multilingual support: quan trọng cho cross-lingual applications
- Context length: giới hạn số tokens input
- Latency và cost

### 2.2.3. Vector Similarity Metrics

**1. Cosine Similarity** (phổ biến nhất):

$$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

- Đo góc giữa hai vectors, không phụ thuộc độ dài
- Giá trị từ -1 đến 1

**2. Euclidean Distance (L2)**:

$$d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$$

- Đo khoảng cách tuyệt đối trong không gian
- Nhạy cảm với magnitude

**3. Dot Product**:

$$\text{dot}(A, B) = \sum_{i=1}^{n} A_i B_i$$

- Với normalized vectors, tương đương cosine similarity
- Đơn giản và nhanh

**Approximate Nearest Neighbor (ANN)**:

Do exact search O(n) chậm với datasets lớn, ANN algorithms được sử dụng:
- **HNSW**: Graph-based, recall cao
- **IVF**: Cluster-based, memory efficient
- **PQ**: Compression, giảm storage

---

## 2.3. Hybrid Search (Dense + Sparse)

### 2.3.1. Dense Retrieval vs Sparse Retrieval

**Sparse Retrieval - BM25**:

BM25 (Best Matching 25) là thuật toán sparse retrieval dựa trên TF-IDF:

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k + 1)}{f(q_i, D) + k \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

Trong đó:
- $f(q_i, D)$: Tần suất query term trong document
- $|D|$: Độ dài document
- $k$, $b$: Hyperparameters (thường k=1.2, b=0.75)

**Đặc điểm**: Exact term matching, nhanh, interpretable, nhưng không hiểu synonyms.

**Dense Retrieval**:

Sử dụng neural network encode query và documents thành dense vectors, so sánh bằng similarity metrics.

**Đặc điểm**: Hiểu ngữ nghĩa, synonyms, cross-lingual, nhưng có thể miss exact keywords.

**So sánh**:

| Tiêu chí | Sparse (BM25) | Dense (Embedding) |
|----------|---------------|-------------------|
| Exact match | ✅ Tốt | ❌ Có thể miss |
| Semantic | ❌ Không | ✅ Tốt |
| Cross-lingual | ❌ Không | ✅ Có |
| Proper nouns | ✅ Tốt | ❌ Kém |

### 2.3.2. Reciprocal Rank Fusion (RRF)

RRF kết hợp kết quả từ nhiều ranking systems mà không cần normalize scores:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Trong đó:
- $\text{rank}_r(d)$: Thứ hạng của document $d$ trong hệ thống $r$
- $k$: Hằng số smoothing (thường = 60)

**Ví dụ**:

| Document | Dense Rank | Sparse Rank | RRF Score |
|----------|------------|-------------|-----------|
| Doc A | 1 | 5 | 1/61 + 1/65 = 0.0318 |
| Doc B | 3 | 1 | 1/63 + 1/61 = 0.0323 |

**Ưu điểm**: Không cần normalize, robust với outliers, đơn giản.

### 2.3.3. Ưu điểm của Hybrid Search

1. **Complementary Strengths**: Dense bắt semantic, sparse bắt exact matches
2. **Robustness**: Backup nếu một hệ thống fail
3. **Long-tail Queries**: Rare terms có thể không có tốt embeddings, BM25 vẫn match được

---

## 2.4. Reranking (Two-Stage Retrieval)

### 2.4.1. Bi-Encoder vs Cross-Encoder

**Bi-Encoder**:
```
Query → Encoder → Query Vector ─┐
                                 ├→ Similarity → Score
Document → Encoder → Doc Vector ─┘
```
- Encode độc lập, vectors có thể pre-compute
- Nhanh: O(log n) với ANN

**Cross-Encoder**:
```
[CLS] Query [SEP] Document [SEP] → Transformer → Score
```
- Encode cùng nhau, full cross-attention
- Chậm: O(n), không thể pre-index
- Accurate hơn

**So sánh**:

| Tiêu chí | Bi-Encoder | Cross-Encoder |
|----------|------------|---------------|
| Speed | ~1ms/doc | ~50-100ms/doc |
| Accuracy | Tốt | Rất tốt |
| Pre-indexing | ✅ Có | ❌ Không |
| Use case | First-stage | Second-stage |

**Two-Stage Pipeline**:
```
Query → Bi-Encoder → Top-K candidates → Cross-Encoder → Top-N final
```

### 2.4.2. Các mô hình Reranker phổ biến

**Commercial**: Cohere Rerank, Jina Reranker

**Open-source**:
- BGE-Reranker: High quality, multilingual
- ms-marco-MiniLM: Fast, English-focused
- mmarco-mMiniLM: Multilingual, balanced

**Trade-offs**: Larger models = higher accuracy, higher latency, higher resource usage.

---

## 2.5. Knowledge Graph và Graph RAG

### 2.5.1. Khái niệm Knowledge Graph

**Định nghĩa**: Knowledge Graph là cấu trúc dữ liệu biểu diễn thông tin dưới dạng đồ thị với nodes (entities) và edges (relationships).

**Thành phần**:

1. **Nodes (Entities)**: Objects, concepts với type và properties
2. **Edges (Relationships)**: Connections có type giữa nodes
3. **Properties**: Attributes key-value

**Triple Pattern**:
```
(Subject) -[Predicate]-> (Object)
(Law) -[HAS_ARTICLE]-> (Article)
(Article) -[REFERENCES]-> (Article)
```

### 2.5.2. Graph Database và Neo4j

**Graph Database**: Hệ thống tối ưu cho lưu trữ và query graph data. Native store nodes và edges, không cần JOIN như relational DB.

**Ưu điểm**:
- Traversal O(1) per hop
- Flexible schema
- Intuitive modeling

**Cypher Query Language** (Neo4j):
```cypher
-- Tìm articles thuộc một law
MATCH (l:Law)-[:HAS_ARTICLE]->(a:Article)
WHERE l.law_id = "law_5"
RETURN a

-- Multi-hop traversal
MATCH (a1:Article)-[:REFERENCES*1..2]->(a2:Article)
RETURN a2
```

### 2.5.3. Graph RAG: Kết hợp Knowledge Graph với RAG

**Motivation**: Vector RAG tốt cho semantic similarity nhưng kém cho:
- Entity-specific queries: "Điều 15 nói gì?"
- Multi-hop reasoning: "Điều nào reference đến Điều 15?"
- Structural queries: "Luật X có bao nhiêu chương?"

**GraphRAG Architecture**:
```
Query → Router → ┌── Vector Search ──┐
                 │                    │
                 └── Graph Search ────┼── Fusion → LLM → Answer
```

**Graph Search Methods**:
1. **Entity Lookup**: Tìm chính xác entity
2. **Relation Traversal**: Đi theo edges
3. **Aggregation**: Count, sum operations

### 2.5.4. So sánh Vector RAG vs Graph RAG

| Query Type | Vector RAG | Graph RAG |
|------------|------------|-----------|
| Semantic similarity | ✅ Xuất sắc | ❌ Không |
| Entity lookup | ❌ Approximate | ✅ Exact |
| Multi-hop | ❌ Không | ✅ Native |
| Cross-lingual | ✅ Có | ❌ Không |
| Aggregation | ❌ Không | ✅ Native |

---

## 2.6. AI Agent với LangGraph

### 2.6.1. Agent và State Machine

**AI Agent**: Hệ thống AI có khả năng:
- Perceive (nhận input)
- Reason (lập kế hoạch)
- Act (gọi tools)
- Adapt (học từ feedback)

**Agent trong RAG** không chỉ linear pipeline, mà có thể:
- Quyết định retrieve thêm nếu chưa đủ
- Rewrite query nếu results không relevant
- Sử dụng nhiều tools (vector, graph, web search)

**State Machine Approach**:

```
     ┌─────────────────────────────────────┐
     │                                     │
     ▼                                     │
[Retrieve] → [Grade] → [Rerank] → [Generate]
     │                               │
     │    Not enough results         │
     └───────────────────────────────┘
             ▲
             │
       [Rewrite Query]
```

**LangGraph**: Framework build stateful applications:
- **Nodes**: Processing steps
- **Edges**: Transitions
- **State**: Shared data
- **Conditional Edges**: Dynamic routing

### 2.6.2. Self-Correction trong RAG

**1. Document Grading**:
- LLM đánh giá mỗi document: relevant hay không?
- Filter irrelevant documents
- Nếu không đủ relevant docs → trigger rewrite

**2. Query Rewriting**:
- Nếu retrieval kém, LLM rewrite query
- Thử alternative phrasings, keywords
- Retry retrieval

**3. Hallucination Check**:
- Verify claims against source documents
- Regenerate hoặc add disclaimer nếu phát hiện unsupported claims

---

## Kết chương

Chương này đã trình bày các cơ sở lý thuyết quan trọng:

1. **RAG**: Kiến trúc kết hợp retrieval và generation, ưu việt về khả năng cập nhật và trích dẫn nguồn.

2. **Vector Embedding**: Chuyển đổi text thành vectors, cho phép semantic search dựa trên similarity metrics.

3. **Hybrid Search**: Kết hợp dense (semantic) và sparse (BM25) bằng RRF fusion để tận dụng ưu điểm cả hai.

4. **Reranking**: Two-stage retrieval với cross-encoder để cải thiện precision.

5. **Knowledge Graph**: Cấu trúc graph cho entity lookup và multi-hop reasoning, GraphRAG kết hợp với vector RAG.

6. **AI Agent**: State machine approach với self-correction để tăng robustness của hệ thống.

Chương tiếp theo sẽ trình bày phân tích yêu cầu và thiết kế hệ thống dựa trên các lý thuyết này.
