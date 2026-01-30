# CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

Chương 5 đã đánh giá hiệu quả của hệ thống với framework RAGAS. Chương này sẽ trình bày các hạn chế của hệ thống và đề xuất hướng phát triển, bao gồm: (i) hạn chế hiện tại trong phần 6.1, (ii) hướng phát triển trong phần 6.2, và (iii) kết luận trong phần 6.3.

---

## 6.1. Hạn chế hiện tại

### 6.1.1. Latency vẫn cao

- 8-10 giây response time vẫn chậm hơn ChatGPT (~1s)
- Bottleneck: LLM calls (translation + generation) chiếm ~70% latency
- Network latency với các external services (Pinecone, OpenAI API)

### 6.1.2. Coverage cần mở rộng

- 431 luật đã cover tốt lĩnh vực tài chính và lao động
- Còn thiếu: dân sự (民法), thương mại (商法), xuất nhập cảnh (出入国管理法)
- Các hiệp định song phương chưa được index

### 6.1.3. Không có conversation memory

- Mỗi query được xử lý độc lập
- Không hỗ trợ follow-up questions như "Vậy còn trường hợp ngoại lệ?"

### 6.1.4. Cross-reference extraction còn sơ sài

- Regex-based extraction bỏ sót nhiều implicit references
- Các cách diễn đạt phức tạp chưa được xử lý

---

## 6.2. Hướng phát triển hệ thống RAG

### 6.2.1. Cải thiện Retrieval Quality

#### 6.2.1.1. Adaptive Chunking Strategy

**Hiện tại:** Fixed-size chunking với overlap 200 tokens gây ra:
- Split mid-sentence làm mất context
- Related clauses bị tách rời

**Đề xuất:** Semantic-aware chunking với LLM assistance:
- Sử dụng LLM để xác định boundary tự nhiên của legal clauses
- Hierarchical chunking với 3 levels: Article → Section → Clause
- Automatic context injection cho các chunks con

**Expected improvement:** +8-12% Context Recall dựa trên benchmarks từ các nghiên cứu tương tự.

#### 6.2.1.2. Query Understanding Enhancement

**Pipeline đề xuất:**
1. **Query Classification**: Phân loại query thành categories (factual, procedural, comparative, hypothetical)
2. **Intent Detection**: Xác định user intent để điều chỉnh retrieval strategy
3. **Query Expansion**: Sinh synonyms và related terms cho tiếng Nhật pháp lý

#### 6.2.1.3. Fine-tuned Embedding Model

**Approach:** Contrastive learning trên legal domain:
- Training data: (query_vi, positive_doc_ja, negative_docs)
- Base model: multilingual-e5-large
- Adapter layer để preserve general knowledge

**Data requirements:**
- 10,000+ triplets từ query logs
- Hard negatives mining từ similar but irrelevant documents

### 6.2.2. Advanced RAG Architectures

#### 6.2.2.1. CRAG (Corrective RAG)

Implement self-correction mechanism:
1. **Confidence Scoring**: LLM đánh giá confidence của retrieved documents
2. **Web Search Fallback**: Nếu confidence < threshold, trigger web search
3. **Knowledge Refinement**: Combine retrieved + web results

#### 6.2.2.2. Self-RAG

Implement reflection tokens để control generation:
- **[Retrieve]**: Quyết định có cần retrieve thêm không
- **[IsRel]**: Đánh giá relevance của retrieved passages
- **[IsSup]**: Verify claims được support bởi context
- **[IsUse]**: Đánh giá usefulness của generated response

#### 6.2.2.3. Agentic RAG với Multi-Step Reasoning

**Architecture đề xuất:**
```
User Query
    ↓
Query Analyzer Agent
    ↓
[Decompose into sub-queries if complex]
    ↓
Parallel Retrieval Agents
    ↓
Evidence Aggregator Agent
    ↓
Reasoning Agent (with chain-of-thought)
    ↓
Verification Agent
    ↓
Response Generator
```

**Tools cho Agent:**
- `search_vector`: Semantic search trên Pinecone
- `search_keyword`: BM25 search trên Elasticsearch
- `search_graph`: Traverse Neo4j knowledge graph
- `calculate_tax`: Tool tính thuế với các parameters
- `lookup_table`: Tra bảng phí, mức lương tối thiểu
- `cross_reference`: Tìm các điều luật liên quan

### 6.2.3. Full GraphRAG Implementation

#### 6.2.3.1. Entity Extraction Enhancement

**Hiện tại:** Regex-based extraction với limited pattern matching
**Đề xuất:** NER model fine-tuned cho Japanese legal text

**Entity types:**
- LAW_ARTICLE: 所得税法第27条
- LEGAL_CONCEPT: 源泉徴収, 確定申告
- ORGANIZATION: 国税庁, 年金機構
- MONETARY_AMOUNT: 103万円, 38万円
- TIME_PERIOD: 毎年6月30日まで
- CONDITION: 居住者である場合

#### 6.2.3.2. Relationship Classification

**Relationship types:**
- AMENDS: Điều này sửa đổi điều khác
- REFERENCES: Điều này tham chiếu điều khác
- SUPERSEDES: Điều này thay thế điều khác
- EXCEPTION_TO: Điều này là ngoại lệ của
- REQUIRES: Điều kiện tiên quyết
- DEFINES: Định nghĩa thuật ngữ

#### 6.2.3.3. Multi-hop Reasoning Queries

**Example flow:**
```
Query: "Người nước ngoài có visa kỹ sư, làm việc 2 năm, có được nhận bảo hiểm thất nghiệp không?"

Step 1: Retrieve visa conditions → 技術・人文知識・国際業務
Step 2: Check insurance eligibility → 雇用保険法第6条
Step 3: Verify work duration requirement → 被保険者期間
Step 4: Cross-check foreign worker exceptions → 外国人労働者特例
Step 5: Aggregate and generate response
```

### 6.2.4. Hybrid Search Optimization

#### 6.2.4.1. Adaptive Fusion Weights

**Hiện tại:** Fixed ratio 0.7 dense / 0.3 sparse
**Đề xuất:** Query-dependent dynamic weights

| Query Type | Dense Weight | Sparse Weight | Rationale |
|------------|--------------|---------------|-----------|
| Conceptual | 0.8 | 0.2 | Semantic understanding |
| Specific article | 0.3 | 0.7 | Exact match needed |
| Mixed | 0.6 | 0.4 | Balanced approach |

**Implementation:** Train classifier để predict optimal weights per query.

#### 6.2.4.2. Late Interaction Models (ColBERT)

Thay vì single-vector representation, sử dụng token-level interaction:
- Preserve fine-grained matching signals
- Better handling cho long documents
- Expected: +5-8% accuracy với minimal latency increase (với GPU)

### 6.2.5. Caching và Optimization

#### 6.2.5.1. Multi-level Caching với Redis

| Cache Level | Content | TTL | Hit Rate Expected |
|-------------|---------|-----|-------------------|
| Query embedding | Vector representation | 24h | 35% |
| Translation | Vi→Ja translations | 7d | 45% |
| Search results | Top-k documents | 1h | 25% |
| Full response | Complete answers | 6h | 15% |

**Expected latency reduction:** < 1s cho cached queries (85% improvement)

#### 6.2.5.2. Speculative Retrieval

Pre-fetch related documents dựa trên:
- Common follow-up patterns
- User browsing behavior
- Related legal concepts

### 6.2.6. Evaluation Pipeline Enhancement

#### 6.2.6.1. Continuous Evaluation

Automated evaluation pipeline chạy hàng tuần:
- Monitor metric drift
- A/B testing cho new configurations
- Alert khi performance degradation

#### 6.2.6.2. Human-in-the-loop Feedback

- Thumbs up/down on responses
- Explicit feedback collection
- Use feedback for model fine-tuning

---

## 6.3. Kết luận

### 6.3.1. Những gì đã đạt được

Qua đồ án này, một hệ thống RAG hoàn chỉnh đã được xây dựng với các thành phần:

- **End-to-end RAG pipeline**: Từ data ingestion đến response generation
- **431 văn bản luật** được index với **206,014 chunks**
- **Cross-lingual retrieval**: Vietnamese query → Japanese documents
- **Hybrid search**: Kết hợp semantic (dense) và keyword (sparse)
- **Two-stage retrieval**: Bi-encoder + cross-encoder reranking
- **LangGraph agent**: Self-correction loop cho complex queries
- **GraphRAG integration**: Knowledge graph với Neo4j

Hệ thống đạt **Faithfulness 0.85** trên RAGAS evaluation, cho thấy câu trả lời được grounding tốt vào context.

### 6.3.2. Bài học kỹ thuật

**1. Data quality > Model size**

Chunking strategy phù hợp và context enrichment có impact lớn hơn việc sử dụng embedding model đắt tiền. Paragraph-level chunking với context prefix cải thiện retrieval 15-20% so với naive fixed-size chunking.

**2. Hybrid approach outperforms single method**

Dense + Sparse embedding, Vector + Graph search, Bi-encoder + Cross-encoder. Combination thường tốt hơn single method.

**3. Efficiency without compromise**

Việc tìm ra model mMiniLM cho thấy có thể giữ high precision của cross-encoder mà vẫn đạt latency thấp. Selection model phù hợp quan trọng hơn là blindly disabling features.

### 6.3.3. Tổng kết

Đồ án này là một trải nghiệm học tập quý giá về việc xây dựng hệ thống AI ứng dụng thực tế. Hệ thống chưa hoàn hảo và còn nhiều điểm cần cải thiện, đặc biệt là hướng phát triển sâu về RAG với các advanced architectures như Self-RAG, CRAG, và Agentic RAG.

Tuy nhiên, mục tiêu ban đầu đã đạt được: một công cụ giúp người Việt Nam tại Nhật Bản tiếp cận thông tin pháp luật dễ dàng hơn, bằng chính ngôn ngữ của họ, với trích dẫn nguồn chính xác.

---

## Kết chương

Chương này đã trình bày các hạn chế hiện tại của hệ thống và đề xuất hướng phát triển tập trung vào cải thiện hệ thống RAG. Các hướng phát triển chính bao gồm:

- **Retrieval quality**: Adaptive chunking, query understanding, fine-tuned embeddings
- **Advanced RAG architectures**: CRAG, Self-RAG, Agentic RAG với multi-step reasoning
- **Full GraphRAG**: NER enhancement, relationship classification, multi-hop queries
- **Optimization**: Multi-level caching, speculative retrieval, continuous evaluation

Đồ án đã hoàn thành mục tiêu xây dựng hệ thống RAG tư vấn pháp luật tài chính Nhật Bản cho người Việt Nam, với khả năng trích dẫn nguồn chính xác và đạt chất lượng đủ tốt cho use case thực tế.
