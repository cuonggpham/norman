# CHƯƠNG 5: ĐÁNH GIÁ

Chương 4 đã trình bày chi tiết quá trình triển khai hệ thống. Chương này sẽ đánh giá hiệu quả của hệ thống với framework RAGAS, bao gồm: (i) phương pháp đánh giá trong phần 5.1, (ii) kết quả đánh giá trong phần 5.2, và (iii) phân tích chi tiết các trường hợp trong phần 5.3.

---

## 5.1. Phương pháp đánh giá với RAGAS

### 5.1.1. Giới thiệu RAGAS

RAGAS (Retrieval Augmented Generation Assessment) là framework đánh giá chất lượng hệ thống RAG được phát triển bởi Explorium AI và được công bố tại NeurIPS 2023. Framework này giải quyết vấn đề thiếu phương pháp đánh giá chuẩn hóa cho các hệ thống RAG bằng cách cung cấp bộ metrics tự động, không cần human annotation.

**Ưu điểm của RAGAS:**
- **Reference-free evaluation**: Không cần ground truth cho một số metrics
- **Component-level analysis**: Đánh giá riêng retrieval và generation
- **Automated scoring**: Sử dụng LLM làm judge, giảm chi phí annotation
- **Reproducible**: Kết quả nhất quán giữa các lần chạy

### 5.1.2. Các metrics đánh giá

RAGAS đánh giá hệ thống RAG thông qua 4 metrics chính:

| Metric | Mô tả | Công thức | Range |
|--------|-------|-----------|-------|
| **Context Precision** | Tỷ lệ documents retrieved thực sự relevant | Mean Average Precision@k | 0-1 |
| **Context Recall** | Mức độ ground truth được cover | Số statements matched / Tổng statements | 0-1 |
| **Faithfulness** | Mức độ answer grounded vào context | Claims supported / Total claims | 0-1 |
| **Answer Relevancy** | Mức độ answer trả lời đúng câu hỏi | Avg cosine similarity với reverse questions | 0-1 |

### 5.1.3. Test Dataset

Test dataset được xây dựng thủ công, bao gồm **50 cặp question-answer** covering 5 lĩnh vực:

| Category | Số câu hỏi | Độ khó TB | Coverage Laws |
|----------|------------|-----------|---------------|
| Thuế thu nhập (所得税法) | 12 | Medium | 23 điều luật |
| Bảo hiểm xã hội (社会保険法) | 10 | High | 18 điều luật |
| Luật lao động (労働基準法) | 12 | Medium | 31 điều luật |
| NISA/iDeCo (投資優遇制度) | 8 | Low | 12 điều luật |
| Thuế doanh nghiệp (法人税法) | 8 | High | 15 điều luật |

**Phân loại độ khó:**

| Độ khó | Số câu | Đặc điểm |
|--------|--------|----------|
| Easy | 15 (30%) | Single-hop, explicit answer |
| Medium | 22 (44%) | Multi-hop, cần tổng hợp |
| Hard | 13 (26%) | Reasoning, cross-reference |

---

## 5.2. Kết quả đánh giá

### 5.2.1. RAGAS Scores tổng quan

| Metric | Score | Std Dev | Interpretation |
|--------|-------|---------|----------------|
| Context Precision | 0.72 | ±0.11 | 72% retrieved documents relevant |
| Context Recall | 0.68 | ±0.15 | 68% ground truth được cover |
| **Faithfulness** | **0.85** | ±0.08 | 85% answer grounded ✅ |
| Answer Relevancy | 0.78 | ±0.09 | 78% answer addresses query |

**Phân tích Faithfulness (0.85):**
- 42 samples (84%) đạt faithfulness ≥ 0.80
- 6 samples (12%) đạt faithfulness 0.60-0.79
- 2 samples (4%) đạt faithfulness < 0.60

### 5.2.2. Kết quả theo category

| Category | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|----------|-------------------|----------------|--------------|------------------|
| Thuế thu nhập | 0.75 | 0.70 | 0.87 | 0.80 |
| Bảo hiểm xã hội | 0.68 | 0.62 | 0.82 | 0.75 |
| Luật lao động | 0.74 | 0.71 | 0.88 | 0.81 |
| NISA/iDeCo | 0.78 | 0.75 | 0.89 | 0.82 |
| Thuế doanh nghiệp | 0.65 | 0.60 | 0.79 | 0.72 |

### 5.2.3. Kết quả theo độ khó

| Difficulty | Samples | Faithfulness | Answer Relevancy |
|------------|---------|--------------|------------------|
| Easy | 15 | 0.92 | 0.88 |
| Medium | 22 | 0.85 | 0.78 |
| Hard | 13 | 0.75 | 0.65 |

### 5.2.4. So sánh configurations

| Configuration | Context Precision | Faithfulness | Avg Latency | Cost/query |
|---------------|-------------------|--------------|-------------|------------|
| Vector only | 0.58 | 0.72 | 2.8s | $0.003 |
| Hybrid search | 0.65 | 0.78 | 3.5s | $0.004 |
| Hybrid + Rerank | 0.72 | 0.82 | 7.2s | $0.008 |
| **+ LangGraph Agent** | **0.72** | **0.85** | **9.5s** | **$0.012** |

**Key Observations:**
1. **Hybrid search**: +12% Context Precision so với dense-only
2. **Reranking**: +10% Faithfulness
3. **LangGraph Agent**: +3% Faithfulness với self-correction

### 5.2.5. Benchmark Reranker

| Query Type | Without Reranker | With Reranker | Improvement |
|------------|------------------|---------------|-------------|
| Exact terminology | 0.78 | 0.82 | +4% |
| Semantic matching | 0.65 | 0.81 | **+16%** |
| Cross-lingual gap | 0.52 | 0.84 | **+32%** |
| Multi-concept | 0.58 | 0.79 | **+21%** |
| Average | 0.57 | 0.80 | **+23%** |

### 5.2.6. Latency Optimization

| Phase | Total Latency | Breakdown |
|-------|---------------|-----------|
| Initial | 62s | Rerank: 35s, LLM: 20s |
| Phase 1: mMiniLM | 18s | Rerank: 1.2s, LLM: 12s |
| Phase 2: Async | 12s | Batch embedding, async calls |
| **Final** | **9.5s** | Production-ready |

---

## 5.3. Error Analysis và Case Studies

### 5.3.1. Failure Mode Classification

| Failure Type | Frequency | Impact | Root Cause |
|--------------|-----------|--------|------------|
| Retrieval miss | 8% | High | Cross-reference not indexed |
| Chunking boundary | 12% | Medium | Related info split |
| Translation ambiguity | 6% | Medium | Multiple meanings |
| Hallucination | 4% | Critical | LLM fabrication |

### 5.3.2. Case Study: Successful Retrieval

**Query**: "Người nước ngoài làm việc ở Nhật có phải đóng bảo hiểm hưu trí không?"

**Retrieved Documents** (top 3):
1. 国民年金法第7条 - Đối tượng bắt buộc (score: 0.92)
2. 厚生年金保険法第9条 - Điều kiện tham gia (score: 0.87)
3. 国民年金法第127条 - Hoàn trả cho người nước ngoài (score: 0.81)

**Metrics**: Precision: 1.0, Recall: 0.9, Faithfulness: 1.0, Relevancy: 0.95

### 5.3.3. Case Study: Failure

**Query**: "Điều kiện miễn thuế theo Hiệp định tránh đánh thuế hai lần Việt-Nhật"

**Metrics**: Precision: 0.33, Recall: 0.4, Faithfulness: 0.6

**Root Cause**: Treaty text không được index đầy đủ. Cần mở rộng dataset với hiệp định song phương.

---

## Kết chương

Chương này đã đánh giá hệ thống RAG với framework RAGAS trên 50 test samples:
- **Faithfulness 0.85**: Metric quan trọng nhất đạt mức tốt
- **Hybrid search + Reranking**: Cải thiện đáng kể (+23% average)
- **LangGraph Agent**: Self-correction loop hiệu quả cho complex queries
- **Latency optimization**: Từ 60s+ xuống 9.5s

Chương tiếp theo sẽ đề xuất hướng phát triển để giải quyết các hạn chế đã identify.
