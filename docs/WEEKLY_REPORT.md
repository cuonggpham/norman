# BÁO CÁO TIẾN ĐỘ HÀNG TUẦN

## Đồ án: Hệ thống RAG Tư vấn Pháp luật Tài chính Nhật Bản cho Người Việt Nam

**Sinh viên thực hiện:** [Họ và tên]  
**GVHD:** [Họ và tên]  
**Thời gian:** Tuần 1 (29/9/2025) - Tuần 16 (18/01/2026)

---

## GIAI ĐOẠN 1: XÁC ĐỊNH ĐỀ TÀI (Week 1-3)

### Week 1 (29/9 - 5/10/2025)

**Đạt được:**

Trong tuần qua, em đã họp với thầy/cô để thảo luận hướng đề tài và bắt đầu tìm hiểu sơ bộ về công nghệ RAG.

- Em có một vài ý tưởng ban đầu như chatbot hỗ trợ du lịch, hệ thống Q&A y tế, nhưng sau khi trao đổi thì thầy/cô gợi ý nên tập trung vào lĩnh vực có dữ liệu chất lượng và ứng dụng thực tế.
- Em đã tìm hiểu về RAG (Retrieval-Augmented Generation) qua các blog và video YouTube, thấy khá hay vì giải quyết được vấn đề hallucination của LLM.
- Em đã brainstorm các lĩnh vực có thể áp dụng: pháp luật, y tế, giáo dục, tài chính. Em nhận thấy pháp luật phù hợp vì yêu cầu độ chính xác cao và cần trích dẫn nguồn.

**Kế hoạch tuần sau:**
- Nghiên cứu kỹ hơn về nhu cầu thực tế của người Việt tại Nhật
- Tìm hiểu các nguồn dữ liệu pháp luật Nhật Bản

---

### Week 2 (6/10 - 12/10/2025)

**Đạt được:**

Trong tuần qua, em đã nghiên cứu về đối tượng người dùng mục tiêu và tìm được nguồn dữ liệu chính thức từ Chính phủ Nhật Bản.

- Em phát hiện cộng đồng người Việt tại Nhật rất đông (~500,000 người) và họ gặp nhiều khó khăn khi tìm hiểu luật tài chính do rào cản ngôn ngữ.
- Em đã tìm được e-Gov API - cổng thông tin pháp luật chính thức. API này cung cấp dữ liệu luật dưới dạng XML, rất chất lượng và được cập nhật thường xuyên.
- Em đã đọc paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" và "Dense Passage Retrieval" để hiểu rõ hơn về lý thuyết RAG.

**Kế hoạch tuần sau:**
- Hoàn thiện đề cương nghiên cứu
- Xác định phạm vi cụ thể: tập trung vào luật tài chính (thuế, bảo hiểm, đầu tư)

---

### Week 3 (13/10 - 19/10/2025)

**Đạt được:**

Trong tuần qua, em đã hoàn thành đề cương nghiên cứu và được thầy/cô duyệt đề tài chính thức.

- Đề tài: "Hệ thống RAG tư vấn pháp luật tài chính Nhật Bản cho người Việt Nam".
- Phạm vi tư vấn bao gồm: thuế thu nhập, thuế cư trú, khai thuế cuối năm, bảo hiểm y tế, lương hưu, và các chương trình đầu tư như NISA, iDeCo.
- Em đã lập kế hoạch tổng thể cho đồ án với các phase: thu thập dữ liệu, xây dựng vector search, tích hợp LLM, và đánh giá hệ thống.

**Kế hoạch tuần sau:**
- Bắt đầu nghiên cứu lý thuyết về RAG pipeline
- Tìm hiểu về vector embedding và semantic search

---

## GIAI ĐOẠN 2: HỌC VỀ RAG (Week 4-8)

### Week 4 (20/10 - 26/10/2025)

**Đạt được:**

Trong tuần qua, em đã học về kiến trúc RAG cơ bản và tìm hiểu về vector embedding.

- Em học qua khóa của DeepLearning.AI và hiểu được flow: nhận query → chuyển thành vector → tìm kiếm văn bản liên quan → đưa vào LLM sinh câu trả lời.
- Em đã tìm hiểu về vector embedding - cách văn bản được chuyển thành vector số trong không gian đa chiều để tính toán độ tương đồng ngữ nghĩa.
- Em đã thử nghiệm OpenAI Embedding API với câu tiếng Việt và tiếng Nhật, thấy model text-embedding-3-large cho kết quả tốt với đa ngôn ngữ.

**Kế hoạch tuần sau:**
- Nghiên cứu về Vector Database (Qdrant, Pinecone, Weaviate)
- Tìm hiểu về chunking strategies

---

### Week 5 (27/10 - 2/11/2025)

**Đạt được:**

Trong tuần qua, em đã so sánh các Vector Database và học về chunking strategies cho văn bản pháp luật.

- Em đã so sánh Qdrant, Pinecone, Weaviate, Milvus và chọn Qdrant Cloud Free Tier vì có hybrid search tích hợp sẵn và documentation tốt.
- Em học về các chiến lược chunking: fixed-size, sentence-level, semantic chunking. Với văn bản pháp luật, em nghĩ nên chunk theo Điều/Khoản để giữ nguyên context.
- Em đọc paper về Dense Passage Retrieval và hiểu sự khác biệt giữa bi-encoder (nhanh) và cross-encoder (chính xác hơn).

**Kế hoạch tuần sau:**
- Tìm hiểu về Hybrid Search (kết hợp vector search và keyword search)
- Nghiên cứu về reranking

---

### Week 6 (3/11 - 9/11/2025)

**Đạt được:**

Trong tuần qua, em đã nghiên cứu về Hybrid Search và Reranking để cải thiện chất lượng retrieval.

- Em hiểu về Hybrid Search - kết hợp vector search (hiểu ngữ nghĩa) và BM25 (khớp từ khóa). Qdrant hỗ trợ Reciprocal Rank Fusion để gộp kết quả.
- Em nghiên cứu về Reranking (two-stage retrieval): dùng bi-encoder lấy nhanh nhiều kết quả, sau đó cross-encoder xếp hạng lại chính xác hơn. Em tìm được model BGE-reranker-large chạy được trên CPU.
- Em thử xây dựng pipeline RAG đơn giản với LangChain nhưng thấy hơi nặng nề, quyết định sẽ tự implement.

**Kế hoạch tuần sau:**
- Nghiên cứu về Query Translation (cross-lingual retrieval)
- Tìm hiểu về LangGraph cho agentic RAG

---

### Week 7 (10/11 - 16/11/2025)

**Đạt được:**

Trong tuần qua, em đã học về Query Translation và các phương pháp self-correction trong RAG.

- Em học về Query Translation để query tiếng Việt tìm được văn bản tiếng Nhật. Em chọn dịch query vì corpus 233 luật quá lớn để dịch hết.
- Em tìm hiểu LangGraph - framework cho agentic workflow. Hay ở chỗ có thể tạo nhánh điều kiện và vòng lặp self-correction.
- Em đọc paper về Self-RAG và CRAG: ý tưởng để LLM tự đánh giá documents retrieved, nếu không đủ relevant thì viết lại query và tìm lại.

**Kế hoạch tuần sau:**
- Nghiên cứu về GraphRAG và Knowledge Graph
- Hoàn thành phần lý thuyết, chuẩn bị bắt tay vào implementation

---

### Week 8 (17/11 - 23/11/2025)

**Đạt được:**

Trong tuần qua, em đã học về GraphRAG và hoàn thành phần lý thuyết cho báo cáo.

- Em học về GraphRAG - kết hợp Knowledge Graph với RAG để xử lý câu hỏi cần reasoning qua nhiều bước (multi-hop QA).
- Em tìm hiểu Neo4j và làm quen với Cypher query language. Em thiết kế schema với các node Law, Chapter, Article, Paragraph.
- Em tổng hợp kiến thức đã học và viết outline cho chương Cơ sở lý thuyết của báo cáo.

**Kế hoạch tuần sau:**
- Bắt đầu phase implementation: thu thập dữ liệu từ e-Gov API
- Setup project structure với FastAPI backend

---

## GIAI ĐOẠN 3: NGHIÊN CỨU + TRIỂN KHAI (Week 9-16)

### Week 9 (24/11 - 30/11/2025)

**Đạt được:**

Trong tuần qua, em đã setup project và thu thập dữ liệu luật từ e-Gov API.

- Em setup cấu trúc project với FastAPI, tổ chức code theo các thư mục riêng biệt cho API, services, LLM integration và data pipeline.
- Em viết script download dữ liệu từ e-Gov API, phải xử lý rate limiting (delay 1.2s giữa các request) để không bị server chặn.
- Em download được **233 văn bản luật** về thuế, bảo hiểm xã hội, lao động dưới dạng XML, tổng dung lượng ~150MB.

**Kế hoạch tuần sau:**
- Implement XML parser để chuyển XML → JSON
- Implement chunking logic

---

### Week 10 (1/12 - 7/12/2025)

**Đạt được:**

Trong tuần qua, em đã parse dữ liệu XML và thực hiện chunking cho toàn bộ văn bản luật.

- Em viết code parse XML của e-Gov thành JSON. Cấu trúc XML khá phức tạp với nhiều tầng lồng nhau nên mất khá thời gian để xử lý hết các trường hợp.
- Em implement chunking với Paragraph (Khoản) làm đơn vị, vì pháp luật Nhật trích dẫn theo format "第X条 第Y項" và mỗi khoản thường chứa ý nghĩa hoàn chỉnh.
- Em thêm context enrichment: gắn tên luật và tên điều vào đầu mỗi chunk. Kết quả: **15,629 chunks** với size trung bình ~800 ký tự.

**Kế hoạch tuần sau:**
- Implement embedding pipeline
- Upload vectors lên Qdrant Cloud

---

### Week 11 (8/12 - 14/12/2025)

**Đạt được:**

Trong tuần qua, em đã tạo embeddings và upload toàn bộ vectors lên Qdrant Cloud.

- Em viết script tạo embeddings với text-embedding-3-large (3072 dimensions), xử lý theo batch 100 chunks một lần.
- Em thêm tính năng resume để tiếp tục từ vị trí dừng nếu bị lỗi, và retry với exponential backoff cho rate limiting.
- Em embed toàn bộ 15,629 chunks trong ~30 phút, chi phí ~$2.
- Em upload lên Qdrant Cloud với cả dense vectors và sparse vectors (BM25) để hỗ trợ hybrid search.

**Kế hoạch tuần sau:**
- Implement basic RAG pipeline
- Implement query translation

---

### Week 12 (15/12 - 21/12/2025)

**Đạt được:**

Trong tuần qua, em đã xây dựng RAG pipeline cơ bản và các API endpoints.

- Em xây dựng pipeline: nhận query tiếng Việt → dịch sang tiếng Nhật → tạo embedding → tìm kiếm → lọc theo score → sinh câu trả lời.
- Em implement query translation với GPT-4o-mini và thêm query expansion để sinh các biến thể câu hỏi tăng khả năng tìm được kết quả.
- Em implement các API endpoints: chat với RAG, search trực tiếp, và health check. Test thử một vài câu hỏi và kết quả khá khả quan!

**Kế hoạch tuần sau:**
- Implement hybrid search và reranking
- Bắt đầu xây dựng frontend UI

---

### Week 13 (22/12 - 28/12/2025)

**Đạt được:**

Trong tuần qua, em đã enable Hybrid Search và implement Reranker.

- Em enable Hybrid Search kết hợp vector search và BM25 với Reciprocal Rank Fusion. Kết quả tốt hơn hẳn với các query có từ khóa cụ thể.
- Em implement BGE Reranker, cải thiện precision +10-20% nhưng chạy trên CPU khá chậm (20-40s). Em để reranker là tính năng optional, mặc định tắt.
- Tuần này nghỉ lễ Giáng sinh nên tiến độ hơi chậm.

**Kế hoạch tuần sau:**
- Implement LangGraph Agent với self-correction
- Bắt đầu xây dựng frontend UI

---

### Week 14 (29/12/2025 - 4/1/2026)

**Đạt được:**

Trong tuần qua, em đã xây dựng LangGraph Agent và bắt đầu làm frontend.

- Em xây dựng Agent với 6 nodes: dịch query, tìm kiếm, đánh giá relevance, rerank, sinh câu trả lời, và viết lại query.
- Em implement self-correction: nếu documents relevant ít hơn 2, hệ thống tự viết lại query và tìm lại (tối đa 2 lần retry).
- Em bắt đầu xây dựng frontend với React + Vite, thiết kế giao diện chat với dark theme.
- Tuần này nghỉ Tết Dương lịch nên em chỉ code được ít.

**Kế hoạch tuần sau:**
- Implement GraphRAG với Neo4j
- Hoàn thiện frontend UI

---

### Week 15 (5/1 - 11/1/2026)

**Đạt được:**

Trong tuần qua, em đã xây dựng Knowledge Graph và tích hợp GraphRAG vào hệ thống.

- Em cài Neo4j và build Knowledge Graph từ dữ liệu luật, tạo được ~25,000 nodes (Law, Chapter, Article, Paragraph) với các relationships thể hiện cấu trúc phân cấp.
- Em implement các chức năng query đồ thị: tìm điều luật theo tên/số, tìm điều liên quan (multi-hop), tìm theo từ khóa.
- Em xây dựng Query Router phân loại câu hỏi để quyết định dùng vector search hay graph search.
- Em hoàn thiện frontend với source cards có thể expand, loading states, responsive design.

**Kế hoạch tuần sau:**
- Implement RAGAS evaluation
- Hoàn thiện báo cáo

---

### Week 16 (12/1 - 18/1/2026)

**Đạt được:**

Trong tuần qua, em đã đánh giá hệ thống với RAGAS và hoàn thiện báo cáo.

- Em implement evaluation với RAGAS framework, tạo bộ test 20+ câu hỏi về thuế, bảo hiểm, lao động cùng ground truth.
- Kết quả: Context Precision 0.72, Context Recall 0.68, Faithfulness 0.85, Answer Relevancy 0.78. Điểm Faithfulness cao cho thấy câu trả lời dựa trên context thực tế.
- Em viết documentation: báo cáo chi tiết, roadmap, và hướng dẫn Quick Start.
- Em deploy demo local và mời một số người dùng thử. Feedback tích cực về giao diện và độ chính xác của câu trả lời.

**Kế hoạch tiếp theo (nếu có thêm thời gian):**
- Deploy lên cloud (Vercel/Railway)
- Thêm conversation memory cho multi-turn chat
- Mở rộng data coverage với thêm các loại luật khác

---

## TỔNG KẾT

### Thành quả đạt được:
- ✅ Xây dựng thành công hệ thống RAG hoàn chỉnh với 233 văn bản luật, 15,629 chunks
- ✅ Implement Hybrid Search kết hợp vector search và keyword search
- ✅ Implement LangGraph Agent với khả năng self-correction
- ✅ Implement GraphRAG với Neo4j Knowledge Graph
- ✅ Đánh giá hệ thống với RAGAS framework, đạt Faithfulness 0.85

### Khó khăn gặp phải:
- Cấu trúc XML của e-Gov khá phức tạp, em mất khá nhiều thời gian để parse đúng hết các trường hợp
- Reranker chạy trên CPU rất chậm do không có GPU, em phải để thành tính năng optional
- Cross-lingual retrieval đôi khi vẫn bị miss do việc dịch query chưa chính xác với thuật ngữ pháp lý

### Bài học kinh nghiệm:
- Em nên test với dataset nhỏ trước khi chạy trên toàn bộ data để tiết kiệm thời gian và chi phí API
- Documentation rất quan trọng, em nên viết song song với code để không quên các chi tiết

---

**Sinh viên thực hiện**

[Chữ ký]
