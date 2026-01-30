# CHƯƠNG 1: TỔNG QUAN

## 1.1. Đặt vấn đề

Theo thống kê từ Bộ Tư pháp Nhật Bản tính đến cuối năm 2024, có hơn 500.000 người Việt Nam đang sinh sống và làm việc tại Nhật Bản, đứng thứ hai chỉ sau cộng đồng Trung Quốc. Nhu cầu tìm hiểu các quy định pháp luật tài chính của nước sở tại là rất lớn, tuy nhiên rào cản ngôn ngữ và khả năng tiếp cận đang là vấn đề nghiêm trọng.

Vấn đề cốt lõi không phải là thiếu thông tin, mà nằm ở khả năng tiếp cận. Các văn bản pháp luật Nhật Bản được viết bằng ngôn ngữ pháp lý phức tạp với nhiều thuật ngữ chuyên ngành như 確定申告 (khai thuế cuối năm), 源泉徴収 (khấu trừ tại nguồn), hay 厚生年金 (bảo hiểm hưu trí). Ngay cả người Nhật bản địa cũng cần thời gian để hiểu những thuật ngữ này, đối với người nước ngoài thì đây gần như là một rào cản không nhỏ.

Câu hỏi nghiên cứu đặt ra là: liệu có thể xây dựng một hệ thống cho phép người Việt đặt câu hỏi bằng tiếng mẹ đẻ, tìm kiếm trong kho văn bản tiếng Nhật gốc theo ngữ nghĩa, và nhận được câu trả lời chính xác kèm trích dẫn nguồn pháp lý?

---

## 1.2. Mục tiêu và phạm vi đề tài

### 1.2.1. Phân tích các phương pháp tiếp cận

Trước khi quyết định hướng tiếp cận, ba phương án chính đã được phân tích:

**Phương án 1: Fine-tuning LLM trên tập dữ liệu pháp luật**

Hướng này có ưu điểm là sau khi fine-tune, mô hình có thể trả lời các câu hỏi pháp lý mà không cần retrieval step. Tuy nhiên, có hai nhược điểm đáng kể:
- Chi phí tính toán cao: Fine-tuning một mô hình như Llama-2-7B hay GPT-3.5 yêu cầu GPU với VRAM lớn và thời gian training đáng kể.
- "Đóng băng" kiến thức: Mô hình sau khi fine-tune sẽ bị đóng băng kiến thức tại thời điểm training. Trong lĩnh vực pháp lý, luật có thể được sửa đổi, bổ sung hàng năm, việc phải re-train mỗi khi có thay đổi là không khả thi.

**Phương án 2: Hệ thống tìm kiếm dựa trên từ khóa (BM25)**

Đây là cách tiếp cận truyền thống với độ phức tạp implementation thấp. BM25 sử dụng Term Frequency-Inverse Document Frequency kết hợp với document length normalization để ranking. Công thức BM25 có dạng:

$$score(D,Q) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}$$

Nhược điểm của BM25 là thiếu khả năng hiểu ngữ nghĩa. Khi người dùng hỏi "làm thêm giờ tối đa bao nhiêu tiếng", hệ thống không thể kết nối với thuật ngữ chính xác trong luật như "法定労働時間" hay "時間外労働の上限規制" do không có token overlap.

**Phương án 3: RAG (Retrieval-Augmented Generation)**

RAG kết hợp ưu điểm của cả hai phương pháp trên. Retrieval step sử dụng dense embedding để tìm kiếm theo ngữ nghĩa, sau đó LLM sinh câu trả lời dựa trên context được retrieve. Điểm mấu chốt là mọi câu trả lời đều được grounding vào dữ liệu thực, giảm thiểu hiện tượng hallucination. Kiến thức có thể được cập nhật chỉ bằng cách re-index mà không cần re-train model.

### 1.2.2. Lựa chọn RAG

Phương pháp RAG được chọn vì phù hợp nhất với yêu cầu của bài toán:
- Cần trích dẫn nguồn chính xác
- Cần khả năng cập nhật dữ liệu linh hoạt
- Cần xử lý cross-lingual query (Việt → Nhật)

### 1.2.3. Phạm vi và nguồn dữ liệu

Đồ án tập trung vào bốn lĩnh vực pháp luật tài chính mà người Việt tại Nhật thường xuyên cần tra cứu:

| Lĩnh vực | Văn bản pháp luật |
|----------|-------------------|
| **Thuế (税金)** | 所得税法 (Luật Thuế thu nhập), 地方税法 (Luật Thuế địa phương), các quy định về 確定申告 |
| **Bảo hiểm xã hội (社会保険)** | 健康保険法 (Luật Bảo hiểm y tế), 厚生年金保険法 (Luật Bảo hiểm hưu trí), 雇用保険法 (Luật Bảo hiểm thất nghiệp) |
| **Đầu tư và tiết kiệm** | NISA (少額投資非課税制度), iDeCo (個人型確定拠出年金) |
| **Lao động** | 労働基準法 (Luật Tiêu chuẩn Lao động), 労働契約法 (Luật Hợp đồng Lao động) |

Nguồn dữ liệu được lấy từ e-Gov Laws API, cổng thông tin pháp luật chính thức của Chính phủ Nhật Bản. Sau quá trình thu thập và lọc, tổng cộng **431 văn bản luật** với **206.014 chunks** được index vào hệ thống.

---

## 1.3. Định hướng giải pháp

### 1.3.1. Hướng tiếp cận

Hệ thống được thiết kế theo kiến trúc RAG (Retrieval-Augmented Generation) với các đặc điểm chính:

- **Hybrid Search**: Kết hợp dense embedding (semantic search) và sparse embedding (BM25) để tối ưu hóa khả năng truy vấn.
- **Two-Stage Retrieval**: Bi-encoder cho recall, Cross-encoder cho precision.
- **LangGraph Agent**: Self-correction loop để xử lý các query phức tạp.
- **Cross-lingual Processing**: Xử lý query tiếng Việt, tìm kiếm trong văn bản tiếng Nhật.

### 1.3.2. Mô tả ngắn gọn giải pháp

Hệ thống bao gồm hai luồng xử lý chính:
1. **Luồng offline (Data Ingestion)**: Thu thập văn bản luật từ e-Gov API → Parse XML → Chunking theo paragraph → Embedding → Index vào Qdrant Cloud.
2. **Luồng online (Query Processing)**: Nhận query tiếng Việt → Dịch và mở rộng query → Hybrid search → Reranking → LLM sinh câu trả lời với citations.

### 1.3.3. Đóng góp chính

- Xây dựng pipeline end-to-end cho việc thu thập, xử lý và index 431 văn bản luật Nhật Bản.
- Thiết kế chiến lược chunking phù hợp với cấu trúc văn bản pháp lý (paragraph-level với context enrichment).
- Triển khai hybrid search với RRF fusion và two-stage retrieval với cross-encoder reranking.
- Phát triển LangGraph agent với self-correction loop để cải thiện chất lượng câu trả lời.
- Đạt **Faithfulness 0.85** trên RAGAS evaluation, cho thấy câu trả lời được grounding tốt vào context.

---

## 1.4. Bố cục đồ án

Phần còn lại của báo cáo đồ án tốt nghiệp này được tổ chức như sau:

**Chương 2: Cơ sở lý thuyết** trình bày các kiến thức nền tảng về Retrieval-Augmented Generation (RAG), bao gồm text embedding, vector database, hybrid search, reranking, và large language models. Chương này cũng giới thiệu về knowledge graph và cách tích hợp vào hệ thống RAG.

**Chương 3: Thiết kế hệ thống** mô tả kiến trúc tổng thể của hệ thống, bao gồm thiết kế luồng xử lý dữ liệu offline và online, thiết kế hệ thống truy vấn với hybrid search và reranking, thiết kế giao diện người dùng, và thiết kế cơ sở dữ liệu vector và knowledge graph.

**Chương 4: Triển khai hệ thống** trình bày chi tiết quá trình triển khai từng thành phần, bao gồm thu thập dữ liệu từ e-Gov API, XML parsing, chunking strategy, embedding với OpenAI, hybrid search với Qdrant Cloud, LangGraph agent với self-correction, và tối ưu hiệu năng.

**Chương 5: Đánh giá** thực hiện đánh giá hệ thống với framework RAGAS, phân tích các chỉ số Context Precision, Context Recall, Faithfulness, và Answer Relevancy. Chương này cũng thảo luận về các hạn chế hiện tại và đề xuất hướng phát triển trong tương lai.

---

## Kết chương

Chương này đã giới thiệu tổng quan về bài toán xây dựng hệ thống RAG tư vấn pháp luật tài chính Nhật Bản cho người Việt Nam. Vấn đề cốt lõi được xác định là rào cản ngôn ngữ và khả năng tiếp cận thông tin pháp lý. Sau khi phân tích ba hướng tiếp cận (fine-tuning, BM25, RAG), phương pháp RAG được chọn vì khả năng trích dẫn nguồn chính xác, cập nhật linh hoạt, và xử lý cross-lingual query.

Hệ thống được thiết kế với hybrid search, two-stage retrieval, và LangGraph agent để đảm bảo chất lượng câu trả lời. Phạm vi dữ liệu bao gồm 431 văn bản luật từ 4 lĩnh vực chính: thuế, bảo hiểm xã hội, đầu tư tiết kiệm, và lao động.

Chương tiếp theo sẽ trình bày cơ sở lý thuyết về các công nghệ và kỹ thuật được sử dụng trong hệ thống.
