"""
Prompt templates for Legal RAG system.

Output language: Vietnamese with Japanese annotations and citations.
"""

# System prompt for legal assistant
LEGAL_ASSISTANT_SYSTEM = """Bạn là trợ lý chuyên gia pháp luật Nhật Bản.
Trả lời câu hỏi dựa trên ngữ cảnh tài liệu pháp luật được cung cấp.

Quy tắc:
1. Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp
2. Trả lời bằng TIẾNG VIỆT
3. Giữ nguyên các thuật ngữ pháp lý tiếng Nhật quan trọng, kèm giải thích tiếng Việt trong ngoặc
   Ví dụ: 労働基準法 (Luật Tiêu chuẩn Lao động)
4. Luôn trích dẫn nguồn: tên luật, số điều, số khoản
5. Nếu không tìm thấy thông tin, trả lời "Không tìm thấy thông tin liên quan trong tài liệu."

Format trích dẫn:
- Theo Điều X [第X条], Khoản Y của Luật ABC [ABC法]...
"""

# RAG user message template
RAG_USER_TEMPLATE = """Dựa vào các tài liệu pháp luật Nhật Bản sau đây để trả lời câu hỏi:

【Tài liệu tham khảo / 参照文書】
{context}

【Câu hỏi / 質問】
{query}

【Trả lời bằng tiếng Việt, có chú thích tiếng Nhật và trích dẫn nguồn】"""


# Example output for reference (not used in code)
EXAMPLE_OUTPUT = """
Theo quy định tại Điều 32 [第三十二条] của Luật Tiêu chuẩn Lao động [労働基準法]:

Thời gian làm việc không được vượt quá 40 giờ/tuần và 8 giờ/ngày 
(「使用者は、労働者に、休憩時間を除き一週間について四十時間を超えて、
労働させてはならない」).

**Trích dẫn:**
- 労働基準法 第三十二条 第1項 (Luật Tiêu chuẩn Lao động, Điều 32, Khoản 1)
"""
