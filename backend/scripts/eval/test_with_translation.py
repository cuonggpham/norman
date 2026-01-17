"""
Comprehensive test for data gaps with various question styles.
Tests 40+ questions with different phrasings.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.qdrant import get_qdrant_client, get_collection_name, search
from app.services.embedding import EmbeddingService
from app.llm.query_translator import QueryTranslator
from app.llm.openai_provider import OpenAIProvider

# 48 câu hỏi tập trung vào LUẬT TÀI CHÍNH NHẬT BẢN cho người Việt Nam
TEST_QUESTIONS = [
    # === THUẾ THU NHẬP CÁ NHÂN (6 câu) ===
    "Thuế thu nhập cá nhân ở Nhật tính như thế nào?",
    "Người nước ngoài ở Nhật có phải đóng thuế không?",
    "Khai thuế hàng năm (確定申告) ở Nhật làm thế nào?",
    "Các khoản được khấu trừ thuế thu nhập là gì?",
    "Thuế suất thu nhập theo từng mức thu nhập?",
    "Cư trú thuế (税務上の居住者) là gì?",
    
    # === THUẾ TIÊU DÙNG VÀ THUẾ KHÁC (6 câu) ===
    "Thuế tiêu dùng (消費税) ở Nhật là bao nhiêu phần trăm?",
    "Thuế bất động sản (固定資産税) được tính như thế nào?",
    "Thuế quà tặng (贈与税) quy định ra sao?",
    "Thuế thừa kế (相続税) ở Nhật thế nào?",
    "Thuế doanh nghiệp (法人税) là bao nhiêu?",
    "Thuế ô tô và phương tiện giao thông?",
    
    # === NGÂN HÀNG VÀ TÀI KHOẢN (6 câu) ===
    "Người nước ngoài mở tài khoản ngân hàng Nhật thế nào?",
    "Điều kiện mở tài khoản ngân hàng ở Nhật là gì?",
    "Chuyển tiền từ Nhật về Việt Nam làm sao?",
    "Giới hạn chuyển tiền quốc tế là bao nhiêu?",
    "Quy định về giao dịch tiền mặt lớn?",
    "Mở tài khoản tiết kiệm ở Nhật như thế nào?",
    
    # === ĐẦU TƯ VÀ CHỨNG KHOÁN (6 câu) ===
    "Người nước ngoài có thể đầu tư chứng khoán Nhật không?",
    "Thuế trên lợi nhuận đầu tư (キャピタルゲイン税) là bao nhiêu?",
    "Tài khoản NISA là gì và lợi ích của nó?",
    "iDeCo (個人型確定拠出年金) là gì?",
    "Thuế cổ tức (配当税) ở Nhật thế nào?",
    "Mở tài khoản chứng khoán ở Nhật cần gì?",
    
    # === BẢO HIỂM XÃ HỘI VÀ HƯU TRÍ (6 câu) ===
    "Bảo hiểm xã hội Nhật Bản bao gồm những gì?",
    "Bảo hiểm y tế quốc dân (国民健康保険) là gì?",
    "Bảo hiểm hưu trí quốc dân (国民年金) ở Nhật thế nào?",
    "Bảo hiểm hưu trí người lao động (厚生年金) là gì?",
    "Lump-sum withdrawal payment (脱退一時金) cho người rời Nhật là gì?",
    "Điều kiện nhận lương hưu ở Nhật?",
    
    # === TÍN DỤNG VÀ VAY VỐN (6 câu) ===
    "Người nước ngoài vay mua nhà ở Nhật được không?",
    "Điều kiện vay ngân hàng (住宅ローン) ở Nhật là gì?",
    "Thẻ tín dụng cho người nước ngoài ở Nhật?",
    "Lãi suất vay mua nhà ở Nhật là bao nhiêu?",
    "Vay tiêu dùng (消費者ローン) ở Nhật thế nào?",
    "Giới hạn vay đối với người nước ngoài?",
    
    # === TIỀN ĐIỆN TỬ VÀ TÀI SẢN KỸ THUẬT SỐ (4 câu) ===
    "Thuế tiền điện tử (仮想通貨) ở Nhật như thế nào?",
    "Quy định về giao dịch Bitcoin ở Nhật?",
    "Khai báo lợi nhuận từ crypto?",
    "Sàn giao dịch tiền điện tử hợp pháp ở Nhật?",
    
    # === BẤT ĐỘNG SẢN VÀ ĐẦU TƯ NHÀ ĐẤT (4 câu) ===
    "Người nước ngoài có thể mua nhà ở Nhật không?",
    "Thuế khi mua bán bất động sản ở Nhật?",
    "Chi phí mua nhà ở Nhật bao gồm những gì?",
    "Quy định về sở hữu đất đai cho người nước ngoài?",
    
    # === KINH DOANH VÀ KHỞI NGHIỆP (4 câu) ===
    "Thành lập công ty ở Nhật cần những gì?",
    "Thuế doanh nghiệp nhỏ (個人事業主) thế nào?",
    "Visa kinh doanh (経営管理ビザ) yêu cầu gì?",
    "Quy định về vốn tối thiểu thành lập công ty?",
]

def test_comprehensive():
    print("=" * 65)
    print("COMPREHENSIVE DATA GAP TEST (40+ Questions)")
    print("=" * 65)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return
    
    embedding = EmbeddingService(api_key=api_key)
    llm = OpenAIProvider(api_key=api_key)
    translator = QueryTranslator(llm=llm)
    client = get_qdrant_client()
    collection_name = get_collection_name()
    
    results = []
    
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] {q[:40]}...")
        
        # Translate if Vietnamese
        ja_query = translator.translate(q)
        
        # Search
        query_vec = embedding.embed_text(ja_query)
        hits = search(client, query_vec, top_k=5, collection_name=collection_name)
        
        if hits:
            scores = [h["score"] for h in hits]
            laws = set(h["payload"].get("law_title", "?") for h in hits)
            top = scores[0]
            
            if top >= 0.6: status = "✅"
            elif top >= 0.4: status = "⚠️"
            else: status = "❌"
            
            print(f"   {status} Top: {top:.3f} | {list(laws)[0][:15]}...")
            results.append({"q": q, "top": top, "laws": list(laws), "status": status})
        else:
            print("   ❌ NO RESULTS")
            results.append({"q": q, "top": 0, "laws": [], "status": "❌"})
    
    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    
    good = [r for r in results if r["status"] == "✅"]
    warn = [r for r in results if r["status"] == "⚠️"]
    bad = [r for r in results if r["status"] == "❌"]
    
    print(f"\n✅ Good (≥0.6): {len(good)} ({len(good)*100//len(results)}%)")
    print(f"⚠️  Medium (0.4-0.6): {len(warn)} ({len(warn)*100//len(results)}%)")
    print(f"❌ Poor (<0.4): {len(bad)} ({len(bad)*100//len(results)}%)")
    
    # Group by topic
    print("\n📊 BY TOPIC:")
    topics = {
        # 48 câu hỏi luật tài chính (9 topics)
        "Thuế thu nhập": results[0:6],
        "Thuế khác": results[6:12],
        "Ngân hàng": results[12:18],
        "Đầu tư": results[18:24],
        "Bảo hiểm XH": results[24:30],
        "Tín dụng": results[30:36],
        "Tiền điện tử": results[36:40],
        "Bất động sản": results[40:44],
        "Kinh doanh": results[44:48],
    }
    
    for topic, items in topics.items():
        if items:
            avg = sum(r["top"] for r in items) / len(items)
            status = "✅" if avg >= 0.6 else ("⚠️" if avg >= 0.4 else "❌")
            print(f"   {status} {topic}: avg={avg:.3f}")
    
    if bad:
        print("\n❌ POOR QUESTIONS:")
        for r in bad:
            print(f"   - {r['q'][:45]}... ({r['top']:.3f})")

if __name__ == "__main__":
    test_comprehensive()
