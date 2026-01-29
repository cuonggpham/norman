
import sys
import os
import logging

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.reranker import get_bge_reranker

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_reranker():
    print("Loading Reranker...")
    reranker = get_bge_reranker()
    
    query = "thuế thu nhập cá nhân"
    
    documents = [
        {
            "payload": {"text": "Luật Thuế thu nhập cá nhân quy định về đối tượng nộp thuế, thu nhập chịu thuế, thu nhập được miễn thuế, giảm thuế và căn cứ tính thuế thu nhập cá nhân."},
            "score": 0.5
        },
        {
            "payload": {"text": "Luật Giao thông đường bộ quy định về quy tắc giao thông, kết cấu hạ tầng giao thông đường bộ, phương tiện và người tham gia giao thông đường bộ."},
            "score": 0.5
        },
        {
            "payload": {"text": "Quy định về thuế GTGT và hóa đơn chứng từ."},
            "score": 0.5
        }
    ]
    
    print(f"\nQuery: {query}")
    print("Reranking 3 documents...")
    
    results = reranker.rerank(query, documents, top_k=3)
    
    print("\nResults:")
    for i, res in enumerate(results):
        print(f"{i+1}. Score: {res['score']:.4f} | Text: {res['payload']['text'][:50]}...")
        
    # Validation
    top_text = results[0]['payload']['text']
    if "thu nhập cá nhân" in top_text:
        print("\nSUCCESS: Top result matches query context.")
    else:
        print("\nFAILURE: Top result does not match query context.")

if __name__ == "__main__":
    test_reranker()
