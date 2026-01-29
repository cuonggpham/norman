
import time
import logging
from typing import List, Dict, Any
from FlagEmbedding import FlagReranker

logging.basicConfig(level=logging.ERROR) # Less logs
logger = logging.getLogger(__name__)

MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", # Corrected Name
    "jinaai/jina-reranker-v2-base-multilingual"
]

TEST_DATA = [
    {
        "query": "Thời gian làm việc tối đa là bao nhiêu giờ?",
        "docs": [
            "労働時間は、休憩時間を除き、1日について8時間を超えて労働させてはならない。",
            "使用者は、労働者に対し、毎週少くとも1回の休日を与えなければならない。",
            "Người sử dụng lao động không được cho người lao động làm việc quá 8 giờ một ngày.",
            "Mỗi tuần người lao động được nghỉ ít nhất 24 giờ liên tục."
        ]
    },
    {
        "query": "Thủ tục xin nghỉ việc như thế nào?",
        "docs": [
            "Đương sự muốn xin thôi việc phải làm đơn trước ít nhất 30 ngày.",
            "Tiền lương được trả bằng tiền mặt hoặc chuyển khoản.",
            "退職しようとする者は、少なくとも30日前に申し出なければならない。",
            "賃金は、通貨で、直接労働者に、その全額を支払わなければならない。"
        ]
    }
]

def benchmark_model(model_name: str):
    print(f"\n{'='*50}")
    print(f"🚀 Benchmarking: {model_name}")
    
    # 1. Load Time
    start_load = time.time()
    try:
        # trust_remote_code=True for Jina
        reranker = FlagReranker(model_name, use_fp16=False, device='cpu', trust_remote_code=True)
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return

    load_time = time.time() - start_load
    print(f"⏱️  Load Time: {load_time:.2f}s")
    
    # 2. Warmup
    try:
        reranker.compute_score(["test", "test"])
    except Exception as e:
        print(f"⚠️  Warmup warning: {e}")
    
    # 3. Inference Benchmark
    total_docs = 0
    total_time = 0
    
    print("\n🔍 Relevance Scores:")
    for i, case in enumerate(TEST_DATA):
        query = case["query"]
        docs = case["docs"]
        pairs = [[query, doc] for doc in docs]
        
        start_inf = time.time()
        scores = reranker.compute_score(pairs)
        duration = time.time() - start_inf
        
        total_time += duration
        total_docs += len(docs)
        
        print(f"  Query {i+1}: ({duration*1000:.1f}ms / {len(docs)} docs)")
        
        # Handle different output formats (list or single float)
        if not isinstance(scores, list):
            scores = [scores]
            
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_doc = scored_docs[0]
        print(f"    ⭐ Top Result ({top_doc[1]:.4f}): {top_doc[0][:50]}...")

    avg_latency = (total_time / total_docs) * 1000
    print(f"\n⚡ Average Latency: {avg_latency:.2f} ms/doc")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("🖥️  System: CPU Inference Check")
    for model in MODELS:
        benchmark_model(model)
