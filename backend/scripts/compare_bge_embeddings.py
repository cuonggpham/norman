#!/usr/bin/env python3
"""
BGE Embedding Model Comparison Script.

Compares retrieval quality between:
- BAAI/bge-base-en-v1.5 (較小, 768d)
- BAAI/bge-m3 (multilingual, 1024d) 

Usage:
    python scripts/compare_bge_embeddings.py
    python scripts/compare_bge_embeddings.py --samples 5 --output results/bge_comparison.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test queries with expected relevant content (Japanese legal domain)
TEST_QUERIES = [
    {
        "query_vi": "Thời gian làm việc tối đa mỗi tuần theo luật lao động là bao nhiêu?",
        "query_ja": "労働基準法における週の労働時間の上限は何時間ですか?",
        "expected_keywords": ["労働時間", "40時間", "週", "労働基準法"],
    },
    {
        "query_vi": "Quy định về nghỉ phép năm có lương",
        "query_ja": "年次有給休暇の規定",
        "expected_keywords": ["年次有給休暇", "有給", "休暇"],
    },
    {
        "query_vi": "Tiền lương làm thêm giờ tính như thế nào?",
        "query_ja": "時間外労働の賃金はどのように計算しますか?",
        "expected_keywords": ["時間外", "割増賃金", "25%", "残業"],
    },
    {
        "query_vi": "Bảo hiểm tai nạn lao động chi trả những gì?",
        "query_ja": "労災保険はどのような給付がありますか?",
        "expected_keywords": ["労災保険", "療養補償", "休業補償", "給付"],
    },
    {
        "query_vi": "Luật phúc lợi trẻ em quy định gì?",
        "query_ja": "児童福祉法の規定は何ですか?",
        "expected_keywords": ["児童福祉法", "児童", "保育"],
    },
]


class BGEEmbedding:
    """BGE Embedding wrapper using FlagEmbedding."""
    
    def __init__(self, model_name: str, use_fp16: bool = True):
        """
        Initialize BGE embedding model.
        
        Args:
            model_name: Full HuggingFace model name
            use_fp16: Use half precision for faster inference
        """
        from FlagEmbedding import FlagModel
        
        logger.info(f"Loading embedding model: {model_name}")
        start = time.time()
        
        self.model_name = model_name
        self.model = FlagModel(
            model_name,
            use_fp16=use_fp16,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章："
        )
        
        elapsed = time.time() - start
        logger.info(f"✓ Model loaded in {elapsed:.1f}s")
    
    def embed(self, text: str) -> list[float]:
        """Embed single text."""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.encode("test").shape[0]


class BGEM3Embedding:
    """BGE-M3 Embedding wrapper (multilingual, supports dense + sparse)."""
    
    def __init__(self, use_fp16: bool = True):
        """
        Initialize BGE-M3 model.
        
        Args:
            use_fp16: Use half precision
        """
        from FlagEmbedding import BGEM3FlagModel
        
        model_name = "BAAI/bge-m3"
        logger.info(f"Loading BGE-M3 model: {model_name}")
        start = time.time()
        
        self.model_name = model_name
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
        )
        
        elapsed = time.time() - start
        logger.info(f"✓ BGE-M3 loaded in {elapsed:.1f}s")
    
    def embed(self, text: str) -> list[float]:
        """Embed single text (dense only)."""
        result = self.model.encode(text, return_dense=True)
        return result["dense_vecs"].tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (dense only)."""
        result = self.model.encode(texts, return_dense=True)
        return result["dense_vecs"].tolist()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        result = self.model.encode("test", return_dense=True)
        return result["dense_vecs"].shape[0]


def compute_retrieval_metrics(
    retrieved_texts: list[str],
    expected_keywords: list[str],
) -> dict:
    """
    Compute simple retrieval metrics.
    
    Args:
        retrieved_texts: List of retrieved document texts
        expected_keywords: Keywords that should appear in good results
        
    Returns:
        Dict with hit rate and keyword coverage
    """
    # Combine all retrieved texts
    combined_text = " ".join(retrieved_texts).lower()
    
    # Count keyword hits
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    keyword_coverage = keyword_hits / len(expected_keywords) if expected_keywords else 0
    
    return {
        "keyword_hits": keyword_hits,
        "total_keywords": len(expected_keywords),
        "keyword_coverage": keyword_coverage,
        "retrieved_count": len(retrieved_texts),
    }


def search_with_embedding(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int = 5,
) -> list[dict]:
    """
    Search Qdrant with given embedding vector.
    
    Args:
        client: Qdrant client
        collection_name: Name of collection to search
        query_vector: Query embedding vector
        top_k: Number of results
        
    Returns:
        List of search results with text and score
    """
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
    )
    
    return [
        {
            "text": r.payload.get("text", ""),
            "score": r.score,
            "law_title": r.payload.get("law_title", ""),
            "article_title": r.payload.get("article_title", ""),
        }
        for r in results
    ]


def evaluate_model(
    model_name: str,
    embedding_model,
    client: QdrantClient,
    collection_name: str,
    queries: list[dict],
    top_k: int = 5,
) -> dict:
    """
    Evaluate a single embedding model.
    
    Args:
        model_name: Name for logging/display
        embedding_model: Embedding model instance
        client: Qdrant client
        collection_name: Collection to search
        queries: Test queries
        top_k: Number of results per query
        
    Returns:
        Evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    query_results = []
    total_embed_time = 0
    total_search_time = 0
    
    for i, q in enumerate(queries, 1):
        query_ja = q["query_ja"]
        expected_kw = q["expected_keywords"]
        
        print(f"\n[{i}/{len(queries)}] Query: {q['query_vi'][:50]}...")
        
        # Embed query
        start = time.time()
        query_vector = embedding_model.embed(query_ja)
        embed_time = time.time() - start
        total_embed_time += embed_time
        
        # Search
        start = time.time()
        results = search_with_embedding(
            client=client,
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
        )
        search_time = time.time() - start
        total_search_time += search_time
        
        # Compute metrics
        retrieved_texts = [r["text"] for r in results]
        metrics = compute_retrieval_metrics(retrieved_texts, expected_kw)
        
        print(f"    Embed: {embed_time*1000:.0f}ms | Search: {search_time*1000:.0f}ms")
        print(f"    Keyword coverage: {metrics['keyword_coverage']:.0%} ({metrics['keyword_hits']}/{metrics['total_keywords']})")
        
        query_results.append({
            "query": q["query_vi"],
            "query_ja": query_ja,
            "embed_time_ms": round(embed_time * 1000, 1),
            "search_time_ms": round(search_time * 1000, 1),
            "metrics": metrics,
            "top_result": results[0] if results else None,
        })
    
    # Aggregate
    avg_coverage = np.mean([r["metrics"]["keyword_coverage"] for r in query_results])
    avg_embed_time = total_embed_time / len(queries) * 1000
    avg_search_time = total_search_time / len(queries) * 1000
    
    print(f"\n{'='*40}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*40}")
    print(f"  Avg Keyword Coverage: {avg_coverage:.1%}")
    print(f"  Avg Embed Time: {avg_embed_time:.1f}ms")
    print(f"  Avg Search Time: {avg_search_time:.1f}ms")
    print(f"  Embedding Dimension: {embedding_model.dimension}")
    
    return {
        "model_name": model_name,
        "dimension": embedding_model.dimension,
        "avg_keyword_coverage": round(avg_coverage, 4),
        "avg_embed_time_ms": round(avg_embed_time, 1),
        "avg_search_time_ms": round(avg_search_time, 1),
        "total_time_ms": round((total_embed_time + total_search_time) * 1000, 1),
        "query_results": query_results,
    }


def run_comparison(
    num_samples: Optional[int] = None,
    top_k: int = 5,
    collection_name: Optional[str] = None,
) -> dict:
    """
    Run full comparison between BGE models.
    
    Args:
        num_samples: Number of test queries (None = all)
        top_k: Number of results per query
        collection_name: Qdrant collection name
        
    Returns:
        Comparison results
    """
    from app.core.config import get_settings
    
    settings = get_settings()
    
    # Initialize Qdrant client
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    
    collection = collection_name or settings.qdrant_collection_name
    queries = TEST_QUERIES[:num_samples] if num_samples else TEST_QUERIES
    
    print("=" * 70)
    print("BGE EMBEDDING MODEL COMPARISON")
    print("=" * 70)
    print(f"Collection: {collection}")
    print(f"Test queries: {len(queries)}")
    print(f"Top-K: {top_k}")
    print(f"Retrieval Multiplier: 2 (new default)")
    print("=" * 70)
    
    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "collection": collection,
            "num_queries": len(queries),
            "top_k": top_k,
        },
        "models": {},
    }
    
    # Test BGE-base-en-v1.5
    print("\n" + "=" * 70)
    print("MODEL 1: BAAI/bge-base-en-v1.5")
    print("=" * 70)
    try:
        bge_base = BGEEmbedding("BAAI/bge-base-en-v1.5")
        results["models"]["bge_base"] = evaluate_model(
            model_name="BAAI/bge-base-en-v1.5",
            embedding_model=bge_base,
            client=client,
            collection_name=collection,
            queries=queries,
            top_k=top_k,
        )
        del bge_base  # Free memory
    except Exception as e:
        logger.error(f"Failed to evaluate bge-base: {e}")
        results["models"]["bge_base"] = {"error": str(e)}
    
    # Test BGE-M3
    print("\n" + "=" * 70)
    print("MODEL 2: BAAI/bge-m3 (Multilingual)")
    print("=" * 70)
    try:
        bge_m3 = BGEM3Embedding()
        results["models"]["bge_m3"] = evaluate_model(
            model_name="BAAI/bge-m3",
            embedding_model=bge_m3,
            client=client,
            collection_name=collection,
            queries=queries,
            top_k=top_k,
        )
        del bge_m3  # Free memory
    except Exception as e:
        logger.error(f"Failed to evaluate bge-m3: {e}")
        results["models"]["bge_m3"] = {"error": str(e)}
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    for name, model_results in results["models"].items():
        if "error" not in model_results:
            print(f"\n{model_results['model_name']}:")
            print(f"  Dimension: {model_results['dimension']}")
            print(f"  Keyword Coverage: {model_results['avg_keyword_coverage']:.1%}")
            print(f"  Embed Time: {model_results['avg_embed_time_ms']:.1f}ms")
            print(f"  Search Time: {model_results['avg_search_time_ms']:.1f}ms")
    
    # Recommend
    valid_models = {k: v for k, v in results["models"].items() if "error" not in v}
    if len(valid_models) >= 2:
        best_model = max(valid_models.items(), key=lambda x: x[1]["avg_keyword_coverage"])
        print(f"\n🏆 RECOMMENDED: {best_model[1]['model_name']}")
        print(f"   (Best keyword coverage: {best_model[1]['avg_keyword_coverage']:.1%})")
        results["recommendation"] = best_model[1]["model_name"]
    
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare BGE embedding models")
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of test queries (default: all)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results per query (default: 5)"
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=None,
        help="Qdrant collection name (default: from settings)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_comparison(
        num_samples=args.samples,
        top_k=args.top_k,
        collection_name=args.collection,
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
