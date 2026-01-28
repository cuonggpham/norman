#!/usr/bin/env python3
"""
RAG Pipeline Benchmark Script.

Measures query performance before/after optimizations.
Tests:
1. First query (cold - no cache)
2. Repeated query (hot - with cache)
3. Multiple different queries

Usage:
    python scripts/benchmark_rag.py
    python scripts/benchmark_rag.py --queries 5 --output results/benchmark.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs for cleaner output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample queries for testing
TEST_QUERIES = [
    "Thời gian làm việc tối đa một tuần là bao nhiêu giờ?",
    "Làm thế nào để khai thuế cuối năm (確定申告)?",
    "Quy định về nghỉ phép có lương trong luật lao động Nhật?",
    "Tiền lương tối thiểu ở Nhật Bản là bao nhiêu?",
    "Chế độ bảo hiểm thất nghiệp ở Nhật hoạt động như thế nào?",
]


def run_benchmark(num_queries: int = 3, top_k: int = 5) -> dict:
    """
    Run benchmark tests on RAG pipeline.
    
    Args:
        num_queries: Number of queries to test
        top_k: Number of results per query
        
    Returns:
        Benchmark results dictionary
    """
    from app.api.deps import get_rag_pipeline
    from app.services.query_cache import get_query_cache
    
    # Clear cache before testing
    cache = get_query_cache()
    cache.clear()
    
    pipeline = get_rag_pipeline()
    
    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_queries": num_queries,
            "top_k": top_k,
        },
        "first_query_times": [],
        "cached_query_times": [],
        "different_queries": [],
        "summary": {},
    }
    
    queries = TEST_QUERIES[:num_queries]
    
    print("=" * 70)
    print("RAG PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"Testing {num_queries} queries with top_k={top_k}\n")
    
    # Test 1: First queries (no cache)
    print("-" * 70)
    print("TEST 1: First queries (no cache)")
    print("-" * 70)
    
    for i, query in enumerate(queries, 1):
        start = time.time()
        response = pipeline.chat(query, top_k=top_k)
        elapsed = (time.time() - start) * 1000
        
        results["first_query_times"].append({
            "query": query[:50] + "...",
            "time_ms": round(elapsed, 0),
            "time_s": round(elapsed / 1000, 2),
            "sources": len(response.sources),
        })
        
        print(f"  [{i}/{num_queries}] {elapsed:.0f}ms ({elapsed/1000:.2f}s) - {query[:40]}...")
    
    avg_first = sum(r["time_ms"] for r in results["first_query_times"]) / len(results["first_query_times"])
    print(f"\n  Average first query: {avg_first:.0f}ms ({avg_first/1000:.2f}s)\n")
    
    # Test 2: Cached queries (same queries again)
    print("-" * 70)
    print("TEST 2: Cached queries (same queries, should be faster)")
    print("-" * 70)
    
    for i, query in enumerate(queries, 1):
        start = time.time()
        response = pipeline.chat(query, top_k=top_k)
        elapsed = (time.time() - start) * 1000
        
        results["cached_query_times"].append({
            "query": query[:50] + "...",
            "time_ms": round(elapsed, 0),
            "time_s": round(elapsed / 1000, 2),
            "sources": len(response.sources),
        })
        
        print(f"  [{i}/{num_queries}] {elapsed:.0f}ms ({elapsed/1000:.2f}s) - {query[:40]}...")
    
    avg_cached = sum(r["time_ms"] for r in results["cached_query_times"]) / len(results["cached_query_times"])
    print(f"\n  Average cached query: {avg_cached:.0f}ms ({avg_cached/1000:.2f}s)\n")
    
    # Summary
    speedup = avg_first - avg_cached
    speedup_pct = (speedup / avg_first) * 100 if avg_first > 0 else 0
    
    results["summary"] = {
        "avg_first_query_ms": round(avg_first, 0),
        "avg_cached_query_ms": round(avg_cached, 0),
        "speedup_ms": round(speedup, 0),
        "speedup_percent": round(speedup_pct, 1),
    }
    
    # Cache stats
    cache_stats = cache.get_stats()
    results["cache_stats"] = cache_stats
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  First query average:  {avg_first:.0f}ms ({avg_first/1000:.2f}s)")
    print(f"  Cached query average: {avg_cached:.0f}ms ({avg_cached/1000:.2f}s)")
    print(f"  Speedup:              {speedup:.0f}ms ({speedup_pct:.1f}%)")
    print(f"  Cache stats:          {cache_stats}")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark")
    parser.add_argument(
        "--queries", "-n",
        type=int,
        default=3,
        help="Number of queries to test (default: 3)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results per query (default: 5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        num_queries=args.queries,
        top_k=args.top_k,
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
