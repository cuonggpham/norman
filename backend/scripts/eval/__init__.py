# RAGAS Evaluation Module
"""
Scripts for evaluating the Japanese Legal RAG system using RAGAS metrics.

Usage:
    # Full evaluation
    python -m scripts.eval.evaluate_ragas
    
    # Compare with/without reranker
    python -m scripts.eval.evaluate_ragas --no-reranker
    
    # Quick test with limited samples
    python -m scripts.eval.evaluate_ragas --limit 3
"""
