"""
RAGAS Evaluation Script for Japanese Legal RAG System.

Evaluates the RAG pipeline using 4 key metrics:
1. Context Recall - Does retrieval get the right documents?
2. Context Precision - Are important docs ranked higher?
3. Faithfulness - Is the response grounded in context?
4. Answer Relevancy - Does the response answer the question?

Usage:
    python scripts/eval/evaluate_ragas.py
    python scripts/eval/evaluate_ragas.py --output results.json
    python scripts/eval/evaluate_ragas.py --no-reranker  # Compare without reranker
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragas import evaluate
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from datasets import Dataset

from app.api.deps import get_embedding_service, get_vector_store, get_llm_provider, get_reranker, get_query_translator
from app.pipelines.rag import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_evaluation_dataset(path: Path) -> list[dict]:
    """Load evaluation samples from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


def create_rag_pipeline(use_reranker: bool = True) -> RAGPipeline:
    """Create RAG pipeline with optional reranker."""
    embedding = get_embedding_service()
    vector_store = get_vector_store()
    llm = get_llm_provider()
    translator = get_query_translator()
    reranker = get_reranker() if use_reranker else None
    
    return RAGPipeline(
        embedding=embedding,
        vector_store=vector_store,
        llm=llm,
        translator=translator,
        reranker=reranker,
        default_top_k=5,
    )


def run_rag_pipeline(pipeline: RAGPipeline, samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Run RAG pipeline for each sample and collect results.
    
    Returns tuple of:
    - results: list of dicts with question, answer, contexts, ground_truth
    - problematic_queries: list of queries that failed or returned no documents
    """
    results = []
    problematic_queries = []
    
    for i, sample in enumerate(samples):
        logger.info(f"Processing {i+1}/{len(samples)}: {sample['question'][:50]}...")
        
        try:
            # Run full RAG pipeline
            response = pipeline.chat(sample["question"], top_k=5)
            
            # Extract contexts from sources
            contexts = [
                f"【{src.law_title} {src.article}】\n{src.text}"
                for src in response.sources
            ]
            
            # Track queries with no documents found
            if not contexts:
                problematic_queries.append({
                    "question": sample["question"],
                    "issue": "No documents retrieved",
                    "expected_law": sample.get("expected_law", ""),
                    "expected_article": sample.get("expected_article", ""),
                })
            
            results.append({
                "question": sample["question"],
                "answer": response.answer,
                "contexts": contexts,
                "ground_truth": sample["ground_truth"],
                "processing_time_ms": response.processing_time_ms,
            })
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            problematic_queries.append({
                "question": sample["question"],
                "issue": f"Error: {e}",
                "expected_law": sample.get("expected_law", ""),
                "expected_article": sample.get("expected_article", ""),
            })
            results.append({
                "question": sample["question"],
                "answer": f"Error: {e}",
                "contexts": [],
                "ground_truth": sample["ground_truth"],
                "processing_time_ms": 0,
            })
    
    return results, problematic_queries


def evaluate_with_ragas(results: list[dict]) -> dict:
    """
    Run RAGAS evaluation on collected results.
    
    Returns dict with overall scores and per-sample details.
    """
    import os
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    # Configure RAGAS LLM and Embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    langchain_llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0,
    )
    langchain_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=openai_api_key,
    )
    
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    # Filter out results with empty contexts or error responses
    valid_results = [
        r for r in results 
        if r["contexts"] and not r["answer"].startswith("Error:")
    ]
    
    if len(valid_results) < len(results):
        logger.warning(
            f"Filtered {len(results) - len(valid_results)} samples with empty contexts or errors"
        )
    
    if not valid_results:
        logger.error("No valid results to evaluate!")
        return {
            "overall_scores": {},
            "dataset_size": 0,
            "raw_results": [],
            "skipped_count": len(results),
        }
    
    # Prepare dataset for RAGAS
    # RAGAS expects: user_input, response, retrieved_contexts, reference
    ragas_data = []
    for r in valid_results:
        ragas_data.append({
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["ground_truth"],
        })
    dataset = Dataset.from_list(ragas_data)
    
    # Define metrics to evaluate (instantiate classes with embeddings)
    context_recall = ContextRecall(llm=ragas_llm)
    context_precision = ContextPrecision(llm=ragas_llm)
    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    
    metrics = [
        context_recall,      # Retrieval quality
        context_precision,   # Ranking quality
        faithfulness,        # Grounded responses
        answer_relevancy,    # Relevant answers
    ]
    
    logger.info("Running RAGAS evaluation...")
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    
    # Convert to pandas and extract mean scores
    df = evaluation_result.to_pandas()
    scores = {}
    
    # Each metric adds a column to the dataframe
    for metric in metrics:
        col_name = metric.name
        if col_name in df.columns:
            # Get mean score, excluding NaN
            mean_score = df[col_name].dropna().mean()
            scores[col_name] = float(mean_score) if not pd.isna(mean_score) else 0.0
    
    return {
        "overall_scores": scores,
        "dataset_size": len(results),
        "raw_results": df.to_dict(orient="records"),
    }


def save_results(results: dict, output_path: Path):
    """Save evaluation results to JSON file."""
    results["timestamp"] = datetime.now().isoformat()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def print_summary(results: dict):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    
    scores = results["overall_scores"]
    
    print("\n📊 Overall Scores:")
    print("-" * 40)
    
    metric_descriptions = {
        "context_recall": "Retrieval Coverage",
        "context_precision": "Ranking Quality",
        "faithfulness": "Response Grounding",
        "answer_relevancy": "Answer Relevance",
    }
    
    for metric, score in scores.items():
        desc = metric_descriptions.get(metric, metric)
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {desc:20} {bar} {score:.2%}")
    
    print("\n" + "-" * 40)
    avg_score = sum(scores.values()) / len(scores)
    print(f"  {'Average':20} {'':20} {avg_score:.2%}")
    
    print("\n📈 Interpretation:")
    if scores.get("context_recall", 0) < 0.7:
        print("  ⚠️  Low Context Recall - Improve retrieval/chunking strategy")
    if scores.get("context_precision", 0) < 0.7:
        print("  ⚠️  Low Context Precision - Tune reranker or increase top_k")
    if scores.get("faithfulness", 0) < 0.8:
        print("  ⚠️  Low Faithfulness - LLM may be hallucinating")
    if scores.get("answer_relevancy", 0) < 0.8:
        print("  ⚠️  Low Answer Relevancy - Improve prompt or response format")
    
    if all(s >= 0.8 for s in scores.values()):
        print("  ✅ All metrics above 80% - RAG system performing well!")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "evaluation_dataset.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "evaluation_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Run evaluation without reranker for comparison",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    samples = load_evaluation_dataset(args.dataset)
    
    if args.limit:
        samples = samples[:args.limit]
        logger.info(f"Limited to {args.limit} samples")
    
    # Create pipeline
    use_reranker = not args.no_reranker
    logger.info(f"Creating RAG pipeline (reranker: {use_reranker})")
    pipeline = create_rag_pipeline(use_reranker=use_reranker)
    
    # Run pipeline on samples
    logger.info("Running RAG pipeline on samples...")
    rag_results, problematic_queries = run_rag_pipeline(pipeline, samples)
    
    # Evaluate with RAGAS
    logger.info("Evaluating with RAGAS metrics...")
    evaluation = evaluate_with_ragas(rag_results)
    
    # Add metadata
    evaluation["config"] = {
        "use_reranker": use_reranker,
        "dataset_path": str(args.dataset),
        "samples_evaluated": len(samples),
    }
    evaluation["problematic_queries"] = problematic_queries
    
    # Save and display results
    save_results(evaluation, args.output)
    print_summary(evaluation)
    
    # Print problematic queries if any
    if problematic_queries:
        print("\n" + "=" * 60)
        print("⚠️  PROBLEMATIC QUERIES (need data supplement)")
        print("=" * 60)
        for i, pq in enumerate(problematic_queries, 1):
            print(f"\n{i}. {pq['question']}")
            print(f"   Issue: {pq['issue']}")
            print(f"   Expected: {pq['expected_law']} {pq['expected_article']}")
        print("\n" + "=" * 60)
    
    return evaluation


if __name__ == "__main__":
    main()
