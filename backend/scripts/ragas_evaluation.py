#!/usr/bin/env python3
"""
RAGAS Evaluation Script for Norman RAG System.

Evaluates the RAG pipeline using RAGAS framework metrics:
- Context Precision: How precise are the retrieved contexts?
- Context Recall: How much of the ground truth context is retrieved?
- Faithfulness: Is the answer faithful to the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?

Usage:
    python scripts/ragas_evaluation.py
    python scripts/ragas_evaluation.py --mode hybrid --output results.json
    python scripts/ragas_evaluation.py --limit 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

# App imports
from app.api.deps import (
    get_rag_pipeline,
    get_embedding_service,
)
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """Evaluates RAG system using RAGAS metrics."""
    
    def __init__(
        self,
        test_questions_path: str,
        ground_truth_path: str,
        use_hybrid: bool = True,
    ):
        """
        Initialize evaluator.
        
        Args:
            test_questions_path: Path to test questions JSON
            ground_truth_path: Path to ground truth JSON
            use_hybrid: Whether to use hybrid search (default: True)
        """
        self.test_questions_path = Path(test_questions_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.use_hybrid = use_hybrid
        
        # Load data
        self.questions = self._load_questions()
        self.ground_truth = self._load_ground_truth()
        
        # Initialize RAG pipeline
        logger.info(f"Initializing RAG pipeline (hybrid={use_hybrid})...")
        self.pipeline = get_rag_pipeline()
        self.embedding_service = get_embedding_service()
        self.settings = get_settings()
        
        logger.info("✓ Evaluator initialized successfully")
    
    def _load_questions(self) -> list[str]:
        """Load test questions from JSON."""
        logger.info(f"Loading questions from {self.test_questions_path}")
        with open(self.test_questions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = data.get("questions", [])
        logger.info(f"✓ Loaded {len(questions)} test questions")
        return questions
    
    def _load_ground_truth(self) -> list[dict[str, Any]]:
        """Load ground truth from JSON."""
        logger.info(f"Loading ground truth from {self.ground_truth_path}")
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ground_truth = data.get("ground_truths", [])
        logger.info(f"✓ Loaded {len(ground_truth)} ground truth entries")
        return ground_truth
    
    def run_rag_queries(self, limit: int | None = None) -> dict[str, list]:
        """
        Run RAG queries and collect results.
        
        Args:
            limit: Maximum number of questions to test (None = all)
            
        Returns:
            Dictionary with questions, answers, contexts, and ground_truth
        """
        questions_to_test = self.questions[:limit] if limit else self.questions
        
        results = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        
        logger.info(f"Running RAG queries for {len(questions_to_test)} questions...")
        
        for i, question in enumerate(questions_to_test, 1):
            logger.info(f"[{i}/{len(questions_to_test)}] Processing: {question[:60]}...")
            
            try:
                # Run RAG query
                start_time = time.time()
                response = self.pipeline.chat(
                    query=question,
                    top_k=5,
                    use_hybrid=self.use_hybrid,
                )
                elapsed = time.time() - start_time
                
                # Extract answer and contexts
                answer = response.answer
                contexts = [source.text for source in response.sources]
                
                # Get ground truth for this question
                gt_entry = next(
                    (gt for gt in self.ground_truth if gt["question"] == question),
                    None
                )
                ground_truth_answer = gt_entry.get("reference_answer", "") if gt_entry else ""
                
                # Store results
                results["question"].append(question)
                results["answer"].append(answer)
                results["contexts"].append(contexts)
                results["ground_truth"].append(ground_truth_answer)
                
                logger.info(f"  ✓ Completed in {elapsed:.2f}s | Contexts: {len(contexts)}")
                
            except Exception as e:
                logger.error(f"  ✗ Error processing question: {e}")
                # Add placeholder to keep arrays aligned
                results["question"].append(question)
                results["answer"].append("")
                results["contexts"].append([])
                results["ground_truth"].append("")
        
        logger.info(f"✓ Completed {len(results['question'])} queries")
        return results
    
    def evaluate_with_ragas(self, results: dict[str, list]) -> dict[str, Any]:
        """
        Evaluate results using RAGAS metrics.
        
        Args:
            results: Dictionary with question, answer, contexts, ground_truth
            
        Returns:
            Dictionary with RAGAS metrics scores
        """
        logger.info("Evaluating with RAGAS metrics...")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict(results)
        
        # Run RAGAS evaluation
        logger.info("Running RAGAS evaluation (this may take a few minutes)...")
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        
        # Convert to dict
        scores = evaluation_result.to_pandas().to_dict()
        
        logger.info("✓ RAGAS evaluation complete")
        return scores
    
    def export_results(
        self,
        results: dict[str, list],
        scores: dict[str, Any],
        output_path: str,
    ):
        """
        Export results to JSON and CSV.
        
        Args:
            results: RAG query results
            scores: RAGAS metric scores
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare export data
        export_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "search_mode": "hybrid" if self.use_hybrid else "vector_only",
                "total_questions": len(results["question"]),
                "model": self.settings.llm_model,
                "embedding_model": self.settings.embedding_model,
            },
            "summary_metrics": {
                metric: (
                    float(sum(values) / len(values))
                    if isinstance(values, list) and len(values) > 0
                    else values
                )
                for metric, values in scores.items()
            },
            "detailed_results": [
                {
                    "question": results["question"][i],
                    "answer": results["answer"][i],
                    "contexts": results["contexts"][i],
                    "ground_truth": results["ground_truth"][i],
                    "scores": {
                        metric: scores[metric][i] if isinstance(scores[metric], list) else None
                        for metric in scores.keys()
                    }
                }
                for i in range(len(results["question"]))
            ]
        }
        
        # Export JSON
        logger.info(f"Exporting results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        # Export CSV
        csv_path = output_path.with_suffix('.csv')
        logger.info(f"Exporting summary to {csv_path}")
        
        import csv
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Question",
                "Answer",
                "Ground Truth",
                "Context Precision",
                "Context Recall",
                "Faithfulness",
                "Answer Relevancy"
            ])
            
            for item in export_data["detailed_results"]:
                writer.writerow([
                    item["question"],
                    item["answer"][:100] + "..." if len(item["answer"]) > 100 else item["answer"],
                    item["ground_truth"][:100] + "..." if len(item["ground_truth"]) > 100 else item["ground_truth"],
                    item["scores"].get("context_precision", ""),
                    item["scores"].get("context_recall", ""),
                    item["scores"].get("faithfulness", ""),
                    item["scores"].get("answer_relevancy", ""),
                ])
        
        logger.info("✓ Results exported successfully")
        
        # Print summary
        print("\n" + "="*60)
        print("RAGAS EVALUATION SUMMARY")
        print("="*60)
        print(f"Search Mode: {'Hybrid' if self.use_hybrid else 'Vector Only'}")
        print(f"Total Questions: {export_data['metadata']['total_questions']}")
        print("\nAverage Metrics:")
        for metric, value in export_data["summary_metrics"].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Norman RAG system with RAGAS"
    )
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to test (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ragas_evaluation.json",
        help="Output path for results (default: results/ragas_evaluation.json)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="tests/data/ragas_test_questions.json",
        help="Path to test questions JSON"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="tests/data/ragas_ground_truth.json",
        help="Path to ground truth JSON"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = RAGASEvaluator(
            test_questions_path=args.questions,
            ground_truth_path=args.ground_truth,
            use_hybrid=(args.mode == "hybrid"),
        )
        
        # Run RAG queries
        results = evaluator.run_rag_queries(limit=args.limit)
        
        # Evaluate with RAGAS
        scores = evaluator.evaluate_with_ragas(results)
        
        # Export results
        evaluator.export_results(results, scores, args.output)
        
        logger.info("✓ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
