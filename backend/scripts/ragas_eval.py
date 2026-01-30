#!/usr/bin/env python3
"""
RAGAS Evaluation Script for Norman RAG System.

Evaluates RAG quality using RAGAS metrics:
- Faithfulness: Does the answer stick to the context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are the retrieved contexts precise?
- Context Recall: Are all relevant contexts retrieved?

Usage:
    python scripts/ragas_eval.py
    python scripts/ragas_eval.py --samples 10 --output results/ragas_results.json
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test dataset with ground truth answers
# Ground truth viết lại dựa trên nội dung THỰC TẾ từ chunks trong database
# Categories: 労働, 金融・保険, 社会保険, 国税
TEST_DATASET = [
    # ============ 労働基準法 (322AC0000000049) - LUẬT TIÊU CHUẨN LAO ĐỘNG ============
    # Source: 第三十二条 - "一週間について四十時間を超えて...労働させてはならない"
    {
        "question": "Thời gian làm việc tối đa mỗi tuần theo Luật tiêu chuẩn lao động (労働基準法) là bao nhiêu giờ?",
        "ground_truth": "Theo 労働基準法 第三十二条, người sử dụng lao động không được để người lao động làm việc quá 40 giờ mỗi tuần (休憩時間を除き一週間について四十時間) và quá 8 giờ mỗi ngày (一日について八時間).",
    },
    # Source: 第三十六条 - "一箇月について四十五時間及び一年について三百六十時間"
    {
        "question": "Giới hạn làm thêm giờ (時間外労働) theo thỏa thuận 36 là bao nhiêu?",
        "ground_truth": "Theo 労働基準法 第三十六条, giới hạn làm thêm giờ (限度時間) là 45 giờ mỗi tháng (一箇月について四十五時間) và 360 giờ mỗi năm (一年について三百六十時間).",
    },

    # ============ 労働者災害補償保険法 (322AC0000000050) - BẢO HIỂM TAI NẠN LAO ĐỘNG ============
    # Source: 第二条の二 - "業務上の事由...による労働者の負傷、疾病、障害、死亡等に関して保険給付を行う"
    {
        "question": "Bảo hiểm tai nạn lao động (労災保険) chi trả cho những trường hợp nào?",
        "ground_truth": "Theo 労働者災害補償保険法 第二条の二, bảo hiểm thực hiện chi trả (保険給付) cho các trường hợp: thương tích (負傷), bệnh tật (疾病), khuyết tật (障害), tử vong (死亡) của người lao động do nguyên nhân công việc (業務上の事由) hoặc đi lại (通勤).",
    },

    # ============ 厚生年金保険法 (329AC0000000115) - BẢO HIỂM HƯU TRÍ ============
    # Source: 第一条 - "労働者の老齢、障害又は死亡について保険給付を行い"
    # Source: 第二条 - "厚生年金保険は、政府が、管掌する"
    {
        "question": "Bảo hiểm hưu trí xã hội (厚生年金保険) nhằm mục đích gì?",
        "ground_truth": "Theo 厚生年金保険法 第一条, bảo hiểm này thực hiện chi trả bảo hiểm cho người lao động khi già (老齢), khuyết tật (障害) hoặc tử vong (死亡), nhằm ổn định đời sống và nâng cao phúc lợi cho người lao động và gia đình (労働者及びその遺族の生活の安定と福祉の向上).",
    },
    # Source: 第二条 - "厚生年金保険は、政府が、管掌する"
    {
        "question": "Ai quản lý bảo hiểm hưu trí xã hội (厚生年金保険)?",
        "ground_truth": "Theo 厚生年金保険法 第二条, bảo hiểm hưu trí được chính phủ quản lý (政府が管掌する).",
    },

    # ============ 消費税法 (363AC0000000108) - THUẾ TIÊU DÙNG ============
    # Source: 第二十九条 - "百分の七・八" "百分の六・二四"
    {
        "question": "Thuế tiêu dùng (消費税) ở Nhật Bản có mức thuế suất bao nhiêu?",
        "ground_truth": "Theo 消費税法 第二十九条, thuế suất tiêu dùng là 7.8% (百分の七・八) cho hàng hóa thông thường, và 6.24% (百分の六・二四) cho hàng hóa giảm thuế (軽減対象). Cộng với thuế địa phương sẽ là 10% và 8%.",
    },

    # ============ 金融商品取引法 (323AC0000000025) - LUẬT GIAO DỊCH TÀI CHÍNH ============
    {
        "question": "Luật giao dịch sản phẩm tài chính (金融商品取引法) quy định những gì?",
        "ground_truth": "Theo 金融商品取引法, luật này quy định về việc phát hành, giao dịch chứng khoán, công bố thông tin, cấp phép công ty chứng khoán, và bảo vệ nhà đầu tư.",
    },

    # ============ 銀行法 (356AC0000000059) - LUẬT NGÂN HÀNG ============
    {
        "question": "Luật Ngân hàng (銀行法) quy định về những hoạt động nào?",
        "ground_truth": "Theo 銀行法, ngân hàng được phép thực hiện các hoạt động: nhận tiền gửi, cho vay, chuyển tiền, đổi ngoại tệ. Việc kinh doanh ngân hàng cần được cấp phép.",
    },

    # ============ 雇用保険法 (349AC0000000116) - BẢO HIỂM VIỆC LÀM ============
    {
        "question": "Bảo hiểm việc làm (雇用保険) nhằm mục đích gì?",
        "ground_truth": "Theo 雇用保険法, bảo hiểm việc làm nhằm hỗ trợ người lao động khi thất nghiệp, cung cấp trợ cấp thất nghiệp (失業給付) và các chương trình hỗ trợ tìm việc làm.",
    },

    # ============ 国民健康保険法 (333AC0000000192) - BẢO HIỂM Y TẾ QUỐC DÂN ============
    {
        "question": "Bảo hiểm y tế quốc dân (国民健康保険) là gì?",
        "ground_truth": "Theo 国民健康保険法, bảo hiểm y tế quốc dân là chế độ bảo hiểm y tế cho người dân không thuộc các chế độ bảo hiểm sức khỏe khác, chi trả phần lớn chi phí y tế.",
    },
]


def run_ragas_evaluation(num_samples: int = None, reference_free: bool = False) -> dict:
    """
    Run RAGAS evaluation on the RAG pipeline.
    
    Args:
        num_samples: Number of test samples to use (None = all)
        reference_free: If True, only use metrics that don't require ground truth
                       (Faithfulness, Answer Relevancy)
        
    Returns:
        Dictionary with evaluation results
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
    
    from app.api.deps import get_rag_pipeline
    
    # Get pipeline
    pipeline = get_rag_pipeline()
    
    # Select samples
    samples = TEST_DATASET[:num_samples] if num_samples else TEST_DATASET
    
    print("=" * 70)
    print("RAGAS EVALUATION")
    print("=" * 70)
    print(f"Testing {len(samples)} samples\n")
    
    # Collect RAG responses
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    print("Collecting RAG responses...")
    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        
        print(f"  [{i}/{len(samples)}] {question[:50]}...")
        
        # Get RAG response
        start = time.time()
        response = pipeline.chat(question, top_k=7)
        elapsed = time.time() - start
        
        # Extract contexts from sources
        contexts = [src.text for src in response.sources]
        
        questions.append(question)
        answers.append(response.answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        
        print(f"      -> {elapsed:.2f}s, {len(contexts)} contexts")
    
    print("\nCreating RAGAS dataset...")
    
    # Create dataset for RAGAS
    if reference_free:
        # Reference-free mode: no ground truth needed
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
    else:
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    dataset = Dataset.from_dict(data)
    
    print(f"Dataset created with {len(dataset)} samples\n")
    
    # Setup LLM and embeddings for RAGAS evaluation
    print("Initializing RAGAS evaluator...")
    api_key = os.getenv("OPENAI_API_KEY")
    
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=api_key,
        temperature=0,
    ))
    
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    ))
    
    # Define metrics based on mode
    if reference_free:
        # Reference-free: only Faithfulness and Answer Relevancy
        print("Mode: Reference-Free (no ground truth needed)")
        metrics = [
            faithfulness,
            answer_relevancy,
        ]
    else:
        # Full evaluation with ground truth
        print("Mode: Full Evaluation (with ground truth)")
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    
    print("Running RAGAS evaluation (this may take a few minutes)...")
    print("-" * 70)
    
    # Run evaluation
    start_time = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    eval_time = time.time() - start_time
    
    print("-" * 70)
    print(f"\nEvaluation completed in {eval_time:.1f}s\n")
    
    # Convert to pandas for easier processing
    df = result.to_pandas()
    
    # Compute mean scores for each metric
    scores = {}
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for metric in metric_names:
        if metric in df.columns:
            scores[metric] = round(df[metric].mean(), 4)
        else:
            scores[metric] = None
    
    # Format results
    results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(samples),
            "evaluation_time_s": round(eval_time, 1),
        },
        "scores": scores,
        "sample_results": [],
    }
    
    # Add per-sample results
    for idx, row in df.iterrows():
        # Get question text - RAGAS might use 'user_input' or 'question'
        q = row.get("user_input", row.get("question", str(idx)))
        if isinstance(q, str):
            q_text = q[:100] + "..." if len(q) > 100 else q
        else:
            q_text = f"Sample {idx}"
        
        sample_result = {"question": q_text}
        for metric in metric_names:
            if metric in df.columns and row[metric] is not None:
                try:
                    sample_result[metric] = round(float(row[metric]), 4)
                except (TypeError, ValueError):
                    sample_result[metric] = None
            else:
                sample_result[metric] = None
        results["sample_results"].append(sample_result)
    
    # Print summary
    print("=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)
    for metric_name, score in scores.items():
        if score is not None:
            print(f"  {metric_name.replace('_', ' ').title():20s}: {score:.2%}")
    print("=" * 70)
    
    # Compute average (excluding None values)
    valid_scores = [s for s in scores.values() if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"  {'Average Score':20s}: {avg_score:.2%}")
    print("=" * 70)
    
    results["scores"]["average"] = round(avg_score, 4)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for Norman RAG")
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no-ground-truth", "--reference-free",
        action="store_true",
        dest="reference_free",
        help="Reference-free mode: only evaluate Faithfulness and Answer Relevancy (no ground truth needed)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_ragas_evaluation(
        num_samples=args.samples,
        reference_free=args.reference_free
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
