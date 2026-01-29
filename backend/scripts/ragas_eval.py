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
# Aligned with actual indexed laws in the system
TEST_DATASET = [
    # 労働基準法 (322AC0000000049) - 429 chunks
    {
        "question": "Thời gian làm việc tối đa mỗi tuần theo Luật tiêu chuẩn lao động (労働基準法) là bao nhiêu giờ?",
        "ground_truth": "Theo 労働基準法 Điều 32, thời gian làm việc không được vượt quá 40 giờ mỗi tuần và 8 giờ mỗi ngày. Có thể làm thêm giờ với thỏa thuận theo Điều 36 (三六協定).",
    },
    {
        "question": "Quy định về nghỉ phép năm có lương (年次有給休暇) trong Luật tiêu chuẩn lao động như thế nào?",
        "ground_truth": "Theo 労働基準法 Điều 39, người lao động được nghỉ phép có lương sau 6 tháng làm việc liên tục với tỷ lệ đi làm trên 80%. Năm đầu được 10 ngày, tăng dần theo thâm niên đến tối đa 20 ngày.",
    },
    {
        "question": "Tiền lương làm thêm giờ (時間外労働) được tính như thế nào theo luật lao động?",
        "ground_truth": "Theo 労働基準法 Điều 37, tiền làm thêm tối thiểu +25% cho giờ ngoài giờ, +35% cho ngày nghỉ, +25% cho làm đêm (22:00-05:00). Làm thêm quá 60 giờ/tháng: +50%.",
    },
    # 職業安定法 (322AC0000000141) - 248 chunks  
    {
        "question": "Quy định về môi giới việc làm (職業紹介) theo Luật ổn định việc làm là gì?",
        "ground_truth": "Theo 職業安定法, việc môi giới việc làm được thực hiện bởi Hello Work (公共職業安定所) miễn phí. Các công ty môi giới tư nhân cần giấy phép và không được thu phí người lao động.",
    },
    # 労働者災害補償保険法 (322AC0000000050) - 381 chunks
    {
        "question": "Bảo hiểm tai nạn lao động (労災保険) chi trả những khoản nào?",
        "ground_truth": "Theo 労働者災害補償保険法, bảo hiểm chi trả: chi phí y tế (療養補償), tiền nghỉ việc (休業補償), trợ cấp thương tật (障害補償), trợ cấp tử vong (遺族補償), chi phí tang lễ.",
    },
    # 金融商品取引法 (323AC0000000025) - 1565 chunks
    {
        "question": "Luật giao dịch sản phẩm tài chính (金融商品取引法) quy định những gì?",
        "ground_truth": "金融商品取引法 quy định về việc phát hành, giao dịch chứng khoán, bảo vệ nhà đầu tư, công bố thông tin, cấp phép công ty chứng khoán, và xử phạt giao dịch nội gián.",
    },
    # 児童福祉法 (322AC0000000164) - 703 chunks
    {
        "question": "Luật phúc lợi trẻ em (児童福祉法) quy định những quyền lợi gì cho trẻ em?",
        "ground_truth": "Theo 児童福祉法, trẻ em có quyền được nuôi dưỡng, giáo dục, bảo vệ. Luật quy định về cơ sở chăm sóc trẻ (保育所), trung tâm hỗ trợ trẻ em, phúc lợi cho trẻ khuyết tật.",
    },
    # 医療法 (323AC0000000205) - 499 chunks
    {
        "question": "Luật y tế (医療法) quy định về bệnh viện và cơ sở y tế như thế nào?",
        "ground_truth": "Theo 医療法, bệnh viện (病院) có từ 20 giường trở lên, phòng khám (診療所) dưới 20 giường. Luật quy định về cấp phép, tiêu chuẩn thiết bị, nhân sự y tế tối thiểu.",
    },
    # 国家公務員法 (322AC0000000120) - 348 chunks
    {
        "question": "Quy định về công chức nhà nước (国家公務員) theo Luật công chức quốc gia?",
        "ground_truth": "Theo 国家公務員法, công chức phải trung thành với hiến pháp, không được tham gia hoạt động chính trị, bị hạn chế quyền đình công. Có hệ thống thi tuyển, đánh giá, kỷ luật.",
    },
    # 地方自治法 (322AC0000000067) - 813 chunks  
    {
        "question": "Luật tự trị địa phương (地方自治法) quy định về chính quyền địa phương như thế nào?",
        "ground_truth": "Theo 地方自治法, chính quyền địa phương gồm: tỉnh (都道府県), thành phố/thị trấn/làng (市町村). Mỗi cấp có hội đồng (議会) và thủ trưởng (知事/市長) do dân bầu.",
    },
]


def run_ragas_evaluation(num_samples: int = None) -> dict:
    """
    Run RAGAS evaluation on the RAG pipeline.
    
    Args:
        num_samples: Number of test samples to use (None = all)
        
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
        response = pipeline.chat(question, top_k=5)
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
    
    # Define metrics
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
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_ragas_evaluation(num_samples=args.samples)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
