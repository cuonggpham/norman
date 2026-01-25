# Tests Directory

Test data and scripts for evaluating the Norman RAG system.

## Structure

```
tests/
├── data/
│   ├── ragas_test_questions.json    # Test questions (Vietnamese + Japanese)
│   └── ragas_ground_truth.json      # Ground truth answers and contexts
├── __init__.py
└── README.md
```

## RAGAS Evaluation

To run RAGAS evaluation:

```bash
cd /home/dell/Documents/Code/norman/backend
python scripts/ragas_evaluation.py
```

See `scripts/ragas_evaluation.py` for more options.
