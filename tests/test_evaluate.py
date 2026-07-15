import numpy as np
import pandas as pd

from src.evaluate import (
    evaluate_retrieval_baseline,
    precision_at_k,
    recall_at_k,
    dcg_at_k,
    ndcg_at_k,
    mrr,
    precision_at_k_relevances,
    recall_at_k_relevances,
    full_evaluation,
)


def test_precision_and_recall_at_k():
    retrieved = [1, 2, 3]
    relevant = [2, 4]

    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 3) == 1 / 2


def test_evaluate_retrieval_baseline_works_with_synthetic_data():
    resumes_df = pd.DataFrame(
        [
            {"id": 1, "Category": "Information-Technology", "Resume_str": "software developer"},
            {"id": 2, "Category": "Finance", "Resume_str": "accountant"},
            {"id": 3, "Category": "Mechanical-Engineer", "Resume_str": "mechanical engineer"},
        ]
    )
    jds_df = pd.DataFrame(
        [
            {"id": 10, "title": "Software Engineer", "description": "Build backend APIs"},
            {"id": 11, "title": "Accountant", "description": "Manage accounts"},
        ]
    )

    resume_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    jd_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    results = evaluate_retrieval_baseline(resumes_df, jds_df, resume_embeddings, jd_embeddings, top_k=2, sample_negatives=0)

    assert len(results) == 2
    assert results.iloc[0]["jd_id"] == 10
    assert results.iloc[0]["precision_at_k"] >= 0.0
    assert results.iloc[0]["recall_at_k"] >= 0.0


def test_ir_metric_helpers():
    assert np.isclose(dcg_at_k([3, 2, 1], 3), (2**3 - 1) / np.log2(2) + (2**2 - 1) / np.log2(3) + (2**1 - 1) / np.log2(4))
    assert np.isclose(ndcg_at_k([3, 2, 1], 3), 1.0)
    assert np.isclose(mrr([[0, 0, 1], [1, 0, 0]]), (1/3 + 1.0) / 2)
    assert np.isclose(precision_at_k_relevances([1, 0, 1], 2), 0.5)
    assert np.isclose(recall_at_k_relevances([1, 0, 1], 2, total_relevant=2), 0.5)
    summary = full_evaluation([[1, 0, 0], [0, 1, 1]], ks=[1, 2])
    assert summary["nDCG@1"] >= 0.0
    assert summary["P@1"] >= 0.0
    assert summary["MRR"] >= 0.0
