from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from src.config import synthesize_labels
from src.retrieval import search


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _token_overlap(resume_text: str, jd_text: str) -> float:
    resume_tokens = set(_normalize_text(resume_text).split())
    jd_tokens = set(_normalize_text(jd_text).split())
    if not resume_tokens or not jd_tokens:
        return 0.0
    return len(resume_tokens & jd_tokens) / len(resume_tokens | jd_tokens)


def build_feature_matrix(
    resumes_df: pd.DataFrame,
    jds_df: pd.DataFrame,
    resume_embeddings: np.ndarray,
    jd_embeddings: np.ndarray,
    top_k: int = 50,
    sample_negatives: int = 5,
) -> pd.DataFrame:
    """Build a reranker training frame from retrieval results and synthetic labels."""
    labels_df = synthesize_labels(resumes_df, jds_df, sample_negatives=sample_negatives)
    label_lookup = {
        (str(row["resume_id"]), str(row["jd_id"])): row["label"]
        for _, row in labels_df.iterrows()
    }

    resume_ids = resumes_df["id"].astype(str).tolist() if "id" in resumes_df.columns else list(range(len(resumes_df)))
    jd_ids = jds_df["id"].astype(str).tolist() if "id" in jds_df.columns else list(range(len(jds_df)))

    scores, indices = search(resume_embeddings, jd_embeddings, top_k=top_k)

    rows: List[Dict[str, Any]] = []
    for jd_idx, jd_id in enumerate(jd_ids):
        jd_row = jds_df.iloc[jd_idx]
        jd_text = " ".join(
            str(v) for v in [jd_row.get("title", ""), jd_row.get("description", "")] if pd.notna(v)
        )
        jd_category = jd_row.get("category")
        jd_broad_field = jd_row.get("broad_field")

        for rank, resume_idx in enumerate(indices[jd_idx], start=1):
            resume_row = resumes_df.iloc[resume_idx]
            resume_id = str(resume_row.get("id", resume_idx))
            resume_text = str(resume_row.get("Resume_str", ""))
            resume_category = resume_row.get("category")
            resume_broad_field = resume_row.get("broad_field")

            label = label_lookup.get((resume_id, str(jd_id)), 0)
            rows.append(
                {
                    "jd_id": str(jd_id),
                    "resume_id": resume_id,
                    "rank": rank,
                    "score": float(scores[jd_idx, rank - 1]),
                    "reciprocal_rank": 1.0 / rank,
                    "score_gap_to_top": float(scores[jd_idx, rank - 1] - scores[jd_idx, 0]),
                    "same_category": int(resume_category == jd_category),
                    "same_broad_field": int(resume_broad_field == jd_broad_field and resume_category != jd_category),
                    "resume_text_length": len(_normalize_text(resume_text)),
                    "jd_text_length": len(_normalize_text(jd_text)),
                    "token_overlap": _token_overlap(resume_text, jd_text),
                    "label": int(label),
                }
            )

    feature_frame = pd.DataFrame(rows)
    feature_frame["jd_id"] = feature_frame["jd_id"].astype(str)
    feature_frame["resume_id"] = feature_frame["resume_id"].astype(str)
    return feature_frame


def _ndcg_at_k(labels: np.ndarray, k: int = 10) -> float:
    labels = np.asarray(labels, dtype=float)[:k]
    if labels.size == 0:
        return 0.0
    gains = np.power(2.0, labels) - 1.0
    discounts = np.log2(np.arange(2, labels.size + 2, dtype=float))
    dcg = np.sum(gains / discounts)
    ideal = np.sort(labels)[::-1]
    ideal_gains = np.power(2.0, ideal) - 1.0
    ideal_discounts = np.log2(np.arange(2, ideal.size + 2, dtype=float))
    ideal_dcg = np.sum(ideal_gains / ideal_discounts)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def evaluate_reranker(
    resumes_df: pd.DataFrame,
    jds_df: pd.DataFrame,
    resume_embeddings: np.ndarray,
    jd_embeddings: np.ndarray,
    top_k: int = 50,
    sample_negatives: int = 5,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a LightGBM reranker with GroupKFold and compare nDCG@10 to the baseline."""
    feature_frame = build_feature_matrix(
        resumes_df,
        jds_df,
        resume_embeddings,
        jd_embeddings,
        top_k=top_k,
        sample_negatives=sample_negatives,
    )

    feature_cols = [
        "rank",
        "score",
        "reciprocal_rank",
        "score_gap_to_top",
        "same_category",
        "same_broad_field",
        "resume_text_length",
        "jd_text_length",
        "token_overlap",
    ]
    X = feature_frame[feature_cols]
    y = feature_frame["label"]
    groups = feature_frame["jd_id"]

    n_unique_groups = feature_frame["jd_id"].nunique()
    n_splits = min(max(2, n_splits), max(2, n_unique_groups))
    splitter = GroupKFold(n_splits=n_splits)

    fold_metrics: List[Dict[str, Any]] = []
    fold_importances: List[np.ndarray] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state + fold_idx,
            verbose=-1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_importances.append(model.feature_importances_)

        valid_frame = feature_frame.iloc[valid_idx].copy()
        valid_frame["predicted_score"] = model.predict(X.iloc[valid_idx])

        baseline_ndcg = []
        reranker_ndcg = []
        for jd_id, jd_rows in valid_frame.groupby("jd_id"):
            jd_rows = jd_rows.sort_values("rank")
            baseline_labels = jd_rows["label"].to_numpy()
            baseline_ndcg.append(_ndcg_at_k(baseline_labels, k=10))

            reranked_rows = jd_rows.sort_values("predicted_score", ascending=False)
            reranker_labels = reranked_rows["label"].to_numpy()
            reranker_ndcg.append(_ndcg_at_k(reranker_labels, k=10))

        fold_metrics.append(
            {
                "fold": fold_idx,
                "baseline_ndcg@10": float(np.mean(baseline_ndcg)),
                "reranker_ndcg@10": float(np.mean(reranker_ndcg)),
            }
        )

    feature_importances = np.mean(np.vstack(fold_importances), axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": feature_importances}
    ).sort_values("importance", ascending=False)

    return {
        "feature_frame": feature_frame,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "feature_importances": importance_df,
        "baseline_ndcg@10": float(np.mean([row["baseline_ndcg@10"] for row in fold_metrics])),
        "reranker_ndcg@10": float(np.mean([row["reranker_ndcg@10"] for row in fold_metrics])),
    }
