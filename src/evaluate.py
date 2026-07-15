from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import synthesize_labels
from src.retrieval import search


def precision_at_k(retrieved_ids: Sequence[object], relevant_ids: Iterable[object], k: int) -> float:
    """Compute precision@k for a ranked list of retrieved IDs."""
    retrieved_slice = list(retrieved_ids[:k]) if k is not None else list(retrieved_ids)
    relevant_set = set(relevant_ids)
    if not retrieved_slice:
        return 0.0
    hits = sum(1 for item in retrieved_slice if item in relevant_set)
    return hits / len(retrieved_slice)


def recall_at_k(retrieved_ids: Sequence[object], relevant_ids: Iterable[object], k: int) -> float:
    """Compute recall@k for a ranked list of retrieved IDs."""
    retrieved_slice = list(retrieved_ids[:k]) if k is not None else list(retrieved_ids)
    relevant_list = list(relevant_ids)
    if not relevant_list:
        return 0.0
    relevant_set = set(relevant_list)
    hits = sum(1 for item in retrieved_slice if item in relevant_set)
    return hits / len(relevant_list)


def _coerce_ids(df: pd.DataFrame, *, id_column: Optional[str] = None) -> np.ndarray:
    if id_column and id_column in df.columns:
        return df[id_column].to_numpy()
    if "id" in df.columns:
        return df["id"].to_numpy()
    if "ID" in df.columns:
        return df["ID"].to_numpy()
    if "job_id" in df.columns:
        return df["job_id"].to_numpy()
    return np.arange(len(df), dtype=object)


def evaluate_retrieval_baseline(
    resumes_df: pd.DataFrame,
    jds_df: pd.DataFrame,
    resume_embeddings: np.ndarray,
    jd_embeddings: np.ndarray,
    top_k: int = 10,
    sample_negatives: int = 5,
) -> pd.DataFrame:
    """Evaluate a raw retrieval baseline using synthetic labels.

    The baseline retrieves resumes for each JD using cosine similarity over the
    embedding vectors and reports precision@k and recall@k against labels
    synthesized from the resume/JD metadata.
    """
    labels_df = synthesize_labels(resumes_df, jds_df, sample_negatives=sample_negatives)
    resume_ids = _coerce_ids(resumes_df, id_column="id")
    jd_ids = _coerce_ids(jds_df, id_column="id")

    if resume_embeddings.ndim != 2 or jd_embeddings.ndim != 2:
        raise ValueError("Resume and JD embeddings must be 2D arrays.")

    scores, indices = search(resume_embeddings, jd_embeddings, top_k=top_k)
    results = []
    for query_idx, jd_id in enumerate(jd_ids):
        relevant_ids = labels_df.loc[(labels_df["jd_id"] == jd_id) & (labels_df["label"] > 0), "resume_id"].tolist()
        if not relevant_ids:
            continue

        retrieved_ids = [resume_ids[idx] for idx in indices[query_idx]]
        results.append(
            {
                "jd_id": jd_id,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, top_k),
                "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, top_k),
            }
        )

    return pd.DataFrame(results)


def evaluate_from_disk(
    resumes_csv: str | Path,
    jds_csv: str | Path,
    resume_embeddings_path: str | Path,
    jd_embeddings_path: str | Path,
    top_k: int = 10,
    sample_negatives: int = 5,
) -> pd.DataFrame:
    """Load data from disk and run the baseline evaluation."""
    resumes_df = pd.read_csv(resumes_csv, low_memory=False)
    jds_df = pd.read_csv(jds_csv, low_memory=False)
    resume_embeddings = np.load(resume_embeddings_path)
    jd_embeddings = np.load(jd_embeddings_path)
    return evaluate_retrieval_baseline(
        resumes_df,
        jds_df,
        resume_embeddings,
        jd_embeddings,
        top_k=top_k,
        sample_negatives=sample_negatives,
    )
