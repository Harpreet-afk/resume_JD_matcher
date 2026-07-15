"""FAISS index building and similarity search."""

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

import numpy as np
from typing import List, Optional, Tuple, Union

Corpus = np.ndarray if faiss is None else Union[faiss.Index, np.ndarray]


def build_index(embeddings: np.ndarray, use_ivf: bool = False, nlist: int = 100):
    """Build a FAISS index for fast similarity search.

    Args:
        embeddings: Normalized vectors (N x D)
        use_ivf: Use IVF index for scalability demo (optional)
        nlist: Number of Voronoi cells for IVF
    """
    if faiss is None:
        raise ImportError("faiss is required to build an index. Install faiss-cpu or disable index usage.")

    d = embeddings.shape[1]

    if use_ivf and embeddings.shape[0] > 1000:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 10  # search 10 nearest cells
    else:
        index = faiss.IndexFlatIP(d)  # exact search, cosine via normalized vecs

    index.add(embeddings)
    return index


def search(
    index_or_embeddings: Corpus,
    query_embeddings: np.ndarray,
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Search a FAISS index or raw embedding matrix.

    If a raw embedding matrix is passed, it must be normalized so that cosine
    similarity equals inner product.

    Returns:
        scores: (num_queries x top_k) similarity scores
        indices: (num_queries x top_k) indices into the indexed corpus
    """
    if hasattr(index_or_embeddings, "search"):
        scores, indices = index_or_embeddings.search(query_embeddings, top_k)
        return scores, indices

    embeddings = index_or_embeddings
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings[np.newaxis, :]

    if embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")

    if embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError(
            "Index/query embeddings dimension mismatch: "
            f"{embeddings.shape[1]} vs {query_embeddings.shape[1]}"
        )

    scores = query_embeddings.dot(embeddings.T)
    if top_k <= 0:
        return np.empty((scores.shape[0], 0), dtype=scores.dtype), np.empty((scores.shape[0], 0), dtype=np.int64)

    top_k = min(top_k, embeddings.shape[0])
    indices = np.argsort(-scores, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(scores, indices, axis=1)
    return top_scores, indices


def search_target_corpus(
    query_texts: List[str],
    target_embeddings: np.ndarray,
    model,
    top_k: int = 10,
    index: Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed query texts and search the target corpus embeddings."""
    query_embeddings = model.encode(
        query_texts,
        batch_size=len(query_texts),
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return search(index if index is not None else target_embeddings, query_embeddings, top_k)


def search_resumes_by_jd(
    jd_texts: List[str],
    resume_embeddings: np.ndarray,
    model,
    top_k: int = 10,
    resume_index: Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search resumes that are most similar to the given job descriptions."""
    return search_target_corpus(jd_texts, resume_embeddings, model, top_k, resume_index)


def search_jds_by_resume(
    resume_texts: List[str],
    jd_embeddings: np.ndarray,
    model,
    top_k: int = 10,
    jd_index: Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search job descriptions that are most similar to the given resumes."""
    return search_target_corpus(resume_texts, jd_embeddings, model, top_k, jd_index)


def save_index(index, path: str) -> None:
    """Persist a FAISS index to disk."""
    if faiss is None:
        raise ImportError("faiss is required to save an index.")
    faiss.write_index(index, path)


def load_index(path: str):
    """Load a FAISS index from disk."""
    if faiss is None:
        raise ImportError("faiss is required to load an index.")
    return faiss.read_index(path)
